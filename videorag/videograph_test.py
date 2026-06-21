import os
import sys
import json
import shutil
import glob
import torch
import asyncio
import multiprocessing
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from transformers import AutoModel, AutoTokenizer
import tiktoken
from google import genai

from ._llm import gemini_complete_with_image_sync, gemini_complete_if_cache
from .llm.qwen_vl import mllm_response
from .ego_prompt import PROMPTS


from ._llm import (
    LLMConfig,
    openai_config,
    azure_openai_config,
    ollama_config
)
from .streaming_op import (
    chunking_by_video_segments,
    streaming_extract_entities,
    streaming_videorag_query, 
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NanoVectorDBVideoSegmentStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    wrap_embedding_func_with_attrs,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from ._videoutil import(
    split_video,
    speech_to_text,
    segment_caption,
    merge_segment_information,
    saving_video_segments,
)
from .video_processing import sample_frames_by_interval


def parse_gemini_json_response(response):
    """
    Parse Gemini response that may be:
    1. A JSON string (empty list [] or JSON array/object)
    2. A markdown code block containing JSON (```json\n...\n```)
    
    Args:
        response: Response string from Gemini
        
    Returns:
        Parsed JSON object (list or dict)
    """
    if not response or not isinstance(response, str):
        return []
    
    response = response.strip()
    
    # Try to extract JSON from markdown code block
    # Pattern: ```json\n...\n``` or ```\n...\n``` (handles cases with or without trailing newline)
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n?```', response, re.DOTALL | re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array or object in the response
    # Look for [ ... ] or { ... }
    array_match = re.search(r'\[.*?\]', response, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass
    
    obj_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try parsing the whole response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # If all parsing fails, return empty list
        logger.warning(f"Failed to parse Gemini response as JSON: {response[:200]}")
        return []


def calculate_time_diff_seconds(time1: int, time2: int) -> float:
    """
    Calculate the time difference between two time numbers in seconds.
    
    Args:
        time1: First time number (e.g., 22040000 for 22:04:00.00)
        time2: Second time number (e.g., 22033000 for 22:03:30.00)
    
    Returns:
        Time difference in seconds (e.g., 30.0 for 30 seconds)
    
    Examples:
        >>> calculate_time_diff_seconds(22040000, 22033000)
        30.0
        >>> calculate_time_diff_seconds(22040000, 22040000)
        0.0
    """
    # Convert time numbers to strings for parsing
    time1_str = str(time1).zfill(8)  # Pad to 8 digits: HHMMSSMM
    time2_str = str(time2).zfill(8)
    
    # Parse hours, minutes, seconds, and milliseconds
    # Format: HHMMSSMM (2 digits each)
    h1 = int(time1_str[0:2])
    m1 = int(time1_str[2:4])
    s1 = int(time1_str[4:6])
    ms1 = int(time1_str[6:8]) if len(time1_str) >= 8 else 0
    
    h2 = int(time2_str[0:2])
    m2 = int(time2_str[2:4])
    s2 = int(time2_str[4:6])
    ms2 = int(time2_str[6:8]) if len(time2_str) >= 8 else 0
    
    # Convert to total seconds
    total_seconds1 = h1 * 3600 + m1 * 60 + s1 + ms1 / 100.0
    total_seconds2 = h2 * 3600 + m2 * 60 + s2 + ms2 / 100.0
    
    # Calculate difference
    diff_seconds = total_seconds1 - total_seconds2
    
    return diff_seconds


MODEL_MAP = {
    "qwenvl_3_4b_instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "qwenvl_2_5_7b_instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwenvl_2_5_3b_instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2vl_7b_instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2vl_2b_instruct": "Qwen/Qwen2-VL-2B-Instruct",
}


@dataclass
class VideoGraph:
    working_dir: str = field(
        default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # video
    threads_for_split: int = 10
    video_segment_length: int = 30 # seconds
    rough_num_frames_per_segment: int = 5 # frames
    fine_num_frames_per_segment: int = 15 # frames
    video_output_format: str = "mp4"
    audio_output_format: str = "mp3"
    video_embedding_batch_num: int = 2
    segment_retrieval_top_k: int = 4
    video_embedding_dim: int = 1024
    gemini_model_name: str = "gemini-2.5-pro"
    
    # query
    retrieval_topk_chunks: int = 2
    query_better_than_threshold: float = 0.2
    
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_video_segments
    chunk_token_size: int = 1200
    # chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # Change to your LLM provider
    llm: LLMConfig = field(default_factory=openai_config)
    
    # entity extraction
    entity_extraction_func: callable = streaming_extract_entities
    
    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vs_vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBVideoSegmentStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json
    
    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"VideoRAG init with param:\n\n  {_print_config}\n")
        
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=asdict(self)
        )
        
        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(wrap_embedding_func_with_attrs(
                embedding_dim = self.llm.embedding_dim,
                max_token_size = self.llm.embedding_max_token_size,
                model_name = self.llm.embedding_model_name)(self.llm.embedding_func))
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )
        
        self.video_segment_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_segment_feature",
                global_config=asdict(self),
                embedding_func=None, # we code the embedding process inside the insert() function.
            )
        )
        
        self.llm.best_model_func = limit_async_func_call(self.llm.best_model_max_async)(
            partial(self.llm.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.llm.cheap_model_func = limit_async_func_call(self.llm.cheap_model_max_async)(
            partial(self.llm.cheap_model_func, hashing_kv=self.llm_response_cache)
        )
        
        
    def load_caption_model(self, model_name: str):
        # 加载对应的模型
        model_path = next(
            (model_path for key, model_path in MODEL_MAP.items() if key in model_name),
            None
        )
        
        if model_path is None:
            raise ValueError(f"Model '{model_name}' not found in MODEL_MAP. Available models: {list(MODEL_MAP.keys())}")

        # 使用 videorag.llm.qwen_vl 模块中的函数
        self.mllm_response = mllm_response
        
        # 创建一个临时类来存储模型和处理器
        class ModelHolder:
            def __init__(self):
                self.video_llm = None
                self.processor = None
                self.image_processor = None
        
        model_holder = ModelHolder()
        
        # 加载模型 - 使用 qwen_vl.py 中的逻辑
        from transformers import AutoProcessor
        
        # 根据模型路径确定使用哪个模型类
        if "Qwen2.5" in model_path or "Qwen2_5" in model_path:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                video_llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
            except ImportError:
                from transformers import Qwen2VLForConditionalGeneration
                video_llm = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
        elif "Qwen3" in model_path:
            from transformers import Qwen3VLForConditionalGeneration
            video_llm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            video_llm = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        
        # 加载 processor
        processor = AutoProcessor.from_pretrained(model_path)
        image_processor = processor
        
        self.caption_model = video_llm
        self.caption_processor = processor
        self.processor = processor
        self.video_llm = video_llm
        self.image_processor = image_processor


    def process_frame_with_proactive_service(self, frame_info, accumulated_captions, max_captions_per_level=3, history_messages=None, retrieved_memory=None):
        """
        Process a single frame with Gemini model for proactive service detection.
        This function is separate from the hierarchical caption generation.
        
        Args:
            frame_info: Dict containing frame information with keys:
                       - 'frame': base64-encoded frame (or 'caption' for text-only)
                       - 'timestamp': frame timestamp
                       - 'datetime': frame datetime
                       - other metadata
            accumulated_captions: Dict containing accumulated captions with structure:
                                 {
                                     'second_captions': [{'time_span': str, 'caption': str, 'timestamp': int}, ...],
                                     'min_captions': [{'time_span': str, 'caption': str, 'timestamp': int}, ...],
                                     'hour_captions': [{'time_span': str, 'caption': str, 'timestamp': int}, ...]
                                 }
                                 All captions should be ordered by time (earliest first)
            proactive_prompt: Optional prompt for proactive service. If None, uses RPOACITVE_SERVICE_PROMPT.
            max_captions_per_level: Maximum number of captions to include per level (default: 3)
            history_messages: Optional string containing previous proactive service history (formatted as text)
            
        Returns:
            Dict containing:
            {
                'frame_timestamp': int,
                'selected_captions': {
                    'second_captions': [...],
                    'min_captions': [...],
                    'hour_captions': [...]
                },
                'gemini_response': str or dict (parsed JSON if possible),
                'prompt_used': str (the prompt that was used)
            }
        """
        # if not self.gemini_available:
        #     logger.warning("Gemini model not available. Skipping proactive service detection.")
        #     return None
        
        if retrieved_memory is None:
            proactive_prompt = PROMPTS["proactive_service_prompt"]
        else:
            proactive_prompt = PROMPTS["proactive_service_prompt_with_memory"]
            proactive_prompt = proactive_prompt + "\n\nRetrieved memory:\n\n" + retrieved_memory
        
        # 将history_messages（字符串）拼接在system_prompt后面
        if history_messages and isinstance(history_messages, str) and history_messages.strip():
            proactive_prompt = proactive_prompt + "\n\n" + "=== Recent Proactive Service History ===\n" + history_messages
        
        # Handle both frame (image) and caption (text) cases
        frame = frame_info.get('frame')
        caption = frame_info.get('caption')  # For text-only mode
        frame_timestamp = frame_info.get('time_span_info', {}).get('timestamp')
        
        if frame is None and caption is None:
            logger.warning("Both frame and caption are None, skipping proactive service detection")
            return None
        
        # Select relevant captions: get the most recent captions before current frame
        # Each level (hour/min/second) should have at most max_captions_per_level captions
        selected_captions = {
            'second_captions': [],
            'min_captions': [],
            'hour_captions': []
        }
        
        # Filter captions that are before the current frame timestamp
        if frame_timestamp is not None:
            for level in ['second_captions', 'min_captions', 'hour_captions']:
                level_captions = accumulated_captions.get(level, [])
                # Filter captions before current frame
                before_frame = [
                    cap for cap in level_captions 
                    if cap.get('timestamp', 0) <= frame_timestamp
                ]
                # Get the most recent max_captions_per_level captions
                selected_captions[level] = before_frame[-max_captions_per_level:] if len(before_frame) > max_captions_per_level else before_frame
        else:
            # If no timestamp, just take the most recent ones
            for level in ['second_captions', 'min_captions', 'hour_captions']:
                level_captions = accumulated_captions.get(level, [])
                selected_captions[level] = level_captions[-max_captions_per_level:] if len(level_captions) > max_captions_per_level else level_captions
        
        # Format captions for prompt
        caption_text_parts = []
        if selected_captions['hour_captions']:
            caption_text_parts.append("=== Hour-level Captions ===")
            for cap in selected_captions['hour_captions']:
                caption_text_parts.append(f"[{cap.get('time_span', 'N/A')}]: {cap.get('caption', '')}")
        
        if selected_captions['min_captions']:
            caption_text_parts.append("\n=== Minute-level Captions ===")
            for cap in selected_captions['min_captions']:
                caption_text_parts.append(f"[{cap.get('time_span', 'N/A')}]: {cap.get('caption', '')}")
        
        if selected_captions['second_captions']:
            caption_text_parts.append("\n=== Second-level Captions ===")
            for cap in selected_captions['second_captions']:
                caption_text_parts.append(f"[{cap.get('time_span', 'N/A')}]: {cap.get('caption', '')}")
        
        caption_text = "\n".join(caption_text_parts)
        
        # Build prompt with captions
        if caption_text:
            full_prompt = f"{proactive_prompt}\n\nHistorical Captions:\n{caption_text}"
        else:
            full_prompt = proactive_prompt
        
        # Prepare content for Gemini (frame + text or text-only)
        frame_image = None
        if frame:
            # Convert base64 string to PIL Image
            import base64
            from PIL import Image
            import io
            
            # Decode base64 string to bytes and create PIL Image
            frame_bytes = base64.b64decode(frame)
            frame_image = Image.open(io.BytesIO(frame_bytes))
        
        # Call Gemini model using _llm.py function
        text_prompt = full_prompt
        images = [frame_image] if frame_image else None  # PIL Image (if available)
        
        # Use appropriate function based on whether we have images
        if images:
            # Image + text mode (synchronous)
            from ._llm import gemini_complete_with_image_sync
            response = gemini_complete_with_image_sync(
                model=self.gemini_model_name,
                prompt=text_prompt,
                images=images,
                system_prompt=None,
                temperature=0.7,
                max_tokens=8192
            )
        else:
            # Text-only mode (use async function with asyncio.run)
            import asyncio
            from ._llm import gemini_complete_if_cache
            response = asyncio.run(gemini_complete_if_cache(
                model=self.gemini_model_name,
                prompt=text_prompt,
                system_prompt=None,
                history_messages=None,  # history_messages已经拼接在prompt中了
                use_cache=False,
                temperature=0.7,
                max_tokens=8192
            ))
        
        parsed_response = parse_gemini_json_response(response)
        
        return {
            'frame_timestamp': frame_timestamp,
            'selected_captions': selected_captions,
            'gemini_response': parsed_response,
            'prompt_used': full_prompt  # Save the prompt used for history
        }


    def streaming_graph_construction(self, data_path=None, anno_path=None, day=None, 
                       interval_seconds=5, window_seconds=30, gap_threshold_seconds=60, 
                       window_minutes=5, window_hours=1,
                       prompt_template=None, prompt_template_min=None, prompt_template_hour=None,
                       max_new_tokens=1024, enable_proactive_service=True, 
                       save_to_videorag=False, frame_time_data=None):
        """
        Construct graph from video frames using streaming input (video by video).
        Process videos one by one, extract frames every 3 seconds, and when accumulated frames 
        reach the window time limit (10s), generate caption using Qwen model and check proactive service.
        
        Args:
            data_path: Path to video data directory (e.g., "/path/to/data/DAY1")
            anno_path: Path to annotation directory (e.g., "/path/to/anno/DAY1")
            day: Day identifier (e.g., "DAY1")
            interval_seconds: Interval between sampled frames in seconds (default: 3)
            window_seconds: Window size in seconds for 10s caption generation (default: 10)
            gap_threshold_seconds: Gap threshold to start new window (default: 60)
            window_minutes: Window size in minutes for 10min caption generation (default: 10)
            window_hours: Window size in hours for 1h caption generation (default: 1)
            prompt_template: Optional prompt template for 10s captions. If None, uses default.
            prompt_template_min: Optional prompt template for 10min captions. If None, uses default.
            prompt_template_hour: Optional prompt template for 1h captions. If None, uses default.
            max_new_tokens: Maximum number of tokens to generate (default: 512)
            enable_proactive_service: If True, call proactive service after generating 10s caption (default: False)
            save_to_videorag: If True, save generated captions to VideoRAG storage (default: False)
            frame_time_data: Optional list of dicts with time information for each frame (for backward compatibility)
                           If provided, uses this instead of processing videos from data_path
            
        Returns:
            Dictionary with generated captions:
            {
                "second_captions": {time_span: caption, ...},
                "min_captions": {time_span: caption, ...},
                "hour_captions": {time_span: caption, ...},
                "proactive_responses": [...]  # Only if enable_proactive_service=True
            }
        """
        # Helper functions for time processing
        def get_time_span_info(anno_file: str) -> Optional[Dict[str, any]]:
            """Get detailed time span information from annotation file."""
            # import json  # Import json in function scope to avoid closure issues
            if not os.path.exists(anno_file):
                logger.warning(f"Annotation file not found: {anno_file}")
                return None
            
            try:
                with open(anno_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                dense_caption = data.get("dense_caption", [])
                
                if not dense_caption:
                    logger.warning(f"No dense_caption found in {anno_file}")
                    return None
                
                first_caption = dense_caption[0]
                last_caption = dense_caption[-1]
                
                start_time = first_caption.get("start_time", "")
                end_time = last_caption.get("end_time", "")
                start_time_number = first_caption.get("start_time_number", 0)
                end_time_number = last_caption.get("end_time_number", 0)
                
                if not start_time or not end_time:
                    logger.warning(f"Missing start_time or end_time in {anno_file}")
                    return None
                
                def time_to_seconds(time_str: str) -> int:
                    """Convert HH:MM:SS to total seconds."""
                    parts = time_str.split(":")
                    if len(parts) != 3:
                        return 0
                    hours, minutes, seconds = map(int, parts)
                    return hours * 3600 + minutes * 60 + seconds
                
                start_seconds = time_to_seconds(start_time)
                end_seconds = time_to_seconds(end_time)
                duration_seconds = end_seconds - start_seconds
                
                return {
                    'time_span': f"{start_time}-{end_time}",
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_time_number': start_time_number,
                    'end_time_number': end_time_number,
                    'duration_seconds': duration_seconds,
                    'caption_count': len(dense_caption)
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON file {anno_file}: {e}")
                return None
            except Exception as e:
                logger.error(f"Error reading annotation file {anno_file}: {e}")
                return None

        def parse_time_string(time_str: str) -> Optional[datetime]:
            """Parse time string in format HH:MM:SS to datetime object."""
            try:
                parts = time_str.split(":")
                if len(parts) != 3:
                    return None
                hours, minutes, seconds = map(int, parts)
                return datetime(2000, 1, 1, hours, minutes, seconds)
            except Exception:
                return None

        def time_number_to_datetime(time_number: int, day_num: int) -> Optional[datetime]:
            """Convert time_number (e.g., 11094300) to datetime."""
            try:
                time_str = str(time_number).zfill(8)
                hours = int(time_str[0:2])
                minutes = int(time_str[2:4])
                seconds = int(time_str[4:6])
                base_date = datetime(2000, 1, 1)
                return base_date + timedelta(days=day_num-1, hours=hours, minutes=minutes, seconds=seconds)
            except Exception as e:
                logger.warning(f"Failed to parse time_number {time_number}: {e}")
                return None

        def get_day_number(day_str: str) -> int:
            """Extract day number from day string (e.g., 'DAY1' -> 1)."""
            try:
                return int(day_str.replace("DAY", ""))
            except Exception:
                return 0
        
        loop = always_get_an_event_loop()
        
        # Default prompts if not provided
        # default_prompt_second = prompt_template if prompt_template else PROMPTS["second_caption_prompt"]
        # default_prompt_min = prompt_template_min if prompt_template_min else PROMPTS["min_caption_prompt"]
        # default_prompt_hour = prompt_template_hour if prompt_template_hour else PROMPTS["hour_caption_prompt"]
        
        # Store results
        second_captions = {}
        min_captions = {}
        hour_captions = {}
        proactive_responses = []  # Store proactive service responses
        
        # Maintain accumulated captions for proactive service (ordered by time)
        accumulated_captions = {
            'second_captions': [],  # List of dicts: {'time_span': str, 'caption': str, 'timestamp': int}
            'min_captions': [],
            'hour_captions': []
        }
        
        # Maintain proactive service history to avoid frequent interruptions
        proactive_service_history = []  # List of dicts: {'service_sub_type': str, 'user_prompt': str, 'trigger_time_window': str}
        
        def format_proactive_history(history_list):
            """Format proactive service history as a string."""
            if not history_list:
                return ""
            history_lines = []
            for i, item in enumerate(history_list, 1):
                history_lines.append(
                    f"{i}. Service Type: {item.get('service_sub_type', 'N/A')}\n"
                    f"   Time Window: {item.get('trigger_time_window', 'N/A')}\n"
                    f"   User Prompt: {item.get('user_prompt', 'N/A')}"
                )
            return "\n\n".join(history_lines)
        
        # 10-second window state
        current_window_frames = []  # List of base64 frames in current window
        current_window_frame_data = []  # List of frame data dicts in current window
        window_start_dt = None
        window_start_day = None
        window_start_timestamp = None
        last_frame_end_time = None  # Track last frame's end time for gap detection
        
        # 10-minute window state
        min_window_second_captions = []  # List of 10s captions in current 10min window
        min_window_start_dt = None
        min_window_start_day = None
        min_window_start_timestamp = None
        
        # 1-hour window state
        hour_window_min_captions = []  # List of 10min captions in current 1h window
        hour_window_start_dt = None
        hour_window_start_day = None
        hour_window_start_timestamp = None
        
        # Process videos if data_path and anno_path are provided
        if data_path and anno_path and day:
            # Get video and annotation file lists
            days_list = sorted(glob.glob(os.path.join(data_path, "*.mp4")))
            anno_list = sorted(glob.glob(os.path.join(anno_path, "*.json")))
            
            if not days_list:
                logger.warning(f"No video files found in {data_path}")
                return {"second_captions": {}, "min_captions": {}, "hour_captions": {}}
            
            # Extract day number from day string
            day_num = get_day_number(day)
            
            logger.info(f"Starting streaming processing: {len(days_list)} videos, window={window_seconds}s, interval={interval_seconds}s")
            
            # Process each video one by one
            for video_idx, (video_path, anno_file) in enumerate(zip(days_list, anno_list)):
                video_name = os.path.basename(video_path).split(".")[0]
                anno_name = os.path.basename(anno_file).split(".")[0]
                assert video_name == anno_name, f"Video name {video_name} and annotation name {anno_name} do not match"
                
                # 以视频为单位，如果视频已经被处理过，则直接跳过
                if video_name in self.video_path_db._data:
                    logger.info(f"Video {video_name} already exists in database, skipping")
                    continue
                
                logger.info(f"Processing video {video_idx + 1}/{len(days_list)}: {video_name}")
                
                loop.run_until_complete(self.video_path_db.upsert(
                        {video_name: video_path}
                    ))
                
                # Extract time span from annotation file
                time_span_info = get_time_span_info(anno_file)
                if not time_span_info:
                    logger.warning(f"No time span info for {video_name}, skipping")
                    continue
                
                # 每5s提取一帧，同时获取各帧的时间范围
                base64_frames, frame_timestamps, frame_time_ranges = sample_frames_by_interval(
                    video_path, 
                    interval_seconds=interval_seconds,
                    output_format='base64',
                    time_span_info=time_span_info
                )
                
                if not base64_frames:
                    logger.warning(f"No frames extracted from {video_name}")
                    continue
                
                # Process frames one by one (streaming)
                for i, frame in enumerate(base64_frames):
                    # Get timestamp for this frame
                    if frame_timestamps and i < len(frame_timestamps):
                        timestamp = frame_timestamps[i]
                    else:
                        # Calculate timestamp from start_time_number
                        start_time_number = time_span_info['start_time_number']
                        start_time_str = str(start_time_number).zfill(8)
                        start_hours = int(start_time_str[0:2])
                        start_minutes = int(start_time_str[2:4])
                        start_seconds = int(start_time_str[4:6])
                        start_centiseconds = int(start_time_str[6:8])
                        start_total_seconds = start_hours * 3600 + start_minutes * 60 + start_seconds + start_centiseconds / 100.0
                        absolute_seconds = start_total_seconds + i * interval_seconds
                        
                        total_hours = int(absolute_seconds // 3600)
                        remaining = absolute_seconds % 3600
                        total_minutes = int(remaining // 60)
                        remaining = remaining % 60
                        total_secs = int(remaining)
                        centiseconds = int(round((remaining - total_secs) * 100))
                        
                        if centiseconds >= 100:
                            total_secs += 1
                            centiseconds = 0
                        if total_secs >= 60:
                            total_minutes += 1
                            total_secs = 0
                        if total_minutes >= 60:
                            total_hours += 1
                            total_minutes = 0
                        
                        timestamp = int(total_hours * 1000000 + total_minutes * 10000 + total_secs * 100 + centiseconds)
                    
                    # Convert timestamp to datetime
                    frame_dt = time_number_to_datetime(timestamp, day_num)
                    if frame_dt is None:
                        start_time_number = time_span_info['start_time_number']
                        start_dt = time_number_to_datetime(start_time_number, day_num)
                        if start_dt:
                            frame_dt = start_dt + timedelta(seconds=i * interval_seconds)
                    
                    if not frame_dt:
                        logger.warning(f"Failed to get datetime for frame {i} in {video_name}, skipping")
                        continue
                    
                    # Create frame info dict
                    frame_info = {
                        'frame': frame,
                        'datetime': frame_dt,
                        'day_num': day_num,
                        'video_name': video_name,
                        'frame_idx': i,
                        'start_time': frame_time_ranges[i]['start'],
                        'end_time': frame_time_ranges[i]['end'],
                        'time_span_info': {
                            **time_span_info,
                            "timestamp": timestamp,
                        }
                    }
                    
                    # Process frame in streaming manner
                    start_time = time_span_info.get('start_time_number')
                    end_time = time_span_info.get('end_time_number')
                    
                    # Initialize first window
                    if window_start_timestamp is None:
                        window_start_timestamp = timestamp
                        window_start_dt = frame_dt
                        window_start_day = day_num
                        current_window_frames = [frame]
                        current_window_frame_data = [frame_info]
                        last_frame_end_time = end_time
                    else:
                        # Check for gap or day change
                        gap_duration = 0
                        if last_frame_end_time is not None:
                            gap_duration = calculate_time_diff_seconds(start_time, last_frame_end_time)
                        is_gap = gap_duration >= gap_threshold_seconds
                        day_changed = (day_num != window_start_day)
                        
                        # Check if window is full
                        window_duration = calculate_time_diff_seconds(timestamp, window_start_timestamp)
                        
                        # If gap, day change, or window full, process current window
                        if day_changed or is_gap or window_duration >= window_seconds:
                            # Generate caption for current window if it has frames
                            if current_window_frames:
                                start_dt = current_window_frame_data[0]['datetime']
                                end_dt = current_window_frame_data[-1]['datetime']
                                start_day = current_window_frame_data[0]['day_num']
                                end_day = current_window_frame_data[-1]['day_num']
                                last_frame_timestamp = current_window_frame_data[-1]['time_span_info'].get('timestamp')
                                
                                time_span = f"{start_day}-{start_dt.strftime('%H:%M:%S')}-{end_dt.strftime('%H:%M:%S')}"
                                
                                # Generate 10s caption using Qwen model with retry mechanism
                                caption_prompt = PROMPTS["simple_second_caption_system_prompt"]
                                max_retries = 3
                                caption_text = None
                                caption = None
                                
                                for retry_count in range(max_retries):
                                    caption_text = self.mllm_response(
                                        self.video_llm,
                                        self.processor,
                                        caption_prompt,
                                        None, 
                                        base64_frames=current_window_frames,
                                        max_new_tokens=max_new_tokens,
                                        has_image=True
                                    )
                                    
                                    # 验证输出格式是否符合要求
                                    try:
                                        caption = json.loads(caption_text)
                                        # 检查是否包含 "frames" 键且为字典类型
                                        if isinstance(caption, dict) and "frames" in caption and isinstance(caption["frames"], dict):
                                            # 格式正确，跳出循环
                                            break
                                        else:
                                            logger.warning(f"Caption format invalid (attempt {retry_count + 1}/{max_retries}): missing 'frames' key or invalid structure")
                                            if retry_count == max_retries - 1:
                                                raise ValueError(f"Caption format invalid after {max_retries} attempts: missing 'frames' key or invalid structure")
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse caption as JSON (attempt {retry_count + 1}/{max_retries}): {e}")
                                        if retry_count == max_retries - 1:
                                            raise ValueError(f"Failed to parse caption as JSON after {max_retries} attempts: {e}")
                                
                                # Qwen输出了各帧的caption，需要将它们整合为一个字典
                                caption_dict = {}
                                for frame_idx, content in caption["frames"].items():
                                    caption_dict[f"DAY{day_num}-{current_window_frame_data[int(frame_idx)]["start_time"]}-{current_window_frame_data[int(frame_idx)]["end_time"]}"] = content
                                second_captions[time_span] = caption_dict
                                logger.info(f"Generated 10s caption for window: {time_span} ({len(current_window_frames)} frames)")
                                
                                # 将caption信息和视频帧存入到video_segment和video_segment_feature_vdb中, 这里不存入单帧的信息，而是整个video clip的信息
                                loop.run_until_complete(self.video_segments.upsert({time_span: {"content": caption_text, "video_frames": current_window_frames, "type": "second"}}))
                                loop.run_until_complete(self.video_segment_feature_vdb.upsert_video_segment(time_span, current_window_frames))
                                loop.run_until_complete(self._save_video_segments())
                                loop.run_until_complete(self.ainsert_streaming_caption({time_span: {"content": json.dumps({time_span: caption_dict}), "video_frames": current_window_frames, "type": "second"}}))    # 根据caption进行流式建图
                                
                                # Add to accumulated_captions for proactive service
                                if enable_proactive_service:
                                    accumulated_captions['second_captions'].append({
                                        'time_span': time_span,
                                        'caption': caption_dict,
                                        'timestamp': last_frame_timestamp, 
                                    })
                                    
                                    # Call proactive service with the 10s caption and accumulated captions
                                    frame_info_for_proactive = {
                                        'caption': caption_dict,
                                        'frame': None,  # Use caption instead of single frame
                                        'time_span_info': {
                                            'timestamp': last_frame_timestamp
                                        }
                                    }
                                    
                                    # 第一次调用：使用Gemini判断是否需要主动服务
                                    # 格式化历史记录为字符串
                                    history_text = format_proactive_history(proactive_service_history)
                                    proactive_result = self.process_frame_with_proactive_service(
                                        frame_info=frame_info_for_proactive,
                                        accumulated_captions=accumulated_captions,
                                        max_captions_per_level=3,
                                        history_messages=history_text,
                                        retrieved_memory=None
                                    )
                                    
                                    if not proactive_result or proactive_result.get('gemini_response') is None:
                                        logger.warning("Proactive service returned None in first call")
                                    else:
                                        gemini_response = proactive_result.get('gemini_response')
                                        
                                        # 解析响应：应该是JSON列表或空列表
                                        service_list = []
                                        if isinstance(gemini_response, list):
                                            service_list = gemini_response
                                        elif isinstance(gemini_response, dict):
                                            # 如果是单个对象，转换为列表
                                            service_list = [gemini_response]
                                        elif isinstance(gemini_response, str):
                                            # 尝试解析JSON字符串
                                            # import json
                                            parsed = json.loads(gemini_response)
                                            if isinstance(parsed, list):
                                                service_list = parsed
                                            elif isinstance(parsed, dict):
                                                service_list = [parsed]
                                        
                                        # 检查是否有需要检索的服务（包含memory_query字段且非空）
                                        needs_retrieval = False
                                        retrieval_query = None
                                        
                                        for service in service_list:
                                            if isinstance(service, dict) and 'memory_query' in service:
                                                memory_query = service.get('memory_query', '')
                                                time_key = service.get('trigger_time_window', '')
                                                service_type = service.get('service_main_type', '')
                                                # memory_query是可选的，只有在需要检索时才存在且非空
                                                if memory_query and isinstance(memory_query, str) and memory_query.strip():
                                                    # 提取query内容（可能在<query>标签中）
                                                    query_match = re.search(r'<query>(.*?)</query>', memory_query, re.IGNORECASE | re.DOTALL)
                                                    if query_match:
                                                        retrieval_query = query_match.group(1).strip()
                                                    else:
                                                        retrieval_query = memory_query.strip()
                                                    
                                                    if retrieval_query:
                                                        needs_retrieval = True
                                                        break
                                        
                                        # 如果需要检索，进行检索并再次调用Gemini
                                        if needs_retrieval and retrieval_query:
                                            logger.info(f"Proactive service requires retrieval. Query: {retrieval_query}")
                                            
                                            from .base import QueryParam
                                            query_param = QueryParam(mode="videorag")
                                            
                                            retrieved_video_context, retrieved_chunk_context = loop.run_until_complete(streaming_videorag_query(
                                                retrieval_query,
                                                time_key, 
                                                service_type,
                                                self.entities_vdb,
                                                self.text_chunks,
                                                self.chunks_vdb,
                                                self.video_path_db,
                                                self.video_segments,
                                                self.video_segment_feature_vdb,
                                                self.chunk_entity_relation_graph,
                                                self.caption_model, 
                                                self.caption_processor,
                                                query_param,
                                                asdict(self),
                                            ))
                                            retrieved_response = retrieved_video_context + "\n" + retrieved_chunk_context
                                            
                                            logger.info(f"Retrieved memory evidence for proactive service")
                                            
                                            # 第二次调用：使用检索到的记忆
                                            # 使用相同的历史记录
                                            proactive_result_with_memory = self.process_frame_with_proactive_service(
                                                frame_info=frame_info_for_proactive,
                                                accumulated_captions=accumulated_captions,
                                                max_captions_per_level=3,
                                                history_messages=history_text,
                                                retrieved_memory=retrieved_response
                                            )
                                            
                                            if proactive_result_with_memory and proactive_result_with_memory.get('gemini_response'):
                                                # 使用带记忆的结果
                                                final_response = proactive_result_with_memory.get('gemini_response')
                                                if isinstance(final_response, list):
                                                    service_list = final_response
                                                elif isinstance(final_response, dict):
                                                    service_list = [final_response]
                                                elif isinstance(final_response, str):
                                                    # import json
                                                    parsed = json.loads(final_response)
                                                    if isinstance(parsed, list):
                                                        service_list = parsed
                                                    elif isinstance(parsed, dict):
                                                        service_list = [parsed]
                                                
                                                proactive_result['gemini_response'] = service_list
                                        
                                        # 保存最终结果并更新历史记录
                                        if service_list:
                                            proactive_responses.append(proactive_result)
                                            logger.info(f"Proactive service triggered: {len(service_list)} service(s)")
                                            
                                            # 将每个服务的service_sub_type, user_prompt, trigger_time_window添加到历史记录
                                            for service in service_list:
                                                if isinstance(service, dict):
                                                    proactive_service_history.append({
                                                        'service_sub_type': service.get('service_sub_type', ''),
                                                        'user_prompt': service.get('user_prompt', ''),
                                                        'trigger_time_window': service.get('trigger_time_window', '')
                                                    })
                                        else:
                                            logger.info("No proactive service needed")
                                    
                                    # Add to 10-minute window
                                    if min_window_start_timestamp is None:
                                        min_window_start_timestamp = window_start_timestamp
                                        min_window_start_dt = start_dt
                                        min_window_start_day = start_day
                                    
                                    min_window_second_captions.append({
                                        'time_span': time_span,
                                        'caption': caption,
                                        'start_timestamp': window_start_timestamp,
                                        'end_timestamp': last_frame_timestamp
                                    })
                                    
                                    # Check if 10-minute window is full
                                    min_window_duration = calculate_time_diff_seconds(last_frame_timestamp, min_window_start_timestamp)
                                    if min_window_duration >= window_minutes * 60:
                                        # Generate 10-minute caption
                                        if min_window_second_captions:
                                            min_start_dt = min_window_start_dt
                                            min_end_dt = end_dt
                                            min_start_day = min_window_start_day
                                            min_end_day = end_day
                                            min_time_span = f"{min_start_day}-{min_start_dt.strftime('%H:%M:%S')}-{min_end_dt.strftime('%H:%M:%S')}"
                                            
                                            # caption_texts = [item['caption'] for item in min_window_second_captions]
                                            caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']['caption']}" for item in min_window_second_captions])
                                            
                                            user_prompt = f"{PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                            
                                            min_caption = self.mllm_response(
                                                self.video_llm,
                                                self.processor,
                                                user_prompt,
                                                None, 
                                                base64_frames=None,
                                                max_new_tokens=max_new_tokens,
                                                has_image=False
                                            )
                                            min_captions[min_time_span] = min_caption
                                            logger.info(f"Generated 10min caption for window: {min_time_span} ({len(min_window_second_captions)} 10s captions)")
                                            
                                            loop.run_until_complete(self.video_segments.upsert({min_time_span: {"content": min_caption, "video_frames": current_window_frames, "type": "minute"}}))
                                            loop.run_until_complete(self._save_video_segments())
                                            loop.run_until_complete(self.ainsert_streaming_caption({min_time_span: {"content": min_caption, "video_frames": current_window_frames, "type": "minute"}}))
                                            
                                            if enable_proactive_service:
                                                accumulated_captions['min_captions'].append({
                                                    'time_span': min_time_span,
                                                    'caption': min_caption,
                                                    'timestamp': last_frame_timestamp
                                                })
                                            
                                            # Add to 1-hour window
                                            if hour_window_start_timestamp is None:
                                                hour_window_start_timestamp = min_window_start_timestamp
                                                hour_window_start_dt = min_start_dt
                                                hour_window_start_day = min_start_day
                                            
                                            hour_window_min_captions.append({
                                                'time_span': min_time_span,
                                                'caption': min_caption,
                                                'start_timestamp': min_window_start_timestamp,
                                                'end_timestamp': last_frame_timestamp
                                            })
                                            
                                            # Check if 1-hour window is full
                                            hour_window_duration = calculate_time_diff_seconds(last_frame_timestamp, hour_window_start_timestamp)
                                            if hour_window_duration >= window_hours * 3600:
                                                # Generate 1-hour caption
                                                if hour_window_min_captions:
                                                    hour_start_dt = hour_window_start_dt
                                                    hour_end_dt = min_end_dt
                                                    hour_start_day = hour_window_start_day
                                                    hour_end_day = min_end_day
                                                    hour_time_span = f"{hour_start_day}-{hour_start_dt.strftime('%H:%M:%S')}-{hour_end_dt.strftime('%H:%M:%S')}"
                                                    
                                                    min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                                                    user_prompt = f"{PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                                                                            
                                                    hour_caption = self.mllm_response(
                                                        self.video_llm,
                                                        self.processor,
                                                        user_prompt,
                                                        None,  
                                                        base64_frames=None,
                                                        max_new_tokens=max_new_tokens,
                                                        has_image=False
                                                    )
                                                    hour_captions[hour_time_span] = hour_caption
                                                    logger.info(f"Generated 1h caption for window: {hour_time_span} ({len(hour_window_min_captions)} 10min captions)")
                                                    
                                                    loop.run_until_complete(self.video_segment.upsert({hour_time_span: {"content": hour_caption, "video_frames": current_window_frames, "type": "hour"}}))
                                                    loop.run_until_complete(self._save_video_segments())
                                                    loop.run_until_complete(self.ainsert_streaming_caption({hour_time_span: {"content": hour_caption, "video_frames": current_window_frames, "type": "hour"}}))
                                                    
                                                    if enable_proactive_service:
                                                        accumulated_captions['hour_captions'].append({
                                                            'time_span': hour_time_span,
                                                            'caption': hour_caption,
                                                            'timestamp': last_frame_timestamp
                                                        })
                                                    
                                                    # Reset 1-hour window
                                                    hour_window_min_captions = []
                                                    hour_window_start_timestamp = None
                                                    hour_window_start_dt = None
                                                    hour_window_start_day = None
                                            
                                            # Reset 10-minute window
                                            min_window_second_captions = []
                                            min_window_start_timestamp = None
                                            min_window_start_dt = None
                                            min_window_start_day = None
                            
                            # Start new window with current frame
                            window_start_timestamp = timestamp
                            window_start_dt = frame_dt
                            window_start_day = day_num
                            current_window_frames = [frame]
                            current_window_frame_data = [frame_info]
                            last_frame_end_time = end_time
                        else:
                            # Add frame to current window
                            current_window_frames.append(frame)
                            current_window_frame_data.append(frame_info)
                            last_frame_end_time = end_time
            
            # Process last 10s window if it has frames    处理最终剩余的视频帧
            if current_window_frames:
                start_dt = current_window_frame_data[0]['datetime']
                end_dt = current_window_frame_data[-1]['datetime']
                start_day = current_window_frame_data[0]['day_num']
                end_day = current_window_frame_data[-1]['day_num']
                last_frame_timestamp = current_window_frame_data[-1]['time_span_info'].get('timestamp')
                
                time_span = f"{start_day}-{start_dt.strftime('%H:%M:%S')}-{end_dt.strftime('%H:%M:%S')}"
                
                # Generate 10s caption using Qwen model with retry mechanism
                caption_prompt = PROMPTS["simple_second_caption_system_prompt"]
                max_retries = 3
                caption_text = None
                caption = None
                
                for retry_count in range(max_retries):
                    caption_text = self.mllm_response(
                        self.video_llm,
                        self.processor,
                        caption_prompt,
                        None, 
                        base64_frames=current_window_frames,
                        max_new_tokens=max_new_tokens,
                        has_image=True
                    )
                    
                    # 验证输出格式是否符合要求
                    try:
                        caption = json.loads(caption_text)
                        # 检查是否包含 "frames" 键且为字典类型
                        if isinstance(caption, dict) and "frames" in caption and isinstance(caption["frames"], dict):
                            # 格式正确，跳出循环
                            break
                        else:
                            logger.warning(f"Caption format invalid (attempt {retry_count + 1}/{max_retries}): missing 'frames' key or invalid structure")
                            if retry_count == max_retries - 1:
                                raise ValueError(f"Caption format invalid after {max_retries} attempts: missing 'frames' key or invalid structure")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse caption as JSON (attempt {retry_count + 1}/{max_retries}): {e}")
                        if retry_count == max_retries - 1:
                            raise ValueError(f"Failed to parse caption as JSON after {max_retries} attempts: {e}")
                
                # Qwen输出了各帧的caption，需要将它们整合为一个字典
                caption_dict = {}
                for frame_idx, content in caption["frames"].items():
                    caption_dict[f"DAY{start_day}-{current_window_frame_data[int(frame_idx)]['start_time']}-{current_window_frame_data[int(frame_idx)]['end_time']}"] = content
                second_captions[time_span] = caption_dict
                logger.info(f"Generated 10s caption for final window: {time_span} ({len(current_window_frames)} frames)")
                
                loop.run_until_complete(self.video_segments.upsert({time_span: {"content": caption_text, "video_frames": current_window_frames, "type": "second"}}))
                loop.run_until_complete(self.video_segment_feature_vdb.upsert_video_segment(time_span, current_window_frames))
                loop.run_until_complete(self._save_video_segments())
                loop.run_until_complete(self.ainsert_streaming_caption({time_span: {"content": caption_text, "video_frames": current_window_frames, "type": "second"}}))
                
                if enable_proactive_service:
                    accumulated_captions['second_captions'].append({
                        'time_span': time_span,
                        'caption': caption_dict,
                        'timestamp': last_frame_timestamp
                    })
                    
                    # Call proactive service for final window
                    frame_info_for_proactive = {
                        'caption': caption_dict,
                        'frame': None,
                        'time_span_info': {'timestamp': last_frame_timestamp}
                    }
                    
                    # 第一次调用：使用Gemini判断是否需要主动服务
                    # 格式化历史记录为字符串
                    history_text = format_proactive_history(proactive_service_history)
                    proactive_result = self.process_frame_with_proactive_service(
                        frame_info=frame_info_for_proactive,
                        accumulated_captions=accumulated_captions,
                        max_captions_per_level=3,
                        history_messages=history_text,
                        retrieved_memory=None
                    )
                    
                    if not proactive_result or proactive_result.get('gemini_response') is None:
                        logger.warning("Proactive service returned None in first call (final window)")
                    else:
                        gemini_response = proactive_result.get('gemini_response')
                        
                        # 解析响应：使用parse_gemini_json_response函数
                        service_list = parse_gemini_json_response(gemini_response)
                        
                        # 检查是否有需要检索的服务（包含memory_query字段且非空）
                        needs_retrieval = False
                        retrieval_query = None
                        
                        for service in service_list:
                            if isinstance(service, dict) and 'memory_query' in service:
                                memory_query = service.get('memory_query', '')
                                time_key = service.get('trigger_time_window', '')
                                service_type = service.get('service_main_type', '')
                                # memory_query是可选的，只有在需要检索时才存在且非空
                                if memory_query and isinstance(memory_query, str) and memory_query.strip():
                                    # 提取query内容（可能在<query>标签中）
                                    query_match = re.search(r'<query>(.*?)</query>', memory_query, re.IGNORECASE | re.DOTALL)
                                    if query_match:
                                        retrieval_query = query_match.group(1).strip()
                                    else:
                                        retrieval_query = memory_query.strip()
                                    
                                    if retrieval_query:
                                        needs_retrieval = True
                                        break
                        
                        # 如果需要检索，进行检索并再次调用Gemini
                        if needs_retrieval and retrieval_query:
                            logger.info(f"Proactive service requires retrieval (final window). Query: {retrieval_query}")
                            
                            from .base import QueryParam
                            query_param = QueryParam(mode="videorag")
                            
                            retrieved_video_context, retrieved_chunk_context = loop.run_until_complete(streaming_videorag_query(
                                retrieval_query,
                                time_key, 
                                service_type, 
                                self.entities_vdb,
                                self.text_chunks,
                                self.chunks_vdb,
                                self.video_path_db,
                                self.video_segments,
                                self.video_segment_feature_vdb,
                                self.chunk_entity_relation_graph,
                                self.caption_model, 
                                self.caption_processor,
                                query_param,
                                asdict(self),
                            ))
                            retrieved_response = retrieved_video_context + "\n" + retrieved_chunk_context
                            
                            logger.info(f"Retrieved memory evidence for proactive service (final window)")
                            
                            # 第二次调用：使用检索到的记忆
                            # 使用相同的历史记录
                            proactive_result_with_memory = self.process_frame_with_proactive_service(
                                frame_info=frame_info_for_proactive,
                                accumulated_captions=accumulated_captions,
                                max_captions_per_level=3,
                                history_messages=history_text,
                                retrieved_memory=retrieved_response
                            )
                            
                            if proactive_result_with_memory and proactive_result_with_memory.get('gemini_response'):
                                # 使用带记忆的结果
                                final_response = proactive_result_with_memory.get('gemini_response')
                                service_list = parse_gemini_json_response(final_response)
                                proactive_result['gemini_response'] = service_list
                        
                        # 保存最终结果并更新历史记录
                        if service_list:
                            proactive_responses.append(proactive_result)
                            logger.info(f"Proactive service triggered (final window): {len(service_list)} service(s)")
                            
                            # 将每个服务的service_sub_type, user_prompt, trigger_time_window添加到历史记录
                            for service in service_list:
                                if isinstance(service, dict):
                                    proactive_service_history.append({
                                        'service_sub_type': service.get('service_sub_type', ''),
                                        'user_prompt': service.get('user_prompt', ''),
                                        'trigger_time_window': service.get('trigger_time_window', '')
                                    })
                        else:
                            logger.info("No proactive service needed (final window)")
                    
                    if min_window_start_timestamp is None:
                        min_window_start_timestamp = window_start_timestamp
                        min_window_start_dt = start_dt
                        min_window_start_day = start_day
                    
                    min_window_second_captions.append({
                        'time_span': time_span,
                        'caption': caption,
                        'start_timestamp': window_start_timestamp,
                        'end_timestamp': last_frame_timestamp
                    })
            
            # Process last 10-minute window if it has captions (for video processing mode)
            if min_window_second_captions:
                min_start_dt = min_window_start_dt
                min_end_dt = current_window_frame_data[-1]['datetime'] if current_window_frame_data else min_start_dt
                min_start_day = min_window_start_day
                min_end_day = current_window_frame_data[-1]['day_num'] if current_window_frame_data else min_start_day
                min_time_span = f"{min_start_day}-{min_start_dt.strftime('%H:%M:%S')}-{min_end_dt.strftime('%H:%M:%S')}"
                
                # caption_texts = [item['caption'] for item in min_window_second_captions]
                caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']['caption']}" for item in min_window_second_captions])
                
                user_prompt = f"{PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                
                min_caption = self.mllm_response(
                    self.video_llm,
                    self.processor,
                    user_prompt,
                    None, 
                    base64_frames=None,
                    max_new_tokens=max_new_tokens,
                    has_image=False
                )
                min_captions[min_time_span] = min_caption
                logger.info(f"Generated 10min caption for final window: {min_time_span} ({len(min_window_second_captions)} 10s captions)")
                
                loop.run_until_complete(self.video_segment.upsert({min_time_span: {"content": min_caption, "video_frames": current_window_frames, "type": "minute"}}))
                loop.run_until_complete(self._save_video_segments())
                loop.run_until_complete(self.ainsert_streaming_caption({min_time_span: {"content": min_caption, "video_frames": current_window_frames, "type": "minute"}}))
                
                if enable_proactive_service:
                    final_timestamp = current_window_frame_data[-1]['time_span_info'].get('timestamp') if current_window_frame_data else min_window_start_timestamp
                    accumulated_captions['min_captions'].append({
                        'time_span': min_time_span,
                        'caption': min_caption,
                        'timestamp': final_timestamp
                    })
                
                # Add to 1-hour window
                if hour_window_start_timestamp is None:
                    hour_window_start_timestamp = min_window_start_timestamp
                    hour_window_start_dt = min_start_dt
                    hour_window_start_day = min_start_day
                
                hour_window_min_captions.append({
                    'time_span': min_time_span,
                    'caption': min_caption,
                    'start_timestamp': min_window_start_timestamp,
                    'end_timestamp': current_window_frame_data[-1]['time_span_info'].get('timestamp') if current_window_frame_data else min_window_start_timestamp
                })
            
            # Process last 1-hour window if it has captions (for video processing mode)
            if hour_window_min_captions:
                hour_start_dt = hour_window_start_dt
                hour_end_dt = current_window_frame_data[-1]['datetime'] if current_window_frame_data else hour_start_dt
                hour_start_day = hour_window_start_day
                hour_end_day = current_window_frame_data[-1]['day_num'] if current_window_frame_data else hour_start_day
                hour_time_span = f"{hour_start_day}-{hour_start_dt.strftime('%H:%M:%S')}-{hour_end_dt.strftime('%H:%M:%S')}"
                
                min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                user_prompt = f"{PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                                                                            
                hour_caption = self.mllm_response(
                    self.video_llm,
                    self.processor,
                    user_prompt,
                    None,  
                    base64_frames=None,
                    max_new_tokens=max_new_tokens,
                    has_image=False
                )
                hour_captions[hour_time_span] = hour_caption
                logger.info(f"Generated 1h caption for final window: {hour_time_span} ({len(hour_window_min_captions)} 10min captions)")
                
                loop.run_until_complete(self.video_segment.upsert({hour_time_span: {"content": hour_caption, "video_frames": current_window_frames, "type": "hour"}}))
                loop.run_until_complete(self._save_video_segments())
                loop.run_until_complete(self.ainsert_streaming_caption({hour_time_span: {"content": hour_caption, "video_frames": current_window_frames, "type": "hour"}}))
                
                if enable_proactive_service:
                    final_timestamp = current_window_frame_data[-1]['time_span_info'].get('timestamp') if current_window_frame_data else hour_window_start_timestamp
                    accumulated_captions['hour_captions'].append({
                        'time_span': hour_time_span,
                        'caption': hour_caption,
                        'timestamp': final_timestamp
                    })
            
      
    async def ainsert_streaming_caption(self, new_video_segments):
        await self._insert_start()
        # 这里不划分chunks，由于每次仅传入一段caption，因此直接对caption提取实体
        captions = [new_video_segments[key]["content"] for key in new_video_segments.keys()][0]
        video_time_span = list(new_video_segments.keys())
        # ENCODER = tiktoken.encoding_for_model("gpt-4o")
        # tokens = ENCODER.encode_batch(captions, num_threads=16)
        client = genai.Client()
        tokens = client.models.count_tokens(
                model="gemini-2.0-flash", contents=captions
            )
        
        caption_dict = {
            "tokens": tokens.total_tokens,
            "content": captions.strip(),
            "chunk_order_index": 0,
            "time_span": [f"{video_time_span[0]}_0"],
        }
        
        inserting_chunks = {compute_mdhash_id(caption_dict["content"], prefix="chunk-"): caption_dict}
        
        _add_chunk_keys = await self.text_chunks.filter_keys(
            list(inserting_chunks.keys())
        )
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }
        if not len(inserting_chunks):
            logger.warning(f"All chunks are already in the storage")
            return
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
        if self.enable_naive_rag:
            logger.info("Insert chunks for naive RAG")
            await self.chunks_vdb.upsert(inserting_chunks)
    
        # ---------- extract/summary entity and upsert to graph
        if [new_video_segments[key]["type"] for key in new_video_segments.keys()][0] == "second":
            logger.info("[Entity Extraction]...")
            maybe_new_kg, _, _ = await streaming_extract_entities(
                list(inserting_chunks.keys())[0],
                inserting_chunks[list(inserting_chunks.keys())[0]],
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                asdict(self),
            )
            
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
        
        # ---------- commit upsertings and indexing
        await self.text_chunks.upsert(inserting_chunks)
    
        await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _save_video_segments(self):
        tasks = []
        for storage_inst in [
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
    
    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
