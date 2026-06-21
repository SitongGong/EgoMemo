import os
import sys
import json
import shutil
import torch
import asyncio
import multiprocessing
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from transformers import AutoModel, AutoTokenizer
import tiktoken
from google import genai

from ._llm import gemini_complete_with_image_sync, gemini_complete_if_cache
from .llm.qwen_vl import mllm_response
from ego_prompt import PROMPTS


from ._llm import (
    LLMConfig,
    openai_config,
    azure_openai_config,
    ollama_config
)
from .streaming_op import (
    chunking_by_video_segments,
    extract_entities,
    get_chunks,
    videorag_query,
    videorag_query_multiple_choice,
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

        try:
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
                    
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise


    def process_frame_with_proactive_service(self, frame_info, accumulated_captions, max_captions_per_level=3, history_messages=None):
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
            history_messages: Optional list of previous conversation messages for iterative calls
            
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
        if not self.gemini_available:
            logger.warning("Gemini model not available. Skipping proactive service detection.")
            return None
        
        if proactive_prompt is None:
            proactive_prompt = PROMPTS["proactive_service_prompt"]
        
        if history_messages is None:
            history_messages = []
        
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
            
            try:
                # Decode base64 string to bytes and create PIL Image
                frame_bytes = base64.b64decode(frame)
                frame_image = Image.open(io.BytesIO(frame_bytes))
            except Exception as e:
                logger.error(f"Failed to prepare frame for Gemini: {e}")
                return {
                    'frame_timestamp': frame_timestamp,
                    'selected_captions': selected_captions,
                    'gemini_response': None,
                    'error': f"Failed to encode frame: {e}",
                    'prompt_used': full_prompt
                }
        
        # Call Gemini model using _llm.py function
        try:
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
                    history_messages=history_messages,
                    use_cache=False,
                    temperature=0.7,
                    max_tokens=8192
                ))
            
            # Try to parse JSON response
            import json
            try:
                # Try to extract JSON from response
                response_text = str(response)
                # Look for JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    parsed_response = json.loads(response_text[json_start:json_end])
                else:
                    parsed_response = response_text
            except:
                parsed_response = response
            
            return {
                'frame_timestamp': frame_timestamp,
                'selected_captions': selected_captions,
                'gemini_response': parsed_response,
                'prompt_used': full_prompt  # Save the prompt used for history
            }
            
        except Exception as e:
            logger.error(f"Error calling Gemini for proactive service detection: {e}")
            return {
                'frame_timestamp': frame_timestamp,
                'selected_captions': selected_captions,
                'gemini_response': None,
                'error': str(e)
            }


    def streaming_graph_construction(self, frame_time_data, interval_seconds=2, 
                       window_seconds=10, gap_threshold_seconds=60, 
                       window_minutes=10, window_hours=1,
                       prompt_template=None, prompt_template_min=None, prompt_template_hour=None,
                       max_new_tokens=512, enable_proactive_service=False, 
                       save_to_videorag=False):
        """
        Construct graph from video frames using streaming input (frame by frame).
        When accumulated frames reach the window time limit, generate caption using Qwen model.
        
        Args:
            frame_time_data: List of dicts with time information for each frame
                           Each dict should have: 'frame', 'datetime', 'day_num', 'time_span_info' keys
            interval_seconds: Interval between sampled frames in seconds (default: 2)
            window_seconds: Window size in seconds for 10s caption generation (default: 10)
            gap_threshold_seconds: Gap threshold to start new window (default: 60)
            window_minutes: Window size in minutes for 10min caption generation (default: 10)
            window_hours: Window size in hours for 1h caption generation (default: 1)
            prompt_template: Optional prompt template for 10s captions. If None, uses default.
            prompt_template_min: Optional prompt template for 10min captions. If None, uses default.
            prompt_template_hour: Optional prompt template for 1h captions. If None, uses default.
            max_new_tokens: Maximum number of tokens to generate (default: 512)
            enable_proactive_service: If True, call process_frame_with_proactive_service for each frame (default: False)
            save_to_videorag: If True, save generated captions to VideoRAG storage (default: False)
            
        Returns:
            Dictionary with generated captions:
            {
                "second_captions": {time_span: caption, ...},
                "min_captions": {time_span: caption, ...},
                "hour_captions": {time_span: caption, ...},
                "proactive_responses": [...]  # Only if enable_proactive_service=True
            }
        """
        if not frame_time_data:
            logger.warning("No frames provided for caption generation")
            return {"second_captions": {}, "min_captions": {}, "hour_captions": {}}
        
        loop = always_get_an_event_loop()
        
        # Default prompts if not provided
        default_prompt_second = PROMPTS["second_caption_prompt"]
        default_prompt_min = PROMPTS["min_caption_prompt"]
        default_prompt_hour = PROMPTS["hour_caption_prompt"]
        
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
        
        # 10-second window state
        current_window_frames = []  # List of base64 frames in current window
        current_window_indices = []  # List of frame indices in current window
        window_start_dt = None
        window_start_day = None
        last_frame_dt = None  # Track last frame's datetime for gap detection
        
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
        
        logger.info(f"Starting streaming processing: {len(frame_time_data)} frames, window={window_seconds}s, interval={interval_seconds}s")
        
        # Process frames one by one (streaming input)
        for frame_idx, frame_info in enumerate(frame_time_data):
            # Call proactive service detection for each frame (if enabled)
            if enable_proactive_service:
                try:
                    # 首先调用Qwen生成该帧的caption
                    caption = self.mllm_response(
                                self.video_llm,
                                self.processor,
                                PROMPTS["frame_caption_user_prompt"],
                                PROMPTS["frame_caption_system_prompt"],
                                base64_frames=frame_info.get('frame'),
                                max_new_tokens=max_new_tokens,
                                has_image=True
                            )
                    
                    # 将生成的caption和之前积累的caption一起使用Gemini生成主动服务结果
                    # 支持迭代调用，直到得到 YES/NO 或达到最大迭代次数
                    max_iterations = 3  # 最大迭代次数
                    iteration_count = 0
                    history_messages = []  # 保存历史上下文
                    final_proactive_result = None
                    
                    while iteration_count < max_iterations:
                        # 准备当前迭代的 prompt（第一次使用默认prompt，后续迭代可以添加检索结果）
                        current_prompt = None if iteration_count == 0 else None
                        
                        # 构建frame_info，包含caption和frame
                        frame_info_for_proactive = {
                            'caption': caption,
                            'frame': frame_info.get('frame'),
                            'time_span_info': frame_info.get('time_span_info', {})
                        }
                        
                        proactive_result = self.process_frame_with_proactive_service(
                            frame_info=frame_info_for_proactive,
                            accumulated_captions=accumulated_captions,
                            proactive_prompt=current_prompt,
                            max_captions_per_level=3,
                            history_messages=history_messages  # 传入历史上下文
                        )
                        
                        if not proactive_result or proactive_result.get('gemini_response') is None:
                            logger.warning(f"Proactive service returned None (iteration {iteration_count + 1})")
                            break
                        
                        # 提取响应内容
                        gemini_response = proactive_result.get('gemini_response')
                        response_text = None
                        
                        # 处理响应可能是字符串或字典的情况
                        if isinstance(gemini_response, str):
                            response_text = gemini_response.strip().upper()
                        elif isinstance(gemini_response, dict):
                            # 尝试从字典中提取文本
                            response_text = str(gemini_response.get('response', gemini_response.get('decision', ''))).strip().upper()
                        else:
                            response_text = str(gemini_response).strip().upper()
                        
                        # 保存当前交互到历史上下文（用于后续迭代）
                        current_prompt_text = proactive_result.get('prompt_used', '')
                        history_messages.append({
                            'role': 'user',
                            'content': current_prompt_text
                        })
                        history_messages.append({
                            'role': 'assistant',
                            'content': str(gemini_response)
                        })
                        
                        # 检查响应内容
                        if response_text in ['YES', 'NO']:
                            # 得到最终决策，跳出循环
                            final_proactive_result = proactive_result
                            logger.info(f"Proactive service decision: {response_text} (iteration {iteration_count + 1})")
                            break
                        elif 'RETRIEVAL' in response_text:
                            # 需要检索，继续迭代
                            iteration_count += 1
                            
                            # 使用正则表达式从响应中提取查询内容
                            # 匹配 <query>...</query> 标签中的内容（不区分大小写，支持多行）
                            query_match = re.search(r'<query>(.*?)</query>', str(gemini_response), re.IGNORECASE | re.DOTALL)
                            retrieval_query = None
                            if query_match:
                                retrieval_query = query_match.group(1).strip()
                                logger.info(f"Proactive service requires retrieval (iteration {iteration_count}). Query: {retrieval_query}")
                            else:
                                logger.warning(f"Proactive service requires retrieval but no query found in response (iteration {iteration_count})")
                            
                            # TODO: 可以在这里添加检索逻辑，使用 retrieval_query 进行检索
                            # 例如：检索相关视频段，添加到 accumulated_captions 或 prompt 中
                            if retrieval_query:
                                # 执行检索操作
                                from .base import QueryParam
                                query_param = QueryParam(mode="videorag")  # 创建查询参数
                                
                                # 使用retrieval_query分别在知识图谱+captions+visual embedding中进行检索
                                retrieved_response = loop.run_until_complete(streaming_videorag_query(
                                    retrieval_query,
                                    self.entities_vdb,
                                    self.text_chunks,
                                    self.chunks_vdb,
                                    self.video_path_db,
                                    self.video_segments,
                                    self.video_segment_feature_vdb,
                                    self.chunk_entity_relation_graph,
                                    self.caption_model, 
                                    self.caption_tokenizer,
                                    query_param,
                                    asdict(self),
                                ))
                                
                                # 将检索结果添加到 history_messages 中，供下一轮迭代使用
                                retrieval_context = f"Retrieved information for query: {retrieval_query}\n\n{retrieved_response}"
                                history_messages.append({
                                    'role': 'user',
                                    'content': retrieval_context
                                })
                                logger.info(f"Added retrieval results to history (iteration {iteration_count})")
                                
                            
                            if iteration_count >= max_iterations:
                                final_proactive_result = proactive_result
                                logger.warning(f"Reached max iterations ({max_iterations}) during RETRIEVAL")
                            continue
                        else:
                            # 其他响应，保存结果但继续迭代
                            final_proactive_result = proactive_result
                            iteration_count += 1
                            if iteration_count >= max_iterations:
                                logger.warning(f"Reached max iterations ({max_iterations}) without YES/NO decision. Last response: {response_text[:50]}")
                            continue
                    
                    if final_proactive_result:
                        proactive_responses.append(final_proactive_result)
                        
                        
                except Exception as e:
                    logger.warning(f"Error in proactive service detection for frame {frame_idx}: {e}")
                    
            base64_frame = frame_info.get('frame')
            frame_day = frame_info.get('day_num')
            start_time = frame_info.get('time_span_info').get('start_time_number')
            end_time = frame_info.get('time_span_info').get('end_time_number')
            frame_timestamp = frame_info.get('time_span_info').get("timestamp")
            
            # Skip frames without valid time information
            if frame_info.get('datetime') is None:
                logger.warning(f"Frame {frame_idx} has no datetime, skipping")
                continue
            
            # Initialize first window
            if window_start_dt is None:
                window_start_dt = frame_timestamp     # 当前视频帧的时间戳作为窗口的开始时间戳
                window_start_day = frame_day
                current_window_frames = [base64_frame]
                current_window_indices = [frame_idx]
                last_frame_dt = end_time       # 当前帧所在批次的结束时间戳作为判断依据
            else:
                # 考虑如下几种情况
                # 1. 当前帧所在clip的开始时间和上一个clip的结束时间是否超过1min
                if last_frame_dt is not None:
                    # Calculate gap: current frame start time - last frame end time
                    gap_duration = calculate_time_diff_seconds(last_frame_dt, start_time)
                    is_gap = gap_duration >= gap_threshold_seconds
                else:
                    is_gap = False
                
                # Check for day change
                day_changed = (frame_day != window_start_day)
                
                # Check if window is full (from window start to current frame)
                # 2. 计算当前帧的时间戳和窗口的开始时间戳的距离
                window_duration = calculate_time_diff_seconds(frame_timestamp, window_start_dt)
                
                # Check if we need to finalize current window and start new one
                if day_changed or is_gap or window_duration >= window_seconds:
                    # Generate caption for current window if it has frames
                    if current_window_frames:
                        start_dt = frame_time_data[current_window_indices[0]]['datetime']
                        end_dt = frame_time_data[current_window_indices[-1]]['datetime']
                        start_day = frame_time_data[current_window_indices[0]]['day_num']
                        end_day = frame_time_data[current_window_indices[-1]]['day_num']
                        
                        time_span = f"Day {start_day}-{start_dt.strftime('%H:%M:%S')} - Day {end_day}-{end_dt.strftime('%H:%M:%S')}"
                        
                        # Generate prompt for 10s caption
                        if callable(default_prompt_second):
                            prompt = default_prompt_second(current_window_frames, time_span)
                        else:
                            prompt = default_prompt_second
                        
                        # Generate 10s caption using Qwen model
                        try:
                            caption = self.mllm_response(
                                self.video_llm,
                                self.processor,
                                prompt, 
                                None, 
                                base64_frames=current_window_frames,
                                max_new_tokens=max_new_tokens,
                                has_image=True
                            )
                            second_captions[time_span] = caption
                            logger.info(f"Generated 10s caption for window: {time_span} ({len(current_window_frames)} frames)")
                            
                            # 将caption信息和视频帧存入到video_segment和video_segment_feature_vdb中
                            loop.run_until_complete(self.video_segments.upsert({"second_captions" + "_" + time_span: caption}))    # 将caption信息存入到video_segment中
                            loop.run_until_complete(self.video_segment_feature_vdb.upsert_video_segment(time_span, current_window_frames))    # 将视频帧存入到video_segment_feature_vdb中
    
                            # Step6: delete the cache file
                            # video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
                            # if os.path.exists(video_segment_cache_path):
                            #     shutil.rmtree(video_segment_cache_path)
                                
                            loop.run_until_complete(self._save_video_segments())
                            
                            # 根据caption进行流式建图
                            loop.run_until_complete(self.ainsert_streaming_caption(self.video_segments._data))
                            
                            # Add to accumulated_captions for proactive service
                            if enable_proactive_service:
                                accumulated_captions['second_captions'].append({
                                    'time_span': time_span,
                                    'caption': caption,
                                    'timestamp': frame_timestamp
                                })
                            
                            # Add to 10-minute window
                            if min_window_start_timestamp is None:
                                min_window_start_timestamp = window_start_dt
                                min_window_start_dt = start_dt
                                min_window_start_day = start_day
                            
                            min_window_second_captions.append({
                                'time_span': time_span,
                                'caption': caption,
                                'start_timestamp': window_start_dt,
                                'end_timestamp': frame_timestamp
                            })
                            
                            # Check if 10-minute window is full
                            min_window_duration = calculate_time_diff_seconds(frame_timestamp, min_window_start_timestamp)
                            if min_window_duration >= window_minutes * 60:
                                # Generate 10-minute caption
                                if min_window_second_captions:
                                    min_start_dt = min_window_start_dt
                                    min_end_dt = end_dt
                                    min_start_day = min_window_start_day
                                    min_end_day = end_day
                                    min_time_span = f"Day {min_start_day}-{min_start_dt.strftime('%H:%M:%S')} - Day {min_end_day}-{min_end_dt.strftime('%H:%M:%S')}"
                                    
                                    # Prepare text input: concatenate all 10s captions
                                    caption_texts = [item['caption'] for item in min_window_second_captions]
                                    caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in min_window_second_captions])
                                    
                                    # Generate prompt for 10min caption
                                    if callable(default_prompt_min):
                                        min_prompt = default_prompt_min(caption_texts, min_time_span)
                                    else:
                                        min_prompt = f"{default_prompt_min}\n\nCaptions:\n{caption_text}"
                                    
                                    try:
                                        min_caption = self.mllm_response(
                                            self.video_llm,
                                            self.processor,
                                            min_prompt,
                                            None, 
                                            base64_frames=None,
                                            max_new_tokens=max_new_tokens,
                                            has_image=False
                                        )
                                        min_captions[min_time_span] = min_caption
                                        logger.info(f"Generated 10min caption for window: {min_time_span} ({len(min_window_second_captions)} 10s captions)")
                                        
                                        # 将caption信息和视频帧存入到video_segment和video_segment_feature_vdb中
                                        loop.run_until_complete(self.video_segment.upsert({"second_captions" + "_" + time_span: caption}))
                                        loop.run_until_complete(self.video_segment_feature_vdb.upsert_video_segment(time_span, current_window_frames))
                
                                        # Step6: delete the cache file
                                        # video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
                                        # if os.path.exists(video_segment_cache_path):
                                        #     shutil.rmtree(video_segment_cache_path)
                                            
                                        loop.run_until_complete(self._save_video_segments())
                                        
                                        loop.run_until_complete(self.ainsert_streaming_caption(self.video_segments._data))
                                            
                                        # Add to accumulated_captions for proactive service
                                        if enable_proactive_service:
                                            accumulated_captions['min_captions'].append({
                                                'time_span': min_time_span,
                                                'caption': min_caption,
                                                'timestamp': frame_timestamp
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
                                            'end_timestamp': frame_timestamp
                                        })
                                        
                                        # Check if 1-hour window is full
                                        hour_window_duration = calculate_time_diff_seconds(frame_timestamp, hour_window_start_timestamp)
                                        if hour_window_duration >= window_hours * 3600:
                                            # Generate 1-hour caption
                                            if hour_window_min_captions:
                                                hour_start_dt = hour_window_start_dt
                                                hour_end_dt = min_end_dt
                                                hour_start_day = hour_window_start_day
                                                hour_end_day = min_end_day
                                                hour_time_span = f"Day {hour_start_day}-{hour_start_dt.strftime('%H:%M:%S')} - Day {hour_end_day}-{hour_end_dt.strftime('%H:%M:%S')}"
                                                
                                                # Prepare text input: concatenate all 10min captions
                                                min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                                                
                                                # Generate prompt for 1h caption
                                                if callable(default_prompt_hour):
                                                    hour_prompt = default_prompt_hour([item['caption'] for item in hour_window_min_captions], hour_time_span)
                                                else:
                                                    hour_prompt = f"{default_prompt_hour}\n\nCaptions:\n{min_caption_text}"
                                                
                                                try:
                                                    hour_caption = self.mllm_response(
                                                        self.video_llm,
                                                        self.processor,
                                                        hour_prompt,
                                                        None, 
                                                        base64_frames=None,
                                                        max_new_tokens=max_new_tokens,
                                                        has_image=False
                                                    )
                                                    hour_captions[hour_time_span] = hour_caption
                                                    logger.info(f"Generated 1h caption for window: {hour_time_span} ({len(hour_window_min_captions)} 10min captions)")
                                                    
                                                    # 将caption信息和视频帧存入到video_segment和video_segment_feature_vdb中
                                                    loop.run_until_complete(self.video_segment.upsert({"second_captions" + "_" + time_span: caption}))
                                                    loop.run_until_complete(self.video_segment_feature_vdb.upsert_video_segment(time_span, current_window_frames))
                            
                                                    # Step6: delete the cache file
                                                    # video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
                                                    # if os.path.exists(video_segment_cache_path):
                                                    #     shutil.rmtree(video_segment_cache_path)
                                                        
                                                    loop.run_until_complete(self._save_video_segments())
                                                    
                                                    loop.run_until_complete(self.ainsert_streaming_caption(self.video_segments._data))
                                                    
                                                    # Add to accumulated_captions for proactive service
                                                    if enable_proactive_service:
                                                        accumulated_captions['hour_captions'].append({
                                                            'time_span': hour_time_span,
                                                            'caption': hour_caption,
                                                            'timestamp': frame_timestamp
                                                        })
                                                    
                                                    # Reset 1-hour window
                                                    hour_window_min_captions = []
                                                    hour_window_start_timestamp = None
                                                    hour_window_start_dt = None
                                                    hour_window_start_day = None
                                                except Exception as e:
                                                    logger.error(f"Error generating 1h caption for window {hour_time_span}: {e}")
                                        
                                    except Exception as e:
                                        logger.error(f"Error generating 10min caption for window {min_time_span}: {e}")
                                    
                                    # Reset 10-minute window
                                    min_window_second_captions = []
                                    min_window_start_timestamp = None
                                    min_window_start_dt = None
                                    min_window_start_day = None
                                    
                        except Exception as e:
                            logger.error(f"Error generating caption for window {time_span}: {e}")
                    
                    # Start new window with current frame
                    window_start_dt = frame_timestamp
                    window_start_day = frame_day
                    current_window_frames = [base64_frame]
                    current_window_indices = [frame_idx]
                    last_frame_dt = end_time
                else:
                    # Add frame to current window
                    current_window_frames.append(base64_frame)
                    current_window_indices.append(frame_idx)
                    last_frame_dt = end_time
        
        # Process last 10s window if it has frames
        if current_window_frames:
            start_dt = frame_time_data[current_window_indices[0]]['datetime']
            end_dt = frame_time_data[current_window_indices[-1]]['datetime']
            start_day = frame_time_data[current_window_indices[0]]['day_num']
            end_day = frame_time_data[current_window_indices[-1]]['day_num']
            last_frame_timestamp = frame_time_data[current_window_indices[-1]]['time_span_info'].get('timestamp')
            
            time_span = f"Day {start_day}-{start_dt.strftime('%H:%M:%S')} - Day {end_day}-{end_dt.strftime('%H:%M:%S')}"
            
            # Generate prompt for 10s caption
            if callable(default_prompt_second):
                prompt = default_prompt_second(current_window_frames, time_span)
            else:
                prompt = default_prompt_second
            
            # Generate caption using Qwen model
            try:
                caption = self.mllm_response(
                    self.video_llm,
                    self.processor,
                    prompt,
                    None, 
                    base64_frames=current_window_frames,
                    max_new_tokens=max_new_tokens,
                    has_image=True
                )
                second_captions[time_span] = caption
                logger.info(f"Generated 10s caption for final window: {time_span} ({len(current_window_frames)} frames)")
                
                # Add to accumulated_captions for proactive service
                if enable_proactive_service:
                    accumulated_captions['second_captions'].append({
                        'time_span': time_span,
                        'caption': caption,
                        'timestamp': last_frame_timestamp
                    })
                
                # Add to 10-minute window
                if min_window_start_timestamp is None:
                    min_window_start_timestamp = window_start_dt
                    min_window_start_dt = start_dt
                    min_window_start_day = start_day
                
                min_window_second_captions.append({
                    'time_span': time_span,
                    'caption': caption,
                    'start_timestamp': window_start_dt,
                    'end_timestamp': last_frame_timestamp
                })
            except Exception as e:
                logger.error(f"Error generating caption for final window {time_span}: {e}")
        
        # Process last 10-minute window if it has captions
        if min_window_second_captions:
            min_start_dt = min_window_start_dt
            min_end_dt = frame_time_data[-1]['datetime'] if frame_time_data else min_start_dt
            min_start_day = min_window_start_day
            min_end_day = frame_time_data[-1]['day_num'] if frame_time_data else min_start_day
            min_time_span = f"Day {min_start_day}-{min_start_dt.strftime('%H:%M:%S')} - Day {min_end_day}-{min_end_dt.strftime('%H:%M:%S')}"
            
            # Prepare text input: concatenate all 10s captions
            caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in min_window_second_captions])
            
            # Generate prompt for 10min caption
            if callable(default_prompt_min):
                min_prompt = default_prompt_min([item['caption'] for item in min_window_second_captions], min_time_span)
            else:
                min_prompt = f"{default_prompt_min}\n\nCaptions:\n{caption_text}"
            
            try:
                min_caption = self.mllm_response(
                    self.video_llm,
                    self.processor,
                    min_prompt,
                    None, 
                    base64_frames=None,
                    max_new_tokens=max_new_tokens,
                    has_image=False
                )
                min_captions[min_time_span] = min_caption
                logger.info(f"Generated 10min caption for final window: {min_time_span} ({len(min_window_second_captions)} 10s captions)")
                
                # Add to accumulated_captions for proactive service
                if enable_proactive_service:
                    final_timestamp = last_frame_timestamp if 'last_frame_timestamp' in locals() else min_window_start_timestamp
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
                    'end_timestamp': last_frame_timestamp if 'last_frame_timestamp' in locals() else min_window_start_timestamp
                })
            except Exception as e:
                logger.error(f"Error generating 10min caption for final window {min_time_span}: {e}")
        
        # Process last 1-hour window if it has captions
        if hour_window_min_captions:
            hour_start_dt = hour_window_start_dt
            hour_end_dt = frame_time_data[-1]['datetime'] if frame_time_data else hour_start_dt
            hour_start_day = hour_window_start_day
            hour_end_day = frame_time_data[-1]['day_num'] if frame_time_data else hour_start_day
            hour_time_span = f"Day {hour_start_day}-{hour_start_dt.strftime('%H:%M:%S')} - Day {hour_end_day}-{hour_end_dt.strftime('%H:%M:%S')}"
            
            # Prepare text input: concatenate all 10min captions
            min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
            
            # Generate prompt for 1h caption
            if callable(default_prompt_hour):
                hour_prompt = default_prompt_hour([item['caption'] for item in hour_window_min_captions], hour_time_span)
            else:
                hour_prompt = f"{default_prompt_hour}\n\nCaptions:\n{min_caption_text}"
            
            try:
                hour_caption = self.mllm_response(
                    self.video_llm,
                    self.processor,
                    hour_prompt,
                    None, 
                    base64_frames=None,
                    max_new_tokens=max_new_tokens,
                    has_image=False
                )
                hour_captions[hour_time_span] = hour_caption
                logger.info(f"Generated 1h caption for final window: {hour_time_span} ({len(hour_window_min_captions)} 10min captions)")
                
                # Add to accumulated_captions for proactive service
                if enable_proactive_service:
                    final_timestamp = last_frame_timestamp if 'last_frame_timestamp' in locals() else hour_window_start_timestamp
                    accumulated_captions['hour_captions'].append({
                        'time_span': hour_time_span,
                        'caption': hour_caption,
                        'timestamp': final_timestamp
                    })
            except Exception as e:
                logger.error(f"Error generating 1h caption for final window {hour_time_span}: {e}")
        
        logger.info(f"Completed streaming processing: generated {len(second_captions)} 10s captions, {len(min_captions)} 10min captions, {len(hour_captions)} 1h captions")
        
        result = {
            "second_captions": second_captions,
            "min_captions": min_captions,
            "hour_captions": hour_captions
        }
        
        if enable_proactive_service:
            result["proactive_responses"] = proactive_responses
            logger.info(f"Generated {len(proactive_responses)} proactive service responses")
        
        # Optionally save to VideoRAG
        if save_to_videorag and self.video_rag:
            self._save_to_videorag(result)
        
        return result
            
      
    async def ainsert_streaming_caption(self, new_video_segments):
        await self._insert_start()
        try:
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
            
        except Exception as e:
            logger.error(f"Error in streaming caption insertion: {e}")
            raise
        finally:
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
            # self.video_path_db,
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
            # self.video_path_db,
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
