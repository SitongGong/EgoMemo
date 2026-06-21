"""
分离的建图和主动服务判断模块

将建图和主动服务判断分成两部分：
1. streaming_graph_construction_only: 只做建图，生成caption并保存
2. process_proactive_service_after_graph: 在caption、visual embedding和构造的图都保存后，进行主动服务判断和检索
"""

import os
import json
import glob
import torch
import asyncio
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from transformers import AutoModel, AutoTokenizer
import tiktoken
from google import genai

from .llm.qwen_vl import mllm_response
from .ego_prompt import PROMPTS
from .holoassist_prompt import HOLOASSIST_PROMPTS

from ._llm import (
    LLMConfig,
    openai_config,
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
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n?```', response, re.DOTALL | re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array or object in the response
    array_match = re.search(r'\[.*?\]', response, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass
    
    object_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return []


def format_timestamp_to_hhmmsscc(timestamp):
    """
    Convert timestamp from integer format (HHMMSSCC) to string format (HH:MM:SS:CC).
    
    Args:
        timestamp: Integer timestamp in format HHMMSSCC (e.g., 10442506)
    
    Returns:
        String in format HH:MM:SS:CC (e.g., "10:44:25:06")
    """
    timestamp_str = str(timestamp).zfill(8)  # Ensure 8 digits with leading zeros
    hours = timestamp_str[0:2]
    minutes = timestamp_str[2:4]
    seconds = timestamp_str[4:6]
    centiseconds = timestamp_str[6:8]
    return f"{hours}:{minutes}:{seconds}:{centiseconds}"


def calculate_time_diff_seconds(timestamp1, timestamp2, datasets_type="egolife"):
    """
    Calculate time difference in seconds between two timestamps.
    
    Args:
        timestamp1: First timestamp
        timestamp2: Second timestamp
        datasets_type: Dataset type, "egolife" for HHMMSSCC format, others for seconds format
    
    Returns:
        Time difference in seconds
    """
    def timestamp_to_seconds(ts, is_egolife):
        
        if is_egolife:
            # Egolife数据集：HHMMSSCC格式（8位整数）
            # 转换为整数并填充到8位
            ts_int = int(ts)
            ts_str = str(ts_int).zfill(8)
            
            # 解析HHMMSSCC格式
            hours = int(ts_str[0:2])
            minutes = int(ts_str[2:4])
            seconds = int(ts_str[4:6])
            centiseconds = int(ts_str[6:8])
            return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0
        else:
            # 其他数据集：直接是秒数（可能是浮点数）
            return float(ts)          
    
    is_egolife = (datasets_type == "egolife")
    return abs(timestamp_to_seconds(timestamp1, is_egolife=True) - timestamp_to_seconds(timestamp2, is_egolife=True))


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


MODEL_MAP = {
    "qwenvl_3_8b_instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "qwenvl_3_4b_instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "qwenvl_2_5_7b_instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwenvl_2_5_3b_instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "minicpm_4_5_v": "openbmb/MiniCPM-V-4_5",
}


class ModelHolder:
    """临时类用于存储模型和处理器"""
    def __init__(self):
        self.video_llm = None
        self.processor = None
        self.image_processor = None


@dataclass
class VideoGraphSeparated:
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
        
        
    def load_caption_model(self, model_name: str, device_map: Optional[Union[str, Dict]] = None):
        # 加载对应的模型
        model_path = next(
            (model_path for key, model_path in MODEL_MAP.items() if key in model_name),
            None
        )
        
        if model_path is None:
            raise ValueError(f"Model '{model_name}' not found in MODEL_MAP. Available models: {list(MODEL_MAP.keys())}")

        # 确定device_map：如果指定了device_map参数，使用它；否则检查CUDA_VISIBLE_DEVICES环境变量
        if device_map is None:
            # 检查是否设置了CUDA_VISIBLE_DEVICES环境变量
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices:
                # 如果设置了CUDA_VISIBLE_DEVICES，明确指定使用cuda:0（因为CUDA_VISIBLE_DEVICES会将指定的物理设备映射为cuda:0）
                device_map = {"": 0}
                logger.info(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} detected, using device_map={{'': 0}}")
            else:
                # 否则使用auto自动分配
                device_map = "auto"
                logger.info("No CUDA_VISIBLE_DEVICES set, using device_map='auto'")
        else:
            logger.info(f"Using specified device_map: {device_map}")
        
        # 检查是否是MiniCPM模型
        if "minicpm" in model_name.lower() or "MiniCPM" in model_path:
            # 使用 MiniCPM 加载方式
            from .llm.minicpm import mllm_response, _load_minicpm_model
            self.mllm_response = mllm_response
            
            model_holder = ModelHolder()
            
            # 加载MiniCPM模型
            video_llm, tokenizer, image_processor = _load_minicpm_model(model_holder, model_path, device_map=device_map)
            
            # MiniCPM使用tokenizer而不是processor
            self.caption_model = video_llm
            self.caption_processor = tokenizer  # MiniCPM使用tokenizer
            self.processor = tokenizer
            self.video_llm = video_llm
            self.image_processor = image_processor
            logger.info(f"Loaded MiniCPM model from {model_path}")
        else:
            # 使用 Qwen-VL 加载方式
            from .llm.qwen_vl import mllm_response
            self.mllm_response = mllm_response
            
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
                        device_map=device_map,
                    )
                except ImportError:
                    from transformers import Qwen2VLForConditionalGeneration
                    video_llm = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        device_map=device_map,
                    )
            elif "Qwen3" in model_path:
                from transformers import Qwen3VLForConditionalGeneration
                video_llm = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=device_map,
                )
            else:
                from transformers import Qwen2VLForConditionalGeneration
                video_llm = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=device_map,
                )
            
            # 加载 processor
            processor = AutoProcessor.from_pretrained(model_path)
            image_processor = processor
            
            self.caption_model = video_llm
            self.caption_processor = processor
            self.processor = processor
            self.video_llm = video_llm
            self.image_processor = image_processor
            logger.info(f"Loaded Qwen-VL model from {model_path}")
    
    def streaming_graph_construction(
        self,
        data_path=None,
        anno_path=None,
        day=None,
        interval_seconds=5,
        window_seconds=30,
        gap_threshold_seconds=60,
        window_minutes=5,
        window_hours=1,
        datasets_type="egolife", 
        max_new_tokens=1024,
    ):
        """
        只进行建图，生成多个阶段的caption并保存，调用self.ainsert_streaming_caption。
        不包含主动服务判断和检索。
        
        时间逻辑说明：
        - 30秒窗口（window_seconds）：累积帧，当窗口满或检测到gap时生成second caption
        - 5分钟窗口（window_minutes）：累积second captions，当窗口满或检测到gap时生成minute caption
        - 1小时窗口（window_hours）：累积minute captions，当窗口满或检测到gap时生成hour caption
        - gap检测：当相邻窗口之间的时间差 >= gap_threshold_seconds 时，先处理当前窗口再开始新窗口
        
        Args:
            data_path: Path to video data directory (e.g., "/path/to/data/DAY1")
            anno_path: Path to annotation directory (e.g., "/path/to/anno/DAY1")
            day: Day identifier (e.g., "DAY1")
            interval_seconds: Interval between sampled frames in seconds (default: 5)
            window_seconds: Window size in seconds for 30s caption generation (default: 30)
            gap_threshold_seconds: Gap threshold to start new window (default: 60)
            window_minutes: Window size in minutes for 5min caption generation (default: 5)
            window_hours: Window size in hours for 1h caption generation (default: 1)
            datasets_type: Dataset type, "egolife" or "holoassist" (default: "egolife")
            max_new_tokens: Maximum number of tokens to generate (default: 1024)
            
        Returns:
            Dictionary with generated captions and accumulated captions:
            {
                "second_captions": {time_span: caption, ...},
                "min_captions": {time_span: caption, ...},
                "hour_captions": {time_span: caption, ...},
                "accumulated_captions": {
                    'second_captions': [{'time_span': str, 'caption': dict, 'timestamp': int}, ...],
                    'min_captions': [{'time_span': str, 'caption': str, 'timestamp': int}, ...],
                    'hour_captions': [{'time_span': str, 'caption': str, 'timestamp': int}, ...]
                }
            }
        """
        # Helper functions for time processing
        def get_time_span_info(anno_file: str) -> Optional[Dict[str, any]]:
            """Get detailed time span information from annotation file."""
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

        def time_number_to_datetime(time_number, day_num: int, is_egolife: bool = True) -> Optional[datetime]:
            """
            Convert time_number to datetime.
            
            Args:
                time_number: For egolife, HHMMSSCC format (e.g., 11094300); for others, seconds (e.g., 117.5)
                day_num: Day number
                is_egolife: Whether this is egolife dataset (default: True for backward compatibility)
            
            Returns:
                datetime object or None if parsing fails
            """
            try:
                if is_egolife:
                    # Egolife数据集：HHMMSSCC格式（8位整数）
                    time_str = str(int(time_number)).zfill(8)
                    hours = int(time_str[0:2])
                    minutes = int(time_str[2:4])
                    seconds = int(time_str[4:6])
                    base_date = datetime(2000, 1, 1)
                    return base_date + timedelta(days=day_num-1, hours=hours, minutes=minutes, seconds=seconds)
                else:
                    # 其他数据集：直接是秒数（可能是浮点数）
                    total_seconds = float(time_number)
                    base_date = datetime(2000, 1, 1)
                    return base_date + timedelta(days=day_num-1, seconds=total_seconds)
            except Exception as e:
                logger.warning(f"Failed to parse time_number {time_number} (type: {type(time_number)}, is_egolife: {is_egolife}): {e}")
                return None

        def get_day_number(day_str: str) -> int:
            """Extract day number from day string (e.g., 'DAY1' -> 1)."""
            try:
                return int(day_str.replace("DAY", ""))
            except Exception:
                return 0
        
        loop = always_get_an_event_loop()
        
        # 定义保存和加载状态的辅助函数
        def save_checkpoint_state(accumulated_captions, window_states, captions_dict=None):
            """保存accumulated_captions、窗口状态和captions字典到文件"""
            checkpoint_file = os.path.join(self.working_dir, "streaming_checkpoint.json")
            try:
                # 将datetime对象转换为字符串
                checkpoint_data = {
                    "accumulated_captions": accumulated_captions,
                    "window_states": {},
                    "captions_dict": captions_dict if captions_dict is not None else {}
                }
                
                # 转换窗口状态中的datetime对象
                for key, value in window_states.items():
                    if isinstance(value, datetime):
                        checkpoint_data["window_states"][key] = value.isoformat()
                    elif key == 'current_window_frame_data' and isinstance(value, list):
                        # 处理current_window_frame_data中的datetime对象
                        serialized_list = []
                        for item in value:
                            if isinstance(item, dict):
                                serialized_item = item.copy()
                                if 'datetime' in serialized_item and isinstance(serialized_item['datetime'], datetime):
                                    serialized_item['datetime'] = serialized_item['datetime'].isoformat()
                                serialized_list.append(serialized_item)
                            else:
                                serialized_list.append(item)
                        checkpoint_data["window_states"][key] = serialized_list
                    elif key in ['min_window_second_captions', 'hour_window_min_captions'] and isinstance(value, list):
                        # 处理min_window_second_captions和hour_window_min_captions中的end_dt datetime对象
                        serialized_list = []
                        for item in value:
                            if isinstance(item, dict):
                                serialized_item = item.copy()
                                if 'end_dt' in serialized_item and isinstance(serialized_item['end_dt'], datetime):
                                    serialized_item['end_dt'] = serialized_item['end_dt'].isoformat()
                                serialized_list.append(serialized_item)
                            else:
                                serialized_list.append(item)
                        checkpoint_data["window_states"][key] = serialized_list
                    elif key == 'frame_time_ranges_list' and isinstance(value, list):
                        # 将元组列表转换为列表列表（JSON可序列化）
                        checkpoint_data["window_states"][key] = [list(item) if isinstance(item, tuple) else item for item in value]
                    else:
                        checkpoint_data["window_states"][key] = value
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Checkpoint saved to {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        
        def load_checkpoint_state():
            """从文件加载accumulated_captions、窗口状态和captions字典"""
            checkpoint_file = os.path.join(self.working_dir, "streaming_checkpoint.json")
            if not os.path.exists(checkpoint_file):
                return None, None, None
            
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                accumulated_captions = checkpoint_data.get("accumulated_captions", {
                    'second_captions': [],
                    'min_captions': [],
                    'hour_captions': []
                })
                
                # 加载captions字典
                captions_dict = checkpoint_data.get("captions_dict", {
                    'second_captions': {},
                    'min_captions': {},
                    'hour_captions': {}
                })
                
                window_states_raw = checkpoint_data.get("window_states", {})
                window_states = {}
                
                # 恢复datetime对象
                datetime_keys = ['window_start_dt', 'last_frame_end_time', 'min_window_start_dt', 'hour_window_start_dt']
                for key in datetime_keys:
                    if key in window_states_raw and window_states_raw[key] is not None:
                        try:
                            window_states[key] = datetime.fromisoformat(window_states_raw[key])
                        except (ValueError, TypeError):
                            window_states[key] = None
                    else:
                        window_states[key] = None
                
                # 处理current_window_frame_data中的datetime对象
                if 'current_window_frame_data' in window_states_raw and isinstance(window_states_raw['current_window_frame_data'], list):
                    restored_list = []
                    for item in window_states_raw['current_window_frame_data']:
                        if isinstance(item, dict) and 'datetime' in item and isinstance(item['datetime'], str):
                            restored_item = item.copy()
                            try:
                                restored_item['datetime'] = datetime.fromisoformat(item['datetime'])
                            except (ValueError, TypeError):
                                restored_item['datetime'] = None
                            restored_list.append(restored_item)
                        else:
                            restored_list.append(item)
                    window_states['current_window_frame_data'] = restored_list
                else:
                    window_states['current_window_frame_data'] = window_states_raw.get('current_window_frame_data', [])
                
                # 处理frame_time_ranges_list：将列表列表转换回元组列表
                if 'frame_time_ranges_list' in window_states_raw and isinstance(window_states_raw['frame_time_ranges_list'], list):
                    restored_ranges = []
                    for item in window_states_raw['frame_time_ranges_list']:
                        if isinstance(item, list) and len(item) == 2:
                            restored_ranges.append(tuple(item))
                        elif isinstance(item, tuple):
                            restored_ranges.append(item)
                        else:
                            restored_ranges.append(item)
                    window_states['frame_time_ranges_list'] = restored_ranges
                else:
                    window_states['frame_time_ranges_list'] = []
                
                # 处理min_window_second_captions中的end_dt datetime对象
                if 'min_window_second_captions' in window_states_raw and isinstance(window_states_raw['min_window_second_captions'], list):
                    restored_min_captions = []
                    for item in window_states_raw['min_window_second_captions']:
                        if isinstance(item, dict) and 'end_dt' in item and isinstance(item['end_dt'], str):
                            restored_item = item.copy()
                            try:
                                restored_item['end_dt'] = datetime.fromisoformat(item['end_dt'])
                            except (ValueError, TypeError):
                                restored_item['end_dt'] = None
                            restored_min_captions.append(restored_item)
                        else:
                            restored_min_captions.append(item)
                    window_states['min_window_second_captions'] = restored_min_captions
                else:
                    window_states['min_window_second_captions'] = []
                
                # 处理hour_window_min_captions中的end_dt datetime对象
                if 'hour_window_min_captions' in window_states_raw and isinstance(window_states_raw['hour_window_min_captions'], list):
                    restored_hour_captions = []
                    for item in window_states_raw['hour_window_min_captions']:
                        if isinstance(item, dict) and 'end_dt' in item and isinstance(item['end_dt'], str):
                            restored_item = item.copy()
                            try:
                                restored_item['end_dt'] = datetime.fromisoformat(item['end_dt'])
                            except (ValueError, TypeError):
                                restored_item['end_dt'] = None
                            restored_hour_captions.append(restored_item)
                        else:
                            restored_hour_captions.append(item)
                    window_states['hour_window_min_captions'] = restored_hour_captions
                else:
                    window_states['hour_window_min_captions'] = []
                
                # 其他状态直接复制
                other_keys = ['current_window_frames', 'window_start_day', 
                             'window_start_timestamp', 'min_window_start_day',
                             'min_window_start_timestamp', 'hour_window_start_day',
                             'hour_window_start_timestamp']
                for key in other_keys:
                    window_states[key] = window_states_raw.get(key, None if 'timestamp' in key or 'day' in key else [])
                
                logger.info(f"Checkpoint loaded from {checkpoint_file}")
                logger.info(f"Loaded {len(accumulated_captions['second_captions'])} second captions, "
                          f"{len(accumulated_captions['min_captions'])} min captions, "
                          f"{len(accumulated_captions['hour_captions'])} hour captions")
                logger.info(f"Loaded {len(window_states.get('frame_time_ranges_list', []))} processed frame time ranges")
                logger.info(f"Loaded {len(captions_dict.get('second_captions', {}))} second caption dicts, "
                          f"{len(captions_dict.get('min_captions', {}))} min caption dicts, "
                          f"{len(captions_dict.get('hour_captions', {}))} hour caption dicts")
                return accumulated_captions, window_states, captions_dict
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
                return None, None, None
        
        # Store results
        # 尝试加载之前保存的状态
        loaded_accumulated_captions, loaded_window_states, loaded_captions_dict = load_checkpoint_state()
        
        # 恢复captions字典
        if loaded_captions_dict is not None:
            second_captions = loaded_captions_dict.get('second_captions', {})
            min_captions = loaded_captions_dict.get('min_captions', {})
            hour_captions = loaded_captions_dict.get('hour_captions', {})
            logger.info("Resuming from previous checkpoint: loaded captions dictionaries")
        else:
            second_captions = {}
            min_captions = {}
            hour_captions = {}
        
        # Maintain accumulated captions for proactive service (ordered by time)
        if loaded_accumulated_captions is not None:
            accumulated_captions = loaded_accumulated_captions
            logger.info("Resuming from previous checkpoint: loaded accumulated_captions")
        else:
            accumulated_captions = {
                'second_captions': [],  # List of dicts: {'time_span': str, 'caption': dict, 'timestamp': int}
                'min_captions': [],
                'hour_captions': []
            }
        
        # 10-second window state
        if loaded_window_states is not None:
            current_window_frames = loaded_window_states.get('current_window_frames', [])
            current_window_frame_data = loaded_window_states.get('current_window_frame_data', [])
            window_start_dt = loaded_window_states.get('window_start_dt')
            window_start_day = loaded_window_states.get('window_start_day')
            window_start_timestamp = loaded_window_states.get('window_start_timestamp')
            last_frame_end_time = loaded_window_states.get('last_frame_end_time')
            logger.info("Resuming from previous checkpoint: loaded 10-second window state")
        else:
            current_window_frames = []
            current_window_frame_data = []
            window_start_dt = None
            window_start_day = None
            window_start_timestamp = None
            last_frame_end_time = None
        
        # 10-minute window state
        if loaded_window_states is not None:
            min_window_second_captions = loaded_window_states.get('min_window_second_captions', [])
            min_window_start_dt = loaded_window_states.get('min_window_start_dt')
            min_window_start_day = loaded_window_states.get('min_window_start_day')
            min_window_start_timestamp = loaded_window_states.get('min_window_start_timestamp')
            logger.info("Resuming from previous checkpoint: loaded 10-minute window state")
        else:
            min_window_second_captions = []
            min_window_start_dt = None
            min_window_start_day = None
            min_window_start_timestamp = None
        
        # 1-hour window state
        if loaded_window_states is not None:
            hour_window_min_captions = loaded_window_states.get('hour_window_min_captions', [])
            hour_window_start_dt = loaded_window_states.get('hour_window_start_dt')
            hour_window_start_day = loaded_window_states.get('hour_window_start_day')
            hour_window_start_timestamp = loaded_window_states.get('hour_window_start_timestamp')
            logger.info("Resuming from previous checkpoint: loaded 1-hour window state")
        else:
            hour_window_min_captions = []
            hour_window_start_dt = None
            hour_window_start_day = None
            hour_window_start_timestamp = None
        
        # 已处理的frame_time_ranges列表
        if loaded_window_states is not None:
            frame_time_ranges_list = loaded_window_states.get('frame_time_ranges_list', [])
            logger.info(f"Resuming from previous checkpoint: loaded {len(frame_time_ranges_list)} processed frame time ranges")
        else:
            frame_time_ranges_list = []
        
        # Process videos if data_path and anno_path are provided
        if datasets_type == "egolife":
        # if data_path and anno_path and day:
            # 对于EgoLife数据集，同时提取视频和标注文件
            days_list = sorted(glob.glob(os.path.join(data_path, "*.mp4")))
            anno_list = sorted(glob.glob(os.path.join(anno_path, "*.json")))
        elif datasets_type == "holoassist":
            days_list = [os.path.join(data_path, "Video_pitchshift.mp4")]
            anno_list = []
        else:
            days_list = sorted(glob.glob(os.path.join(data_path, "*.mp4")))      # 对于其他数据集，应该只有一个视频文件
            anno_list = []
        
            if not days_list:
                raise ValueError(f"No video files found in {data_path}")
            
        if datasets_type == "egolife":
            day_num = get_day_number(day)
            logger.info(f"Starting streaming graph construction (build only): {len(days_list)} videos, window={window_seconds}s, interval={interval_seconds}s")
        else:
            day_num = 1
        
        # Process each video one by one
        for video_idx, video_path in enumerate(days_list):     # 读取视频和注释文件
            video_name = os.path.basename(video_path).split(".")[0]
            
            # 如果是egolife数据集，才提供注释文件
            if datasets_type == "egolife":
                anno_file = anno_list[video_idx]
                anno_name = os.path.basename(anno_file).split(".")[0]
                assert video_name == anno_name, f"Video name {video_name} and annotation name {anno_name} do not match"
                
                # Extract time span from annotation file
                time_span_info = get_time_span_info(anno_file)
                if not time_span_info:
                    logger.warning(f"No time span info for {video_name}, skipping")
                    continue
                
                # 每interval_seconds提取一帧，同时获取各帧的时间范围
                base64_frames, frame_timestamps, frame_time_ranges = sample_frames_by_interval(
                    video_path,
                    interval_seconds=interval_seconds,
                    output_format='base64',
                    time_span_info=time_span_info
                )
            else:
                # 每interval_seconds提取一帧，同时获取各帧的时间范围
                time_span_info = None
                base64_frames, frame_timestamps, frame_time_ranges = sample_frames_by_interval(
                    video_path,
                    interval_seconds=interval_seconds,
                    output_format='base64',
                    time_span_info=time_span_info, 
                )
            
            logger.info(f"Processing video {video_idx + 1}/{len(days_list)}: {video_name}")
            
            if not base64_frames:
                logger.warning(f"No frames extracted from {video_name}")
                continue
            
            # Process frames one by one (streaming)
            for i, frame in enumerate(base64_frames):
                # 检查该帧的frame_time_ranges是否已经处理过
                if i < len(frame_time_ranges):
                    current_frame_time_range = frame_time_ranges[i]
                    # 将frame_time_range转换为可比较的格式（元组）
                    frame_range_key = (current_frame_time_range.get('start'), current_frame_time_range.get('end'))
                    
                    # 检查是否已经处理过
                    if frame_range_key in frame_time_ranges_list:
                        logger.info(f"Frame {i} in video {video_name} already processed (start={current_frame_time_range.get('start')}, end={current_frame_time_range.get('end')}), skipping")
                        continue
                
                # Get timestamp for this frame
                if frame_timestamps and i < len(frame_timestamps):
                    timestamp = frame_timestamps[i]
                
                # Convert timestamp to datetime
                frame_dt = time_number_to_datetime(timestamp, day_num, is_egolife=True)
                if frame_dt is None:
                    if datasets_type == "egolife" and time_span_info:
                        start_time_number = time_span_info.get('start_time_number', 0)
                        start_dt = time_number_to_datetime(start_time_number, day_num, is_egolife=True)
                    else:
                        # 对于非egolife数据集，从0开始计算
                        start_dt = time_number_to_datetime(0, day_num, is_egolife=True)
                    if start_dt:
                        frame_dt = start_dt + timedelta(seconds=i * interval_seconds)
                
                if not frame_dt:
                    logger.warning(f"Failed to get datetime for frame {i} in {video_name}, skipping")
                    continue
                
                # Create frame info dict
                if time_span_info is not None:
                    time_span_info = {
                        **time_span_info,
                        "timestamp": timestamp,
                    }
                else:
                    time_span_info = {
                        "timestamp": timestamp,
                    }
                frame_info = {
                    'frame': frame,
                    'datetime': frame_dt,
                    'day_num': day_num,
                    'video_name': video_name,
                    'frame_idx': i,
                    'start_time': frame_time_ranges[i]['start'],
                    'end_time': frame_time_ranges[i]['end'],
                    'time_span_info': time_span_info, 
                }
                
                # Initialize first window or check for gap/day change
                if window_start_timestamp is None:
                    window_start_timestamp = timestamp
                    window_start_dt = frame_dt
                    window_start_day = day_num
                    current_window_frames = [frame]
                    current_window_frame_data = [frame_info]
                    # 使用当前帧的end_time作为last_frame_end_time（用于gap检测）
                    last_frame_end_time = frame_info.get('end_time', 0) if datasets_type == "egolife" else timestamp
                else:
                    # Check for gap or day change
                    # 获取当前帧的start_time用于gap检测
                    current_frame_start_time = frame_info.get('start_time', 0) if datasets_type == "egolife" else timestamp
                    gap_duration = 0
                    if last_frame_end_time is not None:
                        gap_duration = calculate_time_diff_seconds(current_frame_start_time, last_frame_end_time, datasets_type)
                    is_gap = gap_duration >= gap_threshold_seconds
                    day_changed = (day_num != window_start_day)
                    
                    # Check if window is full
                    window_duration = calculate_time_diff_seconds(timestamp, window_start_timestamp, datasets_type)
                    
                    # If gap, day change, or window full, process current window
                    if day_changed or is_gap or window_duration >= window_seconds:
                        # Generate caption for current window if it has frames
                        if current_window_frames:
                            start_dt = current_window_frame_data[0]['datetime']
                            end_dt = current_window_frame_data[-1]['datetime']
                            start_day = current_window_frame_data[0]['day_num']
                            last_frame_timestamp = current_window_frame_data[-1]['time_span_info'].get('timestamp')
                            
                            time_span = f"{start_day}-{start_dt.strftime('%H:%M:%S')}-{end_dt.strftime('%H:%M:%S')}"
                            
                            # 检查该time_span的caption是否已经生成（避免重复生成）
                            if time_span in second_captions:
                                logger.info(f"Caption for time_span {time_span} already exists, skipping generation")
                                # 如果caption已存在，从second_captions中获取
                                caption_dict = second_captions[time_span]
                            else:
                                # Generate 10s caption using Qwen model with retry mechanism
                                if datasets_type == "egolife":
                                    caption_prompt = PROMPTS["simple_second_caption_system_prompt"]
                                else:
                                    caption_prompt = HOLOASSIST_PROMPTS["simple_second_caption_system_prompt"]
                                max_retries = 3
                                caption_text = None
                                caption = None
                                
                                caption_dict = None
                                caption = None
                                for retry_count in range(max_retries):
                                    try:
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
                                        caption = json.loads(caption_text)
                                        # 检查是否包含 "frames" 键且为字典类型
                                        if not (isinstance(caption, dict) and "frames" in caption and isinstance(caption["frames"], dict)):
                                            logger.warning(f"Caption format invalid (attempt {retry_count + 1}/{max_retries}): missing 'frames' key or invalid structure")
                                            if retry_count == max_retries - 1:
                                                raise ValueError(f"Caption format invalid after {max_retries} attempts: missing 'frames' key or invalid structure")
                                            continue
                                        
                                        # Qwen输出了各帧的caption，需要将它们整合为一个字典
                                        caption_dict = {"dense_caption": {}, "description": caption.get("caption", "")}
                                        for frame_idx, content in caption["frames"].items():
                                            start_time_formatted = format_timestamp_to_hhmmsscc(current_window_frame_data[int(frame_idx)]['start_time'])
                                            end_time_formatted = format_timestamp_to_hhmmsscc(current_window_frame_data[int(frame_idx)]['end_time'])
                                            caption_dict["dense_caption"][f"DAY{day_num}-{start_time_formatted}-{end_time_formatted}"] = content
                                        
                                        # 如果成功处理，跳出循环
                                        break
                                        
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse caption as JSON (attempt {retry_count + 1}/{max_retries}): {e}")
                                        if retry_count == max_retries - 1:
                                            raise ValueError(f"Failed to parse caption as JSON after {max_retries} attempts: {e}")
                                    except (KeyError, IndexError, ValueError, TypeError) as e:
                                        logger.warning(f"Failed to process caption dictionary (attempt {retry_count + 1}/{max_retries}): {e}")
                                        if retry_count == max_retries - 1:
                                            raise ValueError(f"Failed to process caption dictionary after {max_retries} attempts: {e}")
                                
                                # 保存处理后的caption
                                second_captions[time_span] = caption_dict
                                logger.info(f"Generated 10s caption for window: {time_span} ({len(current_window_frames)} frames)")
                                
                                # 将caption信息和视频帧存入到video_segment和video_segment_feature_vdb中
                                loop.run_until_complete(self.video_segments.upsert({time_span: {"content": json.dumps(caption_dict), "video_frames": current_window_frames, "type": "second"}}))
                                loop.run_until_complete(self.video_segment_feature_vdb.upsert_video_segment(time_span, current_window_frames))
                                loop.run_until_complete(self._save_video_segments())
                                loop.run_until_complete(self.ainsert_streaming_caption({time_span: {"content": json.dumps(caption_dict), "video_frames": current_window_frames, "type": "second"}}))
                                
                                # Add to accumulated_captions for proactive service (but don't call it here)
                                accumulated_captions['second_captions'].append({
                                    'time_span': time_span,
                                    'caption': json.dumps(caption_dict),
                                    'timestamp': last_frame_timestamp,
                                })
                            
                            # 保存检查点
                            window_states = {
                                'current_window_frames': current_window_frames,
                                'current_window_frame_data': current_window_frame_data,
                                'window_start_dt': window_start_dt,
                                'window_start_day': window_start_day,
                                'window_start_timestamp': window_start_timestamp,
                                'last_frame_end_time': last_frame_end_time,
                                'min_window_second_captions': min_window_second_captions,
                                'min_window_start_dt': min_window_start_dt,
                                'min_window_start_day': min_window_start_day,
                                'min_window_start_timestamp': min_window_start_timestamp,
                                'hour_window_min_captions': hour_window_min_captions,
                                'hour_window_start_dt': hour_window_start_dt,
                                'hour_window_start_day': hour_window_start_day,
                                'hour_window_start_timestamp': hour_window_start_timestamp,
                                'frame_time_ranges_list': frame_time_ranges_list,
                            }
                            captions_dict = {
                                'second_captions': second_captions,
                                'min_captions': min_captions,
                                'hour_captions': hour_captions,
                            }
                            save_checkpoint_state(accumulated_captions, window_states, captions_dict)
                            
                            # Check for gap before adding to 10-minute window
                            # If there's a gap between the last second caption in min_window and the current one,
                            # process the current min_window first
                            if min_window_second_captions:
                                last_min_window_end_timestamp = min_window_second_captions[-1]['end_timestamp']
                                gap_between_second_captions = calculate_time_diff_seconds(window_start_timestamp, last_min_window_end_timestamp, datasets_type)
                                if gap_between_second_captions >= gap_threshold_seconds:
                                    # There's a gap, process current min_window first
                                    if min_window_second_captions:
                                        last_min_caption_end_timestamp = min_window_second_captions[-1]['end_timestamp']
                                        last_min_caption_end_dt = min_window_second_captions[-1].get('end_dt', end_dt)
                                        
                                        min_start_dt = min_window_start_dt
                                        min_end_dt = last_min_caption_end_dt
                                        min_start_day = min_window_start_day
                                        min_time_span = f"{min_start_day}-{min_start_dt.strftime('%H:%M:%S')}-{min_end_dt.strftime('%H:%M:%S')}"
                                        
                                        caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in min_window_second_captions])
                                        
                                        if datasets_type == "egolife":
                                            user_prompt = f"{PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                        else:
                                            user_prompt = f"{HOLOASSIST_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                     
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
                                        logger.info(f"Generated 10min caption for window (gap detected): {min_time_span} ({len(min_window_second_captions)} 10s captions)")
                                        
                                        loop.run_until_complete(self.video_segments.upsert({min_time_span: {"content": min_caption, "sub_window_captions": [item["time_span"] for item in min_window_second_captions], "type": "minute"}}))
                                        loop.run_until_complete(self._save_video_segments())
                                        loop.run_until_complete(self.ainsert_streaming_caption({min_time_span: {"content": min_caption, "sub_window_captions": [item["time_span"] for item in min_window_second_captions], "type": "minute"}}))
                                        
                                        accumulated_captions['min_captions'].append({
                                            'time_span': min_time_span,
                                            'caption': min_caption,
                                            'timestamp': last_min_caption_end_timestamp
                                        })
                                        
                                        # Check for gap before adding to hour_window
                                        if hour_window_min_captions:
                                            last_hour_window_end_timestamp = hour_window_min_captions[-1]['end_timestamp']
                                            gap_between_min_captions = calculate_time_diff_seconds(min_window_start_timestamp, last_hour_window_end_timestamp, datasets_type)
                                            if gap_between_min_captions >= gap_threshold_seconds:
                                                # There's a gap, process current hour_window first
                                                if hour_window_min_captions:
                                                    last_hour_caption_end_timestamp = hour_window_min_captions[-1]['end_timestamp']
                                                    last_hour_caption_end_dt = hour_window_min_captions[-1].get('end_dt', min_end_dt)
                                                    
                                                    hour_start_dt = hour_window_start_dt
                                                    hour_end_dt = last_hour_caption_end_dt
                                                    hour_start_day = hour_window_start_day
                                                    hour_end_day = hour_window_start_day
                                                    hour_time_span = f"{hour_start_day}-{hour_start_dt.strftime('%H:%M:%S')}-{hour_end_dt.strftime('%H:%M:%S')}"
                                                    
                                                    min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                                                    if datasets_type == "egolife":
                                                        user_prompt = f"{PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                    else:
                                                        user_prompt = f"{HOLOASSIST_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                    
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
                                                    logger.info(f"Generated 1h caption for window (gap detected): {hour_time_span} ({len(hour_window_min_captions)} 10min captions)")
                                                    
                                                    loop.run_until_complete(self.video_segments.upsert({hour_time_span: {"content": hour_caption, "sub_window_captions": [window_caption["time_span"] for window_caption in hour_window_min_captions], "type": "hour"}}))
                                                    loop.run_until_complete(self._save_video_segments())
                                                    loop.run_until_complete(self.ainsert_streaming_caption({hour_time_span: {"content": hour_caption, "sub_window_captions": [window_caption["time_span"] for window_caption in hour_window_min_captions], "type": "hour"}}))
                                                    
                                                    accumulated_captions['hour_captions'].append({
                                                        'time_span': hour_time_span,
                                                        'caption': hour_caption,
                                                        'timestamp': last_hour_caption_end_timestamp
                                                    })
                                                    
                                                    # Reset 1-hour window
                                                    hour_window_min_captions = []
                                                    hour_window_start_timestamp = None
                                                    hour_window_start_dt = None
                                                    hour_window_start_day = None
                                        
                                        # Add to 1-hour window (after gap check)
                                        if hour_window_start_timestamp is None:
                                            hour_window_start_timestamp = min_window_start_timestamp
                                            hour_window_start_dt = min_start_dt
                                            hour_window_start_day = min_start_day
                                        
                                        hour_window_min_captions.append({
                                            'time_span': min_time_span,
                                            'caption': min_caption,
                                            'start_timestamp': min_window_start_timestamp,
                                            'end_timestamp': last_min_caption_end_timestamp
                                        })
                                        
                                        # Reset 10-minute window
                                        min_window_second_captions = []
                                        min_window_start_timestamp = None
                                        min_window_start_dt = None
                                        min_window_start_day = None
                            
                            # Add to 10-minute window
                            if min_window_start_timestamp is None:
                                min_window_start_timestamp = window_start_timestamp
                                min_window_start_dt = start_dt
                                min_window_start_day = start_day
                            
                            min_window_second_captions.append({
                                'time_span': time_span,
                                'caption': json.dumps(caption_dict),  # 使用caption_dict而不是caption
                                'start_timestamp': window_start_timestamp,
                                'end_timestamp': last_frame_timestamp,
                                'end_dt': end_dt  # Store end_dt for gap detection
                            })
                            
                            # Check if 10-minute window is full
                            min_window_duration = calculate_time_diff_seconds(last_frame_timestamp, min_window_start_timestamp, datasets_type)
                            if min_window_duration >= window_minutes * 60:
                                # Generate 10-minute caption
                                if min_window_second_captions:
                                    min_start_dt = min_window_start_dt
                                    min_end_dt = end_dt
                                    min_start_day = min_window_start_day
                                    # Get end_day from current_window_frame_data (same source as end_dt)
                                    min_end_day = current_window_frame_data[-1]['day_num']
                                    min_time_span = f"{min_start_day}-{min_start_dt.strftime('%H:%M:%S')}-{min_end_dt.strftime('%H:%M:%S')}"
                                    
                                    caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in min_window_second_captions])
                                    
                                    if datasets_type == "egolife":
                                        user_prompt = f"{PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                    else:
                                        user_prompt = f"{HOLOASSIST_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                 
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
                                    
                                    loop.run_until_complete(self.video_segments.upsert({min_time_span: {"content": min_caption, "sub_window_captions": [caption_dict_["time_span"] for caption_dict_ in min_window_second_captions], "type": "minute"}}))
                                    loop.run_until_complete(self._save_video_segments())
                                    loop.run_until_complete(self.ainsert_streaming_caption({min_time_span: {"content": min_caption, "sub_window_captions": [caption_dict_["time_span"] for caption_dict_ in min_window_second_captions], "type": "minute"}}))
                                    
                                    accumulated_captions['min_captions'].append({
                                        'time_span': min_time_span,
                                        'caption': min_caption,
                                        'timestamp': last_frame_timestamp
                                    })
                                    
                                    # 保存检查点
                                    window_states = {
                                        'current_window_frames': current_window_frames,
                                        'current_window_frame_data': current_window_frame_data,
                                        'window_start_dt': window_start_dt,
                                        'window_start_day': window_start_day,
                                        'window_start_timestamp': window_start_timestamp,
                                        'last_frame_end_time': last_frame_end_time,
                                        'min_window_second_captions': min_window_second_captions,
                                        'min_window_start_dt': min_window_start_dt,
                                        'min_window_start_day': min_window_start_day,
                                        'min_window_start_timestamp': min_window_start_timestamp,
                                        'hour_window_min_captions': hour_window_min_captions,
                                        'hour_window_start_dt': hour_window_start_dt,
                                        'hour_window_start_day': hour_window_start_day,
                                        'hour_window_start_timestamp': hour_window_start_timestamp,
                                        'frame_time_ranges_list': frame_time_ranges_list,
                                    }
                                    captions_dict = {
                                        'second_captions': second_captions,
                                        'min_captions': min_captions,
                                        'hour_captions': hour_captions,
                                    }
                                    save_checkpoint_state(accumulated_captions, window_states, captions_dict)
                                    
                                    # Check for gap before adding to 1-hour window
                                    # If there's a gap between the last min caption in hour_window and the current one,
                                    # process the current hour_window first
                                    if hour_window_min_captions:
                                        last_hour_window_end_timestamp = hour_window_min_captions[-1]['end_timestamp']
                                        gap_between_min_captions = calculate_time_diff_seconds(min_window_start_timestamp, last_hour_window_end_timestamp, datasets_type)
                                        if gap_between_min_captions >= gap_threshold_seconds:
                                            # There's a gap, process current hour_window first
                                            if hour_window_min_captions:
                                                last_hour_caption_end_timestamp = hour_window_min_captions[-1]['end_timestamp']
                                                last_hour_caption_end_dt = hour_window_min_captions[-1].get('end_dt', min_end_dt)
                                                if last_hour_caption_end_dt is None:
                                                    last_hour_caption_end_dt = min_end_dt
                                                
                                                hour_start_dt = hour_window_start_dt
                                                hour_end_dt = last_hour_caption_end_dt
                                                hour_start_day = hour_window_start_day
                                                hour_end_day = hour_window_start_day
                                                hour_time_span = f"{hour_start_day}-{hour_start_dt.strftime('%H:%M:%S')}-{hour_end_dt.strftime('%H:%M:%S')}"
                                                
                                                min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                                                if datasets_type == "egolife":
                                                    user_prompt = f"{PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                else:
                                                    user_prompt = f"{HOLOASSIST_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                
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
                                                logger.info(f"Generated 1h caption for window (gap detected): {hour_time_span} ({len(hour_window_min_captions)} 10min captions)")
                                                
                                                loop.run_until_complete(self.video_segments.upsert({hour_time_span: {"content": hour_caption, "sub_window_captions": [window_caption["time_span"] for window_caption in hour_window_min_captions], "type": "hour"}}))
                                                loop.run_until_complete(self._save_video_segments())
                                                loop.run_until_complete(self.ainsert_streaming_caption({hour_time_span: {"content": hour_caption, "sub_window_captions": [window_caption["time_span"] for window_caption in hour_window_min_captions], "type": "hour"}}))
                                                
                                                accumulated_captions['hour_captions'].append({
                                                    'time_span': hour_time_span,
                                                    'caption': hour_caption,
                                                    'timestamp': last_hour_caption_end_timestamp
                                                })
                                                
                                                # Reset 1-hour window
                                                hour_window_min_captions = []
                                                hour_window_start_timestamp = None
                                                hour_window_start_dt = None
                                                hour_window_start_day = None
                                    
                                    # Add to 1-hour window
                                    if hour_window_start_timestamp is None:
                                        hour_window_start_timestamp = min_window_start_timestamp
                                        hour_window_start_dt = min_start_dt
                                        hour_window_start_day = min_start_day
                                    
                                    hour_window_min_captions.append({
                                        'time_span': min_time_span,
                                        'caption': min_caption,
                                        'start_timestamp': min_window_start_timestamp,
                                        'end_timestamp': last_frame_timestamp,
                                        'end_dt': min_end_dt  # Store end_dt for gap detection
                                    })
                                    
                                    # Check if 1-hour window is full
                                    hour_window_duration = calculate_time_diff_seconds(last_frame_timestamp, hour_window_start_timestamp, datasets_type)
                                    if hour_window_duration >= window_hours * 3600:
                                        # Generate 1-hour caption
                                        if hour_window_min_captions:
                                            hour_start_dt = hour_window_start_dt
                                            hour_end_dt = min_end_dt
                                            hour_start_day = hour_window_start_day
                                            hour_end_day = min_end_day
                                            hour_time_span = f"{hour_start_day}-{hour_start_dt.strftime('%H:%M:%S')}-{hour_end_dt.strftime('%H:%M:%S')}"
                                            
                                            min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                                            if datasets_type == "egolife":
                                                user_prompt = f"{PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                            else:
                                                user_prompt = f"{HOLOASSIST_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                            
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
                                            
                                            loop.run_until_complete(self.video_segments.upsert({hour_time_span: {"content": hour_caption, "sub_window_captions": [window_caption["time_span"] for window_caption in hour_window_min_captions], "type": "hour"}}))
                                            loop.run_until_complete(self._save_video_segments())
                                            loop.run_until_complete(self.ainsert_streaming_caption({hour_time_span: {"content": hour_caption, "sub_window_captions": [window_caption["time_span"] for window_caption in hour_window_min_captions], "type": "hour"}}))
                                            
                                            accumulated_captions['hour_captions'].append({
                                                'time_span': hour_time_span,
                                                'caption': hour_caption,
                                                'timestamp': last_frame_timestamp
                                            })
                                            
                                            # 保存检查点
                                            window_states = {
                                                'current_window_frames': current_window_frames,
                                                'current_window_frame_data': current_window_frame_data,
                                                'window_start_dt': window_start_dt,
                                                'window_start_day': window_start_day,
                                                'window_start_timestamp': window_start_timestamp,
                                                'last_frame_end_time': last_frame_end_time,
                                                'min_window_second_captions': min_window_second_captions,
                                                'min_window_start_dt': min_window_start_dt,
                                                'min_window_start_day': min_window_start_day,
                                                'min_window_start_timestamp': min_window_start_timestamp,
                                                'hour_window_min_captions': hour_window_min_captions,
                                                'hour_window_start_dt': hour_window_start_dt,
                                                'hour_window_start_day': hour_window_start_day,
                                                'hour_window_start_timestamp': hour_window_start_timestamp,
                                                'frame_time_ranges_list': frame_time_ranges_list,
                                            }
                                            captions_dict = {
                                                'second_captions': second_captions,
                                                'min_captions': min_captions,
                                                'hour_captions': hour_captions,
                                            }
                                            save_checkpoint_state(accumulated_captions, window_states, captions_dict)
                                            
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
                        # 更新last_frame_end_time为当前帧的end_time
                        last_frame_end_time = frame_info.get('end_time', timestamp) if datasets_type == "egolife" else timestamp
                    else:
                        # Add frame to current window
                        current_window_frames.append(frame)
                        current_window_frame_data.append(frame_info)
                        # 更新last_frame_end_time为当前帧的end_time，或者对应的具体时间戳
                        last_frame_end_time = frame_info.get('end_time', timestamp) if datasets_type == "egolife" else timestamp
                
                # 该帧处理完成后，将frame_time_ranges添加到列表中
                if i < len(frame_time_ranges):
                    current_frame_time_range = frame_time_ranges[i]
                    frame_range_key = (current_frame_time_range.get('start'), current_frame_time_range.get('end'))
                    if frame_range_key not in frame_time_ranges_list:
                        frame_time_ranges_list.append(frame_range_key)
                        logger.debug(f"Added frame {i} time range to processed list: start={current_frame_time_range.get('start')}, end={current_frame_time_range.get('end')}")
                
                # 每次循环后保存检查点（包括frame_time_ranges_list）
                window_states = {
                    'current_window_frames': current_window_frames,
                    'current_window_frame_data': current_window_frame_data,
                    'window_start_dt': window_start_dt,
                    'window_start_day': window_start_day,
                    'window_start_timestamp': window_start_timestamp,
                    'last_frame_end_time': last_frame_end_time,
                    'min_window_second_captions': min_window_second_captions,
                    'min_window_start_dt': min_window_start_dt,
                    'min_window_start_day': min_window_start_day,
                    'min_window_start_timestamp': min_window_start_timestamp,
                    'hour_window_min_captions': hour_window_min_captions,
                    'hour_window_start_dt': hour_window_start_dt,
                    'hour_window_start_day': hour_window_start_day,
                    'hour_window_start_timestamp': hour_window_start_timestamp,
                    'frame_time_ranges_list': frame_time_ranges_list,
                }
                captions_dict = {
                    'second_captions': second_captions,
                    'min_captions': min_captions,
                    'hour_captions': hour_captions,
                }
                save_checkpoint_state(accumulated_captions, window_states, captions_dict)
            
            # 确定整个视频都处理结束，再保存路径到数据库中          
            loop.run_until_complete(self.video_path_db.upsert({video_name: video_path}))
        
        # Process last 10s window if it has frames
        if current_window_frames:
            start_dt = current_window_frame_data[0]['datetime']
            end_dt = current_window_frame_data[-1]['datetime']
            start_day = current_window_frame_data[0]['day_num']
            last_frame_timestamp = current_window_frame_data[-1]['time_span_info'].get('timestamp')
            
            time_span = f"{start_day}-{start_dt.strftime('%H:%M:%S')}-{end_dt.strftime('%H:%M:%S')}"
            
            # Generate 10s caption using Qwen model with retry mechanism
            if datasets_type == "egolife":
                caption_prompt = PROMPTS["simple_second_caption_system_prompt"]
            else:
                caption_prompt = HOLOASSIST_PROMPTS["simple_second_caption_system_prompt"]
                
            max_retries = 3
            caption_text = None
            caption = None
            caption_dict = None
            
            for retry_count in range(max_retries):
                try:
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
                    caption = json.loads(caption_text)
                    # 检查是否包含 "frames" 键且为字典类型
                    if not (isinstance(caption, dict) and "frames" in caption and isinstance(caption["frames"], dict)):
                        logger.warning(f"Caption format invalid (attempt {retry_count + 1}/{max_retries}): missing 'frames' key or invalid structure")
                        if retry_count == max_retries - 1:
                            raise ValueError(f"Caption format invalid after {max_retries} attempts: missing 'frames' key or invalid structure")
                        continue
                    
                    # Qwen输出了各帧的caption，需要将它们整合为一个字典
                    caption_dict = {"dense_caption": {}, "description": caption.get("caption", "")}
                    for frame_idx, content in caption["frames"].items():
                        start_time_formatted = format_timestamp_to_hhmmsscc(current_window_frame_data[int(frame_idx)]['start_time'])
                        end_time_formatted = format_timestamp_to_hhmmsscc(current_window_frame_data[int(frame_idx)]['end_time'])
                        caption_dict["dense_caption"][f"DAY{start_day}-{start_time_formatted}-{end_time_formatted}"] = content
                    
                    # 如果成功处理，跳出循环
                    break
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse caption as JSON (attempt {retry_count + 1}/{max_retries}): {e}")
                    if retry_count == max_retries - 1:
                        raise ValueError(f"Failed to parse caption as JSON after {max_retries} attempts: {e}")
                except (KeyError, IndexError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to process caption dictionary (attempt {retry_count + 1}/{max_retries}): {e}")
                    if retry_count == max_retries - 1:
                        raise ValueError(f"Failed to process caption dictionary after {max_retries} attempts: {e}")
            
            # 保存处理后的caption
            second_captions[time_span] = caption_dict
            logger.info(f"Generated 10s caption for final window: {time_span} ({len(current_window_frames)} frames)")
            
            loop.run_until_complete(self.video_segments.upsert({time_span: {"content": json.dumps(caption_dict), "video_frames": current_window_frames, "type": "second"}}))
            loop.run_until_complete(self.video_segment_feature_vdb.upsert_video_segment(time_span, current_window_frames))
            loop.run_until_complete(self._save_video_segments())
            loop.run_until_complete(self.ainsert_streaming_caption({time_span: {"content": json.dumps(caption_dict), "video_frames": current_window_frames, "type": "second"}}))
            
            accumulated_captions['second_captions'].append({
                'time_span': time_span,
                'caption': json.dumps(caption_dict),
                'timestamp': last_frame_timestamp
            })
            
            # 保存检查点（处理最终窗口）
            window_states = {
                'current_window_frames': current_window_frames,
                'current_window_frame_data': current_window_frame_data,
                'window_start_dt': window_start_dt,
                'window_start_day': window_start_day,
                'window_start_timestamp': window_start_timestamp,
                'last_frame_end_time': last_frame_end_time,
                'min_window_second_captions': min_window_second_captions,
                'min_window_start_dt': min_window_start_dt,
                'min_window_start_day': min_window_start_day,
                'min_window_start_timestamp': min_window_start_timestamp,
                'hour_window_min_captions': hour_window_min_captions,
                'hour_window_start_dt': hour_window_start_dt,
                'hour_window_start_day': hour_window_start_day,
                'hour_window_start_timestamp': hour_window_start_timestamp,
                'frame_time_ranges_list': frame_time_ranges_list,
            }
            captions_dict = {
                'second_captions': second_captions,
                'min_captions': min_captions,
                'hour_captions': hour_captions,
            }
            save_checkpoint_state(accumulated_captions, window_states, captions_dict)
            
            # Add to 10-minute window
            if min_window_start_timestamp is None:
                min_window_start_timestamp = window_start_timestamp
                min_window_start_dt = start_dt
                min_window_start_day = start_day
            
            min_window_second_captions.append({
                'time_span': time_span,
                'caption': json.dumps(caption_dict),
                'start_timestamp': window_start_timestamp,
                'end_timestamp': last_frame_timestamp
            })
        
        # Process last 10-minute window if it has captions
        if min_window_second_captions:
            min_start_dt = min_window_start_dt
            min_start_day = min_window_start_day
            
            # 从最后一个caption的time_span中获取实际的结束时间和day
            last_caption = min_window_second_captions[-1]
            last_time_span = last_caption['time_span']
            # time_span格式: "{day}-{HH:MM:SS}-{HH:MM:SS}" 或 "{day}-{HH:MM:SS}-{HH:MM:SS}_index"
            # 移除索引后缀（如果有）
            time_span_clean = last_time_span.rsplit('_', 1)[0] if '_' in last_time_span else last_time_span
            time_span_parts = time_span_clean.split('-')
            if len(time_span_parts) >= 3:
                min_end_day = int(time_span_parts[0])
                min_end_time_str = time_span_parts[2]  # 结束时间
                # 解析时间字符串为datetime
                time_parts = min_end_time_str.split(':')
                if len(time_parts) >= 3:
                    hour = int(time_parts[0])
                    minute = int(time_parts[1])
                    second = int(time_parts[2])
                    base_date = datetime(2000, 1, 1)
                    min_end_dt = base_date + timedelta(days=min_end_day-1, hours=hour, minutes=minute, seconds=second)
                else:
                    min_end_dt = min_start_dt
                    min_end_day = min_start_day
            else:
                # 如果解析失败，使用start时间
                min_end_dt = min_start_dt
                min_end_day = min_start_day
            
            # 计算实际时间范围（秒）
            min_window_duration = calculate_time_diff_seconds(
                min_window_second_captions[-1]['end_timestamp'], 
                min_window_start_timestamp, 
                datasets_type
            )
            
            # 检查时间范围是否达到最小阈值（5分钟 = 300秒）
            min_threshold_seconds = window_minutes * 60  # 5分钟
            if min_window_duration >= min_threshold_seconds:
                min_time_span = f"{min_start_day}-{min_start_dt.strftime('%H:%M:%S')}-{min_end_dt.strftime('%H:%M:%S')}"
                
                caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in min_window_second_captions])
                if datasets_type == "egolife":
                    user_prompt = f"{PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                else:
                    user_prompt = f"{HOLOASSIST_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                
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
                logger.info(f"Generated 10min caption for final window: {min_time_span} ({len(min_window_second_captions)} 10s captions, duration: {min_window_duration:.1f}s)")
                
                loop.run_until_complete(self.video_segments.upsert({min_time_span: {"content": min_caption, "sub_window_captions": [window_caption["time_span"] for window_caption in min_window_second_captions], "type": "minute"}}))
                loop.run_until_complete(self._save_video_segments())
                loop.run_until_complete(self.ainsert_streaming_caption({min_time_span: {"content": min_caption, "sub_window_captions": [window_caption["time_span"] for window_caption in min_window_second_captions], "type": "minute"}}))
                
                # 从最后一个caption的end_timestamp获取final_timestamp
                final_timestamp = min_window_second_captions[-1]['end_timestamp'] if min_window_second_captions else min_window_start_timestamp
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
                    'end_timestamp': final_timestamp
                })
            else:
                logger.info(f"Skipping 10min caption generation for final window: duration {min_window_duration:.1f}s < {min_threshold_seconds}s threshold ({len(min_window_second_captions)} 10s captions)")
        
        # Process last 1-hour window if it has captions
        if hour_window_min_captions:
            hour_start_dt = hour_window_start_dt
            hour_start_day = hour_window_start_day
            
            # 从最后一个caption的time_span中获取实际的结束时间和day
            last_caption = hour_window_min_captions[-1]
            last_time_span = last_caption['time_span']
            # time_span格式: "{day}-{HH:MM:SS}-{HH:MM:SS}" 或 "{day}-{HH:MM:SS}-{HH:MM:SS}_index"
            # 移除索引后缀（如果有）
            time_span_clean = last_time_span.rsplit('_', 1)[0] if '_' in last_time_span else last_time_span
            time_span_parts = time_span_clean.split('-')
            if len(time_span_parts) >= 3:
                hour_end_day = int(time_span_parts[0])
                hour_end_time_str = time_span_parts[2]  # 结束时间
                # 解析时间字符串为datetime
                time_parts = hour_end_time_str.split(':')
                if len(time_parts) >= 3:
                    hour = int(time_parts[0])
                    minute = int(time_parts[1])
                    second = int(time_parts[2])
                    base_date = datetime(2000, 1, 1)
                    hour_end_dt = base_date + timedelta(days=hour_end_day-1, hours=hour, minutes=minute, seconds=second)
                else:
                    hour_end_dt = hour_start_dt
                    hour_end_day = hour_start_day
            else:
                # 如果解析失败，使用start时间
                hour_end_dt = hour_start_dt
                hour_end_day = hour_start_day
            
            # 计算实际时间范围（秒）
            hour_window_duration = calculate_time_diff_seconds(
                hour_window_min_captions[-1]['end_timestamp'], 
                hour_window_start_timestamp, 
                datasets_type
            )
            
            # 检查时间范围是否达到最小阈值（1小时 = 3600秒）
            hour_threshold_seconds = window_hours * 3600  # 1小时
            if hour_window_duration >= hour_threshold_seconds:
                hour_time_span = f"{hour_start_day}-{hour_start_dt.strftime('%H:%M:%S')}-{hour_end_dt.strftime('%H:%M:%S')}"
                
                min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                if datasets_type == "egolife":
                    user_prompt = f"{PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                else:
                    user_prompt = f"{HOLOASSIST_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                
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
                logger.info(f"Generated 1h caption for final window: {hour_time_span} ({len(hour_window_min_captions)} 10min captions, duration: {hour_window_duration:.1f}s)")
                
                loop.run_until_complete(self.video_segments.upsert({hour_time_span: {"content": hour_caption, "sub_window_captions": [window_caption["time_span"] for window_caption in hour_window_min_captions], "type": "hour"}}))
                loop.run_until_complete(self._save_video_segments())
                loop.run_until_complete(self.ainsert_streaming_caption({hour_time_span: {"content": hour_caption, "sub_window_captions": [window_caption["time_span"] for window_caption in hour_window_min_captions], "type": "hour"}}))
                
                # 从最后一个caption的end_timestamp获取final_timestamp
                final_timestamp = hour_window_min_captions[-1]['end_timestamp'] if hour_window_min_captions else hour_window_start_timestamp
                accumulated_captions['hour_captions'].append({
                    'time_span': hour_time_span,
                    'caption': hour_caption,
                    'timestamp': final_timestamp
                })
            else:
                logger.info(f"Skipping 1h caption generation for final window: duration {hour_window_duration:.1f}s < {hour_threshold_seconds}s threshold ({len(hour_window_min_captions)} 10min captions)")
            
            # 保存检查点（处理最终hour窗口）
            window_states = {
                'current_window_frames': current_window_frames,
                'current_window_frame_data': current_window_frame_data,
                'window_start_dt': window_start_dt,
                'window_start_day': window_start_day,
                'window_start_timestamp': window_start_timestamp,
                'last_frame_end_time': last_frame_end_time,
                'min_window_second_captions': min_window_second_captions,
                'min_window_start_dt': min_window_start_dt,
                'min_window_start_day': min_window_start_day,
                'min_window_start_timestamp': min_window_start_timestamp,
                'hour_window_min_captions': hour_window_min_captions,
                'hour_window_start_dt': hour_window_start_dt,
                'hour_window_start_day': hour_window_start_day,
                'hour_window_start_timestamp': hour_window_start_timestamp,
                'frame_time_ranges_list': frame_time_ranges_list,
            }
            captions_dict = {
                'second_captions': second_captions,
                'min_captions': min_captions,
                'hour_captions': hour_captions,
            }
            save_checkpoint_state(accumulated_captions, window_states, captions_dict)
    
        # 函数结束前保存最终检查点
        window_states = {
            'current_window_frames': current_window_frames,
            'current_window_frame_data': current_window_frame_data,
            'window_start_dt': window_start_dt,
            'window_start_day': window_start_day,
            'window_start_timestamp': window_start_timestamp,
            'last_frame_end_time': last_frame_end_time,
            'min_window_second_captions': min_window_second_captions,
            'min_window_start_dt': min_window_start_dt,
            'min_window_start_day': min_window_start_day,
            'min_window_start_timestamp': min_window_start_timestamp,
            'hour_window_min_captions': hour_window_min_captions,
            'hour_window_start_dt': hour_window_start_dt,
            'hour_window_start_day': hour_window_start_day,
            'hour_window_start_timestamp': hour_window_start_timestamp,
            'frame_time_ranges_list': frame_time_ranges_list,
        }
        captions_dict = {
            'second_captions': second_captions,
            'min_captions': min_captions,
            'hour_captions': hour_captions,
        }
        save_checkpoint_state(accumulated_captions, window_states, captions_dict)
        
        return {
            "second_captions": second_captions,
            "min_captions": min_captions,
            "hour_captions": hour_captions,
            "accumulated_captions": accumulated_captions
        }
        
    
    async def ainsert_streaming_caption(self, new_video_segments):
        await self._insert_start()
        # 这里不划分chunks，由于每次仅传入一段caption，因此直接对caption提取实体
        captions = [new_video_segments[key]["content"] for key in new_video_segments.keys()][0]
        video_time_span = list(new_video_segments.keys())
        client = genai.Client()
        tokens = client.models.count_tokens(
                model="gemini-2.0-flash", contents=captions
            )
        
        caption_dict = {
            "tokens": tokens.total_tokens,
            "content": captions.strip(),
            "chunk_order_index": 0,
            "time_span": [f"{video_time_span[0]}_0"],
            "sub_window_captions": [new_video_segments[key]["sub_window_captions"] for key in new_video_segments.keys() if "sub_window_captions" in new_video_segments[key]],
            "type": [new_video_segments[key]["type"] for key in new_video_segments.keys()][0]
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
    
    
    def process_frame_with_proactive_service(self, frame_info, accumulated_captions, datasets_type, max_captions_per_level=3, history_messages=None, retrieved_memory=None):
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
        
        if retrieved_memory is None:
            if datasets_type == "egolife":
                proactive_prompt = PROMPTS["proactive_service_prompt"]
            else:
                proactive_prompt = HOLOASSIST_PROMPTS["proactive_service_prompt"]
        else:
            if datasets_type == "egolife":
                proactive_prompt = PROMPTS["proactive_service_prompt_with_memory"]
            else:
                proactive_prompt = HOLOASSIST_PROMPTS["proactive_service_prompt_with_memory"]
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
        
        # Separate current and historical captions
        # The most recent second_caption is current, all others are historical
        current_second_caption = None
        historical_second_captions = []
        
        if selected_captions['second_captions']:
            # The last one is the most recent (current)
            current_second_caption = selected_captions['second_captions'][-1]
            # All others are historical
            historical_second_captions = selected_captions['second_captions'][:-1]
        
        # Format captions for prompt
        current_caption_parts = []
        historical_caption_parts = []
        
        # Current caption (only the most recent second-level caption)
        if current_second_caption:
            current_caption_parts.append("=== Current Second-level Caption ===")
            current_caption_parts.append(f"[{current_second_caption.get('time_span', 'N/A')}]: {current_second_caption.get('caption', '')}")
        
        # Historical captions: all hour, min, and other second-level captions
        if selected_captions['hour_captions']:
            historical_caption_parts.append("=== Hour-level Captions ===")
            for cap in selected_captions['hour_captions']:
                historical_caption_parts.append(f"[{cap.get('time_span', 'N/A')}]: {cap.get('caption', '')}")
        
        if selected_captions['min_captions']:
            if historical_caption_parts:
                historical_caption_parts.append("")
            historical_caption_parts.append("=== Minute-level Captions ===")
            for cap in selected_captions['min_captions']:
                historical_caption_parts.append(f"[{cap.get('time_span', 'N/A')}]: {cap.get('caption', '')}")
        
        if historical_second_captions:
            if historical_caption_parts:
                historical_caption_parts.append("")
            historical_caption_parts.append("=== Historical Second-level Captions ===")
            for cap in historical_second_captions:
                historical_caption_parts.append(f"[{cap.get('time_span', 'N/A')}]: {cap.get('caption', '')}")
        
        # Build prompt with captions
        prompt_parts = [proactive_prompt]
        
        if current_caption_parts:
            prompt_parts.append("\nCurrent Caption:\n" + "\n".join(current_caption_parts))
        
        if historical_caption_parts:
            prompt_parts.append("\nHistorical Captions:\n" + "\n".join(historical_caption_parts))
        
        full_prompt = "\n".join(prompt_parts)
        
        # Call Gemini model using _llm.py function
        text_prompt = full_prompt
        response = asyncio.run(self.llm.best_model_func_raw(
            self.llm.best_model_name,
            text_prompt,
            system_prompt=None,
            history_messages=[],  # history_messages已经拼接在prompt中了
        ))
        
        parsed_response = parse_gemini_json_response(response)
        
        return {
            'frame_timestamp': frame_timestamp,
            'selected_captions': selected_captions,
            'gemini_response': parsed_response,
            'prompt_used': full_prompt  # Save the prompt used for history
        }
    
    
    def process_proactive_service(
        self,
        datasets_type, 
        accumulated_captions=None,
        proactive_service_history=None,
        load_from_checkpoint=False
    ):
        """
        在caption、visual embedding和构造的图都已经保存后，进行主动服务判断和检索。
        支持从streaming_checkpoint.json离线加载数据进行处理。
        
        Args:
            accumulated_captions: Dict containing accumulated captions with structure:
                                 {
                                     'second_captions': [{'time_span': str, 'caption': dict, 'timestamp': int}, ...],
                                     'min_captions': [{'time_span': str, 'caption': str, 'timestamp': int}, ...],
                                     'hour_captions': [{'time_span': str, 'caption': str, 'timestamp': int}, ...]
                                 }
                                 If None and load_from_checkpoint=True, will load from streaming_checkpoint.json
            proactive_service_history: Optional list of previous proactive service history
            load_from_checkpoint: If True and accumulated_captions is None, load from streaming_checkpoint.json
        
        Returns:
            List of proactive service responses
        """
        loop = always_get_an_event_loop()
        proactive_responses = []
        
        # 如果accumulated_captions为None且需要从检查点加载
        if accumulated_captions is None and load_from_checkpoint:
            checkpoint_file = os.path.join(self.working_dir, "streaming_checkpoint.json")
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    accumulated_captions = checkpoint_data.get("accumulated_captions", {
                        'second_captions': [],
                        'min_captions': [],
                        'hour_captions': []
                    })
                    logger.info(f"Loaded accumulated_captions from checkpoint: {len(accumulated_captions.get('second_captions', []))} second captions, "
                              f"{len(accumulated_captions.get('min_captions', []))} min captions, "
                              f"{len(accumulated_captions.get('hour_captions', []))} hour captions")
                except Exception as e:
                    logger.error(f"Failed to load accumulated_captions from checkpoint: {e}")
                    accumulated_captions = {
                        'second_captions': [],
                        'min_captions': [],
                        'hour_captions': []
                    }
            else:
                logger.warning(f"Checkpoint file not found: {checkpoint_file}, using empty accumulated_captions")
                accumulated_captions = {
                    'second_captions': [],
                    'min_captions': [],
                    'hour_captions': []
                }
        elif accumulated_captions is None:
            logger.warning("accumulated_captions is None and load_from_checkpoint=False, using empty dict")
            accumulated_captions = {
                'second_captions': [],
                'min_captions': [],
                'hour_captions': []
            }
        
        if proactive_service_history is None:
            proactive_service_history = []
        
        # 加载proactive service检查点（用于断点续传）
        proactive_checkpoint_file = os.path.join(self.working_dir, "proactive_service_checkpoint.json")
        processed_time_spans = set()
        if os.path.exists(proactive_checkpoint_file):
            try:
                with open(proactive_checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                processed_time_spans = set(checkpoint_data.get('processed_time_spans', []))
                # 恢复已处理的proactive_responses
                saved_responses = checkpoint_data.get('proactive_responses', [])
                if saved_responses:
                    proactive_responses = saved_responses
                    logger.info(f"Loaded {len(proactive_responses)} proactive responses from checkpoint")
                # 恢复历史记录
                saved_history = checkpoint_data.get('proactive_service_history', [])
                if saved_history:
                    proactive_service_history = saved_history
                    logger.info(f"Loaded {len(proactive_service_history)} proactive service history records from checkpoint")
                logger.info(f"Found checkpoint with {len(processed_time_spans)} processed time_spans, will skip them")
            except Exception as e:
                logger.error(f"Failed to load proactive service checkpoint: {e}, starting from scratch")
        
        # 按照时间顺序排序second_captions（确保按顺序处理）
        second_captions_list = accumulated_captions.get('second_captions', [])
        if second_captions_list:
            # 按timestamp排序
            second_captions_list = sorted(second_captions_list, key=lambda x: x.get('timestamp', 0))
            logger.info(f"Processing {len(second_captions_list)} second captions in chronological order")
        
        # Process each 10s caption for proactive service (按时间顺序)
        for idx, second_caption_info in enumerate(second_captions_list):
            time_span = second_caption_info.get('time_span')
            caption_dict = second_caption_info.get('caption')
            timestamp = second_caption_info.get('timestamp')
            
            if not caption_dict or timestamp is None:
                logger.warning(f"Skipping invalid second_caption_info at index {idx}: missing caption_dict or timestamp")
                continue
            
            # 检查是否已经处理过（断点续传）
            if time_span in processed_time_spans:
                logger.info(f"Skipping already processed caption {idx+1}/{len(second_captions_list)}: {time_span}")
                continue
            
            logger.info(f"Processing proactive service for caption {idx+1}/{len(second_captions_list)}: {time_span}")
            
            # Call proactive service with the 10s caption and accumulated captions
            frame_info_for_proactive = {
                'caption': caption_dict,
                'frame': None,  # Use caption instead of single frame
                'time_span_info': {
                    'timestamp': timestamp
                }
            }
            
            # 第一次调用：使用Gemini判断是否需要主动服务
            history_text = format_proactive_history(proactive_service_history)
            proactive_result = self.process_frame_with_proactive_service(
                datasets_type=datasets_type,
                frame_info=frame_info_for_proactive,
                accumulated_captions=accumulated_captions,
                max_captions_per_level=3,
                history_messages=history_text,
                retrieved_memory=None
            )
            
            if not proactive_result or proactive_result.get('gemini_response') is None:
                logger.warning(f"Proactive service returned None for time_span: {time_span}")
                continue
            
            gemini_response = proactive_result.get('gemini_response')
            
            # 解析响应：应该是JSON列表或空列表
            service_list = []
            if isinstance(gemini_response, list):
                service_list = gemini_response
            elif isinstance(gemini_response, dict):
                service_list = [gemini_response]
            elif isinstance(gemini_response, str):
                parsed = json.loads(gemini_response)
                if isinstance(parsed, list):
                    service_list = parsed
                elif isinstance(parsed, dict):
                    service_list = [parsed]
            
            # 检查是否有需要检索的服务（包含memory_query字段且非空）
            needs_retrieval = False
            retrieval_query = None
            time_key = None
            service_type = None
            
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
                
                query_param = QueryParam(mode="videorag")
                
                retrieved_video_context, retrieved_chunk_context = loop.run_until_complete(streaming_videorag_query(
                    retrieval_query,
                    time_key,
                    service_type,
                    datasets_type, 
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
                proactive_result_with_memory = self.process_frame_with_proactive_service(
                    datasets_type=datasets_type, 
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
                logger.info(f"Proactive service triggered for {time_span}: {len(service_list)} service(s)")
                
                # 将每个服务的service_sub_type, user_prompt, trigger_time_window添加到历史记录
                for service in service_list:
                    if isinstance(service, dict):
                        proactive_service_history.append({
                            'service_sub_type': service.get('service_sub_type', ''),
                            'user_prompt': service.get('user_prompt', ''),
                            'trigger_time_window': service.get('trigger_time_window', '')
                        })
            else:
                logger.info(f"No proactive service needed for {time_span}")
            
            # 标记当前time_span为已处理，并保存检查点（每处理完一个就保存，支持断点续传）
            processed_time_spans.add(time_span)
            try:
                checkpoint_data = {
                    'processed_time_spans': list(processed_time_spans),
                    'proactive_responses': proactive_responses,
                    'proactive_service_history': proactive_service_history,
                    'last_processed_time_span': time_span,
                    'last_processed_timestamp': timestamp,
                    'last_updated': datetime.now().isoformat()
                }
                with open(proactive_checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                logger.debug(f"Saved checkpoint after processing {time_span}")
            except Exception as e:
                logger.error(f"Failed to save proactive service checkpoint after {time_span}: {e}")
        
        # 保存proactive_responses到文件
        proactive_responses_file = os.path.join(self.working_dir, "proactive_responses.json")
        try:
            with open(proactive_responses_file, 'w', encoding='utf-8') as f:
                json.dump(proactive_responses, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(proactive_responses)} proactive responses to {proactive_responses_file}")
        except Exception as e:
            logger.error(f"Failed to save proactive_responses to {proactive_responses_file}: {e}")
        
        return proactive_responses

