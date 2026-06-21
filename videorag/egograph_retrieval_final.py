"""
分离的建图和主动服务判断模块

将建图和主动服务判断分成两部分：
1. streaming_graph_construction_only: 只做建图，生成caption并保存
2. process_proactive_service_after_graph: 在caption、visual embedding和构造的图都保存后，进行主动服务判断和检索
"""

import os
import json
import glob
from ray import data
import torch
import asyncio
import re
import base64
import io
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime  # 仅用于文件命名
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from transformers import AutoModel, AutoTokenizer
import tiktoken
from google import genai
from PIL import Image

from .llm.qwen_vl import mllm_response
from .ego_prompt_ import PROMPTS
from .holoassist_prompt_ import HOLOASSIST_PROMPTS
from .proassist_prompt_ import PROASSIST_PROMPTS, OUTPUT_FORMAT
from .egoextra_prompt import EGOEXTRA_PROMPTS
from .egoschema_prompt import EGOSCHEMA_PROMPTS
from .eyewo_prompt import EYEWO_PROMPTS, OUTPUT_FORMAT

from ._llm import (
    LLMConfig,
    openai_config,
)
from .ego_op import (
    chunking_by_video_segments,
    streaming_extract_entities,
    batch_extract_entities,
    streaming_videorag_query, 
    streaming_egolife_query,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NanoVectorDBVideoSegmentStorage,
    NetworkXStorage,
)
from ._utils import (
    compute_mdhash_id,
    limit_async_func_call,
    wrap_embedding_func_with_attrs,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
    pack_user_ass_to_openai_messages,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from .video_processing import sample_frames_by_interval, extract_eyewo_video_frames


def parse_time_span_to_seconds(time_span: str) -> tuple[float, float]:
    """
    从时间戳字符串中提取开始时间和结束时间（以秒为单位）
    
    支持格式：{day}-{HH:MM:SS} or {HH:MM:SS.ss}-{HH:MM:SS} or {HH:MM:SS.ss}
    例如：
        "1-00:00:00.5-00:00:05" -> (0.5, 5.0)
        "2-00:01:30.25-00:01:45.75" -> (90.25, 105.75)
        "1-00:00:10-00:00:20" -> (10.0, 20.0)
    
    Args:
        time_span: 时间戳字符串，格式为 "day-HH:MM:SS-HH:MM:SS" 或类似格式
    
    Returns:
        tuple: (start_time_seconds, end_time_seconds)
    
    Raises:
        ValueError: 如果时间戳格式无法解析
    """
    if not time_span or not isinstance(time_span, str):
        raise ValueError(f"Invalid time_span: {time_span}")
    
    # 按 '-' 分割，应该得到至少3部分：day, start_time, end_time
    parts = time_span.split('-')
    
    if len(parts) < 3:
        raise ValueError(f"Invalid time_span format: {time_span}. Expected format: 'day-HH:MM:SS-HH:MM:SS'")
    
    # 提取开始时间和结束时间（忽略第一部分，可能是day或其他标识）
    start_time_str = parts[1].strip()
    end_time_str = parts[2].strip()
    
    def hhmmss_to_seconds(time_str: str) -> float:
        """
        将 HH:MM:SS 或 HH:MM:SS.ss 格式转换为秒数
        
        Args:
            time_str: 时间字符串，格式为 "HH:MM:SS" 或 "HH:MM:SS.ss"
        
        Returns:
            float: 总秒数
        """
        if not time_str:
            raise ValueError(f"Empty time string")
        
        # 分割小时、分钟、秒（可能包含小数）
        time_parts = time_str.split(':')
        
        if len(time_parts) != 3:
            raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM:SS or HH:MM:SS.ss")
        
        try:
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = float(time_parts[2])  # 可能是整数或小数
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid time format: {time_str}. Error: {e}")
    
    try:
        start_seconds = hhmmss_to_seconds(start_time_str)
        end_seconds = hhmmss_to_seconds(end_time_str)
        
        return (start_seconds, end_seconds)
    except Exception as e:
        raise ValueError(f"Failed to parse time_span '{time_span}': {e}")


def format_timestamp_to_hhmmss(timestamp, datasets_type="egolife"):
    """
    Convert timestamp to HH:MM:SS format string.
    
    Args:
        timestamp: For egolife, HHMMSSCC format (e.g., 11094300); for others, seconds (e.g., 117.5)
        datasets_type: Dataset type, "egolife" for HHMMSSCC format, others for seconds format
    
    Returns:
        String in format HH:MM:SS or HH:MM:SS.S (e.g., "11:09:43" for egolife, or "00:01:57" or "00:00:50.5" for seconds)
    """
    if datasets_type == "egolife":
        # Egolife数据集：HHMMSSCC格式（8位整数）
        timestamp_str = str(int(timestamp)).zfill(8)
        hours = timestamp_str[0:2]
        minutes = timestamp_str[2:4]
        seconds = timestamp_str[4:6]
        centiseconds = timestamp_str[6:8]
        
        return f"{hours}:{minutes}:{seconds}"
    elif datasets_type == "proassist" or datasets_type == "eyewo":
        timestamp_str = str(int(timestamp)).zfill(8)
        hours = timestamp_str[0:2]
        minutes = timestamp_str[2:4]
        seconds = timestamp_str[4:6]
        centiseconds = timestamp_str[6:8]
        
        # 如果有百分秒（不为00），则显示小数秒
        if centiseconds != "00":
            # 去掉末尾的0，例如 "50" -> "5", "25" -> "25", "05" -> "05"
            fractional_str = centiseconds.rstrip('0')
            return f"{hours}:{minutes}:{seconds}.{fractional_str}"
        else:
            return f"{hours}:{minutes}:{seconds}"
    else:
        # 其他数据集：直接是秒数（可能是浮点数）
        total_seconds_float = float(timestamp)
        total_seconds_int = int(total_seconds_float)
        fractional_part = total_seconds_float - total_seconds_int
        
        hours = total_seconds_int // 3600
        minutes = (total_seconds_int % 3600) // 60
        seconds = total_seconds_int % 60
        
        # 如果有小数部分，则显示小数秒
        if fractional_part > 0:
            # 保留一位小数
            fractional_str = f"{fractional_part:.1f}".lstrip('0')
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}{fractional_str}"
        else:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def tensor_to_pil_images(tensors: torch.Tensor) -> list[Image.Image]:
    """Convert a list of tensor images to PIL images.

    :param tensors: tensor images of shape (B, C, H, W) or (C, H, W)
    :return: list of PIL images
    """
    images = []
    # 如果tensor是4D (B, C, H, W)，迭代batch维度
    if len(tensors.shape) == 4:
        for t in tensors:
            arr = t.permute(1, 2, 0).cpu().numpy()
            # 如果数值范围在[0, 1]，转换为[0, 255]
            if arr.dtype == 'float32' or arr.dtype == 'float64':
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype('uint8')
                else:
                    arr = arr.astype('uint8')
            else:
                arr = arr.astype('uint8')
            images.append(Image.fromarray(arr))
    # 如果tensor是3D (C, H, W)，直接转换
    elif len(tensors.shape) == 3:
        arr = tensors.permute(1, 2, 0).cpu().numpy()
        # 如果数值范围在[0, 1]，转换为[0, 255]
        if arr.dtype == 'float32' or arr.dtype == 'float64':
            if arr.max() <= 1.0:
                arr = (arr * 255).astype('uint8')
            else:
                arr = arr.astype('uint8')
        else:
            arr = arr.astype('uint8')
        images.append(Image.fromarray(arr))
    return images


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

    return abs(timestamp_to_seconds(timestamp1, is_egolife=True) - timestamp_to_seconds(timestamp2, is_egolife=True))


def timestamp_number_to_seconds(timestamp_number):    # 仅在EgoExtra数据集中使用
    """将时间戳数字（HHMMSSCC格式）转换为秒数（浮点数）"""
    ts_int = int(timestamp_number)
    ts_str = str(ts_int).zfill(8)
    hours = int(ts_str[0:2])
    minutes = int(ts_str[2:4])
    seconds = int(ts_str[4:6])
    centiseconds = int(ts_str[6:8])
    return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0


def seconds_to_timestamp_number(seconds):       # 仅在EgoExtra和ProAssist数据集中使用
    """将秒数（浮点数）转换为时间戳数字（HHMMSSCC格式）"""
    total_hours = int(seconds // 3600)
    remaining = seconds % 3600
    total_minutes = int(remaining // 60)
    remaining = remaining % 60
    total_seconds = int(remaining)
    centiseconds = int(round((remaining - total_seconds) * 100))
    
    # 确保centiseconds在有效范围内 [0, 99]
    if centiseconds >= 100:
        total_seconds += 1
        centiseconds = 0
    if total_seconds >= 60:
        total_minutes += 1
        total_seconds = 0
    if total_minutes >= 60:
        total_hours += 1
        total_minutes = 0
    
    # 格式化为HHMMSSCC（小时可以>24）
    timestamp_number = total_hours * 1000000 + total_minutes * 10000 + total_seconds * 100 + centiseconds
    return int(timestamp_number)


def process_proassist_frames(tensor_frames: torch.Tensor, fps: float = 2.0, original_fps: float = 2.0):
    """
    处理 ProAssist 数据集的帧，将 tensor 转换为 base64 编码，并生成时间戳和时间范围。
    支持通过 fps 参数控制采样率，从原始帧中按固定间隔采样。
    
    Args:
        tensor_frames: Tensor 格式的帧，形状为 (num_frames, 3, H, W)
        fps: 目标帧率，默认 2.0（每 0.5 秒一帧）
        original_fps: 原始帧率，默认 2.0（原始视频的帧率）
    
    Returns:
        tuple: (base64_frames, frame_timestamps, frame_time_ranges)
            - base64_frames: list[str], base64 编码的帧列表
            - frame_timestamps: list[int], 每帧的时间戳（HHMMSSCC 格式）
            - frame_time_ranges: list[dict], 每帧的时间范围，格式为 [{'start': int, 'end': int}]
    
    Note:
        如果 fps < original_fps，会按固定间隔采样帧。
        例如：original_fps=2.0, fps=1.0，会每隔1帧采样一次（采样间隔=2）。
    """
    base64_frames = []
    frame_timestamps = []
    frame_time_ranges = []
    
    # 计算采样间隔：如果目标 fps 小于原始 fps，需要按间隔采样
    if fps < original_fps:
        # 计算采样间隔（向上取整）
        sample_interval = int(original_fps / fps)
        logger.info(f"Sampling frames: original_fps={original_fps}, target_fps={fps}, sample_interval={sample_interval}")
    else:
        # 如果目标 fps >= 原始 fps，使用所有帧
        sample_interval = 1
        if fps > original_fps:
            logger.warning(f"Target fps ({fps}) > original_fps ({original_fps}), using all frames")
    
    # 计算每帧的时间间隔（秒）- 基于原始 fps
    original_frame_interval = 1.0 / original_fps
    
    # 按采样间隔遍历帧
    for i in range(0, len(tensor_frames), sample_interval):
        # 计算当前帧在原始视频中的时间戳（秒）
        # 原始帧的时间戳：帧0是0.5秒，帧1是1.0秒，帧2是1.5秒，以此类推
        # 采样后保持原始时间戳
        timestamp_seconds = i * original_frame_interval
        
        # 生成时间戳（HHMMSSCC格式）- 使用原始时间戳
        timestamp = seconds_to_timestamp_number(timestamp_seconds)
        frame_timestamps.append(timestamp)
        
        # 生成时间范围：基于原始帧的时间间隔
        # 采样后的帧i（原始帧索引i）的时间范围：[i * original_frame_interval, (i + 1) * original_frame_interval]
        range_start_seconds = i * original_frame_interval
        range_end_seconds = (i + 1) * original_frame_interval # (i + 1) * original_frame_interval
        
        # 将tensor转换为PIL Image，然后编码为base64字符串
        pil_images = tensor_to_pil_images(tensor_frames[i:i+1])
        if pil_images:
            # 将PIL Image转换为base64编码的字符串
            img_buffer = io.BytesIO()
            pil_images[0].save(img_buffer, format='JPEG')
            img_bytes = img_buffer.getvalue()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            base64_frames.append(base64_str)
        
        time_range = {
            'start': seconds_to_timestamp_number(range_start_seconds),
            'end': seconds_to_timestamp_number(range_end_seconds)
        }
        frame_time_ranges.append(time_range)
    
    return base64_frames, frame_timestamps, frame_time_ranges


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


def format_system_info(system_info_list):
    """
    将对话历史格式化为合理的文本 prompt。
    
    Args:
        system_info_list: 包含对话历史的列表，每个元素包含:
            - role: "system" 或 "user"
            - content: 对话内容
            - timestamp: (可选) 时间戳，仅 user 角色有
    
    Returns:
        格式化后的文本字符串，包含 SYSTEM_PROMPT 和 USER_PROMPT
    """
    if not system_info_list:
        return ""
    
    system_prompts = []
    user_prompts = []
    
    for item in system_info_list:
        role = item.get("role", "")
        content = item.get("content", "")
        timestamp = item.get("timestamp")
        
        if role == "system":
            system_prompts.append(content)
        elif role == "user":
            # 如果有时间戳，格式化时间戳
            if timestamp is not None:
                # 将秒数转换为可读的时间格式
                hours = int(timestamp // 3600)
                minutes = int((timestamp % 3600) // 60)
                seconds = int(timestamp % 60)
                time_str = f"[{hours:02d}:{minutes:02d}:{seconds:02d}] "
            else:
                time_str = ""
            user_prompts.append(f"{time_str}{content}")
    
    # 构建格式化的文本
    formatted_text = ""
    
    # 添加系统提示
    if system_prompts:
        formatted_text += "(1) SYSTEM_PROMPT:\n"
        # 如果有多个系统提示，合并它们（用换行分隔）
        system_content = "\n".join(system_prompts)
        # 为每行添加缩进
        indented_system = "\n".join(f"    {line}" for line in system_content.split("\n"))
        formatted_text += f"{indented_system}\n\n"
    
    # 添加用户提示
    if user_prompts:
        formatted_text += "(2) USER_PROMPT:\n"
        # 如果有多个用户提示，显示所有（但通常最新的最重要）
        if len(user_prompts) == 1:
            formatted_text += f"    {user_prompts[-1]}\n"
        else:
            # 显示所有用户提示，最新的在最后
            for i, user_prompt in enumerate(user_prompts, 1):
                formatted_text += f"    {i}. {user_prompt}\n"
            formatted_text += f"\n    (Most recent: {user_prompts[-1]})\n"
    
    return formatted_text.strip()


def format_assistant_response_history(history_list):
    """
    格式化助手响应历史为文本（用于 proassist）。
    
    Args:
        history_list: 包含助手响应历史的列表，每个元素包含:
            - timestamp: 时间戳
            - response: 助手响应内容
            - decision: 决策类型 ("respond_now" 或 "need_retrieval")
            - evidence: (可选) 证据
    
    Returns:
        格式化后的文本字符串
    """
    if not history_list:
        return ""
    
    history_lines = []
    for i, item in enumerate(history_list, 1):
        timestamp = item.get('timestamp', 'N/A')
        decision = item.get('decision', 'N/A')
        response = item.get('response', '')
        evidence = item.get('evidence', '')
        
        history_lines.append(
            f"{i}. Timestamp: {timestamp}\n"
            f"   Response: {response}"
        )
        # if evidence:
        #     history_lines[-1] += f"\n   Evidence: {evidence}"
    
    return "\n\n".join(history_lines)


def format_assistant_response_history_with_user_questions(history_list, user_questions_list):
    """
    格式化助手响应历史为文本（用于 proassist），包含用户问题和模型回复。
    
    Args:
        history_list: 包含助手响应历史的列表，每个元素包含:
            - timestamp: 时间戳
            - response: 助手响应内容
            - decision: 决策类型 ("respond_now" 或 "need_retrieval")
        user_questions_list: 包含用户问题的列表，每个元素包含:
            - timestamp: 时间戳
            - content: 用户问题内容
    
    Returns:
        格式化后的文本字符串，按时间顺序包含用户问题和模型回复
    """
    # 合并用户问题和模型回复，按时间戳排序
    combined_items = []
    
    # 添加用户问题
    for user_q in user_questions_list:
        timestamp = user_q.get('timestamp', 0)
        content = user_q.get('content', '')
        combined_items.append({
            'timestamp': timestamp,
            'type': 'user',
            'content': content
        })
    
    # 添加模型回复
    for item in history_list:
        timestamp = item.get('timestamp', 'N/A')
        # 如果timestamp是字符串（如time_span），尝试解析
        if isinstance(timestamp, str):
            # 尝试从time_span中提取时间戳（使用开始时间）
            try:
                start_time, _ = parse_time_span_to_seconds(timestamp)
                timestamp = start_time
            except:
                # 如果解析失败，尝试简单的时间格式解析
                try:
                    if '-' in timestamp:
                        # 格式：day-HH:MM:SS-HH:MM:SS 或类似
                        parts = timestamp.split('-')
                        if len(parts) >= 2:
                            time_str = parts[1]  # 取开始时间
                            # 解析时间字符串为秒数（支持 HH:MM:SS 或 HH:MM:SS.ss 格式）
                            if '.' in time_str:
                                time_str, _ = time_str.split('.')
                            time_parts = time_str.split(':')
                            if len(time_parts) == 3:
                                hours = int(time_parts[0])
                                minutes = int(time_parts[1])
                                seconds = int(time_parts[2])
                                timestamp = hours * 3600 + minutes * 60 + seconds
                            else:
                                timestamp = 0
                        else:
                            timestamp = 0
                    elif timestamp.upper().startswith('DAY'):
                        # 支持两种格式：
                        # 1. DAY# HH:MM:SS（例如：DAY1 00:15:40）
                        # 2. DAY#-HH:MM:SS.ss（例如：DAY1-00:12:48.5）
                        # 匹配 "DAY数字 或 - 时间" 格式，时间可能包含小数秒
                        match = re.match(r'DAY\s*\d+\s*-?\s*(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?', timestamp, re.IGNORECASE)
                        if match:
                            hours = int(match.group(1))
                            minutes = int(match.group(2))
                            seconds = int(match.group(3))
                            # 处理小数秒（如果有）
                            fractional_seconds = 0.0
                            if match.group(4):
                                fractional_seconds = float('0.' + match.group(4))
                            timestamp = hours * 3600 + minutes * 60 + seconds + fractional_seconds
                        else:
                            timestamp = 0
                    else:
                        timestamp = 0
                except:
                    timestamp = 0
        elif not isinstance(timestamp, (int, float)):
            timestamp = 0
            
        response = item.get('response', '')
        combined_items.append({
            'timestamp': timestamp,
            'type': 'assistant',
            'content': response
        })
    
    # 按时间戳排序
    combined_items.sort(key=lambda x: x['timestamp'] if isinstance(x['timestamp'], (int, float)) else 0)
    
    if not combined_items:
        return ""
    
    # 格式化输出
    history_lines = []
    for i, item in enumerate(combined_items, 1):
        timestamp = item.get('timestamp', 'N/A')
        item_type = item.get('type', 'unknown')
        content = item.get('content', '')
        
        # 格式化时间戳
        if isinstance(timestamp, (int, float)):
            hours = int(timestamp // 3600)
            minutes = int((timestamp % 3600) // 60)
            seconds = int(timestamp % 60)
            time_str = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
        else:
            time_str = str(timestamp)
        
        if item_type == 'user':
            history_lines.append(
                f"{i}. {time_str} User: {content}"
            )
        else:
            history_lines.append(
                f"{i}. {time_str} Assistant: {content}"
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
        
        
    def load_caption_model(self, model_name: str, device_map: Optional[Union[str, Dict]] = None, **api_kwargs):
        # 清理旧模型（如果存在）以释放显存
        if hasattr(self, 'caption_model') and self.caption_model is not None:
            # 对于本地模型，需要显式释放显存
            if hasattr(self.caption_model, 'cpu'):
                self.caption_model.cpu()
            if hasattr(self.caption_model, 'to'):
                self.caption_model.to('cpu')
            del self.caption_model
            # 清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("已清理旧模型，释放显存")
        
        # 检查是否是API模型（Gemini/GPT-4o/Qwen3-VL API）
        model_name_lower = model_name.lower()
        
        if "gemini" in model_name_lower and ("api" in model_name_lower or "gemini_api" in model_name_lower):
            # 使用 Gemini API
            from .llm.gemini_api import mllm_response as gemini_mllm_response_func
            
            # 创建包装函数以传递API参数
            def gemini_mllm_response(video_llm, processor, user_prompt, system_prompt=None, 
                                    base64_frames=None, max_new_tokens=2048, has_image=True, 
                                    temperature=None, top_p=None):
                return gemini_mllm_response_func(
                    video_llm, processor, user_prompt, system_prompt, base64_frames,
                    max_new_tokens, has_image, temperature, top_p,
                    api_key=os.environ.get("GEMINI_API_KEY", ""),
                    model_name=api_kwargs.get("model_name", "gemini-2.5-pro"),
                    **{k: v for k, v in api_kwargs.items() if k not in ["api_key", "model_name"]}
                )
            
            self.mllm_response = gemini_mllm_response
            
            # 设置占位符对象（API模型不需要实际的模型实例）
            self.caption_model = None
            self.caption_processor = None
            self.processor = None
            self.video_llm = None
            self.image_processor = None
            logger.info(f"Loaded Gemini API model: {api_kwargs.get('model_name', 'gemini-1.5-pro')}")
            return
        
        elif "gpt_4o" in model_name_lower or "gpt4o" in model_name_lower:
            # 使用 GPT-4o API
            from .llm.gpt4o_api import mllm_response as gpt4o_mllm_response_func
            
            # 创建包装函数以传递API参数
            def gpt4o_mllm_response(video_llm, processor, user_prompt, system_prompt=None, 
                                   base64_frames=None, max_new_tokens=2048, has_image=True, 
                                   temperature=None, top_p=None):
                return gpt4o_mllm_response_func(
                    video_llm, processor, user_prompt, system_prompt, base64_frames,
                    max_new_tokens, has_image, temperature, top_p,
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                    model_name=api_kwargs.get("model_name", "gpt-4o"),
                    base_url=api_kwargs.get("base_url"),
                    **{k: v for k, v in api_kwargs.items() if k not in ["api_key", "model_name", "base_url"]}
                )
            
            self.mllm_response = gpt4o_mllm_response
            
            # 设置占位符对象（API模型不需要实际的模型实例）
            self.caption_model = None
            self.caption_processor = None
            self.processor = None
            self.video_llm = None
            self.image_processor = None
            logger.info(f"Loaded GPT-4o API model: {api_kwargs.get('model_name', 'gpt-4o')}")
            return
        
        elif "qwen3" in model_name_lower and "api" in model_name_lower:
            # 使用 Qwen3-VL API
            from .llm.qwen3vl_api import mllm_response as qwen3vl_mllm_response_func
            
            # 创建包装函数以传递API参数
            def qwen3vl_mllm_response(video_llm, processor, user_prompt, system_prompt=None, 
                                     base64_frames=None, max_new_tokens=2048, has_image=True, 
                                     temperature=None, top_p=None):
                return qwen3vl_mllm_response_func(
                    video_llm, processor, user_prompt, system_prompt, base64_frames,
                    max_new_tokens, has_image, temperature, top_p,
                    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
                    api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    model_name=api_kwargs.get("model_name", "qwen3-vl-30b-a3b-instruct"),      # qwen2.5-vl-72b-instruct
                    **{k: v for k, v in api_kwargs.items() if k not in ["api_key", "api_url", "model_name"]}
                )
            
            self.mllm_response = qwen3vl_mllm_response
            
            # 设置占位符对象（API模型不需要实际的模型实例）
            self.caption_model = None
            self.caption_processor = None
            self.processor = None
            self.video_llm = None
            self.image_processor = None
            logger.info(f"Loaded Qwen3-VL API model: {api_kwargs.get('model_name', 'qwen-vl-max')}")
            return
        
        # 加载对应的模型（本地模型）
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
        max_new_tokens=2048,
        video_title=None, 
        task_knowledge=None,
        clip_duration=None,
        task_type=None,
        proassist_fps=1.0,  # ProAssist 数据集的帧率，默认 2.0 fps
        eye_fps=2.0,  # EyeWo 数据集的帧率，默认 2.0 fps
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

        def get_day_number(day_str: str) -> int:
            """Extract day number from day string (e.g., 'DAY1' -> 1)."""
            try:
                return int(day_str.replace("DAY", ""))
            except Exception:
                return 0
        
        loop = always_get_an_event_loop()
        
        # 定义保存和加载状态的辅助函数
        def save_checkpoint_state(accumulated_captions, window_states, captions_dict=None, current_day=None, processed_time_spans=None):
            """保存accumulated_captions、窗口状态和captions字典到文件"""
            checkpoint_file = os.path.join(self.working_dir, "streaming_checkpoint.json")
            try:
                # 将datetime对象转换为字符串
                checkpoint_data = {
                    "accumulated_captions": accumulated_captions,
                    "window_states": {},
                    "captions_dict": captions_dict if captions_dict is not None else {},
                    "current_day": current_day,  # 保存当前的day信息
                    "processed_time_spans": processed_time_spans if processed_time_spans is not None else set()  # 保存已完全处理的time_span
                }
                
                # 将set转换为list以便JSON序列化
                if isinstance(checkpoint_data["processed_time_spans"], set):
                    checkpoint_data["processed_time_spans"] = list(checkpoint_data["processed_time_spans"])
                
                # 直接复制窗口状态（不再需要处理datetime对象）
                for key, value in window_states.items():
                    if key == 'frame_time_ranges_list' and isinstance(value, list):
                        # 将元组列表转换为列表列表（JSON可序列化）
                        checkpoint_data["window_states"][key] = [list(item) if isinstance(item, tuple) else item for item in value]
                    else:
                        checkpoint_data["window_states"][key] = value
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Checkpoint saved to {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        
        def load_checkpoint_state(current_day=None):
            """从文件加载accumulated_captions、窗口状态和captions字典"""
            checkpoint_file = os.path.join(self.working_dir, "streaming_checkpoint.json")
            if not os.path.exists(checkpoint_file):
                return None, None, None, None
            
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:   # 读取之前保存的log文件
                    checkpoint_data = json.load(f)
                
                # 分别读取最后保存的day和当前的day
                saved_day = get_day_number(checkpoint_data.get("current_day", None))
                current_day_num = get_day_number(current_day)
                day_changed = False
                
                # 如果当前day不为None，需要检查匹配：
                if current_day is not None:
                    if saved_day is None:       # 1. 如果保存的day为None但当前day不为None，说明可能是从旧版本升级或之前处理的是非egolife数据集
                        logger.info(f"Checkpoint has no day info but current day is {current_day_num}, resetting window states but keeping captions")
                        day_changed = True
                    elif saved_day < current_day_num:       # 2. 如果当前天数大于保存的天数，说明是新的一天，需要重置窗口状态，保留之前的captions
                        logger.info(f"New day detected: {current_day} (day {current_day_num}) > saved day {saved_day}, "
                                  f"resetting window states but keeping captions from previous day(s)")
                        day_changed = True
                    elif saved_day > current_day_num:       # 3. 如果当前天数小于保存的天数，这是异常情况（回退到之前的天），但为了安全，加载之前的状态
                        logger.warning(f"Day regression detected: {current_day} (day {current_day_num}) < saved day {saved_day}, "
                                     f"this is unusual. Loading previous state anyway.")
                        day_changed = False  # 不重置，加载之前的状态
                    else:  # saved_day == current_day_num
                        # 4. 同一天，继续处理，加载所有状态
                        logger.info(f"Continuing with same day: {current_day} (day {current_day_num}), loading all states")
                        day_changed = False
                
                # 即使day变化，也保留之前day的captions数据（这些数据已经保存在数据库中，保留缓存有助于后续处理）
                accumulated_captions = checkpoint_data.get("accumulated_captions", {
                    'second_captions': [],
                    'min_captions': [],
                    'hour_captions': []
                })
                
                # 加载captions字典（保留之前day的数据）
                captions_dict = checkpoint_data.get("captions_dict", {
                    'second_captions': {},
                    'min_captions': {},
                    'hour_captions': {}
                })
                
                # 如果day变化了，只重置窗口状态，不重置captions数据
                if day_changed:
                    logger.info(f"Keeping {len(accumulated_captions['second_captions'])} second captions, "
                              f"{len(accumulated_captions['min_captions'])} min captions, "
                              f"{len(accumulated_captions['hour_captions'])} hour captions from previous day(s)")
                    logger.info(f"Keeping {len(captions_dict.get('second_captions', {}))} second caption dicts, "
                              f"{len(captions_dict.get('min_captions', {}))} min caption dicts, "
                              f"{len(captions_dict.get('hour_captions', {}))} hour caption dicts from previous day(s)")
                    # 返回None作为window_states，表示需要重置窗口状态，但保留captions数据
                    processed_time_spans = set(checkpoint_data.get("processed_time_spans", []))
                    return accumulated_captions, None, captions_dict, processed_time_spans
                
                window_states_raw = checkpoint_data.get("window_states", {})
                window_states = {}
                
                # 处理frame_time_ranges_list：将列表列表转换回元组列表
                # 支持旧格式：2个元素（start, end）或4个元素（video_name, frame_idx, start, end）
                # 支持新格式：3个元素（day_num, start, end）或5个元素（day_num, video_name, frame_idx, start, end）
                if 'frame_time_ranges_list' in window_states_raw and isinstance(window_states_raw['frame_time_ranges_list'], list):
                    restored_ranges = []
                    for item in window_states_raw['frame_time_ranges_list']:
                        if isinstance(item, list) and (len(item) == 2 or len(item) == 3 or len(item) == 4 or len(item) == 5):
                            restored_ranges.append(tuple(item))
                        elif isinstance(item, tuple):
                            restored_ranges.append(item)
                        else:
                            restored_ranges.append(item)
                    window_states['frame_time_ranges_list'] = restored_ranges
                else:
                    window_states['frame_time_ranges_list'] = []
                
                # 其他状态直接复制（不再需要处理datetime对象）
                other_keys = ['current_window_frames', 'current_window_frame_data', 'window_start_day', 
                             'window_start_timestamp', 'last_frame_end_time', 'min_window_start_day',
                             'min_window_start_timestamp', 'min_window_second_captions', 
                             'hour_window_start_day', 'hour_window_start_timestamp', 
                             'hour_window_min_captions']
                for key in other_keys:
                    window_states[key] = window_states_raw.get(key, None if 'timestamp' in key or 'day' in key or key.endswith('_captions') else [])
                
                logger.info(f"Checkpoint loaded from {checkpoint_file}")
                logger.info(f"Loaded {len(accumulated_captions['second_captions'])} second captions, "
                          f"{len(accumulated_captions['min_captions'])} min captions, "
                          f"{len(accumulated_captions['hour_captions'])} hour captions")
                if window_states is not None:
                    logger.info(f"Loaded {len(window_states.get('frame_time_ranges_list', []))} processed frame time ranges")
                logger.info(f"Loaded {len(captions_dict.get('second_captions', {}))} second caption dicts, "
                          f"{len(captions_dict.get('min_captions', {}))} min caption dicts, "
                          f"{len(captions_dict.get('hour_captions', {}))} hour caption dicts")
                # 加载已完全处理的time_span集合
                processed_time_spans = set(checkpoint_data.get("processed_time_spans", []))
                if processed_time_spans:
                    logger.info(f"Loaded {len(processed_time_spans)} processed time spans (completed caption, DB insertion, and entity extraction)")
                return accumulated_captions, window_states, captions_dict, processed_time_spans
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
                return None, None, None, None
        
        # Store results
        # 尝试加载之前保存的状态（传入当前day进行匹配检查）
        loaded_accumulated_captions, loaded_window_states, loaded_captions_dict, loaded_processed_time_spans = load_checkpoint_state(current_day=day)
        
        # 初始化已完全处理的time_span集合（用于跟踪已完成所有步骤的time_span）
        processed_time_spans = loaded_processed_time_spans if loaded_processed_time_spans is not None else set()
        
        # 从数据库恢复已处理的 time_span（检查 visual embedding 和实体抽取是否已完成）
        # 这样可以确保即使检查点丢失，也能从数据库恢复状态
        try:
            # 检查 video_segment_feature_vdb 中已存在的 time_span
            if hasattr(self, 'video_segment_feature_vdb') and self.video_segment_feature_vdb is not None:
                # 获取所有已存在的 time_span（通过查询数据库）
                # 注意：这里假设 video_segment_feature_vdb 存储的 key 格式为 f"{time_span}_0"
                # 由于无法直接查询所有 keys，我们依赖检查点来恢复
                logger.info("Relying on checkpoint for processed time spans recovery")
            
            # 检查 text_chunks 和 knowledge graph 中已存在的实体
            # 这可以通过检查 second_captions 中哪些已经有对应的实体来判断
            # 但由于实体抽取是批量进行的，我们主要依赖 processed_time_spans 检查点
            if processed_time_spans:
                logger.info(f"Recovered {len(processed_time_spans)} processed time spans from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to recover processed time spans from database: {e}, using checkpoint only")
        
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
        
        # 30-second window state
        if loaded_window_states is not None:
            current_window_frames = loaded_window_states.get('current_window_frames', [])
            current_window_frame_data = loaded_window_states.get('current_window_frame_data', [])
            window_start_day = loaded_window_states.get('window_start_day')
            window_start_timestamp = loaded_window_states.get('window_start_timestamp')
            last_frame_end_time = loaded_window_states.get('last_frame_end_time')
            logger.info("Resuming from previous checkpoint: loaded 10-second window state")
        else:
            current_window_frames = []
            current_window_frame_data = []
            window_start_day = None
            window_start_timestamp = None
            last_frame_end_time = None
        
        # 5-minute window state
        if loaded_window_states is not None:
            min_window_second_captions = loaded_window_states.get('min_window_second_captions', [])
            min_window_start_day = loaded_window_states.get('min_window_start_day')
            min_window_start_timestamp = loaded_window_states.get('min_window_start_timestamp')
            logger.info("Resuming from previous checkpoint: loaded 10-minute window state")
        else:
            min_window_second_captions = []
            min_window_start_day = None
            min_window_start_timestamp = None
        
        # 1-hour window state
        if loaded_window_states is not None:
            hour_window_min_captions = loaded_window_states.get('hour_window_min_captions', [])
            hour_window_start_day = loaded_window_states.get('hour_window_start_day')
            hour_window_start_timestamp = loaded_window_states.get('hour_window_start_timestamp')
            logger.info("Resuming from previous checkpoint: loaded 1-hour window state")
        else:
            hour_window_min_captions = []
            hour_window_start_day = None
            hour_window_start_timestamp = None
        
        # 已处理的frame_time_ranges列表
        if loaded_window_states is not None:
            frame_time_ranges_list = loaded_window_states.get('frame_time_ranges_list', [])
            logger.info(f"Resuming from previous checkpoint: loaded {len(frame_time_ranges_list)} processed frame time ranges")
        else:
            frame_time_ranges_list = []
        
        # 收集所有需要批量处理的 second captions（延迟实体抽取）
        # 注意：如果程序在批量处理之前中断，这些列表会丢失
        # 但已保存到 video_segments 和 video_segment_feature_vdb 的数据不会丢失
        # 我们依赖 processed_time_spans 来避免重复处理
        pending_second_captions_for_entity_extraction = []
        # 收集所有需要批量处理的 video segments（延迟视频编码）
        pending_video_segments_for_encoding = []
        
        # 验证已处理的 time_span：检查数据库中是否真的存在对应的数据
        # 如果检查点标记为已处理，但数据库中不存在，则从 processed_time_spans 中移除
        if processed_time_spans:
            verified_processed = set()
            for time_span in list(processed_time_spans):
                # 检查 video_segments 中是否存在（caption 已保存）
                # 使用 _data 属性直接访问（如果可用），否则信任检查点
                try:
                    if hasattr(self.video_segments, '_data') and self.video_segments._data:
                        if time_span in self.video_segments._data:
                            verified_processed.add(time_span)
                        else:
                            logger.warning(f"Time_span {time_span} marked as processed but not found in video_segments._data, will reprocess")
                    else:
                        # 如果没有 _data 属性，信任检查点（可能是异步加载的数据）
                        # 在实际处理时会再次检查
                        verified_processed.add(time_span)
                        logger.debug(f"Trusting checkpoint for time_span {time_span} (cannot verify from _data)")
                except Exception as e:
                    logger.warning(f"Failed to verify time_span {time_span}: {e}, will reprocess")
                    # 如果验证失败，不添加到 verified_processed，这样会重新处理
            
            # 更新 processed_time_spans，只保留已验证的
            removed_count = len(processed_time_spans) - len(verified_processed)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} unverified time spans from processed list, will reprocess them")
            processed_time_spans = verified_processed
        
        # 恢复待处理的chunks：从数据库中检查哪些time_span已经保存了caption但还没有进行实体抽取
        # 这解决了中断后重新运行只处理最后一个chunks的问题
        try:
            # 获取所有已保存的second captions（从video_segments）
            if hasattr(self.video_segments, '_data') and self.video_segments._data:
                all_saved_time_spans = set(self.video_segments._data.keys())
                # 过滤出second类型的time_span
                second_time_spans = {
                    ts for ts in all_saved_time_spans 
                    if ts not in processed_time_spans  # 排除已完全处理的
                }
                
                # 检查这些time_span是否已经在text_chunks中（已进行实体抽取）
                if second_time_spans:
                    logger.info(f"Checking {len(second_time_spans)} saved time spans for entity extraction status...")
                    # 为每个time_span生成对应的chunk_key
                    time_spans_to_check = []
                    for time_span in second_time_spans:
                        # 检查video_segments中的数据
                        segment_data = self.video_segments._data.get(time_span)
                        if segment_data and segment_data.get("type") == "second":
                            # 生成chunk_key（与处理时使用相同的逻辑）
                            chunk_key = compute_mdhash_id(f"{time_span}_0", prefix="chunk-")
                            time_spans_to_check.append((time_span, chunk_key, segment_data))
                    
                    if time_spans_to_check:
                        # 批量检查哪些chunk_key不存在于text_chunks中
                        chunk_keys_to_check = [chunk_key for _, chunk_key, _ in time_spans_to_check]
                        missing_chunk_keys = loop.run_until_complete(
                            self.text_chunks.filter_keys(chunk_keys_to_check)
                        )
                        
                        # 恢复待处理的chunks
                        for time_span, chunk_key, segment_data in time_spans_to_check:
                            if chunk_key in missing_chunk_keys:
                                # 这个time_span需要重新进行实体抽取
                                pending_second_captions_for_entity_extraction.append({
                                    time_span: segment_data
                                })
                        
                        if pending_second_captions_for_entity_extraction:
                            logger.info(f"Recovered {len(pending_second_captions_for_entity_extraction)} pending time spans for entity extraction from database")
            else:
                logger.info("Cannot access video_segments._data, skipping recovery of pending chunks")
        except Exception as e:
            logger.warning(f"Failed to recover pending chunks from database: {e}, will process new chunks only")
        
        # 恢复待编码的video segments：从数据库中检查哪些time_span已经保存了caption但还没有进行visual embedding
        # 这解决了中断后可能出现"已抽取实体但未保存视觉embedding"的不一致问题
        try:
            # 获取所有已保存的second captions（从video_segments）
            if hasattr(self.video_segments, '_data') and self.video_segments._data:
                all_saved_time_spans = set(self.video_segments._data.keys())
                # 过滤出second类型的time_span
                second_time_spans = {
                    ts for ts in all_saved_time_spans 
                    if ts not in processed_time_spans  # 排除已完全处理的
                }
                
                # 检查这些time_span是否已经在video_segment_feature_vdb中（已进行visual embedding）
                if second_time_spans:
                    logger.info(f"Checking {len(second_time_spans)} saved time spans for visual embedding status...")
                    # 检查哪些time_span需要进行visual embedding
                    time_spans_to_encode = []
                    
                    # 检查video_segment_feature_vdb中是否存在该time_span的embedding
                    if hasattr(self.video_segment_feature_vdb, '_client') and hasattr(self.video_segment_feature_vdb._client, '_data'):
                        # 获取所有已编码的time_span（通过检查key的前缀）
                        encoded_time_spans = set()
                        for key in self.video_segment_feature_vdb._client._data.keys():
                            # key格式为 "time_span_frameindex"，提取time_span部分
                            if '_' in key:
                                time_span_part = '_'.join(key.split('_')[:-1])  # 去掉最后的帧索引
                                encoded_time_spans.add(time_span_part)
                        
                        # 找出未编码的time_span
                        for time_span in second_time_spans:
                            if time_span not in encoded_time_spans:
                                # 这个time_span需要重新进行visual embedding
                                segment_data = self.video_segments._data.get(time_span)
                                if segment_data and segment_data.get("type") == "second":
                                    video_frames = segment_data.get("video_frames", [])
                                    if video_frames:
                                        time_spans_to_encode.append((time_span, video_frames))
                        
                        if time_spans_to_encode:
                            pending_video_segments_for_encoding.extend(time_spans_to_encode)
                            logger.info(f"Recovered {len(time_spans_to_encode)} pending time spans for visual embedding from database")
                    else:
                        logger.info("Cannot access video_segment_feature_vdb._client._data, skipping recovery of pending video segments")
            else:
                logger.info("Cannot access video_segments._data, skipping recovery of pending video segments")
        except Exception as e:
            logger.warning(f"Failed to recover pending video segments from database: {e}, will process new segments only")
        
        # 恢复后检查累积的 captions 是否需要立即处理（如果有大时间间隔）
        # 这确保在恢复后，如果有累积的 min_window_second_captions 或 hour_window_min_captions，
        # 且它们与下一个 caption 之间会有大时间间隔，会在处理新 caption 前先处理这些累积的 captions
        # 注意：这个检查会在处理第一个新 caption 时进行（第1087行和第1279行），所以这里不需要额外处理
        
        # Process videos if data_path and anno_path are provided
        if datasets_type == "egolife":
        # if data_path and anno_path and day:
            # 对于EgoLife数据集，同时提取视频和标注文件
            days_list = sorted(glob.glob(os.path.join(data_path, "*.mp4")))
            anno_list = sorted(glob.glob(os.path.join(anno_path, "*.json")))
        elif datasets_type == "holoassist":
            days_list = [os.path.join(data_path, "Video_pitchshift.mp4")]
            anno_list = []
        elif datasets_type == "proassist":
            days_list = [data_path]
            anno_list = []
        # elif datasets_type == "egoextra":
        #     days_list = sorted(glob.glob(os.path.join(data_path, "*.mp4")))
        #     anno_list = []
        elif datasets_type == "eyewo":
            days_list = [data_path]
            anno_list = []
        elif datasets_type == "egoschema":
            days_list = [data_path]
            anno_list = []
        elif datasets_type == "captioncook4d":
            days_list = [data_path]
            anno_list = []
        else:
            days_list = sorted(glob.glob(os.path.join(data_path, "*.mp4")))      # 对于data_extra数据集，应该包含多个同一天的连续数据集
            anno_list = []
        
            if not days_list:
                raise ValueError(f"No video files found in {data_path}")
            
        day_num = get_day_number(day)
        
        # 对于egoextra数据集，需要跟踪上一个视频的结束时间戳，使时间线连续
        last_video_end_timestamp = 0  # 第一个视频从0开始（00000000格式）
        
        # Process each video one by one
        for video_idx, video_path in enumerate(days_list):     # 读取视频和注释文件
            # 初始化当前视频的变量，用于错误报告
            current_video_name = None
            current_frame_idx = None
            current_frame_timestamp = None
            current_frame_time_range = None
            
            try:
                # 如果是egolife数据集，才提供注释文件
                if datasets_type == "egolife":
                    video_name = os.path.basename(video_path).split(".")[0]
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
                    
                elif datasets_type == "proassist":
                    # proassist数据集的video_path是一个字典，包含已经采样好的帧
                    # 使用 process_proassist_frames 函数处理帧，支持通过 fps 参数控制采样率
                    video_name = "proassist_video"  # proassist数据集没有文件名，使用默认名称
                    tensor_frames = video_path["images"]      # num_frames, 3, H, W
                    
                    # 使用函数处理帧，通过 proassist_fps 参数控制采样率
                    # 原始帧率固定为 2.0 fps，如果目标 fps < 2.0，会按间隔采样
                    base64_frames, frame_timestamps, frame_time_ranges = process_proassist_frames(
                        tensor_frames, 
                        fps=proassist_fps,
                        original_fps=2.0  # ProAssist 数据集的原始帧率固定为 2fps
                    )
                    
                    time_span_info = None
                        
                elif datasets_type == "eyewo":
                    time_span_info = None
                    video_name = os.path.basename(video_path).split(".")[0]
                    base64_frames, frame_timestamps, frame_time_ranges = extract_eyewo_video_frames(
                        video_path,
                        start_time=clip_duration[0],
                        end_time=clip_duration[1],
                        fps=eye_fps,
                    )
                    
                elif datasets_type == "holoassist" or datasets_type == "captioncook4d" or datasets_type == "egoschema" or datasets_type == "egoschema":
                    # 每interval_seconds提取一帧，同时获取各帧的时间范围
                    video_name = os.path.basename(video_path).split(".")[0]
                    time_span_info = None
                    base64_frames, frame_timestamps, frame_time_ranges = sample_frames_by_interval(
                        video_path,
                        interval_seconds=interval_seconds,
                        output_format='base64',
                        time_span_info=time_span_info, 
                    )
                    
                elif datasets_type == "egoextra":
                    # 每interval_seconds提取一帧，同时获取各帧的时间范围
                    video_name = os.path.basename(video_path).split(".")[0]
                    time_span_info = None
                    base64_frames, frame_timestamps, frame_time_ranges = sample_frames_by_interval(
                        video_path,
                        interval_seconds=interval_seconds,
                        output_format='base64',
                        time_span_info=time_span_info, 
                    )
                    
                    # 将上一个视频的结束时间戳作为当前视频的起始时间戳，使时间线连续
                    if last_video_end_timestamp > 0:
                        # 将last_video_end_timestamp转换为秒数
                        offset_seconds = timestamp_number_to_seconds(last_video_end_timestamp)
                        
                        # 调整frame_timestamps
                        if frame_timestamps:
                            adjusted_timestamps = []
                            for ts in frame_timestamps:
                                ts_seconds = timestamp_number_to_seconds(ts)
                                adjusted_ts = seconds_to_timestamp_number(ts_seconds + offset_seconds)
                                adjusted_timestamps.append(adjusted_ts)
                            frame_timestamps = adjusted_timestamps
                        
                        # 调整frame_time_ranges
                        if frame_time_ranges:
                            adjusted_ranges = []
                            for time_range in frame_time_ranges:
                                start_seconds = timestamp_number_to_seconds(time_range['start'])
                                end_seconds = timestamp_number_to_seconds(time_range['end'])
                                adjusted_range = {
                                    'start': seconds_to_timestamp_number(start_seconds + offset_seconds),
                                    'end': seconds_to_timestamp_number(end_seconds + offset_seconds)
                                }
                                adjusted_ranges.append(adjusted_range)
                            frame_time_ranges = adjusted_ranges
                    
                    # 更新last_video_end_timestamp为当前视频的最后一个frame_time_ranges的'end'
                    if frame_time_ranges:
                        last_video_end_timestamp = frame_time_ranges[-1]['end']
                
                logger.info(f"Processing video {video_idx + 1}/{len(days_list)}: {video_name}")
                # 更新当前视频名称，用于错误报告
                current_video_name = video_name
                
                if not base64_frames:
                    logger.warning(f"No frames extracted from {video_name}")
                    continue
                
                # Process frames one by one (streaming)
                for i, frame in enumerate(base64_frames):
                    # # 更新当前帧信息，用于错误报告
                    # current_frame_idx = i
                    
                    # 检查该帧的frame_time_ranges是否已经处理过
                    if i < len(frame_time_ranges):
                        current_frame_time_range = frame_time_ranges[i]
                        # 将frame_time_range转换为可比较的格式（元组）
                        # 需要包含day_num，避免不同天的相同时间范围被误判为重复
                        # 对于proassist数据集，每帧有独立的时间范围，不会重复
                        frame_range_key = (day_num, current_frame_time_range.get('start'), current_frame_time_range.get('end'))
                        
                        # 检查是否已经处理过
                        if frame_range_key in frame_time_ranges_list:
                            logger.info(f"Frame {i} in video {video_name} (day {day_num}) already processed (start={current_frame_time_range.get('start')}, end={current_frame_time_range.get('end')}), skipping")
                            continue
                    
                    # Get timestamp for this frame
                    if frame_timestamps and i < len(frame_timestamps):
                        timestamp = frame_timestamps[i]
                    else:
                        # 如果没有timestamp，跳过该帧
                        logger.warning(f"No timestamp for frame {i} in {video_name}, skipping")
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
                        'day_num': day_num,
                        'video_name': video_name,
                        'frame_idx': i,
                        'start_time': frame_time_ranges[i]['start'],     # 这里对应的是每一帧的时间范围的开始和结束时间
                        'end_time': frame_time_ranges[i]['end'],
                        'time_span_info': time_span_info,       # 这里对应的是该帧对应的video clip的时间范围
                    }
                    
                    # Initialize first window or check for gap/day change
                    if window_start_timestamp is None:
                        window_start_timestamp = timestamp
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
                                # 直接使用 timestamp 生成 time_span，而不是 datetime
                                # start_timestamp = current_window_frame_data[0]['time_span_info'].get('timestamp')
                                # end_timestamp = current_window_frame_data[-1]['time_span_info'].get('timestamp')
                                start_timestamp = current_window_frame_data[0]['start_time']
                                end_timestamp = current_window_frame_data[-1]['end_time']
                                start_day = current_window_frame_data[0]['day_num']
                                last_frame_timestamp = current_window_frame_data[-1]['time_span_info'].get('timestamp')
                            
                                # 使用新的格式化函数直接从 timestamp 生成 time_span
                                if datasets_type == "proassist" or datasets_type == "eyewo":
                                    start_time_str = format_timestamp_to_hhmmss(start_timestamp, datasets_type)
                                    end_time_str = format_timestamp_to_hhmmss(end_timestamp, datasets_type)
                                    time_span = f"{start_day}-{start_time_str}-{end_time_str}"
                                else:
                                    start_time_str = format_timestamp_to_hhmmss(start_timestamp)
                                    end_time_str = format_timestamp_to_hhmmss(end_timestamp)
                                    time_span = f"{start_day}-{start_time_str}-{end_time_str}"
                                
                                # 检查该time_span是否已经完全处理（包括caption生成、数据库插入、实体提取）
                                if time_span in processed_time_spans:
                                    logger.info(f"Time_span {time_span} already fully processed (caption, DB insertion, and entity extraction), skipping all steps")
                                    # 跳过所有处理步骤，但需要确保数据在内存中
                                    if time_span in second_captions:
                                        caption_dict = second_captions[time_span]
                                    else:
                                        # 如果不在内存中，尝试从数据库加载（可选）
                                        logger.warning(f"Time_span {time_span} marked as processed but not in memory, skipping")
                                        continue
                                # 检查该time_span的caption是否已经生成（避免重复生成）
                                elif time_span in second_captions:
                                    logger.info(f"Caption for time_span {time_span} already exists, skipping generation")
                                    # 如果caption已存在，从second_captions中获取
                                    caption_dict = second_captions[time_span]
                                    # 即使caption已存在，如果未完全处理，仍需要执行数据库插入和实体提取
                                    # 继续执行后续步骤（不跳过）
                                else:
                                    # Generate 10s caption using Qwen model with retry mechanism
                                    if datasets_type == "egolife":
                                        caption_prompt = PROMPTS["simple_second_caption_system_prompt"]
                                    elif datasets_type == "proassist":
                                        caption_prompt = PROASSIST_PROMPTS["simple_second_caption_system_prompt"].format(title=video_title, task_knowledge=task_knowledge, output_format=OUTPUT_FORMAT)
                                    elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                                        caption_prompt = EGOEXTRA_PROMPTS["simple_second_caption_system_prompt"]
                                    elif datasets_type == "eyewo":
                                        caption_prompt = EYEWO_PROMPTS["simple_second_caption_system_prompt"].format(task_types=", ".join(task_type), output_format=OUTPUT_FORMAT)
                                    elif datasets_type == "egoschema":
                                        caption_prompt = EGOSCHEMA_PROMPTS["simple_second_caption_system_prompt"]
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
                                                frame_data = current_window_frame_data[int(frame_idx)]
                                                if datasets_type == "proassist" or datasets_type == "eyewo":
                                                    start_time_formatted = format_timestamp_to_hhmmss(frame_data['start_time'], datasets_type)
                                                    end_time_formatted = format_timestamp_to_hhmmss(frame_data['end_time'], datasets_type)
                                                    # 对于proassist数据集，我们需要预测timestamps，而不是时间范围，因此需要添加帧编号
                                                    caption_key = f"DAY{day_num}-{start_time_formatted}"
                                                    caption_dict["dense_caption"][caption_key] = content
                                                elif datasets_type == "holoassist":
                                                    start_time_formatted = format_timestamp_to_hhmmss(frame_data['start_time'])
                                                    end_time_formatted = format_timestamp_to_hhmmss(frame_data['end_time'])
                                                    caption_key = f"DAY{day_num}-{start_time_formatted}"
                                                    caption_dict["dense_caption"][caption_key] = content
                                                else:
                                                    start_time_formatted = format_timestamp_to_hhmmss(frame_data['start_time'])
                                                    end_time_formatted = format_timestamp_to_hhmmss(frame_data['end_time'])
                                                    caption_key = f"DAY{day_num}-{start_time_formatted}-{end_time_formatted}"
                                                    caption_dict["dense_caption"][caption_key] = content
                                            
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
                                    
                                    # 保存处理后的caption（如果之前不存在）
                                    if time_span not in second_captions:
                                        second_captions[time_span] = caption_dict
                                        logger.info(f"Generated 10s caption for window: {time_span} ({len(current_window_frames)} frames)")
                                    
                                    # 无论caption是新生成的还是已存在的，只要未完全处理，都需要执行数据库插入和实体提取
                                    if time_span not in processed_time_spans:
                                        # 将caption信息存入到video_segment中
                                        loop.run_until_complete(self.video_segments.upsert({time_span: {"content": json.dumps(caption_dict), "video_frames": current_window_frames, "type": "second"}}))
                                        # 延迟视频编码：收集到列表中，最后统一批量处理
                                        pending_video_segments_for_encoding.append((time_span, current_window_frames))
                                        loop.run_until_complete(self._save_video_segments())
                                        # 延迟实体抽取：收集到列表中，最后统一批量处理
                                        pending_second_captions_for_entity_extraction.append({
                                            time_span: {"content": json.dumps(caption_dict), "video_frames": current_window_frames, "type": "second"}
                                        })
                                    
                                    # Add to accumulated_captions for proactive service (but don't call it here)
                                    # 只有当time_span不在accumulated_captions中时才添加（避免重复）
                                    if not any(item['time_span'] == time_span for item in accumulated_captions['second_captions']):
                                        accumulated_captions['second_captions'].append({
                                            'time_span': time_span,
                                            'caption': json.dumps(caption_dict),
                                            'timestamp': last_frame_timestamp,
                                        })
                                
                                # 保存检查点
                                window_states = {
                                    'current_window_frames': current_window_frames,
                                    'current_window_frame_data': current_window_frame_data,
                                    'window_start_day': window_start_day,
                                    'window_start_timestamp': window_start_timestamp,
                                    'last_frame_end_time': last_frame_end_time,
                                    'min_window_second_captions': min_window_second_captions,
                                    'min_window_start_day': min_window_start_day,
                                    'min_window_start_timestamp': min_window_start_timestamp,
                                    'hour_window_min_captions': hour_window_min_captions,
                                    'hour_window_start_day': hour_window_start_day,
                                    'hour_window_start_timestamp': hour_window_start_timestamp,
                                    'frame_time_ranges_list': frame_time_ranges_list,
                                }
                                captions_dict = {
                                    'second_captions': second_captions,
                                    'min_captions': min_captions,
                                    'hour_captions': hour_captions,
                                }
                                save_checkpoint_state(accumulated_captions, window_states, captions_dict, current_day=day, processed_time_spans=processed_time_spans)
                                
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
                                            last_min_caption_end_dt = min_window_second_captions[-1].get('end_dt', None)
                                            
                                            # 直接使用 timestamp 生成 time_span
                                            min_start_timestamp = min_window_start_timestamp
                                            min_end_timestamp = last_min_caption_end_timestamp
                                            min_start_day = min_window_start_day
                                            if datasets_type == "proassist" or datasets_type == "eyewo":
                                                min_start_time_str = format_timestamp_to_hhmmss(min_start_timestamp, datasets_type)
                                                min_end_time_str = format_timestamp_to_hhmmss(min_end_timestamp, datasets_type)
                                            else:
                                                min_start_time_str = format_timestamp_to_hhmmss(min_start_timestamp)
                                                min_end_time_str = format_timestamp_to_hhmmss(min_end_timestamp)
                                            min_time_span = f"{min_start_day}-{min_start_time_str}-{min_end_time_str}"
                                            
                                            caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in min_window_second_captions])
                                            
                                            if datasets_type == "egolife":
                                                user_prompt = f"{PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                            elif datasets_type == "proassist":
                                                user_prompt = f"{PROASSIST_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                            elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                                                user_prompt = f"{EGOEXTRA_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                            elif datasets_type == "eyewo":
                                                user_prompt = f"{EYEWO_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                            elif datasets_type == "egoschema":
                                                user_prompt = f"{EGOSCHEMA_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
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
                                                        last_hour_caption_end_dt = hour_window_min_captions[-1].get('end_dt', None)
                                                        
                                                        # 直接使用 timestamp 生成 time_span
                                                        hour_start_timestamp = hour_window_start_timestamp
                                                        hour_end_timestamp = last_frame_timestamp  # 使用 last_frame_timestamp，因为这是正常流程
                                                        hour_start_day = hour_window_start_day
                                                        hour_end_day = hour_window_start_day
                                                        if datasets_type == "proassist" or datasets_type == "eyewo":
                                                            hour_start_time_str = format_timestamp_to_hhmmss(hour_start_timestamp, datasets_type)
                                                            hour_end_time_str = format_timestamp_to_hhmmss(hour_end_timestamp, datasets_type)
                                                        else:
                                                            hour_start_time_str = format_timestamp_to_hhmmss(hour_start_timestamp)
                                                            hour_end_time_str = format_timestamp_to_hhmmss(hour_end_timestamp)
                                                        hour_time_span = f"{hour_start_day}-{hour_start_time_str}-{hour_end_time_str}"
                                                        
                                                        min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                                                        if datasets_type == "egolife":
                                                            user_prompt = f"{PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                        elif datasets_type == "proassist":
                                                            user_prompt = f"{PROASSIST_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                        elif datasets_type == "eyewo":
                                                            user_prompt = f"{EYEWO_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                        elif datasets_type == "egoschema":
                                                            user_prompt = f"{EGOSCHEMA_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                        elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                                                            user_prompt = f"{EGOEXTRA_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
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
                                                        hour_window_start_day = None
                                            
                                            # Add to 1-hour window (after gap check)
                                            if hour_window_start_timestamp is None:
                                                hour_window_start_timestamp = min_window_start_timestamp
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
                                            min_window_start_day = None
                                
                                # Add to 10-minute window
                                if min_window_start_timestamp is None:
                                    min_window_start_timestamp = window_start_timestamp
                                    min_window_start_day = start_day
                                
                                min_window_second_captions.append({
                                    'time_span': time_span,
                                    'caption': json.dumps(caption_dict),  # 使用caption_dict而不是caption
                                    'start_timestamp': window_start_timestamp,
                                    'end_timestamp': last_frame_timestamp
                                })
                                
                                # Check if 10-minute window is full
                                min_window_duration = calculate_time_diff_seconds(last_frame_timestamp, min_window_start_timestamp, datasets_type)
                                if min_window_duration >= window_minutes * 60:
                                    # Generate 10-minute caption
                                    if min_window_second_captions:
                                        # 直接使用 timestamp 生成 time_span
                                        min_start_timestamp = min_window_start_timestamp
                                        min_end_timestamp = last_frame_timestamp
                                        min_start_day = min_window_start_day
                                        # Get end_day from current_window_frame_data
                                        min_end_day = current_window_frame_data[-1]['day_num']
                                        if datasets_type == "proassist" or datasets_type == "eyewo":
                                            min_start_time_str = format_timestamp_to_hhmmss(min_start_timestamp, datasets_type)
                                            min_end_time_str = format_timestamp_to_hhmmss(min_end_timestamp, datasets_type)
                                        else:
                                            min_start_time_str = format_timestamp_to_hhmmss(min_start_timestamp)
                                            min_end_time_str = format_timestamp_to_hhmmss(min_end_timestamp)
                                        min_time_span = f"{min_start_day}-{min_start_time_str}-{min_end_time_str}"
                                        
                                        caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in min_window_second_captions])
                                        
                                        if datasets_type == "egolife":
                                            user_prompt = f"{PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                        elif datasets_type == "proassist":
                                            user_prompt = f"{PROASSIST_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                        elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                                            user_prompt = f"{EGOEXTRA_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                        elif datasets_type == "eyewo":
                                            user_prompt = f"{EYEWO_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                                        elif datasets_type == "egoschema":
                                            user_prompt = f"{EGOSCHEMA_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
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
                                            'window_start_day': window_start_day,
                                            'window_start_timestamp': window_start_timestamp,
                                            'last_frame_end_time': last_frame_end_time,
                                            'min_window_second_captions': min_window_second_captions,
                                            'min_window_start_day': min_window_start_day,
                                            'min_window_start_timestamp': min_window_start_timestamp,
                                            'hour_window_min_captions': hour_window_min_captions,
                                            'hour_window_start_day': hour_window_start_day,
                                            'hour_window_start_timestamp': hour_window_start_timestamp,
                                            'frame_time_ranges_list': frame_time_ranges_list,
                                        }
                                        captions_dict = {
                                            'second_captions': second_captions,
                                            'min_captions': min_captions,
                                            'hour_captions': hour_captions,
                                        }
                                        save_checkpoint_state(accumulated_captions, window_states, captions_dict, current_day=day, processed_time_spans=processed_time_spans)
                                        
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
                                                    # 直接使用 timestamp 生成 time_span
                                                    hour_start_timestamp = hour_window_start_timestamp
                                                    hour_end_timestamp = last_hour_caption_end_timestamp
                                                    hour_start_day = hour_window_start_day
                                                    hour_end_day = hour_window_start_day
                                                    if datasets_type == "proassist" or datasets_type == "eyewo":
                                                        hour_start_time_str = format_timestamp_to_hhmmss(hour_start_timestamp, datasets_type)
                                                        hour_end_time_str = format_timestamp_to_hhmmss(hour_end_timestamp, datasets_type)
                                                    else:
                                                        hour_start_time_str = format_timestamp_to_hhmmss(hour_start_timestamp)
                                                        hour_end_time_str = format_timestamp_to_hhmmss(hour_end_timestamp)
                                                    hour_time_span = f"{hour_start_day}-{hour_start_time_str}-{hour_end_time_str}"
                                                    
                                                    min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                                                    if datasets_type == "egolife":
                                                        user_prompt = f"{PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                    elif datasets_type == "proassist":
                                                        user_prompt = f"{PROASSIST_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                    elif datasets_type == "eyewo":
                                                        user_prompt = f"{EYEWO_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                    elif datasets_type == "egoschema":
                                                        user_prompt = f"{EGOSCHEMA_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                    elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                                                        user_prompt = f"{EGOEXTRA_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
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
                                                    hour_window_start_day = None
                                        
                                        # Add to 1-hour window
                                        if hour_window_start_timestamp is None:
                                            hour_window_start_timestamp = min_window_start_timestamp
                                            hour_window_start_day = min_start_day
                                        
                                        hour_window_min_captions.append({
                                            'time_span': min_time_span,
                                            'caption': min_caption,
                                            'start_timestamp': min_window_start_timestamp,
                                            'end_timestamp': last_frame_timestamp
                                        })
                                        
                                        # Check if 1-hour window is full
                                        hour_window_duration = calculate_time_diff_seconds(last_frame_timestamp, hour_window_start_timestamp, datasets_type)
                                        if hour_window_duration >= window_hours * 3600:
                                            # Generate 1-hour caption
                                            if hour_window_min_captions:
                                                # 直接使用 timestamp 生成 time_span
                                                hour_start_timestamp = hour_window_start_timestamp
                                                hour_end_timestamp = last_frame_timestamp  # 使用 last_frame_timestamp，因为这是正常流程
                                                hour_start_day = hour_window_start_day
                                                hour_end_day = min_start_day
                                                if datasets_type == "proassist" or datasets_type == "eyewo":
                                                    hour_start_time_str = format_timestamp_to_hhmmss(hour_start_timestamp, datasets_type)
                                                    hour_end_time_str = format_timestamp_to_hhmmss(hour_end_timestamp, datasets_type)
                                                else:
                                                    hour_start_time_str = format_timestamp_to_hhmmss(hour_start_timestamp)
                                                    hour_end_time_str = format_timestamp_to_hhmmss(hour_end_timestamp)
                                                hour_time_span = f"{hour_start_day}-{hour_start_time_str}-{hour_end_time_str}"
                                                
                                                min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                                                if datasets_type == "egolife":
                                                    user_prompt = f"{PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                elif datasets_type == "proassist":
                                                    user_prompt = f"{PROASSIST_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                elif datasets_type == "eyewo":
                                                    user_prompt = f"{EYEWO_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                                                    user_prompt = f"{EGOEXTRA_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                                                elif datasets_type == "egoschema":
                                                    user_prompt = f"{EGOSCHEMA_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
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
                                                    'window_start_day': window_start_day,
                                                    'window_start_timestamp': window_start_timestamp,
                                                    'last_frame_end_time': last_frame_end_time,
                                                    'min_window_second_captions': min_window_second_captions,
                                                    'min_window_start_day': min_window_start_day,
                                                    'min_window_start_timestamp': min_window_start_timestamp,
                                                    'hour_window_min_captions': hour_window_min_captions,
                                                    'hour_window_start_day': hour_window_start_day,
                                                    'hour_window_start_timestamp': hour_window_start_timestamp,
                                                    'frame_time_ranges_list': frame_time_ranges_list,
                                                }
                                                captions_dict = {
                                                    'second_captions': second_captions,
                                                    'min_captions': min_captions,
                                                    'hour_captions': hour_captions,
                                                }
                                                save_checkpoint_state(accumulated_captions, window_states, captions_dict, current_day=day, processed_time_spans=processed_time_spans)
                                                
                                                # Reset 1-hour window
                                                hour_window_min_captions = []
                                                hour_window_start_timestamp = None
                                                hour_window_start_day = None
                                        
                                        # Reset 10-minute window
                                        min_window_second_captions = []
                                        min_window_start_timestamp = None
                                        min_window_start_day = None
                        
                            # Start new window with current frame
                            window_start_timestamp = timestamp
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
                        # 需要包含day_num，避免不同天的相同时间范围被误判为重复
                        # 对于proassist数据集，每帧有独立的时间范围，不会重复
                        frame_range_key = (day_num, current_frame_time_range.get('start'), current_frame_time_range.get('end'))
                        if frame_range_key not in frame_time_ranges_list:
                            frame_time_ranges_list.append(frame_range_key)
                            logger.debug(f"Added frame {i} (day {day_num}) time range to processed list: start={current_frame_time_range.get('start')}, end={current_frame_time_range.get('end')}")
                    
                    # 每次循环后保存检查点（包括frame_time_ranges_list）
                    window_states = {
                    'current_window_frames': current_window_frames,
                    'current_window_frame_data': current_window_frame_data,
                    'window_start_day': window_start_day,
                    'window_start_timestamp': window_start_timestamp,
                    'last_frame_end_time': last_frame_end_time,
                    'min_window_second_captions': min_window_second_captions,
                    'min_window_start_day': min_window_start_day,
                    'min_window_start_timestamp': min_window_start_timestamp,
                    'hour_window_min_captions': hour_window_min_captions,
                    'hour_window_start_day': hour_window_start_day,
                    'hour_window_start_timestamp': hour_window_start_timestamp,
                        'frame_time_ranges_list': frame_time_ranges_list,
                    }
                    captions_dict = {
                        'second_captions': second_captions,
                        'min_captions': min_captions,
                        'hour_captions': hour_captions,
                    }
                    save_checkpoint_state(accumulated_captions, window_states, captions_dict, current_day=day)
                
                # 确定整个视频都处理结束，再保存路径到数据库中          
                if datasets_type == "proassist":
                    loop.run_until_complete(self.video_path_db.upsert({video_name: video_path["dataset"]}))
                else:
                    loop.run_until_complete(self.video_path_db.upsert({video_name: video_path}))
            
            except Exception as e:
                # 错误发生时，打印详细的错误信息，包括视频名称和帧信息
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "video_name": current_video_name if current_video_name else f"Unknown (video_idx={video_idx}, path={video_path})",
                    "video_path": str(video_path),
                    "video_index": video_idx + 1,
                    "total_videos": len(days_list),
                }
                
                # 添加帧信息（如果可用）
                if current_frame_idx is not None:
                    error_info["frame_index"] = current_frame_idx
                    error_info["total_frames"] = len(base64_frames) if 'base64_frames' in locals() else "Unknown"
                
                if current_frame_timestamp is not None:
                    error_info["frame_timestamp"] = current_frame_timestamp
                
                if current_frame_time_range is not None:
                    error_info["frame_time_range"] = {
                        "start": current_frame_time_range.get('start'),
                        "end": current_frame_time_range.get('end')
                    }
                
                # 打印错误信息
                logger.error("=" * 80)
                logger.error("ERROR OCCURRED DURING VIDEO PROCESSING")
                logger.error("=" * 80)
                logger.error(f"Error Type: {error_info['error_type']}")
                logger.error(f"Error Message: {error_info['error_message']}")
                logger.error(f"Video Name: {error_info['video_name']}")
                logger.error(f"Video Path: {error_info['video_path']}")
                logger.error(f"Video Index: {error_info['video_index']}/{error_info['total_videos']}")
                
                if 'frame_index' in error_info:
                    logger.error(f"Frame Index: {error_info['frame_index']}/{error_info.get('total_frames', 'Unknown')}")
                
                if 'frame_timestamp' in error_info:
                    logger.error(f"Frame Timestamp: {error_info['frame_timestamp']}")
                
                if 'frame_time_range' in error_info:
                    logger.error(f"Frame Time Range: start={error_info['frame_time_range']['start']}, end={error_info['frame_time_range']['end']}")
                
                logger.error("=" * 80)
                
                # 重新抛出异常，让上层处理
                raise
        
        # Process last 10s window if it has frames
        if current_window_frames:
            # 直接使用 timestamp 生成 time_span，而不是 datetime
            # start_timestamp = current_window_frame_data[0]['time_span_info'].get('timestamp')
            # end_timestamp = current_window_frame_data[-1]['time_span_info'].get('timestamp')
            start_timestamp = current_window_frame_data[0]['start_time']
            end_timestamp = current_window_frame_data[-1]['end_time']
            start_day = current_window_frame_data[0]['day_num']
            last_frame_timestamp = current_window_frame_data[-1]['time_span_info'].get('timestamp')
            
            # 使用新的格式化函数直接从 timestamp 生成 time_span
            if datasets_type == "proassist" or datasets_type == "eyewo":
                start_time_str = format_timestamp_to_hhmmss(start_timestamp, datasets_type)
                end_time_str = format_timestamp_to_hhmmss(end_timestamp, datasets_type)
            else:
                start_time_str = format_timestamp_to_hhmmss(start_timestamp)
                end_time_str = format_timestamp_to_hhmmss(end_timestamp)
            time_span = f"{start_day}-{start_time_str}-{end_time_str}"
            
            # Generate 10s caption using Qwen model with retry mechanism
            if datasets_type == "egolife":
                caption_prompt = PROMPTS["simple_second_caption_system_prompt"]
            elif datasets_type == "proassist":
                caption_prompt = PROASSIST_PROMPTS["simple_second_caption_system_prompt"].format(title=video_title, task_knowledge=task_knowledge, output_format=OUTPUT_FORMAT)
            elif datasets_type == "eyewo":
                caption_prompt = EYEWO_PROMPTS["simple_second_caption_system_prompt"].format(task_types=", ".join(task_type), output_format=OUTPUT_FORMAT)
            elif datasets_type == "egoschema":
                caption_prompt = EGOSCHEMA_PROMPTS["simple_second_caption_system_prompt"]
            elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                caption_prompt = EGOEXTRA_PROMPTS["simple_second_caption_system_prompt"]
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
                        frame_data = current_window_frame_data[int(frame_idx)]
                        if datasets_type == "proassist" or datasets_type == "eyewo":
                            start_time_formatted = format_timestamp_to_hhmmss(frame_data['start_time'], datasets_type)
                            end_time_formatted = format_timestamp_to_hhmmss(frame_data['end_time'], datasets_type)
                            caption_key = f"DAY{start_day}-{start_time_formatted}"
                            caption_dict["dense_caption"][caption_key] = content
                        elif datasets_type == "holoassist":
                            start_time_formatted = format_timestamp_to_hhmmss(frame_data['start_time'])
                            end_time_formatted = format_timestamp_to_hhmmss(frame_data['end_time'])
                            caption_key = f"DAY{start_day}-{start_time_formatted}"
                            caption_dict["dense_caption"][caption_key] = content
                        else:
                            start_time_formatted = format_timestamp_to_hhmmss(frame_data['start_time'])
                            end_time_formatted = format_timestamp_to_hhmmss(frame_data['end_time'])
                            caption_key = f"DAY{start_day}-{start_time_formatted}-{end_time_formatted}"
                            caption_dict["dense_caption"][caption_key] = content
                    
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
            
            # 检查是否需要处理（可能已经处理过）
            if time_span not in processed_time_spans:
                loop.run_until_complete(self.video_segments.upsert({time_span: {"content": json.dumps(caption_dict), "video_frames": current_window_frames, "type": "second"}}))
                # 延迟视频编码：收集到列表中，最后统一批量处理
                pending_video_segments_for_encoding.append((time_span, current_window_frames))
                loop.run_until_complete(self._save_video_segments())
                # 延迟实体抽取：收集到列表中，最后统一批量处理
                pending_second_captions_for_entity_extraction.append({
                    time_span: {"content": json.dumps(caption_dict), "video_frames": current_window_frames, "type": "second"}
                })
            
            accumulated_captions['second_captions'].append({
                'time_span': time_span,
                'caption': json.dumps(caption_dict),
                'timestamp': last_frame_timestamp
            })
            
            # 保存检查点（处理最终窗口）
            window_states = {
                'current_window_frames': current_window_frames,
                'current_window_frame_data': current_window_frame_data,
                'window_start_day': window_start_day,
                'window_start_timestamp': window_start_timestamp,
                'last_frame_end_time': last_frame_end_time,
                'min_window_second_captions': min_window_second_captions,
                'min_window_start_day': min_window_start_day,
                'min_window_start_timestamp': min_window_start_timestamp,
                'hour_window_min_captions': hour_window_min_captions,
                'hour_window_start_day': hour_window_start_day,
                'hour_window_start_timestamp': hour_window_start_timestamp,
                'frame_time_ranges_list': frame_time_ranges_list,
            }
            captions_dict = {
                'second_captions': second_captions,
                'min_captions': min_captions,
                'hour_captions': hour_captions,
            }
            save_checkpoint_state(accumulated_captions, window_states, captions_dict, current_day=day)
            
            # Add to 10-minute window
            if min_window_start_timestamp is None:
                min_window_start_timestamp = window_start_timestamp
                min_window_start_day = start_day
            
            min_window_second_captions.append({
                'time_span': time_span,
                'caption': json.dumps(caption_dict),
                'start_timestamp': window_start_timestamp,
                'end_timestamp': last_frame_timestamp
            })
        
        # Process last 10-minute window if it has captions
        if min_window_second_captions:
            min_start_timestamp = min_window_start_timestamp
            min_start_day = min_window_start_day
            
            # 直接使用最后一个caption的end_timestamp，不需要从time_span解析
            last_caption = min_window_second_captions[-1]
            min_end_timestamp = last_caption['end_timestamp']
            # 从time_span中获取end_day（如果time_span格式正确）
            last_time_span = last_caption['time_span']
            time_span_clean = last_time_span.rsplit('_', 1)[0] if '_' in last_time_span else last_time_span
            time_span_parts = time_span_clean.split('-')
            if len(time_span_parts) >= 1:
                min_end_day = int(time_span_parts[0])  # 从time_span获取day
            else:
                min_end_day = min_start_day
            
            # 计算实际时间范围（秒）
            min_window_duration = calculate_time_diff_seconds(
                min_end_timestamp, 
                min_window_start_timestamp, 
                datasets_type
            )
            
            # 检查时间范围是否达到最小阈值（5分钟 = 300秒）
            if datasets_type == "egoschema":
                min_threshold_seconds = 0
            else:
                min_threshold_seconds = window_minutes * 60  # 5分钟
            if min_window_duration >= min_threshold_seconds:
                # 直接使用 timestamp 生成 time_span
                if datasets_type == "proassist" or datasets_type == "eyewo":
                    min_start_time_str = format_timestamp_to_hhmmss(min_start_timestamp, datasets_type)
                    min_end_time_str = format_timestamp_to_hhmmss(min_end_timestamp, datasets_type)
                else:
                    min_start_time_str = format_timestamp_to_hhmmss(min_start_timestamp)
                    min_end_time_str = format_timestamp_to_hhmmss(min_end_timestamp)
                min_time_span = f"{min_start_day}-{min_start_time_str}-{min_end_time_str}"
                
                caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in min_window_second_captions])
                if datasets_type == "egolife":
                    user_prompt = f"{PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                elif datasets_type == "proassist":
                    user_prompt = f"{PROASSIST_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                elif datasets_type == "eyewo":
                    user_prompt = f"{EYEWO_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                elif datasets_type == "egoschema":
                    user_prompt = f"{EGOSCHEMA_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
                elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                    user_prompt = f"{EGOEXTRA_PROMPTS['min_caption_system_prompt']}\n\nCaptions:\n{caption_text}"
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
            hour_start_timestamp = hour_window_start_timestamp
            hour_start_day = hour_window_start_day
            
            # 直接使用最后一个caption的end_timestamp，不需要从time_span解析
            last_caption = hour_window_min_captions[-1]
            hour_end_timestamp = last_caption['end_timestamp']
            # 从time_span中获取end_day（如果time_span格式正确）
            last_time_span = last_caption['time_span']
            time_span_clean = last_time_span.rsplit('_', 1)[0] if '_' in last_time_span else last_time_span
            time_span_parts = time_span_clean.split('-')
            if len(time_span_parts) >= 1:
                hour_end_day = int(time_span_parts[0])  # 从time_span获取day
            else:
                hour_end_day = hour_start_day
            
            # 计算实际时间范围（秒）
            hour_window_duration = calculate_time_diff_seconds(
                hour_end_timestamp, 
                hour_window_start_timestamp, 
                datasets_type
            )
            
            # 检查时间范围是否达到最小阈值（1小时 = 3600秒）
            hour_threshold_seconds = window_hours * 3600  # 1小时
            if hour_window_duration >= hour_threshold_seconds:
                # 直接使用 timestamp 生成 time_span
                if datasets_type == "proassist" or datasets_type == "eyewo":
                    hour_start_time_str = format_timestamp_to_hhmmss(hour_start_timestamp, datasets_type)
                    hour_end_time_str = format_timestamp_to_hhmmss(hour_end_timestamp, datasets_type)
                else:
                    hour_start_time_str = format_timestamp_to_hhmmss(hour_start_timestamp)
                    hour_end_time_str = format_timestamp_to_hhmmss(hour_end_timestamp)
                hour_time_span = f"{hour_start_day}-{hour_start_time_str}-{hour_end_time_str}"
                
                min_caption_text = "\n".join([f"[{item['time_span']}]: {item['caption']}" for item in hour_window_min_captions])
                if datasets_type == "egolife":
                    user_prompt = f"{PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                elif datasets_type == "proassist":
                    user_prompt = f"{PROASSIST_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                elif datasets_type == "eyewo":
                    user_prompt = f"{EYEWO_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                elif datasets_type == "egoschema":
                    user_prompt = f"{EGOSCHEMA_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
                elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                    user_prompt = f"{EGOEXTRA_PROMPTS['hour_caption_system_prompt']}\n\nCaptions:\n{min_caption_text}"
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
        
        # 函数结束前保存最终检查点
        window_states = {
            'current_window_frames': current_window_frames,
            'current_window_frame_data': current_window_frame_data,
            'window_start_day': window_start_day,
            'window_start_timestamp': window_start_timestamp,
            'last_frame_end_time': last_frame_end_time,
            'min_window_second_captions': min_window_second_captions,
            'min_window_start_day': min_window_start_day,
            'min_window_start_timestamp': min_window_start_timestamp,
            'hour_window_min_captions': hour_window_min_captions,
            'hour_window_start_day': hour_window_start_day,
            'hour_window_start_timestamp': hour_window_start_timestamp,
            'frame_time_ranges_list': frame_time_ranges_list,
        }
        captions_dict = {
            'second_captions': second_captions,
            'min_captions': min_captions,
            'hour_captions': hour_captions,
        }
        save_checkpoint_state(accumulated_captions, window_states, captions_dict, current_day=day)
        
        # 批量处理所有收集的 second captions，统一进行实体抽取
        if pending_second_captions_for_entity_extraction:
            logger.info(f"[Batch Entity Extraction] Processing {len(pending_second_captions_for_entity_extraction)} second captions in parallel...")
            loop.run_until_complete(self._insert_start())
            
            # 准备所有需要处理的chunks
            all_inserting_chunks = {}
            for new_video_segments in pending_second_captions_for_entity_extraction:
                for time_span, segment_data in new_video_segments.items():
                    captions = segment_data["content"]
                    # 处理JSON字符串
                    if isinstance(captions, str):
                        try:
                            caption_dict_content = json.loads(captions)
                            if isinstance(caption_dict_content, dict):
                                captions = caption_dict_content.get("caption", captions)
                        except:
                            pass
                    
                    ENCODER = tiktoken.encoding_for_model("gpt-4o")
                    tokens = ENCODER.encode_batch([captions], num_threads=16)[0]
                    
                    caption_dict = {
                        "tokens": len(tokens),
                        "content": captions.strip() if isinstance(captions, str) else json.dumps(captions).strip(),
                        "chunk_order_index": 0,
                        "time_span": [f"{time_span}_0"],
                        "sub_window_captions": segment_data.get("sub_window_captions", []),
                        "type": segment_data.get("type", "second")
                    }
                    
                    chunk_key = compute_mdhash_id(caption_dict["time_span"][0], prefix="chunk-")
                    all_inserting_chunks[chunk_key] = caption_dict
            
            # 过滤已存在的chunks
            if all_inserting_chunks:
                _add_chunk_keys = loop.run_until_complete(self.text_chunks.filter_keys(list(all_inserting_chunks.keys())))
                all_inserting_chunks = {
                    k: v for k, v in all_inserting_chunks.items() if k in _add_chunk_keys
                }
                
                if all_inserting_chunks:
                    logger.info(f"[Batch Entity Extraction] Inserting {len(all_inserting_chunks)} new chunks")
                    if self.enable_naive_rag:
                        logger.info("Insert chunks for naive RAG")
                        loop.run_until_complete(self.chunks_vdb.upsert(all_inserting_chunks))
                    
                    # 批量并行抽取实体
                    maybe_new_kg, _, _ = loop.run_until_complete(batch_extract_entities(
                        all_inserting_chunks,
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        asdict(self),
                        datasets_type=datasets_type,
                        task_topic=video_title,
                        task_types=task_type,
                    ))
                    
                    if maybe_new_kg is not None:
                        self.chunk_entity_relation_graph = maybe_new_kg
                    else:
                        logger.warning("No new entities found in batch extraction")
                    
                    # 保存chunks
                    loop.run_until_complete(self.text_chunks.upsert(all_inserting_chunks))
                    
                    # 注意：这里不立即标记为已处理，因为还需要完成 visual embedding
                    # 标记将在 visual embedding 完成后统一进行
                    logger.info(f"Entity extraction completed for {len(pending_second_captions_for_entity_extraction)} time spans")
                else:
                    logger.info("All chunks are already in the storage")
                    # 即使chunks已存在，也不立即标记，等待 visual embedding 完成
            
            loop.run_until_complete(self._insert_done())
            logger.info("[Batch Entity Extraction] Completed all entity extractions")
        
        # 批量处理所有收集的 video segments，统一进行编码
        if pending_video_segments_for_encoding:
            logger.info(f"[Batch Video Encoding] Processing {len(pending_video_segments_for_encoding)} video segments in batch...")
            loop.run_until_complete(
                self.video_segment_feature_vdb.upsert_video_segment_batch(
                    pending_video_segments_for_encoding,
                    encode_mode="joint"
                )
            )
            loop.run_until_complete(self._save_video_segments())
            logger.info("[Batch Video Encoding] Completed all video segment encodings")
            
            # 标记所有已编码的 time_span 为完全处理完成
            # 只有当 visual embedding 和实体抽取都完成时才标记
            encoded_time_spans = {time_span for time_span, _ in pending_video_segments_for_encoding}
            entity_extracted_time_spans = {
                time_span 
                for seg in pending_second_captions_for_entity_extraction 
                for time_span in seg.keys()
            }
            
            # 只有同时完成 visual embedding 和实体抽取的 time_span 才标记为完全处理
            fully_processed = encoded_time_spans & entity_extracted_time_spans
            for time_span in fully_processed:
                processed_time_spans.add(time_span)
            
            if fully_processed:
                logger.info(f"Marked {len(fully_processed)} time spans as fully processed (both visual embedding and entity extraction completed)")
            
            # 检查是否有只完成了一部分的 time_span（不应该发生，但记录警告）
            only_encoded = encoded_time_spans - entity_extracted_time_spans
            only_entity_extracted = entity_extracted_time_spans - encoded_time_spans
            if only_encoded:
                logger.warning(f"{len(only_encoded)} time spans have visual embedding but not entity extraction: {list(only_encoded)[:5]}...")
            if only_entity_extracted:
                logger.warning(f"{len(only_entity_extracted)} time spans have entity extraction but not visual embedding: {list(only_entity_extracted)[:5]}...")
            
            # 保存检查点，包含已处理的time_span信息
            window_states = {
                'current_window_frames': current_window_frames,
                'current_window_frame_data': current_window_frame_data,
                'window_start_day': window_start_day,
                'window_start_timestamp': window_start_timestamp,
                'last_frame_end_time': last_frame_end_time,
                'min_window_second_captions': min_window_second_captions,
                'min_window_start_day': min_window_start_day,
                'min_window_start_timestamp': min_window_start_timestamp,
                'hour_window_min_captions': hour_window_min_captions,
                'hour_window_start_day': hour_window_start_day,
                'hour_window_start_timestamp': hour_window_start_timestamp,
                'frame_time_ranges_list': frame_time_ranges_list,
            }
            captions_dict = {
                'second_captions': second_captions,
                'min_captions': min_captions,
                'hour_captions': hour_captions,
            }
            save_checkpoint_state(accumulated_captions, window_states, captions_dict, current_day=day, processed_time_spans=processed_time_spans)
        
        # 函数结束前保存最终检查点
        window_states = {
            'current_window_frames': current_window_frames,
            'current_window_frame_data': current_window_frame_data,
            'window_start_day': window_start_day,
            'window_start_timestamp': window_start_timestamp,
            'last_frame_end_time': last_frame_end_time,
            'min_window_second_captions': min_window_second_captions,
            'min_window_start_day': min_window_start_day,
            'min_window_start_timestamp': min_window_start_timestamp,
            'hour_window_min_captions': hour_window_min_captions,
            'hour_window_start_day': hour_window_start_day,
            'hour_window_start_timestamp': hour_window_start_timestamp,
            'frame_time_ranges_list': frame_time_ranges_list,
        }
        captions_dict = {
            'second_captions': second_captions,
            'min_captions': min_captions,
            'hour_captions': hour_captions,
        }
        save_checkpoint_state(accumulated_captions, window_states, captions_dict, current_day=day, processed_time_spans=processed_time_spans)
        
        return {
            "second_captions": second_captions,
            "min_captions": min_captions,
            "hour_captions": hour_captions,
            "accumulated_captions": accumulated_captions
        }
        
    
    async def ainsert_streaming_caption(self, new_video_segments, datasets_type: str = None, task_topic: str = None, task_types: list[str] = None):
        await self._insert_start()
        # 这里不划分chunks，由于每次仅传入一段caption，因此直接对caption提取实体
        captions = [new_video_segments[key]["content"] for key in new_video_segments.keys()][0]
        video_time_span = list(new_video_segments.keys())
        # client = genai.Client()
        # tokens = client.models.count_tokens(
        #         model="gemini-2.0-flash", contents=captions
        #     )
        
        ENCODER = tiktoken.encoding_for_model("gpt-4o")
        tokens = ENCODER.encode_batch([captions], num_threads=16)[0]
        
        caption_dict = {
            "tokens": len(tokens),
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
                datasets_type=datasets_type,
                task_topic=task_topic,
                task_types=task_types,
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
    
    
    def process_frame_with_proactive_service(self, frame_info, accumulated_captions, datasets_type, max_captions_per_level=3, history_messages=None, retrieved_memory=None, formatted_system_info=None, pre_retrieval_probe=None):
        """
        Args:
            pre_retrieval_probe: 之前的检索信息，格式为 {"suspected_service_type": "...", "retrieval_query": "..."}
        """
        if retrieved_memory is None:
            if datasets_type == "egolife":
                proactive_prompt = PROMPTS["proactive_service_prompt_test"]
            elif datasets_type == "proassist":
                proactive_prompt = PROASSIST_PROMPTS["proactive_service_prompt"]
            elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                proactive_prompt = EGOEXTRA_PROMPTS["proactive_service_prompt"]
            else:
                proactive_prompt = HOLOASSIST_PROMPTS["proactive_service_prompt"]
        else:
            if datasets_type == "egolife":
                proactive_prompt = PROMPTS["proactive_service_prompt_with_memory_test"]
            elif datasets_type == "proassist":
                proactive_prompt = PROASSIST_PROMPTS["proactive_service_prompt_with_memory"]
            elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                proactive_prompt = EGOEXTRA_PROMPTS["proactive_service_prompt_with_memory"]
            else:
                proactive_prompt = HOLOASSIST_PROMPTS["proactive_service_prompt_with_memory"]
            proactive_prompt = proactive_prompt + "\n\nRetrieved memory:\n\n" + retrieved_memory
        
        # 如果有之前的检索信息，添加到prompt中（用于抑制频繁检索）
        if pre_retrieval_probe and isinstance(pre_retrieval_probe, dict):
            time_span_probe = pre_retrieval_probe.get('time_span', '')
            suspected_service_type = pre_retrieval_probe.get('suspected_service_type', '')
            retrieval_query = pre_retrieval_probe.get('retrieval_query', '')
            if suspected_service_type or retrieval_query:
                pre_retrieval_text = "\n\nPREVIOUS RETRIEVAL REQUESTS:\n"
                if time_span_probe:
                    pre_retrieval_text += f"time_span: {time_span_probe}\n"
                if suspected_service_type:
                    pre_retrieval_text += f"suspected_service_type: {suspected_service_type}\n"
                if retrieval_query:
                    pre_retrieval_text += f"retrieval_query: {retrieval_query}\n"
                proactive_prompt = proactive_prompt + pre_retrieval_text
        
        # Handle both frame (image) and caption (text) cases
        frame = frame_info.get('frame')
        caption = frame_info.get('caption')  # For text-only mode
        frame_timestamp = frame_info.get('time_span_info', {}).get('timestamp')
        
        # 如果是 proassist 且有格式化后的 system_info，将其拼接在 prompt 后面
        if datasets_type == "proassist" and formatted_system_info:
            proactive_prompt = proactive_prompt + formatted_system_info
        
        # 将history_messages（字符串）拼接在system_prompt后面
        if history_messages and isinstance(history_messages, str) and history_messages.strip() and datasets_type != "proassist":
            proactive_prompt = proactive_prompt + "\n\n" + "=== Recent Proactive Service History ===\n" + history_messages
        elif history_messages and isinstance(history_messages, str) and history_messages.strip() and datasets_type == "proassist":
            proactive_prompt = proactive_prompt + "\n\n" + "=== (3) INTERACTION HISTORY ===\n" + history_messages
        
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
            # 获取当前帧的天数
            frame_day = frame_info.get('day_num', 0)
            # 如果frame_info中没有day_num，尝试从time_span_info中获取
            if frame_day == 0:
                time_span_info = frame_info.get('time_span_info', {})
                # 如果time_span_info中有time_span字段，从中提取天数
                frame_time_span = time_span_info.get('time_span', '')
                if frame_time_span and '-' in frame_time_span:
                    try:
                        frame_day = int(frame_time_span.split('-')[0])
                    except (ValueError, IndexError):
                        frame_day = 0
            
            # 比较函数：先比较天数，再比较timestamp
            def is_before_frame(cap):
                cap_time_span = cap.get('time_span', '')
                cap_timestamp = cap.get('timestamp', 0)
                
                # 提取caption中的天数
                cap_day = 0
                if cap_time_span and '-' in cap_time_span:
                    try:
                        cap_day = int(cap_time_span.split('-')[0])
                    except (ValueError, IndexError):
                        cap_day = 0
                
                # 先比较天数
                if cap_day < frame_day:
                    return True
                elif cap_day > frame_day:
                    return False
                else:
                    # 天数相同，比较timestamp
                    return cap_timestamp <= frame_timestamp
            
            for level in ['second_captions', 'min_captions', 'hour_captions']:
                level_captions = accumulated_captions.get(level, [])       # 判断每个caption的结束时间是否在当前时间之前
                # Filter captions before current frame (先比较天数，再比较timestamp)
                before_frame = [
                    cap for cap in level_captions 
                    if is_before_frame(cap)
                ]
                # Get the most recent max_captions_per_level captions
                selected_captions[level] = before_frame # [-max_captions_per_level:] if len(before_frame) > max_captions_per_level else before_frame
        else:
            # If no timestamp, just take the most recent ones
            for level in ['second_captions', 'min_captions', 'hour_captions']:
                level_captions = accumulated_captions.get(level, [])
                selected_captions[level] = level_captions[-max_captions_per_level:] if len(level_captions) > max_captions_per_level else level_captions
        
        # Separate current and historical captions
        # The most recent second_caption is current, all others are historical
        current_second_caption = None
        recent_minute_captions = None
        recent_hour_captions = None
        
        if selected_captions['second_captions']:
            # The last one is the most recent (current)
            current_second_caption = selected_captions['second_captions'][-1]
        if datasets_type == "egolife" and selected_captions['min_captions']:
            # All others are historical
            recent_minute_captions = selected_captions['min_captions'][-1]
        if datasets_type == "egolife" and selected_captions['hour_captions']:
            recent_hour_captions = selected_captions['hour_captions'][-1]
        
        # Format captions for prompt
        current_caption_parts = []
        if recent_minute_captions and datasets_type == "egolife":
            current_caption_parts.append("=== Recent Minute-level Caption ===")
            current_caption_parts.append(f"[{recent_minute_captions.get('time_span', 'N/A')}]: {recent_minute_captions.get('caption', '')}")
        # if recent_hour_captions and datasets_type == "egolife":
        #     current_caption_parts.append("=== Recent Hour-level Caption ===")
        #     current_caption_parts.append(f"[{recent_hour_captions.get('time_span', 'N/A')}]: {recent_hour_captions.get('caption', '')}")
        
        # Build prompt with captions
        prompt_parts = [proactive_prompt]
        
        if datasets_type == "proassist":
            prompt_parts.append("\n (4) CURRENT_5S_CAPTION:\n" + f"[{current_second_caption.get('time_span', 'N/A')}]: {current_second_caption.get('caption', '')}")
        else:
            prompt_parts.append("\n CURRENT_CAPTION:\n" + f"[{current_second_caption.get('time_span', 'N/A')}]: {current_second_caption.get('caption', '')}")
        
        if current_caption_parts:
            prompt_parts.append("\nRecent Caption:\n" + "\n".join(current_caption_parts))
        
        # if selected_captions['hour_captions'] and datasets_type == "egolife":        # 对于egolife数据集，我们还提供之前的所有的hour caption的时间窗口供模型选择
        #     all_selected_hour_windows = [cap.get('time_span', '') for cap in selected_captions['hour_captions'][:-1]]
        #     prompt_parts.append("\nAvailable Hour Windows:\n" + "\n".join(all_selected_hour_windows))
        
        # 提取每天视频的开始和结束时间
        if datasets_type == "egolife" and accumulated_captions:
            def extract_day_time_range(time_span_str):
                """从time_span字符串中提取day和开始时间、结束时间"""
                if not time_span_str or not isinstance(time_span_str, str):
                    return None, None, None
                
                # 移除可能的索引后缀（如 "_0"）
                time_span = time_span_str.rsplit('_', 1)[0] if '_' in time_span_str else time_span_str
                
                # 格式可能是: "1-12:00:00-12:00:30" 或 "DAY1-12:00:00-12:00:30"
                parts = time_span.split('-')
                if len(parts) < 3:
                    return None, None, None
                
                # 提取day
                day_str = parts[0]
                try:
                    if day_str.startswith('DAY'):
                        day = int(day_str[3:])
                    else:
                        day = int(day_str)
                except (ValueError, IndexError):
                    return None, None, None
                
                # 提取开始时间和结束时间
                start_time = parts[1]  # HH:MM:SS
                # 结束时间可能在parts[2]，但如果有索引后缀，需要移除
                end_time_str = parts[2] if len(parts) > 2 else parts[1]
                # 移除可能的索引后缀
                end_time = end_time_str.rsplit('_', 1)[0] if '_' in end_time_str else end_time_str
                
                return day, start_time, end_time
            
            # 从所有second_captions中提取每天的时间范围
            day_time_ranges = {}  # {day: {'start': 'HH:MM:SS', 'end': 'HH:MM:SS'}}
            
            all_second_captions = accumulated_captions.get('second_captions', [])
            for cap in all_second_captions:
                time_span = cap.get('time_span', '')
                if not time_span:
                    continue
                
                day, start_time, end_time = extract_day_time_range(time_span)
                if day is None:
                    continue
                
                if day not in day_time_ranges:
                    day_time_ranges[day] = {'start': start_time, 'end': end_time}
                else:
                    # 更新最早开始时间和最晚结束时间
                    if start_time < day_time_ranges[day]['start']:
                        day_time_ranges[day]['start'] = start_time
                    if end_time > day_time_ranges[day]['end']:
                        day_time_ranges[day]['end'] = end_time
            
            # 格式化输出每天的时间范围
            if day_time_ranges:
                day_range_lines = []
                for day in sorted(day_time_ranges.keys()):
                    start = day_time_ranges[day]['start']
                    end = day_time_ranges[day]['end']
                    day_range_lines.append(f"DAY{day}: {start} - {end}")
                
                prompt_parts.append("\nVideo Time Ranges by Day:\n" + "\n".join(day_range_lines))
        
        full_prompt = "\n".join(prompt_parts)
        
        # Call Gemini model using _llm.py function
        response = asyncio.run(self.llm.best_model_func_raw(
            self.llm.best_model_name,
            full_prompt,
            system_prompt=None,
            history_messages=[],  # history_messages已经拼接在prompt中了
        ))
        
        # parsed_response = parse_gemini_json_response(response)
        return {
            'frame_timestamp': frame_timestamp,
            'selected_captions': selected_captions,
            'gemini_response': response,
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
            # 排序函数：先按time_span中的天数排序，再按timestamp排序
            def sort_key(x):
                time_span = x.get('time_span', '')
                timestamp = x.get('timestamp', 0)
                # 提取time_span中的天数（格式：天数-xx:xx:xx-xx:xx:xx）
                day = 0
                if time_span and '-' in time_span:
                    try:
                        day = int(time_span.split('-')[0])
                    except (ValueError, IndexError):
                        day = 0
                return (day, timestamp)
            
            second_captions_list = sorted(second_captions_list, key=sort_key)
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
                    'timestamp': timestamp,
                    'time_span': time_span, 
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
            
            # 检查是否有需要检索的服务（根据新格式：decision == "need_retrieval"）
            needs_retrieval = False
            retrieval_query = None
            time_key = None
            service_type = None
            sub_service_type = None
            
            for service in service_list:
                if isinstance(service, dict):
                    # 根据新格式，检查 decision 字段是否为 "need_retrieval"
                    decision = service.get('decision', '')
                    if decision == "need_retrieval":
                        memory_query = service.get('memory_query', '')
                        suspected_issue = service.get('suspected_issue', '')
                        time_key = proactive_result["selected_captions"]["second_captions"][-1]["time_span"]   #这里修改为当前时间段的时间
                        # 从新格式中不再有 service_main_type 和 service_sub_type，保留为空字符串
                        service_type = ''
                        sub_service_type = ''
                        
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
                                logger.info(f"Retrieval needed. Suspected issue: {suspected_issue}, Query: {retrieval_query}")
                                break
            
            # 如果需要检索，进行检索并再次调用Gemini
            if needs_retrieval and retrieval_query:
                logger.info(f"Proactive service requires retrieval. Query: {retrieval_query}")
                
                query_param = QueryParam(mode="videorag")
                
                # 判断caption_model的类别名称是否包含MiniCPM
                caption_model_class_name = type(self.caption_model).__name__
                use_minicpm = "MiniCPM" in caption_model_class_name
                
                retrieved_video_context, retrieved_chunk_context = loop.run_until_complete(streaming_videorag_query(
                    retrieval_query,
                    time_key,
                    service_type,
                    sub_service_type, 
                    datasets_type, 
                    self.entities_vdb,
                    self.text_chunks,
                    self.chunks_vdb,
                    self.video_segments,
                    self.video_segment_feature_vdb,
                    self.chunk_entity_relation_graph,
                    self.caption_model,
                    self.caption_processor,
                    query_param,
                    asdict(self),
                    use_minicpm=use_minicpm,
                ))
                retrieved_response = retrieved_video_context + "\n" + retrieved_chunk_context
                
                logger.info(f"Retrieved memory evidence for proactive service")
                
                # 第二次调用：使用对话历史的方式，优先使用缓存
                # 获取第一次的prompt和响应
                first_prompt = proactive_result.get('prompt_used', '')
                first_response = proactive_result.get('gemini_response', '')
                
                final_response = None
                
                if first_prompt and first_response:
                    # 构建历史消息
                    history = pack_user_ass_to_openai_messages(first_prompt, first_response)
                    
                    # 构建带记忆的prompt（只需要添加检索到的记忆部分）
                    if datasets_type == "egolife":
                        memory_prompt = PROMPTS["proactive_service_prompt_with_memory_simple"]
                    elif datasets_type == "proassist":
                        memory_prompt = PROASSIST_PROMPTS["proactive_service_prompt_with_memory"]
                    elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                        memory_prompt = EGOEXTRA_PROMPTS["proactive_service_prompt_with_memory_simple"]
                    else:
                        memory_prompt = HOLOASSIST_PROMPTS["proactive_service_prompt_with_memory"]
                    
                    # 添加检索到的记忆
                    continue_prompt = memory_prompt + "\n\nRetrieved memory:\n\n" + retrieved_response
                    
                    # 直接调用LLM（使用缓存版本），传入历史消息
                    try:
                        final_response = asyncio.run(self.llm.best_model_func_raw(
                            self.llm.best_model_name,
                            continue_prompt,
                            system_prompt=None,
                            history_messages=history,
                        ))
                        if final_response:
                            logger.info(f"Used cached response for proactive service with memory")
                    except Exception as e:
                        logger.warning(f"Failed to use cached response, falling back to function call: {e}")
                
                # 如果缓存未命中或失败，使用备选方案
                if not final_response:
                    logger.info(f"Using direct call (fallback) for proactive service with memory")
                    proactive_result_with_memory = self.process_frame_with_proactive_service(
                        datasets_type=datasets_type, 
                        frame_info=frame_info_for_proactive,
                        accumulated_captions=accumulated_captions,
                        max_captions_per_level=3,
                        history_messages=history_text,
                        retrieved_memory=retrieved_response
                    )
                    if proactive_result_with_memory:
                        final_response = proactive_result_with_memory.get('gemini_response')
                
                if final_response:
                    # 使用带记忆的结果
                    # service_list = parse_gemini_json_response(final_response)
                    proactive_result['gemini_next_response'] = final_response
            
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
                    'last_updated': int(time.time())
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
    
    def streaming_retrieval_with_user_queries(
        self,
        # dialog_json_path,
        datasets_type, 
        video_data, 
        accumulated_captions=None,
        proactive_service_history=None,
        load_from_checkpoint=False,
        task_topic=None,
        task_knowledge=None,
    ):
        """
        流式处理视频，同时接收用户在特殊时刻输入的问题，并进行检索。
        
        这个函数在 streaming_graph_construction 的基础上，增加了：
        1. 读取对话JSON文件，提取用户问题和对应的时间戳
        2. 在处理每一帧时，检查当前时间是否有用户问题
        3. 如果有用户问题，进行检索并返回结果
        
        Args:
            dialog_json_path: 对话JSON文件路径，包含用户问题和时间戳
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
            time_tolerance_seconds: 时间容差，用于匹配用户问题的时间戳（默认2秒）
            
        Returns:
            Dictionary with generated captions, accumulated captions, and query responses:
            {
                "second_captions": {time_span: caption, ...},
                "min_captions": {time_span: caption, ...},
                "hour_captions": {time_span: caption, ...},
                "accumulated_captions": {...},
                "user_query_responses": [
                    {
                        "query_time": float,
                        "user_question": str,
                        "frame_timestamp": int,
                        "retrieved_context": str,
                        "response": dict
                    },
                    ...
                ]
            }
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
            # 排序函数：先按time_span中的天数排序，再按timestamp排序
            def sort_key(x):
                time_span = x.get('time_span', '')
                timestamp = x.get('timestamp', 0)
                # 提取time_span中的天数（格式：天数-xx:xx:xx-xx:xx:xx）
                day = 0
                if time_span and '-' in time_span:
                    try:
                        day = int(time_span.split('-')[0])
                    except (ValueError, IndexError):
                        day = 0
                return (day, timestamp)
            
            second_captions_list = sorted(second_captions_list, key=sort_key)
            logger.info(f"Processing {len(second_captions_list)} second captions in chronological order")
        
        # Process each 10s caption for proactive service (按时间顺序)
        for idx, second_caption_info in enumerate(second_captions_list):
            time_span = second_caption_info.get('time_span')
            
            start_time, end_time = parse_time_span_to_seconds(time_span)
            
            system_info = []
            previous_user_questions = []  # 收集之前时刻的用户问题
            for turn_idx, turn in enumerate(video_data["conversation"]):
                if turn["role"] == "user":
                    if start_time <= turn["time"] <= end_time:
                        # 当前时间窗口内的用户问题
                        system_info.append({"role": "user", "content": turn["content"], "timestamp": turn["time"]})
                    elif turn["time"] < start_time:
                        # 之前时刻的用户问题
                        previous_user_questions.append({"timestamp": turn["time"], "content": turn["content"]})
                elif turn["role"] == "system":
                    system_info.append({"role": "system", "content": turn["content"]})
                    
            # 将对话历史格式化为文本 prompt
            formatted_system_info = format_system_info(system_info)
                    
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
            
            # 格式化历史记录（用于 proassist），包含用户问题和模型回复
            history_text = format_assistant_response_history_with_user_questions(
                proactive_service_history, 
                previous_user_questions
            )
        
            # Call proactive service with the 10s caption and accumulated captions
            frame_info_for_proactive = {
                'caption': caption_dict,
                'frame': None,  # Use caption instead of single frame
                'time_span_info': {
                    'timestamp': timestamp,
                    'time_span': time_span, 
                }
            }
            
            # 第一次调用：判断是否需要响应或检索
            proactive_result = self.process_frame_with_proactive_service(
                datasets_type=datasets_type,
                frame_info=frame_info_for_proactive,
                accumulated_captions=accumulated_captions,
                max_captions_per_level=3,
                history_messages=history_text,
                retrieved_memory=None,
                formatted_system_info=formatted_system_info,
            )
            
            if not proactive_result or proactive_result.get('gemini_response') is None:
                logger.warning(f"Proactive service returned None for time_span: {time_span}")
                continue
            
            # 解析模型输出
            gemini_response = proactive_result.get('gemini_response')
            parsed_decision = None
            
            # 尝试解析 JSON 响应
            try:
                # 如果是字符串，尝试解析 JSON
                if isinstance(gemini_response, str):
                    # 移除可能的 markdown 代码块标记
                    cleaned_response = gemini_response.strip()
                    if cleaned_response.startswith("```"):
                        # 移除代码块标记
                        lines = cleaned_response.split("\n")
                        cleaned_response = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned_response
                    
                    # 尝试解析为 JSON
                    if cleaned_response == "[]":
                        parsed_decision = []
                    else:
                        parsed_decision = json.loads(cleaned_response)
                elif isinstance(gemini_response, (dict, list)):
                    parsed_decision = gemini_response
                else:
                    logger.warning(f"Unexpected response type: {type(gemini_response)}")
                    parsed_decision = None
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}, raw response: {gemini_response}")
                parsed_decision = None
            
            # 处理决策
            needs_retrieval = False
            retrieval_query = None
            final_response = None
            
            if parsed_decision is None:
                logger.warning(f"Could not parse decision for time_span: {time_span}")
                continue
            elif parsed_decision == []:
                # 决策：不响应
                logger.info(f"No response needed for {time_span}")
                final_response = {"decision": "no_response"}
            elif isinstance(parsed_decision, dict):
                decision_type = parsed_decision.get("decision", "")
                
                if decision_type == "respond_now":
                    # 决策：立即响应
                    logger.info(f"Responding now for {time_span}")
                    final_response = parsed_decision
                    # 将响应添加到历史记录
                    proactive_service_history.append({
                        'timestamp': parsed_decision.get('timestamp', time_span),
                        'response': parsed_decision.get('response', ''),
                        'decision': 'respond_now',
                        'evidence': parsed_decision.get('evidence', '')
                    })
                elif decision_type == "need_retrieval":
                    # 决策：需要检索
                    needs_retrieval = True
                    retrieval_query_str = parsed_decision.get("retrieval_query", "")
                    
                    # 提取 query 内容（可能在 <query> 标签中）
                    if retrieval_query_str:
                        query_match = re.search(r'<query>(.*?)</query>', retrieval_query_str, re.IGNORECASE | re.DOTALL)
                        if query_match:
                            retrieval_query = query_match.group(1).strip()
                        else:
                            retrieval_query = retrieval_query_str.strip()
                    
                    logger.info(f"Retrieval needed for {time_span}, query: {retrieval_query}")
                else:
                    logger.warning(f"Unknown decision type: {decision_type}")
                    continue
            else:
                logger.warning(f"Unexpected decision format: {type(parsed_decision)}")
                continue
            
            # 如果需要检索，进行检索并再次调用
            if needs_retrieval and retrieval_query:
                from .streaming_op import QueryParam
                query_param = QueryParam(mode="videorag")
                
                # 判断caption_model的类别名称是否包含MiniCPM
                caption_model_class_name = type(self.caption_model).__name__
                use_minicpm = "MiniCPM" in caption_model_class_name
                
                retrieved_video_context, retrieved_chunk_context = loop.run_until_complete(
                    streaming_videorag_query(
                        retrieval_query,
                        time_span,  # 使用 time_span 作为 time_key
                        "",  # service_type
                        "",  # sub_service_type
                        datasets_type,
                        self.entities_vdb,
                        self.text_chunks,
                        self.chunks_vdb,
                        self.video_segments,
                        self.video_segment_feature_vdb,
                        self.chunk_entity_relation_graph,
                        self.caption_model,
                        self.caption_processor,
                        query_param,
                        asdict(self),
                        use_minicpm=use_minicpm,
                    )
                )
                retrieved_response = retrieved_video_context + "\n" + retrieved_chunk_context
                
                logger.info(f"Retrieved memory evidence for proactive service")
                
                # 第二次调用：使用对话历史的方式，优先使用缓存
                # 获取第一次的prompt和响应
                first_prompt = proactive_result.get('prompt_used', '')
                first_response = proactive_result.get('gemini_response', '')
                
                gemini_response_with_memory = None
                
                if first_prompt and first_response:
                    # 构建历史消息
                    history = pack_user_ass_to_openai_messages(first_prompt, first_response)
                    
                    # 构建带记忆的prompt（只需要添加检索到的记忆部分）
                    memory_prompt = PROASSIST_PROMPTS["proactive_service_prompt_with_memory_simple"]
                    
                    # 添加检索到的记忆（使用与 prompt 模板一致的格式）
                    continue_prompt = memory_prompt + "\n\nRETRIEVED_MEMORY_EVIDENCE:\n\n" + retrieved_response
                    
                    # 如果是 proassist 且有格式化后的 system_info，将其拼接在 prompt 后面
                    # formatted_system_info 已经包含了正确的格式（(1) SYSTEM_PROMPT 和 (2) USER_PROMPT），直接添加即可
                    if datasets_type == "proassist" and formatted_system_info:
                        continue_prompt = continue_prompt + "\n\n" + formatted_system_info
                    
                    # 直接调用LLM（使用缓存版本），传入历史消息
                    try:
                        gemini_response_with_memory = asyncio.run(self.llm.best_model_func_raw(
                            self.llm.best_model_name,
                            continue_prompt,
                            system_prompt=None,
                            history_messages=history,
                        ))
                        if gemini_response_with_memory:
                            logger.info(f"Used cached response for proactive service with memory")
                    except Exception as e:
                        logger.warning(f"Failed to use cached response, falling back to function call: {e}")
                
                # 如果缓存未命中或失败，使用备选方案
                if not gemini_response_with_memory:
                    logger.info(f"Using direct call (fallback) for proactive service with memory")
                    proactive_result_with_memory = self.process_frame_with_proactive_service(
                        datasets_type=datasets_type,
                        frame_info=frame_info_for_proactive,
                        accumulated_captions=accumulated_captions,
                        max_captions_per_level=3,
                        history_messages=history_text,
                        retrieved_memory=retrieved_response,
                        formatted_system_info=formatted_system_info,
                    )
                    if proactive_result_with_memory:
                        gemini_response_with_memory = proactive_result_with_memory.get('gemini_response')
                
                if gemini_response_with_memory:
                    # 解析带记忆的响应
                    try:
                        if isinstance(gemini_response_with_memory, str):
                            cleaned_response = gemini_response_with_memory.strip()
                            if cleaned_response.startswith("```"):
                                lines = cleaned_response.split("\n")
                                cleaned_response = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned_response
                            
                            if cleaned_response == "[]":
                                final_response = {"decision": "no_response"}
                            else:
                                final_response = json.loads(cleaned_response)
                        elif isinstance(gemini_response_with_memory, dict):
                            final_response = gemini_response_with_memory
                        else:
                            final_response = None
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response with memory: {e}")
                        final_response = None
                    
                    if final_response and isinstance(final_response, dict):
                        decision_type = final_response.get("decision", "")
                        if decision_type == "respond_now":
                            # 将响应添加到历史记录
                            proactive_service_history.append({
                                'timestamp': final_response.get('timestamp', time_span),
                                'response': final_response.get('response', ''),
                                'decision': 'respond_now',
                                'evidence': final_response.get('evidence', '')
                            })
            
            # 保存结果
            if final_response:
                proactive_result['final_decision'] = final_response
                proactive_responses.append(proactive_result)
                logger.info(f"Proactive service processed for {time_span}: {final_response.get('decision', 'unknown')}")
            
            # 标记当前time_span为已处理，并保存检查点（每处理完一个就保存，支持断点续传）
            processed_time_spans.add(time_span)
            try:
                checkpoint_data = {
                    'processed_time_spans': list(processed_time_spans),
                    'proactive_responses': proactive_responses,
                    'proactive_service_history': proactive_service_history,
                    'last_processed_time_span': time_span,
                    'last_processed_timestamp': timestamp,
                    'last_updated': int(time.time())
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


    def process_egolife_proactive_service(
        self,
        datasets_type, 
        accumulated_captions=None,
        proactive_service_history=None,
        load_from_checkpoint=False
    ):
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
            # 排序函数：先按time_span中的天数排序，再按timestamp排序
            def sort_key(x):
                time_span = x.get('time_span', '')
                timestamp = x.get('timestamp', 0)
                # 提取time_span中的天数（格式：天数-xx:xx:xx-xx:xx:xx）
                day = 0
                if time_span and '-' in time_span:
                    try:
                        day = int(time_span.split('-')[0])
                    except (ValueError, IndexError):
                        day = 0
                return (day, timestamp)
            
            second_captions_list = sorted(second_captions_list, key=sort_key)
            logger.info(f"Processing {len(second_captions_list)} second captions in chronological order")
        
        # Process each 10s caption for proactive service (按时间顺序)
        # 用于保存上一次的检索信息，以抑制频繁检索
        last_pre_retrieval_probe = None
        
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
                    'timestamp': timestamp,
                    'time_span': time_span, 
                }
            }
            
            # 第一次调用：使用Gemini判断是否需要主动服务
            # 传入之前的检索信息以抑制频繁检索
            history_text = format_proactive_history(proactive_service_history)
            proactive_result = self.process_frame_with_proactive_service(
                datasets_type=datasets_type,
                frame_info=frame_info_for_proactive,
                accumulated_captions=accumulated_captions,
                max_captions_per_level=3,
                history_messages=history_text,
                retrieved_memory=None,
                pre_retrieval_probe=last_pre_retrieval_probe
            )
            
            if not proactive_result or proactive_result.get('gemini_response') is None:
                logger.warning(f"Proactive service returned None for time_span: {time_span}")
                continue
            
            gemini_response = proactive_result.get('gemini_response')
            
            # 解析新格式的响应：应该包含 finalized_services 和 episodic_longterm_probe
            parsed_response = None
            if isinstance(gemini_response, str):
                try:
                    parsed_response = json.loads(gemini_response)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    parsed_response = None
            elif isinstance(gemini_response, dict):
                parsed_response = gemini_response
            else:
                logger.warning(f"Unexpected response type: {type(gemini_response)}")
                parsed_response = None
                
            if parsed_response["episodic_longterm_probe"] != {}:
                pass
            
            # 提取 finalized_services 和 episodic_longterm_probe
            finalized_services = {}
            episodic_longterm_probe = {}
            service_list = []
            
            if parsed_response:
                # 提取 finalized_services
                finalized_services = parsed_response.get('finalized_services', {})
                suppressed_reason = None
                
                if isinstance(finalized_services, dict):
                    # 检查是否是 suppressed 格式
                    decision = finalized_services.get('decision', 'none')
                    if decision == 'suppressed':
                        suppressed_reason = finalized_services.get('reason', '')
                        service_list = []  # suppressed 时没有服务
                    else:
                        # 检查是否是包含 services 列表的格式
                        services = finalized_services.get('services', [])
                        if isinstance(services, list) and len(services) > 0:
                            service_list = services
                elif isinstance(finalized_services, list):
                    # 如果 finalized_services 直接是服务列表（新格式）
                    service_list = finalized_services
                
                # 保存 suppressed_reason 到 proactive_result
                if suppressed_reason:
                    proactive_result['suppressed_reason'] = suppressed_reason
                
                # 提取 episodic_longterm_probe
                episodic_longterm_probe = parsed_response.get('episodic_longterm_probe', {})
            
            # 检查是否有需要检索的服务（根据新格式：检查 episodic_longterm_probe 中的 retrieval_query）
            needs_retrieval = False
            retrieval_query = None
            time_key = None
            suspected_service_type = None
            service_type = None
            sub_service_type = None
            
            # 辅助函数：转换时间戳格式从 "1-19:02:42-20:06:57" 转换为 "DAY1-19:02:42-20:06:57"
            def convert_timestamp_format(timestamp_str):
                """将时间戳格式从 '1-19:02:42-20:06:57' 转换为 'DAY1-19:02:42-20:06:57'"""
                if not timestamp_str or not isinstance(timestamp_str, str):
                    return timestamp_str
                # 如果已经是 DAY 格式，直接返回
                if timestamp_str.startswith('DAY'):
                    return timestamp_str
                # 处理格式转换
                parts = timestamp_str.split('-')
                if len(parts) >= 3:
                    day_num = parts[0]
                    return f"DAY{day_num}-{'-'.join(parts[1:])}"
                return timestamp_str
            
            # 辅助函数：将DAY格式时间转换为可比较的秒数
            def day_timestamp_to_seconds(day_timestamp):
                """将DAY格式时间戳（如'DAY1-12:00:00'）转换为总秒数用于比较"""
                if not day_timestamp or not isinstance(day_timestamp, str):
                    return 0
                # 移除DAY前缀
                if day_timestamp.startswith('DAY'):
                    day_timestamp = day_timestamp[3:]
                # 分割day和时间
                parts = day_timestamp.split('-', 1)
                if len(parts) < 2:
                    return 0
                day_num = int(parts[0]) if parts[0].isdigit() else 0
                time_str = parts[1]
                # 解析HH:MM:SS
                time_parts = time_str.split(':')
                if len(time_parts) < 3:
                    return 0
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds = int(float(time_parts[2]))  # 处理可能的秒数小数部分
                # 返回总秒数（假设每天最多24小时，跨天用day_num区分）
                return day_num * 86400 + hours * 3600 + minutes * 60 + seconds
            
            # 辅助函数：比较两个DAY格式时间戳
            def compare_day_timestamps(timestamp1, timestamp2):
                """比较两个DAY格式时间戳，返回-1(t1<t2), 0(t1==t2), 1(t1>t2)"""
                seconds1 = day_timestamp_to_seconds(timestamp1)
                seconds2 = day_timestamp_to_seconds(timestamp2)
                if seconds1 < seconds2:
                    return -1
                elif seconds1 > seconds2:
                    return 1
                else:
                    return 0
            
            # 判断并调整检索时间范围
            def adjust_retrieval_time_range(response_time_range, video_start_time, video_end_time):
                """
                根据规则调整检索时间范围
                
                Args:
                    response_time_range: LLM返回的时间范围，格式为"DAY1-12:00:00-DAY3-12:00:00"或""
                    video_start_time: 视频起始时间，格式为"DAY1-12:00:00"
                    video_end_time: 视频结束时间，格式为"DAY3-12:00:00"
                
                Returns:
                    调整后的时间范围字符串，格式为"DAY1-12:00:00-DAY3-12:00:00"
                """
                # 如果输出是空字符串，返回整个视频范围
                if not response_time_range or not response_time_range.strip():
                    return f"{video_start_time}-{video_end_time}"
                
                # 解析response中的时间范围
                # 格式可能是: DAY1-12:00:00-DAY3-12:00:00
                # 使用正则表达式或简单的字符串分割来解析
                parts = response_time_range.strip().split('-')
                if len(parts) < 4:  # 至少需要DAY1-HH:MM:SS-DAY2-HH:MM:SS
                    # 如果格式不正确，返回整个视频范围
                    logger.warning(f"Invalid time range format: {response_time_range}, using full video range")
                    return f"{video_start_time}-{video_end_time}"
                
                # 找到所有DAY开头的部分
                day_indices = []
                for i, part in enumerate(parts):
                    if part.startswith('DAY'):
                        day_indices.append(i)
                
                if len(day_indices) < 2:
                    logger.warning(f"Could not find two DAY timestamps in: {response_time_range}, using full video range")
                    return f"{video_start_time}-{video_end_time}"
                
                # 构建开始和结束时间字符串
                # 第一个DAY部分：DAY1-12:00:00
                start_day_idx = day_indices[0]
                start_time = f"{parts[start_day_idx]}-{parts[start_day_idx+1]}"
                
                # 第二个DAY部分：DAY3-12:00:00
                end_day_idx = day_indices[1]
                end_time = f"{parts[end_day_idx]}-{parts[end_day_idx+1]}"
                
                # 根据规则调整时间范围
                start_earlier = compare_day_timestamps(start_time, video_start_time) < 0
                end_later = compare_day_timestamps(end_time, video_end_time) > 0
                
                # 规则1: 如果开始时间比视频起始时间早，同时结束时间比视频结束时间晚，返回整个视频范围
                if start_earlier and end_later:
                    logger.info(f"Start time {start_time} is earlier than video start {video_start_time} and end time {end_time} is later than video end {video_end_time}, using full video range")
                    return f"{video_start_time}-{video_end_time}"
                
                # 规则2: 如果开始时间比视频起始时间早（但结束时间在范围内），返回整个视频范围
                if start_earlier:
                    logger.info(f"Start time {start_time} is earlier than video start {video_start_time}, using full video range")
                    return f"{video_start_time}-{video_end_time}"
                
                # 规则3: 如果结束时间比视频结束时间晚（但开始时间在范围内），使用开始时间-视频结束时间
                if end_later:
                    logger.info(f"End time {end_time} is later than video end {video_end_time}, adjusting to {start_time}-{video_end_time}")
                    return f"{start_time}-{video_end_time}"
                
                # 规则4: 如果开始时间仍然比结束时间晚，返回整个视频范围
                if compare_day_timestamps(start_time, end_time) >= 0:
                    logger.warning(f"Start time {start_time} is after end time {end_time}, using full video range")
                    return f"{video_start_time}-{video_end_time}"
                
                return f"{start_time}-{end_time}"
            
            # 从 episodic_longterm_probe 中提取检索查询（新格式：直接包含 retrieval_query 和 suspected_service_type）
            if isinstance(episodic_longterm_probe, dict) and episodic_longterm_probe:
                # 提取 suspected_service_type
                suspected_service_type = episodic_longterm_probe.get('suspected_service_type', '')
                # 提取 retrieval_query（自然语言句子，包含时间提示）
                retrieval_query = episodic_longterm_probe.get('retrieval_query', '')
                
                # 更新last_pre_retrieval_probe，用于下一次调用时抑制频繁检索
                if suspected_service_type or retrieval_query:
                    last_pre_retrieval_probe = {
                        'time_span': time_span,  # 记录激活时间
                        'suspected_service_type': suspected_service_type,
                        'retrieval_query': retrieval_query
                    }
                
                current_timestamp = "DAY" + time_span
                time_span_convert_prompt = PROMPTS["time_convert"].format(current_timestamp=current_timestamp, retrieval_query=retrieval_query)
                response = asyncio.run(self.llm.best_model_func_raw(
                    self.llm.best_model_name,
                    time_span_convert_prompt,
                    system_prompt=None,
                    history_messages=[],  # history_messages已经拼接在prompt中了
                ))
                
                # 从accumulated_captions中获取视频的起始和结束时间
                # 注意：用户会提供视频的起始和结束时间，这里先尝试从captions中提取作为备选
                video_start_time = None
                video_end_time = None
                second_captions_list = accumulated_captions.get('second_captions', [])
                if second_captions_list:
                    # 获取最早和最晚的time_span
                    first_caption = second_captions_list[0]
                    video_start_time = "DAY" + first_caption.get('time_span', '')
                    video_end_time = current_timestamp
                    
                    first_parts = video_start_time.split('-')
                    if len(first_parts) >= 2:
                        video_start_time = f"{first_parts[0]}-{first_parts[1]}"
                    
                    last_parts = video_end_time.split('-')
                    if len(last_parts) >= 2:
                        video_end_time = f"{last_parts[0]}-{last_parts[1]}"
                
                # TODO: 如果用户提供了video_start_time和video_end_time参数，应该使用用户提供的值
                # 这里假设用户会通过某种方式提供这些值（例如作为函数参数或类属性）
                # 如果无法获取，使用从captions提取的值
                if not video_start_time or not video_end_time:
                    logger.warning(f"Could not determine video time range from captions, using response as-is")
                    final_time_range = response.strip() if response else ""
                else:
                    # 调用函数调整时间范围
                    final_time_range = adjust_retrieval_time_range(response, video_start_time, video_end_time)
                    logger.info(f"Adjusted retrieval time range: {final_time_range} (original: {response})")
                
                # 如果有检索查询，设置检索标志
                if retrieval_query and isinstance(retrieval_query, str) and retrieval_query.strip():
                    needs_retrieval = True
                    # 使用调整后的时间范围作为 time_key（如果已计算）
                    if 'final_time_range' in locals() and final_time_range:
                        time_key = final_time_range
                    else:
                        # 否则使用当前时间窗口作为 time_key
                        time_key = proactive_result["selected_captions"]["second_captions"][-1]["time_span"] if proactive_result.get("selected_captions", {}).get("second_captions") else None
                    logger.info(f"Retrieval needed. Suspected service type: {suspected_service_type}, Retrieval query: {retrieval_query}, Time range: {time_key}")
            
            # 如果需要检索，进行检索并再次调用Gemini
            if needs_retrieval and retrieval_query and time_key:
                logger.info(f"Proactive service requires retrieval. Query: {retrieval_query}, Time range: {time_key}")
                
                query_param = QueryParam(mode="videorag")
                
                # 判断caption_model的类别名称是否包含MiniCPM
                caption_model_class_name = type(self.caption_model).__name__
                use_minicpm = "MiniCPM" in caption_model_class_name
                
                # 使用streaming_videorag_query进行检索（对于egolife数据集，这个函数会处理时间范围）
                retrieved_video_context, retrieved_chunk_context = loop.run_until_complete(streaming_egolife_query(
                    retrieval_query,
                    time_key,
                    "",
                    "", 
                    datasets_type, 
                    self.entities_vdb,
                    self.text_chunks,
                    self.chunks_vdb,
                    self.video_segments,
                    self.video_segment_feature_vdb,
                    self.chunk_entity_relation_graph,
                    self.caption_model,
                    self.caption_processor,
                    query_param,
                    asdict(self),
                    use_minicpm=use_minicpm,
                ))
                retrieved_response = json.dumps({"suspect_service_type": suspected_service_type,
                                                 "retrieval_query": retrieval_query,
                                                 "retrieved_memory": retrieved_video_context + "\n" + retrieved_chunk_context})
                
                logger.info(f"Retrieved memory evidence for proactive service")
                
                # 第二次调用：使用对话历史的方式，直接调用LLM
                # 获取第一次的prompt和响应
                first_prompt = proactive_result.get('prompt_used', '')
                first_response = proactive_result.get('gemini_response', '')
                
                if first_prompt and first_response:
                    # 构建历史消息
                    history = pack_user_ass_to_openai_messages(first_prompt, first_response)
                    
                    # 构建带记忆的prompt（只需要添加检索到的记忆部分）
                    if datasets_type == "egolife":
                        memory_prompt = PROMPTS["proactive_service_prompt_with_memory_simple"]
                    elif datasets_type == "proassist":
                        memory_prompt = PROASSIST_PROMPTS["proactive_service_prompt_with_memory"]
                    elif datasets_type == "egoextra" or datasets_type == "captioncook4d":
                        memory_prompt = EGOEXTRA_PROMPTS["proactive_service_prompt_with_memory_simple"]
                    else:
                        memory_prompt = HOLOASSIST_PROMPTS["proactive_service_prompt_with_memory"]
                    
                    # 添加检索到的记忆
                    continue_prompt = memory_prompt + "\n\nRetrieved memory:\n\n" + retrieved_response
                    
                    # 直接调用LLM，传入历史消息
                    final_response = asyncio.run(self.llm.best_model_func_raw(
                        self.llm.best_model_name,
                        continue_prompt,
                        system_prompt=None,
                        history_messages=history,
                    ))
                    
                    # 更新proactive_result
                    proactive_result['gemini_response'] = final_response  # 更新为带记忆的响应
                    proactive_result['retrieved_memory'] = retrieved_response  # 保存检索到的记忆
                else:
                    logger.warning(f"Missing first prompt or response, falling back to function call")
                    # 如果缺少第一次的prompt或响应，回退到原来的方式
                    proactive_result_with_memory = self.process_frame_with_proactive_service(
                        datasets_type=datasets_type, 
                        frame_info=frame_info_for_proactive,
                        accumulated_captions=accumulated_captions,
                        max_captions_per_level=3,
                        history_messages=history_text,
                        retrieved_memory=retrieved_response
                    )
                    
                    if proactive_result_with_memory:
                        final_response = proactive_result_with_memory.get('gemini_response')
                        proactive_result['gemini_response'] = final_response
                        proactive_result['retrieved_memory'] = retrieved_response
                
                # 统一处理：重新解析更新后的响应（如果final_response已定义）
                if 'final_response' in locals() and final_response:
                    if isinstance(final_response, str):
                        try:
                            parsed_response = json.loads(final_response)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON response after retrieval: {e}")
                            parsed_response = None
                    elif isinstance(final_response, dict):
                        parsed_response = final_response
                    else:
                        parsed_response = None
                    
                    # 重新提取服务列表
                    if parsed_response:
                        finalized_services = parsed_response  # 新格式：响应直接是finalized_services
                        suppressed_reason = None
                        
                        if isinstance(finalized_services, dict):
                            # 检查是否是 suppressed 格式
                            decision = finalized_services.get('decision', 'none')
                            if decision == 'suppressed':
                                suppressed_reason = finalized_services.get('reason', '')
                                service_list = []  # suppressed 时没有服务
                            else:
                                # 检查是否是包含 services 列表的格式
                                services = finalized_services.get('services', [])
                                if isinstance(services, list) and len(services) > 0:
                                    service_list = services
                        elif isinstance(finalized_services, list):
                            # 如果 finalized_services 直接是服务列表（新格式）
                            service_list = finalized_services
                        
                        # 保存 suppressed_reason 到 proactive_result
                        if suppressed_reason:
                            proactive_result['suppressed_reason'] = suppressed_reason
            
            # 保存最终结果并更新历史记录
            if service_list:
                proactive_responses.append(proactive_result)
                logger.info(f"Proactive service triggered for {time_span}: {len(service_list)} service(s)")
                
                # 将每个服务的service_sub_type, user_prompt, trigger_time_window添加到历史记录
                for service in service_list:
                    if isinstance(service, dict):
                        # 转换时间戳格式：从 "1-19:02:42-20:06:57" 转换为 "DAY1-19:02:42-20:06:57"
                        trigger_time_window = convert_timestamp_format(service.get('trigger_time_window', ''))
                        
                        proactive_service_history.append({
                            'service_main_type': service.get('service_main_type', ''),
                            'service_sub_type': service.get('service_sub_type', ''),
                            'user_prompt': service.get('user_prompt', ''),
                            'trigger_time_window': trigger_time_window,
                            'trigger_evidence': service.get('trigger_evidence', ''),
                            'confidence': service.get('confidence', '')
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
                    'last_updated': int(time.time())
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
    