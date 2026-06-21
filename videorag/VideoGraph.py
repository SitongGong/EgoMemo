import importlib
import logging
import json
import torch
import logging
import asyncio
from datetime import timedelta
from transformers import AutoProcessor

from ._llm import gemini_complete_with_image_sync, gemini_complete_if_cache
from .llm.qwen_vl import mllm_response
from prompt import *
from videograph import VideoRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

class VideoGraphAgent:
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize VideoGraphAgent with specified model.
        
        Args:
            model_name: Name of the model to use (e.g., "qwenvl_2_5_7b_instruct")
            **kwargs: Additional arguments for model initialization
        """
        self.model_name = model_name
        
        # 首先加载Qwen-VL模型用来为视频生成captions
        # 使用 videorag.llm.qwen_vl 模块
        model_path = next(
            (model_path for key, model_path in MODEL_MAP.items() if key in model_name),
            None
        )
        
        if model_path is None:
            raise ValueError(f"Model '{model_name}' not found in MODEL_MAP. Available models: {list(MODEL_MAP.keys())}")
        
        # 加载对应的模型
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
                        device_map=kwargs.get("device_map", "auto"),
                    )
                except ImportError:
                    from transformers import Qwen2VLForConditionalGeneration
                    video_llm = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        device_map=kwargs.get("device_map", "auto"),
                    )
            elif "Qwen3" in model_path:
                from transformers import Qwen3VLForConditionalGeneration
                video_llm = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=kwargs.get("device_map", "auto"),
                )
            else:
                from transformers import Qwen2VLForConditionalGeneration
                video_llm = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map=kwargs.get("device_map", "auto"),
                )
            
            # 加载 processor
            processor = AutoProcessor.from_pretrained(model_path)
            image_processor = processor
            
            # 如果没有使用 device_map，手动移动到设备
            if kwargs.get("device_map") is None:
                device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
                video_llm.to(device)
            
            self.processor = processor
            self.video_llm = video_llm
            self.image_processor = image_processor
                    
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        logger.info(f"Successfully loaded model: {model_name} from {model_path}")
    
        # 加载Gemini模型用来建图（可选）
        # 使用 _llm.py 中的 Gemini 函数
        self.gemini_model_name = kwargs.get("gemini_model", "gemini-1.5-pro")
        gemini_api_key = kwargs.get("gemini_api_key", None)
        
        # 设置环境变量（如果提供了 API key）
        if gemini_api_key:
            import os
            os.environ['GOOGLE_API_KEY'] = gemini_api_key
            self.gemini_available = True
        else:
            # 检查环境变量中是否已有 API key
            import os
            self.gemini_available = os.environ.get('GOOGLE_API_KEY') is not None
            if not self.gemini_available:
                logger.warning("Gemini API key not provided. Gemini model will not be available.")
    
    def process_frame_with_proactive_service(self, frame_info, accumulated_captions, 
                                            proactive_prompt=None, max_captions_per_level=3):
        """
        Process a single frame with Gemini model for proactive service detection.
        This function is separate from the hierarchical caption generation.
        
        Args:
            frame_info: Dict containing frame information with keys:
                       - 'frame': base64-encoded frame
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
            
        Returns:
            Dict containing:
            {
                'frame_timestamp': int,
                'selected_captions': {
                    'second_captions': [...],
                    'min_captions': [...],
                    'hour_captions': [...]
                },
                'gemini_response': str or dict (parsed JSON if possible)
            }
        """
        if not self.gemini_available:
            logger.warning("Gemini model not available. Skipping proactive service detection.")
            return None
        
        if proactive_prompt is None:
            proactive_prompt = RPOACITVE_SERVICE_PROMPT
        
        frame = frame_info.get('frame')
        frame_timestamp = frame_info.get('time_span_info', {}).get('timestamp')
        
        if frame is None:
            logger.warning("Frame is None, skipping proactive service detection")
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
        
        # Prepare content for Gemini (frame + text)
        # Convert base64 string to PIL Image
        import base64
        from PIL import Image
        import io
        
        try:
            # Decode base64 string to bytes and create PIL Image
            frame_bytes = base64.b64decode(frame)
            frame_image = Image.open(io.BytesIO(frame_bytes))
            
            # Build Gemini content format (using the format expected by GeminiModel.generate)
            # The _process_content method will handle encoding the PIL Image
            gemini_content = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image", "image": frame_image}  # Pass PIL Image directly
                    ]
                }
            ]
        except Exception as e:
            logger.error(f"Failed to prepare frame for Gemini: {e}")
            return {
                'frame_timestamp': frame_timestamp,
                'selected_captions': selected_captions,
                'gemini_response': None,
                'error': f"Failed to encode frame: {e}"
            }
        
        # Call Gemini model using _llm.py function
        try:
            # Extract text and image from gemini_content
            text_prompt = full_prompt
            images = [frame_image]  # PIL Image
            
            # Use synchronous function from _llm.py
            response = gemini_complete_with_image_sync(
                model=self.gemini_model_name,
                prompt=text_prompt,
                images=images,
                system_prompt=None,
                temperature=0.7,
                max_tokens=8192
            )
            
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
                'gemini_response': parsed_response
            }
            
        except Exception as e:
            logger.error(f"Error calling Gemini for proactive service detection: {e}")
            return {
                'frame_timestamp': frame_timestamp,
                'selected_captions': selected_captions,
                'gemini_response': None,
                'error': str(e)
            }
            
    
    def _save_to_videorag(self, result: dict):
        """
        Save generated captions to VideoRAG storage.
        
        Args:
            result: Dictionary containing second_captions, min_captions, hour_captions
        """
        try:
            # Convert captions to VideoRAG format
            # VideoRAG's ainsert expects: {video_name: {segment_id: {content: str, ...}}}
            # We'll use a single video name for all streaming captions
            video_segment = {
                'streaming_captions': {}
            }
            
            # Add all captions as segments
            segment_index = 0
            
            # Add second-level captions
            for time_span, caption in result.get("second_captions", {}).items():
                segment_id = f"second_{segment_index}"
                video_segment['streaming_captions'][segment_id] = {
                    'content': caption,
                }
                segment_index += 1
            
            # Add minute-level captions
            for time_span, caption in result.get("min_captions", {}).items():
                segment_id = f"min_{segment_index}"
                video_segment['streaming_captions'][segment_id] = {
                    'content': caption,
                }
                segment_index += 1
            
            # Add hour-level captions
            for time_span, caption in result.get("hour_captions", {}).items():
                segment_id = f"hour_{segment_index}"
                video_segment['streaming_captions'][segment_id] = {
                    'content': caption,
                }
                segment_index += 1
            
            if not video_segment['streaming_captions']:
                logger.warning("No captions to save to VideoRAG")
                return
            
            # Call VideoRAG's ainsert asynchronously
            # Since construct_graph is synchronous, we use asyncio.run
            async def _async_insert():
                await self.video_rag.ainsert(video_segment)
            
            total_segments = len(video_segment['streaming_captions'])
            logger.info(f"Saving {total_segments} caption segments to VideoRAG...")
            asyncio.run(_async_insert())
            logger.info("Successfully saved captions to VideoRAG")
            
        except Exception as e:
            logger.error(f"Failed to save to VideoRAG: {e}", exc_info=True)
        
    def construct_graph(self, frame_time_data, interval_seconds=2, 
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
        
        # Default prompts if not provided
        if prompt_template is None:
            default_prompt_second = "Please describe what is happening in these video frames in detail."
        elif callable(prompt_template):
            default_prompt_second = prompt_template
        else:
            default_prompt_second = prompt_template
        
        if prompt_template_min is None:
            default_prompt_min = "Based on the following 10-second captions, generate a comprehensive 10-minute summary caption."
        elif callable(prompt_template_min):
            default_prompt_min = prompt_template_min
        else:
            default_prompt_min = prompt_template_min
        
        if prompt_template_hour is None:
            default_prompt_hour = "Based on the following 10-minute captions, generate a comprehensive 1-hour summary caption."
        elif callable(prompt_template_hour):
            default_prompt_hour = prompt_template_hour
        else:
            default_prompt_hour = prompt_template_hour
        
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
                    proactive_result = self.process_frame_with_proactive_service(
                        frame_info=frame_info,
                        accumulated_captions=accumulated_captions,
                        proactive_prompt=None,
                        max_captions_per_level=3
                    )
                    if proactive_result:
                        proactive_responses.append(proactive_result)
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
                                base64_frames=current_window_frames,
                                max_new_tokens=max_new_tokens,
                                has_image=True
                            )
                            second_captions[time_span] = caption
                            logger.info(f"Generated 10s caption for window: {time_span} ({len(current_window_frames)} frames)")
                            
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
                                            base64_frames=None,
                                            max_new_tokens=max_new_tokens,
                                            has_image=False
                                        )
                                        min_captions[min_time_span] = min_caption
                                        logger.info(f"Generated 10min caption for window: {min_time_span} ({len(min_window_second_captions)} 10s captions)")
                                        
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
                                                        base64_frames=None,
                                                        max_new_tokens=max_new_tokens,
                                                        has_image=False
                                                    )
                                                    hour_captions[hour_time_span] = hour_caption
                                                    logger.info(f"Generated 1h caption for window: {hour_time_span} ({len(hour_window_min_captions)} 10min captions)")
                                                    
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