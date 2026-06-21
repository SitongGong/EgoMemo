# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import logging
import os
import tempfile
import math
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import subprocess
from decord import VideoReader, cpu
from typing import Optional

# Disable moviepy logging
logging.getLogger('moviepy').setLevel(logging.ERROR)
# Disable moviepy's tqdm progress bar
logging.getLogger('moviepy.video.io.VideoFileClip').setLevel(logging.ERROR)
logging.getLogger('moviepy.audio.io.AudioFileClip').setLevel(logging.ERROR)

# Configure logging
logger = logging.getLogger(__name__)

def get_video_info(file_path):
    """Get video/audio information using appropriate libraries.
    
    Args:
        file_path (str): Path to video or audio file
        
    Returns:
        dict: Dictionary containing media metadata
    """
    file_info = {}
    file_info["path"] = file_path
    file_info["name"] = file_path.split("/")[-1]
    file_info["format"] = os.path.splitext(file_path)[1][1:].lower()
        
    # Handle video files using moviepy
    
    video = VideoFileClip(file_path)  # Disable logging for this instance
    
    # Get basic properties from moviepy
    file_info["fps"] = video.fps
    file_info["frames"] = int(video.fps * video.duration)
    file_info["duration"] = video.duration
    file_info["width"] = video.size[0]
    file_info["height"] = video.size[1]
    
    video.close()
    return file_info

def extract_frames(video, start_time=None, interval=None, sample_fps=10):
    # if start_time and interval are not provided, sample the whole video at sample_fps
    if start_time is None and interval is None:
        start_time = 0
        interval = video.duration

    frames = []
    frame_interval = 1.0 / sample_fps

    # Extract frames at specified intervals
    for t in np.arange(
        start_time, min(start_time + interval, video.duration), frame_interval
    ):
        frame = video.get_frame(t)
        # Convert frame to jpg and base64
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(base64.b64encode(buffer).decode("utf-8"))
        
    return frames


def _seconds_to_timestamp_number(seconds, start_time_number=None):
    """Convert seconds to timestamp number format (HHMMSSCC).
    
    Args:
        seconds (float): Time in seconds
        start_time_number (int, optional): Base time number to add to seconds
        
    Returns:
        int: Timestamp number in HHMMSSCC format
    """
    if start_time_number is not None:
        # Convert start_time_number to total seconds
        start_time_str = str(start_time_number).zfill(8)
        start_hours = int(start_time_str[0:2])
        start_minutes = int(start_time_str[2:4])
        start_seconds = int(start_time_str[4:6])
        start_centiseconds = int(start_time_str[6:8])
        start_total_seconds = start_hours * 3600 + start_minutes * 60 + start_seconds + start_centiseconds / 100.0
        absolute_seconds = start_total_seconds + seconds
    else:
        absolute_seconds = seconds
    
    # Convert to time_number format (HHMMSSCC)
    total_hours = int(absolute_seconds // 3600)
    remaining = absolute_seconds % 3600
    total_minutes = int(remaining // 60)
    remaining = remaining % 60
    total_seconds = int(remaining)
    centiseconds = int(round((remaining - total_seconds) * 100))
    
    # Ensure centiseconds is in valid range [0, 99]
    if centiseconds >= 100:
        total_seconds += 1
        centiseconds = 0
    if total_seconds >= 60:
        total_minutes += 1
        total_seconds = 0
    if total_minutes >= 60:
        total_hours += 1
        total_minutes = 0
    
    # Format as HHMMSSCC (hours can be > 24)
    timestamp_number = total_hours * 1000000 + total_minutes * 10000 + total_seconds * 100 + centiseconds
    return int(timestamp_number)


def sample_frames_by_interval(video_path, interval_seconds=2, output_format='base64', time_span_info=None,
                               max_seconds: Optional[float] = None,
                               min_seconds: Optional[float] = None):
    """Sample frames from video by taking the middle frame of each time interval.

    The video is divided into intervals of length interval_seconds, and the middle frame
    of each interval is sampled. For example, with interval_seconds=5:
    - Interval [0, 5s]: samples frame at 2.5s
    - Interval [5s, 10s]: samples frame at 7.5s
    - Interval [10s, 15s]: samples frame at 12.5s
    - etc.

    Args:
        video_path (str): Path to the video file
        interval_seconds (float): Length of each time interval in seconds (default: 2.0)
        output_format (str): Output format, either 'base64' or 'numpy' (default: 'base64')
        time_span_info (dict, optional): Dictionary containing time span information for the video.
            Should contain:
            - 'start_time_number': Start time as number (e.g., 11094300 for 11:09:43.00)
            - 'end_time_number': End time as number (e.g., 11100000 for 11:10:00.00)
            If provided, timestamps will be calculated relative to this time span.
        max_seconds (float, optional): Hard upper bound on采样时刻 (秒). 用于 OVO-Bench / StreamingBench
            等需要"截到 question time_stamp 之前"的协议: 不读取/解码超过该时刻的帧, 严格不泄漏未来.
            None = 用整段视频 duration. 默认 None.
        
    Returns:
        tuple: (frames, timestamps, time_ranges) where:
            - frames: List of frames. If output_format='base64', returns list of base64-encoded JPEG strings.
                     If output_format='numpy', returns list of numpy arrays (RGB format).
            - timestamps: List of timestamp numbers corresponding to each frame (same format as start_time_number).
                         If time_span_info is None, returns None.
            - time_ranges: List of dictionaries, each containing 'start' and 'end' representing the time range 
                          (interval) that each frame represents.
                          If time_span_info is provided, 'start' and 'end' are timestamp numbers (HHMMSSCC format).
                          If time_span_info is None, 'start' and 'end' are relative time in seconds.
    
    Example:
        >>> frames, timestamps, time_ranges = sample_frames_by_interval('video.mp4', interval_seconds=5, 
        ...                                                 time_span_info={'start_time_number': 11094300, 
        ...                                                                 'end_time_number': 11100000})
        >>> # Returns frames sampled at 2.5s, 7.5s, 12.5s, etc., their timestamps, and time ranges [0,5], [5,10], [10,15], etc.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")
    
    video = VideoFileClip(video_path)
    frames = []
    timestamps = []
    time_ranges = []
    
    try:
        # Calculate sampling times (relative to video start, in seconds)
        # Each interval [i*interval, (i+1)*interval] takes the middle frame at (i*interval + interval/2)
        duration = video.duration
        # max_seconds: OVO-Bench / StreamingBench 协议下"截到 max_ts"的早停 (避免泄漏未来)
        if max_seconds is not None and max_seconds > 0:
            duration = min(duration, float(max_seconds))
        # min_seconds: ESTP/ProAssist 等 clip 协议下跳过 [0, clip_start] 这段无关内容
        # 只对采样起点做 clip, 不平移时间轴: hypergraph 里 fact.temporal 仍是绝对秒
        skip_intervals = 0
        if min_seconds is not None and min_seconds > 0:
            skip_intervals = int(min_seconds // interval_seconds)
        exact_intervals = duration / interval_seconds
        num_intervals = int(np.ceil(exact_intervals))
        # 如果duration超过interval_seconds的倍数但不足1秒，则向下取整
        remainder = duration - int(exact_intervals) * interval_seconds
        if remainder > 0 and remainder < 1:
            num_intervals = int(exact_intervals)
        sampling_times = []

        for i in range(skip_intervals, num_intervals):
            # Calculate the middle point of each interval
            interval_start = i * interval_seconds
            interval_end = min((i + 1) * interval_seconds, duration)
            middle_time = (interval_start + interval_end) / 2.0

            # Clamp to valid range
            middle_time = min(middle_time, duration - 0.01)
            if middle_time >= 0:
                sampling_times.append(middle_time)
        
        sampling_times = np.array(sampling_times)
        
        # Get start_time_number for time range calculation
        if time_span_info is not None:
            start_time_number = time_span_info.get('start_time_number')
        else:
            start_time_number = "00000000"
        
        # Extract frames at specified intervals
        for i, t in enumerate(sampling_times):
            # Clamp time to video duration
            t = min(t, duration - 0.01)
            frame = video.get_frame(t)
            
            if output_format == 'base64':
                # Convert frame to JPEG and encode as base64
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames.append(base64.b64encode(buffer).decode("utf-8"))
            elif output_format == 'numpy':
                # Return numpy array directly (RGB format)
                frames.append(frame)
            else:
                raise ValueError(f"Unsupported output_format: {output_format}. Use 'base64' or 'numpy'")
            
            # Calculate timestamp if time_span_info is provided
            timestamp_number = _seconds_to_timestamp_number(t, start_time_number)
            timestamps.append(timestamp_number)
            
            # Frame at time t (middle of interval) represents [interval_start, interval_end]
            # 注意: i 是 sampling_times 里的索引 (0-based), 不是 interval_index;
            # 当 min_seconds > 0 时, i=0 对应的真实 interval_index = skip_intervals.
            # 用 t (frame 真实时间) 反推 interval_index 才是正确的绝对时间
            interval_index = int(round((t - interval_seconds / 2.0) / interval_seconds))
            range_start = interval_index * interval_seconds
            range_end = min((interval_index + 1) * interval_seconds, duration)
            
            # Convert time range to timestamp numbers if time_span_info is provided
            # Otherwise, return relative time ranges in seconds
            time_range = {
                'start': _seconds_to_timestamp_number(range_start, start_time_number),
                'end': _seconds_to_timestamp_number(range_end, start_time_number)
            }
            time_ranges.append(time_range)
        
        return frames, timestamps, time_ranges
        
    except Exception as e:
        logger.error(f"Error sampling frames from {video_path}: {str(e)}")
        raise
    finally:
        video.close()

def sample_base64_frames(video_path, interval_seconds=2, output_format='base64', time_span_info=None,
                          max_seconds: Optional[float] = None):
    """Sample frames from video by taking the middle frame of each time interval.
    
    The video is divided into intervals of length interval_seconds, and the middle frame
    of each interval is sampled. For example, with interval_seconds=5:
    - Interval [0, 5s]: samples frame at 2.5s
    - Interval [5s, 10s]: samples frame at 7.5s
    - Interval [10s, 15s]: samples frame at 12.5s
    - etc.
    
    Args:
        video_path (str): Path to the video file
        interval_seconds (float): Length of each time interval in seconds (default: 2.0)
        output_format (str): Output format, either 'base64' or 'numpy' (default: 'base64')
        time_span_info (dict, optional): Dictionary containing time span information for the video.
            Should contain:
            - 'start_time_number': Start time as number (e.g., 11094300 for 11:09:43.00)
            - 'end_time_number': End time as number (e.g., 11100000 for 11:10:00.00)
            If provided, timestamps will be calculated relative to this time span.
        
    Returns:
        tuple: (frames, timestamps, time_ranges) where:
            - frames: List of frames. If output_format='base64', returns list of base64-encoded JPEG strings.
                     If output_format='numpy', returns list of numpy arrays (RGB format).
            - timestamps: List of timestamp numbers corresponding to each frame (same format as start_time_number).
                         If time_span_info is None, returns None.
            - time_ranges: List of dictionaries, each containing 'start' and 'end' representing the time range 
                          (interval) that each frame represents.
                          If time_span_info is provided, 'start' and 'end' are timestamp numbers (HHMMSSCC format).
                          If time_span_info is None, 'start' and 'end' are relative time in seconds.
    
    Example:
        >>> frames, timestamps, time_ranges = sample_frames_by_interval('video.mp4', interval_seconds=5, 
        ...                                                 time_span_info={'start_time_number': 11094300, 
        ...                                                                 'end_time_number': 11100000})
        >>> # Returns frames sampled at 2.5s, 7.5s, 12.5s, etc., their timestamps, and time ranges [0,5], [5,10], [10,15], etc.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")
    
    video = VideoFileClip(video_path)
    frames = []
    timestamps = []
    time_ranges = []
    
    try:
        # Calculate sampling times (relative to video start, in seconds)
        # Each interval [i*interval, (i+1)*interval] takes the middle frame at (i*interval + interval/2)
        duration = video.duration
        # max_seconds: OVO-Bench / StreamingBench 协议下"截到 max_ts"的早停 (避免泄漏未来)
        if max_seconds is not None and max_seconds > 0:
            duration = min(duration, float(max_seconds))
        # min_seconds: ESTP/ProAssist 等 clip 协议下跳过 [0, clip_start] 这段无关内容
        # 只对采样起点做 clip, 不平移时间轴: hypergraph 里 fact.temporal 仍是绝对秒
        skip_intervals = 0
        if min_seconds is not None and min_seconds > 0:
            skip_intervals = int(min_seconds // interval_seconds)
        exact_intervals = duration / interval_seconds
        num_intervals = int(np.ceil(exact_intervals))
        # 如果duration超过interval_seconds的倍数但不足1秒，则向下取整
        remainder = duration - int(exact_intervals) * interval_seconds
        if remainder > 0 and remainder < 1:
            num_intervals = int(exact_intervals)
        sampling_times = []

        for i in range(skip_intervals, num_intervals):
            # Calculate the middle point of each interval
            interval_start = i * interval_seconds
            interval_end = min((i + 1) * interval_seconds, duration)
            middle_time = (interval_start + interval_end) / 2.0

            # Clamp to valid range
            middle_time = min(middle_time, duration - 0.01)
            if middle_time >= 0:
                sampling_times.append(middle_time)
        
        sampling_times = np.array(sampling_times)
        
        # Get start_time_number for time range calculation
        if time_span_info is not None:
            start_time_number = time_span_info.get('start_time_number')
        else:
            start_time_number = "00000000"
        
        # Extract frames at specified intervals
        for i, t in enumerate(sampling_times):
            # Clamp time to video duration
            t = min(t, duration - 0.01)
            frame = video.get_frame(t)
            
            if output_format == 'base64':
                # Convert frame to JPEG and encode as base64
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames.append(base64.b64encode(buffer).decode("utf-8"))
            elif output_format == 'numpy':
                # Return numpy array directly (RGB format)
                frames.append(frame)
            else:
                raise ValueError(f"Unsupported output_format: {output_format}. Use 'base64' or 'numpy'")
            
            # Calculate timestamp if time_span_info is provided
            if time_span_info is not None and start_time_number is not None:
                timestamp_number = _seconds_to_timestamp_number(t, start_time_number)
                timestamps.append(timestamp_number)
            else:
                timestamp_number = _seconds_to_timestamp_number(t, start_time_number)
                timestamps.append(timestamp_number)
            
            # Frame at time t (middle of interval) represents [interval_start, interval_end]
            # 注意: i 是 sampling_times 里的索引 (0-based), 不是 interval_index;
            # 当 min_seconds > 0 时, i=0 对应的真实 interval_index = skip_intervals.
            # 用 t (frame 真实时间) 反推 interval_index 才是正确的绝对时间
            interval_index = int(round((t - interval_seconds / 2.0) / interval_seconds))
            range_start = interval_index * interval_seconds
            range_end = min((interval_index + 1) * interval_seconds, duration)
            
            # Convert time range to timestamp numbers if time_span_info is provided
            # Otherwise, return relative time ranges in seconds
            if time_span_info is not None and start_time_number is not None:
                time_range = {
                    'start': _seconds_to_timestamp_number(range_start, start_time_number),
                    'end': _seconds_to_timestamp_number(range_end, start_time_number)
                }
                time_ranges.append(time_range)
            else:
                # Return relative time ranges in seconds
                time_range = {
                    'start': _seconds_to_timestamp_number(range_start, start_time_number),
                    'end': _seconds_to_timestamp_number(range_end, start_time_number)
                }
                time_ranges.append(time_range)
        
        if time_span_info is None:
            return frames, timestamps, time_ranges
        else:
            return frames, timestamps, time_ranges
        
    except Exception as e:
        logger.error(f"Error sampling frames from {video_path}: {str(e)}")
        raise
    finally:
        video.close()

# TODO: check if there is a better way to do this without repeatedly opening and closing the video file
def process_video_clip(video_path, fps=5, audio_fps=16000): 
    try: 
        base64_data = {}
        video = VideoFileClip(video_path)
        base64_data["video"] = base64.b64encode(open(video_path, "rb").read())
        base64_data["frames"] = extract_frames(video, sample_fps=fps)
        
        if video.audio is None:
            base64_data["audio"] = None
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav") as audio_tempfile:
                video.audio.write_audiofile(audio_tempfile.name, codec="pcm_s16le", fps=audio_fps)
                audio_tempfile.seek(0)
                base64_data["audio"] = base64.b64encode(audio_tempfile.read())
        
        video.close()
        return base64_data["video"], base64_data["frames"], base64_data["audio"]

    except Exception as e:
        logger.error(f"Error processing video clip: {str(e)}")
        raise

def verify_video_processing(video_path, output_dir, interval, strict=False):
    """Verify that a video was properly split into clips by checking the number of clips.
    
    Args:
        video_path (str): Path to original video file
        output_dir (str): Directory containing the split clips
        interval (float): Interval length in seconds used for splitting
        
    Returns:
        bool: True if verification passes, False otherwise
    """

    def has_video_and_audio(file_path):
        def has_stream(stream_type):
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", stream_type,
                "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                capture_output=True, text=True
            )
            return bool(result.stdout.strip())

        return has_stream("v:0") and has_stream("a:0")

    def has_static_segment(
        video_path,
        min_static_duration=5.0,
        diff_threshold=0.001,
    ) -> bool:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        min_static_frames = int(min_static_duration * fps)

        prev_gray = None
        consecutive_static_frames = 0

        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = np.mean(diff)

                if mean_diff < diff_threshold:
                    consecutive_static_frames += 1
                    if consecutive_static_frames >= min_static_frames:
                        cap.release()
                        return True
                else:
                    consecutive_static_frames = 0

            prev_gray = gray

        cap.release()
        return False

    try:
        if not os.path.exists(video_path):
            with open("logs/video_processing_failed.log", "a") as f:
                f.write(f"Error processing {video_path}: Video file not found.\n")
            logger.error(f"Error processing {video_path}: Video file not found.")
            return False
        # Get expected number of clips based on video duration
        video_info = get_video_info(video_path)
        expected_clips_num = math.ceil(int(video_info["duration"]) / interval)
        
        # Get actual number of clips in output directory
        clip_dir = output_dir
        
        if not os.path.exists(clip_dir):
            with open("logs/video_processing_failed.log", "a") as f:
                f.write(f"Error processing {video_path}: Clip directory {clip_dir} not found.\n")
            logger.error(f"Error processing {video_path}: Clip directory {clip_dir} not found.")
            return False
            
        actual_clips = [f for f in os.listdir(clip_dir) if os.path.isfile(os.path.join(clip_dir, f)) and f.split('.')[-1] in ['mp4', 'mov', 'webm']]
        actual_clips_num = len(actual_clips)
        
        if actual_clips_num != expected_clips_num:
            with open("logs/video_processing_failed.log", "a") as f:
                f.write(f"Error processing {video_path}: Expected {video_info['duration']}/{interval}={expected_clips_num} clips, but found {actual_clips_num} clips.\n")
            logger.error(f"Error processing {video_path}: Expected {video_info['duration']}/{interval}={expected_clips_num} clips, but found {actual_clips_num} clips.")
            return False

        if strict:
            clip_files = [os.path.join(clip_dir, clip) for clip in actual_clips]
            for clip_file in clip_files:
                clip_id = clip_file.split("/")[-1].split(".")[0]
                if not has_video_and_audio(clip_file):
                    with open("logs/video_processing_failed.log", "a") as f:
                        f.write(f"Error processing {clip_file}: No video or audio streams found.\n")
                    logger.error(f"Error processing {clip_file}: No video or audio streams found.")
                    return False
                if int(clip_id) < len(clip_files)-2 and has_static_segment(clip_file):
                    with open("logs/video_processing_failed.log", "a") as f:
                        f.write(f"Error processing {clip_file}: Has static segment.\n")
                    logger.error(f"Error processing {clip_file}: Has static segment.")
                    return False

        return True
        
    except Exception as e:
        with open("logs/video_processing_failed.log", "a") as f:
            f.write(f"Error verifying {video_path}: {e}\n")
        logger.error(f"Error verifying {video_path}: {e}")
        return False

def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    """Round time up to the nearest frame boundary based on fps.
    
    Args:
        time: Time in seconds
        fps: Frames per second
        min_time: Minimum allowed time
        max_time: Maximum allowed time
        
    Returns:
        Adjusted time value
    """
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

def extract_eyewo_video_frames(video_path, start_time, end_time, fps=2, output_format='base64'):
    """Extract frames from eyewo dataset video between start_time and end_time.
    
    This function extracts video frames from the specified time range, sampling at specified fps.
    The first frame timestamp starts from 1/(2*fps) seconds (e.g., 0.5s for fps=2).
    
    Args:
        video_path (str): Path to the video file
        start_time (float): Start time in seconds (relative to video start)
        end_time (float): End time in seconds (relative to video start)
        fps (int): Frames per second to sample (default: 2)
        output_format (str): Output format, either 'base64' or 'numpy' (default: 'base64')
        
    Returns:
        tuple: (frames, frame_timestamps, frame_time_ranges) where:
            - frames: List of frames. If output_format='base64', returns list of base64-encoded JPEG strings.
                     If output_format='numpy', returns list of numpy arrays (RGB format).
            - frame_timestamps: List of timestamps in seconds, starting from 1/(2*fps)
            - frame_time_ranges: List of dictionaries with 'start' and 'end' keys,
                                 representing the time range each frame represents,
                                 starting from 0
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if start_time is None or end_time is None:
        raise ValueError("start_time and end_time must be provided")
    
    if start_time >= end_time:
        raise ValueError("start_time must be less than end_time")
    
    # Load video using decord
    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)  # Total number of frames in the video
    video_duration = total_frames / video_fps
    
    # Calculate frame_interval (how many frames to skip between samples)
    # fps=2 means sample 2 frames per second, so we sample every video_fps/fps frames
    frame_interval = round(video_fps / fps)  # e.g., video_fps=30, fps=2 -> frame_interval=15
    
    # Adjust start_time and end_time to frame boundaries using video_fps
    start_time = ceil_time_by_fps(start_time, video_fps, min_time=0, max_time=video_duration)
    end_time = ceil_time_by_fps(end_time, video_fps, min_time=0, max_time=video_duration)
    
    # Calculate frame indices in the original video (use video_fps, not frame_interval)
    start_frame = int(start_time * video_fps)
    # Calculate end_frame, but ensure it doesn't exceed total_frames
    # Note: end_frame is exclusive (not included), so we use end_time * video_fps without +1
    # to avoid exceeding the video bounds
    end_frame_calculated = int(end_time * video_fps)
    # Ensure end_frame doesn't exceed total_frames (since frame indices are 0-indexed, max is total_frames-1)
    # But since range() is exclusive, we can use total_frames as the upper bound
    end_frame = min(end_frame_calculated + 1, total_frames)
    
    # Sample frames: each frame_interval frames sample one frame
    # Filter out any indices that exceed total_frames (safety check)
    frame_idx = [i for i in range(start_frame, end_frame, frame_interval) if i < total_frames]
    
    if len(frame_idx) == 0:
        logger.warning(f"No frames found in range [{start_time}, {end_time}]")
        return [], [], []
    
    # Extract frames
    frames = []
    for idx in frame_idx:
        frame = vr[idx].asnumpy()  # Get single frame as numpy array (RGB format)
        
        if output_format == 'base64':
            # Convert frame to JPEG and encode as base64
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frames.append(base64.b64encode(buffer).decode("utf-8"))
        elif output_format == 'numpy':
            # Return numpy array directly (RGB format)
            frames.append(frame)
        else:
            raise ValueError(f"Unsupported output_format: {output_format}. Use 'base64' or 'numpy'")
    
    # Calculate frame_timestamps (starting from 1/(2*fps))
    # For fps=2: first frame at 0.5s, second at 1.0s, third at 1.5s, etc.
    frame_interval = 1.0 / fps  # Time interval between frames (0.5s for fps=2)
    frame_timestamps = []
    for i in range(len(frame_idx)):
        # First frame at 0.5s for fps=2, second at 1.0s, etc.
        timestamp = (i + 1) * frame_interval
        frame_timestamps.append(_seconds_to_timestamp_number(timestamp, start_time_number=0))
    
    # Calculate frame_time_ranges (starting from 0)
    # Each frame represents a 1/fps second interval
    frame_time_ranges = []
    for i in range(len(frame_idx)):
        time_range = {
            'start': _seconds_to_timestamp_number(i * frame_interval, start_time_number=0),
            'end': _seconds_to_timestamp_number(i * frame_interval + frame_interval, start_time_number=0)
        }
        frame_time_ranges.append(time_range)
    
    return frames, frame_timestamps, frame_time_ranges

def sample_frames_qaego4d(video_path, video_time_span, interval_seconds=1, output_format='base64'):
    """Sample frames from qaego4d dataset video within specified time span.

    This function samples frames from a video clip defined by video_time_span,
    dividing the time span into intervals and taking the middle frame of each interval.

    Args:
        video_path (str): Path to the video file
        video_time_span (tuple): (start_time, end_time) in seconds, defining the clip to sample from
        interval_seconds (float): Length of each time interval in seconds (default: 2.0)
        output_format (str): Output format, either 'base64' or 'numpy' (default: 'base64')

    Returns:
        tuple: (frames, timestamps, time_ranges) where:
            - frames: List of frames. If output_format='base64', returns list of base64-encoded JPEG strings.
                     If output_format='numpy', returns list of numpy arrays (RGB format).
            - timestamps: List of timestamp numbers corresponding to each frame (HHMMSSCC format).
            - time_ranges: List of dictionaries, each containing 'start' and 'end' representing the time range
                          (interval) that each frame represents (in HHMMSSCC format).

    Example:
        >>> frames, timestamps, time_ranges = sample_frames_qaego4d('video.mp4',
        ...                                                          video_time_span=(10.5, 25.3),
        ...                                                          interval_seconds=2.0)
        >>> # Samples frames from the clip [10.5s, 25.3s] at 2-second intervals
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")

    if not isinstance(video_time_span, (tuple, list)) or len(video_time_span) != 2:
        raise ValueError("video_time_span must be a tuple or list of (start_time, end_time)")

    clip_start, clip_end = video_time_span

    if clip_start > clip_end:
        raise ValueError("video_time_span start_time must be less than or equal to end_time")

    if clip_start < 0:
        raise ValueError("video_time_span start_time must be non-negative")

    video = VideoFileClip(video_path)
    frames = []
    timestamps = []
    time_ranges = []

    try:
        video_duration = video.duration

        # Validate time span against video duration
        if clip_start >= video_duration:
            raise ValueError(f"video_time_span start_time ({clip_start}s) exceeds video duration ({video_duration}s)")

        # Clamp clip_end to video duration
        clip_end = min(clip_end, video_duration)

        # clip_start == clip_end: 提取该时刻的单帧
        if clip_start == clip_end:
            absolute_time = min(clip_start, video_duration - 0.01)
            frame = video.get_frame(absolute_time)

            if output_format == 'base64':
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames.append(base64.b64encode(buffer).decode("utf-8"))
            elif output_format == 'numpy':
                frames.append(frame)
            else:
                raise ValueError(f"Unsupported output_format: {output_format}. Use 'base64' or 'numpy'")

            timestamps.append(_seconds_to_timestamp_number(0, start_time_number=0))
            time_ranges.append({
                'start': _seconds_to_timestamp_number(0, start_time_number=0),
                'end': _seconds_to_timestamp_number(0, start_time_number=0)
            })
            return frames, timestamps, time_ranges

        # Calculate clip duration
        clip_duration = clip_end - clip_start

        # Calculate number of intervals in the clip
        exact_intervals = clip_duration / interval_seconds
        num_intervals = int(np.ceil(exact_intervals))

        # If remainder is less than 1 second, round down
        remainder = clip_duration - int(exact_intervals) * interval_seconds
        if remainder > 0 and remainder < 1:
            num_intervals = int(exact_intervals)

        # clip_duration 不超过 1s 时 num_intervals 可能为 0，此时提取中间一帧
        if num_intervals == 0:
            mid_time = clip_start + clip_duration / 2.0
            mid_time = min(mid_time, video_duration - 0.01)
            frame = video.get_frame(mid_time)

            if output_format == 'base64':
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames.append(base64.b64encode(buffer).decode("utf-8"))
            elif output_format == 'numpy':
                frames.append(frame)
            else:
                raise ValueError(f"Unsupported output_format: {output_format}. Use 'base64' or 'numpy'")

            timestamps.append(_seconds_to_timestamp_number(clip_duration / 2.0, start_time_number=0))
            time_ranges.append({
                'start': _seconds_to_timestamp_number(0, start_time_number=0),
                'end': _seconds_to_timestamp_number(clip_duration, start_time_number=0)
            })
            return frames, timestamps, time_ranges

        sampling_times = []

        for i in range(num_intervals):
            # Calculate the middle point of each interval (relative to clip start)
            interval_start = i * interval_seconds
            interval_end = min((i + 1) * interval_seconds, clip_duration)
            middle_time = (interval_start + interval_end) / 2.0

            # Convert to absolute time in video
            absolute_time = clip_start + middle_time

            # Clamp to valid range
            absolute_time = min(absolute_time, video_duration - 0.01)
            if absolute_time >= clip_start:
                sampling_times.append((absolute_time, interval_start, interval_end))

        # Extract frames at specified intervals
        for absolute_time, interval_start, interval_end in sampling_times:
            # Clamp time to video duration
            absolute_time = min(absolute_time, video_duration - 0.01)
            frame = video.get_frame(absolute_time)

            if output_format == 'base64':
                # Convert frame to JPEG and encode as base64
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames.append(base64.b64encode(buffer).decode("utf-8"))
            elif output_format == 'numpy':
                # Return numpy array directly (RGB format)
                frames.append(frame)
            else:
                raise ValueError(f"Unsupported output_format: {output_format}. Use 'base64' or 'numpy'")

            # Calculate timestamp for the middle of the interval (relative to clip start)
            relative_time = absolute_time - clip_start
            timestamp_number = _seconds_to_timestamp_number(relative_time, start_time_number=0)
            timestamps.append(timestamp_number)

            # Calculate time range for this frame (relative to clip start)
            time_range = {
                'start': _seconds_to_timestamp_number(interval_start, start_time_number=0),
                'end': _seconds_to_timestamp_number(interval_end, start_time_number=0)
            }
            time_ranges.append(time_range)

        return frames, timestamps, time_ranges

    except Exception as e:
        logger.error(f"Error sampling frames from {video_path} with time_span {video_time_span}: {str(e)}")
        raise
    finally:
        video.close()
