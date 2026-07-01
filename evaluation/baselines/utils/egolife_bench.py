"""
Base class for EgoLife Proactive Service Benchmark.

Handles:
- Loading EgoLife video segments and annotations
- Uniformly sampling 8 frames per segment
- Iterating over all segments for a person/day
- Collecting and saving results
"""

import abc
import os
import re
import json
import glob
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu

from prompts import PROACTIVE_DETECTION_PROMPT

logger = logging.getLogger(__name__)

# ============================================================================
# Time utilities
# ============================================================================

def timestamp_number_to_str(ts_num):
    """Convert HHMMSSCC integer to 'HH:MM:SS' string."""
    s = str(ts_num).zfill(8)
    return f"{s[0:2]}:{s[2:4]}:{s[4:6]}"


def _relative_seconds_to_absolute(relative_sec, start_time_number):
    """
    Convert relative seconds (from segment start) to absolute HHMMSSCC number.

    Args:
        relative_sec: float, seconds from segment start
        start_time_number: int, HHMMSSCC of segment start time

    Returns:
        int: absolute HHMMSSCC number
    """
    s = str(int(start_time_number)).zfill(8)
    base_sec = int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) + int(s[6:8]) / 100.0
    abs_sec = base_sec + float(relative_sec)

    h = int(abs_sec // 3600)
    m = int((abs_sec % 3600) // 60)
    sec = int(abs_sec % 60)
    cs = int(round((abs_sec - int(abs_sec)) * 100))
    if cs >= 100:
        sec += 1
        cs = 0
    return h * 1000000 + m * 10000 + sec * 100 + cs


def filename_to_time_number(filename):
    """Extract time number from filename like 'DAY5_A1_JAKE_12460000.mp4'."""
    base = os.path.splitext(os.path.basename(filename))[0]
    # Try patterns: DAY#_PERSON_TIME or PERSON_DAY#_TIME
    match = re.search(r'(\d{8,})$', base)
    if match:
        return int(match.group(1))
    return None


def get_video_time_window(video_path, annotation_path=None):
    """
    Get the time window string for a video segment.
    Returns: (day_str, time_window_str) e.g. ('DAY1', 'DAY1 11:00:00-11:00:30')
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]

    # Extract day number
    day_match = re.search(r'DAY(\d+)', basename, re.IGNORECASE)
    day_num = int(day_match.group(1)) if day_match else 1
    day_str = f"DAY{day_num}"

    # Extract start time from filename
    time_match = re.search(r'(\d{8,})$', basename)
    if time_match:
        start_ts = int(time_match.group(1))
        start_str = timestamp_number_to_str(start_ts)

        # Estimate end time: try annotation, or fallback to video duration
        if annotation_path and os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    anno = json.load(f)
                captions = anno.get('dense_caption', [])
                if captions:
                    last = captions[-1]
                    end_str = last.get('end_time', start_str)
                    return day_str, f"{day_str} {start_str}-{end_str}"
            except Exception:
                pass

        # Fallback: use video duration
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            duration = len(vr) / vr.get_avg_fps()
            end_ts = start_ts + int(duration) * 100  # rough
            end_str = timestamp_number_to_str(end_ts)
            return day_str, f"{day_str} {start_str}-{end_str}"
        except Exception:
            return day_str, f"{day_str} {start_str}-{start_str}"

    return "DAY1", "DAY1 00:00:00-00:00:30"


# ============================================================================
# Frame sampling
# ============================================================================

def sample_uniform_frames(video_path, num_frames=8, output_format='pil'):
    """
    Uniformly sample frames from a video file.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample (default: 8)
        output_format: 'pil' for PIL Images, 'numpy' for numpy arrays

    Returns:
        list of frames, list of frame timestamps (seconds), duration (seconds)
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frames / fps

    # Ensure we don't sample more frames than available
    num_frames = min(num_frames, total_frames)
    if num_frames <= 0:
        return [], [], 0.0

    # Uniform sampling indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    frames_data = vr.get_batch(indices).asnumpy()

    frames = []
    timestamps = []
    for i, idx in enumerate(indices):
        ts = round(idx / fps, 2)
        timestamps.append(ts)
        if output_format == 'pil':
            frames.append(Image.fromarray(frames_data[i]))
        else:
            frames.append(frames_data[i])

    return frames, timestamps, duration


# ============================================================================
# Data discovery
# ============================================================================

def discover_video_segments(data_dir, person, day=None):
    """
    Discover all video segments for a person (and optionally a specific day).

    Args:
        data_dir: Root directory of EgoLife data (e.g. /mnt/tokyo-ai/gst/GST_EgoLife/egolife)
        person: Person ID (e.g. 'A1_JAKE')
        day: Optional day filter (e.g. 'DAY1')

    Returns:
        List of dicts with keys: video_path, annotation_path, person, day, segment_id
    """
    person_dir = os.path.join(data_dir, person)
    anno_base = os.path.join(data_dir, "annotation_segments", person)

    segments = []

    if day:
        day_dirs = [os.path.join(person_dir, day)]
    else:
        day_dirs = sorted(glob.glob(os.path.join(person_dir, "DAY*")))

    for day_dir in day_dirs:
        if not os.path.isdir(day_dir):
            continue
        day_name = os.path.basename(day_dir)

        video_files = sorted(
            glob.glob(os.path.join(day_dir, "*.mp4"))
        )

        for vf in video_files:
            basename = os.path.splitext(os.path.basename(vf))[0]
            # Try to find matching annotation
            anno_path = os.path.join(anno_base, day_name, f"{basename}.json")

            segments.append({
                "video_path": vf,
                "annotation_path": anno_path if os.path.exists(anno_path) else None,
                "person": person,
                "day": day_name,
                "segment_id": basename,
            })

    return segments


# ============================================================================
# Base Benchmark Class
# ============================================================================

class EgoLifeProactiveBench(abc.ABC):
    """
    Base class for evaluating a model's proactive service detection
    on EgoLife video segments.

    Subclasses must implement:
        - _init_model(): Initialize the model
        - inference(frames, prompt): Run model inference given frames + text prompt
    """

    def __init__(self, args):
        self.args = args
        self.num_frames = getattr(args, 'num_frames', 8)
        self.data_dir = args.data_dir
        self.result_dir = args.result_dir
        self.persons = args.persons
        self.days = args.days
        self._init_model()

    @abc.abstractmethod
    def _init_model(self):
        """Initialize the model. Called during __init__."""
        pass

    @abc.abstractmethod
    def inference(self, frames, prompt):
        """
        Run inference on a set of frames with a text prompt.

        Args:
            frames: List of PIL Image objects (uniformly sampled from video)
            prompt: Text prompt string

        Returns:
            str: Model's text response (should be JSON)
        """
        pass

    def build_prompt(self, segment_info, time_window, duration_seconds, frame_timestamps):
        """Build the proactive detection prompt for a segment."""
        ts_str = ", ".join(f"{t:.1f}s" for t in frame_timestamps)
        return PROACTIVE_DETECTION_PROMPT.format(
            num_frames=self.num_frames,
            person_id=segment_info["person"],
            day_id=segment_info["day"],
            time_window=time_window,
            duration_seconds=round(duration_seconds, 1),
            frame_timestamps=ts_str,
        )

    def parse_response(self, response_text, start_time_number=None):
        """
        Parse JSON from model response and convert relative time_span
        (seconds from segment start) to absolute time (HHMMSSCC / HH:MM:SS).
        """
        if response_text is None:
            return None
        # Try to extract JSON from response
        text = response_text.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split('\n')
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = '\n'.join(lines)
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        if parsed is None:
            logger.warning(f"Failed to parse JSON response: {text[:200]}")
            return {"raw_response": response_text, "services": []}

        # Convert relative time_span to absolute time for each service
        services = parsed.get("services", [])
        if isinstance(services, list) and start_time_number is not None:
            for svc in services:
                ts = svc.get("time_span")
                if isinstance(ts, list) and len(ts) == 2:
                    abs_start = _relative_seconds_to_absolute(ts[0], start_time_number)
                    abs_end = _relative_seconds_to_absolute(ts[1], start_time_number)
                    svc["time_span_absolute"] = {
                        "start": timestamp_number_to_str(abs_start),
                        "end": timestamp_number_to_str(abs_end),
                        "start_number": abs_start,
                        "end_number": abs_end,
                    }
                    svc["time_span_relative_seconds"] = ts

        return parsed

    def eval(self):
        """
        Run evaluation across all specified persons and days.
        Iterates over video segments, samples frames, runs inference,
        and saves results.
        """
        all_results = {}

        for person in self.persons:
            logger.info(f"Processing person: {person}")
            person_results = {}

            for day in self.days:
                logger.info(f"  Processing {day}")
                segments = discover_video_segments(self.data_dir, person, day)

                if not segments:
                    logger.warning(f"  No video segments found for {person}/{day}")
                    continue

                day_results = []

                for seg in tqdm(segments, desc=f"{person}/{day}"):
                    video_path = seg["video_path"]

                    # Get time window
                    day_str, time_window = get_video_time_window(
                        video_path, seg["annotation_path"]
                    )

                    # Sample frames
                    try:
                        frames, timestamps, duration = sample_uniform_frames(
                            video_path,
                            num_frames=self.num_frames,
                            output_format='pil'
                        )
                    except Exception as e:
                        logger.error(f"  Failed to sample frames from {video_path}: {e}")
                        continue

                    if not frames:
                        logger.warning(f"  No frames sampled from {video_path}")
                        continue

                    # Build prompt and run inference
                    prompt = self.build_prompt(seg, time_window, duration, timestamps)

                    try:
                        response = self.inference(frames, prompt)
                    except Exception as e:
                        logger.error(f"  Inference failed for {seg['segment_id']}: {e}")
                        response = None

                    # Parse response — convert relative seconds to absolute time
                    start_time_number = filename_to_time_number(seg["segment_id"])
                    parsed = self.parse_response(response, start_time_number)

                    result = {
                        "segment_id": seg["segment_id"],
                        "person": person,
                        "day": day,
                        "time_window": time_window,
                        "video_path": video_path,
                        "num_frames_sampled": len(frames),
                        "response": parsed,
                        "raw_response": response,
                    }
                    day_results.append(result)

                person_results[day] = day_results
                logger.info(f"  {day}: processed {len(day_results)} segments")

            all_results[person] = person_results

        # Save results
        self._save_results(all_results)

        # Generate summary
        summary = self._generate_summary(all_results)

        return all_results, summary

    def _save_results(self, all_results):
        """Save per-person, per-day results to JSON files."""
        model_name = getattr(self.args, 'model', 'unknown')
        output_dir = os.path.join(self.result_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)

        for person, person_data in all_results.items():
            for day, day_results in person_data.items():
                out_path = os.path.join(output_dir, f"{person}_{day}_proactive.json")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(day_results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved: {out_path}")

        # Also save a combined file
        combined_path = os.path.join(output_dir, "all_results.json")
        # Flatten for combined
        flat = []
        for person, person_data in all_results.items():
            for day, day_results in person_data.items():
                flat.extend(day_results)
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(flat, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved combined: {combined_path}")

    def _generate_summary(self, all_results):
        """
        Generate a summary of detected proactive services.

        Returns a dict with counts per service type.
        """
        summary = {
            "total_segments": 0,
            "segments_with_service": 0,
            "total_services": 0,
            "service_counts": {
                "Instant": {"Safety": 0, "Tool Use": 0},
                "Short-Term": {"Error-Recovery": 0, "Next-Step Guidance": 0, "Resource Reminder": 0},
                "Episodic": {"Episodic Task Reminder": 0, "Episodic Memory Recall": 0},
                "Long-Term": {
                    "Long-Horizon Memory-Link": 0,
                    "Routine Optimization": 0,
                    "Personal Progress Feedback": 0,
                    "Habit-Coaching": 0,
                },
            },
            "per_person": {},
            "per_day": {},
        }

        for person, person_data in all_results.items():
            person_count = 0
            for day, day_results in person_data.items():
                day_key = f"{person}/{day}"
                day_service_count = 0

                for result in day_results:
                    summary["total_segments"] += 1
                    resp = result.get("response")
                    if not isinstance(resp, dict):
                        continue

                    services = resp.get("services", [])
                    if isinstance(services, list) and len(services) > 0:
                        summary["segments_with_service"] += 1
                        summary["total_services"] += len(services)
                        day_service_count += len(services)
                        person_count += len(services)

                        for svc in services:
                            main_type = svc.get("service_main_type", "")
                            sub_type = svc.get("service_sub_type", "")
                            if main_type in summary["service_counts"]:
                                type_dict = summary["service_counts"][main_type]
                                if sub_type in type_dict:
                                    type_dict[sub_type] += 1

                summary["per_day"][day_key] = day_service_count

            summary["per_person"][person] = person_count

        # Print summary
        model_name = getattr(self.args, 'model', 'unknown')
        logger.info(f"\n{'='*60}")
        logger.info(f"PROACTIVE SERVICE DETECTION SUMMARY — {model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Total segments processed: {summary['total_segments']}")
        logger.info(f"Segments with service triggered: {summary['segments_with_service']}")
        logger.info(f"Total services detected: {summary['total_services']}")
        logger.info(f"\nService type breakdown:")
        for main_type, subtypes in summary["service_counts"].items():
            total = sum(subtypes.values())
            if total > 0:
                logger.info(f"  {main_type}: {total}")
                for sub, count in subtypes.items():
                    if count > 0:
                        logger.info(f"    {sub}: {count}")

        logger.info(f"\nPer-person totals:")
        for person, count in summary["per_person"].items():
            logger.info(f"  {person}: {count} services")

        # Save summary
        model_name = getattr(self.args, 'model', 'unknown')
        output_dir = os.path.join(self.result_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"\nSummary saved to: {summary_path}")

        return summary
