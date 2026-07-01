"""
Base class for Proactive Service Benchmark on long-form video datasets.

Supports:
- HoloAssist: instructional videos with error recovery annotations
- CaptionCook4D: cooking videos with step-level error annotations

Unlike EgoLife (many pre-split ~30s segments), these datasets have single long
videos per recording. This class splits each video into fixed-length windows
(default 30s), uniformly samples frames per window, and runs model inference.
"""

import abc
import os
import re
import json
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu

from prompts import PROACTIVE_DETECTION_PROMPT, HOLOASSIST_DETECTION_PROMPT, CAPTIONCOOK4D_DETECTION_PROMPT

# Dataset -> prompt template mapping
DATASET_PROMPTS = {
    "holoassist": HOLOASSIST_DETECTION_PROMPT,
    "captioncook4d": CAPTIONCOOK4D_DETECTION_PROMPT,
}

logger = logging.getLogger(__name__)


# ============================================================================
# Time utilities
# ============================================================================

def seconds_to_timestr(seconds):
    """Convert seconds (float) to 'HH:MM:SS' string."""
    seconds = max(0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================================
# Frame sampling
# ============================================================================

def sample_frames_from_window(video_path, start_sec, end_sec, num_frames=8):
    """
    Uniformly sample frames from a time window [start_sec, end_sec] of a video.

    Returns:
        list of PIL Images, list of relative timestamps (seconds from window start),
        window duration (seconds)
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    video_duration = total_frames / fps

    # Clamp to video bounds
    start_sec = max(0, start_sec)
    end_sec = min(end_sec, video_duration)
    if end_sec <= start_sec:
        return [], [], 0.0

    duration = end_sec - start_sec
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames - 1)

    n_available = end_frame - start_frame + 1
    num_frames = min(num_frames, n_available)
    if num_frames <= 0:
        return [], [], duration

    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int).tolist()
    frames_data = vr.get_batch(indices).asnumpy()

    frames = []
    timestamps = []
    for i, idx in enumerate(indices):
        rel_ts = round((idx / fps) - start_sec, 2)
        timestamps.append(rel_ts)
        frames.append(Image.fromarray(frames_data[i]))

    return frames, timestamps, duration


# ============================================================================
# Data loading — HoloAssist
# ============================================================================

def load_holoassist_videos(
    video_dir,
    val_anno_path,
):
    """
    Load HoloAssist validation videos and their service annotations.

    Args:
        video_dir: Directory containing extracted videos
                   (e.g. /mnt/workspace/gst/HoloAssist/videos_extracted)
        val_anno_path: Path to val_service_annotations.json

    Returns:
        list of dicts: [{video_id, video_path, annotations}, ...]
    """
    with open(val_anno_path, 'r') as f:
        annotations = json.load(f)

    videos = []
    for video_id, anno in annotations.items():
        video_path = os.path.join(video_dir, f"{video_id}_Video_pitchshift.mp4")
        if not os.path.exists(video_path):
            logger.warning(f"Video not found, skipping: {video_path}")
            continue
        videos.append({
            "video_id": video_id,
            "video_path": video_path,
            "annotations": anno,
        })

    logger.info(f"Loaded {len(videos)} HoloAssist videos from {val_anno_path}")
    return videos


# ============================================================================
# Data loading — CaptionCook4D
# ============================================================================

def load_captioncook4d_videos(
    video_dir,
    data_splits_path,
    error_anno_path,
    splits=("val", "test"),
):
    """
    Load CaptionCook4D error videos from specified splits.

    Args:
        video_dir: Directory with video files (e.g. {id}_360p.mp4)
        data_splits_path: Path to recordings_data_split_combined.json
        error_anno_path: Path to error_annotations.json
        splits: Which splits to include (default: val + test)

    Returns:
        list of dicts: [{video_id, video_path, annotations}, ...]
    """
    with open(data_splits_path, 'r') as f:
        data_splits = json.load(f)
    split_ids = set()
    for s in splits:
        split_ids.update(data_splits.get(s, []))

    with open(error_anno_path, 'r') as f:
        all_annos = json.load(f)

    videos = []
    for record in all_annos:
        rid = record["recording_id"]
        if rid not in split_ids:
            continue
        if not record.get("is_error", False):
            continue
        video_path = os.path.join(video_dir, f"{rid}_360p.mp4")
        if not os.path.exists(video_path):
            logger.warning(f"Video not found, skipping: {video_path}")
            continue
        videos.append({
            "video_id": rid,
            "video_path": video_path,
            "annotations": record,
        })

    logger.info(f"Loaded {len(videos)} CaptionCook4D error videos from splits {splits}")
    return videos


# ============================================================================
# Base Benchmark Class
# ============================================================================

class VideoBench(abc.ABC):
    """
    Base class for evaluating proactive service detection on long-form video
    datasets (HoloAssist, CaptionCook4D).

    Splits each video into fixed-length windows, samples frames, runs model
    inference, and saves results.

    Subclasses must implement:
        - _init_model(): Initialize the model
        - inference(frames, prompt): Run model inference
    """

    def __init__(self, args):
        self.args = args
        self.num_frames = getattr(args, 'num_frames', 8)
        self.window_seconds = getattr(args, 'window_seconds', 30)
        self.result_dir = args.result_dir
        self.dataset = args.dataset  # "holoassist" or "captioncook4d"
        self._init_model()

    @abc.abstractmethod
    def _init_model(self):
        pass

    @abc.abstractmethod
    def inference(self, frames, prompt):
        pass

    def build_prompt(self, video_id, window_idx, time_window_str, duration_seconds, frame_timestamps):
        """Build proactive detection prompt for a video window.

        Uses dataset-specific prompt template:
        - holoassist:    Instant + Short-Term only
        - captioncook4d: Instant + Short-Term + Episodic
        - fallback:      full taxonomy (all 4 categories)
        """
        ts_str = ", ".join(f"{t:.1f}s" for t in frame_timestamps)
        prompt_template = DATASET_PROMPTS.get(self.dataset, PROACTIVE_DETECTION_PROMPT)
        return prompt_template.format(
            num_frames=self.num_frames,
            person_id=video_id,
            day_id=self.dataset,
            time_window=time_window_str,
            duration_seconds=round(duration_seconds, 1),
            frame_timestamps=ts_str,
        )

    def parse_response(self, response_text):
        """Parse JSON from model response."""
        if response_text is None:
            return None
        text = response_text.strip()
        # Remove markdown code fences
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

        # Convert relative time_span to absolute seconds within the video
        services = parsed.get("services", [])
        if isinstance(services, list):
            for svc in services:
                ts = svc.get("time_span")
                if isinstance(ts, list) and len(ts) == 2:
                    svc["time_span_relative_seconds"] = ts

        return parsed

    def _get_video_duration(self, video_path):
        """Get video duration in seconds."""
        vr = VideoReader(video_path, ctx=cpu(0))
        return len(vr) / vr.get_avg_fps()

    def eval(self):
        """Run evaluation on all videos."""
        # Load videos based on dataset
        if self.dataset == "holoassist":
            videos = load_holoassist_videos(
                video_dir=self.args.video_dir,
                val_anno_path=self.args.anno_path,
            )
        elif self.dataset == "captioncook4d":
            videos = load_captioncook4d_videos(
                video_dir=self.args.video_dir,
                data_splits_path=self.args.data_splits_path,
                error_anno_path=self.args.error_anno_path,
                splits=tuple(self.args.splits),
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # 可选:视频列表分片(多进程并行用)。shard_total=1 时不分片,行为不变。
        shard_total = getattr(self.args, 'shard_total', 1)
        shard_idx = getattr(self.args, 'shard_idx', 0)
        if shard_total and shard_total > 1:
            videos = videos[shard_idx::shard_total]
            logger.info(f"Shard {shard_idx}/{shard_total}: processing {len(videos)} videos")

        all_results = []
        model_name = getattr(self.args, 'model', 'unknown')
        output_dir = os.path.join(self.result_dir, model_name, self.dataset)
        os.makedirs(output_dir, exist_ok=True)

        for video_info in tqdm(videos, desc=f"Videos ({self.dataset})"):
            video_id = video_info["video_id"]
            video_path = video_info["video_path"]

            # Skip if result already exists
            result_path = os.path.join(output_dir, f"{video_id}.json")
            if self.args.skip_existing and os.path.exists(result_path):
                logger.info(f"Skipping {video_id} (result exists)")
                try:
                    with open(result_path, 'r') as f:
                        all_results.append(json.load(f))
                except Exception:
                    pass
                continue

            # Get video duration and split into windows
            try:
                duration = self._get_video_duration(video_path)
            except Exception as e:
                logger.error(f"Cannot read video {video_path}: {e}")
                continue

            num_windows = max(1, int(np.ceil(duration / self.window_seconds)))
            window_results = []

            for w_idx in range(num_windows):
                start_sec = w_idx * self.window_seconds
                end_sec = min((w_idx + 1) * self.window_seconds, duration)

                # Sample frames
                try:
                    frames, timestamps, w_dur = sample_frames_from_window(
                        video_path, start_sec, end_sec,
                        num_frames=self.num_frames,
                    )
                except Exception as e:
                    logger.error(f"  Frame sampling failed for {video_id} window {w_idx}: {e}")
                    continue

                if not frames:
                    continue

                # Build prompt
                time_window_str = (
                    f"{seconds_to_timestr(start_sec)}-{seconds_to_timestr(end_sec)}"
                )
                prompt = self.build_prompt(
                    video_id, w_idx, time_window_str, w_dur, timestamps
                )

                # Run inference
                try:
                    response = self.inference(frames, prompt)
                except Exception as e:
                    logger.error(f"  Inference failed for {video_id} window {w_idx}: {e}")
                    response = None

                parsed = self.parse_response(response)

                # Add absolute time offset to each detected service
                if parsed and isinstance(parsed.get("services"), list):
                    for svc in parsed["services"]:
                        rel = svc.get("time_span_relative_seconds")
                        if isinstance(rel, list) and len(rel) == 2:
                            svc["time_span_absolute_seconds"] = [
                                round(start_sec + rel[0], 2),
                                round(start_sec + rel[1], 2),
                            ]
                            svc["time_span_absolute"] = {
                                "start": seconds_to_timestr(start_sec + rel[0]),
                                "end": seconds_to_timestr(start_sec + rel[1]),
                            }

                window_results.append({
                    "window_idx": w_idx,
                    "start_sec": round(start_sec, 2),
                    "end_sec": round(end_sec, 2),
                    "time_window": time_window_str,
                    "num_frames_sampled": len(frames),
                    "response": parsed,
                    "raw_response": response,
                })

            # Assemble per-video result
            video_result = {
                "video_id": video_id,
                "video_path": video_path,
                "dataset": self.dataset,
                "duration_seconds": round(duration, 2),
                "window_seconds": self.window_seconds,
                "num_windows": num_windows,
                "annotations": video_info.get("annotations"),
                "windows": window_results,
            }

            # Save per-video result
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(video_result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved: {result_path}")

            all_results.append(video_result)

        # Save combined results
        combined_path = os.path.join(output_dir, "all_results.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved combined: {combined_path}")

        # Generate summary
        summary = self._generate_summary(all_results, output_dir)

        return all_results, summary

    def _generate_summary(self, all_results, output_dir):
        """Generate summary statistics."""
        # Build service_counts based on dataset-specific taxonomy
        base_counts = {
            "Instant": {"Safety": 0, "Tool Use": 0},
            "Short-Term": {"Error-Recovery": 0, "Next-Step Guidance": 0, "Resource Reminder": 0},
        }
        if self.dataset == "captioncook4d":
            base_counts["Episodic"] = {"Episodic Task Reminder": 0, "Episodic Memory Recall": 0}
        elif self.dataset not in ("holoassist",):
            # Full taxonomy for unknown datasets
            base_counts["Episodic"] = {"Episodic Task Reminder": 0, "Episodic Memory Recall": 0}
            base_counts["Long-Term"] = {
                "Long-Horizon Memory-Link": 0,
                "Routine Optimization": 0,
                "Personal Progress Feedback": 0,
                "Habit-Coaching": 0,
            }

        summary = {
            "dataset": self.dataset,
            "total_videos": len(all_results),
            "total_windows": 0,
            "windows_with_service": 0,
            "total_services": 0,
            "service_counts": base_counts,
            "per_video": {},
        }

        for vr in all_results:
            video_id = vr.get("video_id", "unknown")
            video_svc_count = 0

            for wr in vr.get("windows", []):
                summary["total_windows"] += 1
                resp = wr.get("response")
                if not isinstance(resp, dict):
                    continue
                services = resp.get("services", [])
                if isinstance(services, list) and len(services) > 0:
                    summary["windows_with_service"] += 1
                    summary["total_services"] += len(services)
                    video_svc_count += len(services)
                    for svc in services:
                        main_type = svc.get("service_main_type", "")
                        sub_type = svc.get("service_sub_type", "")
                        if main_type in summary["service_counts"]:
                            type_dict = summary["service_counts"][main_type]
                            if sub_type in type_dict:
                                type_dict[sub_type] += 1

            summary["per_video"][video_id] = video_svc_count

        # Log summary
        model_name = getattr(self.args, 'model', 'unknown')
        logger.info(f"\n{'='*60}")
        logger.info(f"PROACTIVE SERVICE DETECTION SUMMARY — {model_name} / {self.dataset}")
        logger.info(f"{'='*60}")
        logger.info(f"Total videos: {summary['total_videos']}")
        logger.info(f"Total windows: {summary['total_windows']}")
        logger.info(f"Windows with service: {summary['windows_with_service']}")
        logger.info(f"Total services detected: {summary['total_services']}")
        for main_type, subtypes in summary["service_counts"].items():
            total = sum(subtypes.values())
            if total > 0:
                logger.info(f"  {main_type}: {total}")
                for sub, count in subtypes.items():
                    if count > 0:
                        logger.info(f"    {sub}: {count}")

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved to: {summary_path}")

        return summary
