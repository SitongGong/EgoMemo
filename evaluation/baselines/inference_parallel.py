"""
EgoLife Proactive Service Benchmark — Parallel Inference (5 processes for DAY1-DAY5).

Spawns one process per day so that DAY1-DAY5 run concurrently.
Each process independently initialises its own model client, processes all
video segments for that day, and saves per-day results.  After all workers
finish, the main process merges results and generates a combined summary.

Usage:
    # Gemini (default)
    python inference_parallel.py \
        --model Gemini \
        --gemini_project <api_key> \
        --persons A1_JAKE

    # QWen3VL
    python inference_parallel.py \
        --model QWen3VL \
        --persons A1_JAKE

    # Specify days explicitly (default: DAY1-DAY5)
    python inference_parallel.py --model Gemini --days DAY1 DAY2 DAY3

    # Resume from where we left off (skips existing result files)
    python inference_parallel.py --model Gemini --skip_existing
"""

import argparse
import json
import logging
import os
import sys
import time
import multiprocessing as mp
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ALL_PERSONS = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]
ALL_DAYS = ["DAY1", "DAY2", "DAY3", "DAY4", "DAY5", "DAY6", "DAY7"]


# ---------------------------------------------------------------------------
# Argument parsing — identical to inference.py
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="EgoLife Proactive Service Benchmark — Parallel Inference"
    )

    # Data
    parser.add_argument(
        "--data_dir", type=str,
        default="./data/egolife",
        help="Root directory of EgoLife data"
    )
    parser.add_argument(
        "--result_dir", type=str,
        default="./outputs/GPT_results",
        help="Root directory for saving results"
    )

    # Scope
    parser.add_argument(
        "--persons", type=str, nargs='+',
        default=["A1_JAKE", "A3_TASHA", "A5_KATRINA"],
        help=f"Person IDs to evaluate. Available: {ALL_PERSONS}"
    )
    parser.add_argument(
        "--days", type=str, nargs='+',
        default=["DAY1", "DAY2", "DAY3", "DAY4", "DAY5"],
        help=f"Days to evaluate. Available: {ALL_DAYS}"
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="GPT",
        choices=["Gemini", "GPT", "QWen2VL_7B", "QWen2VL_72B", "QWen3VL"],
        help="Model to use for evaluation"
    )

    # Frame sampling
    parser.add_argument(
        "--num_frames", type=int, default=8,
        help="Number of frames to uniformly sample per video segment (default: 8)"
    )

    # Model-specific arguments
    parser.add_argument(
        "--gemini_project", type=str, default="AIzaSyBV5hl3rjJWxAR_zIn7QjYW-CY-rdCgV08",
        help="Google Cloud project ID for Gemini"
    )
    parser.add_argument(
        "--gemini_model", type=str, default="gemini-2.5-flash",
        help="Gemini model name"
    )
    parser.add_argument(
        "--gpt_api", type=str, default=os.environ.get("OPENAI_API_KEY",""),
        help="OpenAI API key for GPT"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to local model weights (for QWen2VL)"
    )

    # QWen3-VL API arguments
    parser.add_argument(
        "--qwen_api_key", type=str, default=os.environ.get("DASHSCOPE_API_KEY",""),
        help="DashScope API key for QWen3-VL"
    )
    parser.add_argument(
        "--qwen_model", type=str, default="qwen2.5-72b-instruct", # "qwen3-vl-plus",
        help="QWen3-VL model name"
    )
    parser.add_argument(
        "--qwen_base_url", type=str,
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        help="QWen3-VL API base URL"
    )

    # Parallel-specific
    parser.add_argument(
        "--skip_existing", type=bool, default=True,
        help="Skip days whose result files already exist"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Worker function — runs in a subprocess, one per day
# ---------------------------------------------------------------------------

def worker_fn(day: str, args_dict: dict):
    """
    Process all video segments for a single day.
    Each worker creates its own model instance and saves results independently.
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - [{day}] %(levelname)s - %(message)s'
    )
    wlogger = logging.getLogger(f"worker.{day}")

    # Reconstruct args namespace
    class Args:
        pass
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    # Override days to this single day
    args.days = [day]

    model_name = args.model
    wlogger.info(f"Starting {model_name} inference for {args.persons} / {day}")

    # Check if results already exist (skip_existing)
    if args_dict.get("skip_existing", False):
        output_dir = os.path.join(args.result_dir, model_name)
        all_exist = True
        for person in args.persons:
            out_path = os.path.join(output_dir, f"{person}_{day}_proactive.json")
            if not os.path.exists(out_path):
                all_exist = False
                break
        if all_exist:
            wlogger.info(f"Results already exist for {day}, skipping")
            return

    # Import and instantiate the model
    if model_name == "Gemini":
        from models.Gemini import EvalGemini
        model = EvalGemini(args)
    elif model_name == "GPT":
        from models.GPT import EvalGPT
        model = EvalGPT(args)
    elif model_name in ("QWen2VL_7B", "QWen2VL_72B"):
        from models.QWen2VL import EvalQWen2VL
        model = EvalQWen2VL(args)
    elif model_name == "QWen3VL":
        from models.QWen3VL import EvalQWen3VL
        model = EvalQWen3VL(args)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Run evaluation (single day)
    start_time = time.time()
    results, summary = model.eval()
    elapsed = time.time() - start_time

    wlogger.info(f"Finished {day} in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Post-merge: combine per-day files into all_results.json + summary
# ---------------------------------------------------------------------------

def merge_results(args):
    """Merge per-day result files into combined output + summary."""
    model_name = args.model
    output_dir = os.path.join(args.result_dir, model_name)

    flat = []
    per_person_day = {}

    for person in args.persons:
        per_person_day[person] = {}
        for day in args.days:
            out_path = os.path.join(output_dir, f"{person}_{day}_proactive.json")
            if os.path.exists(out_path):
                with open(out_path, 'r', encoding='utf-8') as f:
                    day_results = json.load(f)
                per_person_day[person][day] = day_results
                flat.extend(day_results)
            else:
                logger.warning(f"Missing result file: {out_path}")

    # Save combined
    combined_path = os.path.join(output_dir, "all_results.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved combined results: {combined_path} ({len(flat)} segments)")

    # Generate summary
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

    for person, person_data in per_person_day.items():
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

    logger.info(f"\nPer-day totals:")
    for day_key in sorted(summary["per_day"].keys()):
        logger.info(f"  {day_key}: {summary['per_day'][day_key]} services")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSummary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logger.info(f"Model: {args.model}")
    logger.info(f"Persons: {args.persons}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Frames per segment: {args.num_frames}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Result dir: {args.result_dir}")
    logger.info(f"Parallel workers: {len(args.days)} (one per day)")

    # Validate model choice early
    if args.model == "GPT":
        assert args.gpt_api is not None, "--gpt_api is required for GPT model"
    elif args.model in ("QWen2VL_7B", "QWen2VL_72B"):
        assert args.model_path is not None and os.path.exists(args.model_path), \
            f"--model_path must point to a valid model directory, got: {args.model_path}"

    # Build picklable args dict
    args_dict = vars(args).copy()

    # Launch one process per day
    mp.set_start_method("spawn", force=True)

    processes = []
    overall_start = time.time()

    for day in args.days:
        p = mp.Process(
            target=worker_fn,
            args=(day, args_dict),
            name=f"worker-{day}",
        )
        processes.append(p)

    for p in processes:
        p.start()
        logger.info(f"Started {p.name} (PID={p.pid})")

    for p in processes:
        p.join()
        logger.info(f"Finished {p.name} (exit code={p.exitcode})")

    overall_elapsed = time.time() - overall_start
    logger.info(f"All workers finished. Total wall time: {overall_elapsed:.1f}s")

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        logger.error(f"{len(failed)} worker(s) failed:")
        for p in failed:
            logger.error(f"  {p.name} exit code={p.exitcode}")

    # Merge results from all days
    logger.info("\nMerging results from all days...")
    merge_results(args)

    logger.info(f"\nDone! Results saved to: {os.path.join(args.result_dir, args.model)}")


if __name__ == "__main__":
    main()
