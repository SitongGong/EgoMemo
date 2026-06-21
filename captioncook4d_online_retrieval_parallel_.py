"""
Parallel video retrieval script for CaptionCook4D.
Splits error videos from val+test into N workers for multi-process execution.

Automatically detects embedding dimension (1024 vs 1536) per video from
vdb_chunks.json and selects the matching LLM config.

Usage examples:
  # Single GPU, 4 workers:
  python captioncook4d_online_retrieval_parallel.py --cuda 0 --num_workers 4

  # 2 GPUs, 4 workers (round-robin):
  python captioncook4d_online_retrieval_parallel.py --cuda 0,1 --num_workers 4

  # Run only specific workers:
  python captioncook4d_online_retrieval_parallel.py --cuda 0,1 --num_workers 4 --workers 0,2
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import argparse
import json
import logging
import time
import multiprocessing as mp
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument('--cuda', type=str, default='0')
    _pre_args, _ = _pre_parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.cuda

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_error_video_ids(
    data_splits_path: str,
    error_annotations_path: str,
    base_data_path: str,
    caption_model_name: str,
    response_name: str,
    results_base: str,
) -> List[str]:
    """Load val+test error video IDs, filter out completed and missing ones."""
    with open(data_splits_path) as f:
        data_splits = json.load(f)
    val_test_ids = set(data_splits["val"] + data_splits["test"])

    with open(error_annotations_path) as f:
        error_annotations = json.load(f)
    error_ids = set(
        item["recording_id"] for item in error_annotations
        if item.get("is_error") is True and item["recording_id"] in val_test_ids
    )

    filtered = []
    for recording_id in sorted(error_ids):
        video_path = os.path.join(base_data_path, recording_id + "_360p.mp4")
        if not os.path.exists(video_path):
            continue
        response_path = os.path.join(
            results_base, f"{recording_id}_{caption_model_name}", f"{response_name}.json"
        )
        if os.path.exists(response_path):
            continue
        filtered.append(recording_id)

    return filtered


# ---------------------------------------------------------------------------
# Worker function — runs in a subprocess
# ---------------------------------------------------------------------------

def worker_fn(
    worker_id: int,
    video_ids: List[str],
    cuda_device: str,
    args_dict: dict,
):
    """
    Each worker:
      1. Sets CUDA_VISIBLE_DEVICES to the assigned GPU.
      2. Imports heavy modules (model loading happens per-process).
      3. Processes its partition of videos sequentially.
    """
    # ---- Set GPU before any CUDA-related import ----
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    os.environ["OPENAI_API_KEY"] = args_dict["openai_api_key"]
    os.environ["GOOGLE_API_KEY"] = args_dict["google_api_key"]
    os.environ["DEEPSEEK_API_KEY"] = args_dict["deepseek_api_key"]
    os.environ["SILICONFLOW_API_KEY"] = args_dict["siliconflow_api_key"]

    logger.info(
        f"[Worker {worker_id}] PID={os.getpid()} | "
        f"{len(video_ids)} videos | GPU={cuda_device}"
    )

    # ---- Lazy imports (after CUDA env is set) ----
    from videorag.egograph_retrieval_optimize_ import VideoGraphSeparated
    from videorag._llm import (
        openai_embedding, bge_m3_embedding, gpt_4o_mini_complete, LLMConfig,
    )

    qwen_openai_config = LLMConfig(
        embedding_func_raw=bge_m3_embedding,
        embedding_model_name="BAAI/bge-m3",
        embedding_dim=1024,
        embedding_max_token_size=8192,
        embedding_batch_num=32,
        embedding_func_max_async=16,
        query_better_than_threshold=0.2,
        best_model_func_raw=gpt_4o_mini_complete,
        best_model_name="gpt-5-mini",
        best_model_max_token_size=32768,
        best_model_max_async=16,
        cheap_model_func_raw=gpt_4o_mini_complete,
        cheap_model_name="gpt-4o-mini",
        cheap_model_max_token_size=32768,
        cheap_model_max_async=16,
    )

    gpt5_2_config = LLMConfig(
        embedding_func_raw=openai_embedding,
        embedding_model_name="text-embedding-3-small",
        embedding_dim=1536,
        embedding_max_token_size=8192,
        embedding_batch_num=32,
        embedding_func_max_async=16,
        query_better_than_threshold=0.2,
        best_model_func_raw=gpt_4o_mini_complete,
        best_model_name="gpt-5-mini",
        best_model_max_token_size=32768,
        best_model_max_async=16,
        cheap_model_func_raw=gpt_4o_mini_complete,
        cheap_model_name="gpt-4o-mini",
        cheap_model_max_token_size=32768,
        cheap_model_max_async=16,
    )

    # ---- Reconstruct a simple namespace from args_dict ----
    class Args:
        pass
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    results_base = args_dict["results_base"]
    base_data_path = args_dict["base_data_path"]

    # ---- Process each video ----
    for idx, video_id in enumerate(video_ids):
        working_dir = os.path.join(results_base, f"{video_id}_{args.caption_model_name}")

        # 根据 vdb_chunks.json 的 embedding_dim 选择 LLM config
        vdb_chunks_path = os.path.join(working_dir, "vdb_chunks.json")
        llm_config = qwen_openai_config  # 默认 bge-m3 (dim=1024)
        if os.path.exists(vdb_chunks_path):
            with open(vdb_chunks_path, 'r') as f:
                vdb_meta = json.load(f)
            embed_dim = vdb_meta.get("embedding_dim", 1024)
            if embed_dim == 1536:
                llm_config = gpt5_2_config
                logger.info(f"[Worker {worker_id}][{video_id}] embedding_dim=1536, using text-embedding-3-small")
            else:
                logger.info(f"[Worker {worker_id}][{video_id}] embedding_dim={embed_dim}, using bge-m3")

        args.data_path = os.path.join(base_data_path, video_id + "_360p.mp4")

        videorag = VideoGraphSeparated(
            llm=llm_config,
            working_dir=working_dir,
            video_embedding_gpu=args.cuda,  # 映射后始终是 cuda:0
        )
        videorag.load_caption_model(model_name="qwen3_api")

        start_time = time.time()
        videorag.process_proactive_service(
            datasets_type=args.datasets_type,
            accumulated_captions=None,
            load_from_checkpoint=True,
            args=args,
        )
        elapsed = time.time() - start_time

        logger.info(
            f"[Worker {worker_id}] Done {idx+1}/{len(video_ids)} | "
            f"video={video_id} | time={elapsed:.1f}s"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel video retrieval for CaptionCook4D"
    )
    parser.add_argument(
        "--data_path", type=str,
        default=os.environ.get("CAPTIONCOOK4D_DATA_PATH", "./data/captioncook4d/hololens/sync/pv"),
        help="Base path to video data",
    )
    parser.add_argument(
        "--data_splits_path", type=str,
        default=os.environ.get("CAPTIONCOOK4D_DATA_SPLITS_PATH", "./data/captioncook4d/annotations/data_splits/recordings_data_split_combined.json"),
    )
    parser.add_argument(
        "--error_annotations_path", type=str,
        default=os.environ.get("CAPTIONCOOK4D_ERROR_ANNOTATIONS_PATH", "./data/captioncook4d/annotations/annotation_json/error_annotations.json"),
    )
    parser.add_argument(
        "--results_base", type=str,
        default=os.environ.get("RESULTS_ROOT", "./results") + "/captioncook4d_rebuttal",
        help="Base directory for results",
    )

    parser.add_argument("--caption_model_name", type=str, default="qwenvl_3_8b_instruct")
    parser.add_argument(
        "--cuda", type=str, default="1",
        help="Comma-separated GPU IDs. Workers are round-robin assigned.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of parallel workers to split videos across",
    )
    parser.add_argument(
        "--workers", type=str, default=None,
        help="Comma-separated worker indices to run. Default: all. "
             "E.g. '0,2' runs only workers 0 and 2",
    )

    parser.add_argument("--save_name", type=str, default="proactive_service_checkpoint_gpt_5")
    parser.add_argument("--response_name", type=str, default="proactive_response_gpt_5")

    parser.add_argument("--caption_retrieval", type=bool, default=True)
    parser.add_argument("--visual_retrieval", type=bool, default=True)
    parser.add_argument("--entity_retrieval", type=bool, default=True)
    parser.add_argument("--need_retrieval", type=bool, default=True)
    parser.add_argument("--filter_captions", type=bool, default=False)
    parser.add_argument("--multiscale", type=bool, default=True)
    parser.add_argument("--reconstruct_caption", type=bool, default=True)

    parser.add_argument("--datasets_type", type=str, default="captioncook4d")

    args = parser.parse_args()

    # ---- Load and filter video list ----
    all_videos = load_error_video_ids(
        data_splits_path=args.data_splits_path,
        error_annotations_path=args.error_annotations_path,
        base_data_path=args.data_path,
        caption_model_name=args.caption_model_name,
        response_name=args.response_name,
        results_base=args.results_base,
    )
    logger.info(f"Total videos to process: {len(all_videos)}")

    if not all_videos:
        logger.info("No videos to process. All done or no matching videos found.")
        return

    # ---- Parse GPU list ----
    gpu_ids = [g.strip() for g in args.cuda.split(",")]
    logger.info(f"Available GPUs: {gpu_ids}")

    # ---- Split videos across workers ----
    num_workers = min(args.num_workers, len(all_videos))
    video_partitions: List[List[str]] = [[] for _ in range(num_workers)]
    for i, video in enumerate(all_videos):
        video_partitions[i % num_workers].append(video)

    # ---- Determine which workers to launch ----
    if args.workers is not None:
        worker_indices = [int(w.strip()) for w in args.workers.split(",")]
    else:
        worker_indices = list(range(num_workers))

    logger.info(f"Will launch {len(worker_indices)} workers out of {num_workers}")
    for wi in worker_indices:
        assigned_gpu = gpu_ids[wi % len(gpu_ids)]
        logger.info(
            f"  Worker {wi}: {len(video_partitions[wi])} videos, GPU={assigned_gpu}"
        )

    # ---- Build a picklable args dict ----
    args_dict = vars(args).copy()
    args_dict["base_data_path"] = args.data_path
    args_dict["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
    args_dict["google_api_key"] = os.environ.get("GEMINI_API_KEY", "")
    args_dict["deepseek_api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
    args_dict["siliconflow_api_key"] = os.environ.get("SILICONFLOW_API_KEY", "")

    # ---- Launch processes ----
    mp.set_start_method("spawn", force=True)

    processes: List[mp.Process] = []
    for wi in worker_indices:
        assigned_gpu = gpu_ids[wi % len(gpu_ids)]
        p = mp.Process(
            target=worker_fn,
            args=(wi, video_partitions[wi], assigned_gpu, args_dict),
            name=f"worker-{wi}",
        )
        processes.append(p)

    overall_start = time.time()

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
    else:
        logger.info("All workers completed successfully.")


if __name__ == "__main__":
    main()
