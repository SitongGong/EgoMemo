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
import gc
import multiprocessing as mp
import torch
from typing import List, Dict

from videorag.egograph_retrieval_optimize_ import VideoGraphSeparated
from videorag._llm import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][Process-%(process)d] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# _qwen_openai_config_ = LLMConfig(
#     embedding_func_raw=bge_m3_embedding,
#     embedding_model_name="BAAI/bge-m3",
#     embedding_dim=1024,
#     embedding_max_token_size=8192,
#     embedding_batch_num=32,
#     embedding_func_max_async=16,
#     query_better_than_threshold=0.2,

#     best_model_func_raw=qwen_complete,
#     best_model_name="qwen3-30b-a3b-instruct-2507",
#     best_model_max_token_size=32768,
#     best_model_max_async=16,

#     cheap_model_func_raw=qwen_complete,
#     cheap_model_name="qwen3-30b-a3b-instruct-2507",
#     cheap_model_max_token_size=32768,
#     cheap_model_max_async=16,
# )

gpt5_2_llm_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 12,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM (we utilize gpt-4o-mini for all experiments)
    best_model_func_raw = gpt_4o_mini_complete,
    best_model_name = "gpt-4o-mini",
    best_model_max_token_size = 32768,
    best_model_max_async = 16,

    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16,
)


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def process_video_batch(recording_ids: List[str], args, gpu_id: int, process_id: int):
    """
    在指定GPU上顺序处理一批 CaptionCook4D 视频。

    Args:
        recording_ids: 要处理的 recording_id 列表
        args: 命令行参数
        gpu_id: GPU设备ID
        process_id: 进程标识符
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.environ["GOOGLE_API_KEY"] = args.google_api_key
    os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key
    os.environ["SILICONFLOW_API_KEY"] = args.siliconflow_api_key

    logger.info(f"Process {process_id} starting on GPU {gpu_id} with {len(recording_ids)} videos")

    processing_times = []

    for idx, recording_id in enumerate(recording_ids):
        video_start_time = time.time()

        working_dir = f"{os.environ.get('RESULTS_ROOT', './results')}/captioncook4d_rebuttal/{recording_id}_{args.caption_model_name}"

        # 如果结果目录已存在，跳过
        if os.path.exists(working_dir):
            logger.info(f"Process {process_id}: [{idx+1}/{len(recording_ids)}] 跳过已处理视频: {recording_id}")
            continue

        video_path = os.path.join(args.data_path, recording_id + "_360p.mp4")
        if not os.path.exists(video_path):
            logger.warning(f"Process {process_id}: [{idx+1}/{len(recording_ids)}] 视频文件不存在，跳过: {video_path}")
            continue

        try:
            videorag = VideoGraphSeparated(
                # llm=_qwen_openai_config_,
                llm=gpt5_2_llm_config,
                working_dir=working_dir,
                video_embedding_gpu=int(args.cuda),
            )
            videorag.load_caption_model(model_name=args.caption_model_name)

            videorag.streaming_graph_construction(
                data_path=video_path,
                anno_path=None,
                day="DAY1",
                interval_seconds=2,
                window_seconds=10,
                gap_threshold_seconds=60,
                window_minutes=1,
                window_hours=1/6,
                max_new_tokens=1024,
                datasets_type=args.datasets_type,
            )

            elapsed = time.time() - video_start_time
            processing_times.append(elapsed)
            avg = sum(processing_times) / len(processing_times)
            remaining = (len(recording_ids) - idx - 1) * avg
            logger.info(
                f"Process {process_id}: [{idx+1}/{len(recording_ids)}] "
                f"video={recording_id} 耗时={elapsed:.1f}s "
                f"平均={avg:.1f}s/条 预计剩余={remaining/60:.1f}min"
            )
            clear_gpu_memory()

        except Exception as e:
            elapsed = time.time() - video_start_time
            logger.error(
                f"Process {process_id}: [{idx+1}/{len(recording_ids)}] "
                f"video={recording_id} 失败: {e} (耗时{elapsed:.1f}s)"
            )
            continue

    if processing_times:
        avg = sum(processing_times) / len(processing_times)
        logger.info(f"Process {process_id} 完成: 处理={len(processing_times)}个, 平均={avg:.1f}s, 最快={min(processing_times):.1f}s, 最慢={max(processing_times):.1f}s")

    return processing_times


def split_list(data_list, n):
    k, m = divmod(len(data_list), n)
    return [data_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description="CaptionCook4D parallel processing (multi-process)")
    parser.add_argument("--data_path", type=str,
                        default=os.environ.get("CAPTIONCOOK4D_DATA_PATH", "./data/captioncook4d/hololens/sync/pv"),
                        help="视频文件所在目录（每个视频为 {recording_id}_360p.mp4）")
    parser.add_argument("--caption_model_name", type=str, default="qwenvl_3_8b_instruct")
    parser.add_argument("--cuda", type=str, default="0",
                        help="CUDA device ID，传给 video_embedding_gpu 和 CUDA_VISIBLE_DEVICES")
    parser.add_argument("--datasets_type", type=str, default="captioncook4d")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="并行进程数 (default: 4)")
    parser.add_argument("--openai_api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--google_api_key", type=str,
                        default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--deepseek_api_key", type=str,
                        default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--siliconflow_api_key", type=str,
                        default=os.environ.get("SILICONFLOW_API_KEY", ""))

    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.environ["GOOGLE_API_KEY"] = args.google_api_key
    os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key
    os.environ["SILICONFLOW_API_KEY"] = args.siliconflow_api_key

    # 提取 val + test 中 is_error=True 的视频
    with open(os.environ.get("CAPTIONCOOK4D_DATA_SPLITS_PATH", "./data/captioncook4d/annotations/data_splits/recordings_data_split_combined.json")) as f:
        data_splits = json.load(f)
    val_test_ids = set(data_splits["val"] + data_splits["test"])

    with open(os.environ.get("CAPTIONCOOK4D_ERROR_ANNOTATIONS_PATH", "./data/captioncook4d/annotations/annotation_json/error_annotations.json")) as f:
        error_annotations = json.load(f)
    error_ids = sorted(
        item["recording_id"] for item in error_annotations
        if item.get("is_error") is True and item["recording_id"] in val_test_ids
    )
    logger.info(f"Val+Test 中 is_error=True 的视频数: {len(error_ids)}, 示例: {error_ids[:5]}")

    # 过滤已处理（结果目录已存在）
    unprocessed = [
        rid for rid in error_ids
        if not os.path.exists(
            f"{os.environ.get('RESULTS_ROOT', './results')}/captioncook4d_rebuttal/{rid}_{args.caption_model_name}"
        )
    ]
    skipped = len(error_ids) - len(unprocessed)
    if skipped:
        logger.info(f"跳过已处理的 {skipped} 个视频，剩余 {len(unprocessed)} 个待处理")
    if not unprocessed:
        logger.info("所有视频已处理完毕，无需重新运行")
        return

    # 分批并行处理
    num_processes = min(args.num_processes, len(unprocessed))
    batches = split_list(unprocessed, num_processes)
    logger.info(f"使用 {num_processes} 个进程，GPU={args.cuda}")
    for i, batch in enumerate(batches):
        logger.info(f"  Batch {i}: {len(batch)} 个视频")

    overall_start = time.time()
    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=num_processes) as pool:
        tasks = [
            pool.apply_async(process_video_batch, args=(batch, args, int(args.cuda), i))
            for i, batch in enumerate(batches)
        ]
        all_times = []
        for i, task in enumerate(tasks):
            try:
                times = task.get()
                all_times.extend(times)
                logger.info(f"Process {i} 完成")
            except Exception as e:
                logger.error(f"Process {i} 失败: {e}", exc_info=True)

    total = time.time() - overall_start
    logger.info(f"\n{'='*80}")
    logger.info(f"所有视频处理完成! 总耗时={total:.1f}s ({total/60:.1f}min)")
    if all_times:
        logger.info(f"成功处理={len(all_times)}个, 平均={sum(all_times)/len(all_times):.1f}s, 最快={min(all_times):.1f}s, 最慢={max(all_times):.1f}s")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
