"""
批处理版本：将视频数据集切分为多份，使用多个GPU并行处理
基于 egoschema_graph_ablation.py 的处理逻辑
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
from datetime import datetime
from pathlib import Path

# 导入原始处理函数所需的所有模块
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from egoschema_graph_ablation import streaming_processing, gpt5_2_llm_config

# 配置日志（每个进程会有自己的日志文件）
def setup_logger(process_id, log_dir):
    """为每个进程设置独立的日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"egoschema_ablation_batch_process_{process_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(f"Process_{process_id}")
    logger.setLevel(logging.INFO)

    # 清除已有的 handlers，避免重复
    logger.handlers = []

    # 文件处理器
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter('%(asctime)s - [P%(process)d] %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"进程 {process_id} 日志保存到: {log_file}")
    return logger, log_file


def worker_process(process_id, gpu_id, data_chunk, video_dict, args, log_dir, results_queue):
    """
    工作进程函数

    Args:
        process_id: 进程ID
        gpu_id: GPU设备ID
        data_chunk: 该进程需要处理的数据列表 [(video_name, cor_answer), ...]
        video_dict: 视频信息字典
        args: 命令行参数
        log_dir: 日志目录
        results_queue: 用于返回结果的队列
    """
    # 设置该进程使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 设置OpenAI和Google API密钥
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

    # 为每个进程设置独立的日志
    logger, log_file = setup_logger(process_id, log_dir)

    logger.info(f"=" * 80)
    logger.info(f"进程 {process_id} 启动")
    logger.info(f"GPU设备: {gpu_id} (映射到 cuda:0)")
    logger.info(f"分配任务数: {len(data_chunk)}")
    logger.info(f"=" * 80)

    # 统计变量
    processing_times = []
    success_count = 0
    fail_count = 0
    skip_count = 0
    process_start_time = time.time()

    for idx, (video_name, cor_answer) in enumerate(data_chunk):
        # 获取视频信息
        if video_name not in video_dict:
            logger.warning(f"[P{process_id}][{idx + 1}/{len(data_chunk)}] 视频 {video_name} 不在 video_dict 中，跳过")
            skip_count += 1
            continue

        question = video_dict[video_name]["question"]
        option = video_dict[video_name]["option"]
        question_idx = video_dict[video_name]["question_idx"]
        
        if os.path.exists(f"/root/githubs/VideoRAG/all_results/egoschema_results_question/{video_name}_{args.caption_model_name}_{question_idx}"):
            continue

        # 将 question 和 options 结合
        question_with_options = question + " Options: " + " ".join([f"{opt}" for opt in option])

        # 记录单个视频处理开始时间
        video_start_time = time.time()

        logger.info(f"\n{'=' * 80}")
        logger.info(f"[P{process_id}][{idx + 1}/{len(data_chunk)}] 开始处理视频: {video_name}")
        logger.info(f"问题索引: {question_idx}")
        logger.info(f"{'=' * 80}\n")

        try:
            streaming_processing(args, question_with_options, video_name, option, question_idx, cor_answer)

            # 记录成功
            video_end_time = time.time()
            video_processing_time = video_end_time - video_start_time
            processing_times.append(video_processing_time)
            success_count += 1

            logger.info(f"\n{'=' * 80}")
            logger.info(f"[P{process_id}][{idx + 1}/{len(data_chunk)}] 视频处理完成: {video_name}")
            logger.info(f"处理耗时: {video_processing_time:.2f} 秒 ({video_processing_time / 60:.2f} 分钟)")
            logger.info(f"{'=' * 80}\n")

            # 计算并显示进度
            if len(processing_times) > 0:
                avg_time = sum(processing_times) / len(processing_times)
                remaining_videos = len(data_chunk) - (idx + 1)
                estimated_remaining_time = avg_time * remaining_videos

                logger.info(f"进程 {process_id} 统计:")
                logger.info(f"  - 已处理: {idx + 1}/{len(data_chunk)} 个视频")
                logger.info(f"  - 成功: {success_count}, 失败: {fail_count}, 跳过: {skip_count}")
                logger.info(f"  - 平均耗时: {avg_time:.2f} 秒/视频 ({avg_time / 60:.2f} 分钟/视频)")
                logger.info(f"  - 预计剩余: {estimated_remaining_time:.2f} 秒 ({estimated_remaining_time / 60:.2f} 分钟)")
                logger.info(f"  - 已用总时间: {time.time() - process_start_time:.2f} 秒 ({(time.time() - process_start_time) / 60:.2f} 分钟)\n")

        except Exception as e:
            video_end_time = time.time()
            video_processing_time = video_end_time - video_start_time
            fail_count += 1

            logger.error(f"\n{'=' * 80}")
            logger.error(f"[P{process_id}][{idx + 1}/{len(data_chunk)}] 视频处理失败: {video_name}")
            logger.error(f"错误: {e}")
            logger.error(f"失败前耗时: {video_processing_time:.2f} 秒 ({video_processing_time / 60:.2f} 分钟)")
            logger.error(f"{'=' * 80}\n")
            import traceback
            logger.error(traceback.format_exc())
            # 继续处理下一个视频

    # 进程处理完成总结
    process_end_time = time.time()
    total_processing_time = process_end_time - process_start_time

    logger.info(f"\n{'=' * 80}")
    logger.info(f"进程 {process_id} 处理完成!")
    logger.info(f"{'=' * 80}")
    logger.info(f"总体统计:")
    logger.info(f"  - 处理视频数: {success_count + fail_count}/{len(data_chunk)}")
    logger.info(f"  - 成功: {success_count}, 失败: {fail_count}, 跳过: {skip_count}")
    logger.info(f"  - 总耗时: {total_processing_time:.2f} 秒 ({total_processing_time / 60:.2f} 分钟 / {total_processing_time / 3600:.2f} 小时)")
    if processing_times:
        logger.info(f"  - 平均耗时: {sum(processing_times) / len(processing_times):.2f} 秒/视频")
        logger.info(f"  - 最快: {min(processing_times):.2f} 秒")
        logger.info(f"  - 最慢: {max(processing_times):.2f} 秒")
    logger.info(f"  - 日志文件: {log_file}")
    logger.info(f"{'=' * 80}\n")

    # 将结果放入队列
    results_queue.put({
        'process_id': process_id,
        'gpu_id': gpu_id,
        'total_videos': len(data_chunk),
        'success_count': success_count,
        'fail_count': fail_count,
        'skip_count': skip_count,
        'total_time': total_processing_time,
        'processing_times': processing_times,
        'log_file': log_file
    })


def main():
    parser = argparse.ArgumentParser(description="批处理版本：并行处理 EgoSchema 视频数据集（消融实验）")
    parser.add_argument("--data_path", type=str,
                        default="/data/gst/dataset/EgoSchema/GENERATION/test-00000-of-00001.parquet",
                        help="Path to parquet file")
    parser.add_argument("--retrieval_path", type=str,
                        default="/root/githubs/EgoSchema/subset_answers.json",
                        help="Path to retrieval answers JSON file")
    parser.add_argument("--video_path", type=str,
                        default="/data/gst/dataset/EgoSchema/videos",
                        help="Base path to video files")
    parser.add_argument("--caption_model_name", type=str,
                        default="qwenvl_3_8b_instruct",      # "qwenvl_3_8b_instruct"   "minicpm_4_5_v"   gemini_api    gpt_4o_api
                        help="Caption model name")
    parser.add_argument("--gpu_ids", type=str,
                        default="0",
                        help="GPU IDs to use, comma-separated (default: 0,1,2,3)")
    parser.add_argument("--num_processes", type=int,
                        default=1,
                        help="Total number of processes (default: 4)")
    parser.add_argument("--log_dir", type=str,
                        default="/root/githubs/VideoRAG/videorag_logs/ablation_batch_processing",
                        help="Directory for log files")
    parser.add_argument('--graph_construction_stage', type=bool, default=True)
    parser.add_argument('--retrieval_stage', type=bool, default=False)
    parser.add_argument('--datasets_type', type=str, default="egoschema")

    args = parser.parse_args()

    # 解析GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    num_processes = args.num_processes

    # 确保进程数不超过 GPU 数量（或者循环分配）
    if len(gpu_ids) < num_processes:
        print(f"警告: GPU数量 ({len(gpu_ids)}) 少于进程数 ({num_processes})，将循环分配GPU")
        gpu_ids = [gpu_ids[i % len(gpu_ids)] for i in range(num_processes)]

    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)

    # 读取数据集
    import pandas as pd
    print(f"正在读取 parquet 文件: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    print(f"成功读取 parquet 文件，共 {len(df)} 行")

    # 构建 video_dict
    video_dict = {}
    for idx, row_dict in enumerate(df.to_dict('records')):
        video_dict[row_dict["video_idx"]] = {
            "question": row_dict["question"],
            "option": row_dict["option"],
            "question_idx": row_dict["question_idx"]
        }

    # 读取 retrieval_data
    print(f"正在读取 retrieval 文件: {args.retrieval_path}")
    with open(args.retrieval_path, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)
    print(f"成功读取 retrieval 文件，共 {len(retrieval_data)} 条记录")

    # 将 retrieval_data 转换为列表 [(video_name, cor_answer), ...]
    data_list = list(retrieval_data.items())

    # 将数据切分为 num_processes 份
    chunk_size = len(data_list) // num_processes
    data_chunks = []

    for i in range(num_processes):
        start_idx = i * chunk_size
        if i == num_processes - 1:
            # 最后一个进程处理剩余所有数据
            end_idx = len(data_list)
        else:
            end_idx = (i + 1) * chunk_size
        data_chunks.append(data_list[start_idx:end_idx])

    print(f"\n{'=' * 80}")
    print(f"批处理配置:")
    print(f"  - 总视频数: {len(data_list)}")
    print(f"  - 进程数: {num_processes}")
    print(f"  - GPU设备: {gpu_ids[:num_processes]}")
    for i, chunk in enumerate(data_chunks):
        gpu_id = gpu_ids[i]
        print(f"  - 进程 {i}: {len(chunk)} 个视频, GPU {gpu_id}")
    print(f"  - 日志目录: {args.log_dir}")
    print(f"  - Caption模型: {args.caption_model_name}")
    print(f"{'=' * 80}\n")

    # 创建结果队列
    results_queue = mp.Queue()

    # 创建并启动进程
    processes = []
    overall_start_time = time.time()

    for i in range(num_processes):
        # 为每个进程分配GPU
        gpu_id = gpu_ids[i]

        # 创建进程
        p = mp.Process(
            target=worker_process,
            args=(i, gpu_id, data_chunks[i], video_dict, args, args.log_dir, results_queue)
        )
        processes.append(p)
        p.start()
        print(f"进程 {i} 已启动 (GPU {gpu_id})")

    # 等待所有进程完成
    for i, p in enumerate(processes):
        p.join()
        print(f"进程 {i} 已完成")

    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time

    # 收集所有进程的结果
    all_results = []
    while not results_queue.empty():
        all_results.append(results_queue.get())

    # 排序结果
    all_results.sort(key=lambda x: x['process_id'])

    # 打印总体统计
    print(f"\n{'=' * 80}")
    print(f"所有进程处理完成!")
    print(f"{'=' * 80}")
    print(f"总体统计:")
    print(f"  - 总耗时: {overall_time:.2f} 秒 ({overall_time / 60:.2f} 分钟 / {overall_time / 3600:.2f} 小时)")

    total_videos = 0
    total_success = 0
    total_fail = 0
    total_skip = 0
    all_times = []

    for result in all_results:
        total_videos += result['total_videos']
        total_success += result['success_count']
        total_fail += result['fail_count']
        total_skip += result.get('skip_count', 0)
        all_times.extend(result['processing_times'])

        print(f"\n  进程 {result['process_id']} (GPU {result['gpu_id']}):")
        print(f"    - 视频数: {result['total_videos']}")
        print(f"    - 成功: {result['success_count']}, 失败: {result['fail_count']}, 跳过: {result.get('skip_count', 0)}")
        print(f"    - 耗时: {result['total_time']:.2f} 秒 ({result['total_time'] / 60:.2f} 分钟)")
        if result['processing_times']:
            avg = sum(result['processing_times']) / len(result['processing_times'])
            print(f"    - 平均: {avg:.2f} 秒/视频")
        print(f"    - 日志: {result['log_file']}")

    print(f"\n  汇总:")
    print(f"    - 总视频数: {total_videos}")
    print(f"    - 成功: {total_success}, 失败: {total_fail}, 跳过: {total_skip}")
    if all_times:
        print(f"    - 平均处理时间: {sum(all_times) / len(all_times):.2f} 秒/视频")
        print(f"    - 最快: {min(all_times):.2f} 秒")
        print(f"    - 最慢: {max(all_times):.2f} 秒")
        print(f"    - 吞吐量: {total_success / (overall_time / 60):.2f} 个视频/分钟")
    print(f"{'=' * 80}\n")

    # 保存汇总结果到JSON
    summary_file = os.path.join(args.log_dir, f"ablation_batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    summary = {
        'total_videos': total_videos,
        'total_success': total_success,
        'total_fail': total_fail,
        'total_skip': total_skip,
        'overall_time_seconds': overall_time,
        'overall_time_minutes': overall_time / 60,
        'overall_time_hours': overall_time / 3600,
        'average_time_per_video': sum(all_times) / len(all_times) if all_times else 0,
        'throughput_videos_per_minute': total_success / (overall_time / 60) if overall_time > 0 else 0,
        'num_processes': num_processes,
        'gpu_ids': gpu_ids[:num_processes],
        'caption_model_name': args.caption_model_name,
        'process_results': all_results
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"汇总结果已保存到: {summary_file}")


if __name__ == "__main__":
    # 设置启动方法为 'spawn' (更安全，特别是在使用CUDA时)
    mp.set_start_method('spawn', force=True)
    main()
