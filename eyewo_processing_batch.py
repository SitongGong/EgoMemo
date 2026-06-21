"""
批处理版本：将视频数据集切分为多份，使用多GPU并行处理
一个进程使用一块GPU卡
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

from eyewo_processing import streaming_processing

# 配置日志（每个进程会有自己的日志文件）
def setup_logger(process_id, log_dir):
    """为每个进程设置独立的日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"eyewo_batch_process_{process_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger(f"Process_{process_id}")
    logger.setLevel(logging.INFO)
    
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


def collect_all_clips(video_data):
    """
    收集所有需要处理的clips
    
    Returns:
        List of tuples: (video_name, clip_name, clip_duration, task_type, user_query_list)
    """
    all_clips = []
    
    for video_name, clip_info in video_data.items():
        
        # if "goalstep" in list(clip_info.keys()) and len(list(clip_info.keys())) > 1:
        #     print(video_name)
        
        for clip_name, clip_list in clip_info.items():
            
            if clip_name == "goalstep":
                for idx, clip_list in enumerate(clip_info["goalstep"]):
                    start_time = clip_list["start_time"]
                    end_time = clip_list["end_time"]
                    clip_duration = (start_time, end_time)
                    clip_name = f"{clip_list['video_uid']}_{idx}"
                    
                    task_type = [clip_list["Task Type"]]
                    user_query_list = [clip_list["conversation"][0]["content"]]
                    
                    all_clips.append((video_name, clip_name, clip_duration, task_type, user_query_list))
            else:
                start_time = clip_list[0]["clip_start_time"]
                end_time = clip_list[0]["clip_end_time"]
                clip_duration = (start_time, end_time)
                
                task_type = set()
                user_query_list = []
                for question_pair in clip_list:
                    task_type.add(question_pair["Task Type"])
                    user_query_list.append(question_pair["question"])
                task_type = list(task_type)
                
                all_clips.append((video_name, clip_name, clip_duration, task_type, user_query_list))
        
        # if "goalstep" in list(clip_info.keys()):
        #     goal_step = clip_info["goalstep"]
        #     for idx, clip_list in enumerate(goal_step):
        #         start_time = clip_list["start_time"]
        #         end_time = clip_list["end_time"]
        #         clip_duration = (start_time, end_time)
        #         clip_name = f"{clip_list['video_uid']}_{idx}"
                
        #         task_type = [clip_list["Task Type"]]
        #         user_query_list = [clip_list["conversation"][0]["content"]]
                
        #         all_clips.append((video_name, clip_name, clip_duration, task_type, user_query_list))
        # else:
        #     for clip_name, clip_list in clip_info.items():
        #         start_time = clip_list[0]["clip_start_time"]
        #         end_time = clip_list[0]["clip_end_time"]
        #         clip_duration = (start_time, end_time)
                
        #         if clip_name == "3039bfc5-9856-4c65-8908-d8928d6b3579" or clip_name == "55ceecc4-af50-42f6-b258-aff17f17e128":
        #             pass
                
        #         task_type = set()
        #         user_query_list = []
        #         for question_pair in clip_list:
        #             task_type.add(question_pair["Task Type"])
        #             user_query_list.append(question_pair["question"])
        #         task_type = list(task_type)
                
        #         all_clips.append((video_name, clip_name, clip_duration, task_type, user_query_list))
    
    return all_clips


def worker_process(process_id, gpu_id, clips_chunk, args, log_dir, results_queue):
    """
    工作进程函数
    
    Args:
        process_id: 进程ID (0-N)
        gpu_id: GPU设备ID
        clips_chunk: 该进程需要处理的clips列表 [(video_name, clip_name, clip_duration, task_type, user_query_list), ...]
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
    logger.info(f"分配任务数: {len(clips_chunk)}")
    logger.info(f"=" * 80)
    
    # 统计变量
    processing_times = []
    success_count = 0
    fail_count = 0
    process_start_time = time.time()
    
    for idx, (video_name, clip_name, clip_duration, task_type, user_query_list) in enumerate(clips_chunk):
        # 计算视频时长（秒）
        start_time, end_time = clip_duration
        video_duration = end_time - start_time
        
        # 记录单个视频处理开始时间
        video_start_time = time.time()
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"[P{process_id}][{idx + 1}/{len(clips_chunk)}] 开始处理视频: {clip_name}")
        logger.info(f"视频名称: {video_name}")
        logger.info(f"视频时长: {video_duration:.2f} 秒")
        logger.info(f"任务类型: {task_type}")
        logger.info(f"{'=' * 80}\n")
        
        if clip_name == "3039bfc5-9856-4c65-8908-d8928d6b3579" or clip_name == "55ceecc4-af50-42f6-b258-aff17f17e128":
            pass
        
        if os.path.exists(f"{os.environ.get('RESULTS_ROOT', './results')}/eyewo_results_cor/{clip_name}_{args.caption_model_name}"): 
            continue
        
        try:
            streaming_processing(args, video_name, clip_name, clip_duration, task_type, user_query_list)
            
            # 记录成功
            video_end_time = time.time()
            video_processing_time = video_end_time - video_start_time
            success_count += 1
            
            # 保存处理时间信息
            time_record = {
                "video_name": video_name,
                "clip_name": clip_name,
                "video_duration_seconds": video_duration,
                "processing_time_seconds": video_processing_time,
                "processing_time_minutes": video_processing_time / 60.0,
                "speed_ratio": video_duration / video_processing_time if video_processing_time > 0 else 0,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }
            processing_times.append(time_record)
            
            logger.info(f"\n{'=' * 80}")
            logger.info(f"[P{process_id}][{idx + 1}/{len(clips_chunk)}] 视频处理完成: {clip_name}")
            logger.info(f"处理耗时: {video_processing_time:.2f} 秒 ({video_processing_time / 60:.2f} 分钟)")
            logger.info(f"速度比: {time_record['speed_ratio']:.2f}x")
            logger.info(f"{'=' * 80}\n")
            
            # 计算并显示进度
            if len(processing_times) > 0:
                avg_time = sum(t["processing_time_seconds"] for t in processing_times) / len(processing_times)
                remaining_clips = len(clips_chunk) - (idx + 1)
                estimated_remaining_time = avg_time * remaining_clips
                
                logger.info(f"进程 {process_id} 统计:")
                logger.info(f"  - 已处理: {idx + 1}/{len(clips_chunk)} 个视频")
                logger.info(f"  - 成功: {success_count}, 失败: {fail_count}")
                logger.info(f"  - 平均耗时: {avg_time:.2f} 秒/视频 ({avg_time / 60:.2f} 分钟/视频)")
                logger.info(f"  - 预计剩余: {estimated_remaining_time:.2f} 秒 ({estimated_remaining_time / 60:.2f} 分钟)")
                logger.info(f"  - 已用总时间: {time.time() - process_start_time:.2f} 秒 ({(time.time() - process_start_time) / 60:.2f} 分钟)\n")
        
        except Exception as e:
            video_end_time = time.time()
            video_processing_time = video_end_time - video_start_time
            fail_count += 1
            
            logger.error(f"\n{'=' * 80}")
            logger.error(f"[P{process_id}][{idx + 1}/{len(clips_chunk)}] 视频处理失败: {clip_name}")
            logger.error(f"错误: {e}")
            logger.error(f"失败前耗时: {video_processing_time:.2f} 秒 ({video_processing_time / 60:.2f} 分钟)")
            logger.error(f"{'=' * 80}\n")
            # 继续处理下一个视频
    
    # 进程处理完成总结
    process_end_time = time.time()
    total_processing_time = process_end_time - process_start_time
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"进程 {process_id} 处理完成!")
    logger.info(f"{'=' * 80}")
    logger.info(f"总体统计:")
    logger.info(f"  - 处理视频数: {success_count + fail_count}/{len(clips_chunk)}")
    logger.info(f"  - 成功: {success_count}, 失败: {fail_count}")
    logger.info(f"  - 总耗时: {total_processing_time:.2f} 秒 ({total_processing_time / 60:.2f} 分钟 / {total_processing_time / 3600:.2f} 小时)")
    if processing_times:
        avg_time = sum(t["processing_time_seconds"] for t in processing_times) / len(processing_times)
        logger.info(f"  - 平均耗时: {avg_time:.2f} 秒/视频")
        logger.info(f"  - 最快: {min(t['processing_time_seconds'] for t in processing_times):.2f} 秒")
        logger.info(f"  - 最慢: {max(t['processing_time_seconds'] for t in processing_times):.2f} 秒")
        avg_speed_ratio = sum(t['speed_ratio'] for t in processing_times) / len(processing_times)
        logger.info(f"  - 平均速度比: {avg_speed_ratio:.2f}x")
    logger.info(f"  - 日志文件: {log_file}")
    logger.info(f"{'=' * 80}\n")
    
    # 将结果放入队列
    results_queue.put({
        'process_id': process_id,
        'gpu_id': gpu_id,
        'total_clips': len(clips_chunk),
        'success_count': success_count,
        'fail_count': fail_count,
        'total_time': total_processing_time,
        'processing_times': processing_times,
        'log_file': log_file
    })


def main():
    parser = argparse.ArgumentParser(description="批处理版本：并行处理EyeWO视频数据集")
    parser.add_argument("--data_path", type=str, 
                        default=os.environ.get("EYEWO_DATA_PATH", "./data/eyewo/estp_bench_sq.json"),
                        help="Path to JSON data file")
    parser.add_argument("--video_path", type=str, 
                        default=os.environ.get("EYEWO_VIDEO_PATH", "./data/eyewo/ESTP-Bench/full_scale_2fps_max384"),
                        help="Base path to video files")
    parser.add_argument("--caption_model_name", type=str, default="qwenvl_3_8b_instruct", # "qwenvl_3_8b_instruct",
                        help="Caption model name")
    parser.add_argument("--gpu_ids", type=str, 
                        default="0",
                        help="GPU IDs to use, comma-separated (default: 0,1,2,3)")
    parser.add_argument("--num_processes", type=int, 
                        default=1,
                        help="Total number of processes (default: 4)")
    parser.add_argument("--log_dir", type=str,
                        default="/root/githubs/VideoRAG/videorag_logs/eyewo_batch_processing",
                        help="Directory for log files")
    parser.add_argument('--graph_construction_stage', type=bool, default=True)
    parser.add_argument('--retrieval_stage', type=bool, default=False)
    parser.add_argument('--datasets_type', type=str, default="eyewo")
    
    args = parser.parse_args()
    
    # 解析GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    if len(gpu_ids) < args.num_processes:
        print(f"警告: 进程数 ({args.num_processes}) 大于GPU数 ({len(gpu_ids)})")
        print(f"将循环使用GPU: {gpu_ids}")
        # 扩展GPU列表以匹配进程数
        while len(gpu_ids) < args.num_processes:
            gpu_ids.extend(gpu_ids[:args.num_processes - len(gpu_ids)])
    elif len(gpu_ids) > args.num_processes:
        print(f"信息: GPU数 ({len(gpu_ids)}) 大于进程数 ({args.num_processes})")
        print(f"将只使用前 {args.num_processes} 个GPU: {gpu_ids[:args.num_processes]}")
        gpu_ids = gpu_ids[:args.num_processes]
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 读取数据集
    print(f"正在读取数据集: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        video_data = json.load(f)
    print(f"成功读取数据集，共 {len(video_data)} 个视频")
    
    # 收集所有需要处理的clips
    print("正在收集所有clips...")
    all_clips = collect_all_clips(video_data)
    print(f"共收集到 {len(all_clips)} 个clips")
    
    # 将数据切分为num_processes份
    chunk_size = len(all_clips) // args.num_processes
    clips_chunks = []
    
    for i in range(args.num_processes):
        start_idx = i * chunk_size
        if i == args.num_processes - 1:
            # 最后一个进程处理剩余所有数据
            end_idx = len(all_clips)
        else:
            end_idx = (i + 1) * chunk_size
        clips_chunks.append(all_clips[start_idx:end_idx])
    
    print(f"\n{'=' * 80}")
    print(f"批处理配置:")
    print(f"  - 总clips数: {len(all_clips)}")
    print(f"  - 进程数: {args.num_processes}")
    print(f"  - GPU设备: {gpu_ids}")
    for i, chunk in enumerate(clips_chunks):
        gpu_id = gpu_ids[i]  # 一个进程使用一块卡
        print(f"  - 进程 {i}: {len(chunk)} 个clips, GPU {gpu_id}")
    print(f"  - 日志目录: {args.log_dir}")
    print(f"{'=' * 80}\n")
    
    # 创建结果队列
    results_queue = mp.Queue()
    
    # 创建并启动进程
    processes = []
    overall_start_time = time.time()
    
    for i in range(args.num_processes):
        # 为每个进程分配GPU (一个进程使用一块卡)
        gpu_id = gpu_ids[i]
        
        # 创建进程
        p = mp.Process(
            target=worker_process,
            args=(i, gpu_id, clips_chunks[i], args, args.log_dir, results_queue)
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
    
    # 收集所有处理时间记录
    all_processing_times = []
    for result in all_results:
        all_processing_times.extend(result['processing_times'])
    
    # 打印总体统计
    print(f"\n{'=' * 80}")
    print(f"所有进程处理完成!")
    print(f"{'=' * 80}")
    print(f"总体统计:")
    print(f"  - 总耗时: {overall_time:.2f} 秒 ({overall_time / 60:.2f} 分钟 / {overall_time / 3600:.2f} 小时)")
    
    total_clips = 0
    total_success = 0
    total_fail = 0
    
    for result in all_results:
        total_clips += result['total_clips']
        total_success += result['success_count']
        total_fail += result['fail_count']
        
        print(f"\n  进程 {result['process_id']} (GPU {result['gpu_id']}):")
        print(f"    - Clips数: {result['total_clips']}")
        print(f"    - 成功: {result['success_count']}, 失败: {result['fail_count']}")
        print(f"    - 耗时: {result['total_time']:.2f} 秒 ({result['total_time'] / 60:.2f} 分钟)")
        if result['processing_times']:
            avg = sum(t["processing_time_seconds"] for t in result['processing_times']) / len(result['processing_times'])
            print(f"    - 平均: {avg:.2f} 秒/clip")
        print(f"    - 日志: {result['log_file']}")
    
    print(f"\n  汇总:")
    print(f"    - 总clips数: {total_clips}")
    print(f"    - 成功: {total_success}, 失败: {total_fail}")
    if all_processing_times:
        avg_time = sum(t["processing_time_seconds"] for t in all_processing_times) / len(all_processing_times)
        print(f"    - 平均处理时间: {avg_time:.2f} 秒/clip")
        print(f"    - 最快: {min(t['processing_time_seconds'] for t in all_processing_times):.2f} 秒")
        print(f"    - 最慢: {max(t['processing_time_seconds'] for t in all_processing_times):.2f} 秒")
        print(f"    - 吞吐量: {total_success / (overall_time / 60):.2f} 个clips/分钟")
    print(f"{'=' * 80}\n")
    
    # 计算详细统计信息（按视频时长分组）
    if all_processing_times:
        total_video_duration = sum(t["video_duration_seconds"] for t in all_processing_times)
        total_processing_time = sum(t["processing_time_seconds"] for t in all_processing_times)
        avg_speed_ratio = sum(t["speed_ratio"] for t in all_processing_times) / len(all_processing_times) if all_processing_times else 0
        
        # 按视频时长分组统计
        duration_groups = {
            "short (< 30s)": [],
            "medium (30s - 2min)": [],
            "long (2min - 5min)": [],
            "very_long (> 5min)": []
        }
        
        for record in all_processing_times:
            duration = record["video_duration_seconds"]
            if duration < 30:
                duration_groups["short (< 30s)"].append(record)
            elif duration < 120:
                duration_groups["medium (30s - 2min)"].append(record)
            elif duration < 300:
                duration_groups["long (2min - 5min)"].append(record)
            else:
                duration_groups["very_long (> 5min)"].append(record)
        
        # 计算各组的平均处理时间
        group_stats = {}
        for group_name, records in duration_groups.items():
            if records:
                avg_processing = sum(r["processing_time_seconds"] for r in records) / len(records)
                avg_duration = sum(r["video_duration_seconds"] for r in records) / len(records)
                avg_ratio = sum(r["speed_ratio"] for r in records) / len(records)
                group_stats[group_name] = {
                    "count": len(records),
                    "avg_video_duration_seconds": avg_duration,
                    "avg_processing_time_seconds": avg_processing,
                    "avg_processing_time_minutes": avg_processing / 60.0,
                    "avg_speed_ratio": avg_ratio
                }
        
        # 构建保存的数据
        save_data = {
            "summary": {
                "total_clips": total_clips,
                "total_success": total_success,
                "total_fail": total_fail,
                "total_video_duration_seconds": total_video_duration,
                "total_video_duration_minutes": total_video_duration / 60.0,
                "total_processing_time_seconds": total_processing_time,
                "total_processing_time_minutes": total_processing_time / 60.0,
                "total_processing_time_hours": total_processing_time / 3600.0,
                "overall_time_seconds": overall_time,
                "overall_time_minutes": overall_time / 60.0,
                "overall_time_hours": overall_time / 3600.0,
                "average_speed_ratio": avg_speed_ratio,
                "num_processes": args.num_processes,
                "gpu_ids": gpu_ids,
                "timestamp": datetime.now().isoformat()
            },
            "duration_group_statistics": group_stats,
            "process_results": all_results,
            "detailed_records": all_processing_times
        }
        
        # 保存到 JSON 文件
        summary_file = os.path.join(args.log_dir, f"eyewo_batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"汇总结果已保存到: {summary_file}")
        print(f"\n按视频时长分组的统计:")
        for group_name, stats in group_stats.items():
            print(f"  {group_name}: {stats['count']} 个视频, 平均处理时间: {stats['avg_processing_time_minutes']:.2f} 分钟, 平均速度比: {stats['avg_speed_ratio']:.2f}x")


if __name__ == "__main__":
    # 设置启动方法为 'spawn' (更安全，特别是在使用CUDA时)
    mp.set_start_method('spawn', force=True)
    main()
