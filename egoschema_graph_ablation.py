import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import argparse
import glob
import json
import base64
import logging
import re
import time
from PIL import Image
import io
from typing import List, Optional, Callable, Dict, Tuple
from datetime import datetime, timedelta
from collections import Counter

from openai.resources.conversations.conversations import Conversations
# from videorag.egovideo_graph_retrieval import VideoGraphSeparated
from videorag.egograph_retrieval_optimize_ import VideoGraphSeparated

import pyarrow.ipc as ipc
import datasets as hf_datasets
from videorag._videoutil.dataset import BaseDataset

hf_datasets.disable_progress_bars()

from videorag.video_processing import sample_frames_by_interval

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from videorag._llm import *
from videorag.videorag import VideoRAG, QueryParam
from videorag.egoschema_prompt import EGOSCHEMA_PROMPTS
from videorag.ego_op import streaming_videorag_query
from videorag.streaming_op import QueryParam as StreamingQueryParam
from videorag._utils import pack_user_ass_to_openai_messages, always_get_an_event_loop
from dataclasses import asdict
import asyncio


gpt5_2_llm_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM (we utilize gpt-4o-mini for all experiments)   
    best_model_func_raw = gpt_4o_mini_complete,
    best_model_name = "gpt-4o-mini", 
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",     # gpt-5-mini
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16,
)


def streaming_processing(args, question, video_name, option, question_idx, cor_answer, output_results=None):
    """
    Process videos for a specific day and generate streaming captions.
    This function now directly calls streaming_graph_construction which handles
    video processing internally in a streaming manner.
    
    Args:
        args: Command line arguments
        video_data: Video data dictionary
        idx: Index to distinguish different processing instances of the same video
    """
    video_path = os.path.join(args.video_path, video_name + ".mp4")
    
    # 设置VideoRAG参数，使用idx来区分同一个视频的不同处理实例
    videorag = VideoGraphSeparated(llm=gpt5_2_llm_config, working_dir=f"{os.environ.get('RESULTS_ROOT', './results')}/egoschema_results_question_500/{video_name}_{args.caption_model_name}_{question_idx}")     # {args.caption_model_name}
    videorag.load_caption_model(model_name=args.caption_model_name)
    
    # 直接调用streaming_graph_construction，视频处理逻辑已集成到函数内部
    # 函数会逐个视频处理，每3秒提取一帧，积累10秒后生成caption并判断主动服务
    results_dict = None
    if args.graph_construction_stage:
        day = "DAY1"
        results_dict = videorag.streaming_graph_construction(
            data_path=video_path,
            anno_path=None,
            day=day,
            interval_seconds=3,   # 1, 3
            window_seconds=15,   # 5, 15
            gap_threshold_seconds=60,
            window_minutes=1,
            window_hours=1/6,     #10分钟
            max_new_tokens=2048,
            datasets_type=args.datasets_type, 
            question_with_option=question, 
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming caption generation for video frames")
    parser.add_argument("--data_path", type=str, default=os.environ.get("EGOSCHEMA_DATA_PATH", "./data/egoschema/test-00000-of-00001.parquet"),        # "/data/gst/dataset/egolife/A1_JAKE"
                        help="Base path to video data")
    parser.add_argument("--retrieval_path", type=str, default=os.environ.get("EGOSCHEMA_RETRIEVAL_PATH", "./data/egoschema/subset_answers.json"))
    parser.add_argument("--video_path", type=str, default=os.environ.get("EGOSCHEMA_VIDEO_PATH", "./data/egoschema/videos"))
    
    parser.add_argument("--caption_model_name", type=str, default="qwenvl_3_8b_instruct")      # "qwenvl_3_8b_instruct"   "minicpm_4_5_v"   gemini_api    gpt_4o_api
    
    parser.add_argument('--cuda', type=str, default='1',
                       help="CUDA device ID(s) to use. Sets CUDA_VISIBLE_DEVICES environment variable. "
                            "Model will run on cuda:0 (mapped from the specified physical device).")
    
    parser.add_argument('--graph_construction_stage', type=bool, default=True)
    parser.add_argument('--retrieval_stage', type=bool, default=False)
    parser.add_argument('--datasets_type', type=str, default="egoschema")

    args = parser.parse_args()
    
    # 读取 parquet 文件
    import pandas as pd
    logger.info(f"正在读取 parquet 文件: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    logger.info(f"成功读取 parquet 文件，共 {len(df)} 行，{len(df.columns)} 列")
    logger.info(f"列名: {list(df.columns)}")
    
    # 在模型加载之前设置CUDA_VISIBLE_DEVICES，确保模型固定在该设备上运行
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    logger.info(f"Set CUDA_VISIBLE_DEVICES={args.cuda}, model will run on cuda:0 (mapped from physical device {args.cuda})")
    
    
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
    
    # 统计变量
    total_videos = len(df)
    processing_times = []
    overall_start_time = time.time()
    
    logger.info(f"=" * 80)
    logger.info(f"开始处理 {total_videos} 个视频")
    logger.info(f"=" * 80)
    
    # 读取 subset_answers.json 文件
    with open(args.retrieval_path, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)
    
    video_dict = {}
    for idx, row_dict in enumerate(df.to_dict('records')):
        video_dict[row_dict["video_idx"]] = {"question": row_dict["question"], "option": row_dict["option"], "question_idx": row_dict["question_idx"]}

    for loop_idx, (video_name, cor_answer) in enumerate(retrieval_data.items()):    # 单独处理每一个视频，使用idx区分同一个视频的不同处理实例
        question = video_dict[video_name]["question"]
        option = video_dict[video_name]["option"]
        question_idx = video_dict[video_name]["question_idx"]

        question_with_options = question + " Options: " + " ".join([f"{opt}" for opt in option])
        
        # 记录单个视频处理开始时间
        video_start_time = time.time()
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"[{loop_idx + 1}/{total_videos}] 开始处理视频: {video_name}")
        logger.info(f"问题索引: {question_idx}")
        logger.info(f"{'=' * 80}\n")
        
        try:
            streaming_processing(args, question_with_options, video_name, option, question_idx, cor_answer)
            
            # 记录单个视频处理结束时间
            video_end_time = time.time()
            video_processing_time = video_end_time - video_start_time
            processing_times.append(video_processing_time)
            
            # 打印单个视频处理时间
            logger.info(f"\n{'=' * 80}")
            logger.info(f"[{loop_idx + 1}/{total_videos}] 视频处理完成: {video_name}")
            logger.info(f"处理耗时: {video_processing_time:.2f} 秒 ({video_processing_time / 60:.2f} 分钟)")
            logger.info(f"{'=' * 80}\n")
            
            # 计算并显示平均时间和预计剩余时间
            if len(processing_times) > 0:
                avg_time = sum(processing_times) / len(processing_times)
                remaining_videos = total_videos - (loop_idx + 1)
                estimated_remaining_time = avg_time * remaining_videos
                
                logger.info(f"统计信息:")
                logger.info(f"  - 已处理: {idx + 1}/{total_videos} 个视频")
                logger.info(f"  - 平均耗时: {avg_time:.2f} 秒/视频 ({avg_time / 60:.2f} 分钟/视频)")
                logger.info(f"  - 预计剩余: {estimated_remaining_time:.2f} 秒 ({estimated_remaining_time / 60:.2f} 分钟)")
                logger.info(f"  - 已用总时间: {time.time() - overall_start_time:.2f} 秒 ({(time.time() - overall_start_time) / 60:.2f} 分钟)\n")
        
        except Exception as e:
            video_end_time = time.time()
            video_processing_time = video_end_time - video_start_time
            logger.error(f"\n{'=' * 80}")
            logger.error(f"[{loop_idx + 1}/{total_videos}] 视频处理失败: {video_name}")
            logger.error(f"错误: {e}")
            logger.error(f"失败前耗时: {video_processing_time:.2f} 秒 ({video_processing_time / 60:.2f} 分钟)")
            logger.error(f"{'=' * 80}\n")
            # 可选：继续处理下一个视频或者中断
            # raise  # 取消注释以在出错时中断
    
    # 遍历结束，保存 output_results
    # _save_output_results()

    # 处理完所有视频后的总结
    overall_end_time = time.time()
    total_processing_time = overall_end_time - overall_start_time
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"所有视频处理完成!")
    logger.info(f"{'=' * 80}")
    logger.info(f"总体统计:")
    logger.info(f"  - 处理视频数: {len(processing_times)}/{total_videos}")
    logger.info(f"  - 总耗时: {total_processing_time:.2f} 秒 ({total_processing_time / 60:.2f} 分钟 / {total_processing_time / 3600:.2f} 小时)")
    if processing_times:
        logger.info(f"  - 平均耗时: {sum(processing_times) / len(processing_times):.2f} 秒/视频")
        logger.info(f"  - 最快: {min(processing_times):.2f} 秒")
        logger.info(f"  - 最慢: {max(processing_times):.2f} 秒")
    logger.info(f"{'=' * 80}\n")