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
import time
from PIL import Image
import io
from typing import List, Optional, Callable, Dict, Tuple
from datetime import datetime, timedelta

from openai.resources.conversations.conversations import Conversations
# from videorag.egovideo_graph_retrieval import VideoGraphSeparated
# from videorag.egograph_retrieval_cor import VideoGraphSeparated
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
from videorag.eyewo_prompt import EYEWO_PROMPTS
from videorag.ego_op import streaming_videorag_query
from videorag.streaming_op import QueryParam as StreamingQueryParam
from videorag._utils import pack_user_ass_to_openai_messages, always_get_an_event_loop
from dataclasses import asdict
import asyncio

longervideos_llm_config = LLMConfig(
    embedding_func_raw = gemini_embedding,
    embedding_model_name = "models/text-embedding-004",
    embedding_dim = 768,
    embedding_max_token_size = 2048,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM (we utilize Gemini for all experiments)   
    best_model_func_raw = gemini_pro_complete,
    best_model_name = "gemini-3-pro-preview", 
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gemini_flash_complete,
    cheap_model_name = "gemini-2.5-flash",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

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
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16,
)

def parse_time_span_to_seconds(time_span: str) -> tuple[float, float]:
    """
    从时间戳字符串中提取开始时间和结束时间（以秒为单位）
    
    支持格式：{day}-{HH:MM:SS} or {HH:MM:SS.ss}-{HH:MM:SS} or {HH:MM:SS.ss}
    例如：
        "1-00:00:00.5-00:00:05" -> (0.5, 5.0)
        "2-00:01:30.25-00:01:45.75" -> (90.25, 105.75)
        "1-00:00:10-00:00:20" -> (10.0, 20.0)
    
    Args:
        time_span: 时间戳字符串，格式为 "day-HH:MM:SS-HH:MM:SS" 或类似格式
    
    Returns:
        tuple: (start_time_seconds, end_time_seconds)
    """
    if not time_span or not isinstance(time_span, str):
        raise ValueError(f"Invalid time_span: {time_span}")
    
    parts = time_span.split('-')
    if len(parts) < 3:
        raise ValueError(f"Invalid time_span format: {time_span}. Expected format: 'day-HH:MM:SS-HH:MM:SS'")
    
    start_time_str = parts[1].strip()
    end_time_str = parts[2].strip()
    
    def hhmmss_to_seconds(time_str: str) -> float:
        time_parts = time_str.split(':')
        if len(time_parts) != 3:
            raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM:SS or HH:MM:SS.ss")
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    start_seconds = hhmmss_to_seconds(start_time_str)
    end_seconds = hhmmss_to_seconds(end_time_str)
    return (start_seconds, end_seconds)

def format_interaction_history(history_list):
    """格式化交互历史记录为文本"""
    if not history_list:
        return "(no previous answer)"
    
    history_lines = []
    for item in history_list:
        if item.get('decision') == 'answer':
            timestamp = item.get('timestamp', 'N/A')
            answer = item.get('answer', '')
            history_lines.append(f"[{timestamp}] Answered: {answer}")
        elif item.get('decision') == 'need_retrieval':
            query = item.get('retrieval_query', '')
            history_lines.append(f"Requested retrieval: {query}")
    
    return "\n".join(history_lines) if history_lines else "(no previous answer)"

def eyewo_streaming_retrieval_with_user_queries(
    checkpoint_path: str,
    user_query_list: List[str],
    llm_config: LLMConfig,
    videorag: VideoGraphSeparated,
    datasets_type: str = "eyewo"
) -> Dict:
    """
    对 EyeWO 数据集进行流式检索，处理用户查询列表
    
    Args:
        checkpoint_path: checkpoint.json 文件路径
        user_query_list: 用户查询列表
        llm_config: LLM 配置
        datasets_type: 数据集类型，默认 "eyewo"
    
    Returns:
        包含检索结果和历史记录的字典
    """
    # 读取 checkpoint.json
    logger.info(f"正在读取 checkpoint 文件: {checkpoint_path}")
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        checkpoint_data = json.load(f)
    
    # 获取 second_captions（使用 second-level captions 而不是 min_captions）
    accumulated_captions = checkpoint_data.get('accumulated_captions', {})
    second_captions = accumulated_captions.get('second_captions', [])
    
    if not second_captions:
        logger.warning("未找到 second_captions，使用空列表")
        second_captions = []
    
    # 获取 prompt
    prompt_template = EYEWO_PROMPTS.get("proactive_service_prompt", "")
    memory_prompt_template = EYEWO_PROMPTS.get("proactive_service_prompt_with_memory_simple", "")
    
    # 存储所有查询的结果
    all_query_results = {}
    loop = always_get_an_event_loop()
    
    # 对每个用户查询进行处理
    for query_idx, user_query in enumerate(user_query_list):
        logger.info(f"处理查询 {query_idx + 1}/{len(user_query_list)}: {user_query}")
        
        # 初始化该查询的交互历史
        interaction_history = []
        query_responses = []
        has_retrieved = False  # 标记是否已经执行过检索（只有一次机会）
        
        # 按时间顺序处理 second_captions
        def sort_key(x):
            time_span = x.get('time_span', '')
            timestamp = x.get('timestamp', 0)
            day = 0
            if time_span and '-' in time_span:
                try:
                    day = int(time_span.split('-')[0])
                except (ValueError, IndexError):
                    day = 0
            return (day, timestamp)
        
        sorted_second_captions = sorted(second_captions, key=sort_key)
        logger.info(f"处理 {len(sorted_second_captions)} 个 second_captions")
        
        # 处理每个 second_caption
        for sec_cap_idx, sec_cap_info in enumerate(sorted_second_captions):
            time_span = sec_cap_info.get('time_span', '')
            caption_dict = sec_cap_info.get('caption', '')
            timestamp = sec_cap_info.get('timestamp', 0)
            
            # 处理 caption_dict（可能是 JSON 字符串或字典）
            if isinstance(caption_dict, str):
                try:
                    caption_dict = json.loads(caption_dict)
                except:
                    caption_dict = {"description": caption_dict}
            
            # 提取 caption 文本（优先使用 description）
            caption_text = str(caption_dict)
            
            if not caption_text:
                continue
            
            # 从 time_span 提取时间戳字符串（格式：DAY# HH:MM:SS）
            # time_span 格式：1-00:00:00-00:00:10，提取开始时间
            timestamp_str = ""
            # if time_span and '-' in time_span:
            #     parts = time_span.split('-')
            #     if len(parts) >= 2:
            #         day = parts[0] if parts[0].isdigit() else "1"
            #         time_part = parts[1]  # 开始时间，格式：HH:MM:SS
            #         timestamp_str = f"DAY{day} {time_part}"
            
            # # 如果 caption 中不包含时间戳，添加时间戳
            # if timestamp_str and timestamp_str not in caption_text:
            #     caption_with_timestamp = f"{timestamp_str} {caption_text}"
            # else:
            #     caption_with_timestamp = caption_text
            
            # 格式化交互历史
            history_text = format_interaction_history(interaction_history)
            
            # 构建完整的 prompt
            full_prompt = prompt_template + f"""
------------------------------------------------------------
USER_QUERY
------------------------------------------------------------
{user_query}

------------------------------------------------------------
INTERACTION_HISTORY
------------------------------------------------------------
{history_text}

------------------------------------------------------------
CURRENT_5S_CAPTION
------------------------------------------------------------
{caption_text}
"""
            
            # 调用 LLM 进行决策
            try:
                response = asyncio.run(llm_config.best_model_func_raw(
                    llm_config.best_model_name,
                    full_prompt,
                    system_prompt=None,
                    history_messages=[],
                ))
                logger.debug(f"LLM 响应 (second_cap {sec_cap_idx + 1}): {response[:200]}...")
                
                # 解析 JSON 响应
                import re
                cleaned_response = response.strip()
                
                # 移除可能的 markdown 代码块标记
                if cleaned_response.startswith("```"):
                    lines = cleaned_response.split("\n")
                    cleaned_response = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned_response
                
                # 解析决策
                if cleaned_response == "[]":
                    decision = []
                else:
                    json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                    if json_match:
                        decision = json.loads(json_match.group())
                    else:
                        logger.warning(f"无法解析响应: {cleaned_response[:100]}")
                        decision = []
            except Exception as e:
                logger.error(f"LLM 调用失败 (second_cap {sec_cap_idx + 1}): {e}")
                decision = []
            
            # 处理决策
            if decision == []:
                # 不响应
                continue
            elif isinstance(decision, dict):
                decision_type = decision.get("decision", "")
                
                if decision_type == "answer":
                    # 直接回答
                    answer_timestamp = decision.get("timestamp", timestamp_str)
                    answer_text = decision.get("answer", "")
                    reasoning = decision.get("reasoning", "")
                    
                    logger.info(f"在 {answer_timestamp} 回答: {answer_text[:50]}...")
                    
                    # 添加到交互历史
                    interaction_history.append({
                        'decision': 'answer',
                        'timestamp': answer_timestamp,
                        'answer': answer_text,
                        'reasoning': reasoning,
                        'time_span': time_span
                    })
                    
                    # 添加到查询响应
                    query_responses.append({
                        'time_span': time_span,
                        'timestamp': answer_timestamp,
                        'decision': 'answer',
                        'answer': answer_text,
                        'reasoning': reasoning
                    })
                    
                    # 流式问答任务：继续处理后续的 caption，可能需要在不同时刻多次回答
                    # 交互历史会记录之前的回答，避免重复回答相同内容
                    
                elif decision_type == "need_retrieval" and not has_retrieved:
                    # 需要检索（只有一次机会）
                    retrieval_query = decision.get("retrieval_query", "")
                    logger.info(f"需要检索: {retrieval_query}")
                    
                    # 执行检索
                    try:
                        query_param = StreamingQueryParam(mode="videorag")
                        
                        # 判断 caption_model 的类别名称是否包含 MiniCPM
                        caption_model_class_name = type(videorag.caption_model).__name__
                        use_minicpm = "MiniCPM" in caption_model_class_name
                        
                        retrieved_video_context, retrieved_chunk_context = loop.run_until_complete(
                            streaming_videorag_query(
                                retrieval_query,
                                time_span,  # 使用 time_span 作为 time_key
                                "",  # service_type
                                "",  # sub_service_type
                                datasets_type,
                                videorag.entities_vdb,
                                videorag.text_chunks,
                                videorag.chunks_vdb,
                                videorag.video_segments,
                                videorag.video_segment_feature_vdb,
                                videorag.chunk_entity_relation_graph,
                                videorag.caption_model,
                                videorag.caption_processor,
                                query_param,
                                asdict(videorag),
                                use_minicpm=use_minicpm,
                            )
                        )
                        retrieved_response = retrieved_video_context + "\n" + retrieved_chunk_context
                        logger.info(f"检索完成，获得上下文长度: {len(retrieved_response)} 字符")
                        
                        # 标记已执行检索
                        has_retrieved = True
                        
                        # 构建带记忆的 prompt
                        # 构建历史消息（使用第一次的 prompt 和响应）
                        history = pack_user_ass_to_openai_messages(full_prompt, response)
                        
                        # 添加检索到的记忆
                        continue_prompt = memory_prompt_template + "\n\nRetrieved memory:\n\n" + retrieved_response
                        
                        # 再次调用 LLM（使用带记忆的 prompt）
                        try:
                            gemini_response_with_memory = asyncio.run(llm_config.best_model_func_raw(
                                llm_config.best_model_name,
                                continue_prompt,
                                system_prompt=None,
                                history_messages=history,
                            ))
                            if gemini_response_with_memory:
                                logger.info(f"使用带记忆的响应")
                                response_with_memory = gemini_response_with_memory
                        except Exception as e:
                            logger.warning(f"使用缓存响应失败，使用直接调用: {e}")
                            # 如果失败，直接调用
                            response_with_memory = asyncio.run(llm_config.best_model_func_raw(
                                llm_config.best_model_name,
                                continue_prompt,
                                system_prompt=None,
                                history_messages=[],
                            ))
                        
                        # 解析带记忆的响应
                        cleaned_response_memory = response_with_memory.strip()
                        if cleaned_response_memory.startswith("```"):
                            lines = cleaned_response_memory.split("\n")
                            cleaned_response_memory = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned_response_memory
                        
                        if cleaned_response_memory == "[]":
                            decision_with_memory = []
                        else:
                            json_match = re.search(r'\{.*\}', cleaned_response_memory, re.DOTALL)
                            if json_match:
                                decision_with_memory = json.loads(json_match.group())
                            else:
                                logger.warning(f"无法解析带记忆的响应: {cleaned_response_memory[:100]}")
                                decision_with_memory = []
                        
                        # 处理带记忆的决策
                        if decision_with_memory == []:
                            # 检索后仍然不回答
                            interaction_history.append({
                                'decision': 'need_retrieval',
                                'retrieval_query': retrieval_query,
                                'time_span': time_span,
                                'retrieved_context': retrieved_response[:200] + "...",
                                'final_decision': 'no_answer_after_retrieval'
                            })
                            query_responses.append({
                                'time_span': time_span,
                                'timestamp': timestamp,
                                'decision': 'need_retrieval',
                                'retrieval_query': retrieval_query,
                                'retrieved_context': retrieved_response[:200] + "...",
                                'final_decision': 'no_answer_after_retrieval'
                            })
                        elif isinstance(decision_with_memory, dict) and decision_with_memory.get("decision") == "answer":
                            # 检索后回答
                            answer_timestamp = decision_with_memory.get("timestamp", timestamp_str)
                            answer_text = decision_with_memory.get("answer", "")
                            reasoning = decision_with_memory.get("reasoning", "")
                            
                            logger.info(f"检索后在 {answer_timestamp} 回答: {answer_text[:50]}...")
                            
                            interaction_history.append({
                                'decision': 'answer',
                                'timestamp': answer_timestamp,
                                'answer': answer_text,
                                'reasoning': reasoning,
                                'time_span': time_span,
                                'retrieved_context': retrieved_response[:200] + "..."
                            })
                            
                            query_responses.append({
                                'time_span': time_span,
                                'timestamp': answer_timestamp,
                                'decision': 'answer',
                                'answer': answer_text,
                                'reasoning': reasoning,
                                'retrieved_context': retrieved_response[:200] + "..."
                            })
                    except Exception as e:
                        logger.error(f"检索失败: {e}")
                        # 检索失败，记录但继续
                        interaction_history.append({
                            'decision': 'need_retrieval',
                            'retrieval_query': retrieval_query,
                            'time_span': time_span,
                            'error': str(e)
                        })
                        query_responses.append({
                            'time_span': time_span,
                            'timestamp': timestamp,
                            'decision': 'need_retrieval',
                            'retrieval_query': retrieval_query,
                            'error': str(e)
                        })
                        
                # elif decision_type == "need_retrieval" and has_retrie:
                #     # 已经检索过，不能再检索
                #     logger.warning(f"已执行过检索，跳过检索请求: {retrieval_query}")
                #     continue
                    
        
        # 保存该查询的结果
        all_query_results[user_query] = {
            'query': user_query,
            'interaction_history': interaction_history,
            'responses': query_responses
        }
        
        logger.info(f"查询 '{user_query}' 处理完成，共 {len(query_responses)} 个响应")
    
    # 保存历史记录到文件
    history_output_path = checkpoint_path.replace("streaming_checkpoint.json", "retrieval_history.json")
    with open(history_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_query_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"检索历史已保存到: {history_output_path}")
    
    return {
        'accumulated_captions': accumulated_captions,
        'query_results': all_query_results
    }

def streaming_processing(args, video_name, clip_name, clip_duration, task_type, user_query_list=None):
    """
    Process videos for a specific day and generate streaming captions.
    This function now directly calls streaming_graph_construction which handles
    video processing internally in a streaming manner.
    
    Args:
        args: Command line arguments
        video_data: Video data dictionary
        idx: Index to distinguish different processing instances of the same video
    """
    
    # 设置VideoRAG参数，使用idx来区分同一个视频的不同处理实例
    video_path = os.path.join(args.video_path, video_name + ".mp4")
    videorag = VideoGraphSeparated(llm=gpt5_2_llm_config, working_dir=f"{os.environ.get('RESULTS_ROOT', './results')}/eyewo_results_cor/{clip_name}_{args.caption_model_name}")     # {args.caption_model_name}
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
            interval_seconds=1,
            window_seconds=5,
            gap_threshold_seconds=60,
            window_minutes=1,
            window_hours=1/6,     #10分钟
            max_new_tokens=2048,
            datasets_type=args.datasets_type, 
            clip_duration=clip_duration,
            task_type=task_type, 
            eye_fps=2, 
            user_query_list=user_query_list, 
        )
        
    if args.retrieval_stage:
        # 构建 checkpoint 路径
        checkpoint_path = f"/root/githubs/VideoRAG/VideoRAG-algorithm/eyewo_videos/{clip_name}_{args.caption_model_name}/streaming_checkpoint.json"
        videorag.load_caption_model(model_name=args.caption_model_name)# "qwenvl_3_8b_instruct")
        
        # 确保 user_query_list 不为空
        if user_query_list is None or len(user_query_list) == 0:
            logger.warning(f"user_query_list 为空，跳过检索")
            return results_dict
        
        logger.info(f"开始 EyeWO 流式检索")
        logger.info(f"查询列表: {user_query_list}")
        logger.info(f"Checkpoint 路径: {checkpoint_path}")
        
        # 调用新的检索函数
        retrieval_result = eyewo_streaming_retrieval_with_user_queries(
            checkpoint_path=checkpoint_path,
            user_query_list=user_query_list,
            llm_config=gpt5_2_llm_config,
            videorag=videorag,
            datasets_type=args.datasets_type
        )
        
        logger.info(f"检索完成，共处理 {len(user_query_list)} 个查询")
        
        # 将检索结果合并到 results_dict
        if results_dict is None:
            results_dict = {}
        results_dict['retrieval_result'] = retrieval_result
    
    return results_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming caption generation for video frames")
    parser.add_argument("--data_path", type=str, default="/root/githubs/EyeWO/data/estp_dataset/estp_bench_sq.json",        # "/data/gst/dataset/egolife/A1_JAKE"
                        help="Base path to video data")
    parser.add_argument("--video_path", type=str, default="/data/gst/dataset/ESTP-IT/zhangyl9/ESTP-Bench/full_scale_2fps_max384")
    parser.add_argument("--days", type=str, nargs='+', 
                       default=["DAY1"],
                       help="List of days to process")
    parser.add_argument("--qa_path", type=str, default="/data/gst/dataset/ProAssist-Dataset/processed_data/ego4d/generated_dialogs/val") # /data/gst/dataset/egolife/annotation_segments/A1_JAKE
    
    parser.add_argument("--caption_model_name", type=str, default="minicpm_4_5_o")      # "qwen3_api" "qwenvl_3_8b_instruct"   "minicpm_4_5_v"   gemini_api    gpt_4o_api
    parser.add_argument("--interval_seconds", type=int, default=3,
                       help="Interval between sampled frames in seconds (default: 3)")
    parser.add_argument("--window_seconds", type=int, default=10,
                       help="Time window in seconds for each caption (default: 10)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for saving captions (optional)")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of retry attempts")
    parser.add_argument("--max_size_video", type=int, nargs=2, default=[256, 256],
                       help="Maximum size for video frames [width, height]")
    parser.add_argument("--quality", type=int, default=85,
                       help="JPEG quality for encoding (1-100)")
    
    parser.add_argument('--cuda', type=str, default='6',
                       help="CUDA device ID(s) to use. Sets CUDA_VISIBLE_DEVICES environment variable. "
                            "Model will run on cuda:0 (mapped from the specified physical device).")
    
    parser.add_argument('--graph_construction_stage', type=bool, default=True)
    parser.add_argument('--retrieval_stage', type=bool, default=False)
    parser.add_argument('--datasets_type', type=str, default="eyewo")

    args = parser.parse_args()
    
    # 在模型加载之前设置CUDA_VISIBLE_DEVICES，确保模型固定在该设备上运行
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.cuda}, model will run on cuda:0 (mapped from physical device {args.cuda})")
    else:
        # 如果没有指定，检查环境变量是否已设置
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            logger.info("CUDA_VISIBLE_DEVICES not set, using default device allocation")
        else:
            logger.info(f"Using existing CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
    
    video_data = json.load(open(args.data_path, "r"))
    
    # 用于保存处理时间的数据结构
    processing_times = []
    
    for video_name, clip_info in video_data.items():    # 单独处理每一个视频，使用idx区分同一个视频的不同处理实例
        if "goalstep" in list(clip_info.keys()):
            goal_step = clip_info["goalstep"]
            for idx, clip_list in enumerate(goal_step):
                start_time = clip_list["start_time"]
                end_time = clip_list["end_time"]
                clip_duration = (start_time, end_time)
                clip_name = f"{clip_list['video_uid']}_{idx}"
                
                # 计算视频时长（秒）
                video_duration = end_time - start_time
                
                # 记录处理开始时间
                processing_start = time.time()
                
                task_type = [clip_list["Task Type"]]
                user_query_list = [clip_list["conversation"][0]["content"]]
                streaming_processing(args, video_name, clip_name, clip_duration, task_type, user_query_list)
                
                # 记录处理结束时间并计算耗时
                processing_end = time.time()
                processing_time = processing_end - processing_start
                
                # 保存处理时间信息
                time_record = {
                    "video_name": video_name,
                    "clip_name": clip_name,
                    "video_duration_seconds": video_duration,
                    "processing_time_seconds": processing_time,
                    "processing_time_minutes": processing_time / 60.0,
                    "speed_ratio": video_duration / processing_time if processing_time > 0 else 0,  # 视频时长/处理时间，值越大表示处理越快
                    "task_type": task_type,
                    "timestamp": datetime.now().isoformat()
                }
                processing_times.append(time_record)
                
                logger.info(f"视频 {clip_name} 处理完成 - 视频时长: {video_duration:.2f}秒, 处理时间: {processing_time:.2f}秒 ({processing_time/60:.2f}分钟), 速度比: {time_record['speed_ratio']:.2f}x")
        else:
            for clip_name, clip_list in clip_info.items():
                
                # if clip_name != "068cb0a2-d3fa-4e1a-947b-2612d9785aa2":
                #     continue
                
                start_time = clip_list[0]["clip_start_time"]
                end_time = clip_list[0]["clip_end_time"]
                clip_duration = (start_time, end_time)
                
                # 计算视频时长（秒）
                video_duration = end_time - start_time
                
                # 记录处理开始时间
                processing_start = time.time()
                
                task_type = set()
                user_query_list = []
                for question_pair in clip_list:
                    task_type.add(question_pair["Task Type"])
                    user_query_list.append(question_pair["question"])
                task_type = list(task_type)
                
                streaming_processing(args, video_name, clip_name, clip_duration, task_type, user_query_list)
                
                # 记录处理结束时间并计算耗时
                processing_end = time.time()
                processing_time = processing_end - processing_start
                
                # 保存处理时间信息
                time_record = {
                    "video_name": video_name,
                    "clip_name": clip_name,
                    "video_duration_seconds": video_duration,
                    "processing_time_seconds": processing_time,
                    "processing_time_minutes": processing_time / 60.0,
                    "speed_ratio": video_duration / processing_time if processing_time > 0 else 0,  # 视频时长/处理时间，值越大表示处理越快
                    "task_type": task_type,
                    "timestamp": datetime.now().isoformat()
                }
                processing_times.append(time_record)
                
                logger.info(f"视频 {clip_name} 处理完成 - 视频时长: {video_duration:.2f}秒, 处理时间: {processing_time:.2f}秒 ({processing_time/60:.2f}分钟), 速度比: {time_record['speed_ratio']:.2f}x")
    
    # 保存处理时间数据到文件
    if processing_times:
        # 生成输出文件名（包含时间戳）
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"processing_times_{timestamp_str}.json"
        
        # 计算统计信息
        total_videos = len(processing_times)
        total_video_duration = sum(record["video_duration_seconds"] for record in processing_times)
        total_processing_time = sum(record["processing_time_seconds"] for record in processing_times)
        avg_speed_ratio = sum(record["speed_ratio"] for record in processing_times) / total_videos if total_videos > 0 else 0
        
        # 按视频时长分组统计
        duration_groups = {
            "short (< 30s)": [],
            "medium (30s - 2min)": [],
            "long (2min - 5min)": [],
            "very_long (> 5min)": []
        }
        
        for record in processing_times:
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
                "total_videos": total_videos,
                "total_video_duration_seconds": total_video_duration,
                "total_video_duration_minutes": total_video_duration / 60.0,
                "total_processing_time_seconds": total_processing_time,
                "total_processing_time_minutes": total_processing_time / 60.0,
                "total_processing_time_hours": total_processing_time / 3600.0,
                "average_speed_ratio": avg_speed_ratio,
                "timestamp": datetime.now().isoformat()
            },
            "duration_group_statistics": group_stats,
            "detailed_records": processing_times
        }
        
        # 保存到 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"处理时间统计已保存到: {output_file}")
        logger.info(f"{'='*60}")
        logger.info(f"总视频数: {total_videos}")
        logger.info(f"总视频时长: {total_video_duration/60:.2f} 分钟 ({total_video_duration/3600:.2f} 小时)")
        logger.info(f"总处理时间: {total_processing_time/60:.2f} 分钟 ({total_processing_time/3600:.2f} 小时)")
        logger.info(f"平均速度比: {avg_speed_ratio:.2f}x")
        logger.info(f"\n按视频时长分组的统计:")
        for group_name, stats in group_stats.items():
            logger.info(f"  {group_name}: {stats['count']} 个视频, 平均处理时间: {stats['avg_processing_time_minutes']:.2f} 分钟, 平均速度比: {stats['avg_speed_ratio']:.2f}x")
        logger.info(f"{'='*60}\n")