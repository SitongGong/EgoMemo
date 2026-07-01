import os
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

# Suppress verbose logging from OpenAI and httpx libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from videorag._llm import *
from videorag.videorag import VideoRAG, QueryParam
from videorag.eyewo_prompt_ import EYEWO_PROMPTS
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
    best_model_name = "gpt-5.2", 
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-5-mini",
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
    datasets_type: str = "eyewo",
    args=None, 
    task_type=None, 
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
    prompt_template = EYEWO_PROMPTS.get("proactive_service_prompt_", "")
    memory_prompt_template = EYEWO_PROMPTS.get("proactive_service_prompt_with_memory_simple", "")
    
    # 存储所有查询的结果
    all_query_results = {}
    loop = always_get_an_event_loop()
    
    # 对每个用户查询进行处理
    for query_idx, user_query in enumerate(user_query_list):
        logger.info(f"处理查询 {query_idx + 1}/{len(user_query_list)}: {user_query}")
        
        # if task_type[query_idx] != " Text-Rich Understanding":
        #     continue
        
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
            
            # 格式化交互历史
            history_text = format_interaction_history(interaction_history)
            
            # 构建完整的 prompt
            full_prompt = prompt_template + f"""
------------------------------------------------------------
USER_QUERY
------------------------------------------------------------
{user_query}

------------------------------------------------------------
TASK_TYPE
------------------------------------------------------------
{task_type[query_idx]}

------------------------------------------------------------
CURRENT_5S_CAPTION
------------------------------------------------------------
{caption_text}

------------------------------------------------------------
INTERACTION_HISTORY
------------------------------------------------------------
{history_text}
"""
            
            # 调用 LLM 进行决策
            try:
                response = asyncio.run(llm_config.best_model_func_raw(
                    llm_config.best_model_name,
                    full_prompt,
                    system_prompt=None,
                    history_messages=[],
                ))
                # logger.debug(f"LLM 响应 (second_cap {sec_cap_idx + 1}): {response[:200]}...")
                
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
                                use_minicpm=False,
                                args=args, 
                                ori_query=user_query, 
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
                        continue_prompt = memory_prompt_template + "\n\nRETRIEVED_MEMORY_EVIDENCE\n\n" + retrieved_response
                        
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
        
        # logger.info(f"查询 '{user_query}' 处理完成，共 {len(query_responses)} 个响应")
    
    # 保存历史记录到文件
    history_output_path = checkpoint_path.replace("streaming_checkpoint.json", "retrieval_history.json")
    with open(history_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_query_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"检索历史已保存到: {history_output_path}")
    
    return {
        'accumulated_captions': accumulated_captions,
        'query_results': all_query_results
    }

def streaming_processing(args, video_name, clip_name, clip_duration, task_type, results_dict, user_query_list=None):
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
    videorag = VideoGraphSeparated(llm=gpt5_2_llm_config, working_dir=f"{args.root_path}/{clip_name}_{args.caption_model_name}")     # {args.caption_model_name}
    # videorag.load_caption_model(model_name=args.caption_model_name)

    # 构建 checkpoint 路径
    checkpoint_path = f"{args.root_path}/{clip_name}_{args.caption_model_name}/streaming_checkpoint.json"
    videorag.load_caption_model(model_name="qwenvl_3_8b_instruct")

    if not os.path.exists(checkpoint_path):
        return

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
        datasets_type=args.datasets_type,
        args=args,
        task_type=task_type,
    )

    logger.info(f"检索完成，共处理 {len(user_query_list)} 个查询")

    # 将当前视频的结果添加到全局results_dict中
    if retrieval_result and 'query_results' in retrieval_result:
        results_dict[clip_name] = {
            'video_name': video_name,
            'clip_name': clip_name,
            'clip_duration': clip_duration,
            'task_type': task_type,
            'query_results': retrieval_result['query_results']
        }

    return results_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming caption generation for video frames")
    parser.add_argument("--data_path", type=str, default=os.environ.get("EYEWO_DATA_PATH", "./data/eyewo/estp_bench_sq.json"),
                        help="Base path to video data")
    parser.add_argument("--video_path", type=str, default=os.environ.get("EYEWO_VIDEO_PATH", "./data/eyewo/ESTP-Bench/full_scale_2fps_max384"))
    parser.add_argument("--root_path", type=str, default=os.environ.get("RESULTS_ROOT", "./results") + "/eyewo_results_cor")
    parser.add_argument("--days", type=str, nargs='+', 
                       default=["DAY1"],
                       help="List of days to process")    
    parser.add_argument("--caption_model_name", type=str, default="qwenvl_3_8b_instruct")      # "qwen3_api" "qwenvl_3_8b_instruct"   "minicpm_4_5_v"   gemini_api    gpt_4o_api
    
    parser.add_argument("--caption_retrieval", type=bool, default=True)
    parser.add_argument("--visual_retrieval", type=bool, default=True)
    parser.add_argument("--entity_retrieval", type=bool, default=True)
    parser.add_argument("--need_retrieval", type=bool, default=True)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--filter_captions", type=bool, default=False)
    
    parser.add_argument('--cuda', type=str, default='1',
                       help="CUDA device ID(s) to use. Sets CUDA_VISIBLE_DEVICES environment variable. "
                            "Model will run on cuda:0 (mapped from the specified physical device).")
    
    parser.add_argument('--graph_construction_stage', type=bool, default=False)
    parser.add_argument('--retrieval_stage', type=bool, default=True)
    parser.add_argument('--datasets_type', type=str, default="eyewo")

    args = parser.parse_args()
    
    # 在模型加载之前设置CUDA_VISIBLE_DEVICES，确保模型固定在该设备上运行
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    logger.info(f"Set CUDA_VISIBLE_DEVICES={args.cuda}, model will run on cuda:0 (mapped from physical device {args.cuda})")
    
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
    
    video_data = json.load(open(args.data_path, "r"))
    
    # 用于保存处理时间的数据结构
    processing_times = []
    results_dict = {}
    
    # 已完成结果缓存（用于断点续跑，跳过已处理的 clip/question）；首次运行可不提供，默认为空
    _prev_results_path = os.environ.get("EYEWO_PREV_RESULTS_PATH", "")
    if _prev_results_path and os.path.exists(_prev_results_path):
        all_results = json.load(open(_prev_results_path))
    else:
        all_results = {}

    # ========== Step 1: 对 video_data 中的问题进行去重 (同一 clip 下的相同 question) ==========
    # 同时统一提取 question（兼容 "question" 字段和 conversation[role=user] 两种格式）
    deduped_video_data = {}  # {video_name: {clip_name: [deduped_items]}}
    total_before_dedup = 0
    total_after_dedup = 0
    for video_name, clip_info in video_data.items():
        deduped_video_data[video_name] = {}
        for clip_name, clip_list in clip_info.items():
            seen_questions = set()
            deduped_items = []
            for item in clip_list:
                # 提取 question：优先使用 "question" 字段，否则从 conversation 中取 user 消息
                if "question" in item:
                    question_text = item["question"]
                else:
                    user_msgs = [m for m in item.get("conversation", []) if m.get("role") == "user"]
                    question_text = user_msgs[0]["content"] if user_msgs else None

                total_before_dedup += 1
                if question_text in seen_questions:
                    continue  # 跳过同一 clip 下的重复问题
                seen_questions.add(question_text)
                deduped_items.append(item)
                total_after_dedup += 1
            if deduped_items:
                deduped_video_data[video_name][clip_name] = deduped_items

    logger.info(f"去重完成: 去重前 {total_before_dedup} 个问题, 去重后 {total_after_dedup} 个问题, 去除 {total_before_dedup - total_after_dedup} 个重复")

    # ========== Step 2: 从 all_results 中构建已完成的 (clip_name, question) 集合 ==========
    existing_pairs = set()
    for clip_id, clip_data in all_results.items():
        query_results = clip_data.get("query_results", {})
        for question_text in query_results.keys():
            existing_pairs.add((clip_id, question_text))
    logger.info(f"已有结果: {len(all_results)} 个 clip, {len(existing_pairs)} 个 (clip, question) 对")

    # ========== Step 3: 筛选缺失的 clip/question，只重新检索缺失项 ==========
    total_missing = 0
    total_skipped = 0
    for video_name, clip_info in deduped_video_data.items():
        for clip_name, clip_list in clip_info.items():
            # 提取时间信息（兼容两种字段名）
            first_item = clip_list[0]
            start_time = first_item.get("clip_start_time", first_item.get("start_time", 0))
            end_time = first_item.get("clip_end_time", first_item.get("end_time", 0))
            clip_duration = (start_time, end_time)

            # 计算视频时长（秒）
            video_duration = end_time - start_time

            # 筛选出 all_results 中缺失的问题
            task_type = []
            user_query_list = []
            for item in clip_list:
                if "question" in item:
                    question_text = item["question"]
                else:
                    user_msgs = [m for m in item.get("conversation", []) if m.get("role") == "user"]
                    question_text = user_msgs[0]["content"] if user_msgs else None

                if (clip_name, question_text) in existing_pairs:
                    total_skipped += 1
                    continue  # 已有结果，跳过

                task_type.append(item["Task Type"])
                user_query_list.append(question_text)
                total_missing += 1

            # 如果该 clip 的所有问题都已有结果，跳过
            if not user_query_list:
                # 从 all_results 加载已有结果到 results_dict
                if clip_name in all_results:
                    results_dict[clip_name] = all_results[clip_name]
                logger.info(f"Clip {clip_name} 所有问题已有结果，跳过")
                continue

            logger.info(f"Clip {clip_name} 缺失 {len(user_query_list)} 个问题，开始检索: {user_query_list}")

            # 记录处理开始时间
            processing_start = time.time()

            streaming_processing(args, video_name, clip_name, clip_duration, task_type, results_dict, user_query_list)

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
                "speed_ratio": video_duration / processing_time if processing_time > 0 else 0,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }
            processing_times.append(time_record)

            logger.info(f"视频 {clip_name} 处理完成 - 视频时长: {video_duration:.2f}秒, 处理时间: {processing_time:.2f}秒 ({processing_time/60:.2f}分钟), 速度比: {time_record['speed_ratio']:.2f}x")

    logger.info(f"检索统计: 跳过 {total_skipped} 个已有问题, 重新检索 {total_missing} 个缺失问题")
            
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

    # 将新检索的结果与 all_results 合并后保存
    # all_results 为底，results_dict 中的新结果覆盖/补充上去
    merged_results = dict(all_results)  # 先复制原有结果
    for clip_name, clip_data in results_dict.items():
        if clip_name in merged_results:
            # 该 clip 在 all_results 中已有部分结果，合并 query_results
            existing_qr = merged_results[clip_name].get("query_results", {})
            new_qr = clip_data.get("query_results", {})
            existing_qr.update(new_qr)  # 新结果覆盖/补充到已有结果
            merged_results[clip_name]["query_results"] = existing_qr
        else:
            # 该 clip 是全新的，直接添加
            merged_results[clip_name] = clip_data

    logger.info(f"合并完成: all_results 原有 {len(all_results)} 个 clip, 新增/更新后共 {len(merged_results)} 个 clip")

    if merged_results:
        # 生成输出文件名（包含时间戳）
        results_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results_file = f"all_retrieval_results_{results_timestamp_str}.json"

        try:
            # Use /tmp for temporary file to avoid space issues
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir='/tmp') as tmp_f:
                temp_path = tmp_f.name
                json.dump(merged_results, tmp_f, indent=2, ensure_ascii=False)

            # Move the temp file to final destination
            import shutil
            shutil.move(temp_path, all_results_file)

            logger.info(f"\n{'='*60}")
            logger.info(f"合并后的检索结果已保存到: {all_results_file}")
            logger.info(f"总共包含 {len(merged_results)} 个 clip 的检索结果")
            logger.info(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"保存合并检索结果失败: {e}")
            # Try to save without indentation to reduce size
            try:
                with open(all_results_file, 'w', encoding='utf-8') as f:
                    json.dump(merged_results, f, ensure_ascii=False)
                logger.info(f"合并后的检索结果已保存到（无格式化）: {all_results_file}")
            except Exception as e2:
                logger.error(f"保存合并检索结果失败（第二次尝试）: {e2}")