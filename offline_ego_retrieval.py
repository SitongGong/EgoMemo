import os
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
    embedding_batch_num = 12,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM (we utilize gpt-4o-mini for all experiments)   
    best_model_func_raw = gpt_4o_mini_complete,
    best_model_name = "gpt-5.2", 
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-5.2",     # gpt-5-mini
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16,
)

def egoschema_iterative_retrieval(
    question: str,
    options: List[str],
    checkpoint_path: str,
    llm_config: LLMConfig,
    videorag: VideoGraphSeparated,
    max_rounds: int = 3
) -> Dict:
    """
    实现 EgoSchema 的迭代检索框架
    
    Args:
        question: 问题文本
        options: 选项列表
        checkpoint_path: checkpoint.json 文件路径
        llm_config: LLM 配置
        max_rounds: 最大检索轮数，默认3
    
    Returns:
        包含最终答案和检索历史的字典
    """
    # 读取 checkpoint.json
    logger.info(f"正在读取 checkpoint 文件: {checkpoint_path}")
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        checkpoint_data = json.load(f)
    
    # 获取 min_captions (全局1分钟caption)
    accumulated_captions = checkpoint_data.get('accumulated_captions', {})
    min_captions = accumulated_captions.get('min_captions', [])
    second_captions = accumulated_captions.get('second_captions', [])
    
    if not min_captions:
        logger.warning("未找到 min_captions，使用空列表")
        min_captions = []
    
    # 格式化全局 captions
    global_captions_text = "\n".join([
        f"[{i+1}] Time: {cap.get('time_span', 'N/A')}\n{cap.get('caption', '')}"
        for i, cap in enumerate(min_captions)
    ])
    
    # 格式化选项
    options_text = "\n".join([f"- {opt}" for opt in options])
    
    # 获取 prompt
    prompt_template = EGOSCHEMA_PROMPTS.get("proactive_service_prompt_with_memory_simple", "")
    
    # 初始化检索历史
    retrieved_contexts = []
    retrieval_history = []
    
    # 迭代检索循环
    for round_idx in range(1, max_rounds + 1):
        logger.info(f"开始第 {round_idx}/{max_rounds} 轮检索")
        
        # 构建检索上下文文本
        retrieved_context_text = ""
        if retrieved_contexts:
            retrieved_context_text = "\n\n".join([
                f"[Retrieved {i+1}] {ctx}"
                for i, ctx in enumerate(retrieved_contexts)
            ])
        
        # 构建完整的 prompt
        full_prompt = f"""{prompt_template}

------------------------------------------------------------
QUESTION
------------------------------------------------------------
{question}

------------------------------------------------------------
OPTIONS
------------------------------------------------------------
{options_text}

------------------------------------------------------------
GLOBAL_CAPTIONS
------------------------------------------------------------
{global_captions_text}

------------------------------------------------------------
RETRIEVED_CONTEXT
------------------------------------------------------------
{retrieved_context_text if retrieved_context_text else "None (first round)"}

------------------------------------------------------------
ROUND_INDEX
------------------------------------------------------------
{round_idx}

------------------------------------------------------------
MAX_ROUNDS
------------------------------------------------------------
{max_rounds}
"""
        
        # 调用 LLM 进行决策
        logger.info(f"调用 LLM 进行第 {round_idx} 轮决策...")
        try:
            response = asyncio.run(llm_config.best_model_func_raw(
                llm_config.best_model_name,
                full_prompt,
                system_prompt=None,
                history_messages=[],
            ))
            logger.info(f"LLM 响应: {response[:200]}...")
            
            # 解析 JSON 响应
            cleaned_response = response.strip()
            # 移除可能的 markdown 代码块标记
            if cleaned_response.startswith("```"):
                lines = cleaned_response.split("\n")
                cleaned_response = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned_response
            
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                decision_json = json.loads(json_match.group())
            else:
                logger.error(f"无法从响应中提取 JSON: {response}")
                decision_json = {"decision": "error", "error": response}
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            decision_json = {"decision": "error", "error": str(e)}
        
        # 记录决策
        decision_json['round'] = round_idx
        retrieval_history.append(decision_json)
        
        decision = decision_json.get('decision', '')
        
        # 处理决策
        if decision == 'answer':
            logger.info(f"第 {round_idx} 轮：模型决定直接回答")
            return {
                'answer': decision_json.get('answer', ''),
                'reasoning': decision_json.get('reasoning', ''),
                'round': round_idx,
                'retrieval_history': retrieval_history
            }
        
        elif decision == 'forced_answer':
            logger.info(f"第 {round_idx} 轮：达到最大轮数，强制回答")
            return {
                'answer': decision_json.get('answer', ''),
                'reasoning': decision_json.get('reasoning', ''),
                'round': round_idx,
                'retrieval_history': retrieval_history
            }
        
        elif decision == 'need_retrieval':
            if round_idx >= max_rounds:
                logger.warning(f"第 {round_idx} 轮：需要检索但已达到最大轮数，强制回答")
                # 使用可用的上下文强制回答
                return {
                    'answer': options[0] if options else '',  # 默认选择第一个选项
                    'reasoning': '已达到最大检索轮数，基于可用信息选择',
                    'round': round_idx,
                    'retrieval_history': retrieval_history
                }
            
            retrieval_query = decision_json.get('retrieval_query', '')
            logger.info(f"第 {round_idx} 轮：需要检索，查询: {retrieval_query}")
            
            # 执行检索：使用 streaming_videorag_query 进行检索
            try:
                loop = always_get_an_event_loop()
                query_param = StreamingQueryParam(mode="videorag")
                
                # 判断 caption_model 的类别名称是否包含 MiniCPM
                caption_model_class_name = type(videorag.caption_model).__name__
                use_minicpm = "MiniCPM" in caption_model_class_name
                
                # 使用当前时间戳作为 time_key（从 checkpoint 中获取最后一个 second_caption 的时间）
                time_key = ""
                if second_captions:
                    last_caption = second_captions[-1]
                    time_key = last_caption.get('time_span', '')
                
                retrieved_video_context, retrieved_chunk_context = loop.run_until_complete(
                    streaming_videorag_query(
                        retrieval_query,
                        time_key,  # 使用 time_span 作为 time_key
                        "",  # service_type
                        "",  # sub_service_type
                        "egoschema",  # datasets_type
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
                
                # 将检索到的上下文添加到列表中，继续下一轮迭代
                retrieved_contexts.append(retrieved_response)
                logger.info(f"检索完成，已添加到上下文，继续下一轮迭代")
                    
            except Exception as e:
                logger.error(f"检索失败: {e}")
                # 检索失败，使用空上下文继续
                retrieved_contexts.append(f"检索失败: {str(e)}")
        
        else:
            logger.error(f"未知的决策类型: {decision}")
            if round_idx >= max_rounds:
                return {
                    'answer': options[0] if options else '',
                    'reasoning': '达到最大轮数但决策失败',
                    'round': round_idx,
                    'retrieval_history': retrieval_history
                }
    
    # 如果所有轮次都完成但没有返回答案，返回默认答案
    logger.warning("所有检索轮次完成但未获得答案")
    return {
        'answer': options[0] if options else '',
        'reasoning': '检索完成但未获得明确答案',
        'round': max_rounds,
        'retrieval_history': retrieval_history
    }

def streaming_processing(args, question, video_name, option, question_idx, cor_answer, output_results):
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
    results_root = os.environ.get("RESULTS_ROOT", "./results")
    videorag = VideoGraphSeparated(llm=gpt5_2_llm_config, working_dir=f"{results_root}/egoschema_results_cor/{video_name}_{args.caption_model_name}_{question_idx}")     # {args.caption_model_name}
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
            interval_seconds=3,
            window_seconds=15,
            gap_threshold_seconds=60,
            window_minutes=1,
            window_hours=1/6,     #10分钟
            max_new_tokens=2048,
            datasets_type=args.datasets_type, 
        )
        
    if args.retrieval_stage:
        # 构建 checkpoint 路径
        results_root = os.environ.get("RESULTS_ROOT", "./results")
        checkpoint_path = f"{results_root}/egoschema_results_cor/{video_name}_{args.caption_model_name}_{question_idx}/streaming_checkpoint.json"
        
        # 解析选项（option 可能是字符串或列表）
        if isinstance(option, str):
            # 尝试解析为列表（可能是 JSON 字符串）
            try:
                options_list = json.loads(option)
                if not isinstance(options_list, list):
                    options_list = [option]
            except:
                # 如果不是 JSON，可能是用分隔符分隔的字符串
                options_list = [opt.strip() for opt in option.split(',') if opt.strip()]
                if len(options_list) == 1:
                    options_list = [option]
        elif isinstance(option, list):
            options_list = option
        else:
            options_list = [str(option)]
        
        logger.info(f"开始 EgoSchema 迭代检索")
        logger.info(f"问题: {question}")
        logger.info(f"选项: {options_list}")
        logger.info(f"Checkpoint 路径: {checkpoint_path}")
        
        # 首先应该检查一下记忆的时间的完整性：解析 streaming_checkpoint 中各处的 time_span 是否重复
        def _check_time_span_duplicates(name: str, items: list, key_time_span: str = "time_span") -> bool:
            """若 items 为列表且元素含 time_span，检查是否有重复；返回 True 表示无重复。"""
            if not isinstance(items, list):
                logger.warning(f"[Checkpoint 校验] {name}: 非列表，跳过")
                return True
            time_spans = []
            for el in items:
                if isinstance(el, dict) and key_time_span in el:
                    time_spans.append(el[key_time_span])
                else:
                    ts = getattr(el, key_time_span, None)
                    if ts is not None:
                        time_spans.append(ts)
            if len(time_spans) != len(set(time_spans)):
                dups = {k: c for k, c in Counter(time_spans).items() if c > 1}
                logger.warning(f"[Checkpoint 校验] {name}: 存在重复 time_span: {dups}")
                return False
            logger.info(f"[Checkpoint 校验] {name}: 共 {len(time_spans)} 个元素，time_span 无重复")
            return True

        def _check_dict_time_span_keys(name: str, d: dict) -> bool:
            """captions_dict 的 second_captions/min_captions 的 key 即为 time_span，检查是否有重复 key（理论上不会）。"""
            if not isinstance(d, dict):
                logger.warning(f"[Checkpoint 校验] {name}: 非字典，跳过")
                return True
            keys = list(d.keys())
            if len(keys) != len(set(keys)):
                logger.warning(f"[Checkpoint 校验] {name}: 存在重复 key(time_span): {[k for k in keys if keys.count(k) > 1]}")
                return False
            logger.info(f"[Checkpoint 校验] {name}: 共 {len(keys)} 个 key，无重复")
            return True

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            all_ok = True
            acc = ckpt.get("accumulated_captions") or {}
            all_ok &= _check_time_span_duplicates("accumulated_captions.second_captions", acc.get("second_captions", []))
            all_ok &= _check_time_span_duplicates("accumulated_captions.min_captions", acc.get("min_captions", []))
            ws = ckpt.get("window_states") or {}
            all_ok &= _check_time_span_duplicates("window_states.min_window_second_captions", ws.get("min_window_second_captions", []))
            all_ok &= _check_time_span_duplicates("window_states.hour_window_min_captions", ws.get("hour_window_min_captions", []))
            cd = ckpt.get("captions_dict") or {}
            all_ok &= _check_dict_time_span_keys("captions_dict.second_captions", cd.get("second_captions", {}))
            all_ok &= _check_dict_time_span_keys("captions_dict.min_captions", cd.get("min_captions", {}))
            if not all_ok:
                logger.warning("[Checkpoint 校验] 存在 time_span 重复，建议检查或修复 checkpoint 后再进行检索")
            else:
                logger.info("[Checkpoint 校验] 所有 time_span 检查通过")
        except FileNotFoundError:
            logger.warning(f"[Checkpoint 校验] 文件不存在: {checkpoint_path}，跳过校验")
        except json.JSONDecodeError as e:
            logger.warning(f"[Checkpoint 校验] JSON 解析失败: {e}，跳过校验")
        except Exception as e:
            logger.warning(f"[Checkpoint 校验] 校验过程异常: {e}")

        # 调用迭代检索函数
        retrieval_result = egoschema_iterative_retrieval(
            question=question,
            options=options_list,
            checkpoint_path=checkpoint_path,
            llm_config=gpt5_2_llm_config,
            videorag=videorag,
            max_rounds=3
        )
        output_results[video_name] = retrieval_result
        
        return output_results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming caption generation for video frames")
    parser.add_argument("--data_path", type=str, default=os.environ.get("EGOSCHEMA_DATA_PATH", "./data/egoschema/test-00000-of-00001.parquet"),
                        help="Base path to video data")
    parser.add_argument("--retrieval_path", type=str, default=os.environ.get("EGOSCHEMA_RETRIEVAL_PATH", "./data/egoschema/subset_answers.json"))
    parser.add_argument("--video_path", type=str, default=os.environ.get("EGOSCHEMA_VIDEO_PATH", "./data/egoschema/videos"))
    
    parser.add_argument("--caption_model_name", type=str, default="qwenvl_3_8b_instruct")      # "qwenvl_3_8b_instruct"   "minicpm_4_5_v"   gemini_api    gpt_4o_api
    
    parser.add_argument('--cuda', type=str, default='7',
                       help="CUDA device ID(s) to use. Sets CUDA_VISIBLE_DEVICES environment variable. "
                            "Model will run on cuda:0 (mapped from the specified physical device).")
    
    parser.add_argument('--graph_construction_stage', type=bool, default=False)
    parser.add_argument('--retrieval_stage', type=bool, default=True)
    parser.add_argument('--datasets_type', type=str, default="egoschema")

    args = parser.parse_args()
    
    # 读取 parquet 文件
    import pandas as pd
    logger.info(f"正在读取 parquet 文件: {args.data_path}")
    df = pd.read_parquet(args.data_path)
    logger.info(f"成功读取 parquet 文件，共 {len(df)} 行，{len(df.columns)} 列")
    logger.info(f"列名: {list(df.columns)}")
    
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

    # 配置代理（如果需要）
    # 请根据你的代理服务器设置，取消注释并修改以下配置
    # os.environ["HTTP_PROXY"] = "http://your_proxy:port"
    # os.environ["HTTPS_PROXY"] = "http://your_proxy:port"
    # 或者使用 socks5 代理
    # os.environ["HTTP_PROXY"] = "socks5://127.0.0.1:7890"
    # os.environ["HTTPS_PROXY"] = "socks5://127.0.0.1:7890"
    
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
    
    # for idx, row_dict in enumerate(df.to_dict('records')):、
    _results_root = os.environ.get("RESULTS_ROOT", "./results")
    cor_root = f"{_results_root}/egoschema_results_cor"
    output_results_path = f"{_results_root}/output_results.json"

    # 加载已有的 output_results.json（如果存在）
    output_results = {}
    if os.path.exists(output_results_path):
        try:
            with open(output_results_path, "r", encoding="utf-8") as f:
                output_results = json.load(f)
            logger.info(f"成功加载已有的 output_results，共 {len(output_results)} 条记录")
        except json.JSONDecodeError as e:
            logger.warning(f"加载 output_results.json 失败: {e}，将从空字典开始")
            output_results = {}
        except Exception as e:
            logger.warning(f"读取 output_results.json 时出错: {e}，将从空字典开始")
            output_results = {}
    else:
        logger.info(f"未找到已有的 output_results.json，将创建新文件")

    def _save_output_results():
        with open(output_results_path, "w", encoding="utf-8") as f:
            json.dump(output_results, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存 output_results 至 {output_results_path}")

    for loop_idx, (video_name, cor_answer) in enumerate(retrieval_data.items()):    # 单独处理每一个视频，使用idx区分同一个视频的不同处理实例
        question = video_dict[video_name]["question"]
        option = video_dict[video_name]["option"]
        question_idx = video_dict[video_name]["question_idx"]

        # 检查该视频是否已经处理过
        if video_name in output_results:
            logger.info(f"[{loop_idx + 1}/{total_videos}] {video_name} 已在 output_results 中，跳过处理")
            continue

        # 若该视频在 egoschema_results_cor 中不存在对应目录，则保存当前 output_results 并跳过
        expected_dir = f"{video_name}_{args.caption_model_name}_{question_idx}/streaming_checkpoint.json"
        if not os.path.exists(os.path.join(cor_root, expected_dir)):
            logger.warning(f"[{loop_idx + 1}/{total_videos}] {video_name} 在 egoschema_results_cor 中不存在（目录 {expected_dir}），保存 output_results 并跳过")
            _save_output_results()
            continue
        
        # 记录单个视频处理开始时间
        video_start_time = time.time()
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"[{loop_idx + 1}/{total_videos}] 开始处理视频: {video_name}")
        logger.info(f"问题索引: {question_idx}")
        logger.info(f"{'=' * 80}\n")
        
        try:
            output_results = streaming_processing(args, question, video_name, option, question_idx, cor_answer, output_results)
            
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
    _save_output_results()

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