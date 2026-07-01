import os
import argparse
import glob
import json
import base64
import logging
import random
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
# from videorag.egoschema_prompt import EGOSCHEMA_PROMPTS
from videorag.offline_prompt import OFFLINE_PROMPTS
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
    best_model_name = "o3", 
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-5-mini",     # gpt-5-mini
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16,
)

def _parse_llm_json(response: str) -> dict:
    """从 LLM 响应中提取 JSON 对象，处理 markdown 代码块等包装。"""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return None


def _extract_option_letter(text: str) -> str:
    """
    从回答或正确答案中提取选项字母（A/B/C/D/E）。

    支持格式：
      "A"  "A."  "A. some text"  "(A)"  "Option A"  "a"  "A. "
    返回大写字母，如 "A"；无法提取则返回 None。
    """
    if not text:
        return None
    text = text.strip()
    # 先尝试精确匹配：整个字符串就是一个字母（可能带点/括号）
    m = re.match(r'^[(\s]*([A-Ea-e])[).\s]', text)
    if m:
        return m.group(1).upper()
    # 单独一个字母
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    # 从开头提取 "A." "B:" 等
    m = re.match(r'^([A-Ea-e])[.:\s)\-]', text)
    if m:
        return m.group(1).upper()
    return None


def _single_question_retrieval(
    question: str,
    options: List[str],
    global_captions_text: str,
    second_captions: List[dict],
    is_short_video: bool,
    llm_config: LLMConfig,
    videorag: VideoGraphSeparated,
    max_rounds: int = 2,
    args=None
) -> Dict:
    """
    对单个问题执行检索+回答流程。

    Args:
        question: 单个问题文本
        options: 该问题的选项列表
        global_captions_text: 预先格式化好的全局 captions 文本
        second_captions: second_captions 原始列表（用于检索时获取 time_key）
        is_short_video: True 则直接回答（无检索），False 则走迭代检索
        llm_config: LLM 配置
        videorag: VideoGraphSeparated 实例
        max_rounds: 最大检索轮数

    Returns:
        单个问题的结果字典，包含 answer / reasoning / retrieved_context / retrieval_history 等
    """
    if is_short_video:
        # ----------------------------------------------------------
        # 短视频路径：直接使用所有 second_captions，无需检索
        # ----------------------------------------------------------
        prompt_template = OFFLINE_PROMPTS.get("proactive_service_prompt_without_memory", "")
        full_prompt = f"""{prompt_template}

------------------------------------------------------------
QUESTION
------------------------------------------------------------
{question}

------------------------------------------------------------
OPTIONS
------------------------------------------------------------
{options}

------------------------------------------------------------
GLOBAL_CAPTIONS
------------------------------------------------------------
{global_captions_text}
"""
        try:
            response = asyncio.run(llm_config.best_model_func_raw(
                llm_config.best_model_name,
                full_prompt,
                system_prompt=None,
                history_messages=[],
            ))
            logger.info(f"LLM 响应: {response[:200]}...")
            result_json = _parse_llm_json(response)
            if result_json is None:
                logger.error(f"无法从响应中提取 JSON: {response}")
                result_json = {"answer": "", "reasoning": response}
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            result_json = {"answer": options[0] if options else "", "reasoning": str(e)}

        return {
            'question': question,
            'options': options,
            'answer': result_json.get('answer', ''),
            'reasoning': result_json.get('reasoning', ''),
            'round': 0,
            'retrieved_context': [],
            'retrieval_history': [{"mode": "direct_answer", "second_captions_count": len(second_captions), **result_json}],
        }

    # ----------------------------------------------------------
    # 长视频路径：迭代检索
    # ----------------------------------------------------------
    prompt_template = OFFLINE_PROMPTS.get("proactive_service_prompt_with_memory_simple", "")

    retrieved_contexts = []
    retrieval_history = []

    for round_idx in range(1, max_rounds + 1):
        logger.info(f"  [Q] 开始第 {round_idx}/{max_rounds} 轮检索")

        retrieved_context_text = ""
        if retrieved_contexts:
            retrieved_context_text = "\n\n".join([
                f"[Retrieved {i+1}] {ctx}"
                for i, ctx in enumerate(retrieved_contexts)
            ])

        full_prompt = f"""{prompt_template}

------------------------------------------------------------
QUESTION
------------------------------------------------------------
{question}

------------------------------------------------------------
OPTIONS
------------------------------------------------------------
{options}

------------------------------------------------------------
GLOBAL_CAPTIONS
------------------------------------------------------------
{global_captions_text}

------------------------------------------------------------
RETRIEVED_CONTEXT
------------------------------------------------------------
{retrieved_context_text if retrieved_context_text else "None (first round)"}
"""

        try:
            response = asyncio.run(llm_config.best_model_func_raw(
                llm_config.best_model_name,
                full_prompt,
                system_prompt=None,
                history_messages=[],
            ))
            logger.info(f"  LLM 响应: {response[:200]}...")
            decision_json = _parse_llm_json(response)
            if decision_json is None:
                logger.error(f"  无法从响应中提取 JSON: {response}")
                decision_json = {"decision": "error", "error": response}
        except Exception as e:
            logger.error(f"  LLM 调用失败: {e}")
            decision_json = {"decision": "error", "error": str(e)}

        decision_json['round'] = round_idx
        retrieval_history.append(decision_json)
        decision = decision_json.get('decision', '')

        def _build_result(answer, reasoning):
            return {
                'question': question,
                'options': options,
                'answer': answer,
                'reasoning': reasoning,
                'round': round_idx,
                'retrieved_context': retrieved_contexts.copy(),
                'retrieval_history': retrieval_history,
            }

        if decision in ('answer', 'forced_answer'):
            logger.info(f"  第 {round_idx} 轮：模型决定回答 (decision={decision})")
            return _build_result(
                decision_json.get('answer', ''),
                decision_json.get('reasoning', ''),
            )

        elif decision == 'need_retrieval':
            if round_idx >= max_rounds:
                logger.warning(f"  第 {round_idx} 轮：需要检索但已达到最大轮数，强制回答")
                return _build_result(
                    options[0] if options else '',
                    '已达到最大检索轮数，基于可用信息选择',
                )

            retrieval_query = decision_json.get('retrieval_query', '')
            logger.info(f"  第 {round_idx} 轮：需要检索，查询: {retrieval_query}")

            try:
                loop = always_get_an_event_loop()
                query_param = StreamingQueryParam(mode="videorag")

                caption_model_class_name = type(videorag.caption_model).__name__
                use_minicpm = "MiniCPM" in caption_model_class_name

                time_key = ""
                if second_captions:
                    last_caption = second_captions[-1]
                    time_key = last_caption.get('time_span', '')

                retrieved_video_context, retrieved_chunk_context = loop.run_until_complete(
                    streaming_videorag_query(
                        retrieval_query,
                        time_key,
                        "",  # service_type
                        "",  # sub_service_type
                        args.datasets_type,  # datasets_type
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
                        reconstruct_caption=True, 
                        ori_query=question + " Options: " + options,  
                    )
                )
                retrieved_response = retrieved_video_context + "\n" + retrieved_chunk_context
                logger.info(f"  检索完成，获得上下文长度: {len(retrieved_response)} 字符")
                retrieved_contexts.append(retrieved_response)

            except Exception as e:
                logger.error(f"  检索失败: {e}")
                retrieved_contexts.append(f"检索失败: {str(e)}")

        else:
            logger.error(f"  未知的决策类型: {decision}")
            if round_idx >= max_rounds:
                return _build_result(
                    options[0] if options else '',
                    '达到最大轮数但决策失败',
                )

    logger.warning("  所有检索轮次完成但未获得答案")
    return {
        'question': question,
        'options': options,
        'answer': options[0] if options else '',
        'reasoning': '检索完成但未获得明确答案',
        'round': max_rounds,
        'retrieved_context': retrieved_contexts.copy(),
        'retrieval_history': retrieval_history,
    }


def egoschema_iterative_retrieval(
    questions: list,
    options_list: list,
    checkpoint_path: str,
    llm_config: LLMConfig,
    videorag: VideoGraphSeparated,
    max_rounds: int = 2,
    ground_truth_answers: list = None,
    args=None,
    existing_results: list = None,
) -> Dict:
    """
    对一个视频的多个问题+选项进行迭代检索，返回包含所有问题结果的字典。

    checkpoint / captions 在同一个视频中是共享的，只需加载一次。
    每个问题独立走检索-回答流程。

    Args:
        questions: 问题列表，如 ["What is ...?", "How does ...?"]
        options_list: 与 questions 一一对应的选项列表，如 [["A","B","C"], ["A","B","C"]]
        checkpoint_path: checkpoint.json 文件路径
        llm_config: LLM 配置
        videorag: VideoGraphSeparated 实例
        max_rounds: 最大检索轮数
        ground_truth_answers: 正确答案列表（可选），用于在结果中保存，方便统计正确率

    Returns:
        {
          "video_summary": { total_questions, correct, accuracy, ... },
          "results": [
            {
              "question_index": 0,
              "question": "...",
              "options": [...],
              "answer": "模型回答",
              "ground_truth": "正确答案",
              "is_correct": True/False,
              "reasoning": "推理过程",
              "round": 2,
              "retrieved_context": [...],
              "retrieval_history": [...]
            },
            ...
          ]
        }
    """
    # 读取 checkpoint.json
    logger.info(f"正在读取 checkpoint 文件: {checkpoint_path}")
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        checkpoint_data = json.load(f)

    # 获取 captions（一个视频只加载一次）
    accumulated_captions = checkpoint_data.get('accumulated_captions', {})
    min_captions = accumulated_captions.get('min_captions', [])
    second_captions = accumulated_captions.get('second_captions', [])

    is_short_video = len(second_captions) <= 12

    if is_short_video:
        logger.info(f"second_captions 数量为 {len(second_captions)}（<= 6），使用直接回答模式")
        global_captions_text = "\n".join([
            f"[{i+1}] Time: {cap.get('time_span', 'N/A')}\n{cap.get('caption', '')}"
            for i, cap in enumerate(second_captions)
        ])
    else:
        logger.info(f"second_captions 数量为 {len(second_captions)}（> 6），使用检索模式")
        if not min_captions:
            logger.warning("未找到 min_captions，使用空列表")
            min_captions = []
        global_captions_text = "\n".join([
            f"[{i+1}] Time: {cap.get('time_span', 'N/A')}\n{cap.get('caption', '')}"
            for i, cap in enumerate(min_captions)
        ])

    # 兼容单问题调用：将单个 question/options 包装成列表
    if isinstance(questions, str):
        questions = [questions]
    # if isinstance(options_list, list) and options_list and not isinstance(options_list[0], list):
    #     options_list = [options_list]
    if ground_truth_answers is not None and not isinstance(ground_truth_answers, list):
        ground_truth_answers = [ground_truth_answers]

    # 判断单条结果是否为错误/无效结果（API 报错伪装成正常回答）
    _ERROR_PATTERNS = ("RetryError", "RateLimitError", "APIError", "TimeoutError", "ConnectionError", "检索失败")

    def _is_error_result(r: dict) -> bool:
        """检查结果是否包含错误标志（reasoning 或 error 字段含有已知错误关键词）"""
        for field in ("reasoning", "error"):
            val = r.get(field)
            if val and isinstance(val, str):
                for pat in _ERROR_PATTERNS:
                    if pat in val:
                        return True
        return False

    # 构建已有结果的索引（按 question_index 查找）
    existing_by_idx = {}
    if existing_results:
        for r in existing_results:
            q_idx_key = r.get('question_index')
            if q_idx_key is not None and 'answer' in r and r.get('is_correct') is not None:
                if _is_error_result(r):
                    logger.info(f"  跳过错误结果 question_index={q_idx_key}: {r.get('reasoning', r.get('error', ''))[:80]}")
                    continue
                existing_by_idx[q_idx_key] = r

    # 逐个问题处理
    all_results = []
    correct_count = 0

    for q_idx, (question, options) in enumerate(zip(questions, options_list)):
        # 跳过已有成功结果的问题
        if q_idx in existing_by_idx:
            cached = existing_by_idx[q_idx]
            logger.info(f"========== 问题 {q_idx + 1}/{len(questions)} [已有结果，跳过] ==========")
            logger.info(f"  Q: {question}")
            logger.info(f"  Cached Answer: {cached.get('answer')} | Correct: {cached.get('is_correct')}")
            all_results.append(cached)
            if cached.get('is_correct'):
                correct_count += 1
            continue

        logger.info(f"========== 问题 {q_idx + 1}/{len(questions)} ==========")
        logger.info(f"Q: {question}")
        logger.info(f"Options: {options}")

        options_text = "\n".join([f"{opt}: {text}" for opt, text in options.items()])

        result = _single_question_retrieval(
            question=question,
            options=options_text,
            global_captions_text=global_captions_text,
            second_captions=second_captions,
            is_short_video=is_short_video,
            llm_config=llm_config,
            videorag=videorag,
            max_rounds=max_rounds,
            args=args, 
        )

        # 附加元信息
        result['question_index'] = q_idx

        # 如果提供了正确答案，计算是否正确
        if ground_truth_answers is not None and q_idx < len(ground_truth_answers):
            gt = ground_truth_answers[q_idx]
            result['ground_truth'] = gt
            # 提取选项字母进行比较（如 "A. xxx" -> "A", "B" -> "B"）
            pred_letter = _extract_option_letter(str(result.get('answer', '')))
            gt_letter = _extract_option_letter(str(gt))
            result['pred_letter'] = pred_letter
            result['gt_letter'] = gt_letter
            result['is_correct'] = (pred_letter is not None
                                    and gt_letter is not None
                                    and pred_letter == gt_letter)
            if result['is_correct']:
                correct_count += 1
        else:
            result['ground_truth'] = None
            result['is_correct'] = None

        logger.info(f"  Answer: {result['answer']} | GT: {result.get('ground_truth')} | Correct: {result.get('is_correct')}")
        all_results.append(result)

    # 汇总统计
    total_q = len(all_results)
    summary = {
        'total_questions': total_q,
        'correct': correct_count,
        'accuracy': correct_count / total_q if total_q > 0 else 0.0,
        'mode': 'direct_answer' if is_short_video else 'retrieval',
        'second_captions_count': len(second_captions),
    }
    logger.info(f"视频汇总: {total_q} 个问题, 正确 {correct_count}, 准确率 {summary['accuracy']:.2%}")

    return {
        'video_summary': summary,
        'results': all_results,
    }

def streaming_processing(args, video_name, video_data, existing_results=None):
    """
    对一个视频的所有问题进行检索+回答。

    Args:
        args: Command line arguments
        video_name: 视频名称
        video_data: 视频数据字典
            - egotaskqa: {"question": [q1, q2, ...], "options": [opts1, opts2, ...], "answer": [a1, a2, ...]}
            - qaego4d:   {"question": str, "options": [5 items], "answer": str, ...}
        existing_results: 该视频之前已有的 results 列表（用于跳过已处理的问题）

    Returns:
        egoschema_iterative_retrieval 返回的结果字典
    """
    # ------ 统一成 questions / options_list / answers 列表形式 ------
    if args.datasets_type == "qaego4d":
        raw_options = video_data["options"]
        options_list = [{"A.": raw_options[0], 
                         "B.": raw_options[1],
                         "C.": raw_options[2], 
                         "D.": raw_options[3]}]
        questions = [video_data["question"]]
        answers = [video_data["answer"]]
    elif args.datasets_type == "egotaskqa":
        questions = video_data["question"]       # 已经是列表
        options_list = video_data["options"]      # 已经是列表的列表
        answers = video_data["answer"]            # 已经是列表
    else:
        questions = video_data.get("question", [])
        options_list = video_data.get("options", [])
        answers = video_data.get("answer", [])
        if isinstance(questions, str):
            questions = [questions]
            options_list = [options_list]
            answers = [answers] if not isinstance(answers, list) else answers

    # ------ 加载 VideoRAG ------
    video_name_clean = video_name.split(".")[0]
    results_root = os.environ.get("RESULTS_ROOT", "./results")
    working_dir = f"{results_root}/{args.datasets_type}_cor/{video_name_clean}_{args.caption_model_name}"
    videorag = VideoGraphSeparated(llm=gpt5_2_llm_config, working_dir=working_dir)
    videorag.load_caption_model(model_name=args.caption_model_name)

    checkpoint_path = os.path.join(working_dir, "streaming_checkpoint.json")

    # ------ 调用多问题检索 ------
    result = egoschema_iterative_retrieval(
        questions=questions,
        options_list=options_list,
        checkpoint_path=checkpoint_path,
        llm_config=gpt5_2_llm_config,
        videorag=videorag,
        max_rounds=2,
        ground_truth_answers=answers,
        args=args,
        existing_results=existing_results,
    )

    return result
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming caption generation for video frames")
    parser.add_argument("--data_path", type=str, default=os.environ.get("EGOSCHEMA_DATA_PATH", "./data/egoschema/test-00000-of-00001.parquet"),
                        help="Base path to video data")
    parser.add_argument("--retrieval_path", type=str, default=os.environ.get("EGOSCHEMA_RETRIEVAL_PATH", "./data/egoschema/subset_answers.json"))

    parser.add_argument("--egotaskqa_path", type=str, default=os.environ.get("EGOTASKQA_VIDEO_PATH", "./data/egotaskqa/qa_videos"))
    parser.add_argument("--egotaskqa_anno", type=str, default=os.environ.get("EGOTASKQA_ANNO_PATH", "./data/egotaskqa/egotaskqa_test.json"))
    parser.add_argument("--qaego4d_path", type=str, default=os.environ.get("QAEGO4D_VIDEO_PATH", "./data/qaego4d/Ego4D"))
    parser.add_argument("--qaego4d_anno", type=str, default=os.environ.get("QAEGO4D_ANNO_PATH", "./data/qaego4d/annotations.QaEgo4D_test_close.json"))
    
    parser.add_argument("--caption_retrieval", type=bool, default=True)
    parser.add_argument("--visual_retrieval", type=bool, default=True)
    parser.add_argument("--entity_retrieval", type=bool, default=True)
    parser.add_argument("--need_retrieval", type=bool, default=True)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--filter_captions", type=bool, default=False)
    
    parser.add_argument("--caption_model_name", type=str, default="qwenvl_3_8b_instruct")      # "qwenvl_3_8b_instruct"   "minicpm_4_5_v"   gemini_api    gpt_4o_api
    
    parser.add_argument('--cuda', type=str, default='0',
                       help="CUDA device ID(s) to use. Sets CUDA_VISIBLE_DEVICES environment variable. "
                            "Model will run on cuda:0 (mapped from the specified physical device).")
    
    parser.add_argument('--graph_construction_stage', type=bool, default=True)
    parser.add_argument('--retrieval_stage', type=bool, default=False)
    parser.add_argument('--datasets_type', type=str, default="qaego4d")     # "egotaskqa" "qaego4d"

    args = parser.parse_args()
    
    # 在模型加载之前设置CUDA_VISIBLE_DEVICES，确保模型固定在该设备上运行
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    logger.info(f"Set CUDA_VISIBLE_DEVICES={args.cuda}, model will run on cuda:0 (mapped from physical device {args.cuda})")
    
    
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
    
    # 统计变量
    processing_times = []
    overall_start_time = time.time()
    
    # # 读取 subset_answers.json 文件
    # with open(args.egotaskqa_anno, "r", encoding="utf-8") as f:
    #     retrieval_data = json.load(f)
        
    # with open(args.qaego4d_anno, "r", encoding="utf-8") as f:
    #     retrieval_data_ = json.load(f)
    
    if args.datasets_type == "egotaskqa":
        with open(args.egotaskqa_anno, "r", encoding="utf-8") as f:
            retrieval_data = json.load(f)
        video_dict = dict()
        for video_data in retrieval_data:
            vname = video_data["video_path"]
            if vname not in video_dict:
                video_dict[vname] = {"question": [], "options": [], "answer": []}
            video_dict[vname]["question"].append(video_data["q"])
            video_dict[vname]["options"].append(video_data["option"])
            video_dict[vname]["answer"].append(video_data["a"])                                       
        
    elif args.datasets_type == "qaego4d":
        with open(args.qaego4d_anno, "r", encoding="utf-8") as f:
            retrieval_data = json.load(f)
        video_dict = dict()
        option_dict = {0: "A. ", 1: "B. ", 2: "C. ", 3: "D. "}
        for num, video_data in enumerate(retrieval_data):
            # 打乱选项顺序
            options = video_data["wrong_answers"]
            options.append(video_data["answer"])
            random.shuffle(options)
            video_data_ = {"video_id": video_data["video_id"],
                           "sample_id": video_data["sample_id"],
                           "answer": option_dict[options.index(video_data["answer"])], # video_data["answer"],
                           "question": video_data["question"],
                           "options": options,
                           "video_time_span": (video_data["video_start_sec"], video_data["video_end_sec"]),
                           "moment_time_span": (video_data["moment_start_frame"], video_data["moment_end_frame"])
                           }
            
            video_dict[video_data["video_uid"] + "_" + str(num)] = video_data_
            
    # ================================================================
    # 结果收集与保存
    # ================================================================
    output_results = {}  # video_name -> 该视频的完整结果
    output_results_path = f"{os.environ.get('RESULTS_ROOT', './results')}/output_results_{args.datasets_type}_o3.json"

    # 尝试加载已有结果（支持断点续跑）
    if os.path.exists(output_results_path):
        try:
            with open(output_results_path, "r", encoding="utf-8") as f:
                output_results = json.load(f)
            logger.info(f"加载已有结果 {len(output_results)} 条: {output_results_path}")
        except Exception as e:
            logger.warning(f"加载已有结果失败: {e}，从空字典开始")
            output_results = {}

    def _save_output_results():
        """保存结果并附带全局准确率统计"""
        # 计算全局统计
        global_correct = 0
        global_total = 0
        for vname, vresult in output_results.items():
            if vname == "__global_summary__":
                continue
            summary = vresult.get('video_summary', {})
            global_correct += summary.get('correct', 0)
            global_total += summary.get('total_questions', 0)

        output_results["__global_summary__"] = {
            "total_videos": len([k for k in output_results if k != "__global_summary__"]),
            "total_questions": global_total,
            "total_correct": global_correct,
            "accuracy": global_correct / global_total if global_total > 0 else 0.0,
        }

        with open(output_results_path, "w", encoding="utf-8") as f:
            json.dump(output_results, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存结果至 {output_results_path} "
                     f"(问题数={global_total}, 正确={global_correct}, "
                     f"准确率={output_results['__global_summary__']['accuracy']:.2%})")

    total_videos = len(video_dict)
    for loop_idx, (video_name, video_data) in enumerate(video_dict.items()):

        # 跳过已成功处理的视频（断点续跑）；有 error 或部分问题缺失的会重新处理
        prev_results = None  # 该视频之前已有的单题结果列表
        if video_name in output_results:
            existing = output_results[video_name]
            if "video_summary" in existing and "error" not in existing:
                # 检查该视频的所有问题是否都已有结果
                prev_result_list = existing.get("results", [])
                total_expected = existing["video_summary"].get("total_questions", 0)
                answered_count = sum(1 for r in prev_result_list if r.get("is_correct") is not None)
                if answered_count >= total_expected and total_expected > 0:
                    logger.info(f"[{loop_idx+1}/{total_videos}] {video_name} 所有 {total_expected} 个问题已有结果，跳过")
                    continue
                else:
                    logger.info(f"[{loop_idx+1}/{total_videos}] {video_name} 已有 {answered_count}/{total_expected} 个问题结果，补充处理")
                    prev_results = prev_result_list
            else:
                logger.info(f"[{loop_idx+1}/{total_videos}] {video_name} 之前处理失败，重新处理")

        video_start_time = time.time()
        logger.info(f"[{loop_idx+1}/{total_videos}] 开始处理: {video_name}")

        try:
            result = streaming_processing(args, video_name, video_data, existing_results=prev_results)
            output_results[video_name] = result

            video_processing_time = time.time() - video_start_time
            processing_times.append(video_processing_time)

            vs = result.get('video_summary', {})
            logger.info(
                f"[{loop_idx+1}/{total_videos}] {video_name} 完成 | "
                f"耗时 {video_processing_time:.1f}s | "
                f"问题数 {vs.get('total_questions',0)} | "
                f"正确 {vs.get('correct',0)} | "
                f"准确率 {vs.get('accuracy',0):.2%}"
            )

            # 每处理 5 个视频自动保存一次
            if (loop_idx + 1) % 5 == 0:
                _save_output_results()

        except Exception as e:
            video_processing_time = time.time() - video_start_time
            logger.error(f"[{loop_idx+1}/{total_videos}] {video_name} 失败: {e} (耗时 {video_processing_time:.1f}s)")
            output_results[video_name] = {"error": str(e)}

    # 最终保存
    _save_output_results()

    # 处理完所有视频后的总结
    total_processing_time = time.time() - overall_start_time
    gs = output_results.get("__global_summary__", {})

    logger.info(f"\n{'=' * 80}")
    logger.info(f"所有视频处理完成!")
    logger.info(f"{'=' * 80}")
    logger.info(f"总体统计:")
    logger.info(f"  - 总耗时: {total_processing_time:.2f} 秒 ({total_processing_time / 60:.2f} 分钟)")
    logger.info(f"  - 视频数: {gs.get('total_videos', 0)}")
    logger.info(f"  - 总问题数: {gs.get('total_questions', 0)}")
    logger.info(f"  - 总正确数: {gs.get('total_correct', 0)}")
    logger.info(f"  - 总准确率: {gs.get('accuracy', 0):.2%}")
    if processing_times:
        logger.info(f"  - 平均耗时: {sum(processing_times) / len(processing_times):.2f} 秒/视频")
        logger.info(f"  - 最快: {min(processing_times):.2f} 秒")
        logger.info(f"  - 最慢: {max(processing_times):.2f} 秒")
    logger.info(f"  - 结果文件: {output_results_path}")
    logger.info(f"{'=' * 80}\n")