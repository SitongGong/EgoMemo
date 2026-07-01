import os
import argparse
import json
import logging
import re
import time
from typing import List, Dict
from dataclasses import asdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if __name__ == "__main__":
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument('--cuda', type=str, default='1')
    _pre_args, _ = _pre_parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.cuda

from videorag.egograph_retrieval_optimize_ import VideoGraphSeparated
from videorag._llm import *
from videorag.ego_op import streaming_videorag_query
from videorag.streaming_op import QueryParam as StreamingQueryParam
from videorag._utils import always_get_an_event_loop
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# Task type classification
# ================================================================
# Real-Time Visual Perception: directly answer using recent captions
REALTIME_TASKS = {"STU", "OJR", "ATR", "ACR", "OCR", "FPD"}
# Backward Tracing: may need retrieval from full video history
BACKWARD_TASKS = {"EPM", "ASI", "HLD"}

# ================================================================
# LLM Config
# ================================================================
gpt5_2_llm_config1 = LLMConfig(
    embedding_func_raw = bge_m3_embedding,
    embedding_model_name = "BAAI/bge-m3",
    embedding_dim = 1024,  # bge-m3的嵌入维度
    embedding_max_token_size = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    best_model_func_raw=gpt_4o_mini_complete,
    best_model_name="gpt-5-mini",
    best_model_max_token_size=32768,
    best_model_max_async=16,

    cheap_model_func_raw=gpt_4o_mini_complete,
    cheap_model_name="gpt-4o-mini",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16,
)

gpt5_2_llm_config2 = LLMConfig(
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

# ================================================================
# Prompt templates
# ================================================================

OVOBENCH_REALTIME_PROMPT = """
You are an egocentric video question-answering assistant
designed for the OVO-Bench real-time visual perception tasks.

This is a TRAINING-FREE, DIRECT-ANSWER reasoning process.

You are given real-time egocentric video captions:
- RECENT_CAPTIONS: detailed second-level captions from the LAST 10 SECONDS of the video (the moment the question is asked).
- GLOBAL_CAPTIONS: minute-level summaries covering the entire video up to the current moment.

You must answer the question directly based on the provided captions.
No retrieval is allowed.

------------------------------------------------------------
Task Types You Handle
------------------------------------------------------------
- STU (Spatial-Temporal Understanding): spatial layout, positions, temporal order of visible events
- OJR (Object Relationship): relationships between visible objects/people
- ATR (Attribute Recognition): colors, shapes, sizes, materials of visible objects
- ACR (Action Recognition): what actions are currently happening or just happened
- OCR (Text Recognition): reading text, signs, labels visible in the scene
- FPD (Future Prediction): predicting what will happen next based on current visual context

------------------------------------------------------------
Inputs
------------------------------------------------------------

(1) QUESTION
A natural-language question about the egocentric video at the current moment.

(2) OPTIONS
Multiple-choice options. Exactly ONE is correct.

(3) RECENT_CAPTIONS
Detailed second-level captions from the last ~10 seconds of the video.
These contain the most relevant visual information for real-time perception.

(4) GLOBAL_CAPTIONS
Minute-level summaries of the entire video up to the current moment.
Use these for broader context if needed.

------------------------------------------------------------
Reasoning Requirements
------------------------------------------------------------

- Focus primarily on RECENT_CAPTIONS for real-time perception questions.
- Use GLOBAL_CAPTIONS for additional context when needed.
- Ground all reasoning in the provided captions.
- Do NOT invent objects, actions, or states not mentioned in captions.
- Choose the option most consistent with available evidence.

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

{
  "answer": "<exact option text>",
  "reasoning": "<concise explanation grounded in captions>"
}

Output ONLY the JSON object. No additional text.
"""

OVOBENCH_BACKWARD_RETRIEVAL_PROMPT = """
You are an egocentric video question-answering decision assistant
designed for the OVO-Bench backward tracing tasks.

This is a TRAINING-FREE, ITERATIVE decision-and-retrieval process.
At most TWO retrieval rounds are allowed.

You are given:
- GLOBAL_CAPTIONS: minute-level summaries covering the entire video.
- RETRIEVED_CONTEXT: additional retrieved captions from previous rounds (may be empty).

------------------------------------------------------------
Task Types You Handle
------------------------------------------------------------
- EPM (Episodic Memory): questions about past events, who/what/where in earlier parts of the video
- ASI (Action Sequence Inference): reasoning about sequences of actions that happened earlier
- HLD (Historical Location/Detail): tracking where objects were or what happened at earlier times

------------------------------------------------------------
Inputs
------------------------------------------------------------

(1) QUESTION
A natural-language question requiring backward tracing through the video.

(2) OPTIONS
Multiple-choice options. Exactly ONE is correct.

(3) GLOBAL_CAPTIONS
Minute-level summaries of the entire video.

(4) RETRIEVED_CONTEXT (OPTIONAL)
Additional retrieved captions from previous retrieval rounds. May be empty.

------------------------------------------------------------
Decision Rules
------------------------------------------------------------

CASE 1 — RETRIEVED_CONTEXT is EMPTY (first round)

You may ANSWER immediately ONLY IF:
- The answer can be clearly determined from GLOBAL_CAPTIONS alone.
- Exactly ONE option is consistent with the evidence.

Otherwise → request retrieval.

CASE 2 — RETRIEVED_CONTEXT EXISTS (after retrieval)

If this is the second retrieval round, you MUST answer.
If this is the first retrieval round, you may request one more retrieval OR answer.

When answering after retrieval:
- Choose the option most consistent with all available evidence.
- Do NOT invent new evidence.

------------------------------------------------------------
Retrieval Query Requirements
------------------------------------------------------------

If requesting retrieval:
- Output ONE concise English sentence as the retrieval query.
- The query MUST specify:
  • The temporal segment or event needed
  • The specific object/person/action to look for
  • Include answer options in parentheses for context

------------------------------------------------------------
Output Format (STRICT)
------------------------------------------------------------

Case 1 — Answer:
{
  "decision": "answer",
  "answer": "<exact option text>",
  "reasoning": "<concise explanation grounded in captions>"
}

Case 2 — Need Retrieval:
{
  "decision": "need_retrieval",
  "retrieval_query": "<one-sentence retrieval query including options>"
}

Output ONLY the JSON object. No additional text.
"""

# ================================================================
# Utility functions
# ================================================================

def _parse_llm_json(response: str) -> dict:
    """Extract JSON object from LLM response, handling markdown code blocks."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return None


def _extract_option_letter(text: str, options: list = None) -> str:
    """
    Extract option letter (A/B/C/D/E) from answer text.

    Handles multiple cases:
      1. Direct letter: "A", "B.", "(C)", "A. xxx"
      2. Full option text: matches against the options list
      3. Partial match: answer contains or is contained by an option
    """
    if not text:
        return None
    text = text.strip()
    labels = "ABCDEFGHIJ"

    # Case 1: answer starts with an option letter
    m = re.match(r'^[(\s]*([A-Ea-e])[).\s:,\-]', text)
    if m:
        return m.group(1).upper()
    if len(text) == 1 and text.upper() in labels:
        return text.upper()
    m = re.match(r'^([A-Ea-e])[.:\s)\-]', text)
    if m:
        return m.group(1).upper()

    # Case 2: answer matches option text (with or without "A." prefix)
    if options:
        text_lower = text.lower().strip()
        # Strip leading label like "A. ", "A: ", "(A) " from the answer
        text_clean = re.sub(r'^[(\s]*[A-Ea-e][).\s:,\-]+\s*', '', text_lower).strip()

        for i, opt in enumerate(options):
            if i >= len(labels):
                break
            opt_lower = opt.lower().strip()
            # Exact match (option text == answer text)
            if text_lower == opt_lower or text_clean == opt_lower:
                return labels[i]
            # Answer starts with "A. option_text" pattern
            prefixed = f"{labels[i].lower()}. {opt_lower}"
            if text_lower == prefixed or text_lower.startswith(prefixed):
                return labels[i]

        # Case 3: fuzzy - answer is contained in option or option is contained in answer
        for i, opt in enumerate(options):
            if i >= len(labels):
                break
            opt_lower = opt.lower().strip()
            if len(opt_lower) > 3 and (opt_lower in text_lower or text_clean in opt_lower):
                return labels[i]

    return None


def _time_span_to_seconds(time_span: str) -> float:
    """Parse time_span like '1-00:03:20-00:03:30' and return end time in seconds."""
    parts = time_span.split("-", 1)
    if len(parts) < 2:
        return 0.0
    time_part = parts[1]  # '00:03:20-00:03:30'
    times = time_part.split("-")
    if len(times) < 2:
        return 0.0
    end_time_str = times[1]  # '00:03:30'
    h, m, s = end_time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _get_last_n_seconds_captions(second_captions: list, n_seconds: int = 10) -> list:
    """Get second_captions from the last n_seconds of the video."""
    if not second_captions:
        return []
    # Find the end time of the last caption
    last_end = _time_span_to_seconds(second_captions[-1].get('time_span', ''))
    threshold = last_end - n_seconds
    result = []
    for cap in second_captions:
        end_time = _time_span_to_seconds(cap.get('time_span', ''))
        if end_time > threshold:
            result.append(cap)
    return result if result else [second_captions[-1]]


def _format_captions(captions: list) -> str:
    """Format a list of caption dicts into text."""
    return "\n".join([
        f"[{i+1}] Time: {cap.get('time_span', 'N/A')}\n{cap.get('caption', '')}"
        for i, cap in enumerate(captions)
    ])


def _format_options(options: list, gt_index: int = None) -> str:
    """Format options list into labeled text (A. xxx, B. xxx, ...)."""
    labels = "ABCDEFGHIJ"
    parts = []
    for i, opt in enumerate(options):
        label = labels[i] if i < len(labels) else str(i)
        parts.append(f"{label}. {opt}")
    return "\n".join(parts)


def _gt_to_letter(gt_index: int) -> str:
    """Convert ground truth index to option letter."""
    labels = "ABCDEFGHIJ"
    if gt_index is not None and 0 <= gt_index < len(labels):
        return labels[gt_index]
    return None


# ================================================================
# Real-time task: direct answer
# ================================================================

def _handle_realtime_task(
    question: str,
    options_text: str,
    recent_captions_text: str,
    global_captions_text: str,
    llm_config: LLMConfig,
) -> dict:
    """Handle real-time visual perception tasks with direct answering."""

    full_prompt = f"""{OVOBENCH_REALTIME_PROMPT}

------------------------------------------------------------
QUESTION
------------------------------------------------------------
{question}

------------------------------------------------------------
OPTIONS
------------------------------------------------------------
{options_text}

------------------------------------------------------------
RECENT_CAPTIONS (last ~10 seconds)
------------------------------------------------------------
{recent_captions_text}

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
        logger.info(f"  LLM response: {response[:200]}...")
        result_json = _parse_llm_json(response)
        if result_json is None:
            logger.error(f"  Failed to parse JSON from response: {response}")
            result_json = {"answer": "", "reasoning": response}
    except Exception as e:
        logger.error(f"  LLM call failed: {e}")
        result_json = {"answer": "", "reasoning": str(e)}

    return {
        'answer': result_json.get('answer', ''),
        'reasoning': result_json.get('reasoning', ''),
        'round': 0,
        'mode': 'realtime_direct',
        'retrieved_context': [],
        'retrieval_history': [{"mode": "realtime_direct", **result_json}],
    }


# ================================================================
# Backward tracing task: iterative retrieval
# ================================================================

def _handle_backward_task(
    question: str,
    options_text: str,
    global_captions_text: str,
    second_captions: list,
    llm_config: LLMConfig,
    videorag: VideoGraphSeparated,
    max_rounds: int = 2,
    args=None,
) -> dict:
    """Handle backward tracing tasks with iterative retrieval."""

    retrieved_contexts = []
    retrieval_history = []

    for round_idx in range(1, max_rounds + 1):
        logger.info(f"    [Backward] Round {round_idx}/{max_rounds}")

        retrieved_context_text = ""
        if retrieved_contexts:
            retrieved_context_text = "\n\n".join([
                f"[Retrieved {i+1}] {ctx}"
                for i, ctx in enumerate(retrieved_contexts)
            ])

        full_prompt = f"""{OVOBENCH_BACKWARD_RETRIEVAL_PROMPT}

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
"""
        try:
            response = asyncio.run(llm_config.best_model_func_raw(
                llm_config.best_model_name,
                full_prompt,
                system_prompt=None,
                history_messages=[],
            ))
            logger.info(f"    LLM response: {response[:200]}...")
            decision_json = _parse_llm_json(response)
            if decision_json is None:
                logger.error(f"    Failed to parse JSON: {response}")
                decision_json = {"decision": "error", "error": response}
        except Exception as e:
            logger.error(f"    LLM call failed: {e}")
            decision_json = {"decision": "error", "error": str(e)}

        decision_json['round'] = round_idx
        retrieval_history.append(decision_json)
        decision = decision_json.get('decision', '')

        def _build_result(answer, reasoning):
            return {
                'answer': answer,
                'reasoning': reasoning,
                'round': round_idx,
                'mode': 'backward_retrieval',
                'retrieved_context': retrieved_contexts.copy(),
                'retrieval_history': retrieval_history,
            }

        if decision in ('answer', 'forced_answer'):
            logger.info(f"    Round {round_idx}: model decided to answer (decision={decision})")
            return _build_result(
                decision_json.get('answer', ''),
                decision_json.get('reasoning', ''),
            )

        elif decision == 'need_retrieval':
            if round_idx >= max_rounds:
                logger.warning(f"    Round {round_idx}: needs retrieval but max rounds reached, forcing answer")
                return _build_result(
                    decision_json.get('answer', options_text.split('\n')[0] if options_text else ''),
                    'Max retrieval rounds reached, answering with available info',
                )

            retrieval_query = decision_json.get('retrieval_query', '')
            logger.info(f"    Round {round_idx}: needs retrieval, query: {retrieval_query}")

            try:
                loop = always_get_an_event_loop()
                query_param = StreamingQueryParam(mode="videorag")

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
                        ori_query=question + " Options: " + options_text,
                    )
                )
                retrieved_response = retrieved_video_context + "\n" + retrieved_chunk_context
                logger.info(f"    Retrieval done, context length: {len(retrieved_response)} chars")
                retrieved_contexts.append(retrieved_response)

            except Exception as e:
                logger.error(f"    Retrieval failed: {e}")
                retrieved_contexts.append(f"Retrieval failed: {str(e)}")

        else:
            logger.error(f"    Unknown decision type: {decision}")
            if round_idx >= max_rounds:
                return _build_result(
                    options_text.split('\n')[0] if options_text else '',
                    'Max rounds reached with unknown decision',
                )

    logger.warning("    All retrieval rounds completed without answer")
    return {
        'answer': options_text.split('\n')[0] if options_text else '',
        'reasoning': 'Retrieval completed without clear answer',
        'round': max_rounds,
        'mode': 'backward_retrieval',
        'retrieved_context': retrieved_contexts.copy(),
        'retrieval_history': retrieval_history,
    }


# ================================================================
# Main per-question processing
# ================================================================

def process_single_question(
    video_data: dict,
    args,
) -> dict:
    """
    Process a single OVO-Bench question.

    Routes to real-time or backward tracing based on task type.

    Args:
        video_data: single question dict from ovo_bench_filtered.json
        args: command line arguments

    Returns:
        Result dict with answer, reasoning, is_correct, etc.
    """
    question_id = video_data["id"]
    task = video_data["task"]
    question = video_data["question"]
    options = video_data["options"]
    gt_index = video_data["gt"]

    options_text = _format_options(options)
    gt_letter = _gt_to_letter(gt_index)

    # Load checkpoint
    video_name_clean = str(question_id)
    results_root = os.environ.get("RESULTS_ROOT", "./results")
    working_dir = f"{results_root}/{args.datasets_type}_cor/{video_name_clean}_{args.caption_model_name}"

    if not os.path.exists(working_dir):
        logger.warning(f"  Working dir not found, skipping: {working_dir}")
        return {
            'question_id': question_id,
            'task': task,
            'question': question,
            'options': options,
            'answer': '',
            'reasoning': f'Working dir not found: {working_dir}',
            'gt': gt_index,
            'gt_letter': gt_letter,
            'pred_letter': None,
            'is_correct': None,
            'error': 'working_dir_not_found',
        }

    checkpoint_path = os.path.join(working_dir, "streaming_checkpoint.json")

    if not os.path.exists(checkpoint_path):
        logger.error(f"  Checkpoint not found: {checkpoint_path}")
        return {
            'question_id': question_id,
            'task': task,
            'question': question,
            'options': options,
            'answer': '',
            'reasoning': f'Checkpoint not found: {checkpoint_path}',
            'gt': gt_index,
            'gt_letter': gt_letter,
            'pred_letter': None,
            'is_correct': None,
            'error': 'checkpoint_not_found',
        }

    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        checkpoint_data = json.load(f)

    accumulated_captions = checkpoint_data.get('accumulated_captions', {})
    second_captions = accumulated_captions.get('second_captions', [])
    min_captions = accumulated_captions.get('min_captions', [])

    global_captions_text = _format_captions(min_captions)

    # 根据 vdb_chunks.json 的 embedding_dim 选择对应的 LLM config
    vdb_chunks_path = os.path.join(working_dir, "vdb_chunks.json")
    gpt5_2_llm_config = gpt5_2_llm_config1  # 默认 bge-m3 (dim=1024)
    if os.path.exists(vdb_chunks_path):
        with open(vdb_chunks_path, 'r') as f:
            vdb_meta = json.load(f)
        embed_dim = vdb_meta.get("embedding_dim", 1024)
        if embed_dim == 1536:
            gpt5_2_llm_config = gpt5_2_llm_config2
            logger.info(f"  embedding_dim=1536, using gpt5_2_llm_config2 (text-embedding-3-small)")
        else:
            logger.info(f"  embedding_dim={embed_dim}, using gpt5_2_llm_config1 (bge-m3)")

    if task in REALTIME_TASKS:
        # ---- Real-Time Visual Perception ----
        # Use last 10s of second_captions + all min_captions
        recent_captions = _get_last_n_seconds_captions(second_captions, n_seconds=10)
        recent_captions_text = _format_captions(recent_captions)

        logger.info(f"  [Realtime] task={task}, recent_captions={len(recent_captions)}, min_captions={len(min_captions)}")

        result = _handle_realtime_task(
            question=question,
            options_text=options_text,
            recent_captions_text=recent_captions_text,
            global_captions_text=global_captions_text,
            llm_config=gpt5_2_llm_config,
        )

    elif task in BACKWARD_TASKS:
        
        # ---- Backward Tracing ----
        # Use all min_captions, with iterative retrieval up to 2 rounds
        videorag = VideoGraphSeparated(llm=gpt5_2_llm_config, working_dir=working_dir, video_embedding_gpu=args.cuda)
        videorag.load_caption_model(model_name=args.caption_model_name)

        logger.info(f"  [Backward] task={task}, second_captions={len(second_captions)}, min_captions={len(min_captions)}")

        result = _handle_backward_task(
            question=question,
            options_text=options_text,
            global_captions_text=global_captions_text,
            second_captions=second_captions,
            llm_config=gpt5_2_llm_config,
            videorag=videorag,
            max_rounds=2,
            args=args,
        )

    else:
        logger.warning(f"  Unknown task type: {task}, falling back to realtime mode")
        recent_captions = _get_last_n_seconds_captions(second_captions, n_seconds=10)
        recent_captions_text = _format_captions(recent_captions)

        result = _handle_realtime_task(
            question=question,
            options_text=options_text,
            recent_captions_text=recent_captions_text,
            global_captions_text=global_captions_text,
            llm_config=gpt5_2_llm_config,
        )

    # Evaluate correctness
    pred_letter = _extract_option_letter(str(result.get('answer', '')), options=options)
    is_correct = (pred_letter is not None
                  and gt_letter is not None
                  and pred_letter == gt_letter)

    result.update({
        'question_id': question_id,
        'task': task,
        'question': question,
        'options': options,
        'gt': gt_index,
        'gt_letter': gt_letter,
        'pred_letter': pred_letter,
        'is_correct': is_correct,
    })

    return result


# ================================================================
# Main entry point
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OVO-Bench retrieval and evaluation")
    parser.add_argument("--data_path", type=str,
                        default=os.environ.get("OVOBENCH_DATA_PATH", "./data/ovobench/ovo_bench_filtered.json"),
                        help="Path to ovo_bench_filtered.json")
    parser.add_argument("--video_path", type=str,
                        default=os.environ.get("OVOBENCH_VIDEO_PATH", "./data/ovobench/chunked_videos"))

    parser.add_argument("--caption_retrieval", type=bool, default=True)
    parser.add_argument("--visual_retrieval", type=bool, default=True)
    parser.add_argument("--entity_retrieval", type=bool, default=True)
    parser.add_argument("--need_retrieval", type=bool, default=True)
    parser.add_argument("--max_rounds", type=int, default=2)
    parser.add_argument("--filter_captions", type=bool, default=False)
    parser.add_argument("--multiscale", type=bool, default=True)
    parser.add_argument("--reconstruct_caption", type=bool, default=True)

    parser.add_argument("--caption_model_name", type=str, default="qwenvl_3_8b_instruct")

    parser.add_argument('--cuda', type=str, default='3',
                        help="CUDA device ID(s) to use.")

    parser.add_argument('--datasets_type', type=str, default="ovobench")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    logger.info(f"Set CUDA_VISIBLE_DEVICES={args.cuda}")

    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

    # Load dataset
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    logger.info(f"Loaded {len(all_questions)} questions from {args.data_path}")

    # ================================================================
    # Results collection with checkpoint resume
    # ================================================================
    output_results_path = f"{os.environ.get('RESULTS_ROOT', './results')}/output_results_{args.datasets_type}_cor.json"
    output_results = {}

    if os.path.exists(output_results_path):
        try:
            with open(output_results_path, "r", encoding="utf-8") as f:
                output_results = json.load(f)
            logger.info(f"Loaded existing results: {len(output_results)} entries from {output_results_path}")
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}, starting fresh")
            output_results = {}

    # Error patterns for detecting failed results
    _ERROR_PATTERNS = ("RetryError", "RateLimitError", "APIError", "TimeoutError",
                       "ConnectionError", "Retrieval failed", "checkpoint_not_found")

    def _is_error_result(r: dict) -> bool:
        for field in ("reasoning", "error"):
            val = r.get(field)
            if val and isinstance(val, str):
                for pat in _ERROR_PATTERNS:
                    if pat in val:
                        return True
        return False

    def _save_output_results():
        """Save results with global accuracy stats, including per-task breakdown."""
        global_correct = 0
        global_total = 0
        task_stats = {}

        for qid_str, qresult in output_results.items():
            if qid_str == "__global_summary__":
                continue
            if qresult.get('is_correct') is None:
                continue
            task = qresult.get('task', 'unknown')
            if task not in task_stats:
                task_stats[task] = {'correct': 0, 'total': 0}
            task_stats[task]['total'] += 1
            global_total += 1
            if qresult.get('is_correct'):
                task_stats[task]['correct'] += 1
                global_correct += 1

        # Per-task accuracy
        for task in task_stats:
            t = task_stats[task]
            t['accuracy'] = t['correct'] / t['total'] if t['total'] > 0 else 0.0
            t['category'] = 'realtime' if task in REALTIME_TASKS else 'backward'

        # Category-level accuracy
        category_stats = {}
        for task, stats in task_stats.items():
            cat = stats['category']
            if cat not in category_stats:
                category_stats[cat] = {'correct': 0, 'total': 0}
            category_stats[cat]['correct'] += stats['correct']
            category_stats[cat]['total'] += stats['total']
        for cat in category_stats:
            c = category_stats[cat]
            c['accuracy'] = c['correct'] / c['total'] if c['total'] > 0 else 0.0

        output_results["__global_summary__"] = {
            "total_questions": global_total,
            "total_correct": global_correct,
            "accuracy": global_correct / global_total if global_total > 0 else 0.0,
            "per_task": task_stats,
            "per_category": category_stats,
        }

        with open(output_results_path, "w", encoding="utf-8") as f:
            json.dump(output_results, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Saved results to {output_results_path} "
            f"(total={global_total}, correct={global_correct}, "
            f"accuracy={global_correct / global_total:.2%})" if global_total > 0 else
            f"Saved results to {output_results_path} (no results yet)"
        )
        for task in sorted(task_stats.keys()):
            s = task_stats[task]
            logger.info(f"  {task} ({s['category']}): {s['correct']}/{s['total']} = {s['accuracy']:.2%}")

    # ================================================================
    # Main processing loop
    # ================================================================
    processing_times = []
    overall_start_time = time.time()
    total_questions = len(all_questions)

    for loop_idx, video_data in enumerate(all_questions):
        question_id = video_data["id"]
        qid_str = str(question_id)
        task = video_data["task"]

        # Skip already successfully processed questions (unless they have errors)
        if qid_str in output_results and qid_str != "__global_summary__":
            existing = output_results[qid_str]
            if existing.get('is_correct') is not None and not _is_error_result(existing):
                logger.info(f"[{loop_idx+1}/{total_questions}] id={question_id} task={task} already done, skipping")
                continue
            else:
                logger.info(f"[{loop_idx+1}/{total_questions}] id={question_id} task={task} previous result has error, reprocessing")

        q_start_time = time.time()
        logger.info(f"[{loop_idx+1}/{total_questions}] Processing id={question_id} task={task}")
        logger.info(f"  Q: {video_data['question']}")

        try:
            result = process_single_question(video_data, args)
            output_results[qid_str] = result

            q_time = time.time() - q_start_time
            processing_times.append(q_time)

            logger.info(
                f"[{loop_idx+1}/{total_questions}] id={question_id} done | "
                f"time={q_time:.1f}s | "
                f"pred={result.get('pred_letter')} gt={result.get('gt_letter')} | "
                f"correct={result.get('is_correct')} | "
                f"mode={result.get('mode')} round={result.get('round')}"
            )

            # Auto-save every 10 questions
            if (loop_idx + 1) % 10 == 0:
                _save_output_results()

        except Exception as e:
            q_time = time.time() - q_start_time
            logger.error(f"[{loop_idx+1}/{total_questions}] id={question_id} failed: {e} (time={q_time:.1f}s)")
            output_results[qid_str] = {
                'question_id': question_id,
                'task': task,
                'question': video_data['question'],
                'error': str(e),
                'is_correct': None,
            }

    # Final save
    _save_output_results()

    # Summary
    total_time = time.time() - overall_start_time
    gs = output_results.get("__global_summary__", {})

    logger.info(f"\n{'=' * 80}")
    logger.info(f"OVO-Bench Retrieval Complete!")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Total time: {total_time:.2f}s ({total_time / 60:.2f} min)")
    logger.info(f"  Total questions: {gs.get('total_questions', 0)}")
    logger.info(f"  Total correct: {gs.get('total_correct', 0)}")
    logger.info(f"  Accuracy: {gs.get('accuracy', 0):.2%}")
    if processing_times:
        logger.info(f"  Avg time: {sum(processing_times) / len(processing_times):.2f}s/question")
        logger.info(f"  Fastest: {min(processing_times):.2f}s")
        logger.info(f"  Slowest: {max(processing_times):.2f}s")
    logger.info(f"  Results: {output_results_path}")

    per_task = gs.get('per_task', {})
    if per_task:
        logger.info(f"\nPer-task breakdown:")
        for task in sorted(per_task.keys()):
            s = per_task[task]
            logger.info(f"  {task} ({s.get('category', '?')}): {s['correct']}/{s['total']} = {s.get('accuracy', 0):.2%}")

    per_cat = gs.get('per_category', {})
    if per_cat:
        logger.info(f"\nPer-category breakdown:")
        for cat in sorted(per_cat.keys()):
            c = per_cat[cat]
            logger.info(f"  {cat}: {c['correct']}/{c['total']} = {c.get('accuracy', 0):.2%}")

    logger.info(f"{'=' * 80}\n")
