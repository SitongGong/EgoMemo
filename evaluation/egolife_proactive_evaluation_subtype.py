"""
EgoLife Proactive Service Evaluation Script — Sub-Type Level.

Extended version of egolife_proactive_evaluation.py that adds:
  1. Per sub-type Precision / Recall / F1 (in addition to per main-type)
  2. Macro-averaged P / R / F1 across all sub-types
  3. GPT-4o-mini text scoring (Rationality & Effectiveness) for
     matched pairs whose **sub-type also matches exactly**

Usage:
  python egolife_proactive_evaluation_subtype.py \
      --prediction_base /path/to/egolife_results \
      --gt_base /path/to/gemini_annotation_segments \
      --persons A5_KATRINA \
      --days DAY1 DAY2 DAY3 DAY4 DAY5 \
      --tolerance 60

  # With LLM scoring enabled:
  python egolife_proactive_evaluation_subtype.py --no_llm_scoring False
"""

import os
import re
import json
import argparse
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================

ALL_PERSONS = ["A1_JAKE", "A3_TASHA", "A5_KATRINA"]     # "A3_TASHA", "A4_LUCIA", "A5_KATRINA"

# 4 main service categories and their sub-types
MAIN_TYPE_MAP = {
    "safety":               "Instant",
    "tool_use":             "Instant",
    "next_step_guidance":   "Short-Term",
    "error_recovery":       "Short-Term",
    "resource_reminder":    "Short-Term",
    "habit_coaching":       "Long-Term",
    "memory_link_contextual": "Long-Term",
    "routine_optimization": "Long-Term",
    # personal_progressive 已从 benchmark 移除(camera-ready),不计入评测
    "memory_recall":        "Episodic",
    "task_reminder":        "Episodic",
}

SUB_TYPES = list(MAIN_TYPE_MAP.keys())

GT_SERVICE_DIRS = [
    ("egolife_instant/safety",                  "safety"),
    ("egolife_instant/tool_use",                "tool_use"),
    ("egolife_short_term/next_step_guidance",   "next_step_guidance"),
    ("egolife_short_term/error_recovery",       "error_recovery"),
    ("egolife_short_term/resource_reminder",    "resource_reminder"),
    ("egolife_long_term/habit_coaching",        "habit_coaching"),
    ("egolife_long_term/memory_link_contextual","memory_link_contextual"),
    ("egolife_long_term/routine_optimization",  "routine_optimization"),
    ("egolife_episodic/memory_recall",          "memory_recall"),
    ("egolife_episodic/task_reminder",          "task_reminder"),
]

PRED_MAIN_TYPE_NORMALIZE = {
    "instant":                          "Instant",
    "instant proactive service":        "Instant",
    "instant proactive services":       "Instant",
    "short-term":                       "Short-Term",
    "short-term proactive service":     "Short-Term",
    "short-term proactive services":    "Short-Term",
    "long-term":                        "Long-Term",
    "long-term proactive service":      "Long-Term",
    "long-term proactive services":     "Long-Term",
    "episodic":                         "Episodic",
    "episodic proactive service":       "Episodic",
    "episodic proactive services":      "Episodic",
}

# Normalize prediction sub-type strings to canonical sub_type keys.
# Built from actual model outputs across A1-A6 DAY1-DAY5 predictions.
PRED_SUB_TYPE_NORMALIZE = {
    # safety
    "safety":                       "safety",
    # tool_use
    "tool use":                     "tool_use",
    "tool_use":                     "tool_use",
    # next_step_guidance
    "next-step guidance":           "next_step_guidance",
    "next_step_guidance":           "next_step_guidance",
    "next step guidance":           "next_step_guidance",
    # error_recovery
    "error-recovery":               "error_recovery",
    "error_recovery":               "error_recovery",
    "error recovery":               "error_recovery",
    # resource_reminder
    "resource reminder":            "resource_reminder",
    "resource_reminder":            "resource_reminder",
    # habit_coaching
    "habit-coaching":               "habit_coaching",
    "habit_coaching":               "habit_coaching",
    "habit coaching":               "habit_coaching",
    # memory_link_contextual
    "long-horizon memory-link":     "memory_link_contextual",
    "long horizon memory-link":     "memory_link_contextual",
    "long-horizon memory link":     "memory_link_contextual",
    "memory_link_contextual":       "memory_link_contextual",
    "memory link contextual":       "memory_link_contextual",
    # personal_progressive 已从 benchmark 移除 -> 映射为 None, 预测中出现则跳过
    "personal progress feedback":   None,
    "personal_progressive":         None,
    "personal progressive":         None,
    # routine_optimization
    "routine optimization":         "routine_optimization",
    "routine_optimization":         "routine_optimization",
    # memory_recall
    "episodic memory recall":       "memory_recall",
    "memory_recall":                "memory_recall",
    "memory recall":                "memory_recall",
    # task_reminder
    "episodic task reminder":       "task_reminder",
    "task_reminder":                "task_reminder",
    "task reminder":                "task_reminder",
}

MAIN_TYPES = ["Instant", "Short-Term", "Long-Term", "Episodic"]

# ===========================================================================
# LLM evaluation prompts
# ===========================================================================

LLM_RATIONALITY_PROMPT = """\
You are an expert evaluator of proactive assistant systems.

Your task: evaluate the RATIONALITY of a predicted proactive service by
comparing it against the ground truth (GT). Focus on semantic similarity
between the GT and predicted text.

------------------------------------------------------------
Input
------------------------------------------------------------

GT Service Type : {gt_service_type}
GT Message      : {gt_user_prompt}

Predicted Service Type : {pred_service_type}
Predicted Message      : {pred_user_prompt}

------------------------------------------------------------
Evaluation Criteria — Rationality (1-5)
------------------------------------------------------------

Assess the semantic similarity between the predicted message and the GT:
  - Do both messages address the same underlying situation or need?
  - Is the predicted content semantically consistent with the GT?
  - Does the predicted service type align with the GT service type?
  - Are the key information elements preserved?

  1 = Completely irrelevant or contradicts GT
  2 = Major semantic mismatch; addresses a different need
  3 = Partially aligned; captures some aspects but misses key points
  4 = Mostly aligned; minor semantic differences
  5 = Fully semantically equivalent to GT

------------------------------------------------------------
Output Format (STRICT — nothing else)
------------------------------------------------------------

Rationality: <1-5>
Justification: <1-3 sentences>
"""

LLM_EFFECTIVENESS_PROMPT = """\
You are an expert evaluator of proactive assistant systems.

Your task: evaluate the EFFECTIVENESS of a predicted proactive service.
Focus on whether the message logically and helpfully assists the user.

------------------------------------------------------------
Input
------------------------------------------------------------

GT Service Type : {gt_service_type}
GT Message      : {gt_user_prompt}

Predicted Service Type : {pred_service_type}
Predicted Message      : {pred_user_prompt}

------------------------------------------------------------
Evaluation Criteria — Effectiveness (1-5)
------------------------------------------------------------

Assess whether the predicted message provides logical, helpful assistance:
  - Does it logically address the user's situation?
  - Is the advice or information actionable and useful?
  - Is the reasoning behind the service sound?
  - Does it proactively help the user in a meaningful way?
  - Is the tone and framing appropriate for a proactive assistant?

  1 = Illogical or unhelpful; provides no meaningful assistance
  2 = Poorly reasoned; the help offered is confusing or misguided
  3 = Somewhat helpful but lacks clear logic or actionability
  4 = Logically sound and helpful, minor issues
  5 = Excellent — clearly reasoned, actionable, and genuinely helpful

------------------------------------------------------------
Output Format (STRICT — nothing else)
------------------------------------------------------------

Effectiveness: <1-5>
Justification: <1-3 sentences>
"""

# ===========================================================================
# Time parsing utilities
# ===========================================================================

def parse_day_time_window(time_window_str: str):
    """
    Parse time window strings from both GT and prediction formats.
    Returns (day_num, start_seconds, end_seconds) or None on failure.
    """
    pattern = r'DAY(\d+)\s*[-\s]\s*(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s*-\s*(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)'
    match = re.search(pattern, time_window_str)
    if not match:
        return None

    day_num = int(match.group(1))

    def to_seconds(t):
        parts = t.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    start = to_seconds(match.group(2))
    end = to_seconds(match.group(3))
    return day_num, start, end


def time_window_center(day_num, start, end):
    center = (start + end) / 2.0
    return day_num * 86400 + center


# ===========================================================================
# GT extraction
# ===========================================================================

def extract_ground_truth(gt_base: str, person: str, allowed_days: set):
    results = []
    for subdir, sub_type in GT_SERVICE_DIRS:
        fpath = os.path.join(gt_base, person, subdir, "conversation_results_correct.json")
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {fpath}: {e}")
            continue
        if not isinstance(data, list):
            continue

        main_type = MAIN_TYPE_MAP[sub_type]

        for entry in data:
            source = entry.get("supporting_source",
                               entry.get("current_supporting_source", ""))
            if source == "speakers_say":
                continue
            # past_source = entry.get("past_supporting_source", "")
            # if past_source == "speakers_say":
            #     continue

            tw = entry.get("current_time_window", "")
            parsed = parse_day_time_window(tw)
            if parsed is None:
                continue
            day_num, start_s, end_s = parsed

            if day_num not in allowed_days:
                continue

            utterance = ""
            dialogue = entry.get("proactive_dialogue") or entry.get("dialogue") or []
            for turn in dialogue:
                if turn.get("role") == "assistant":
                    utterance = turn.get("utterance", "").strip()
                    break
            if not utterance:
                continue

            results.append({
                "person": person,
                "day_num": day_num,
                "main_type": main_type,
                "sub_type": sub_type,
                "time_center": time_window_center(day_num, start_s, end_s),
                "time_start": day_num * 86400 + start_s,
                "time_end": day_num * 86400 + end_s,
                "user_prompt": utterance,
                "service_type_raw": sub_type,
            })

    results.sort(key=lambda x: x["time_center"])
    return results


# ===========================================================================
# Prediction extraction
# ===========================================================================

def extract_predictions(prediction_base: str, person: str,
                        caption_model: str, allowed_days: set,
                        response_name: str = "proactive_response_gpt_5"):
    person_dir = f"{person}-{caption_model}_restart"
    base = os.path.join(prediction_base, person_dir)

    results = []
    for day_num in sorted(allowed_days):
        day_tag = f"DAY{day_num}"
        fpath = os.path.join(base, day_tag, f"{response_name}.json")
        if not os.path.exists(fpath):
            logger.warning(f"Prediction file not found: {fpath}")
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {fpath}: {e}")
            continue

        for entry in data:
            gr = entry.get("gemini_response")
            if gr is None:
                continue

            services = []
            if isinstance(gr, list):
                services = gr
            elif isinstance(gr, dict):
                if gr.get("decision") == "suppressed":
                    continue
                fs = gr.get("finalized_services", {})
                if isinstance(fs, dict):
                    svcs = fs.get("services", [])
                    if isinstance(svcs, list):
                        services = svcs
                elif isinstance(fs, list):
                    services = fs

            for svc in services:
                if not isinstance(svc, dict):
                    continue

                tw = svc.get("trigger_time_window", "")
                parsed = parse_day_time_window(tw)
                if parsed is None:
                    continue
                svc_day, start_s, end_s = parsed

                if svc_day not in allowed_days:
                    continue

                raw_main = svc.get("service_main_type", "")
                main_type = PRED_MAIN_TYPE_NORMALIZE.get(
                    raw_main.lower().strip(), None
                )
                if main_type is None:
                    logger.warning(
                        f"Unmapped main_type: '{raw_main}' — add to PRED_MAIN_TYPE_NORMALIZE"
                    )
                    main_type = raw_main  # keep raw so it's visible

                raw_sub = svc.get("service_sub_type", "")
                _key = raw_sub.lower().strip()
                # personal_progressive 已移除评测: 预测里出现则跳过, 不计入
                if _key in ("personal progress feedback", "personal_progressive", "personal progressive"):
                    continue
                sub_type = PRED_SUB_TYPE_NORMALIZE.get(_key, None)
                if sub_type is None:
                    logger.warning(
                        f"Unmapped sub_type: '{raw_sub}' — add to PRED_SUB_TYPE_NORMALIZE"
                    )
                    sub_type = raw_sub  # keep raw so it's visible

                results.append({
                    "person": person,
                    "day_num": svc_day,
                    "main_type": main_type,
                    "sub_type": sub_type,
                    "time_center": time_window_center(svc_day, start_s, end_s),
                    "time_start": svc_day * 86400 + start_s,
                    "time_end": svc_day * 86400 + end_s,
                    "user_prompt": svc.get("user_prompt", ""),
                    "confidence": svc.get("confidence", ""),
                    "trigger_evidence": svc.get("trigger_evidence", ""),
                })

    results.sort(key=lambda x: x["time_center"])
    return results


# ===========================================================================
# Matching
# ===========================================================================

def match_predictions_to_gt(predictions, ground_truth, tolerance=30.0,
                            match_main_type=True):
    """
    Greedy matching: for each GT, find the closest unmatched prediction
    within tolerance whose main_type matches (if required).
    """
    used_pred = set()
    matched = []

    for gt in ground_truth:
        gt_start = gt["time_start"]
        gt_end = gt["time_end"]
        best_idx = None
        best_dist = float('inf')

        for i, pred in enumerate(predictions):
            if i in used_pred:
                continue
            if match_main_type and pred["main_type"] != gt["main_type"]:
                continue

            pc = pred["time_center"]
            if pc < gt_start - tolerance or pc > gt_end + tolerance:
                continue

            if pc < gt_start:
                dist = gt_start - pc
            elif pc > gt_end:
                dist = pc - gt_end
            else:
                dist = 0.0

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            matched.append((predictions[best_idx], gt))
            used_pred.add(best_idx)

    missed_gt = [gt for gt in ground_truth
                 if not any(g is gt for _, g in matched)]
    redundant_pred = [pred for i, pred in enumerate(predictions)
                      if i not in used_pred]

    return matched, missed_gt, redundant_pred


# ===========================================================================
# Metrics computation
# ===========================================================================

def compute_prf(num_matched, num_pred, num_gt):
    precision = num_matched / num_pred if num_pred > 0 else 0.0
    recall = num_matched / num_gt if num_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ===========================================================================
# LLM scoring
# ===========================================================================

def parse_single_score(text: str, key: str):
    """Parse a single score (Rationality or Effectiveness) from LLM output."""
    r = re.search(rf'{key}:\s*(\d+)', text, re.IGNORECASE)
    if r:
        return min(int(r.group(1)), 5)
    return None


def llm_score_matched_pairs(matched_pairs, model_name="gpt-4o-mini"):
    """
    Score each matched pair using GPT-4o-mini.
    Only scores pairs where the sub-type matches exactly.

    Returns list of score dicts (or None for pairs that weren't scored).
    """
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        logger.error(f"Failed to init OpenAI client: {e}")
        return []

    all_scores = []
    for pred, gt in tqdm(matched_pairs, desc=f"LLM scoring ({model_name})"):
        # Only score pairs with exact sub-type match
        if pred["sub_type"] != gt["sub_type"]:
            all_scores.append(None)
            continue

        gt_type_str = f"{gt['main_type']} / {gt['sub_type']}"
        pred_type_str = f"{pred['main_type']} / {pred['sub_type']}"

        scores = {}

        # --- Rationality scoring ---
        prompt_r = LLM_RATIONALITY_PROMPT.format(
            gt_service_type=gt_type_str,
            gt_user_prompt=gt["user_prompt"],
            pred_service_type=pred_type_str,
            pred_user_prompt=pred["user_prompt"],
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_r}],
                temperature=0.0,
            )
            text_r = response.choices[0].message.content
            val = parse_single_score(text_r, "Rationality")
            if val is not None:
                scores["rationality"] = val
                scores["rationality_justification"] = text_r
        except Exception as e:
            logger.warning(f"LLM Rationality call failed: {e}")

        # --- Effectiveness scoring ---
        prompt_e = LLM_EFFECTIVENESS_PROMPT.format(
            gt_service_type=gt_type_str,
            gt_user_prompt=gt["user_prompt"],
            pred_service_type=pred_type_str,
            pred_user_prompt=pred["user_prompt"],
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_e}],
                temperature=0.0,
            )
            text_e = response.choices[0].message.content
            val = parse_single_score(text_e, "Effectiveness")
            if val is not None:
                scores["effectiveness"] = val
                scores["effectiveness_justification"] = text_e
        except Exception as e:
            logger.warning(f"LLM Effectiveness call failed: {e}")

        if "rationality" in scores and "effectiveness" in scores:
            all_scores.append(scores)
        else:
            logger.warning(f"Incomplete LLM scores for pair, skipping")
            all_scores.append(None)

    return all_scores


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_person(predictions, ground_truth, tolerance=30.0):
    """
    Run full evaluation for one person.

    Returns a dict with:
      - per_main_type: {main_type: {precision, recall, f1, ...}}
      - per_sub_type:  {sub_type: {precision, recall, f1, ...}}
      - overall: {precision, recall, f1, ...}
      - matched_pairs: all matched (pred, gt) tuples for LLM scoring
    """
    results = {
        "per_main_type": {},
        "per_sub_type": {},
        "overall": {},
    }

    # --- Per main-type matching ---
    all_matched = []
    all_missed = []
    all_redundant = []

    for mt in MAIN_TYPES:
        preds_mt = [p for p in predictions if p["main_type"] == mt]
        gts_mt = [g for g in ground_truth if g["main_type"] == mt]

        matched, missed, redundant = match_predictions_to_gt(
            preds_mt, gts_mt, tolerance=tolerance, match_main_type=False
        )

        p, r, f = compute_prf(len(matched), len(preds_mt), len(gts_mt))
        results["per_main_type"][mt] = {
            "precision": p, "recall": r, "f1": f,
            "num_pred": len(preds_mt), "num_gt": len(gts_mt),
            "num_matched": len(matched),
        }

        all_matched.extend(matched)
        all_missed.extend(missed)
        all_redundant.extend(redundant)

    # --- Per sub-type metrics ---
    # Count sub-type level stats from the already-matched pairs
    sub_type_stats = defaultdict(lambda: {"num_pred": 0, "num_gt": 0, "num_matched": 0})

    for st in SUB_TYPES:
        sub_type_stats[st]["num_pred"] = sum(
            1 for p in predictions if p["sub_type"] == st
        )
        sub_type_stats[st]["num_gt"] = sum(
            1 for g in ground_truth if g["sub_type"] == st
        )
        # For sub-type level matching: a pair counts as matched for sub_type st
        # only when BOTH pred and GT have sub_type == st (strict sub-type match)
        sub_type_stats[st]["num_matched"] = sum(
            1 for pred, gt in all_matched
            if gt["sub_type"] == st and pred["sub_type"] == st
        )

    # Sanity check: sub-type prediction counts must sum to total predictions
    known_sub_pred = sum(sub_type_stats[st]["num_pred"] for st in SUB_TYPES)
    if known_sub_pred != len(predictions):
        unmapped = [p for p in predictions if p["sub_type"] not in SUB_TYPES]
        unmapped_types = set(p["sub_type"] for p in unmapped)
        logger.error(
            f"  SUB-TYPE MISMATCH: {known_sub_pred} sub-type preds vs "
            f"{len(predictions)} total preds. "
            f"Unmapped sub_types: {unmapped_types}"
        )

    for st in SUB_TYPES:
        s = sub_type_stats[st]
        p, r, f = compute_prf(s["num_matched"], s["num_pred"], s["num_gt"])
        results["per_sub_type"][st] = {
            "precision": p, "recall": r, "f1": f,
            **s,
        }

    # --- Overall ---
    total_pred = len(predictions)
    total_gt = len(ground_truth)
    total_matched = len(all_matched)
    p, r, f = compute_prf(total_matched, total_pred, total_gt)
    results["overall"] = {
        "precision": p, "recall": r, "f1": f,
        "num_pred": total_pred, "num_gt": total_gt,
        "num_matched": total_matched,
    }

    results["matched_pairs"] = all_matched

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EgoLife Proactive Service Evaluation — Sub-Type Level"
    )
    parser.add_argument(
        "--prediction_base", type=str,
        default=paths.EGOLIFE_PRED,
    )
    parser.add_argument(
        "--gt_base", type=str,
        default=paths.EGOLIFE_GT,
    )
    parser.add_argument(
        "--persons", type=str, nargs='+', default=ALL_PERSONS,
    )
    parser.add_argument(
        "--days", type=str, nargs='+',
        default=["DAY1", "DAY2", "DAY3", "DAY4", "DAY5"],
    )
    parser.add_argument(
        "--caption_model", type=str, default=paths.CAPTION_MODEL,
    )
    parser.add_argument(
        "--tolerance", type=float, default=60.0,
    )
    parser.add_argument(
        "--no_llm_scoring", action="store_true", default=False,
        help="Skip LLM-based scoring",
    )
    parser.add_argument(
        "--llm_model", type=str, default="gpt-4o-mini",
    )
    parser.add_argument(
        "--response_name", type=str,
        default="proactive_response_gpt_5",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(paths.OUTPUT_ROOT, "egolife_proactive_eval_subtype_results.json"),
    )
    args = parser.parse_args()

    allowed_days = set()
    for d in args.days:
        num = int(d.replace("DAY", ""))
        allowed_days.add(num)
    logger.info(f"Allowed days: {sorted(allowed_days)}")

    # ---- Aggregate across all persons ----
    agg_per_main = {mt: {"num_pred": 0, "num_gt": 0, "num_matched": 0}
                    for mt in MAIN_TYPES}
    agg_per_sub = {st: {"num_pred": 0, "num_gt": 0, "num_matched": 0}
                   for st in SUB_TYPES}
    agg_total = {"num_pred": 0, "num_gt": 0, "num_matched": 0}
    all_person_results = {}
    all_matched_for_llm = []

    for person in args.persons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {person}")
        logger.info(f"{'='*60}")

        person_dir = os.path.join(
            args.prediction_base,
            f"{person}-{args.caption_model}_restart"
        )
        person_days = set()
        for d in allowed_days:
            fpath = os.path.join(person_dir, f"DAY{d}", f"{args.response_name}.json")
            if os.path.exists(fpath):
                person_days.add(d)
        if not person_days:
            logger.warning(f"  No prediction files found for {person}, skipping")
            continue
        logger.info(f"  Available days: {sorted(person_days)}")

        gt = extract_ground_truth(args.gt_base, person, person_days)
        preds = extract_predictions(
            args.prediction_base, person, args.caption_model, person_days,
            response_name=args.response_name
        )
        logger.info(f"  GT annotations: {len(gt)}  |  Predictions: {len(preds)}")

        person_result = evaluate_person(preds, gt, tolerance=args.tolerance)
        all_person_results[person] = person_result

        # --- Print per main-type ---
        logger.info(f"\n  --- Per Main-Type ---")
        for mt in MAIN_TYPES:
            r = person_result["per_main_type"].get(mt, {})
            if r.get("num_gt", 0) > 0 or r.get("num_pred", 0) > 0:
                logger.info(
                    f"  {mt:12s}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                    f"F1={r['f1']:.3f}  (pred={r['num_pred']}, gt={r['num_gt']}, "
                    f"matched={r['num_matched']})"
                )
            agg_per_main[mt]["num_pred"] += r.get("num_pred", 0)
            agg_per_main[mt]["num_gt"] += r.get("num_gt", 0)
            agg_per_main[mt]["num_matched"] += r.get("num_matched", 0)

        # --- Print per sub-type ---
        logger.info(f"\n  --- Per Sub-Type ---")
        for st in SUB_TYPES:
            r = person_result["per_sub_type"].get(st, {})
            if r.get("num_gt", 0) > 0 or r.get("num_pred", 0) > 0:
                mt_label = MAIN_TYPE_MAP[st]
                logger.info(
                    f"  [{mt_label:10s}] {st:25s}  P={r['precision']:.3f}  "
                    f"R={r['recall']:.3f}  F1={r['f1']:.3f}  "
                    f"(pred={r['num_pred']}, gt={r['num_gt']}, "
                    f"matched={r['num_matched']})"
                )
            agg_per_sub[st]["num_pred"] += r.get("num_pred", 0)
            agg_per_sub[st]["num_gt"] += r.get("num_gt", 0)
            agg_per_sub[st]["num_matched"] += r.get("num_matched", 0)

        # --- Print overall ---
        ov = person_result["overall"]
        logger.info(
            f"\n  {'Overall':12s}  P={ov['precision']:.3f}  R={ov['recall']:.3f}  "
            f"F1={ov['f1']:.3f}  (pred={ov['num_pred']}, gt={ov['num_gt']}, "
            f"matched={ov['num_matched']})"
        )
        agg_total["num_pred"] += ov["num_pred"]
        agg_total["num_gt"] += ov["num_gt"]
        agg_total["num_matched"] += ov["num_matched"]

        all_matched_for_llm.extend(person_result["matched_pairs"])

    # ================================================================
    # AGGREGATED RESULTS
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATED RESULTS (all persons)")
    logger.info(f"{'='*60}")

    # --- Per main-type ---
    logger.info(f"\n  --- Per Main-Type ---")
    for mt in MAIN_TYPES:
        a = agg_per_main[mt]
        p, r, f = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
        logger.info(
            f"  {mt:12s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  "
            f"(pred={a['num_pred']}, gt={a['num_gt']}, matched={a['num_matched']})"
        )

    # --- Per sub-type ---
    logger.info(f"\n  --- Per Sub-Type ---")
    sub_type_p_list = []
    sub_type_r_list = []
    sub_type_f1_list = []

    for st in SUB_TYPES:
        a = agg_per_sub[st]
        p, r, f = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
        mt_label = MAIN_TYPE_MAP[st]
        logger.info(
            f"  [{mt_label:10s}] {st:25s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  "
            f"(pred={a['num_pred']}, gt={a['num_gt']}, matched={a['num_matched']})"
        )
        # Only include sub-types that have GT or predictions for macro-average
        if a["num_gt"] > 0 or a["num_pred"] > 0:
            sub_type_p_list.append(p)
            sub_type_r_list.append(r)
            sub_type_f1_list.append(f)

    # --- Macro-average across sub-types ---
    if sub_type_p_list:
        macro_p = np.mean(sub_type_p_list)
        macro_r = np.mean(sub_type_r_list)
        macro_f1 = np.mean(sub_type_f1_list)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    logger.info(f"\n  --- Sub-Type Macro Average ---")
    logger.info(
        f"  {'Macro-Avg':25s}  P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}  "
        f"(over {len(sub_type_p_list)} active sub-types)"
    )

    # --- Overall (micro) ---
    p_all, r_all, f_all = compute_prf(
        agg_total["num_matched"], agg_total["num_pred"], agg_total["num_gt"]
    )
    logger.info(
        f"\n  {'Overall':12s}  P={p_all:.3f}  R={r_all:.3f}  F1={f_all:.3f}  "
        f"(pred={agg_total['num_pred']}, gt={agg_total['num_gt']}, "
        f"matched={agg_total['num_matched']})"
    )

    # ================================================================
    # LLM scoring (only for sub-type exact matches)
    # ================================================================
    llm_results = {}
    llm_per_sub = {}
    if not args.no_llm_scoring and all_matched_for_llm:
        # Filter to only sub-type exact matches
        subtype_matched = [(pred, gt) for pred, gt in all_matched_for_llm
                           if pred["sub_type"] == gt["sub_type"]]
        logger.info(
            f"\nLLM scoring: {len(subtype_matched)} pairs with exact sub-type match "
            f"(out of {len(all_matched_for_llm)} total matched)"
        )

        if subtype_matched:
            scores = llm_score_matched_pairs(
                subtype_matched, model_name=args.llm_model
            )

            # --- Global LLM scores ---
            valid = [s for s in scores if s is not None]
            if valid:
                avg_r = np.mean([s["rationality"] for s in valid])
                avg_e = np.mean([s["effectiveness"] for s in valid])
                llm_results = {
                    "avg_rationality": round(avg_r, 3),
                    "avg_effectiveness": round(avg_e, 3),
                    "num_scored": len(valid),
                }
                logger.info(f"\n  --- LLM Scores (Global, {len(valid)} pairs) ---")
                logger.info(f"    Avg Rationality:    {avg_r:.3f}")
                logger.info(f"    Avg Effectiveness:  {avg_e:.3f}")

            # --- Per sub-type LLM scores ---
            logger.info(f"\n  --- LLM Scores Per Sub-Type ---")
            sub_scores_map = defaultdict(list)
            for (pred, gt), score in zip(subtype_matched, scores):
                if score is not None:
                    sub_scores_map[gt["sub_type"]].append(score)

            for st in SUB_TYPES:
                if st not in sub_scores_map or not sub_scores_map[st]:
                    continue
                ss = sub_scores_map[st]
                st_avg_r = np.mean([s["rationality"] for s in ss])
                st_avg_e = np.mean([s["effectiveness"] for s in ss])
                llm_per_sub[st] = {
                    "avg_rationality": round(st_avg_r, 3),
                    "avg_effectiveness": round(st_avg_e, 3),
                    "num_scored": len(ss),
                }
                mt_label = MAIN_TYPE_MAP[st]
                logger.info(
                    f"    [{mt_label:10s}] {st:25s}  "
                    f"Rationality={st_avg_r:.3f}  Effectiveness={st_avg_e:.3f}  "
                    f"(n={len(ss)})"
                )

    # ================================================================
    # Save results
    # ================================================================
    output_data = {
        "config": {
            "persons": args.persons,
            "days": args.days,
            "tolerance": args.tolerance,
            "caption_model": args.caption_model,
            "llm_model": args.llm_model,
        },
        "aggregated": {
            "per_main_type": {},
            "per_sub_type": {},
            "sub_type_macro_average": {
                "precision": round(macro_p, 4),
                "recall": round(macro_r, 4),
                "f1": round(macro_f1, 4),
                "num_active_sub_types": len(sub_type_p_list),
            },
            "overall": {
                "precision": round(p_all, 4),
                "recall": round(r_all, 4),
                "f1": round(f_all, 4),
                **agg_total,
            },
            "llm_scores": llm_results,
            "llm_scores_per_sub_type": llm_per_sub,
        },
        "per_person": {},
    }

    for mt in MAIN_TYPES:
        a = agg_per_main[mt]
        pp, rr, ff = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
        output_data["aggregated"]["per_main_type"][mt] = {
            "precision": round(pp, 4), "recall": round(rr, 4),
            "f1": round(ff, 4), **a,
        }

    for st in SUB_TYPES:
        a = agg_per_sub[st]
        pp, rr, ff = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
        output_data["aggregated"]["per_sub_type"][st] = {
            "main_type": MAIN_TYPE_MAP[st],
            "precision": round(pp, 4), "recall": round(rr, 4),
            "f1": round(ff, 4), **a,
        }

    for person, res in all_person_results.items():
        output_data["per_person"][person] = {
            "per_main_type": res["per_main_type"],
            "per_sub_type": res["per_sub_type"],
            "overall": res["overall"],
        }

    output_path = args.output or os.path.join(
        os.path.dirname(__file__),
        "egolife_proactive_eval_subtype_results.json"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
