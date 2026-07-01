"""
CaptionCook4D Proactive Service Evaluation Script — Sub-Type Level.

Evaluates predicted proactive services against ground truth annotations.
  - GT: error_to_dialogue_results_.json  ->  dialogue.items[]
  - Predictions: captioncook4d/{recording_id}_qwenvl_3_8b_instruct/proactive_response_*.json
  - Only recordings present in the prediction directory are evaluated

Computes:
  1. Per sub-type Precision / Recall / F1 (strict sub-type match)
  2. Per main-type P / R / F1
  3. Macro-averaged P / R / F1 across sub-types
  4. GPT-4o-mini text scoring (Rationality & Effectiveness) for
     matched pairs whose sub-type also matches exactly

Usage:
  python captioncook4d_proactive_evaluation_subtype.py
  python captioncook4d_proactive_evaluation_subtype.py --no_llm_scoring
  python captioncook4d_proactive_evaluation_subtype.py --response_name proactive_response_gpt_5_wo_visual
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

# Canonical sub_type -> main_type mapping
MAIN_TYPE_MAP = {
    "safety":             "Instant",
    "tool_use":           "Instant",
    "next_step_guidance": "Short-Term",
    "error_recovery":     "Short-Term",
    "resource_reminder":  "Short-Term",
    # Episodic(memory_recall / task_reminder)已从 CaptainCook4D 移除:
    # 该数据集无 Episodic 标注,预测中出现则跳过,不计入评测。
}

SUB_TYPES = list(MAIN_TYPE_MAP.keys())

MAIN_TYPES = ["Instant", "Short-Term"]

# 评测时跳过的视频(recording_id)。
# 这些视频会从有效录像集合中剔除,不参与 GT/预测的提取与指标计算。
SKIP_VIDEOS = {
    "15_46", "16_39", "16_42", "16_44", "20_47",
    "20_48", "23_39", "23_41", "28_50", "4_44",
}

# Normalize GT and prediction main type strings to canonical form
MAIN_TYPE_NORMALIZE = {
    "instant":                          "Instant",
    "instant proactive service":        "Instant",
    "instant proactive services":       "Instant",
    "short-term":                       "Short-Term",
    "short-term proactive service":     "Short-Term",
    "short-term proactive services":    "Short-Term",
    # "episodic":                         "Episodic",
    # "episodic proactive service":       "Episodic",
    # "episodic proactive services":      "Episodic",
}

# Normalize GT and prediction sub type strings to canonical form
# Both GT and predictions use display names like "Error-Recovery", "Tool Use", etc.
SUB_TYPE_NORMALIZE = {
    # safety
    "safety":                   "safety",
    # tool_use
    "tool use":                 "tool_use",
    "tool_use":                 "tool_use",
    # next_step_guidance
    "next-step guidance":       "next_step_guidance",
    "next_step_guidance":       "next_step_guidance",
    "next step guidance":       "next_step_guidance",
    # error_recovery
    "error-recovery":           "error_recovery",
    "error_recovery":           "error_recovery",
    "error recovery":           "error_recovery",
    # resource_reminder
    "resource reminder":        "resource_reminder",
    "resource_reminder":        "resource_reminder",
    # Episodic(memory_recall / task_reminder)已从 CaptainCook4D 移除, 不再映射
    # "episodic memory recall":   "memory_recall",
    # "memory_recall":            "memory_recall",
    # "memory recall":            "memory_recall",
    # "episodic task reminder":   "task_reminder",
    # "task_reminder":            "task_reminder",
    # "task reminder":            "task_reminder",
}

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

def parse_pred_time_window(time_window_str: str):
    """
    Parse prediction time_window: "DAY1 00:00:06-00:00:08" or "DAY1-00:00:06-00:00:08"
    Returns (start_seconds, end_seconds) or None.
    """
    if not time_window_str:
        return None

    pattern = r'DAY\d+\s*[-\s]\s*(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s*-\s*(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)'
    match = re.search(pattern, time_window_str)
    if not match:
        return None

    def hms_to_seconds(t):
        p = t.split(':')
        return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])

    return hms_to_seconds(match.group(1)), hms_to_seconds(match.group(2))


# ===========================================================================
# GT extraction
# ===========================================================================

def extract_ground_truth(gt_path: str, valid_recordings: set):
    """
    Extract GT from error_to_dialogue_results_.json.

    Each recording has dialogue.items[], where each item has:
      - start_time, end_time (seconds, -1 if missing step)
      - service_type.main, service_type.sub
      - dialogue[] with role/utterance pairs

    Only recordings in valid_recordings are included.
    Items with start_time == -1 are skipped (no valid time window).

    Returns list of dicts.
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for entry in data:
        rec_id = entry.get("recording_id", "")
        if rec_id not in valid_recordings:
            continue

        dlg = entry.get("dialogue", {})
        items = dlg.get("items", [])

        for item in items:
            start_t = item.get("start_time", -1.0)
            end_t = item.get("end_time", -1.0)

            # Skip items without valid time (e.g. missing steps)
            if start_t < 0 or end_t < 0:
                continue

            # Normalize service types
            st = item.get("service_type", {})
            raw_main = st.get("main", "")
            main_type = MAIN_TYPE_NORMALIZE.get(raw_main.lower().strip(), None)
            if main_type is None:
                logger.warning(f"GT unmapped main_type: '{raw_main}'")
                main_type = raw_main

            raw_sub = st.get("sub", "")
            sub_type = SUB_TYPE_NORMALIZE.get(raw_sub.lower().strip(), None)
            if sub_type is None:
                logger.warning(f"GT unmapped sub_type: '{raw_sub}'")
                sub_type = raw_sub

            # Extract first assistant utterance
            utterance = ""
            for turn in item.get("dialogue", []):
                if turn.get("role") == "assistant":
                    utterance = turn.get("utterance", "").strip()
                    break
            if not utterance:
                continue

            results.append({
                "recording_id": rec_id,
                "main_type": main_type,
                "sub_type": sub_type,
                "time_start": start_t,
                "time_end": end_t,
                "time_center": (start_t + end_t) / 2.0,
                "user_prompt": utterance,
                "error_tag": item.get("error_tag", ""),
                "step_id": item.get("step_id", ""),
            })

    results.sort(key=lambda x: (x["recording_id"], x["time_center"]))
    return results


# ===========================================================================
# Prediction extraction
# ===========================================================================

def extract_predictions(pred_base: str, valid_recordings: set, response_name: str):
    results = []
    count = 0
    for rec_id in sorted(valid_recordings):
        working_dir = os.path.join(pred_base, f"{rec_id}_qwenvl_3_8b_instruct")
        fpath = os.path.join(working_dir, f"{response_name}.json")
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            count += len(data)
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
                parsed = parse_pred_time_window(tw)
                if parsed is None:
                    continue
                start_s, end_s = parsed

                raw_main = svc.get("service_main_type", "")
                main_type = MAIN_TYPE_NORMALIZE.get(
                    raw_main.lower().strip(), None
                )
                if main_type is None:
                    logger.warning(
                        f"Pred unmapped main_type: '{raw_main}' — add to MAIN_TYPE_NORMALIZE"
                    )
                    main_type = raw_main

                raw_sub = svc.get("service_sub_type", "")
                _key = raw_sub.lower().strip()
                # Episodic 已从 CaptainCook4D 移除: 预测里出现 memory_recall/task_reminder 则跳过
                if _key in ("episodic memory recall", "memory_recall", "memory recall",
                            "episodic task reminder", "task_reminder", "task reminder"):
                    continue
                sub_type = SUB_TYPE_NORMALIZE.get(_key, None)
                if sub_type is None:
                    logger.warning(
                        f"Pred unmapped sub_type: '{raw_sub}' — add to SUB_TYPE_NORMALIZE"
                    )
                    sub_type = raw_sub

                results.append({
                    "recording_id": rec_id,
                    "main_type": main_type,
                    "sub_type": sub_type,
                    "time_start": start_s,
                    "time_end": end_s,
                    "time_center": (start_s + end_s) / 2.0,
                    "user_prompt": svc.get("user_prompt", ""),
                    "confidence": svc.get("confidence", ""),
                    "trigger_evidence": svc.get("trigger_evidence", ""),
                })

    results.sort(key=lambda x: (x["recording_id"], x["time_center"]))
    return results


# ===========================================================================
# Matching
# ===========================================================================

def match_predictions_to_gt(predictions, ground_truth, tolerance=30.0):
    """
    Greedy matching per recording: for each GT, find closest unmatched
    prediction within tolerance.
    """
    pred_by_rec = defaultdict(list)
    gt_by_rec = defaultdict(list)
    for p in predictions:
        pred_by_rec[p["recording_id"]].append(p)
    for g in ground_truth:
        gt_by_rec[g["recording_id"]].append(g)

    all_recs = set(list(pred_by_rec.keys()) + list(gt_by_rec.keys()))

    matched = []
    missed_gt = []
    redundant_pred = []

    for rec in all_recs:
        r_preds = pred_by_rec.get(rec, [])
        r_gts = gt_by_rec.get(rec, [])

        used_pred = set()

        for gt in r_gts:
            gt_start = gt["time_start"]
            gt_end = gt["time_end"]
            best_idx = None
            best_dist = float('inf')

            for i, pred in enumerate(r_preds):
                if i in used_pred:
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
                matched.append((r_preds[best_idx], gt))
                used_pred.add(best_idx)
            else:
                missed_gt.append(gt)

        for i, pred in enumerate(r_preds):
            if i not in used_pred:
                redundant_pred.append(pred)

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
    r = re.search(rf'{key}:\s*(\d+)', text, re.IGNORECASE)
    if r:
        return min(int(r.group(1)), 5)
    return None


def llm_score_matched_pairs(matched_pairs, model_name="gpt-4o-mini"):
    """
    Score each matched pair using GPT-4o-mini.
    Only scores pairs where the sub-type matches exactly.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        logger.error(f"Failed to init OpenAI client: {e}")
        return []

    all_scores = []
    for pred, gt in tqdm(matched_pairs, desc=f"LLM scoring ({model_name})"):
        if pred["sub_type"] != gt["sub_type"]:
            all_scores.append(None)
            continue

        gt_type_str = f"{gt['main_type']} / {gt['sub_type']}"
        pred_type_str = f"{pred['main_type']} / {pred['sub_type']}"

        scores = {}

        # --- Rationality ---
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

        # --- Effectiveness ---
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
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CaptionCook4D Proactive Service Evaluation — Sub-Type Level"
    )
    parser.add_argument(
        "--prediction_base", type=str,
        default=paths.CC4D_PRED,
    )
    parser.add_argument(
        "--gt_path", type=str,
        default=paths.CC4D_GT,
    )
    parser.add_argument(
        "--response_name", type=str, default="proactive_response_gpt_5_wo_visual",
        help="Name of prediction JSON file (without .json)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=25.0,
    )
    parser.add_argument(
        "--no_llm_scoring", action="store_true", default=False,
        help="Skip LLM-based scoring",
    )
    parser.add_argument(
        "--llm_model", type=str, default="gpt-4o-mini",
    )
    parser.add_argument(
        "--output", type=str, default=None,
    )
    args = parser.parse_args()

    # ---- Discover valid recordings (those with prediction dirs) ----
    pred_recordings = set()
    if os.path.isdir(args.prediction_base):
        suffix = "_qwenvl_3_8b_instruct"
        for d in os.listdir(args.prediction_base):
            if d.endswith(suffix):
                rec_id = d[:-len(suffix)]
                fpath = os.path.join(args.prediction_base, d, f"{args.response_name}.json")
                if os.path.exists(fpath):
                    pred_recordings.add(rec_id)

    # 剔除 SKIP_VIDEOS 中指定的视频
    if SKIP_VIDEOS:
        skipped = pred_recordings & SKIP_VIDEOS
        pred_recordings -= SKIP_VIDEOS
        logger.info(f"Skipped {len(skipped)} videos: {sorted(skipped)}")

    logger.info(f"Prediction recordings: {len(pred_recordings)}")

    if not pred_recordings:
        logger.error("No prediction recordings found. Check paths.")
        return

    # ---- Extract GT and predictions ----
    gt_all = extract_ground_truth(args.gt_path, pred_recordings)
    pred_all = extract_predictions(args.prediction_base, pred_recordings, args.response_name)

    # Count GT recordings that actually have dialogue items
    gt_rec_ids = set(g["recording_id"] for g in gt_all)
    logger.info(f"GT recordings with valid items: {len(gt_rec_ids)}")

    logger.info(f"\nGT annotations: {len(gt_all)}")
    for st in SUB_TYPES:
        cnt = sum(1 for g in gt_all if g["sub_type"] == st)
        if cnt > 0:
            logger.info(f"  {st}: {cnt}")

    logger.info(f"Predictions: {len(pred_all)}")
    for st in SUB_TYPES:
        cnt = sum(1 for p in pred_all if p["sub_type"] == st)
        if cnt > 0:
            logger.info(f"  {st}: {cnt}")

    # Sanity check: all predictions map to known sub_types
    known_sub_pred = sum(1 for p in pred_all if p["sub_type"] in SUB_TYPES)
    if known_sub_pred != len(pred_all):
        unmapped = [p for p in pred_all if p["sub_type"] not in SUB_TYPES]
        unmapped_types = set(p["sub_type"] for p in unmapped)
        logger.error(
            f"SUB-TYPE MISMATCH: {known_sub_pred} mapped vs {len(pred_all)} total. "
            f"Unmapped: {unmapped_types}"
        )

    # Sanity check: all GT items map to known sub_types
    known_sub_gt = sum(1 for g in gt_all if g["sub_type"] in SUB_TYPES)
    if known_sub_gt != len(gt_all):
        unmapped_gt = [g for g in gt_all if g["sub_type"] not in SUB_TYPES]
        unmapped_gt_types = set(g["sub_type"] for g in unmapped_gt)
        logger.error(
            f"GT SUB-TYPE MISMATCH: {known_sub_gt} mapped vs {len(gt_all)} total. "
            f"Unmapped: {unmapped_gt_types}"
        )

    # ================================================================
    # Per main-type matching
    # ================================================================
    main_type_results = {}
    all_matched_main = []

    for mt in MAIN_TYPES:
        sub_types_for_mt = [st for st, m in MAIN_TYPE_MAP.items() if m == mt]
        preds_mt = [p for p in pred_all if p["sub_type"] in sub_types_for_mt]
        gts_mt = [g for g in gt_all if g["sub_type"] in sub_types_for_mt]

        matched_mt, missed_mt, redundant_mt = match_predictions_to_gt(
            preds_mt, gts_mt, tolerance=args.tolerance
        )
        p, r, f = compute_prf(len(matched_mt), len(preds_mt), len(gts_mt))
        main_type_results[mt] = {
            "precision": p, "recall": r, "f1": f,
            "num_pred": len(preds_mt), "num_gt": len(gts_mt),
            "num_matched": len(matched_mt),
        }
        all_matched_main.extend(matched_mt)

    # ================================================================
    # Per sub-type evaluation (strict match)
    # ================================================================
    sub_type_results = {}

    for st in SUB_TYPES:
        preds_st = [p for p in pred_all if p["sub_type"] == st]
        gts_st = [g for g in gt_all if g["sub_type"] == st]

        matched_st, missed_st, redundant_st = match_predictions_to_gt(
            preds_st, gts_st, tolerance=args.tolerance
        )

        p, r, f = compute_prf(len(matched_st), len(preds_st), len(gts_st))
        sub_type_results[st] = {
            "precision": p, "recall": r, "f1": f,
            "num_pred": len(preds_st), "num_gt": len(gts_st),
            "num_matched": len(matched_st),
        }

    # ================================================================
    # Macro average across sub-types
    # ================================================================
    sub_p_list, sub_r_list, sub_f1_list = [], [], []
    for st in SUB_TYPES:
        s = sub_type_results[st]
        if s["num_gt"] > 0 or s["num_pred"] > 0:
            sub_p_list.append(s["precision"])
            sub_r_list.append(s["recall"])
            sub_f1_list.append(s["f1"])

    if sub_p_list:
        macro_p = np.mean(sub_p_list)
        macro_r = np.mean(sub_r_list)
        macro_f1 = np.mean(sub_f1_list)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    # ================================================================
    # Overall (from main-type aggregation)
    # ================================================================
    total_pred = len(pred_all)
    total_gt = len(gt_all)
    total_matched = sum(main_type_results[mt]["num_matched"] for mt in MAIN_TYPES)
    p_all, r_all, f_all = compute_prf(total_matched, total_pred, total_gt)

    # ================================================================
    # Print results
    # ================================================================
    logger.info(f"\n{'='*70}")
    logger.info("PER SUB-TYPE RESULTS (strict sub-type match)")
    logger.info(f"{'='*70}")
    for st in SUB_TYPES:
        r = sub_type_results[st]
        if r["num_gt"] > 0 or r["num_pred"] > 0:
            mt_label = MAIN_TYPE_MAP[st]
            logger.info(
                f"  [{mt_label:10s}] {st:22s}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                f"F1={r['f1']:.3f}  (pred={r['num_pred']}, gt={r['num_gt']}, "
                f"matched={r['num_matched']})"
            )

    logger.info(f"\n  --- Sub-Type Macro Average ---")
    logger.info(
        f"  {'Macro-Avg':22s}  P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}  "
        f"(over {len(sub_p_list)} active sub-types)"
    )

    logger.info(f"\n{'='*70}")
    logger.info("PER MAIN-TYPE RESULTS")
    logger.info(f"{'='*70}")
    for mt in MAIN_TYPES:
        r = main_type_results[mt]
        if r["num_gt"] > 0 or r["num_pred"] > 0:
            logger.info(
                f"  {mt:12s}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                f"F1={r['f1']:.3f}  (pred={r['num_pred']}, gt={r['num_gt']}, "
                f"matched={r['num_matched']})"
            )

    logger.info(f"\n{'='*70}")
    logger.info("OVERALL RESULTS")
    logger.info(f"{'='*70}")
    logger.info(
        f"  {'Overall':12s}  P={p_all:.3f}  R={r_all:.3f}  F1={f_all:.3f}  "
        f"(pred={total_pred}, gt={total_gt}, matched={total_matched})"
    )

    # ================================================================
    # LLM scoring (only for sub-type exact matches)
    # ================================================================
    llm_results = {}
    llm_per_sub = {}
    if not args.no_llm_scoring and all_matched_main:
        subtype_matched = [(pred, gt) for pred, gt in all_matched_main
                           if pred["sub_type"] == gt["sub_type"]]
        logger.info(
            f"\nLLM scoring: {len(subtype_matched)} pairs with exact sub-type match "
            f"(out of {len(all_matched_main)} total matched)"
        )

        if subtype_matched:
            scores = llm_score_matched_pairs(
                subtype_matched, model_name=args.llm_model
            )

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

            # Per sub-type LLM scores
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
                    f"    [{mt_label:10s}] {st:22s}  "
                    f"Rationality={st_avg_r:.3f}  Effectiveness={st_avg_e:.3f}  "
                    f"(n={len(ss)})"
                )

    # ================================================================
    # Save results
    # ================================================================
    output_data = {
        "config": {
            "prediction_base": args.prediction_base,
            "gt_path": args.gt_path,
            "response_name": args.response_name,
            "tolerance": args.tolerance,
            "llm_model": args.llm_model,
            "num_recordings": len(pred_recordings),
        },
        "per_sub_type": {},
        "sub_type_macro_average": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
            "num_active_sub_types": len(sub_p_list),
        },
        "per_main_type": {},
        "overall": {
            "precision": round(p_all, 4),
            "recall": round(r_all, 4),
            "f1": round(f_all, 4),
            "num_pred": total_pred,
            "num_gt": total_gt,
            "num_matched": total_matched,
        },
        "llm_scores": llm_results,
        "llm_scores_per_sub_type": llm_per_sub,
    }

    for st in SUB_TYPES:
        r = sub_type_results[st]
        output_data["per_sub_type"][st] = {
            "main_type": MAIN_TYPE_MAP[st],
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1"], 4),
            "num_pred": r["num_pred"],
            "num_gt": r["num_gt"],
            "num_matched": r["num_matched"],
        }

    for mt in MAIN_TYPES:
        r = main_type_results[mt]
        output_data["per_main_type"][mt] = {
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1"], 4),
            "num_pred": r["num_pred"],
            "num_gt": r["num_gt"],
            "num_matched": r["num_matched"],
        }

    output_path = args.output or os.path.join(
        os.path.dirname(__file__),
        "captioncook4d_proactive_eval_subtype_results.json"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
