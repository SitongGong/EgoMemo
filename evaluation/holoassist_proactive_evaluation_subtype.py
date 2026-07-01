"""
HoloAssist Proactive Service Evaluation Script — Sub-Type Level.

Extended version of holoassist_proactive_evaluation.py that adds:
  1. Per sub-type Precision / Recall / F1 (with strict sub-type matching)
  2. Per main-type P / R / F1
  3. Macro-averaged P / R / F1 across all sub-types
  4. GPT-4o-mini text scoring (Rationality & Effectiveness) for
     matched pairs whose sub-type also matches exactly

Usage:
  python holoassist_proactive_evaluation_subtype.py

  # With LLM scoring enabled:
  python holoassist_proactive_evaluation_subtype.py  (default: LLM scoring ON)

  # Without LLM scoring:
  python holoassist_proactive_evaluation_subtype.py --no_llm_scoring
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

# Videos excluded from evaluation (kept identical to combined_proactive_evaluation_subtype.py).
HOLOASSIST_SKIP_VIDEOS = {
    "R073-20July-SmallPrinter",
    "z040-june-23-22-printer_big",
    "z052-july-11-22-printer_big",
    "z062-june-29-22-printer_small-bad",
    "z109-july-27-22-printer_small",
    "z114-aug-03-22-espresso",
    "z117-aug-05-22-printer_big",
    "z127-aug-10-22-printer_big",
    "z132-aug-12-22-printer_big",
    "z146-aug-19-22-printer_big",
    "z171-sep-03-22-printer_big",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================

MAIN_TYPE_MAP = {
    "safety":             "Instant",
    "tool_use":           "Instant",
    "next_step_guidance": "Short-Term",
    "error_recovery":     "Short-Term",
    "resource_reminder":  "Short-Term",
}

SUB_TYPES = list(MAIN_TYPE_MAP.keys())

GT_SERVICE_DIRS = [
    ("instant/safety",               "safety",             "safety_instant_events"),
    ("instant/tool_use",             "tool_use",           "tool_use_instant_events"),
    ("short_term/error_recovery",    "error_recovery",     "error_recovery_events"),
    ("short_term/next_step_guidance","next_step_guidance",  "next_step_events"),
    ("short_term/resource_reminder", "resource_reminder",   "resource_reminder_events"),
]

# Normalize prediction service_main_type
# Built from actual model outputs across HoloAssist predictions.
PRED_MAIN_TYPE_NORMALIZE = {
    "instant":                          "Instant",
    "instant proactive service":        "Instant",
    "instant proactive services":       "Instant",
    "short-term":                       "Short-Term",
    "short-term proactive service":     "Short-Term",
    "short-term proactive services":    "Short-Term",
}

# Normalize prediction service_sub_type to canonical GT sub_type.
# Built from actual model outputs: Safety, Tool Use, Error-Recovery,
# Next-Step Guidance, Resource Reminder.
PRED_SUB_TYPE_NORMALIZE = {
    # safety
    "safety":                   "safety",
    # tool_use
    "tool use":                 "tool_use",
    "tool_use":                 "tool_use",
    # error_recovery
    "error-recovery":           "error_recovery",
    "error_recovery":           "error_recovery",
    "error recovery":           "error_recovery",
    # next_step_guidance
    "next-step guidance":       "next_step_guidance",
    "next_step_guidance":       "next_step_guidance",
    "next step guidance":       "next_step_guidance",
    # resource_reminder
    "resource reminder":        "resource_reminder",
    "resource_reminder":        "resource_reminder",
}

MAIN_TYPES = ["Instant", "Short-Term"]

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

def parse_gt_time_window(time_window_str: str):
    """
    Parse HoloAssist GT time_window strings. Three formats:
      1) Seconds:   "19.503-21.433"
      2) HH:MM:SS:  "00:00:19.503-00:00:21.433"
      3) MM:SS:     "01:24.100-01:24.700"   (分:秒, 部分视频用此格式)
    Returns (start_seconds, end_seconds) or None.
    """
    if not time_window_str:
        return None

    if ':' in time_window_str:
        # 同时支持 HH:MM:SS 和 MM:SS 两种(冒号段数不同)
        match = re.match(
            r'(\d{1,2}(?::\d{2})+(?:\.\d+)?)\s*-\s*(\d{1,2}(?::\d{2})+(?:\.\d+)?)',
            time_window_str
        )
        if not match:
            return None

        def hms_to_seconds(t):
            p = t.split(':')
            # 2段=MM:SS, 3段=HH:MM:SS
            if len(p) == 2:
                return int(p[0]) * 60 + float(p[1])
            return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])

        return hms_to_seconds(match.group(1)), hms_to_seconds(match.group(2))

    parts = time_window_str.split('-')
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    return None


def parse_pred_time_window(time_window_str: str):
    """
    Parse prediction time_window: "DAY1 00:00:06-00:00:08"
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

def extract_ground_truth(gt_base: str, video_names: list):
    """
    Extract GT annotations. When error_recovery and tool_use share the same
    time_window for a video, only error_recovery is kept.
    """
    video_entries = defaultdict(list)

    for subdir, sub_type, events_key in GT_SERVICE_DIRS:
        main_type = MAIN_TYPE_MAP[sub_type]
        cat_dir = os.path.join(gt_base, subdir)
        if not os.path.isdir(cat_dir):
            continue

        for video_name in video_names:
            fpath = os.path.join(cat_dir, f"{video_name}.json")
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}")
                continue

            dialogs = data.get("dialogs", [])
            for dialog_entry in dialogs:
                tw_str = dialog_entry.get("time_window", "")
                parsed = parse_gt_time_window(tw_str)
                if parsed is None:
                    continue
                start_s, end_s = parsed

                utterance = ""
                for turn in dialog_entry.get("dialogue", []):
                    if turn.get("role") == "assistant":
                        utterance = turn.get("utterance", "").strip()
                        break
                if not utterance:
                    continue

                observation = dialog_entry.get("observation", "")

                video_entries[video_name].append({
                    "video_name": video_name,
                    "main_type": main_type,
                    "sub_type": sub_type,
                    "time_start": start_s,
                    "time_end": end_s,
                    "time_center": (start_s + end_s) / 2.0,
                    "user_prompt": utterance,
                    "observation": observation,
                    "raw_time_window": tw_str,
                })

    # Resolve conflicts: error_recovery > tool_use at same time
    results = []
    for video_name, entries in video_entries.items():
        time_groups = defaultdict(list)
        for e in entries:
            key = (round(e["time_start"], 2), round(e["time_end"], 2))
            time_groups[key].append(e)

        for key, group in time_groups.items():
            if len(group) == 1:
                results.append(group[0])
            else:
                sub_types_in_group = {e["sub_type"] for e in group}
                if "error_recovery" in sub_types_in_group and "tool_use" in sub_types_in_group:
                    for e in group:
                        # if e["sub_type"] == "error_recovery":
                        if e["sub_type"] == "tool_use":
                            results.append(e)
                            break
                else:
                    results.extend(group)

    results.sort(key=lambda x: (x["video_name"], x["time_center"]))
    return results


# ===========================================================================
# Prediction extraction
# ===========================================================================

def extract_predictions(pred_base: str, video_names: list, response_name: str):
    results = []
    count = 0
    for video_name in video_names:
        working_dir = os.path.join(pred_base, f"{video_name}-qwenvl_3_8b_instruct")
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
                main_type = PRED_MAIN_TYPE_NORMALIZE.get(
                    raw_main.lower().strip(), None
                )
                if main_type is None:
                    logger.warning(
                        f"Unmapped main_type: '{raw_main}' — add to PRED_MAIN_TYPE_NORMALIZE"
                    )
                    main_type = raw_main

                raw_sub = svc.get("service_sub_type", "")
                sub_type = PRED_SUB_TYPE_NORMALIZE.get(
                    raw_sub.lower().strip(), None
                )
                if sub_type is None:
                    logger.warning(
                        f"Unmapped sub_type: '{raw_sub}' — add to PRED_SUB_TYPE_NORMALIZE"
                    )
                    sub_type = raw_sub

                results.append({
                    "video_name": video_name,
                    "main_type": main_type,
                    "sub_type": sub_type,
                    "time_start": start_s,
                    "time_end": end_s,
                    "time_center": (start_s + end_s) / 2.0,
                    "user_prompt": svc.get("user_prompt", ""),
                    "confidence": svc.get("confidence", ""),
                    "trigger_evidence": svc.get("trigger_evidence", ""),
                })

    results.sort(key=lambda x: (x["video_name"], x["time_center"]))
    return results


# ===========================================================================
# Matching
# ===========================================================================

def match_predictions_to_gt(predictions, ground_truth, tolerance=30.0):
    """
    Greedy matching per video: for each GT, find closest unmatched prediction
    within tolerance.
    """
    pred_by_video = defaultdict(list)
    gt_by_video = defaultdict(list)
    for p in predictions:
        pred_by_video[p["video_name"]].append(p)
    for g in ground_truth:
        gt_by_video[g["video_name"]].append(g)

    all_videos = set(list(pred_by_video.keys()) + list(gt_by_video.keys()))

    matched = []
    missed_gt = []
    redundant_pred = []

    for video in all_videos:
        v_preds = pred_by_video.get(video, [])
        v_gts = gt_by_video.get(video, [])

        used_pred = set()

        for gt in v_gts:
            gt_start = gt["time_start"]
            gt_end = gt["time_end"]
            best_idx = None
            best_dist = float('inf')

            for i, pred in enumerate(v_preds):
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
                matched.append((v_preds[best_idx], gt))
                used_pred.add(best_idx)
            else:
                missed_gt.append(gt)

        for i, pred in enumerate(v_preds):
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
    Returns list of score dicts (or None for unscored pairs).
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
        description="HoloAssist Proactive Service Evaluation — Sub-Type Level"
    )
    parser.add_argument(
        "--prediction_base", type=str,
        default=paths.HOLOASSIST_PRED,
    )
    parser.add_argument(
        "--gt_base", type=str,
        default=paths.HOLOASSIST_GT,
    )
    parser.add_argument(
        "--response_name", type=str, default="proactive_response_gpt_5_wo_entity",
    )
    parser.add_argument(
        "--tolerance", type=float, default=10.0,
    )
    parser.add_argument(
        "--no_llm_scoring", action="store_true", default=False,
        help="Skip LLM-based scoring",
    )
    parser.add_argument(
        "--llm_model", type=str, default="gpt-4o-mini",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(paths.OUTPUT_ROOT, "holoassist_proactive_eval_subtype_results.json"),
    )
    args = parser.parse_args()

    # ---- Discover common videos ----
    pred_videos = set()
    if os.path.isdir(args.prediction_base):
        for d in os.listdir(args.prediction_base):
            if d.endswith("-qwenvl_3_8b_instruct"):
                video_name = d[:-len("-qwenvl_3_8b_instruct")]
                fpath = os.path.join(args.prediction_base, d, f"{args.response_name}.json")
                if os.path.exists(fpath):
                    pred_videos.add(video_name)

    gt_videos = set()
    for subdir, sub_type, events_key in GT_SERVICE_DIRS:
        cat_dir = os.path.join(args.gt_base, subdir)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if fname.endswith(".json"):
                gt_videos.add(fname[:-5])

    common_videos_all = sorted(pred_videos & gt_videos)

    # Exclude HOLOASSIST_SKIP_VIDEOS up-front
    common_videos_all = [v for v in common_videos_all if v not in HOLOASSIST_SKIP_VIDEOS]

    # Filter: only keep videos that have at least one GT dialogue entry
    common_videos = []
    for video_name in common_videos_all:
        has_gt = False
        for subdir, sub_type, events_key in GT_SERVICE_DIRS:
            fpath = os.path.join(args.gt_base, subdir, f"{video_name}.json")
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("dialogs"):
                    has_gt = True
                    break
            except Exception:
                continue
        if has_gt:
            common_videos.append(video_name)

    logger.info(f"Prediction videos: {len(pred_videos)}")
    logger.info(f"GT videos: {len(gt_videos)}")
    logger.info(f"Skipped (HOLOASSIST_SKIP_VIDEOS): {len(HOLOASSIST_SKIP_VIDEOS)}")
    logger.info(f"Common videos (with GT): {len(common_videos)} "
                f"(filtered out {len(common_videos_all) - len(common_videos)} without GT)")

    if not common_videos:
        logger.error("No common videos found. Check paths.")
        return

    # ---- Extract GT and predictions ----
    gt_all = extract_ground_truth(args.gt_base, common_videos)
    pred_all = extract_predictions(args.prediction_base, common_videos, args.response_name)

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

    # ================================================================
    # Per main-type matching (for overall aggregation)
    # ================================================================
    main_type_results = {}
    all_matched_main = []  # matched pairs from main-type level matching

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
    # Per sub-type evaluation (strict: both pred and GT must have same sub_type)
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
            "gt_base": args.gt_base,
            "response_name": args.response_name,
            "tolerance": args.tolerance,
            "llm_model": args.llm_model,
            "num_common_videos": len(common_videos),
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
        "holoassist_proactive_eval_subtype_results.json"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
