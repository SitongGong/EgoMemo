"""
Combined Proactive Service Evaluation — EgoLife + HoloAssist + CaptionCook4D.

Runs all three dataset-specific subtype evaluations, then aggregates raw counts
(num_pred, num_gt, num_matched) across datasets to produce unified per-sub-type,
per-main-type, macro-average, and overall P / R / F1.

Also runs GPT-4o-mini text scoring on all sub-type-matched pairs across datasets.

Usage:
  python combined_proactive_evaluation_subtype.py --no_llm_scoring
  python combined_proactive_evaluation_subtype.py   # with LLM scoring
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add evaluation dir to path so we can import siblings
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

# ===========================================================================
# Import dataset-specific modules
# ===========================================================================
import paths
import egolife_proactive_evaluation_subtype as ego_eval
import holoassist_proactive_evaluation_subtype as holo_eval
import captioncook4d_proactive_evaluation_subtype as cc4d_eval

# ===========================================================================
# Constants — union of all sub-types
# ===========================================================================
MAIN_TYPE_MAP = {
    "safety":               "Instant",
    "tool_use":             "Instant",
    "next_step_guidance":   "Short-Term",
    "error_recovery":       "Short-Term",
    "resource_reminder":    "Short-Term",
    "habit_coaching":       "Long-Term",
    "memory_link_contextual": "Long-Term",
    "routine_optimization": "Long-Term",
    "memory_recall":        "Episodic",
    "task_reminder":        "Episodic",
    # personal_progressive 已从 benchmark 移除(camera-ready),不计入评测
}

SUB_TYPES = list(MAIN_TYPE_MAP.keys())
MAIN_TYPES = ["Instant", "Short-Term", "Long-Term", "Episodic"]

# HoloAssist videos with missing ablation prediction files — skip in all experiments
# 原先跳过 11 个打印机/咖啡机视频(标记为 incomplete)。
# 经核查:这 11 个视频在全部 6 个消融 + 2 个 baseline 上均有预测,
# 不再缺失,因此清空跳过列表,评测覆盖 final.json 全部 193 个视频。
HOLOASSIST_SKIP_VIDEOS = set()

# ===========================================================================
# HoloAssist GT — 改用最新清洗版单文件 final.json
#   结构: {video: {instant:{safety:[...],tool_use:[...]},
#                  short_term:{error_recovery:[...],next_step_guidance:[...],resource_reminder:[...]}}}
#   每个事件: {time_window, dialogue:[{role,utterance}], observation, ...}
# 用 monkey-patch 替换 holo_eval.extract_ground_truth, 并在 run_holoassist 里
# 基于 final.json 的 video keys 确定 GT 视频集合。
# ===========================================================================
HOLOASSIST_FINAL_GT = paths.HOLOASSIST_GT
_HOLO_GT_CACHE = {}

def _load_holo_final_gt():
    if "data" not in _HOLO_GT_CACHE:
        with open(HOLOASSIST_FINAL_GT, "r", encoding="utf-8") as f:
            _HOLO_GT_CACHE["data"] = json.load(f)
    return _HOLO_GT_CACHE["data"]

# (main_bucket, sub_type) 遍历顺序
_HOLO_FINAL_DIRS = [
    ("instant",    "safety"),
    ("instant",    "tool_use"),
    ("short_term", "error_recovery"),
    ("short_term", "next_step_guidance"),
    ("short_term", "resource_reminder"),
]

def holo_extract_gt_final(gt_base, video_names):
    """与 holo_eval.extract_ground_truth 输出同格式, 但从 final.json 读取。
    gt_base 参数保留以兼容签名(忽略, 固定读 HOLOASSIST_FINAL_GT)。"""
    data = _load_holo_final_gt()
    video_set = set(video_names)
    video_entries = {}
    for vn in video_names:
        vv = data.get(vn)
        if not vv:
            continue
        for bucket, sub_type in _HOLO_FINAL_DIRS:
            main_type = holo_eval.MAIN_TYPE_MAP[sub_type]
            for e in (vv.get(bucket) or {}).get(sub_type, []) or []:
                parsed = holo_eval.parse_gt_time_window(e.get("time_window", ""))
                if parsed is None:
                    continue
                start_s, end_s = parsed
                utterance = ""
                for turn in e.get("dialogue", []):
                    if turn.get("role") == "assistant":
                        utterance = turn.get("utterance", "").strip(); break
                if not utterance:
                    continue
                video_entries.setdefault(vn, []).append({
                    "video_name": vn, "main_type": main_type, "sub_type": sub_type,
                    "time_start": start_s, "time_end": end_s,
                    "time_center": (start_s + end_s) / 2.0,
                    "user_prompt": utterance,
                    "observation": e.get("observation", ""),
                    "raw_time_window": e.get("time_window", ""),
                })
    # final.json 已清洗冲突, 直接展开
    results = []
    for vn, entries in video_entries.items():
        results.extend(entries)
    results.sort(key=lambda x: (x["video_name"], x["time_center"]))
    return results

# 用 final.json 版替换原 GT 读取
holo_eval.extract_ground_truth = holo_extract_gt_final

# ===========================================================================
# CaptionCook4D — 统一所有模型评测同一批 recording(公平对比)
#   主模型/消融跑了 87 个 recording, baseline 多跑了 2_41/7_50(共89)。
#   取主模型实际有预测的 recording 作为公共集合, 所有配置(含 baseline)都限定在此集合,
#   避免 baseline 因多测 2 个视频导致 num_gt 不一致。
# ===========================================================================
_CC4D_MAIN_DIR = paths.CC4D_PRED
_CC4D_MAIN_RESP = "proactive_response_gpt_5"
_CC4D_COMMON_CACHE = {}

def cc4d_common_recordings():
    """主模型(full)实际有预测的 recording 集合, 作为所有配置的公共评测集。"""
    if "set" not in _CC4D_COMMON_CACHE:
        recs = set()
        suffix = "_qwenvl_3_8b_instruct"
        if os.path.isdir(_CC4D_MAIN_DIR):
            for d in os.listdir(_CC4D_MAIN_DIR):
                if d.endswith(suffix):
                    rid = d[:-len(suffix)]
                    if os.path.exists(os.path.join(_CC4D_MAIN_DIR, d, f"{_CC4D_MAIN_RESP}.json")):
                        recs.add(rid)
        _CC4D_COMMON_CACHE["set"] = recs
    return _CC4D_COMMON_CACHE["set"]

# ===========================================================================
# Metrics
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

LLM_RATIONALITY_PROMPT = ego_eval.LLM_RATIONALITY_PROMPT
LLM_EFFECTIVENESS_PROMPT = ego_eval.LLM_EFFECTIVENESS_PROMPT


def parse_single_score(text: str, key: str):
    r = re.search(rf'{key}:\s*(\d+)', text, re.IGNORECASE)
    if r:
        return min(int(r.group(1)), 5)
    return None


OPENAI_API_KEY = paths.openai_api_key()


def llm_score_matched_pairs(matched_pairs, model_name="gpt-4o"):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to init OpenAI client: {e}")
        return []

    all_scores = []
    for pred, gt in tqdm(matched_pairs, desc=f"LLM scoring ({model_name})"):
        if pred["sub_type"] != gt["sub_type"]:
            all_scores.append(None)
            continue

        # Skip pairs with empty user_prompt
        gt_prompt = (gt.get("user_prompt") or "").strip()
        pred_prompt = (pred.get("user_prompt") or "").strip()
        if not gt_prompt or not pred_prompt:
            all_scores.append(None)
            continue

        gt_type_str = f"{gt['main_type']} / {gt['sub_type']}"
        pred_type_str = f"{pred['main_type']} / {pred['sub_type']}"
        scores = {}

        prompt_r = LLM_RATIONALITY_PROMPT.format(
            gt_service_type=gt_type_str, gt_user_prompt=gt_prompt,
            pred_service_type=pred_type_str, pred_user_prompt=pred_prompt,
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

        prompt_e = LLM_EFFECTIVENESS_PROMPT.format(
            gt_service_type=gt_type_str, gt_user_prompt=gt_prompt,
            pred_service_type=pred_type_str, pred_user_prompt=pred_prompt,
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
            all_scores.append(None)

    return all_scores

# ===========================================================================
# Dataset runners — each returns (per_sub_type_counts, per_main_type_counts,
#                                  overall_counts, matched_pairs)
# where counts = {key: {"num_pred": N, "num_gt": N, "num_matched": N}}
# ===========================================================================

def evaluate_subtype_strict(predictions, ground_truth, sub_types, main_type_map, match_fn, tolerance):
    """
    Evaluate per sub-type with strict sub-type isolation:
    each sub-type's predictions are matched only against that sub-type's GT.

    This ensures per_sub["num_matched"] and matched_pairs are consistent:
    matched_pairs == union of all per-sub_type matches, so
    len(subtype_matched) == sum(per_sub[st]["num_matched"]).

    Returns (per_sub, per_main, overall_counts, matched_pairs).
    """
    per_sub = {}
    all_matched = []

    for st in sub_types:
        preds_st = [p for p in predictions if p["sub_type"] == st]
        gts_st = [g for g in ground_truth if g["sub_type"] == st]
        matched_st, _, _ = match_fn(preds_st, gts_st, tolerance=tolerance)
        per_sub[st] = {
            "num_pred": len(preds_st),
            "num_gt": len(gts_st),
            "num_matched": len(matched_st),
        }
        all_matched.extend(matched_st)

    # Derive per_main from per_sub results (consistent counts)
    per_main = {}
    for mt in set(main_type_map.values()):
        sts_for_mt = [st for st, m in main_type_map.items() if m == mt]
        per_main[mt] = {
            "num_pred":    sum(per_sub.get(st, {}).get("num_pred", 0)    for st in sts_for_mt),
            "num_gt":      sum(per_sub.get(st, {}).get("num_gt", 0)      for st in sts_for_mt),
            "num_matched": sum(per_sub.get(st, {}).get("num_matched", 0) for st in sts_for_mt),
        }

    overall = {
        "num_pred":    len(predictions),
        "num_gt":      len(ground_truth),
        "num_matched": len(all_matched),
    }

    return per_sub, per_main, overall, all_matched


def run_egolife(args):
    """Run EgoLife evaluation and return raw counts + matched pairs.

    Matches per-person (to avoid cross-person matching) and per-sub-type
    (strict isolation), so matched_pairs == sub-type-strict matches.
    """
    persons = args.egolife_persons
    allowed_days = set()
    for d in args.egolife_days:
        allowed_days.add(int(d.replace("DAY", "")))

    per_main = {mt: {"num_pred": 0, "num_gt": 0, "num_matched": 0}
                for mt in ego_eval.MAIN_TYPES}
    per_sub = {st: {"num_pred": 0, "num_gt": 0, "num_matched": 0}
               for st in ego_eval.SUB_TYPES}
    total_pred = 0
    total_gt = 0
    total_matched = 0
    all_matched = []

    for person in persons:
        person_dir = os.path.join(
            args.egolife_prediction_base,
            f"{person}-{args.egolife_caption_model}_restart"
        )
        person_days = set()
        for d in allowed_days:
            fpath = os.path.join(person_dir, f"DAY{d}",
                                 f"{args.egolife_response_name}.json")
            if os.path.exists(fpath):
                person_days.add(d)
        if not person_days:
            continue

        gt = ego_eval.extract_ground_truth(args.egolife_gt_base, person, person_days)
        preds = ego_eval.extract_predictions(
            args.egolife_prediction_base, person,
            args.egolife_caption_model, person_days,
            response_name=args.egolife_response_name
        )

        p_sub, p_main, p_ov, p_matched = evaluate_subtype_strict(
            preds, gt,
            sub_types=ego_eval.SUB_TYPES,
            main_type_map=ego_eval.MAIN_TYPE_MAP,
            match_fn=ego_eval.match_predictions_to_gt,
            tolerance=args.egolife_tolerance,
        )

        for mt in ego_eval.MAIN_TYPES:
            r = p_main.get(mt, {})
            per_main[mt]["num_pred"]    += r.get("num_pred", 0)
            per_main[mt]["num_gt"]      += r.get("num_gt", 0)
            per_main[mt]["num_matched"] += r.get("num_matched", 0)
        for st in ego_eval.SUB_TYPES:
            r = p_sub.get(st, {})
            per_sub[st]["num_pred"]    += r.get("num_pred", 0)
            per_sub[st]["num_gt"]      += r.get("num_gt", 0)
            per_sub[st]["num_matched"] += r.get("num_matched", 0)
        total_pred    += p_ov["num_pred"]
        total_gt      += p_ov["num_gt"]
        total_matched += p_ov["num_matched"]
        all_matched.extend(p_matched)

    logger.info(f"  EgoLife: GT={total_gt}, Pred={total_pred}")

    overall = {"num_pred": total_pred, "num_gt": total_gt, "num_matched": total_matched}
    return per_sub, per_main, overall, all_matched


def run_holoassist(args):
    """Run HoloAssist evaluation and return raw counts + matched pairs."""
    pred_videos = set()
    if os.path.isdir(args.holoassist_prediction_base):
        for d in os.listdir(args.holoassist_prediction_base):
            if d.endswith("-qwenvl_3_8b_instruct"):
                video_name = d[:-len("-qwenvl_3_8b_instruct")]
                fpath = os.path.join(args.holoassist_prediction_base, d,
                                     f"{args.holoassist_response_name}.json")
                if os.path.exists(fpath):
                    pred_videos.add(video_name)

    # GT 视频集合来自 final.json 的 keys
    gt_videos = set(_load_holo_final_gt().keys())

    common_videos_all = sorted((pred_videos & gt_videos) - HOLOASSIST_SKIP_VIDEOS)

    # 只保留 final.json 中至少有一条有效 GT 事件的视频
    gt_all = holo_eval.extract_ground_truth(args.holoassist_gt_base, common_videos_all)
    videos_with_gt = set(g["video_name"] for g in gt_all)
    common_videos = [v for v in common_videos_all if v in videos_with_gt]
    gt_all = [g for g in gt_all if g["video_name"] in set(common_videos)]

    if not common_videos:
        logger.warning("  HoloAssist: no common videos found")
        return {}, {}, {"num_pred": 0, "num_gt": 0, "num_matched": 0}, []

    pred_all = holo_eval.extract_predictions(
        args.holoassist_prediction_base, common_videos, args.holoassist_response_name
    )

    logger.info(f"  HoloAssist: GT={len(gt_all)}, Pred={len(pred_all)}, Videos={len(common_videos)} "
                f"(filtered out {len(common_videos_all) - len(common_videos)} without GT, "
                f"skipped {len(HOLOASSIST_SKIP_VIDEOS)} incomplete videos)")

    per_sub, per_main, overall, all_matched = evaluate_subtype_strict(
        pred_all, gt_all,
        sub_types=holo_eval.SUB_TYPES,
        main_type_map=holo_eval.MAIN_TYPE_MAP,
        match_fn=holo_eval.match_predictions_to_gt,
        tolerance=args.holoassist_tolerance,
    )
    return per_sub, per_main, overall, all_matched


def run_captioncook4d(args):
    """Run CaptionCook4D evaluation and return raw counts + matched pairs."""
    pred_recordings = set()
    suffix = "_qwenvl_3_8b_instruct"
    if os.path.isdir(args.cc4d_prediction_base):
        for d in os.listdir(args.cc4d_prediction_base):
            if d.endswith(suffix):
                rec_id = d[:-len(suffix)]
                fpath = os.path.join(args.cc4d_prediction_base, d,
                                     f"{args.cc4d_response_name}.json")
                if os.path.exists(fpath):
                    pred_recordings.add(rec_id)

    # 限定到所有模型的公共 recording 集合(保证 8 个配置评测同一批视频)
    common = cc4d_common_recordings()
    if common:
        pred_recordings = pred_recordings & common
    if not pred_recordings:
        logger.warning("  CaptionCook4D: no prediction recordings found")
        return {}, {}, {"num_pred": 0, "num_gt": 0, "num_matched": 0}, []

    gt_all = cc4d_eval.extract_ground_truth(args.cc4d_gt_path, pred_recordings)
    pred_all = cc4d_eval.extract_predictions(
        args.cc4d_prediction_base, pred_recordings, args.cc4d_response_name
    )

    logger.info(f"  CaptionCook4D: GT={len(gt_all)}, Pred={len(pred_all)}, Recordings={len(pred_recordings)}")

    per_sub, per_main, overall, all_matched = evaluate_subtype_strict(
        pred_all, gt_all,
        sub_types=cc4d_eval.SUB_TYPES,
        main_type_map=cc4d_eval.MAIN_TYPE_MAP,
        match_fn=cc4d_eval.match_predictions_to_gt,
        tolerance=args.cc4d_tolerance,
    )
    return per_sub, per_main, overall, all_matched


# ===========================================================================
# Baseline prediction extraction (GPT / Qwen flat-file format)
# ===========================================================================

def _parse_egolife_day_time_window(tw_str: str):
    """Parse EgoLife time_window like 'DAY1 11:09:42-11:10:00', return abs seconds."""
    import re
    m = re.match(r'DAY(\d+)\s+(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)-(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)', tw_str or "")
    if not m:
        return None, None
    def hms(t):
        p = t.split(':')
        return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
    day_n = int(m.group(1))
    seg_start = hms(m.group(2))
    return day_n, seg_start


def extract_egolife_baseline(pred_dir: str, persons: list, allowed_days: set):
    """Extract predictions from GPT/Qwen flat EgoLife files: {PERSON}_{DAY}_proactive.json."""
    results_by_person = defaultdict(list)
    for person in persons:
        for day_n in sorted(allowed_days):
            fname = f"{person}_DAY{day_n}_proactive.json"
            fpath = os.path.join(pred_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    segs = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}")
                continue
            for seg in segs:
                tw = seg.get("time_window", "")
                _, seg_start_s = _parse_egolife_day_time_window(tw)
                if seg_start_s is None:
                    continue
                resp = seg.get("response", {})
                if not isinstance(resp, dict):
                    continue
                for svc in resp.get("services", []):
                    if not isinstance(svc, dict):
                        continue
                    time_span = svc.get("time_span", [0, 0])
                    if not time_span or len(time_span) < 2:
                        continue
                    # seg_start_s is seconds-of-day; add day offset to match GT time encoding
                    day_offset = day_n * 86400
                    abs_start = day_offset + seg_start_s + float(time_span[0])
                    abs_end   = day_offset + seg_start_s + float(time_span[1])
                    raw_sub = svc.get("service_sub_type", "")
                    sub_type = ego_eval.PRED_SUB_TYPE_NORMALIZE.get(raw_sub.lower().strip(), raw_sub.lower().strip())
                    raw_main = svc.get("service_main_type", "")
                    main_type = ego_eval.PRED_MAIN_TYPE_NORMALIZE.get(raw_main.lower().strip(), raw_main)
                    results_by_person[person].append({
                        "person": person,
                        "main_type": main_type,
                        "sub_type": sub_type,
                        "time_start": abs_start,
                        "time_end": abs_end,
                        "time_center": (abs_start + abs_end) / 2.0,
                        "user_prompt": svc.get("user_prompt", ""),
                        "confidence": svc.get("confidence", ""),
                    })
    return results_by_person


def extract_holoassist_baseline(pred_dir: str, video_names: list):
    """Extract predictions from GPT/Qwen flat HoloAssist files: {video_id}.json."""
    results = []
    for video_name in video_names:
        fpath = os.path.join(pred_dir, f"{video_name}.json")
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {fpath}: {e}")
            continue
        if not isinstance(data, dict):
            continue
        for win in data.get("windows", []):
            start_sec = float(win.get("start_sec", 0))
            resp = win.get("response", {})
            if not isinstance(resp, dict):
                continue
            for svc in resp.get("services", []):
                if not isinstance(svc, dict):
                    continue
                time_span = svc.get("time_span", [0, 0])
                if not time_span or len(time_span) < 2:
                    continue
                abs_start = start_sec + float(time_span[0])
                abs_end = start_sec + float(time_span[1])
                raw_sub = svc.get("service_sub_type", "")
                sub_type = holo_eval.PRED_SUB_TYPE_NORMALIZE.get(raw_sub.lower().strip(), raw_sub.lower().strip())
                raw_main = svc.get("service_main_type", "")
                main_type = holo_eval.PRED_MAIN_TYPE_NORMALIZE.get(raw_main.lower().strip(), raw_main)
                results.append({
                    "video_name": video_name,
                    "main_type": main_type,
                    "sub_type": sub_type,
                    "time_start": abs_start,
                    "time_end": abs_end,
                    "time_center": (abs_start + abs_end) / 2.0,
                    "user_prompt": svc.get("user_prompt", ""),
                    "confidence": svc.get("confidence", ""),
                })
    results.sort(key=lambda x: (x["video_name"], x["time_center"]))
    return results


def extract_captioncook4d_baseline(pred_dir: str, rec_ids: set):
    """Extract predictions from GPT/Qwen flat CC4D files: {rec_id}.json."""
    results = []
    for rec_id in rec_ids:
        fpath = os.path.join(pred_dir, f"{rec_id}.json")
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {fpath}: {e}")
            continue
        if not isinstance(data, dict):
            continue
        for win in data.get("windows", []):
            start_sec = float(win.get("start_sec", 0))
            resp = win.get("response", {})
            if not isinstance(resp, dict):
                continue
            for svc in resp.get("services", []):
                if not isinstance(svc, dict):
                    continue
                time_span = svc.get("time_span", [0, 0])
                if not time_span or len(time_span) < 2:
                    continue
                abs_start = start_sec + float(time_span[0])
                abs_end = start_sec + float(time_span[1])
                raw_sub = svc.get("service_sub_type", "")
                sub_type = cc4d_eval.SUB_TYPE_NORMALIZE.get(raw_sub.lower().strip(), raw_sub.lower().strip())
                raw_main = svc.get("service_main_type", "")
                main_type = cc4d_eval.MAIN_TYPE_NORMALIZE.get(raw_main.lower().strip(), raw_main)
                results.append({
                    "recording_id": rec_id,
                    "main_type": main_type,
                    "sub_type": sub_type,
                    "time_start": abs_start,
                    "time_end": abs_end,
                    "time_center": (abs_start + abs_end) / 2.0,
                    "user_prompt": svc.get("user_prompt", ""),
                    "confidence": svc.get("confidence", ""),
                })
    results.sort(key=lambda x: (x["recording_id"], x["time_center"]))
    return results


def run_egolife_baseline(args, pred_dir: str):
    """Run EgoLife baseline evaluation (GPT/Qwen flat files), per-person matching."""
    allowed_days = set()
    for d in args.egolife_days:
        allowed_days.add(int(d.replace("DAY", "")))
    persons = args.egolife_persons

    preds_by_person = extract_egolife_baseline(pred_dir, persons, allowed_days)

    per_main = {mt: {"num_pred": 0, "num_gt": 0, "num_matched": 0} for mt in ego_eval.MAIN_TYPES}
    per_sub = {st: {"num_pred": 0, "num_gt": 0, "num_matched": 0} for st in ego_eval.SUB_TYPES}
    total_pred = total_gt = total_matched = 0
    all_matched = []

    for person in persons:
        person_days = set()
        for d in allowed_days:
            fname = f"{person}_DAY{d}_proactive.json"
            if os.path.exists(os.path.join(pred_dir, fname)):
                person_days.add(d)
        if not person_days:
            continue
        gt = ego_eval.extract_ground_truth(args.egolife_gt_base, person, person_days)
        preds = preds_by_person.get(person, [])

        p_sub, p_main, p_ov, p_matched = evaluate_subtype_strict(
            preds, gt,
            sub_types=ego_eval.SUB_TYPES,
            main_type_map=ego_eval.MAIN_TYPE_MAP,
            match_fn=ego_eval.match_predictions_to_gt,
            tolerance=args.egolife_tolerance,
        )
        for mt in ego_eval.MAIN_TYPES:
            r = p_main.get(mt, {})
            per_main[mt]["num_pred"]    += r.get("num_pred", 0)
            per_main[mt]["num_gt"]      += r.get("num_gt", 0)
            per_main[mt]["num_matched"] += r.get("num_matched", 0)
        for st in ego_eval.SUB_TYPES:
            r = p_sub.get(st, {})
            per_sub[st]["num_pred"]    += r.get("num_pred", 0)
            per_sub[st]["num_gt"]      += r.get("num_gt", 0)
            per_sub[st]["num_matched"] += r.get("num_matched", 0)
        total_pred    += p_ov["num_pred"]
        total_gt      += p_ov["num_gt"]
        total_matched += p_ov["num_matched"]
        all_matched.extend(p_matched)

    logger.info(f"  EgoLife (baseline): GT={total_gt}, Pred={total_pred}")
    overall = {"num_pred": total_pred, "num_gt": total_gt, "num_matched": total_matched}
    return per_sub, per_main, overall, all_matched


def run_holoassist_baseline(args, pred_dir: str):
    """Run HoloAssist baseline evaluation (GPT/Qwen flat files)."""
    # GT 视频集合来自 final.json 的 keys
    gt_videos = set(_load_holo_final_gt().keys())

    # Find videos with baseline predictions
    pred_videos = {f[:-5] for f in os.listdir(pred_dir) if f.endswith(".json")}
    common_videos_all = sorted((pred_videos & gt_videos) - HOLOASSIST_SKIP_VIDEOS)

    # 只保留 final.json 中至少有一条有效 GT 事件的视频
    gt_all = holo_eval.extract_ground_truth(args.holoassist_gt_base, common_videos_all)
    videos_with_gt = set(g["video_name"] for g in gt_all)
    common_videos = [v for v in common_videos_all if v in videos_with_gt]
    gt_all = [g for g in gt_all if g["video_name"] in set(common_videos)]

    if not common_videos:
        logger.warning("  HoloAssist (baseline): no common videos found")
        return {}, {}, {"num_pred": 0, "num_gt": 0, "num_matched": 0}, []

    pred_all = extract_holoassist_baseline(pred_dir, common_videos)
    logger.info(f"  HoloAssist (baseline): GT={len(gt_all)}, Pred={len(pred_all)}, Videos={len(common_videos)}")

    per_sub, per_main, overall, all_matched = evaluate_subtype_strict(
        pred_all, gt_all,
        sub_types=holo_eval.SUB_TYPES,
        main_type_map=holo_eval.MAIN_TYPE_MAP,
        match_fn=holo_eval.match_predictions_to_gt,
        tolerance=args.holoassist_tolerance,
    )
    return per_sub, per_main, overall, all_matched


def run_captioncook4d_baseline(args, pred_dir: str):
    """Run CaptionCook4D baseline evaluation (GPT/Qwen flat files)."""
    rec_ids = {f[:-5] for f in os.listdir(pred_dir) if f.endswith(".json")}
    # 限定到所有模型的公共 recording 集合(排除 baseline 多跑的 2_41/7_50 及噪声文件)
    common = cc4d_common_recordings()
    if common:
        rec_ids = rec_ids & common
    if not rec_ids:
        logger.warning("  CaptionCook4D (baseline): no prediction files found")
        return {}, {}, {"num_pred": 0, "num_gt": 0, "num_matched": 0}, []

    gt_all = cc4d_eval.extract_ground_truth(args.cc4d_gt_path, rec_ids)
    pred_all = extract_captioncook4d_baseline(pred_dir, rec_ids)
    logger.info(f"  CaptionCook4D (baseline): GT={len(gt_all)}, Pred={len(pred_all)}, Recordings={len(rec_ids)}")

    per_sub, per_main, overall, all_matched = evaluate_subtype_strict(
        pred_all, gt_all,
        sub_types=cc4d_eval.SUB_TYPES,
        main_type_map=cc4d_eval.MAIN_TYPE_MAP,
        match_fn=cc4d_eval.match_predictions_to_gt,
        tolerance=args.cc4d_tolerance,
    )
    return per_sub, per_main, overall, all_matched


# ===========================================================================
# Core evaluation logic (shared by single-run and batch run_all)
# ===========================================================================

def _dump_matched_pairs_for_human_eval(ego_matched, holo_matched, cc4d_matched,
                                       dump_path: str):
    """
    把三个数据集的 strict sub-type matched (pred, gt) 对保存成一份便于人工打分的 JSON。
    每条记录包含必要的定位信息（数据集、person/video/recording、时间窗、main/sub_type）
    以及 GT / Pred 的文本，并预留 human_rationality / human_effectiveness / notes 字段。
    """
    def _seconds_to_clock(s):
        try:
            s = float(s)
        except Exception:
            return ""
        sign = "-" if s < 0 else ""
        s = abs(s)
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s - h * 3600 - m * 60
        return f"{sign}{h:02d}:{m:02d}:{sec:06.3f}"

    def _pack(pred, gt, dataset):
        if dataset == "EgoLife":
            ident = {
                "person": gt.get("person") or pred.get("person", ""),
                "day_num": gt.get("day_num") or pred.get("day_num"),
            }
            day_n = ident.get("day_num")
            offset = (day_n * 86400) if isinstance(day_n, int) else 0
            gt_clock = (
                f"DAY{day_n} {_seconds_to_clock(gt.get('time_start', 0) - offset)}"
                f"-{_seconds_to_clock(gt.get('time_end', 0) - offset)}"
                if day_n is not None else ""
            )
            pred_clock = (
                f"DAY{day_n} {_seconds_to_clock(pred.get('time_start', 0) - offset)}"
                f"-{_seconds_to_clock(pred.get('time_end', 0) - offset)}"
                if day_n is not None else ""
            )
        elif dataset == "HoloAssist":
            ident = {"video_name": gt.get("video_name") or pred.get("video_name", "")}
            gt_clock = (
                f"{_seconds_to_clock(gt.get('time_start', 0))}"
                f"-{_seconds_to_clock(gt.get('time_end', 0))}"
            )
            pred_clock = (
                f"{_seconds_to_clock(pred.get('time_start', 0))}"
                f"-{_seconds_to_clock(pred.get('time_end', 0))}"
            )
        else:  # CaptionCook4D
            ident = {"recording_id": gt.get("recording_id") or pred.get("recording_id", "")}
            gt_clock = (
                f"{_seconds_to_clock(gt.get('time_start', 0))}"
                f"-{_seconds_to_clock(gt.get('time_end', 0))}"
            )
            pred_clock = (
                f"{_seconds_to_clock(pred.get('time_start', 0))}"
                f"-{_seconds_to_clock(pred.get('time_end', 0))}"
            )

        return {
            "dataset": dataset,
            **ident,
            "main_type": gt.get("main_type", ""),
            "sub_type": gt.get("sub_type", ""),
            "gt": {
                "time_window": gt_clock,
                "time_start": gt.get("time_start"),
                "time_end": gt.get("time_end"),
                "main_type": gt.get("main_type", ""),
                "sub_type": gt.get("sub_type", ""),
                "user_prompt": gt.get("user_prompt", ""),
            },
            "pred": {
                "time_window": pred_clock,
                "time_start": pred.get("time_start"),
                "time_end": pred.get("time_end"),
                "main_type": pred.get("main_type", ""),
                "sub_type": pred.get("sub_type", ""),
                "user_prompt": pred.get("user_prompt", ""),
                "confidence": pred.get("confidence", ""),
                "trigger_evidence": pred.get("trigger_evidence", ""),
            },
            # —— 人工打分预留字段 ——
            "human_rationality": None,
            "human_effectiveness": None,
            "notes": "",
        }

    pairs = []
    for pred, gt in ego_matched:
        pairs.append(_pack(pred, gt, "EgoLife"))
    for pred, gt in holo_matched:
        pairs.append(_pack(pred, gt, "HoloAssist"))
    for pred, gt in cc4d_matched:
        pairs.append(_pack(pred, gt, "CaptionCook4D"))

    payload = {
        "description": (
            "Strict sub-type matched (Pred, GT) pairs from the FULL VideoRAG model "
            "for human rationality / effectiveness scoring. Each item exposes GT and "
            "Pred text + time windows, plus empty fields for human ratings."
        ),
        "scoring_guide": {
            "rationality": "1-5: semantic similarity between Pred and GT message.",
            "effectiveness": "1-5: how logically/helpfully Pred assists the user.",
        },
        "num_pairs": len(pairs),
        "num_by_dataset": {
            "EgoLife": len(ego_matched),
            "HoloAssist": len(holo_matched),
            "CaptionCook4D": len(cc4d_matched),
        },
        "pairs": pairs,
    }

    os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(
        f"Human-eval matched pairs saved to: {dump_path} "
        f"(EgoLife={len(ego_matched)}, HoloAssist={len(holo_matched)}, "
        f"CaptionCook4D={len(cc4d_matched)})"
    )


def evaluate_and_save(args, ego_sub, ego_main, ego_overall, ego_matched,
                      holo_sub, holo_main, holo_overall, holo_matched,
                      cc4d_sub, cc4d_main, cc4d_overall, cc4d_matched,
                      output_path: str, dump_pairs_path: str = None):
    """Aggregate results, run optional LLM scoring, save JSON. Returns output_data.

    若提供 dump_pairs_path，会额外把 strict sub-type 匹配的 (pred, gt) 对
    导出到该路径，供人工打分使用（仅主实验需要）。
    """
    all_matched = ego_matched + holo_matched + cc4d_matched

    # 仅在调用方指定路径时（主实验 full）保存匹配对，供人工打分使用
    if dump_pairs_path:
        _dump_matched_pairs_for_human_eval(
            ego_matched, holo_matched, cc4d_matched, dump_pairs_path
        )
    dataset_results = {
        "EgoLife":       {"per_sub": ego_sub,  "per_main": ego_main,  "overall": ego_overall},
        "HoloAssist":    {"per_sub": holo_sub, "per_main": holo_main, "overall": holo_overall},
        "CaptionCook4D": {"per_sub": cc4d_sub, "per_main": cc4d_main, "overall": cc4d_overall},
    }

    # Per-dataset macro averages
    dataset_macro = {}
    for ds_name, ds in dataset_results.items():
        ov = ds["overall"]
        if ov["num_pred"] == 0 and ov["num_gt"] == 0:
            dataset_macro[ds_name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_active": 0}
            continue
        ds_p_list, ds_r_list, ds_f1_list = [], [], []
        for st in SUB_TYPES:
            if st not in ds["per_sub"]:
                continue
            a = ds["per_sub"][st]
            if a["num_gt"] > 0 or a["num_pred"] > 0:
                sp, sr, sf = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
                ds_p_list.append(sp); ds_r_list.append(sr); ds_f1_list.append(sf)
        dataset_macro[ds_name] = {
            "precision": round(np.mean(ds_p_list) if ds_p_list else 0.0, 4),
            "recall":    round(np.mean(ds_r_list)  if ds_r_list  else 0.0, 4),
            "f1":        round(np.mean(ds_f1_list) if ds_f1_list else 0.0, 4),
            "num_active": len(ds_p_list),
        }

    # Print per-dataset
    for ds_name, ds in dataset_results.items():
        ov = ds["overall"]
        if ov["num_pred"] == 0 and ov["num_gt"] == 0:
            continue
        p, r, f = compute_prf(ov["num_matched"], ov["num_pred"], ov["num_gt"])
        dm = dataset_macro[ds_name]
        logger.info(f"\n{'='*70}")
        logger.info(f"{ds_name} RESULTS")
        logger.info(f"{'='*70}")
        for st in SUB_TYPES:
            if st not in ds["per_sub"]:
                continue
            a = ds["per_sub"][st]
            if a["num_gt"] > 0 or a["num_pred"] > 0:
                sp, sr, sf = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
                mt_label = MAIN_TYPE_MAP[st]
                logger.info(
                    f"  [{mt_label:10s}] {st:25s}  P={sp:.3f}  R={sr:.3f}  F1={sf:.3f}  "
                    f"(pred={a['num_pred']}, gt={a['num_gt']}, matched={a['num_matched']})"
                )
        logger.info(
            f"  {'Sub-Type Macro Avg':25s}  P={dm['precision']:.3f}  R={dm['recall']:.3f}  "
            f"F1={dm['f1']:.3f}  (over {dm['num_active']} active sub-types)"
        )
        logger.info(
            f"  {'Overall':25s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  "
            f"(pred={ov['num_pred']}, gt={ov['num_gt']}, matched={ov['num_matched']})"
        )

    # Aggregate across datasets
    agg_sub = {st: {"num_pred": 0, "num_gt": 0, "num_matched": 0} for st in SUB_TYPES}
    for ds in dataset_results.values():
        for st in SUB_TYPES:
            if st in ds["per_sub"]:
                agg_sub[st]["num_pred"]    += ds["per_sub"][st]["num_pred"]
                agg_sub[st]["num_gt"]      += ds["per_sub"][st]["num_gt"]
                agg_sub[st]["num_matched"] += ds["per_sub"][st]["num_matched"]

    agg_main = {mt: {"num_pred": 0, "num_gt": 0, "num_matched": 0} for mt in MAIN_TYPES}
    for ds in dataset_results.values():
        for mt in MAIN_TYPES:
            if mt in ds["per_main"]:
                agg_main[mt]["num_pred"]    += ds["per_main"][mt]["num_pred"]
                agg_main[mt]["num_gt"]      += ds["per_main"][mt]["num_gt"]
                agg_main[mt]["num_matched"] += ds["per_main"][mt]["num_matched"]

    total_pred = sum(ds["overall"]["num_pred"] for ds in dataset_results.values())
    total_gt   = sum(ds["overall"]["num_gt"]   for ds in dataset_results.values())
    total_matched = sum(ds["overall"]["num_matched"] for ds in dataset_results.values())
    p_all, r_all, f_all = compute_prf(total_matched, total_pred, total_gt)

    sub_p_list, sub_r_list, sub_f1_list = [], [], []
    for st in SUB_TYPES:
        a = agg_sub[st]
        if a["num_gt"] > 0 or a["num_pred"] > 0:
            p, r, f = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
            sub_p_list.append(p); sub_r_list.append(r); sub_f1_list.append(f)
    macro_p  = np.mean(sub_p_list)  if sub_p_list  else 0.0
    macro_r  = np.mean(sub_r_list)  if sub_r_list  else 0.0
    macro_f1 = np.mean(sub_f1_list) if sub_f1_list else 0.0

    # Print combined
    logger.info(f"\n{'='*70}")
    logger.info("COMBINED RESULTS — ALL DATASETS")
    logger.info(f"{'='*70}")
    logger.info(f"\n  --- Per Sub-Type (strict match) ---")
    for st in SUB_TYPES:
        a = agg_sub[st]
        if a["num_gt"] > 0 or a["num_pred"] > 0:
            p, r, f = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
            mt_label = MAIN_TYPE_MAP[st]
            ds_tags = [ds_name[0] for ds_name, ds in dataset_results.items()
                       if st in ds["per_sub"] and (ds["per_sub"][st]["num_gt"] > 0 or ds["per_sub"][st]["num_pred"] > 0)]
            logger.info(
                f"  [{mt_label:10s}] {st:25s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  "
                f"(pred={a['num_pred']}, gt={a['num_gt']}, matched={a['num_matched']})  [{','.join(ds_tags)}]"
            )
    logger.info(f"\n  --- Sub-Type Macro Average ---")
    logger.info(
        f"  {'Macro-Avg':25s}  P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}  "
        f"(over {len(sub_p_list)} active sub-types)"
    )
    logger.info(f"\n  --- Per Main-Type ---")
    for mt in MAIN_TYPES:
        a = agg_main[mt]
        if a["num_gt"] > 0 or a["num_pred"] > 0:
            p, r, f = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
            logger.info(
                f"  {mt:12s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  "
                f"(pred={a['num_pred']}, gt={a['num_gt']}, matched={a['num_matched']})"
            )
    logger.info(f"\n  --- Overall ---")
    logger.info(
        f"  {'Overall':12s}  P={p_all:.3f}  R={r_all:.3f}  F1={f_all:.3f}  "
        f"(pred={total_pred}, gt={total_gt}, matched={total_matched})"
    )

    # LLM scoring
    llm_results = {}
    llm_per_sub = {}
    if not args.no_llm_scoring and all_matched:
        subtype_matched = [(pred, gt) for pred, gt in all_matched if pred["sub_type"] == gt["sub_type"]]
        logger.info(
            f"\nLLM scoring: {len(subtype_matched)} pairs with exact sub-type match "
            f"(out of {len(all_matched)} total matched)"
        )
        if subtype_matched:
            scores = llm_score_matched_pairs(subtype_matched, model_name=args.llm_model)
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
            sub_scores_map = defaultdict(list)
            for (pred, gt), score in zip(subtype_matched, scores):
                if score is not None:
                    sub_scores_map[gt["sub_type"]].append(score)
            logger.info(f"\n  --- LLM Scores Per Sub-Type ---")
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
                logger.info(
                    f"    [{MAIN_TYPE_MAP[st]:10s}] {st:25s}  "
                    f"Rationality={st_avg_r:.3f}  Effectiveness={st_avg_e:.3f}  (n={len(ss)})"
                )

    # Build output JSON
    output_data = {
        "per_dataset": {},
        "combined": {
            "per_sub_type": {},
            "sub_type_macro_average": {
                "precision": round(macro_p, 4), "recall": round(macro_r, 4),
                "f1": round(macro_f1, 4), "num_active_sub_types": len(sub_p_list),
            },
            "per_main_type": {},
            "overall": {
                "precision": round(p_all, 4), "recall": round(r_all, 4), "f1": round(f_all, 4),
                "num_pred": total_pred, "num_gt": total_gt, "num_matched": total_matched,
            },
            "llm_scores": llm_results,
            "llm_scores_per_sub_type": llm_per_sub,
        },
    }
    for ds_name, ds in dataset_results.items():
        ov = ds["overall"]
        p, r, f = compute_prf(ov["num_matched"], ov["num_pred"], ov["num_gt"])
        ds_sub, ds_main = {}, {}
        for st in SUB_TYPES:
            if st in ds["per_sub"]:
                a = ds["per_sub"][st]
                if a["num_gt"] > 0 or a["num_pred"] > 0:
                    sp, sr, sf = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
                    ds_sub[st] = {"precision": round(sp,4), "recall": round(sr,4), "f1": round(sf,4), **a}
        for mt in MAIN_TYPES:
            if mt in ds["per_main"]:
                a = ds["per_main"][mt]
                if a["num_gt"] > 0 or a["num_pred"] > 0:
                    mp, mr, mf = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
                    ds_main[mt] = {"precision": round(mp,4), "recall": round(mr,4), "f1": round(mf,4), **a}
        output_data["per_dataset"][ds_name] = {
            "per_sub_type": ds_sub,
            "sub_type_macro_average": dataset_macro[ds_name],
            "per_main_type": ds_main,
            "overall": {"precision": round(p,4), "recall": round(r,4), "f1": round(f,4), **ov},
        }
    for st in SUB_TYPES:
        a = agg_sub[st]
        if a["num_gt"] > 0 or a["num_pred"] > 0:
            p, r, f = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
            output_data["combined"]["per_sub_type"][st] = {
                "main_type": MAIN_TYPE_MAP[st],
                "precision": round(p,4), "recall": round(r,4), "f1": round(f,4), **a,
            }
    for mt in MAIN_TYPES:
        a = agg_main[mt]
        if a["num_gt"] > 0 or a["num_pred"] > 0:
            p, r, f = compute_prf(a["num_matched"], a["num_pred"], a["num_gt"])
            output_data["combined"]["per_main_type"][mt] = {
                "precision": round(p,4), "recall": round(r,4), "f1": round(f,4), **a,
            }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to: {output_path}")
    return output_data


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined Proactive Service Evaluation — All Datasets"
    )

    # --- EgoLife ---
    parser.add_argument("--egolife_prediction_base", type=str,
        default=paths.EGOLIFE_PRED)
    parser.add_argument("--egolife_gt_base", type=str,
        default=paths.EGOLIFE_GT)
    parser.add_argument("--egolife_persons", type=str, nargs='+',
        default=["A1_JAKE", "A4_LUCIA", "A5_KATRINA"])
    parser.add_argument("--egolife_days", type=str, nargs='+',
        default=["DAY1", "DAY2", "DAY3", "DAY4", "DAY5"])
    parser.add_argument("--egolife_caption_model", type=str,
        default="qwenvl_3_8b_instruct")
    parser.add_argument("--egolife_response_name", type=str,
        default="proactive_response_gpt_5")
    parser.add_argument("--egolife_tolerance", type=float, default=60.0)

    # --- HoloAssist ---
    parser.add_argument("--holoassist_prediction_base", type=str,
        default=paths.HOLOASSIST_PRED)
    parser.add_argument("--holoassist_gt_base", type=str,
        default=paths.HOLOASSIST_GT)
    parser.add_argument("--holoassist_response_name", type=str,
        default="proactive_response_gpt_5")
    parser.add_argument("--holoassist_tolerance", type=float, default=10.0)

    # --- CaptionCook4D ---
    parser.add_argument("--cc4d_prediction_base", type=str,
        default=paths.CC4D_PRED)
    parser.add_argument("--cc4d_gt_path", type=str,
        default=paths.CC4D_GT)
    parser.add_argument("--cc4d_response_name", type=str,
        default="proactive_response_gpt_5")
    parser.add_argument("--cc4d_tolerance", type=float, default=25.0)

    # --- Global ---
    parser.add_argument("--no_llm_scoring", action="store_true", default=True,
        help="Skip LLM-based scoring (default: scoring enabled)")
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    parser.add_argument("--output", type=str, default="new_result_full",
        help="Output JSON path (single-run mode). Defaults to <egolife_response_name>.json")
    parser.add_argument("--run_all", type=bool, default=True,
        help="Run all 5 ablations + full model + GPT + Qwen baselines and save each result")
    args = parser.parse_args()

    if args.run_all:
        # ================================================================
        # Batch mode: evaluate all configs and save individual JSON files
        # ================================================================
        _BASE_EGO  = paths.EGOLIFE_PRED
        _BASE_HOLO = paths.HOLOASSIST_PRED
        _BASE_CC4D = paths.CC4D_PRED
        # GPT-5-mini baseline
        _GPT_EGO   = paths.GPT_EGO
        _GPT_HOLO  = paths.GPT_HOLO
        _GPT_CC4D  = paths.GPT_CC4D
        # Qwen3-VL-Plus baseline
        _QWEN_EGO  = paths.QWEN_EGO
        _QWEN_HOLO = paths.QWEN_HOLO
        _QWEN_CC4D = paths.QWEN_CC4D

        # (label, ego_response, holo_response, cc4d_response, is_baseline, ego_dir, holo_dir, cc4d_dir)
        configs = [
            # 5 ablations (VideoRAG w/o X)
            ("wo_caption",    "proactive_response_gpt_5_wo_caption",   "proactive_response_gpt_5_wo_caption",      "proactive_response_gpt_5_wo_caption",  False, _BASE_EGO, _BASE_HOLO, _BASE_CC4D),
            ("wo_visual",     "proactive_response_gpt_5_wo_visual",    "proactive_response_gpt_5_wo_visual",       "proactive_response_gpt_5_wo_visual",   False, _BASE_EGO, _BASE_HOLO, _BASE_CC4D),
            ("wo_entity",     "proactive_response_gpt_5_wo_entity",    "proactive_response_gpt_5_wo_entity",       "proactive_response_gpt_5_wo_entity",   False, _BASE_EGO, _BASE_HOLO, _BASE_CC4D),
            ("wo_multiscale", "proactive_response_gpt_5_wo_multiscale","proactive_response_gpt_5_wo_multiscale",   "proactive_response_gpt_5_wo_multiscale",False, _BASE_EGO, _BASE_HOLO, _BASE_CC4D),
            ("wo_recons",     "proactive_response_gpt_5_wo_recons_",   "proactive_response_gpt_5_wo_recons_","proactive_response_gpt_5_wo_recons_",   False, _BASE_EGO, _BASE_HOLO, _BASE_CC4D),
            # Full VideoRAG model
            ("full",          "proactive_response_gpt_5",              "proactive_response_gpt_5",                 "proactive_response_gpt_5",             False, _BASE_EGO, _BASE_HOLO, _BASE_CC4D),
            # Baselines
            ("GPT",  None, None, None, True,  _GPT_EGO,  _GPT_HOLO,  _GPT_CC4D),
            ("Qwen", None, None, None, True,  _QWEN_EGO, _QWEN_HOLO, _QWEN_CC4D),
        ]

        for (label, ego_resp, holo_resp, cc4d_resp, is_baseline, ego_dir, holo_dir, cc4d_dir) in configs:
            logger.info(f"\n{'#'*70}")
            logger.info(f"# Running config: {label}")
            logger.info(f"{'#'*70}")

            if is_baseline:
                ego_sub, ego_main, ego_overall, ego_matched = run_egolife_baseline(args, ego_dir)
                holo_sub, holo_main, holo_overall, holo_matched = run_holoassist_baseline(args, holo_dir)
                cc4d_sub, cc4d_main, cc4d_overall, cc4d_matched = run_captioncook4d_baseline(args, cc4d_dir)
            else:
                args.egolife_response_name  = ego_resp
                args.holoassist_response_name = holo_resp
                args.cc4d_response_name     = cc4d_resp
                args.egolife_prediction_base  = ego_dir
                args.holoassist_prediction_base = holo_dir
                args.cc4d_prediction_base   = cc4d_dir
                ego_sub, ego_main, ego_overall, ego_matched   = run_egolife(args)
                holo_sub, holo_main, holo_overall, holo_matched = run_holoassist(args)
                cc4d_sub, cc4d_main, cc4d_overall, cc4d_matched = run_captioncook4d(args)

            # 输出统一为 result_clean_*(已移除 personal_progressive / CaptainCook4D-Episodic),
            # 避免覆盖历史的 result_*.json
            out_path = os.path.join(_EVAL_DIR, f"result_clean_{label}.json")
            # 所有配置（含 GPT / Qwen baseline）都保存匹配对，用于人工打分
            dump_path = os.path.join(
                _EVAL_DIR, f"result_clean_{label}_matched_pairs_for_human_eval.json"
            )
            evaluate_and_save(
                args,
                ego_sub, ego_main, ego_overall, ego_matched,
                holo_sub, holo_main, holo_overall, holo_matched,
                cc4d_sub, cc4d_main, cc4d_overall, cc4d_matched,
                output_path=out_path,
                dump_pairs_path=dump_path,
            )

        logger.info("\nAll configs done.")
        return

    # ================================================================
    # Single-run mode (original behaviour)
    # ================================================================
    logger.info("Running EgoLife evaluation...")
    ego_sub, ego_main, ego_overall, ego_matched = run_egolife(args)

    logger.info("Running HoloAssist evaluation...")
    holo_sub, holo_main, holo_overall, holo_matched = run_holoassist(args)

    logger.info("Running CaptionCook4D evaluation...")
    cc4d_sub, cc4d_main, cc4d_overall, cc4d_matched = run_captioncook4d(args)

    out_name = args.output or f"{args.egolife_response_name}.json"
    output_path = os.path.join(_EVAL_DIR, out_name)
    evaluate_and_save(
        args,
        ego_sub, ego_main, ego_overall, ego_matched,
        holo_sub, holo_main, holo_overall, holo_matched,
        cc4d_sub, cc4d_main, cc4d_overall, cc4d_matched,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
