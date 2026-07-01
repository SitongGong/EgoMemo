#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Table S4(EgoLife subset)——口径与论文一致版。

与 combined_proactive_evaluation_subtype_reserve.py 的 EgoLife 计算路径
**完全一致**：
  - per-person 匹配(避免跨人误配),再累加 num_pred/num_gt/num_matched
  - 用普通版 ego_eval.match_predictions_to_gt(非 strict 脚本)
  - F1 由累加 counts 经 compute_prf 得到
  - 容差 60s,指标 F1
  - 保留 personal_progressive(PF 列)

唯一区别:GT 源换成修正后的 final_gt(有 correct 用 correct)。
做法:monkey-patch ego_eval.extract_ground_truth 为 correct 优先,GT_BASE 指向 final_gt。
原脚本不改动。
"""
import os
import paths, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egolife_proactive_evaluation_subtype as ego_eval

GT_BASE   = paths.EGOLIFE_GT
PRED_BASE = paths.EGOLIFE_PRED
CAPTION   = paths.CAPTION_MODEL
PERSONS   = ["A1_JAKE", "A4_LUCIA", "A5_KATRINA"]
DAYS      = {1, 2, 3, 4, 5}
TOL       = 60.0


def compute_prf(num_matched, num_pred, num_gt):
    p = num_matched / num_pred if num_pred > 0 else 0.0
    r = num_matched / num_gt if num_gt > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f


# ---- correct 优先的 GT 读取(其余逻辑与原 extract_ground_truth 完全相同) ----
_orig_parse = ego_eval.parse_day_time_window
_orig_center = ego_eval.time_window_center

def extract_gt_correct_first(gt_base, person, allowed_days):
    results = []
    for subdir, sub_type in ego_eval.GT_SERVICE_DIRS:
        base = os.path.join(gt_base, person, subdir)
        fpath = os.path.join(base, "conversation_results_correct.json")
        if not os.path.exists(fpath):
            fpath = os.path.join(base, "conversation_results.json")
        if not os.path.exists(fpath):
            continue
        try:
            data = json.load(open(fpath, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        main_type = ego_eval.MAIN_TYPE_MAP[sub_type]
        for entry in data:
            if not isinstance(entry, dict):
                continue
            source = entry.get("supporting_source", entry.get("current_supporting_source", ""))
            if source == "speakers_say":
                continue
            parsed = _orig_parse(entry.get("current_time_window", ""))
            if parsed is None:
                continue
            day_num, start_s, end_s = parsed
            if day_num not in allowed_days:
                continue
            utterance = ""
            for turn in (entry.get("proactive_dialogue") or entry.get("dialogue") or []):
                if turn.get("role") == "assistant":
                    utterance = turn.get("utterance", "").strip(); break
            if not utterance:
                continue
            results.append({
                "person": person, "day_num": day_num,
                "main_type": main_type, "sub_type": sub_type,
                "time_center": _orig_center(day_num, start_s, end_s),
                "time_start": day_num * 86400 + start_s,
                "time_end": day_num * 86400 + end_s,
                "user_prompt": utterance, "service_type_raw": sub_type,
            })
    results.sort(key=lambda x: x["time_center"])
    return results


def eval_egolife(response_name):
    """完全复用 reserve.run_egolife 的 per-person 累加逻辑,返回各子类型 counts。"""
    per_sub = {st: {"num_pred": 0, "num_gt": 0, "num_matched": 0} for st in ego_eval.SUB_TYPES}
    tot = {"num_pred": 0, "num_gt": 0, "num_matched": 0}
    for person in PERSONS:
        pdir = os.path.join(PRED_BASE, f"{person}-{CAPTION}_restart")
        person_days = {d for d in DAYS
                       if os.path.exists(os.path.join(pdir, f"DAY{d}", f"{response_name}.json"))}
        if not person_days:
            continue
        gt = extract_gt_correct_first(GT_BASE, person, person_days)
        preds = ego_eval.extract_predictions(PRED_BASE, person, CAPTION, person_days,
                                             response_name=response_name)
        # strict 子类型隔离 + per-person 匹配(与 reserve.evaluate_subtype_strict 同口径)
        for st in ego_eval.SUB_TYPES:
            preds_st = [p for p in preds if p["sub_type"] == st]
            gts_st = [g for g in gt if g["sub_type"] == st]
            matched, _, _ = ego_eval.match_predictions_to_gt(preds_st, gts_st, tolerance=TOL)
            per_sub[st]["num_pred"]    += len(preds_st)
            per_sub[st]["num_gt"]      += len(gts_st)
            per_sub[st]["num_matched"] += len(matched)
            tot["num_pred"]    += len(preds_st)
            tot["num_gt"]      += len(gts_st)
            tot["num_matched"] += len(matched)
    f1 = {st: compute_prf(per_sub[st]["num_matched"], per_sub[st]["num_pred"],
                          per_sub[st]["num_gt"])[2] * 100 for st in ego_eval.SUB_TYPES}
    # Overall = 活跃子类型(gt或pred>0)的 F1 macro-average(与论文 Table S4 口径一致)
    import numpy as np
    active = [f1[st] for st in ego_eval.SUB_TYPES
              if per_sub[st]["num_gt"] > 0 or per_sub[st]["num_pred"] > 0]
    ov = float(np.mean(active)) if active else 0.0
    return f1, ov, per_sub


ROWS = [
    ("w/o MS",          "proactive_response_gpt_5_wo_multiscale"),
    ("w/o Recons.",     "proactive_response_gpt_5_wo_recons_"),
    ("w/o VSR",         "proactive_response_gpt_5_wo_visual"),
    ("w/o GSR",         "proactive_response_gpt_5_wo_entity"),
    ("w/o MTR",         "proactive_response_gpt_5_wo_caption"),
    ("EgoMemo (Ours)",  "proactive_response_gpt_5"),
]
COLS = [("SA","safety"),("TU","tool_use"),("NSG","next_step_guidance"),("ER","error_recovery"),
        ("RR","resource_reminder"),("MR","memory_recall"),("TR","task_reminder"),
        ("HC","habit_coaching"),("ML","memory_link_contextual"),
        ("PF","personal_progressive"),("RO","routine_optimization")]


def main():
    data = {}
    for name, rn in ROWS:
        f1, ov, _ = eval_egolife(rn)
        data[name] = {**f1, "__overall__": ov}

    head = f"{'Model':16s}" + "".join(f"{ab:>6s}" for ab, _ in COLS) + f"{'Overall':>9s}"
    print(head); print("-" * len(head))
    for name, _ in ROWS:
        d = data[name]
        print(f"{name:16s}" + "".join(f"{d[st]:>6.1f}" for _, st in COLS) + f"{d['__overall__']:>9.1f}")

    out_path = os.path.join(paths.OUTPUT_ROOT, "table_s4.json")
    os.makedirs(paths.OUTPUT_ROOT, exist_ok=True)
    json.dump(data, open(out_path, "w"),
              ensure_ascii=False, indent=2)
    print(f"\n已存: {out_path}")


if __name__ == "__main__":
    main()
