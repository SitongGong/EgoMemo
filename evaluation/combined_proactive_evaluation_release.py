#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测的 RELEASE 版:GT 全部读取自 /mnt/workspace/gst/EgoServe_release(开源发布的标注),
其余评测逻辑(匹配/PRF1/strict)完全复用 combined_proactive_evaluation_subtype_correct.py。

与 _correct.py 的唯一区别 = GT 来源:
  - EgoLife:      EgoServe_release/EgoLife/{person}/{instant,short_term,long_term,episodic}.json
                  (已是过滤后的发布版, 含 sub_type 字段, 不再二次过滤)
  - HoloAssist:   EgoServe_release/HoloAssist/holoassist_service_annotations.json (191 video)
  - CaptainCook4D:EgoServe_release/CaptainCook4D/captaincook4d_service_annotations.json (87 rec)

原 _correct.py 完全不动(原路径/处理方式保留)。本文件用 monkey-patch 重定向 GT 读取后,
直接调用 _correct.main() 跑全部 8 个配置, 结果写到 result_release_*.json。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import combined_proactive_evaluation as C
import egolife_proactive_evaluation_subtype as ego
import holoassist_proactive_evaluation_subtype as holo
import captioncook4d_proactive_evaluation_subtype as cc4d

REL = paths.RELEASE_ROOT

# ---- EgoLife: 读 release 的 4 个主类型文件(已过滤, 直接用) ----
EGO_MAIN_FILES = ["instant", "short_term", "long_term", "episodic"]

def ego_extract_gt_release(gt_base, person, allowed_days):
    """从 release 读 EgoLife GT。release 已过滤(无 speakers_say/无效/PF),
    每事件带 sub_type; 仍按 allowed_days 限制天数(与 _correct 行为对齐)。"""
    results = []
    pdir = os.path.join(REL, "EgoLife", person)
    if not os.path.isdir(pdir):
        return results
    for mf in EGO_MAIN_FILES:
        fp = os.path.join(pdir, f"{mf}.json")
        if not os.path.exists(fp):
            continue
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for e in data:
            if not isinstance(e, dict):
                continue
            sub_type = e.get("sub_type")
            if sub_type not in ego.MAIN_TYPE_MAP:
                continue
            parsed = ego.parse_day_time_window(e.get("current_time_window", ""))
            if parsed is None:
                continue
            day_num, start_s, end_s = parsed
            if day_num not in allowed_days:
                continue
            utterance = ""
            for turn in (e.get("proactive_dialogue") or e.get("dialogue") or []):
                if turn.get("role") == "assistant":
                    utterance = turn.get("utterance", "").strip(); break
            if not utterance:
                continue
            results.append({
                "person": person, "day_num": day_num,
                "main_type": ego.MAIN_TYPE_MAP[sub_type], "sub_type": sub_type,
                "time_center": ego.time_window_center(day_num, start_s, end_s),
                "time_start": day_num * 86400 + start_s,
                "time_end": day_num * 86400 + end_s,
                "user_prompt": utterance, "service_type_raw": sub_type,
            })
    results.sort(key=lambda x: x["time_center"])
    return results

# ---- HoloAssist: 读 release 单文件(与 _correct 的 final.json 适配函数同逻辑) ----
HOLO_REL = os.path.join(REL, "HoloAssist", "holoassist_service_annotations.json")
_HOLO_REL_CACHE = {}
def _load_holo_rel():
    if "d" not in _HOLO_REL_CACHE:
        _HOLO_REL_CACHE["d"] = json.load(open(HOLO_REL, encoding="utf-8"))
    return _HOLO_REL_CACHE["d"]

_HOLO_DIRS = [("instant", "safety"), ("instant", "tool_use"),
              ("short_term", "error_recovery"), ("short_term", "next_step_guidance"),
              ("short_term", "resource_reminder")]

def holo_extract_gt_release(gt_base, video_names):
    data = _load_holo_rel()
    results = []
    for vn in video_names:
        vv = data.get(vn)
        if not vv:
            continue
        for bucket, sub in _HOLO_DIRS:
            mt = holo.MAIN_TYPE_MAP[sub]
            for e in (vv.get(bucket) or {}).get(sub, []) or []:
                parsed = holo.parse_gt_time_window(e.get("time_window", ""))
                if parsed is None:
                    continue
                s, en = parsed
                ut = ""
                for t in e.get("dialogue", []):
                    if t.get("role") == "assistant":
                        ut = t.get("utterance", "").strip(); break
                if not ut:
                    continue
                results.append({"video_name": vn, "main_type": mt, "sub_type": sub,
                    "time_start": s, "time_end": en, "time_center": (s + en) / 2.0,
                    "user_prompt": ut, "observation": e.get("observation", ""),
                    "raw_time_window": e.get("time_window", "")})
    results.sort(key=lambda x: (x["video_name"], x["time_center"]))
    return results

# ---- 应用 patch ----
ego.extract_ground_truth = ego_extract_gt_release
holo.extract_ground_truth = holo_extract_gt_release
# _correct 模块里也直接引用了 holo_eval.extract_ground_truth(已被上面替换)
C.holo_eval.extract_ground_truth = holo_extract_gt_release
# HoloAssist 视频集合: release 的 keys
C._load_holo_final_gt = _load_holo_rel
# CaptainCook4D: GT 路径换成 release 单文件(结构同 consolidated, extract_ground_truth 直接用)
# 通过改 main() 的默认参数实现 —— 在调用前改 sys.argv 不便, 直接 patch 路径常量
# _correct.run_captioncook4d 用 args.cc4d_gt_path, 我们改 main 里的默认值

# EgoLife GT base 改成 release(其实 ego_extract_gt_release 已硬编码 REL, 但 _correct 调用时传 person/days)
# _correct.extract_gt_correct_first? 不存在; _correct.run_egolife 调 ego_eval.extract_ground_truth(gt_base, person, days)
# 已被 patch, 忽略 gt_base 参数。

def main():
    # 改 CC4D GT 路径 -> release
    argv_bak = sys.argv[:]
    sys.argv = [sys.argv[0],
                "--cc4d_gt_path", os.path.join(REL, "CaptainCook4D", "captaincook4d_service_annotations.json")]
    # 改输出前缀: 让 _correct 写 result_release_*.json
    _orig_join = os.path.join
    # _correct.main 用 os.path.join(_EVAL_DIR, f"result_{label}.json")
    # 简单起见: 跑完后重命名。先记录 label 集合
    C.main()
    sys.argv = argv_bak
    # 重命名 result_clean_*.json -> result_release_*.json(仅本次跑出的 8 个)
    EVAL = os.path.dirname(os.path.abspath(__file__))
    for label in ["wo_caption","wo_visual","wo_entity","wo_multiscale","wo_recons","full","GPT","Qwen"]:
        src = _orig_join(EVAL, f"result_clean_{label}.json")
        dst = _orig_join(EVAL, f"result_release_{label}.json")
        if os.path.exists(src):
            os.rename(src, dst)
    print("\n[release] 结果已写为 result_release_*.json")

if __name__ == "__main__":
    main()
