#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从已清洗合并的 HoloAssist 主动服务大 JSON 中，提取"rebuttal 预测目录里出现的视频"子集。

背景：
    合并清洗后的全量 GT 在
        HoloAssist/holoassist_service_annotations_merged.json
    （由 consolidate_service_annotations.py 生成，含全部 1758 个视频）。
    而实际做评测/rebuttal 用到的视频是
        GST_EGOSERVE/all_results/holoassist_rebuttal/<video>-qwenvl_3_8b_instruct/
    下出现的那一批（207 个文件夹）。

本脚本只做"按视频名筛选"的提取，不改动任何标注内容、不做二次清洗：
    1) 扫描 rebuttal 目录，收集所有 "<video>-qwenvl_3_8b_instruct" 文件夹对应的视频名。
    2) 按用户选定的范围过滤：默认去掉"无 GT 对话"的视频（不动 skip 列表）。
    3) 从合并大 JSON 中原样取出这些视频的标注（保留 instant/short_term 分组结构），
       写出一个新的子集 JSON。

不修改：
    * consolidate_service_annotations.py（之前整理 HoloAssist 用的脚本）
    * holoassist_service_annotations_merged.json（全量合并 GT）
    * rebuttal 预测目录

用法：
    python extract_rebuttal_subset.py
    # 自定义范围：
    python extract_rebuttal_subset.py --scope has_gt   # 207 去掉无GT (默认)
    python extract_rebuttal_subset.py --scope all       # rebuttal 全部 207
    python extract_rebuttal_subset.py --scope eval182    # 去 skip + 去无GT = 182
"""

import os
import json
import argparse

# 评测脚本中固定排除的视频（与 holoassist 评测脚本保持一致）
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

PRED_SUFFIX = "-qwenvl_3_8b_instruct"


def collect_rebuttal_videos(pred_base):
    """扫描 rebuttal 目录，返回其中出现的全部视频名集合。"""
    videos = set()
    if not os.path.isdir(pred_base):
        raise FileNotFoundError(f"rebuttal 预测目录不存在: {pred_base}")
    for d in os.listdir(pred_base):
        if d.endswith(PRED_SUFFIX):
            videos.add(d[:-len(PRED_SUFFIX)])
    return videos


def video_has_gt(video_entry):
    """判断某视频在合并 GT 中是否含有 GT 对话条目（任一类型事件带非空 dialogue）。"""
    if not video_entry:
        return False
    for high in video_entry:
        for sub_type, events in video_entry[high].items():
            for e in events:
                if e.get("dialogue"):
                    return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="从合并 GT 中提取 rebuttal 视频子集"
    )
    parser.add_argument(
        "--merged_gt", type=str,
        default="./data/HoloAssist/holoassist_service_annotations_merged.json",
        help="全量合并清洗后的大 GT JSON",
    )
    parser.add_argument(
        "--pred_base", type=str,
        default="./data/predictions/holoassist_rebuttal",
        help="rebuttal 预测目录（用于确定视频集合）",
    )
    parser.add_argument(
        "--scope", type=str, default="has_gt",
        choices=["all", "has_gt", "eval182"],
        help="提取范围：all=全部rebuttal视频; has_gt=去掉无GT(默认); "
             "eval182=去skip+去无GT(评测口径)",
    )
    parser.add_argument(
        "--output", type=str,
        default="./data/HoloAssist/holoassist_service_annotations_merged_rebuttal.json",
        help="提取出的子集 JSON 输出路径",
    )
    args = parser.parse_args()

    # 读取全量合并 GT
    with open(args.merged_gt, "r", encoding="utf-8") as f:
        merged = json.load(f)

    # rebuttal 视频集合
    rebuttal_videos = collect_rebuttal_videos(args.pred_base)
    print(f"rebuttal 目录视频数: {len(rebuttal_videos)}")

    # rebuttal 中不在合并 GT 里的（理论上应为 0）
    missing = sorted(rebuttal_videos - set(merged.keys()))
    if missing:
        print(f"警告: {len(missing)} 个 rebuttal 视频不在合并 GT 中，将被忽略: {missing[:5]}")

    # 候选 = rebuttal ∩ 合并GT
    candidates = sorted(rebuttal_videos & set(merged.keys()))

    # 按 scope 过滤
    selected = []
    dropped_skip = []
    dropped_no_gt = []
    for vn in candidates:
        if args.scope == "eval182" and vn in HOLOASSIST_SKIP_VIDEOS:
            dropped_skip.append(vn)
            continue
        if args.scope in ("has_gt", "eval182") and not video_has_gt(merged[vn]):
            dropped_no_gt.append(vn)
            continue
        selected.append(vn)

    # 组装子集（原样取出，不改内容）
    subset = {vn: merged[vn] for vn in selected}

    # 统计各子类型事件数（便于核对）
    sub_counts = {}
    layout = {"instant": ["safety", "tool_use"],
              "short_term": ["error_recovery", "next_step_guidance", "resource_reminder"]}
    for vn in selected:
        v = merged[vn]
        for high, subs in layout.items():
            for st in subs:
                sub_counts[st] = sub_counts.get(st, 0) + len(v.get(high, {}).get(st, []))

    # 写出
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    # 同时写一份视频名单，便于换机核对
    list_path = os.path.splitext(args.output)[0] + "_video_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for vn in selected:
            f.write(vn + "\n")

    print(f"\n提取范围 scope = {args.scope}")
    print(f"候选(rebuttal∩合并GT): {len(candidates)}")
    if args.scope == "eval182":
        print(f"  去掉 skip: {len(dropped_skip)}")
    if args.scope in ("has_gt", "eval182"):
        print(f"  去掉无GT对话: {len(dropped_no_gt)}")
    print(f"最终提取视频数: {len(selected)}")
    print(f"各子类型事件数: {sub_counts}")
    print(f"\n子集已保存: {args.output}")
    print(f"视频名单已保存: {list_path}")


if __name__ == "__main__":
    main()
