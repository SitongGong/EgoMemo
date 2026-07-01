#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloAssist 主动服务标注整理与清洗脚本
=====================================

背景：
    我们提出的 benchmark 中，主动服务标注分散在
        holoassist_service_annotations_rec_/instant/{safety, tool_use}
        holoassist_service_annotations_rec_/short_term/{error_recovery, next_step_guidance, resource_reminder}
    下的多个 json 文件中（每个视频一个文件，共 5 种服务类型）。
    论文提交后发现标注存在三类问题，需要重新整理并剔除/合并有问题的标注，
    用于代码开源时在 GitHub 上修正。

本脚本完成两件事：
    1) 合并：把 5 种服务类型按视频名合并成一个大 JSON
       —— key 为视频名，value 按 instant / short_term 两大类分组。
    2) 清洗：修正三类已知标注问题（详见下方"清洗规则"）。

重要原则（遵循项目规则）：
    - 绝不修改原始 json 文件，只读取它们并生成新的合并文件。
    - 原始标注的产生逻辑见
        gemini_generation/holoassist_data_generator.py
        gemini_generation/holoassist_prompt.py

清洗规则：
    规则 A —— tool_use / error_recovery 重复：
        同一视频中若某个 time_window 同时出现在 tool_use 和 error_recovery 里
        （这是因为我们把同一条人工标注分别送进两种 prompt 导致的重复），
        统一只保留 tool_use 类型，从 error_recovery 中剔除该事件。

    规则 B —— 一处错误对应多个错误类型：
        同一 time_window 在同一服务类型内部出现多个事件时去重；
        若跨类型仍有冲突（除 A 之外的情况），按"更合理"的优先级保留一个。
        优先级见 SERVICE_PRIORITY。

    规则 C —— 时间上连续、描述同一错误的多段标注合并为一段：
        因为我们当时是直接把 HoloAssist 人工细粒度标注按错误类型逐段转换，
        所以经常出现多段时间几乎首尾相连、且类型相同、描述同一个错误的情况
        （例如"镜头没装正"被切成十几段）。
        对这种 gap <= MERGE_GAP_SECONDS 且类型相同的连续事件合并为一个大事件。

输出结构（按用户确认）：
    {
      "<video_name>": {
        "instant": {
          "safety": [ {event..., "dialogue":[...]}, ... ],
          "tool_use": [ ... ]
        },
        "short_term": {
          "error_recovery": [ ... ],
          "next_step_guidance": [ ... ],
          "resource_reminder": [ ... ]
        }
      },
      ...
    }
    每个 event 内嵌入与之配对的 dialogue（按 time_window 对齐）。

用法：
    conda activate egoserve
    python consolidate_service_annotations.py \
        --input_root /mnt/workspace/gst/HoloAssist/holoassist_service_annotations_rec_ \
        --output /mnt/workspace/gst/HoloAssist/holoassist_service_annotations_merged.json \
        --merge_gap 1.5
"""

import os
import json
import glob
import argparse
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# 配置：服务类型 -> (高层类别, 事件数组的 key 名, 类型字段名)
# ---------------------------------------------------------------------------
SERVICE_CONFIG = {
    "safety":            ("instant",    "safety_instant_events",     "risk_type"),
    "tool_use":          ("instant",    "tool_use_instant_events",   "risk_type"),
    "error_recovery":    ("short_term", "error_recovery_events",     "error_type"),
    "next_step_guidance":("short_term", "next_step_events",          "guidance_type"),
    "resource_reminder": ("short_term", "resource_reminder_events",  "reminder_type"),
}

# 子目录布局：高层类别 -> 子目录名（即低层服务类型）
DIR_LAYOUT = {
    "instant":    ["safety", "tool_use"],
    "short_term": ["error_recovery", "next_step_guidance", "resource_reminder"],
}

# 规则 B 跨类型冲突时的保留优先级（数字越小越优先保留）。
# 规则 A 已单独把 tool_use > error_recovery 固定下来；这里给出更一般的兜底优先级：
# safety（涉及人身安全）最高，其次 tool_use，再 error_recovery，
# 然后 resource_reminder，最后 next_step_guidance（仅是建议性，冲突时最先让位）。
SERVICE_PRIORITY = {
    "safety": 0,
    "tool_use": 1,
    "error_recovery": 2,
    "resource_reminder": 3,
    "next_step_guidance": 4,
}

# 规则 C 合并阈值（秒）；可由命令行覆盖。
DEFAULT_MERGE_GAP = 1.5


# ---------------------------------------------------------------------------
# 时间窗解析 / 格式化
# ---------------------------------------------------------------------------
def parse_time_window(tw):
    """
    解析 time_window 字符串为 (start_sec, end_sec)。
    支持两种格式：
        "00:00:22.200-00:00:25.130"  (HH:MM:SS.mmm)
        "138.124-140.095"            (纯秒)
    解析失败返回 None。
    """
    if not isinstance(tw, str):
        return None
    parts = tw.split("-")
    if len(parts) != 2:
        return None

    def to_sec(s):
        s = s.strip()
        if ":" in s:
            # 兼容 HH:MM:SS.mmm 与 MM:SS.mmm 两种写法（原始数据里两者混用）
            comps = [float(c) for c in s.split(":")]
            while len(comps) < 3:
                comps.insert(0, 0.0)
            hh, mm, ss = comps[0], comps[1], comps[2]
            return hh * 3600 + mm * 60 + ss
        return float(s)

    try:
        return to_sec(parts[0]), to_sec(parts[1])
    except (ValueError, AttributeError):
        return None


def is_colon_format(tw):
    """判断原始 time_window 是否为 HH:MM:SS 格式。"""
    return isinstance(tw, str) and ":" in tw


def sec_to_colon(sec):
    """秒 -> HH:MM:SS.mmm 字符串。"""
    if sec < 0:
        sec = 0.0
    hh = int(sec // 3600)
    mm = int((sec % 3600) // 60)
    ss = sec - hh * 3600 - mm * 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def format_time_window(start_sec, end_sec, colon):
    """按指定格式重新生成 time_window 字符串。"""
    if colon:
        return f"{sec_to_colon(start_sec)}-{sec_to_colon(end_sec)}"
    return f"{start_sec:.3f}-{end_sec:.3f}"


# ---------------------------------------------------------------------------
# 读取单个服务类型的某个视频文件
# ---------------------------------------------------------------------------
def load_service_file(path, event_key):
    """
    读取一个服务标注文件，返回 (events, dialogs, status)。
    status:
        "ok"      正常（可能为空标注）
        "failed"  失败请求（文件里 dump 的是 schema / error，没有事件 key）
    失败请求会被视为"无有效标注"，事件按空处理。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], [], "failed"

    if not isinstance(data, dict):
        return [], [], "failed"

    # 失败请求特征：含 generation_config / error，而没有真正的事件数组
    if event_key not in data:
        return [], [], "failed"

    events = data.get(event_key) or []
    dialogs = data.get("dialogs") or []
    if not isinstance(events, list):
        events = []
    if not isinstance(dialogs, list):
        dialogs = []
    return events, dialogs, "ok"


def attach_dialogues(events, dialogs):
    """
    按 time_window 把 dialogs 嵌入到对应 event 上（写入 event['dialogue']）。
    已确认原始数据中二者 time_window 完全一一对应。
    若某 event 无匹配 dialog，则其 dialogue 为空列表。
    """
    tw_to_dialogue = {}
    for d in dialogs:
        tw = d.get("time_window")
        if tw is not None and tw not in tw_to_dialogue:
            tw_to_dialogue[tw] = d.get("dialogue", [])

    enriched = []
    for e in events:
        e = dict(e)  # 拷贝，避免影响原始读入对象
        e["dialogue"] = tw_to_dialogue.get(e.get("time_window"), [])
        enriched.append(e)
    return enriched


# ---------------------------------------------------------------------------
# 规则 A + B：跨类型去重（处理同一 time_window 在多个类型出现的冲突）
# ---------------------------------------------------------------------------
def normalize_tw_key(tw):
    """把 time_window 归一化为可比较的 (start, end) 元组（四舍五入到毫秒），用作冲突匹配键。"""
    parsed = parse_time_window(tw)
    if parsed is None:
        return ("raw", tw)  # 解析失败时退化为原字符串比较
    return (round(parsed[0], 3), round(parsed[1], 3))


def resolve_cross_type_conflicts(per_type_events, stats):
    """
    规则 A & B：同一 time_window 跨类型冲突时，按优先级只保留一个类型。

    per_type_events: {service_name: [event, ...]}（同一视频内、已附带 dialogue）
    就地修改并返回新的 per_type_events。
    """
    # 1) 收集每个 time_window 出现在哪些类型
    tw_to_types = defaultdict(set)
    for svc, events in per_type_events.items():
        for e in events:
            tw_to_types[normalize_tw_key(e.get("time_window"))].add(svc)

    # 2) 对有冲突（出现在 >1 个类型）的 time_window，决定保留哪个类型
    keep_type_for_tw = {}
    for twk, types in tw_to_types.items():
        if len(types) <= 1:
            continue
        # 规则 A：tool_use 与 error_recovery 同时出现 -> 保留 tool_use
        if "tool_use" in types and "error_recovery" in types:
            stats["rule_A_tooluse_over_errorrecovery"] += 1
        # 规则 B（含规则 A 的一致方向）：统一用 SERVICE_PRIORITY 选最高优先级
        winner = min(types, key=lambda s: SERVICE_PRIORITY.get(s, 99))
        keep_type_for_tw[twk] = winner
        if not ("tool_use" in types and "error_recovery" in types):
            stats["rule_B_other_conflicts"] += 1

    # 3) 过滤掉非胜出类型里的冲突事件
    cleaned = {}
    for svc, events in per_type_events.items():
        kept = []
        for e in events:
            twk = normalize_tw_key(e.get("time_window"))
            if twk in keep_type_for_tw and keep_type_for_tw[twk] != svc:
                stats["removed_conflict_events"] += 1
                continue
            kept.append(e)
        cleaned[svc] = kept
    return cleaned


def dedup_within_type(events, type_field, stats):
    """
    规则 B 的同类型内部去重：完全相同的 (time_window, 类型) 只保留一个，
    被丢弃的事件其来源（observation）并入保留事件的 merged_sources 以便追溯。
    """
    seen = {}
    result = []
    for e in events:
        key = (normalize_tw_key(e.get("time_window")), e.get(type_field))
        if key in seen:
            stats["dedup_within_type"] += 1
            primary = seen[key]
            primary.setdefault("merged_sources", [])
            obs = e.get("observation")
            if obs and obs not in primary["merged_sources"]:
                primary["merged_sources"].append(obs)
            continue
        seen[key] = e
        result.append(e)
    return result


# ---------------------------------------------------------------------------
# 规则 C：连续同类型事件合并
# ---------------------------------------------------------------------------
def merge_consecutive(events, type_field, gap_threshold, stats):
    """
    规则 C：把时间上连续（gap <= gap_threshold）且类型相同的事件合并为一个大事件。

    合并策略：
        - time_window 扩展为整组的 [最早 start, 最晚 end]（保持原始格式）。
        - 保留组内第一个事件的字段作为主体（类型、source、segment_id 等）。
        - observation 合并为有序去重后的列表，写入 merged_observations；
          同时把主 observation 设为组内第一个，便于阅读。
        - dialogue 保留第一个事件的（最早触发那一刻的对话最合理）。
        - 新增 merged_count 记录该大事件由几段合并而来。
    无法解析时间窗的事件不参与合并，原样保留。
    """
    if not events:
        return events

    # 拆出可解析时间的事件，按 start 排序；不可解析的原样放回
    parsable = []
    unparsable = []
    for e in events:
        parsed = parse_time_window(e.get("time_window"))
        if parsed is None:
            unparsable.append(e)
        else:
            parsable.append((parsed[0], parsed[1], e))
    parsable.sort(key=lambda x: (x[0], x[1]))

    merged = []
    i = 0
    n = len(parsable)
    while i < n:
        group = [parsable[i]]
        j = i
        while j + 1 < n:
            cur_end = group[-1][1]
            nxt_start, nxt_end, nxt_e = parsable[j + 1]
            same_type = nxt_e.get(type_field) == group[-1][2].get(type_field)
            # gap：下一段起点 - 当前段终点（允许轻微重叠，即 gap 为负）
            gap = nxt_start - cur_end
            if same_type and gap <= gap_threshold:
                group.append(parsable[j + 1])
                j += 1
            else:
                break

        if len(group) == 1:
            merged.append(group[0][2])
        else:
            # 执行合并
            stats["rule_C_merge_groups"] += 1
            stats["rule_C_events_merged"] += len(group)
            base = dict(group[0][2])  # 以第一个事件为主体
            start_sec = min(g[0] for g in group)
            end_sec = max(g[1] for g in group)
            colon = is_colon_format(group[0][2].get("time_window"))
            base["time_window"] = format_time_window(start_sec, end_sec, colon)

            observations = []
            for _, _, e in group:
                obs = e.get("observation")
                if obs and obs not in observations:
                    observations.append(obs)
            base["merged_observations"] = observations
            if observations:
                base["observation"] = observations[0]
            base["merged_count"] = len(group)
            # dialogue 保留第一个事件的
            base["dialogue"] = group[0][2].get("dialogue", [])
            merged.append(base)
        i = j + 1

    # 不可解析事件原样保留在末尾
    merged.extend(unparsable)
    return merged


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def consolidate(input_root, merge_gap):
    """读取所有服务类型文件，执行合并 + 清洗，返回 (merged_dict, stats)。"""
    stats = Counter()
    raw_event_counts = Counter()   # 清洗前各类型事件数
    failed_files = []

    # video_name -> {service_name -> [enriched events]}
    per_video = defaultdict(lambda: {svc: [] for svc in SERVICE_CONFIG})

    # 1) 遍历每个服务类型目录，读取每个视频文件
    for svc, (high, event_key, type_field) in SERVICE_CONFIG.items():
        svc_dir = os.path.join(input_root, high, svc)
        files = sorted(glob.glob(os.path.join(svc_dir, "*.json")))
        for fp in files:
            video_name = os.path.splitext(os.path.basename(fp))[0]
            events, dialogs, status = load_service_file(fp, event_key)
            if status == "failed":
                failed_files.append((svc, os.path.basename(fp)))
                stats["failed_request_files"] += 1
                continue
            enriched = attach_dialogues(events, dialogs)
            raw_event_counts[svc] += len(enriched)
            per_video[video_name][svc].extend(enriched)

    # 2) 对每个视频执行清洗
    merged = {}
    for video_name in sorted(per_video.keys()):
        per_type = per_video[video_name]

        # 先做同类型内部去重(规则B) + 连续合并(规则C)，再做跨类型冲突解决(规则A&B)。
        # 顺序很重要：合并会生成新的(更长的)time_window，可能与另一类型的窗口
        # 恰好相同，因此跨类型冲突必须在合并之后判定，否则会有漏网之鱼。
        for svc in per_type:
            type_field = SERVICE_CONFIG[svc][2]
            per_type[svc] = dedup_within_type(per_type[svc], type_field, stats)
            per_type[svc] = merge_consecutive(per_type[svc], type_field, merge_gap, stats)

        # 规则 A & B：跨类型冲突解决（在合并完成后，基于最终窗口判定）
        per_type = resolve_cross_type_conflicts(per_type, stats)

        # 各类型按起始时间排序，保证有序
        for svc in per_type:
            per_type[svc].sort(
                key=lambda e: (parse_time_window(e.get("time_window")) or (0.0, 0.0))[0]
            )

        # 3) 按 instant / short_term 分组组织输出
        out = {"instant": {}, "short_term": {}}
        for svc, (high, _, _) in SERVICE_CONFIG.items():
            out[high][svc] = per_type[svc]
        merged[video_name] = out

    # 统计清洗后事件数
    clean_event_counts = Counter()
    for video_name, out in merged.items():
        for high in out:
            for svc, evs in out[high].items():
                clean_event_counts[svc] += len(evs)

    stats["videos"] = len(merged)
    return merged, stats, raw_event_counts, clean_event_counts, failed_files


def main():
    parser = argparse.ArgumentParser(description="HoloAssist 主动服务标注整理与清洗")
    parser.add_argument(
        "--input_root",
        type=str,
        default="./data/HoloAssist/holoassist_service_annotations_rec_",
        help="包含 instant/ 与 short_term/ 的根目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/HoloAssist/holoassist_service_annotations_merged.json",
        help="合并后的大 JSON 输出路径",
    )
    parser.add_argument(
        "--merge_gap",
        type=float,
        default=DEFAULT_MERGE_GAP,
        help="规则 C 连续合并的时间间隔阈值（秒）",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="./outputs/consolidate_report.json",
        help="清洗统计报告输出路径",
    )
    args = parser.parse_args()

    print(f"读取根目录: {args.input_root}")
    print(f"合并阈值 merge_gap = {args.merge_gap}s")

    merged, stats, raw_counts, clean_counts, failed_files = consolidate(
        args.input_root, args.merge_gap
    )

    # 写出合并文件
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n合并文件已保存: {args.output}")

    # 组织报告
    report = {
        "merge_gap_seconds": args.merge_gap,
        "videos": stats["videos"],
        "raw_event_counts": dict(raw_counts),
        "clean_event_counts": dict(clean_counts),
        "cleaning_stats": {
            "rule_A_tooluse_over_errorrecovery (tool_use/error_recovery 重复, 保留 tool_use)":
                stats["rule_A_tooluse_over_errorrecovery"],
            "rule_B_other_conflicts (其它跨类型冲突, 按优先级保留)":
                stats["rule_B_other_conflicts"],
            "removed_conflict_events (因跨类型冲突被剔除的事件总数)":
                stats["removed_conflict_events"],
            "dedup_within_type (同类型内部完全重复被去重数)":
                stats["dedup_within_type"],
            "rule_C_merge_groups (连续合并产生的大事件组数)":
                stats["rule_C_merge_groups"],
            "rule_C_events_merged (被合并的原始事件总数)":
                stats["rule_C_events_merged"],
            "failed_request_files (失败/被取消的原始请求文件数, 按空处理)":
                stats["failed_request_files"],
        },
        "failed_files": [f"{svc}/{name}" for svc, name in failed_files],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印简报
    print("\n================ 清洗统计简报 ================")
    print(f"视频总数: {stats['videos']}")
    print("\n各类型事件数（清洗前 -> 清洗后）:")
    for svc in SERVICE_CONFIG:
        print(f"  {svc:20s}: {raw_counts[svc]:5d} -> {clean_counts[svc]:5d}")
    print("\n清洗动作:")
    for k, v in report["cleaning_stats"].items():
        print(f"  {k}: {v}")
    print(f"\n详细报告已保存: {args.report}")
    print("=" * 46)


if __name__ == "__main__":
    main()
