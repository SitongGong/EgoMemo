#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正 rebuttal next_step：两条标准（在 merged_rebuttal_llm 副本上做，不改原文件）。

标准一（必须对应 Conversation）：
  next_step 必须对应 trainval data-annotation-trainval-v1_1.json 里的 label:Conversation 事件。
  - 剔除无法对应任何 Conversation 的 next_step；
  - 对受影响的视频，用 gpt-5.2 从该视频的 Conversation 列表按 next_step 定义重新提取被遗漏的，
    新事件 time_window 来自对应 Conversation 的 start-end（保证满足标准一）。
  - 原有"能对应 Conversation"的 next_step 保留不动。

标准二（相邻不连续）：
  相邻两个 next_step 若 gap <= 1.5s 视为连续，合并为一个（与 consolidate 规则C一致）。
  对全部 193 个视频执行。

匹配口径（稳健）：next_step 视为"对应某 Conversation"当且仅当
  segment_id 末尾时间 ≈ 某 Conversation.start（±0.5s），或
  time_window 起点 ≈ 某 Conversation.start（±0.5s），或
  time_window 与某 Conversation 区间重叠。

执行顺序：先 LLM 重提取(标准一) -> 合并入该视频 next_step -> 再对所有视频做标准二合并。

用法：
  python fix_next_step.py                # gpt-5.2 在线
  python fix_next_step.py --dry_run      # 只做剔除+合并，不调LLM重提取（验证流程）
"""

import os
import re
import json
import argparse
from collections import Counter

HOLO = os.environ.get("HOLOASSIST_DIR", "./data/HoloAssist")
MERGED_LLM = f"{HOLO}/holoassist_service_annotations_merged_rebuttal_llm.json"
# 原始(未经规则C合并)的 next_step 目录——从这里重建，避免继承超长合并窗
RAW_NS_DIR = f"{HOLO}/holoassist_service_annotations_rec_/short_term/next_step_guidance"
TRAINVAL = f"{HOLO}/data-annotation-trainval-v1_1.json"
ENV_FILE = os.environ.get("EGOSERVE_ENV_FILE", ".env")
MERGE_GAP = 1.5

NEXT_STEP_DEF = """Next-Step Guidance (Short-Horizon proactive service): triggered AFTER the user has
just completed a CORRECT step; the assistant proactively suggests the next logical action in the
multi-step workflow, grounded in the instructor/student CONVERSATION. It is NOT error correction,
NOT tool-use technique, NOT safety. Only conversation segments where the instructor (or the dialogue)
guides the next workflow action after a step qualify."""


def parse_tw(s):
    a, b = s.split("-", 1)
    def sec(x):
        x = x.strip()
        if ":" in x:
            p = [float(z) for z in x.split(":")]
            while len(p) < 3:
                p.insert(0, 0.0)
            return p[0] * 3600 + p[1] * 60 + p[2]
        return float(x)
    return round(sec(a), 3), round(sec(b), 3)


def sec_to_hms(t):
    if t < 0: t = 0.0
    h = int(t // 3600); m = int((t % 3600) // 60); s = t - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def fmt_tw(s, e):
    return f"{sec_to_hms(s)}-{sec_to_hms(e)}"


def seg_time(sid):
    m = re.search(r"_(\d+\.?\d*)$", sid or "")
    return float(m.group(1)) if m else None


def load_key():
    for line in open(ENV_FILE):
        if line.strip().startswith("OPENAI_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def get_conversations(rec):
    """返回 [(start,end,purpose,text), ...]"""
    out = []
    for e in rec.get("events", []):
        if e.get("label") == "Conversation":
            a = e.get("attributes", {})
            out.append((round(e["start"], 3), round(e["end"], 3),
                        a.get("Conversation Purpose", ""), a.get("Transcription", "")))
    return out


def corresponds_conversation(ev, convs, tol=1.0):
    """
    判断 next_step 是否对应某 Conversation —— 严格口径：起点 ≈ 某 Conversation.start。
    next_step 本就源于"某条对话引导下一步"，因此其起点应与该对话起点对齐。
    NOT 用"时间窗与对话区间重叠"——那会让几分钟的大窗(把多条短对话套住)被误判为对应。
    """
    st = seg_time(ev.get("segment_id", ""))
    tw = parse_tw(ev["time_window"])
    if st is not None and any(abs(st - c0) < tol for c0, c1, *_ in convs):
        return True
    if any(abs(tw[0] - c0) < tol for c0, c1, *_ in convs):
        return True
    return False


MAX_MERGED_DUR = 60.0  # 合并封顶时长(秒)：超过 1 分钟视为过度合并，停止再并(用户口径)


def _merge_once(evs, gap, max_dur):
    """对已按起点排序的事件做一遍合并：相邻 gap<=阈值 且 合并后实际时长<=max_dur 才并。
    返回 (新列表, 是否发生过合并)。"""
    out = []
    i = 0
    changed = False
    while i < len(evs):
        group_start = parse_tw(evs[i]["time_window"])[0]
        group_end = parse_tw(evs[i]["time_window"])[1]
        j = i
        while j + 1 < len(evs):
            nxt = parse_tw(evs[j + 1]["time_window"])
            # 连续(下一段起点 - 当前组终点 <= gap) 且 合并后整组实际时长 <= max_dur
            new_end = max(group_end, nxt[1])
            if nxt[0] - group_end <= gap and (new_end - group_start) <= max_dur:
                group_end = new_end
                j += 1
            else:
                break
        if j == i:
            out.append(evs[i])
        else:
            changed = True
            members = evs[i:j + 1]
            base = dict(members[0])
            base["time_window"] = fmt_tw(group_start, group_end)
            # 累计已有的 merged_observations / merged_count（支持迭代）
            obs = []
            cnt = 0
            for e in members:
                obs.extend(e.get("merged_observations", [e.get("observation", "")]))
                cnt += e.get("merged_count", 1)
            base["merged_observations"] = obs
            base["merged_count"] = cnt
            out.append(base)
        i = j + 1
    return out, changed


def merge_consecutive(events, gap=MERGE_GAP, max_dur=MAX_MERGED_DUR):
    """
    标准二：相邻 gap<=阈值 合并为一个，合并后整组时长封顶 max_dur 秒（超 1 分钟视为过度合并）。
    迭代到收敛——因为剔除+重提取后插入的新事件可能与已合并组再次构成相邻，单遍不够。
    """
    if len(events) <= 1:
        return events
    evs = sorted(events, key=lambda e: parse_tw(e["time_window"])[0])
    for _ in range(20):  # 收敛上限，足够
        evs, changed = _merge_once(evs, gap, max_dur)
        if not changed:
            break
    return evs


def llm_reextract(client, model, video, convs, existing_starts):
    """用 LLM 从 Conversation 重提取被遗漏的 next_step。existing_starts: 已保留的对应窗起点集合，避免重复。"""
    conv_payload = [{"start": c0, "end": c1, "purpose": p, "text": t} for c0, c1, p, t in convs]
    prompt = (
        NEXT_STEP_DEF
        + f"\n\nVideo '{video}' conversation segments (manual annotation):\n"
        + json.dumps(conv_payload, ensure_ascii=False, indent=2)
        + "\n\nIdentify which conversation segments constitute Next-Step Guidance. For each, output an "
          "event whose time_window comes EXACTLY from that conversation's start-end. Write a 2-turn "
          "proactive dialogue (assistant first, then user). Do NOT include timestamps in dialogue.\n"
          'Return STRICT JSON: {"events":[{"conversation_start":<float>,"conversation_end":<float>,'
          '"guidance_type":"next_step_install|next_step_mixing|next_step_measure|next_step_cleanup|'
          'next_step_save_export|other","observation":"<objective>",'
          '"dialogue":[{"role":"assistant","utterance":"..."},{"role":"user","utterance":"..."}]}]}'
    )
    for _ in range(4):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"})
            res = json.loads(r.choices[0].message.content)
            break
        except Exception as e:
            print(f"    LLM重试: {e}")
            res = None
    if not res:
        return []
    new_events = []
    for it in res.get("events", []):
        cs, ce = it.get("conversation_start"), it.get("conversation_end")
        if cs is None or ce is None:
            continue
        # 不和已保留的重复（起点 ±0.5s）
        if any(abs(cs - s) < 0.5 for s in existing_starts):
            continue
        new_events.append({
            "clip_id": video,
            "segment_id": f"conversation_{cs}",
            "time_window": fmt_tw(float(cs), float(ce)),
            "guidance_type": it.get("guidance_type", "other"),
            "observation": it.get("observation", ""),
            "source": "manual_annotation_reextracted",
            "confidence": 0.8,
            "dialogue": it.get("dialogue", []),
            "reextracted_from_conversation": True,
        })
    return new_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--input", default=MERGED_LLM,
                    help="第一步(refine_rebuttal_with_llm.py)的输出文件")
    ap.add_argument("--raw_ns_dir", default=RAW_NS_DIR,
                    help="原始 next_step 目录(未经合并)，用于重建")
    ap.add_argument("--max_dur", type=float, default=MAX_MERGED_DUR,
                    help="next_step 连续合并封顶时长(秒)，超过视为过度合并")
    ap.add_argument("--output", default=f"{HOLO}/holoassist_service_annotations_final.json")
    ap.add_argument("--report", default="./outputs/next_step_fix_report.json")
    args = ap.parse_args()

    m = json.load(open(args.input, encoding="utf-8"))
    trainval = {r["video_name"]: r for r in json.load(open(TRAINVAL, encoding="utf-8"))}
    out = json.loads(json.dumps(m, ensure_ascii=False))  # 深拷贝

    client = None
    if not args.dry_run:
        from openai import OpenAI
        client = OpenAI(api_key=load_key())

    stats = Counter()
    report = {"model": args.model, "dry_run": args.dry_run, "per_video": {}}

    for vn, v in out.items():
        # 从原始目录(未经规则C合并)重建该视频的 next_step，避免继承超长合并窗
        raw_fp = os.path.join(args.raw_ns_dir, vn + ".json")
        ns = []
        if os.path.exists(raw_fp):
            try:
                rawd = json.load(open(raw_fp, encoding="utf-8"))
                if isinstance(rawd, dict):
                    # 原始 next_step 的对话在 dialogs 里，按 time_window 对齐补回
                    tw2dlg = {d.get("time_window"): d.get("dialogue", [])
                              for d in (rawd.get("dialogs") or [])}
                    for e in (rawd.get("next_step_events") or []):
                        ns.append({
                            "clip_id": vn,
                            "segment_id": e.get("segment_id", ""),
                            "time_window": e["time_window"],
                            "guidance_type": e.get("guidance_type", "other"),
                            "observation": e.get("observation", ""),
                            "source": e.get("source", "manual_annotation"),
                            "confidence": e.get("confidence", 1.0),
                            "dialogue": tw2dlg.get(e["time_window"], []),
                        })
            except Exception as ex:
                print(f"  读原始 next_step 失败 {vn}: {ex}")
        rec = trainval.get(vn)
        convs = get_conversations(rec) if rec else []

        # 标准一：剔除无对应 Conversation 的
        kept = [e for e in ns if corresponds_conversation(e, convs)]
        removed = len(ns) - len(kept)
        stats["removed_no_conv"] += removed

        added = 0
        if removed > 0 and not args.dry_run and convs:
            # 该视频受影响 -> LLM 重提取遗漏的
            existing_starts = [parse_tw(e["time_window"])[0] for e in kept]
            new_events = llm_reextract(client, args.model, vn, convs, existing_starts)
            kept.extend(new_events)
            added = len(new_events)
            stats["reextracted"] += added

        # 标准二：合并连续
        before = len(kept)
        kept = merge_consecutive(kept, MERGE_GAP, max_dur=args.max_dur)
        merged_reduced = before - len(kept)
        stats["merged_reduced"] += merged_reduced

        kept.sort(key=lambda e: parse_tw(e["time_window"])[0])
        v["short_term"]["next_step_guidance"] = kept

        if removed or added or merged_reduced:
            report["per_video"][vn] = {"removed": removed, "reextracted": added,
                                       "merged_reduced": merged_reduced, "final": len(kept)}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    report["stats"] = dict(stats)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print(f"标准一 剔除无对应Conversation: {stats['removed_no_conv']}")
    print(f"标准一 LLM重提取补入: {stats['reextracted']}" + ("（dry-run跳过）" if args.dry_run else ""))
    print(f"标准二 合并减少: {stats['merged_reduced']}")
    print(f"受影响视频数: {len(report['per_video'])}")
    print(f"输出: {args.output}")
    print(f"报告: {args.report}")


if __name__ == "__main__":
    main()
