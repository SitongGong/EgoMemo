#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloAssist 主动服务标注修正 —— 第 1 个开源脚本（处理 Wrong Action，自包含）
============================================================================

本脚本从【原始 5 目录】直接起步，无需任何中间产物。核心是一条**单向管线**——
合并(会改变时间窗)永远放在最后，所有判断都在"原始小窗"状态下完成，
从根上杜绝"先合并/压平、LLM 判决拆不开"导致的跨类型重复或落地失败：

  ① 读取 + 同类型去重（consolidate_step0）
       读 instant/{safety,tool_use} 与 short_term/{error_recovery,next_step_guidance,
       resource_reminder} 5 目录，仅去掉同类型内部完全重复。不解冲突、不合并。
  ② 跨类型冲突 LLM 重判（合并之前，小窗状态）
       同一时间窗被标成多个服务类型时，把各候选类型的 observation 交 LLM 按定义判单一
       sub_type；因为还没合并，冲突窗是原始小窗，能干净归位、不被大窗吞并。
  ③ 漏网 Wrong Action 补判
       trainval 里 Action Correctness 以 'Wrong Action' 开头、却未被任何错误类事件覆盖的
       动作，交 LLM 判 sub_type(限 error_recovery/tool_use/safety) + 生成对话，补入。
  ④ 最后合并（类型已定型）
       4a 同类型连续合并(规则C，gap<=阈值)；4b 错误类相邻事件交 LLM 逐对判"是否同一错误"，
       是则合并、否则保留，迭代到收敛。此时跨类型冲突已解，合并只在同类型内，绝不产生跨类型重复。

输出 holoassist_service_annotations_merged_refined.json，作为 fix_next_step.py 的输入。
范围可选：默认全量；--videos_file 传视频名列表(如 rebuttal 子集)；--limit N 取前 N 个。
（next_step 的对应核对/封顶合并由第 2 个脚本 fix_next_step.py 负责，本脚本不动 next_step。）

服务类型范围（与 HoloAssist 体系一致，五种）：
    safety / tool_use (Instant)
    error_recovery / next_step_guidance / resource_reminder (Short-Term)

数据来源：
  - 全量清洗结果：holoassist_service_annotations_merged.json（用于取 rebuttal 视频的事件）
  - rebuttal 子集：holoassist_service_annotations_merged_rebuttal.json（193 视频，本脚本的修改对象）
  - 原始 5 目录：holoassist_service_annotations_rec_/...（用于重建冲突对）
  - 人工标注：data-annotation-trainval-v1_1.json（用于找 Wrong Action）

输出（新文件，不覆盖任何输入）：
  - holoassist_service_annotations_merged_rebuttal_llm.json
        在 merged_rebuttal 基础上：冲突窗按 LLM 重判结果归类；新增补判的漏网事件。
  - refine_rebuttal_report.json
        记录所有 LLM 决策、改动统计，便于追溯。

LLM 配置：默认 gpt-5.2，key 从 egomemo_demo/.env 的 OPENAI_API_KEY 读取。
带 --dry_run 时不调模型（任务A回落到原清洗保留的类型；任务B用 error_recovery 占位），
用于先验证流程与产出结构。

用法：
    conda activate egoserve
    python refine_rebuttal_with_llm.py                 # 在线，gpt-5.2
    python refine_rebuttal_with_llm.py --limit 5       # 只处理前5个视频（小样本验证）
    python refine_rebuttal_with_llm.py --dry_run       # 离线跑通流程
"""

import os
import re
import json
import time
import glob
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, Counter


# ===========================================================================
# 路径
# ===========================================================================
HOLO = os.environ.get("HOLOASSIST_DIR", "./data/HoloAssist")
MERGED_FULL = f"{HOLO}/holoassist_service_annotations_merged.json"
MERGED_REBUTTAL = f"{HOLO}/holoassist_service_annotations_merged_rebuttal.json"
RAW_ROOT = f"{HOLO}/holoassist_service_annotations_rec_"
TRAINVAL = f"{HOLO}/data-annotation-trainval-v1_1.json"
ENV_FILE = os.environ.get("EGOSERVE_ENV_FILE", ".env")

# 原始 5 目录布局：sub_type -> (high, events_key)
RAW_CFG = {
    "safety":            ("instant",    "safety_instant_events"),
    "tool_use":          ("instant",    "tool_use_instant_events"),
    "error_recovery":    ("short_term", "error_recovery_events"),
    "next_step_guidance":("short_term", "next_step_events"),
    "resource_reminder": ("short_term", "resource_reminder_events"),
}
ERROR_SUBS = {"safety", "tool_use", "error_recovery", "resource_reminder"}  # short_term/instant 错误类(参与覆盖判定/合并)
# 必须对应 Wrong Action 的子类型（①.5 约束用）：只含"做错动作"类。
# resource_reminder 本质是"该做未做/遗漏收尾"(火没关/门没锁/忘拿)，不是做错动作，
# 故不要求它对应 Wrong Action（与 next_step 同类，来源不同）。
WA_REQUIRED_SUBS = {"safety", "tool_use", "error_recovery"}
SUB_TO_HIGH = {
    "safety": "instant", "tool_use": "instant",
    "error_recovery": "short_term", "next_step_guidance": "short_term",
    "resource_reminder": "short_term",
}
FIVE_SUBS = ["safety", "tool_use", "error_recovery", "next_step_guidance", "resource_reminder"]
# 漏网 Wrong Action（已经做错的动作）的候选类型：语义上只可能是
# error_recovery（需回退重做）/ tool_use（仅技术问题）/ safety（造成即时人身危险）。
# 绝不可能是 next_step_guidance（那是步骤已做对、提示下一步）；
# resource_reminder（遗留未处理状态）对"刚做错的动作"也基本不适用，故排除。
WRONG_ACTION_SUBS = ["error_recovery", "tool_use", "safety"]


# ===========================================================================
# 步骤 0：从原始 5 目录 合并 + 清洗（内联自 consolidate_service_annotations.py）
#   功能与 consolidate_service_annotations.py 完全一致：把 instant/{safety,tool_use} 与
#   short_term/{error_recovery,next_step_guidance,resource_reminder} 5 个目录的逐视频
#   标注合并成一个大 dict，并执行三条清洗规则：
#     A) tool_use / error_recovery 同窗重复 -> 保留 tool_use
#     B) 其它跨类型同窗冲突 -> 按 _CONS_PRIORITY 保留其一；同类型内部完全重复去重
#     C) gap<=阈值 的同类型连续段合并为一个大事件
#   合并(C)在前、跨类型冲突(A/B)在后（合并产生的新窗才能正确判冲突）。
#   这样本脚本可直接从原始注释起步，无需先跑 consolidate 生成中间文件。
# ===========================================================================
_CONS_SERVICE_CONFIG = {
    "safety":            ("instant",    "safety_instant_events",    "risk_type"),
    "tool_use":          ("instant",    "tool_use_instant_events",  "risk_type"),
    "error_recovery":    ("short_term", "error_recovery_events",    "error_type"),
    "next_step_guidance":("short_term", "next_step_events",         "guidance_type"),
    "resource_reminder": ("short_term", "resource_reminder_events", "reminder_type"),
}
_CONS_PRIORITY = {"safety": 0, "tool_use": 1, "error_recovery": 2,
                  "resource_reminder": 3, "next_step_guidance": 4}


def _cons_parse_tw(tw):
    if not isinstance(tw, str) or tw.count("-") < 1:
        return None
    parts = tw.split("-")
    if len(parts) != 2:
        return None
    def to_sec(s):
        s = s.strip()
        if ":" in s:
            comps = [float(c) for c in s.split(":")]
            while len(comps) < 3:
                comps.insert(0, 0.0)
            return comps[0] * 3600 + comps[1] * 60 + comps[2]
        return float(s)
    try:
        return to_sec(parts[0]), to_sec(parts[1])
    except (ValueError, AttributeError):
        return None


def _cons_is_colon(tw):
    return isinstance(tw, str) and ":" in tw


def _cons_sec_to_colon(sec):
    if sec < 0:
        sec = 0.0
    hh = int(sec // 3600); mm = int((sec % 3600) // 60); ss = sec - hh * 3600 - mm * 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def _cons_fmt_tw(s, e, colon):
    if colon:
        return f"{_cons_sec_to_colon(s)}-{_cons_sec_to_colon(e)}"
    return f"{s:.3f}-{e:.3f}"


def _cons_load_file(path, event_key):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], [], "failed"
    if not isinstance(data, dict) or event_key not in data:
        return [], [], "failed"
    events = data.get(event_key) or []
    dialogs = data.get("dialogs") or []
    if not isinstance(events, list): events = []
    if not isinstance(dialogs, list): dialogs = []
    return events, dialogs, "ok"


def _cons_attach_dialogues(events, dialogs):
    tw2d = {}
    for d in dialogs:
        tw = d.get("time_window")
        if tw is not None and tw not in tw2d:
            tw2d[tw] = d.get("dialogue", [])
    out = []
    for e in events:
        e = dict(e)
        e["dialogue"] = tw2d.get(e.get("time_window"), [])
        out.append(e)
    return out


def _cons_norm_key(tw):
    p = _cons_parse_tw(tw)
    return ("raw", tw) if p is None else (round(p[0], 3), round(p[1], 3))


# 注：旧版用"固定优先级"解跨类型冲突的 _cons_resolve_conflicts 已删除。
# 新管线里跨类型冲突一律交给 ②(LLM 重判)处理，不再用优先级提前压平，
# 避免"先压平/合并、LLM 判决拆不开"的问题。_CONS_PRIORITY 仅 dry-run 兜底用。


def _cons_dedup(events, type_field, stats):      # 对于某一个视频中的服务类型去重，判断其是否与其他的服务类型重合
    seen = {}; res = []
    for e in events:
        key = (_cons_norm_key(e.get("time_window")), e.get(type_field))
        if key in seen:
            stats["consolidate_dedup"] += 1
            seen[key].setdefault("merged_sources", [])
            obs = e.get("observation")
            if obs and obs not in seen[key]["merged_sources"]:
                seen[key]["merged_sources"].append(obs)
            continue
        seen[key] = e; res.append(e)
    return res


def _cons_merge(events, type_field, gap, stats, barriers=None):
    """同类型连续合并(规则C)。barriers: 其它类型事件的 (s,e) 列表 —
    合并组的时间跨度不得吞并任何 barrier 窗(避免大窗跨越/包住别类小窗，产生跨类型重叠)。"""
    barriers = barriers or []
    if not events:
        return events
    parsable, unparsable = [], []
    for e in events:
        p = _cons_parse_tw(e.get("time_window"))
        (unparsable if p is None else parsable).append(e if p is None else (p[0], p[1], e))
    parsable.sort(key=lambda x: (x[0], x[1]))

    def swallows_barrier(gs, ge):
        # 合并窗 [gs,ge] 是否与某 barrier 窗有实质重叠(>0.05s)
        return any(min(ge, b1) - max(gs, b0) > 0.05 for b0, b1 in barriers)

    out = []; i = 0; n = len(parsable)
    while i < n:       # 对于每个时刻的该种服务类型
        group = [parsable[i]]; j = i
        while j + 1 < n:     # 从当前时刻开始遍历
            cur_end = group[-1][1]      # 当前服务的结束时间
            ns, ne, nev = parsable[j + 1]       # 下一个服务类型
            gstart = group[0][0]
            cand_end = max(cur_end, ne)
            if (nev.get(type_field) == group[-1][2].get(type_field)
                    and ns - cur_end <= gap
                    and not swallows_barrier(gstart, cand_end)):  # 护栏：合并后不得吞并别类窗
                group.append(parsable[j + 1]); j += 1
            else:
                break
        if len(group) == 1:    # 不存在连续现象
            out.append(group[0][2])
        else:
            stats["consolidate_ruleC_groups"] += 1
            base = dict(group[0][2])
            s = min(g[0] for g in group); e_ = max(g[1] for g in group)     # 对连续时间段的服务类型的时间窗口进行合并
            base["time_window"] = _cons_fmt_tw(s, e_, _cons_is_colon(group[0][2].get("time_window")))
            obs = []
            for _, _, e in group:
                o = e.get("observation")
                if o and o not in obs: obs.append(o)
            base["merged_observations"] = obs
            if obs: base["observation"] = obs[0]
            base["merged_count"] = len(group)
            base["dialogue"] = group[0][2].get("dialogue", [])
            out.append(base)
        i = j + 1
    out.extend(unparsable)
    return out


def consolidate_step0(input_root, stats):
    """
    管线第①步：读取原始 5 目录 + **仅做同类型内部完全重复去重**。
    刻意【不做】跨类型冲突解决，也【不做】规则C连续合并 ——
    这两件事分别由 ②(LLM 解冲突) 和 ④(最后合并) 负责。
    保持每个事件在"原始小窗"状态，便于 ② 干净地按窗归类、不被大窗吞并。
    返回 merged dict (与 merged.json 同构)。
    """
    per_video = defaultdict(lambda: {svc: [] for svc in _CONS_SERVICE_CONFIG})
    for svc, (high, ekey, _) in _CONS_SERVICE_CONFIG.items():
        for fp in sorted(glob.glob(os.path.join(input_root, high, svc, "*.json"))):
            vn = os.path.splitext(os.path.basename(fp))[0]
            events, dialogs, status = _cons_load_file(fp, ekey)
            if status == "failed":
                stats["consolidate_failed_files"] += 1
                continue
            per_video[vn][svc].extend(_cons_attach_dialogues(events, dialogs))
    merged = {}
    for vn in sorted(per_video.keys()):
        pt = per_video[vn]
        for svc in pt:
            tf = _CONS_SERVICE_CONFIG[svc][2]
            pt[svc] = _cons_dedup(pt[svc], tf, stats)   # 只去同类型完全重复，不合并、不解冲突
            pt[svc].sort(key=lambda e: (_cons_parse_tw(e.get("time_window")) or (0.0, 0.0))[0])
        out = {"instant": {}, "short_term": {}}
        for svc, (high, _, _) in _CONS_SERVICE_CONFIG.items():
            out[high][svc] = pt[svc]
        merged[vn] = out
    return merged


def merge_within_types(out, videos, merge_gap, stats):
    """
    管线第④步（最后一步）：对每个视频、每个错误类子类型，做规则C连续合并。
    此时跨类型冲突已由 ② 解决、漏网 WA 已由 ③ 补入，所有事件类型已定型，
    故合并只在同类型内进行，绝不会把不同类型的窗合到一起 -> 不产生跨类型重复。
    注：错误类相邻"是否同一错误"的 LLM 逐对判定由 consolidate_adjacent_same_type(任务C)负责，
        本函数只做确定性的同类型连续合并(规则C)。next_step 的合并留给 fix_next_step。
    """
    for vn in videos:
        # 收集该视频【所有事件】的时间窗，按类型分组，用于"合并不得跨越别类事件"的护栏
        all_wins = {}  # svc -> [(s,e), ...]
        for high in out[vn]:
            for svc, evs in out[vn][high].items():
                ws = []
                for e in evs:
                    p = parse_tw(e.get("time_window", ""))
                    if p:
                        ws.append(p)
                all_wins[svc] = ws

        for high in out[vn]:
            for svc in out[vn][high]:
                if svc not in ERROR_SUBS:
                    continue
                tf = _CONS_SERVICE_CONFIG[svc][2]
                # 护栏：其它类型事件的时间窗。合并 svc 段时不得跨过它们(否则会吞并别类小窗)。
                barriers = [w for s2, ws in all_wins.items() if s2 != svc for w in ws]
                before = len(out[vn][high][svc])
                out[vn][high][svc] = _cons_merge(out[vn][high][svc], tf, merge_gap, stats,
                                                 barriers=barriers)
                stats["step4_merged_reduced"] += before - len(out[vn][high][svc])


# ===========================================================================
# 服务类型定义（喂给 LLM，简明版，与 HoloAssist prompt 语义一致）
# ===========================================================================
SUB_DEFINITIONS = """HoloAssist proactive service sub-types (choose exactly ONE):
- safety (Instant): an immediate bodily hazard within seconds (burn, cut, electric shock, slip,
  rotating part, falling/heavy object). If any acute physical danger is present, choose safety.
- tool_use (Instant): improper tool operation / technique (grip, posture, orientation, contact,
  guard/clamp, power-off timing) where the step choice itself is CORRECT and NO rollback is needed;
  fixing it means adjusting the ongoing action.
- error_recovery (Short-Term): a wrong workflow state already produced (wrong object/target/slot/
  order/parameter, or a skipped required step) that must be undone and redone (rollback + redo).
- next_step_guidance (Short-Term): the current step is done correctly; the user is pausing and the
  assistant suggests the next logical action.
- resource_reminder (Short-Term): an end-state left unhandled (power/heat left on, door/cap open,
  unsaved work, leftover item, low supply) that should be closed/saved/taken/refilled."""


# ===========================================================================
# 时间窗解析
# ===========================================================================
def parse_tw(s):
    """'HH:MM:SS.mmm-...' 或 'MM:SS.mmm-...' 或 '12.3-15.6' -> (start,end) 秒。"""
    if not isinstance(s, str) or "-" not in s:
        return None
    a, b = s.split("-", 1)
    def sec(x):
        x = x.strip()
        if ":" in x:
            p = [float(z) for z in x.split(":")]
            while len(p) < 3:
                p.insert(0, 0.0)
            return p[0] * 3600 + p[1] * 60 + p[2]
        return float(x)
    try:
        return round(sec(a), 3), round(sec(b), 3)
    except ValueError:
        return None


def sec_to_hms(t):
    if t < 0:
        t = 0.0
    h = int(t // 3600); m = int((t % 3600) // 60); s = t - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def fmt_tw(s, e):
    return f"{sec_to_hms(s)}-{sec_to_hms(e)}"


def get_sub_of_event(high, sub_type):
    return sub_type


# ===========================================================================
# LLM 客户端
# ===========================================================================
class LLM:
    def __init__(self, model, dry_run=False, max_retries=4, retry_delay=3.0):
        self.model = model
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        if not dry_run:
            key = None
            for line in open(ENV_FILE):
                if line.strip().startswith("OPENAI_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if not key:
                raise ValueError(f"未在 {ENV_FILE} 找到 OPENAI_API_KEY")
            from openai import OpenAI
            self.client = OpenAI(api_key=key)

    def json_call(self, prompt):
        """调用模型并解析 JSON 对象；失败重试，最终失败返回 None。"""
        if self.dry_run:
            return None
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                txt = r.choices[0].message.content
                return json.loads(txt)
            except Exception as e:
                last = e
                time.sleep(self.retry_delay)
        print(f"  [LLM失败] {last}")
        return None


# ===========================================================================
# 任务A：重建跨类型冲突对（从原始 5 目录）
# ===========================================================================
def build_conflicts(video_names, input_root=RAW_ROOT):
    """
    返回 {video: {tw_key(start,end): {sub_type: observation, ...}}}，
    只保留出现在 >1 个类型的时间窗（即冲突）。从原始 5 目录(input_root)重建。
    """
    conflicts = {}
    for vn in video_names:
        tw_map = defaultdict(dict)   # (s,e) -> {sub: observation}
        for sub, (high, ek) in RAW_CFG.items():
            fp = os.path.join(input_root, high, sub, vn + ".json")
            if not os.path.exists(fp):
                continue
            try:
                d = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(d, dict) or ek not in d:
                continue
            for e in (d.get(ek) or []):
                key = parse_tw(e.get("time_window", ""))
                if key is None:
                    continue
                # 同一类型同窗可能多条，保留第一条 observation
                if sub not in tw_map[key]:
                    tw_map[key][sub] = e.get("observation", "")
        vc = {k: v for k, v in tw_map.items() if len(v) > 1}
        if vc:
            conflicts[vn] = vc
    return conflicts


def llm_resolve_conflict(llm, video, tw_key, type2obs):
    """对一个冲突窗用 LLM 判定单一 sub_type，并顺带生成一段统一对话。

    返回 (sub_type, reason, dialogue)；失败返回 (None, None, None)。
    生成对话是为了：当判出的类型在清洗后已无该窗事件、需要新建归位事件时，
    该事件也能带上对话（而不是留空）。
    """
    payload = {
        "video": video,
        "time_window": fmt_tw(*tw_key),
        "candidate_types": sorted(type2obs.keys()),
        "observations": type2obs,
    }
    prompt = (
        f"{SUB_DEFINITIONS}\n\n"
        "A single annotated error segment was assigned MULTIPLE conflicting service "
        "sub-types (one observation per type below). Re-judge by the definitions and "
        "output the ONE most appropriate sub-type for this segment, then write a short "
        "proactive dialogue for it.\n\n"
        f"Segment (JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Dialogue rules: assistant speaks first; 2 turns (assistant, user); polite and "
        "supportive; address this specific situation; no timestamps; do not say "
        "'error'/'mistake'/'annotation'.\n\n"
        'Return STRICT JSON: {"sub_type":"<one of: safety|tool_use|error_recovery|'
        'next_step_guidance|resource_reminder>","reason":"<short>",'
        '"dialogue":[{"role":"assistant","utterance":"..."},{"role":"user","utterance":"..."}]}'
    )
    res = llm.json_call(prompt)
    if res and res.get("sub_type") in FIVE_SUBS:
        return res["sub_type"], res.get("reason", ""), res.get("dialogue", [])
    return None, None, None


# ===========================================================================
# 任务B：找漏网 Wrong Action（与 merged_rebuttal 错误类事件无时间重叠）
# ===========================================================================
def collect_error_windows(video_entry):
    wins = []
    for high in video_entry:
        for sub, evs in video_entry[high].items():
            if sub not in ERROR_SUBS:
                continue
            for e in evs:
                p = parse_tw(e.get("time_window", ""))
                if p:
                    wins.append(p)
    return wins


def overlaps(a0, a1, wins):
    return any(a0 <= w1 and a1 >= w0 for w0, w1 in wins)


def wrong_action_starts(rec):
    """返回该视频 trainval 里所有 Wrong Action 的起点列表。"""
    starts = []
    for e in rec.get("events", []):
        if e.get("label") == "Fine grained action" and \
           e.get("attributes", {}).get("Action Correctness", "").startswith("Wrong Action"):
            starts.append(round(e["start"], 3))
    return starts


def filter_error_must_match_wa(out, videos, trainval, stats, tol=1.0):
    """
    管线 ①.5 步：做错动作类(safety/tool_use/error_recovery)必须对应 trainval 的某个
    Wrong Action —— 起点 ≈ 某 Wrong Action 起点(±tol)。不满足的(实为 Correct Action /
    otherwise 等被误转成错误类的)直接剔除。
    resource_reminder 不在此约束内（它是"该做未做/遗漏收尾"，不源于 Wrong Action）。
    放在冲突重判(②)之前：先去掉挂错的，再判冲突，避免它们干扰冲突判定。
    注：此时是 ①(去重)之后的原始小窗状态，起点对齐判定可靠。
    """
    for vn in videos:
        rec = trainval.get(vn, {})
        wa_starts = wrong_action_starts(rec)
        v = out[vn]
        for high in v:
            for sub in list(v[high].keys()):
                if sub not in WA_REQUIRED_SUBS:   # 只约束做错动作类，排除 resource_reminder
                    continue
                kept = []
                for e in v[high][sub]:
                    p = parse_tw(e.get("time_window", ""))
                    ok = p is not None and any(abs(p[0] - ws) < tol for ws in wa_starts)
                    if ok:
                        kept.append(e)
                    else:
                        stats["err_not_wa_removed"] += 1
                        stats[f"err_not_wa_{sub}"] += 1
                v[high][sub] = kept


def find_missed_wrong_actions(merged_rebuttal, trainval_by_video):
    """返回 {video: [wrong_action_event, ...]}（未被错误类事件时间窗覆盖的）。"""
    missed = defaultdict(list)
    for vn, v in merged_rebuttal.items():
        rec = trainval_by_video.get(vn)
        if not rec:
            continue
        wins = collect_error_windows(v)
        for ev in rec.get("events", []):
            if ev.get("label") != "Fine grained action":
                continue
            ac = ev.get("attributes", {}).get("Action Correctness", "")
            if not ac.startswith("Wrong Action"):
                continue
            s, e = ev.get("start"), ev.get("end")
            if not isinstance(s, (int, float)) or not isinstance(e, (int, float)):
                continue
            if not overlaps(s, e, wins):
                missed[vn].append(ev)
    return missed


def llm_judge_missed(llm, video, ev):
    """对一个漏网 Wrong Action 判服务类型 + 生成对话。"""
    attr = ev.get("attributes", {})
    payload = {
        "video": video,
        "time_window": fmt_tw(ev["start"], ev["end"]),
        "action": {
            "verb": attr.get("Verb"), "noun": attr.get("Noun"),
            "adjective": attr.get("Adjective"), "adverbial": attr.get("adverbial"),
        },
        "action_correctness": attr.get("Action Correctness"),
        "incorrect_action_explanation": attr.get("Incorrect Action Explanation"),
    }
    prompt = (
        f"{SUB_DEFINITIONS}\n\n"
        "Below is a human-annotated WRONG action from an instructional video — the user "
        "performed an action that was already judged incorrect. Decide the single most "
        "appropriate service sub-type, then write a short proactive dialogue.\n\n"
        "STRICT constraint — because this is an ALREADY-WRONG action, you MUST choose "
        "ONLY from these three sub-types:\n"
        "  - error_recovery: the wrong step/object/target/order produced a wrong state "
        "that must be undone and redone (this is the DEFAULT for most wrong actions).\n"
        "  - tool_use: the step choice was correct but the manipulation technique was "
        "wrong, fixable on the spot with NO rollback.\n"
        "  - safety: the wrong action created an immediate bodily hazard.\n"
        "Do NOT use next_step_guidance (that is for a CORRECTLY-completed step) or "
        "resource_reminder. A wrong action can never be next_step_guidance.\n"
        "When in doubt between error_recovery and tool_use, prefer error_recovery.\n\n"
        f"Wrong action (JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Dialogue rules: assistant speaks first; 2 turns (assistant, user); polite and "
        "supportive; clearly address this specific problem; no timestamps; do not say "
        "'error'/'mistake'/'annotation'.\n\n"
        'Return STRICT JSON: {"sub_type":"<error_recovery|tool_use|safety>",'
        '"observation":"<objective one-line>",'
        '"dialogue":[{"role":"assistant","utterance":"..."},{"role":"user","utterance":"..."}]}'
    )
    res = llm.json_call(prompt)
    if res and res.get("sub_type") in WRONG_ACTION_SUBS:
        return res
    return None


def make_missed_event(video, ev, judge):
    attr = ev.get("attributes", {})
    sub = judge["sub_type"]
    return {
        "clip_id": video,
        "segment_id": f"wrong_action_{ev.get('id','')}",
        "time_window": fmt_tw(ev["start"], ev["end"]),
        # 与各类型事件的类型字段名对齐：safety/tool_use 用 risk_type；
        # error_recovery 用 error_type；next_step 用 guidance_type；resource 用 reminder_type
        **_type_field(sub, attr),
        "observation": judge.get("observation", attr.get("Incorrect Action Explanation", "")),
        "source": "manual_annotation_recovered",
        "confidence": 0.8,
        "dialogue": judge.get("dialogue", []),
        "recovered_from_wrong_action": True,
        "action_correctness": attr.get("Action Correctness"),
    }


def merge_consecutive_missed(events, gap=1.5):
    """
    仿 consolidate 规则C：把同一视频里时间上连续（下一段 start - 当前段 end <= gap）
    的漏网 Wrong Action 合并成组。返回 [[ev,...], ...]，每个子列表是一组（可能只含1个）。
    按 start 排序后顺序扫描；组内不强制同 verb/noun（连续的反复尝试常用不同动词描述同一错误）。
    """
    evs = sorted(events, key=lambda e: (e.get("start", 0), e.get("end", 0)))
    groups = []
    i = 0
    while i < len(evs):
        j = i
        while j + 1 < len(evs) and (evs[j + 1].get("start", 0) - evs[j].get("end", 0)) <= gap:
            j += 1
        groups.append(evs[i:j + 1])
        i = j + 1
    return groups


def llm_judge_missed_group(llm, video, group):
    """
    对一个连续合并组（>=1 个 Wrong Action）判服务类型 + 生成一段统一对话。
    单元素组等价于原 llm_judge_missed；多元素组把组内所有动作一起喂给模型。
    """
    actions = []
    for ev in group:
        a = ev.get("attributes", {})
        actions.append({
            "time_window": fmt_tw(ev["start"], ev["end"]),
            "verb": a.get("Verb"), "noun": a.get("Noun"),
            "adjective": a.get("Adjective"), "adverbial": a.get("adverbial"),
            "action_correctness": a.get("Action Correctness"),
            "incorrect_action_explanation": a.get("Incorrect Action Explanation"),
        })
    s = min(ev["start"] for ev in group)
    e = max(ev["end"] for ev in group)
    payload = {
        "video": video,
        "time_window": fmt_tw(s, e),
        "note": ("These consecutive wrong actions are time-adjacent and describe the SAME "
                 "ongoing mistake; treat them as ONE error." if len(group) > 1
                 else "A single wrong action."),
        "wrong_actions": actions,
    }
    prompt = (
        f"{SUB_DEFINITIONS}\n\n"
        "Below are one or more human-annotated WRONG actions from an instructional video "
        "that were missed by the proactive-service annotations. When several are listed, "
        "they are time-adjacent and describe the SAME ongoing mistake, so judge them as a "
        "SINGLE error and produce ONE service event with ONE dialogue.\n\n"
        "STRICT constraint — because this is an ALREADY-WRONG action, choose ONLY from:\n"
        "  - error_recovery: a wrong step/object/target/order produced a wrong state that "
        "must be undone and redone (DEFAULT for most wrong actions).\n"
        "  - tool_use: the step choice was correct but the technique was wrong, fixable on "
        "the spot with NO rollback.\n"
        "  - safety: the wrong action created an immediate bodily hazard.\n"
        "Do NOT use next_step_guidance or resource_reminder. A wrong action can never be "
        "next_step_guidance. When unsure between error_recovery and tool_use, prefer error_recovery.\n\n"
        f"Wrong action(s) (JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Dialogue rules: assistant speaks first; 2 turns (assistant, user); polite and "
        "supportive; address the combined situation; no timestamps; do not say "
        "'error'/'mistake'/'annotation'.\n\n"
        'Return STRICT JSON: {"sub_type":"<error_recovery|tool_use|safety>",'
        '"observation":"<objective one-line covering the whole group>",'
        '"dialogue":[{"role":"assistant","utterance":"..."},{"role":"user","utterance":"..."}]}'
    )
    res = llm.json_call(prompt)
    if res and res.get("sub_type") in WRONG_ACTION_SUBS:
        return res
    return None


def make_missed_group_event(video, group, judge):
    """用合并组 + LLM 结果组装一个事件；time_window 取整组 [min start, max end]。"""
    sub = judge["sub_type"]
    s = min(ev["start"] for ev in group)
    e = max(ev["end"] for ev in group)
    first_attr = group[0].get("attributes", {})
    return {
        "clip_id": video,
        "segment_id": "wrong_action_" + "_".join(str(ev.get("id", "")) for ev in group),
        "time_window": fmt_tw(s, e),
        **_type_field(sub, first_attr),
        "observation": judge.get("observation", first_attr.get("Incorrect Action Explanation", "")),
        "source": "manual_annotation_recovered",
        "confidence": 0.8,
        "dialogue": judge.get("dialogue", []),
        "recovered_from_wrong_action": True,
        "num_merged": len(group),
        "merged_observations": [ev.get("attributes", {}).get("Incorrect Action Explanation", "") for ev in group],
        "action_correctness": first_attr.get("Action Correctness"),
    }


# ===========================================================================
# 任务C：错误类内部时间相邻事件，LLM 逐对判"是否同一错误"，是则合并
#   经任务A/B 后，同一错误子类型(tool_use/error_recovery/safety/resource_reminder)内部
#   仍可能残留时间几乎首尾相接(gap<=阈值)的两段。它们有的是同一错误被切两段(该合)，
#   有的是相邻的不同错误(不该合)。机械按 gap 合并会误合，故用 LLM 逐对判定。
# ===========================================================================
def llm_same_error(llm, video, sub_type, ev_a, ev_b):
    """问 LLM：两个相邻事件是否描述同一个错误。返回 (is_same, merged_observation)。"""
    payload = {
        "video": video, "service_sub_type": sub_type,
        "event_A": {"time_window": ev_a.get("time_window"), "observation": ev_a.get("observation", "")},
        "event_B": {"time_window": ev_b.get("time_window"), "observation": ev_b.get("observation", "")},
    }
    prompt = (
        "Two time-adjacent proactive-service events of the SAME sub-type were annotated on an "
        "instructional video. Decide whether they describe the SAME single underlying mistake "
        "(i.e. one continuous error split into two segments) or TWO DIFFERENT mistakes that "
        "merely happen to be adjacent in time.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "If they are the SAME mistake, also write one merged observation covering both.\n"
        'Return STRICT JSON: {"same_error": true/false, "merged_observation": "<one line, only if same>"}'
    )
    res = llm.json_call(prompt)
    if res is None or "same_error" not in res:
        return None, None
    return bool(res["same_error"]), res.get("merged_observation", "")


def merge_event_pair(ev_a, ev_b, merged_obs):
    """把判定为同一错误的相邻两事件合并为一个：time_window 取并集，记录合并信息与对话。"""
    pa, pb = parse_tw(ev_a["time_window"]), parse_tw(ev_b["time_window"])
    s, e = min(pa[0], pb[0]), max(pa[1], pb[1])
    base = dict(ev_a)
    base["time_window"] = fmt_tw(s, e)
    if merged_obs:
        base["observation"] = merged_obs
    # 累计合并来源
    prev = ev_a.get("merged_observations", [ev_a.get("observation", "")])
    base["merged_observations"] = prev + [ev_b.get("observation", "")]
    base["merged_count"] = ev_a.get("merged_count", 1) + ev_b.get("merged_count", 1)
    # 对话：保留较早事件的（若无则取后者的）
    base["dialogue"] = ev_a.get("dialogue") or ev_b.get("dialogue") or []
    base["llm_merged_same_error"] = True
    return base


def consolidate_adjacent_same_type(out, videos, llm, gap, stats, report):
    """任务C 主逻辑：对每视频每错误子类型，相邻 gap<=阈值 的事件用 LLM 判同一错误则合并。
    迭代到收敛（合并后可能与下一段又相邻）。"""
    for vn in videos:
        v = out[vn]
        for high in v:
            for sub in list(v[high].keys()):
                if sub not in ERROR_SUBS:
                    continue
                # 护栏：该视频【其它类型】事件的时间窗。合并后窗不得吞并它们(防跨类型重叠)。
                barriers = []
                for h2 in v:
                    for s2, evl in v[h2].items():
                        if s2 == sub:
                            continue
                        for e2 in evl:
                            p2 = parse_tw(e2.get("time_window", ""))
                            if p2:
                                barriers.append(p2)
                def _swallows(gs, ge):
                    return any(min(ge, b1) - max(gs, b0) > 0.05 for b0, b1 in barriers)

                evs = sorted(v[high][sub], key=lambda e: parse_tw(e["time_window"])[0])
                changed = True
                guard = 0
                while changed and guard < 50:
                    changed = False
                    guard += 1
                    i = 0
                    new_list = []
                    while i < len(evs):
                        if i + 1 < len(evs):
                            ca = parse_tw(evs[i]["time_window"])
                            cb = parse_tw(evs[i + 1]["time_window"])
                            # 相邻判定：起点差 gap 内（含重叠，即 cb 起点 <= ca 终点 + gap）
                            # 且合并后不得吞并别类事件窗（护栏）
                            merged_s = min(ca[0], cb[0]); merged_e = max(ca[1], cb[1])
                            if cb[0] - ca[1] <= gap and not _swallows(merged_s, merged_e):
                                stats["adj_pairs_checked"] += 1
                                same, mobs = (None, None)
                                if not llm.dry_run:
                                    same, mobs = llm_same_error(llm, vn, sub, evs[i], evs[i + 1])
                                if same:
                                    merged = merge_event_pair(evs[i], evs[i + 1], mobs)
                                    new_list.append(merged)
                                    stats["adj_merged"] += 1
                                    report["task_C_merged"].append({
                                        "video": vn, "sub_type": sub,
                                        "A": evs[i]["time_window"], "B": evs[i + 1]["time_window"],
                                    })
                                    i += 2
                                    changed = True
                                    continue
                                else:
                                    if same is False:
                                        stats["adj_kept_diff"] += 1
                        new_list.append(evs[i])
                        i += 1
                    evs = sorted(new_list, key=lambda e: parse_tw(e["time_window"])[0])
                v[high][sub] = evs


def _type_field(sub, attr):
    """按 sub_type 生成对应的类型字段（值用动词+名词概述，保持与原schema字段名一致）。"""
    tag = "other"
    if sub in ("safety", "tool_use"):
        return {"risk_type": tag}
    if sub == "error_recovery":
        return {"error_type": tag}
    if sub == "next_step_guidance":
        return {"guidance_type": tag}
    if sub == "resource_reminder":
        return {"reminder_type": tag}
    return {}


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="用 gpt-5.2 修正 rebuttal 标注（冲突重判+漏网补判）")
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--limit", type=int, default=100, help="只处理前 N 个视频（0=全部）")
    ap.add_argument("--missed_merge_gap", type=float, default=1.5,
                    help="任务B：连续漏网 Wrong Action 合并的时间间隔阈值（秒），与 consolidate 规则C一致")
    ap.add_argument("--adjacent_gap", type=float, default=1.5,
                    help="任务C：错误类内部相邻事件判定阈值（秒），gap<=此值的相邻对交 LLM 判是否同一错误")
    ap.add_argument("--dry_run", action="store_true")
    # 输入来源：默认从原始 5 目录直接合并清洗(自包含)；也可读现成 merged json
    ap.add_argument("--input_root", default=RAW_ROOT,
                    help="原始 5 目录根(instant/ short_term/)，从这里合并清洗起步")
    ap.add_argument("--workers", type=int, default=16,
                    help="任务②/③ LLM 调用并发线程数（独立调用，可并行加速）")
    ap.add_argument("--merge_gap", type=float, default=1.5,
                    help="第④步：同类型连续合并(规则C)阈值(秒)")
    ap.add_argument("--videos_file", default="",
                    help="可选：只处理该文件里列出的视频名(每行一个)；留空=全部")
    ap.add_argument("--output", default=f"{HOLO}/holoassist_service_annotations_merged_refined_debug.json")
    ap.add_argument("--report", default="./outputs/refine_rebuttal_report.json")
    args = ap.parse_args()

    trainval = {r["video_name"]: r for r in json.load(open(TRAINVAL, encoding="utf-8"))}

    # ===== 管线第①步：读取原始 5 目录 + 同类型去重（不解冲突、不合并）=====
    cons_stats = Counter()
    print(f"① 读取原始目录 + 同类型去重(不合并、不解冲突): {args.input_root}")
    merged = consolidate_step0(args.input_root, cons_stats)
    print(f"  读入 {len(merged)} 个视频")

    # 选择处理范围
    videos = sorted(merged.keys())
    if args.videos_file and os.path.exists(args.videos_file):
        want = {ln.strip() for ln in open(args.videos_file) if ln.strip()}
        videos = [v for v in videos if v in want]
        print(f"  按 --videos_file 限定到 {len(videos)} 个视频")
    if args.limit:
        videos = videos[:args.limit]
    print(f"处理 {len(videos)} 个视频" + ("（dry-run）" if args.dry_run else f"（{args.model}）"))

    llm = LLM(args.model, dry_run=args.dry_run)
    out = json.loads(json.dumps({vn: merged[vn] for vn in videos}, ensure_ascii=False))

    report = {"model": args.model, "dry_run": args.dry_run,
              "task_A_conflicts": [], "task_B_missed": [], "task_C_merged": [],
              "stats": Counter()}

    # ===== 管线 ①.5 步：错误类必须对应 Wrong Action，剔除挂错的 =====
    print("\n①.5 剔除不对应 Wrong Action 的错误类事件...")
    filter_error_must_match_wa(out, videos, trainval, report["stats"])
    print(f"  剔除: {report['stats'].get('err_not_wa_removed',0)} "
          f"(按类型 " + ", ".join(f"{k.replace('err_not_wa_','')}={v}" for k,v in report['stats'].items() if k.startswith('err_not_wa_') and k!='err_not_wa_removed') + ")")

    # ===== 管线第②步：跨类型冲突，LLM 重判（此时是原始小窗，归位干净）=====
    print("\n② 跨类型冲突 LLM 重判（合并之前，小窗状态）...")
    conflicts = build_conflicts(videos, input_root=args.input_root)
    n_conf = sum(len(v) for v in conflicts.values())
    print(f"  冲突时间窗总数: {n_conf}", flush=True)
    # 先收集所有需重判的冲突窗(跳过不对应WA的)
    conf_jobs = []
    for vn in videos:
        if vn not in conflicts:
            continue
        wa_starts = wrong_action_starts(trainval.get(vn, {}))
        for tw_key, type2obs in conflicts[vn].items():
            if not any(abs(tw_key[0] - ws) < 1.0 for ws in wa_starts):
                report["stats"]["conflict_skipped_not_wa"] += 1
                continue
            conf_jobs.append((vn, tw_key, type2obs))
    report["stats"]["conflict_windows"] = len(conf_jobs)

    # LLM 调用并发执行，结果按序串行 apply（apply 改共享 dict，串行避免竞态）
    def _resolve(job):
        vn, tw_key, type2obs = job
        if args.dry_run:
            prio = {"safety":0,"tool_use":1,"error_recovery":2,"resource_reminder":3,"next_step_guidance":4}
            return min(type2obs.keys(), key=lambda s: prio.get(s, 9)), "dry-run priority", []
        return llm_resolve_conflict(llm, vn, tw_key, type2obs)
    if args.dry_run:
        conf_results = [_resolve(j) for j in conf_jobs]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            conf_results = list(ex.map(_resolve, conf_jobs))
    for (vn, tw_key, type2obs), (chosen, reason, dialogue) in zip(conf_jobs, conf_results):
        if chosen is None:
            report["stats"]["conflict_llm_failed"] += 1
            continue
        apply_conflict_resolution(out[vn], tw_key, chosen, type2obs, video=vn, dialogue=dialogue)
        report["stats"]["conflict_resolved"] += 1
        report["task_A_conflicts"].append({
            "video": vn, "time_window": fmt_tw(*tw_key),
            "candidates": sorted(type2obs.keys()), "chosen": chosen, "reason": reason,
        })

    # ===== 管线第③步：漏网 Wrong Action 补判（仍是小窗，未合并）=====
    print("\n③ 查找漏网 Wrong Action 并用 LLM 补判...")
    missed = find_missed_wrong_actions({vn: out[vn] for vn in videos}, trainval)
    n_missed = sum(len(v) for v in missed.values())
    # 仿 consolidate 规则C：把时间上连续(gap<=阈值)的漏网动作先合并成一个错误，再补判
    # 先把每视频的漏网动作连续合并成组，收集所有组(group)作为并发任务
    miss_jobs = []  # (vn, group)
    for vn in videos:
        report["stats"]["missed_total"] += len(missed.get(vn, []))
        for group in merge_consecutive_missed(missed.get(vn, []), gap=args.missed_merge_gap):
            miss_jobs.append((vn, group))
    n_groups = len(miss_jobs)

    def _judge(job):
        vn, group = job
        if args.dry_run:
            return {"sub_type": "error_recovery",
                    "observation": group[0].get("attributes", {}).get("Incorrect Action Explanation", ""),
                    "dialogue": []}
        return llm_judge_missed_group(llm, vn, group)
    if args.dry_run:
        miss_results = [_judge(j) for j in miss_jobs]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            miss_results = list(ex.map(_judge, miss_jobs))

    for (vn, group), judge in zip(miss_jobs, miss_results):
        if len(group) > 1:
            report["stats"]["missed_merge_groups"] += 1
            report["stats"]["missed_merged_actions"] += len(group)
        if judge is None:
            report["stats"]["missed_llm_failed"] += 1
            continue
        new_e = (vn, group, judge)
        high = SUB_TO_HIGH[judge["sub_type"]]
        out[vn].setdefault(high, {}).setdefault(judge["sub_type"], []).append(new_e)
        report["stats"]["missed_added"] += 1
        report["stats"][f"missed_as_{judge['sub_type']}"] += 1
        report["task_B_missed"].append({
            "video": vn, "time_window": new_e["time_window"],
            "sub_type": judge["sub_type"], "num_merged": len(group),
            "action_correctness": group[0].get("attributes", {}).get("Action Correctness"),
        })
    print(f"  漏网 Wrong Action 总数: {n_missed} -> 连续合并(gap<={args.missed_merge_gap}s)后事件数: {n_groups}", flush=True)

    # ===== 管线第④步：最后合并（冲突已解、漏网已补，类型已定型）=====
    # 4a) 确定性的同类型连续合并(规则C)
    print("\n④ 最后合并：同类型连续合并(规则C)...")
    merge_within_types(out, videos, args.merge_gap, report["stats"])
    print(f"  规则C合并减少事件: {report['stats'].get('step4_merged_reduced',0)}")
    # 4b) 错误类相邻"是否同一错误"的 LLM 逐对判定(迭代收敛)
    print("   错误类相邻事件 LLM 逐对判'是否同一错误'...")
    consolidate_adjacent_same_type(out, videos, llm, args.adjacent_gap, report["stats"], report)
    print(f"   检查相邻对: {report['stats'].get('adj_pairs_checked',0)}, "
          f"判为同一错误合并: {report['stats'].get('adj_merged',0)}, "
          f"判为不同保留: {report['stats'].get('adj_kept_diff',0)}")

    # 各类型事件按时间排序
    for vn in videos:
        for high in out[vn]:
            for sub in out[vn][high]:
                out[vn][high][sub].sort(key=lambda e: (parse_tw(e.get("time_window","")) or (0,0))[0])

    # ---------- 写出 ----------
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for k, v in cons_stats.items():
        report["stats"][k] = v
    report["stats"] = dict(report["stats"])
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n================ 简报 ================")
    print(f"视频数: {len(videos)}")
    print(f"[A] 冲突窗: {report['stats'].get('conflict_windows',0)}, "
          f"重判成功: {report['stats'].get('conflict_resolved',0)}, "
          f"LLM失败: {report['stats'].get('conflict_llm_failed',0)}")
    print(f"[B] 漏网WrongAction: {report['stats'].get('missed_total',0)}, "
          f"补入: {report['stats'].get('missed_added',0)}, "
          f"LLM失败: {report['stats'].get('missed_llm_failed',0)}")
    print(f"    补入类型分布: " + ", ".join(
        f"{k.replace('missed_as_','')}={v}" for k,v in report['stats'].items() if k.startswith('missed_as_')))
    print(f"[C] 相邻对检查: {report['stats'].get('adj_pairs_checked',0)}, "
          f"判同一错误合并: {report['stats'].get('adj_merged',0)}, "
          f"判不同保留: {report['stats'].get('adj_kept_diff',0)}")
    print(f"输出: {args.output}")
    print(f"报告: {args.report}")
    print("=" * 38)


def apply_conflict_resolution(video_entry, tw_key, chosen, type2obs, video="", dialogue=None):
    """
    一个时间窗被标成了多个服务类型（冲突），LLM 已判它该是 chosen 这一种。
    本函数就是把这个窗变成"只属于 chosen"，消除跨类型重复。

    数据是按类型分桶存的（video_entry[high][sub] 各是一个事件列表），所以"改类型"
    在代码上表现为在列表之间挪动。两种情况：
      情况1：chosen 类型里本来就有这个窗
             -> 只需把【其它类型】里这个窗的重复事件删掉。
      情况2：chosen 类型里没有这个窗（它原本属于别的类型）
             -> 把那条事件的类型字段替换成 chosen（即从原类型列表移到 chosen 列表）。
      （兜底：万一连可替换的事件都找不到，才用 LLM 信息新建一条；正常数据走不到。）

    本函数在【合并之前】(管线第②步) 执行，冲突窗都还是原始小窗，与各类型事件
    精确相等(±0.05s)，故用精确匹配，不会出现旧版"被合并进大窗、拆不开"的问题。
    """
    s, e = tw_key

    def exact(ev):
        p = parse_tw(ev.get("time_window", ""))
        return p is not None and abs(p[0] - s) < 0.05 and abs(p[1] - e) < 0.05

    chosen_high = SUB_TO_HIGH[chosen]
    chosen_lst = video_entry.get(chosen_high, {}).get(chosen, [])
    chosen_has = any(exact(ev) for ev in chosen_lst)

    removed_event = None
    for sub in list(type2obs.keys()):
        if sub == chosen:
            continue
        high = SUB_TO_HIGH[sub]
        lst = video_entry.get(high, {}).get(sub, [])
        keep, hit = [], []
        for ev in lst:
            (hit if exact(ev) else keep).append(ev)
        if hit:
            video_entry[high][sub] = keep
            if removed_event is None:
                removed_event = hit[0]   # 留一条用于在 chosen 缺失时移过去

    if not chosen_has:
        if removed_event is not None:
            ev = removed_event
            for f in ("risk_type", "error_type", "guidance_type", "reminder_type"):
                ev.pop(f, None)
            ev.update(_type_field(chosen, {}))
            ev["reassigned_from_conflict"] = True
            video_entry.setdefault(chosen_high, {}).setdefault(chosen, []).append(ev)
        else:
            video_entry.setdefault(chosen_high, {}).setdefault(chosen, []).append({
                "clip_id": video, "segment_id": "conflict_reassigned",
                "time_window": fmt_tw(s, e),
                **_type_field(chosen, {}),
                "observation": type2obs.get(chosen, type2obs[next(iter(type2obs))]),
                "source": "manual_annotation", "confidence": 0.8,
                "dialogue": dialogue or [], "reassigned_from_conflict": True,
            })


if __name__ == "__main__":
    main()
