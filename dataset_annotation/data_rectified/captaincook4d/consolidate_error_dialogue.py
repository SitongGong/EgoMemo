#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CaptionCook4D 错误→服务对话标注的整理与类型校验脚本
======================================================

背景：
    annotations/annotation_json/error_to_dialogue_results_.json 是由
    CaptionCook4D 的人工错误标注直接转换成的主动服务类型标注
    （原转换流程见 EgoLife/gemini_generation/ego_error_transform.py）。
    论文相关数据，发现两个问题需要修正（不改原文件，生成新文件）：

    问题1（同一错误被拆成多种服务类型 + 建议分开）：
        同一个 step 上若标注了多个错误，会被分别生成多个 item，
        每个 item 各有自己的 service_type 和一段对话。应当把同一 step 的
        这些 item 合并为"单一事件 + 一段统一对话"。

    问题2（服务类型划分不准/不一致）：
        同样性质的错误（如 Order Error）在不同样本里被划成不同 sub_type。
        需要重新校验每个事件的 service_type 是否合理，统一为单一类型。

方案（两步在同一次 LLM 调用内完成）：
    按 (recording_id, step_id) 对 dialogue.items 分组；对每组把该 step 的
    全部错误信息（error_tag / description / 原对话 / 原 service_type）一起
    交给 Gemini，让模型：
        (a) 重新判定一个最合理的 service_type（校验/纠正问题2），
        (b) 重写一段连贯的 1-2 轮统一对话（合并问题1）。
    单 item 的 step 也走同一流程，相当于对其类型再校验一遍。

    服务类型定义、main/sub 取值、调用方式均沿用 ego_error_transform.py。

输出：
    与原文件同构的 list；每个 recording 的 dialogue.items 经过合并与类型校验，
    每个 step 至多一个 item。新增字段（便于追溯，不影响原有字段）：
        - merged_from_error_tags: 该事件由哪些 error_tag 合并而来
        - original_sub_types:     合并前各 item 的原 sub_type
        - num_merged:             合并的 item 数

用法：
    conda activate egoserve
    # 在线（调用 Gemini Batch API）：
    python consolidate_error_dialogue.py
    # 离线 dry-run（确定性规则合并+类型统一，不调模型，先验证流程/产出结构）：
    python consolidate_error_dialogue.py --dry_run
"""

import os
import json
import time
import argparse
from typing import Optional, Dict, Any, List, Literal
from collections import defaultdict

from pydantic import BaseModel, Field


# ===========================================================================
# 服务类型定义（与 ego_error_transform.py 完全一致）
# ===========================================================================
# 注：Episodic 系列（Episodic Proactive Service / Episodic Task Reminder /
# Episodic Memory Recall）已从 CaptionCook4D 处理中移除，不再使用。
MAIN_TYPES = [
    "Instant Proactive Service",
    "Short-Term Proactive Service",
]
SUB_TYPES = [
    "Safety",
    "Tool Use",
    "Error-Recovery",
    "Next-Step Guidance",
    "Resource Reminder",
]

# sub -> main 的对应（用于校验/补全 main 字段）
SUB_TO_MAIN = {
    "Safety": "Instant Proactive Service",
    "Tool Use": "Instant Proactive Service",
    "Error-Recovery": "Short-Term Proactive Service",
    "Next-Step Guidance": "Short-Term Proactive Service",
    "Resource Reminder": "Short-Term Proactive Service",
}

# dry-run 用的 error_tag -> 默认 sub_type（仅离线兜底；在线时由 LLM 判定）
# 依据数据主导分布：Technique -> Tool Use，其余 -> Error-Recovery。
DRYRUN_TAG_DEFAULT_SUB = {
    "Technique Error": "Tool Use",
    "Order Error": "Error-Recovery",
    "Preparation Error": "Error-Recovery",
    "Measurement Error": "Error-Recovery",
    "Missing Step": "Error-Recovery",
    "Timing Error": "Error-Recovery",
    "Temperature Error": "Error-Recovery",
    "Other": "Error-Recovery",
}


# ===========================================================================
# LLM 输出 schema（一个 step 合并后只产出一个事件）
# ===========================================================================
class DialogueUtterance(BaseModel):
    role: Literal["assistant", "user"] = Field(..., description="Role of the speaker")
    utterance: str = Field(..., description="The spoken message (natural, respectful, concise)")


class ServiceType(BaseModel):
    main: Literal[
        "Instant Proactive Service",
        "Short-Term Proactive Service",
    ] = Field(..., description="Main category of the proactive service")
    sub: Literal[
        "Safety", "Tool Use", "Error-Recovery", "Next-Step Guidance",
        "Resource Reminder",
    ] = Field(..., description="Sub-category of the proactive service")


class MergedEvent(BaseModel):
    """同一 step 合并校验后的单个事件。"""
    service_type: ServiceType = Field(..., description="The single, validated service type for this step")
    dialogue: List[DialogueUtterance] = Field(
        ..., min_length=1, max_length=2,
        description="One unified proactive dialogue (1-2 turns) covering all errors on this step",
    )


OUTPUT_SCHEMA = MergedEvent


# ===========================================================================
# 校验 / 合并用的 system prompt（在 ego 原定义基础上，强调"合并+单一类型"）
# ===========================================================================
CONSOLIDATE_SYSTEM_PROMPT = """
You are a proactive-service annotation reviewer for egocentric instructional videos
(cooking, assembly, lab work). You are given ALL annotated errors that occur on ONE
single step of a task, together with the previously generated service types and
dialogues for those errors.

Two issues must be fixed:

(1) One step may have been split into several proactive-service items (one per error),
    each with its own service type and its own dialogue. You MUST merge them into a
    SINGLE proactive service for this step, with ONE unified, coherent dialogue that
    addresses the combined situation.

(2) The previously assigned service types are often inconsistent and may be wrong.
    You MUST re-judge and output exactly ONE most-appropriate service type for this
    step, based on the semantics of the errors — not by copying the old labels.

You MUST NOT invent new errors. Only reason about the errors that are explicitly given.

------------------------------------------------------------
Proactive Service Types (choose exactly ONE)
------------------------------------------------------------

### Instant Proactive Service (immediate, < 10 seconds)
1. Safety — the error could plausibly cause bodily harm or an immediate accident
   (burns, cuts, shock, unsafe proximity, hazardous spills/heat). If any error could
   endanger the body, it MUST be Safety, even if a tool/technique is also involved.
2. Tool Use — improper tool operation / unstable handling / incorrect technique that
   does NOT yet require rollback (bad pouring technique, unstable grip, poor control).

### Short-Term Proactive Service (10 seconds - 10 minutes)
3. Error-Recovery — the user already completed a wrong step and it must be corrected
   or redone (wrong measurement added, wrong component assembled, wrong configuration).
4. Next-Step Guidance — the step is done correctly and the user is transitioning to the
   next stage; guidance moves the workflow forward.
5. Resource Reminder — something is left unfinished/unresolved while moving on
   (leftover material, missing cleanup, container not closed, power/heat not turned off).

(Note: there are exactly FIVE sub-types above — Safety, Tool Use, Error-Recovery,
 Next-Step Guidance, Resource Reminder. Choose ONLY from these five.)

------------------------------------------------------------
Priority rules when multiple errors are on the same step
------------------------------------------------------------
- If ANY error could cause bodily injury -> Safety.
- Else if the dominant problem is improper technique/tool handling with no rollback
  needed -> Tool Use.
- Else if any error requires correcting/redoing what was done -> Error-Recovery.
- Use Next-Step Guidance / Resource Reminder only when they clearly fit
  better than the above per the definitions.
- Output ONE type that best captures the merged situation.

------------------------------------------------------------
Dialogue rules
------------------------------------------------------------
- Output ONE dialogue of 1-2 turns; the assistant speaks first.
- Polite, calm, respectful, supportive — never scold or command.
- Do NOT mention "error", "mistake", or "annotation".
- The single dialogue must naturally cover all the given errors on this step together.

------------------------------------------------------------
Output (STRICT JSON, a single object)
------------------------------------------------------------
{
  "service_type": { "main": "<Instant Proactive Service | Short-Term Proactive Service>",
                    "sub": "<Safety | Tool Use | Error-Recovery | Next-Step Guidance | Resource Reminder>" },
  "dialogue": [
    { "role": "assistant", "utterance": "<unified proactive message>" },
    { "role": "user", "utterance": "<optional short user reply>" }
  ]
}
"""


# ===========================================================================
# 分组：按 (recording_id, step_id) 把同 step 的 item 聚合
# ===========================================================================
def group_items_by_step(items: List[Dict[str, Any]]):
    """
    返回有序的分组列表：[(step_id, [item, ...]), ...]
    保持每个 step 第一次出现的顺序，组内保持原始顺序。
    """
    order = []
    groups = defaultdict(list)
    for it in items:
        sid = it.get("step_id")
        if sid not in groups:
            order.append(sid)
        groups[sid].append(it)
    return [(sid, groups[sid]) for sid in order]


def build_group_payload(step_id, group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把一个 step 的多个 item 打包成喂给 LLM 的结构化输入。"""
    errors = []
    for it in group:
        errors.append({
            "error_tag": it.get("error_tag"),
            "step_description": it.get("description"),
            "previous_service_type": it.get("service_type"),
            "previous_dialogue": it.get("dialogue"),
        })
    # 时间窗：取组内的最小 start / 最大 end（同 step 通常一致）
    starts = [it.get("start_time") for it in group if isinstance(it.get("start_time"), (int, float))]
    ends = [it.get("end_time") for it in group if isinstance(it.get("end_time"), (int, float))]
    return {
        "step_id": step_id,
        "description": group[0].get("description"),
        "start_time": min(starts) if starts else group[0].get("start_time"),
        "end_time": max(ends) if ends else group[0].get("end_time"),
        "errors_on_this_step": errors,
    }


def assemble_merged_item(payload, group, llm_result) -> Dict[str, Any]:
    """
    用 LLM 结果（或 dry-run 结果）组装最终合并 item，保留与原文件一致的字段。
    """
    sub = llm_result["service_type"]["sub"]
    # 防御：万一出现已废弃的 Episodic（schema 已禁止），重映射到 Error-Recovery
    if sub not in SUB_TO_MAIN:
        sub = "Error-Recovery"
    # main 以 sub 推导，保证内部一致
    main = SUB_TO_MAIN[sub]

    return {
        "step_id": payload["step_id"],
        "description": payload["description"],
        "start_time": payload["start_time"],
        "end_time": payload["end_time"],
        "error_tag": group[0].get("error_tag"),  # 主 error_tag（第一个）
        "service_type": {"main": main, "sub": sub},
        "dialogue": llm_result["dialogue"],
        # 追溯字段
        "merged_from_error_tags": [it.get("error_tag") for it in group],
        "original_sub_types": [it.get("service_type", {}).get("sub") for it in group],
        "num_merged": len(group),
    }


# ===========================================================================
# dry-run：离线确定性合并 + 类型统一（不调模型）
# ===========================================================================
def dryrun_merge(payload, group) -> Dict[str, Any]:
    """
    离线兜底逻辑：
      - service_type：组内若有 Safety 则取 Safety（安全优先）；否则按"出现最多的
        error_tag 的默认 sub"。这里用简单稳健规则，仅用于流程验证。
        （Episodic 系列已废弃，不会产生。）
      - dialogue：取组内第一条对话作为统一对话（不调模型生成）。
    """
    subs = [it.get("service_type", {}).get("sub") for it in group]
    tags = [it.get("error_tag") for it in group]

    if "Safety" in subs:
        chosen_sub = "Safety"
    else:
        # 取出现最多的 error_tag，再映射默认 sub
        from collections import Counter
        top_tag = Counter(tags).most_common(1)[0][0]
        chosen_sub = DRYRUN_TAG_DEFAULT_SUB.get(top_tag, "Error-Recovery")

    dialogue = group[0].get("dialogue") or [
        {"role": "assistant", "utterance": payload.get("description", "")}
    ]
    return {
        "service_type": {"sub": chosen_sub, "main": SUB_TO_MAIN.get(chosen_sub)},
        "dialogue": dialogue,
    }


# ===========================================================================
# Gemini Batch API 调用（沿用 ego_error_transform.py 的方式）
# ===========================================================================
class GeminiConsolidator:
    def __init__(self, api_key, model_name, check_interval=30,
                 max_retries=3, retry_delay=2.0):
        import google.generativeai as genai
        from google import genai as genai_batch
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("未提供 API key，请用 --api_key 或设置 GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.batch_client = genai_batch.Client(api_key=self.api_key)
        self.model_name = model_name
        self.check_interval = check_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _build_request(self, payload):
        prompt = (CONSOLIDATE_SYSTEM_PROMPT
                  + "\n\nErrors on this single step (JSON):\n"
                  + json.dumps(payload, ensure_ascii=False, indent=2))
        return {
            "contents": [{"parts": [{"text": prompt}], "role": "user"}],
            "config": {
                "response_mime_type": "application/json",
                "response_schema": OUTPUT_SCHEMA.model_json_schema(),
            },
        }

    def run_batch(self, payloads, display_name):
        """对一批 step payload 调用 Batch API，返回与输入等长的结果列表（失败位为 None）。"""
        requests = [self._build_request(p) for p in payloads]
        last_err = None
        for attempt in range(self.max_retries):
            try:
                job = self.batch_client.batches.create(
                    model=f"models/{self.model_name}",
                    src=requests,
                    config={"display_name": display_name},
                )
                name = job.name
                bad = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
                while True:
                    job = self.batch_client.batches.get(name=name)
                    if job.state.name == "JOB_STATE_SUCCEEDED":
                        break
                    if job.state.name in bad:
                        raise RuntimeError(f"Batch job 失败: {job.state.name}")
                    time.sleep(self.check_interval)

                results = [None] * len(payloads)
                responses = job.dest.inlined_responses if job.dest else []
                for i, resp in enumerate(responses):
                    if i >= len(results):
                        break
                    if getattr(resp, "error", None) or not getattr(resp, "response", None):
                        continue
                    try:
                        cand = resp.response.candidates[0]
                        text = cand.content.parts[0].text
                        results[i] = json.loads(text)
                    except Exception as e:
                        print(f"  解析响应 {i} 失败: {e}")
                        results[i] = None
                return results
            except Exception as e:
                last_err = e
                print(f"  尝试 {attempt+1}/{self.max_retries} 失败: {e}")
                time.sleep(self.retry_delay)
        raise RuntimeError(f"Batch API 调用失败: {last_err}")


# ===========================================================================
# OpenAI 后端（gpt-5.2 等），逐请求 chat.completions + JSON mode
# 与 GeminiConsolidator 提供相同的 run_batch 接口，便于在 main 里切换。
# key 默认从 egomemo_demo/.env 的 OPENAI_API_KEY 读取（实测可用）。
# ===========================================================================
OPENAI_ENV_FILE = "/mnt/workspace/gst/egomemo_demo/.env"


def _load_openai_key(explicit=None):
    if explicit:
        return explicit
    env = os.getenv("OPENAI_API_KEY")
    if env:
        return env
    if os.path.exists(OPENAI_ENV_FILE):
        for line in open(OPENAI_ENV_FILE):
            if line.strip().startswith("OPENAI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class OpenAIConsolidator:
    def __init__(self, api_key, model_name, max_retries=4, retry_delay=3.0, workers=16, **_):
        key = _load_openai_key(api_key)
        if not key:
            raise ValueError("未找到 OPENAI_API_KEY（--api_key / 环境变量 / egomemo_demo/.env）")
        from openai import OpenAI
        self.client = OpenAI(api_key=key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.workers = workers

    def _prompt(self, payload):
        return (CONSOLIDATE_SYSTEM_PROMPT
                + "\n\nErrors on this single step (JSON):\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
                + "\n\nReturn STRICT JSON with exactly the schema described above "
                  '(a single object with "service_type" and "dialogue").')

    def _one(self, payload):
        last = None
        for _ in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": self._prompt(payload)}],
                    response_format={"type": "json_object"},
                )
                return json.loads(r.choices[0].message.content)
            except Exception as e:
                last = e
                time.sleep(self.retry_delay)
        print(f"  [OpenAI失败] {last}")
        return None

    def run_batch(self, payloads, display_name):
        """并发处理一批 payload，返回与输入等长的结果列表（失败位为 None）。"""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self._one, payloads))


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="CaptionCook4D 错误对话标注整理+类型校验")
    parser.add_argument(
        "--input", type=str,
        default="./data/CaptainCook4D/error_to_dialogue_results_.json",
    )
    parser.add_argument(
        "--output", type=str,
        default="./data/CaptainCook4D/error_to_dialogue_results_consolidated.json",
    )
    parser.add_argument("--backend", type=str, default="openai",
                        choices=["openai", "gemini"],
                        help="LLM 后端：openai(gpt-5.2,默认,实测可用) 或 gemini(原Batch API)")
    parser.add_argument("--api_key", type=str, default="",
                        help="留空则按后端自动取 key：openai 读 egomemo_demo/.env，gemini 读 GEMINI_API_KEY")
    parser.add_argument("--model_name", type=str, default="gpt-5.2",
                        help="openai 默认 gpt-5.2；gemini 可用 gemini-2.5-pro")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="每批提交的 step 数")
    parser.add_argument("--workers", type=int, default=16,
                        help="openai 后端并发线程数（每个 step 独立调用，可并行加速）")
    parser.add_argument("--check_interval", type=int, default=30)
    parser.add_argument("--dry_run", action="store_true",
                        help="离线确定性合并，不调用 LLM（用于流程/结构验证）")
    parser.add_argument(
        "--report", type=str,
        default="./outputs/cc4d_consolidate_report.json",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"读取 {len(data)} 个 recording")

    # 1) 收集所有 step 分组（跨 recording 拉平，便于批量送 LLM），记录归属
    #    flat_jobs: [(rec_idx, step_id, payload, group), ...]
    flat_jobs = []
    per_rec_groups = []  # 与 data 对齐：每个 recording 的 [(step_id, group), ...]
    for rec_idx, rec in enumerate(data):
        items = rec.get("dialogue", {}).get("items", []) or []
        grouped = group_items_by_step(items)
        per_rec_groups.append(grouped)
        for sid, group in grouped:
            payload = build_group_payload(sid, group)
            flat_jobs.append((rec_idx, sid, payload, group))

    n_steps = len(flat_jobs)
    n_multi = sum(1 for *_, g in flat_jobs if len(g) > 1)
    print(f"待处理 step 事件数: {n_steps}（其中需合并多 item 的: {n_multi}）")

    # 2) 逐 step 得到合并校验结果
    results_per_job = [None] * n_steps
    if args.dry_run:
        print("== dry-run：离线确定性合并，不调用 LLM ==")
        for idx, (_, _, payload, group) in enumerate(flat_jobs):
            results_per_job[idx] = dryrun_merge(payload, group)
    else:
        if args.backend == "openai":
            consolidator = OpenAIConsolidator(
                api_key=args.api_key, model_name=args.model_name, workers=args.workers,
            )
        else:
            consolidator = GeminiConsolidator(
                api_key=args.api_key or os.getenv("GEMINI_API_KEY", ""),
                model_name=args.model_name, check_interval=args.check_interval,
            )
        payloads = [j[2] for j in flat_jobs]
        for start in range(0, n_steps, args.batch_size):
            end = min(start + args.batch_size, n_steps)
            print(f"== Batch {start}-{end-1} / {n_steps} ==")
            batch_res = consolidator.run_batch(
                payloads[start:end], display_name=f"cc4d-consolidate-{start}"
            )
            for k, r in enumerate(batch_res):
                # LLM 失败时回落到 dry-run 规则，保证不丢事件
                if r is None:
                    _, _, p, g = flat_jobs[start + k]
                    r = dryrun_merge(p, g)
                    r["_fallback"] = True
                results_per_job[start + k] = r

    # 3) 组装回每个 recording，保持原文件结构
    out_data = []
    job_cursor = 0
    stats = {"total_recordings": len(data), "items_before": 0, "items_after": 0,
             "merged_groups": 0, "items_removed_by_merge": 0, "fallback_used": 0,
             "subtype_after": {}, "subtype_changed": 0}
    # job 按 flat 顺序，per_rec_groups 也按相同顺序生成，可对齐消费
    for rec_idx, rec in enumerate(data):
        grouped = per_rec_groups[rec_idx]
        new_items = []
        for sid, group in grouped:
            llm_result = results_per_job[job_cursor]
            job_cursor += 1
            stats["items_before"] += len(group)
            if len(group) > 1:
                stats["merged_groups"] += 1
                stats["items_removed_by_merge"] += (len(group) - 1)
            if isinstance(llm_result, dict) and llm_result.get("_fallback"):
                stats["fallback_used"] += 1
            merged_item = assemble_merged_item(
                build_group_payload(sid, group), group, llm_result
            )
            # 统计类型变化
            new_sub = merged_item["service_type"]["sub"]
            stats["subtype_after"][new_sub] = stats["subtype_after"].get(new_sub, 0) + 1
            orig_subs = set(merged_item["original_sub_types"])
            if not (len(orig_subs) == 1 and new_sub in orig_subs):
                stats["subtype_changed"] += 1
            new_items.append(merged_item)
        stats["items_after"] += len(new_items)

        out_rec = {
            "recording_id": rec.get("recording_id"),
            "activity_id": rec.get("activity_id"),
            "step_annotations": rec.get("step_annotations"),
            "dialogue": {"items": new_items},
        }
        out_data.append(out_rec)

    # 4) 写出
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n合并校验结果已保存: {args.output}")

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n================ 统计简报 ================")
    print(f"recording 数: {stats['total_recordings']}")
    print(f"items 合并前: {stats['items_before']} -> 合并后: {stats['items_after']}")
    print(f"  需合并的 step 组数: {stats['merged_groups']}（消去 item {stats['items_removed_by_merge']} 个）")
    print(f"  类型相对原标注发生变化的事件数: {stats['subtype_changed']}")
    if not args.dry_run:
        print(f"  LLM 失败回落 dry-run 的 step 数: {stats['fallback_used']}")
    print(f"合并后各 sub_type 分布: {stats['subtype_after']}")
    print(f"详细报告: {args.report}")
    print("=" * 42)


if __name__ == "__main__":
    main()
