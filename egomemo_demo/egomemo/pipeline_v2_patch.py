"""
将 StreamingPipeline 的推理方法替换为 v2 版本。

v2 修改点：
1. [respond] 时间戳必须是精确时间点，不允许 time range
2. 一次性问题：回复时间戳 = 问题提问时间，检索截止 = 问题提问时间
3. recurring 问题：回复时间戳 = 当前 caption 中的具体时间点，检索截止 = 当前 caption 起始时间
4. 主动服务同样使用精确时间点格式

用法：
    from egomemo.pipeline_v2_patch import patch_pipeline_v2
    patch_pipeline_v2()
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# 与 egolife_gpt_retrieval_.py 对齐的 prompt 拼装辅助函数
# ============================================================================

def _format_day_time_ranges_from_captions(captions: List[Dict]) -> str:
    """聚合 caption 列表中每天的视频起止时间，构造 ego 原版要求的
    "Video Time Ranges by Day" 段。

    captions: 每个元素含 "time_span" 形如 "1-00:00:00-00:00:10"。
    返回多行字符串，如:
        DAY1: 00:00:00 - 00:11:34
    """
    day_time_ranges = {}
    for cap in captions or []:
        ts = cap.get("time_span", "") if isinstance(cap, dict) else ""
        if not ts or "-" not in ts:
            continue
        # time_span 可能含尾缀 "_0" / "_1" 等
        ts_clean = ts.rsplit("_", 1)[0] if "_" in ts.split("-")[-1] else ts
        parts = ts_clean.split("-")
        if len(parts) < 3:
            continue
        day_str = parts[0]
        try:
            day = int(day_str[3:]) if day_str.startswith("DAY") else int(day_str)
        except (ValueError, IndexError):
            continue
        start_t = parts[1]
        end_t = parts[2] if len(parts) > 2 else parts[1]
        if day not in day_time_ranges:
            day_time_ranges[day] = {"start": start_t, "end": end_t}
        else:
            if start_t < day_time_ranges[day]["start"]:
                day_time_ranges[day]["start"] = start_t
            if end_t > day_time_ranges[day]["end"]:
                day_time_ranges[day]["end"] = end_t
    if not day_time_ranges:
        return ""
    lines = [
        f"DAY{day}: {day_time_ranges[day]['start']} - {day_time_ranges[day]['end']}"
        for day in sorted(day_time_ranges)
    ]
    return "Video Time Ranges by Day:\n" + "\n".join(lines)


def _build_egolife_pass1_prompt(
    base_system_prompt: str,
    current_caption_segment: str,
    current_caption_timespan: str,
    recent_minute_caption: Optional[str],
    recent_minute_timespan: Optional[str],
    proactive_history: Optional[List[Dict]],
    pre_retrieval_probe: Optional[Dict],
    day_time_ranges_block: str,
) -> str:
    """按 egolife_gpt_retrieval_._proactive_service_detection 的 8 步拼装方式
    构造 Pass 1 prompt。

    基本结构（与 ego 原版一致）：
        [base_system_prompt]
        [+ PREVIOUS RETRIEVAL REQUESTS（如有）]
        [+ === (1) Proactive Service History === <JSON dump>]
        [+\n (2) CURRENT_CAPTION:\n[time_span]: caption]
        [+ === (3) RECENT_5MIN_CAPTION === <如有 mid-level>]
        [+ Video Time Ranges by Day:\n...]
    """
    prompt = base_system_prompt

    # ---- (a) PREVIOUS RETRIEVAL REQUESTS（用于抑制重复检索）----
    if pre_retrieval_probe and isinstance(pre_retrieval_probe, dict):
        time_span_probe = pre_retrieval_probe.get("time_span", "")
        suspected = pre_retrieval_probe.get("suspected_service_type", "")
        retrieval_query = pre_retrieval_probe.get("retrieval_query", "")
        if suspected or retrieval_query:
            txt = "\n\nPREVIOUS RETRIEVAL REQUESTS:\n"
            if time_span_probe:
                txt += f"time_span: {time_span_probe}\n"
            if suspected:
                txt += f"suspected_service_type: {suspected}\n"
            if retrieval_query:
                txt += f"retrieval_query: {retrieval_query}\n"
            prompt += txt

    # ---- (b) (1) Proactive Service History（JSON dump）----
    if proactive_history:
        history_text = json.dumps(proactive_history, ensure_ascii=False, indent=2)
        prompt += "\n\n" + "=== (1) Proactive Service History ===\n" + history_text

    # ---- (c) (2) CURRENT_CAPTION ----
    prompt += (
        "\n (2) CURRENT_CAPTION:\n"
        + f"[{current_caption_timespan}]: {current_caption_segment}"
    )

    # ---- (d) (3) RECENT_5MIN_CAPTION（可选 mid-level）----
    if recent_minute_caption and recent_minute_timespan:
        prompt += (
            "\n=== (3) RECENT_5MIN_CAPTION ===\n"
            + f"[{recent_minute_timespan}]: {recent_minute_caption}"
        )

    # ---- (e) Video Time Ranges by Day ----
    if day_time_ranges_block:
        prompt += "\n" + day_time_ranges_block

    return prompt


def _build_egolife_pass2_prompt(
    mem_system_prompt: str,
    suspected_service_type: str,
    retrieval_query: str,
    retrieved_memory_text: str,
) -> str:
    """与 egolife_gpt_retrieval_ 的 Pass 2 调用对齐：
        mem_prompt + "\n\nRETRIEVED_MEMORY_EVIDENCE:\n\n" + retrieved_memory_json

    其中 retrieved_memory_json 是
        {"suspect_service_type", "retrieval_query", "retrieved_memory"} 的 JSON dump。
    """
    retrieved_memory_obj = json.dumps(
        {
            "suspect_service_type": suspected_service_type or "",
            "retrieval_query": retrieval_query or "",
            "retrieved_memory": retrieved_memory_text or "",
        },
        ensure_ascii=False,
    )
    return mem_system_prompt + "\n\nRETRIEVED_MEMORY_EVIDENCE:\n\n" + retrieved_memory_obj


def _parse_caption_start_time_from_span(time_span: str) -> float:
    """从 time_span 字符串（如 '1-00:00:14-00:00:16'）中解析出起始时间的秒数。"""
    # 格式: day-HH:MM:SS-HH:MM:SS
    parts = time_span.split('-', 1)
    if len(parts) < 2:
        return 0.0
    time_range = parts[1]  # HH:MM:SS-HH:MM:SS
    start_str = time_range.split('-')[0]  # HH:MM:SS
    h, m, s = start_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def _format_day_timestamp(day: int, seconds: float) -> str:
    """将秒数格式化为 DAY{N}-HH:MM:SS 格式。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"DAY{day}-{h:02d}:{m:02d}:{s:02d}"


def _format_caption_time_span_display(time_span: str) -> str:
    """将 time_span '1-00:00:14-00:00:16' 转为显示格式 'DAY1-00:00:14-00:00:16'。"""
    parts = time_span.split('-', 1)
    if len(parts) < 2:
        return time_span
    day = parts[0]
    return f"DAY{day}-{parts[1]}"


def _reason_for_question_v2(
    self, question, video_time: float, time_span: str,
    caption_text: str, frames_base64: List[str],
) -> Optional[Dict]:
    """v2 版本的 per-question reasoning。

    与原版区别：
    - 使用 v2 prompt templates（精确时间点约束）
    - 根据 question.recurring 区分 one-time / recurring
    - 一次性问题：检索截止 = ask_time，回复时间戳 = ask_time
    - recurring 问题：检索截止 = caption 起始时间，回复时间戳 = caption 中具体时间点
    """
    from .prompt_templates_v2 import (
        QA_SYSTEM_PROMPT_V2,
        PER_QUESTION_PROMPT_V2,
        PER_QUESTION_WITH_MEMORY_PROMPT_V2,
        make_timestamp_rule,
        make_retrieval_time_span,
    )
    from .memory_bridge import _make_time_span, _seconds_to_hhmmss
    from .streaming_pipeline import _parse_response_timestamp

    # 场景扩展：勾选后把鸡蛋羹引导规则追加到 QA system prompt 末尾
    qa_system_prompt = QA_SYSTEM_PROMPT_V2
    if bool(getattr(self.config, "egg_recipe_guidance_enabled", False)):
        from .prompt_templates_v2 import EGG_RECIPE_GUIDANCE_RULES
        qa_system_prompt = qa_system_prompt + EGG_RECIPE_GUIDANCE_RULES

    outputs = []
    actions_taken = []

    mem_result = self.question_queue.consume_pending_mem_read(question.qid)
    previous_answers_str = self.working_memory.format_previous_answers(question.qid, max_count=5)

    # 确定问题类型
    question_type = "recurring" if question.recurring else "one-time"

    # 解析 caption 的起始时间和显示格式
    caption_start_time = _parse_caption_start_time_from_span(time_span)
    caption_time_span_display = _format_caption_time_span_display(time_span)

    # 生成 ask timestamp 的 DAY 格式
    ask_timestamp_str = _format_day_timestamp(1, question.ask_time)

    # 生成时间戳约束说明
    response_timestamp_rule = make_timestamp_rule(
        question_type, ask_timestamp_str, caption_time_span_display,
    )

    if mem_result:
        prompt = PER_QUESTION_WITH_MEMORY_PROMPT_V2.format(
            system_prompt=qa_system_prompt,
            current_timestamp=f"{_seconds_to_hhmmss(video_time)} ({video_time:.1f}s)",
            caption_time_span=caption_time_span_display,
            current_caption=caption_text,
            retrieved_memory=mem_result,
            qid=question.qid,
            question_text=question.text,
            question_type=question_type,
            response_timestamp_rule=response_timestamp_rule,
        )
    else:
        prompt = PER_QUESTION_PROMPT_V2.format(
            system_prompt=qa_system_prompt,
            current_timestamp=f"{_seconds_to_hhmmss(video_time)} ({video_time:.1f}s)",
            caption_time_span=caption_time_span_display,
            current_caption=caption_text,
            recent_history=(
                self.memory.format_recent_history_before(time_span, self.config.recent_history_count)
                if hasattr(self.memory, 'format_recent_history_before')
                else self.memory.format_recent_history(self.config.recent_history_count)
            ),
            qid=question.qid,
            question_text=question.text,
            question_ask_time=f"{question.ask_time:.1f}",
            question_type=question_type,
            question_status=question.status.value,
            question_evidence="; ".join(question.evidence_notes[-3:]) if question.evidence_notes else "(none)",
            previous_answers=previous_answers_str,
            response_timestamp_rule=response_timestamp_rule,
        )

    # 广播: 推理开始
    if self.event_callback:
        self.event_callback("pipeline_stage", {
            "phase": "reasoning", "stage": "think_start",
            "qid": question.qid,
            "question_text": question.text,
            "time": video_time,
        })

    # Pass 1
    t_reason_start = time.time()
    response = self._call_reasoning_vlm(prompt, [])
    t_reason_end = time.time()
    self._timing_reasoning.append(t_reason_end - t_reason_start)
    logger.info(f"[Timing] Reasoning VLM (Pass 1): {t_reason_end-t_reason_start:.3f}s")

    # 广播: 思考完成，含模型原始输出预览
    if self.event_callback:
        self.event_callback("pipeline_stage", {
            "phase": "reasoning", "stage": "think_done",
            "qid": question.qid, "time": video_time,
            "raw_response": response[:300] if response else "",
        })

    actions = self.action_router.parse_llm_output(response)
    if not actions:
        return None

    action = actions[0]
    action_type = action.action.upper()

    if action_type == "SILENT":
        actions_taken.append(f"SILENT_{question.qid}")
        if self.event_callback:
            self.event_callback("pipeline_stage", {
                "phase": "reasoning", "stage": "silent",
                "qid": question.qid, "time": video_time,
            })

    elif action_type == "RESPOND":
        raw_answer = action.answer_text or ""
        ref_timestamp, answer_text = _parse_response_timestamp(raw_answer)
        # v2 硬约束：one-time 问题的回复时间戳必须等于提问时间，不信任模型输出
        if not question.recurring:
            logger.info(f"[v2-override] Pass1 one-time {question.qid}: model said {ref_timestamp!r}, forcing to {ask_timestamp_str}")
            ref_timestamp = ask_timestamp_str
        self.question_queue.record_answer(question.qid, answer_text, video_time)
        self.recorder.record_answer(question.qid, answer_text, video_time, action.evidence)
        self.working_memory.add_answer(question.qid, answer_text, video_time, action.evidence)
        # 构建推理轨迹
        reasoning_trace = [
            {"stage": "think", "content": f"Analyzing question [{question.qid}]: {question.text}"},
            {"stage": "respond", "content": answer_text},
        ]
        outputs.append({
            "type": "answer", "qid": question.qid,
            "answer": answer_text, "time": video_time,
            "ref_timestamp": ref_timestamp,
            "reasoning_trace": reasoning_trace,
        })
        actions_taken.append(f"RESPOND_{question.qid}")
        logger.info(f"[RESPOND_{question.qid}] {f'@{ref_timestamp} ' if ref_timestamp else ''}{answer_text[:100]}...")

        # 立即广播回答（让前端同步显示气泡+TTS，不等 step_complete）
        if self.event_callback:
            self.event_callback("answer_ready", {
                "qid": question.qid,
                "answer": answer_text,
                "time": video_time,
                "ref_timestamp": ref_timestamp,
            })

    elif action_type == "SEARCH":
        query = action.search_query or ""
        if query:
            # 广播: 检索开始
            if self.event_callback:
                self.event_callback("pipeline_stage", {
                    "phase": "reasoning", "stage": "search_start",
                    "qid": question.qid, "time": video_time,
                    "search_query": query,
                })

            # v2: 根据问题类型决定检索截止时间
            retrieval_time_span = make_retrieval_time_span(
                question_type=question_type,
                ask_time=question.ask_time,
                caption_start_time=caption_start_time,
                make_time_span_func=_make_time_span,
                day=1,
            )
            t_retrieval_start = time.time()
            retrieved = self.memory.read(query, retrieval_time_span)
            t_retrieval_end = time.time()
            self._timing_retrieval.append(t_retrieval_end - t_retrieval_start)
            logger.info(f"[Timing] Memory retrieval: {t_retrieval_end-t_retrieval_start:.3f}s")
            self.working_memory.add_mem_read(question.qid, query, retrieved, video_time)
            actions_taken.append(f"SEARCH_{question.qid}")
            logger.info(f"[SEARCH_{question.qid}] query: {query[:80]}... | time_span: {retrieval_time_span}")

            # 广播: 检索结果
            if self.event_callback:
                self.event_callback("pipeline_stage", {
                    "phase": "reasoning", "stage": "observation",
                    "qid": question.qid, "time": video_time,
                    "retrieved_preview": retrieved[:300],
                })

            # Pass 2
            history = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": response}]},
            ]

            pass2_prompt = (
                f"[observation]\n{retrieved}\n[/observation]\n\n"
                f"Based on the retrieved information above, answer the question [{question.qid}] now.\n"
                f"IMPORTANT: {response_timestamp_rule}\n"
                f"[respond] "
            )
            t_reason2_start = time.time()
            response2 = self._call_reasoning_vlm(pass2_prompt, [], history_messages=history)
            t_reason2_end = time.time()
            self._timing_reasoning.append(t_reason2_end - t_reason2_start)
            logger.info(f"[Timing] Reasoning VLM (Pass 2): {t_reason2_end-t_reason2_start:.3f}s")

            raw_answer2 = response2 or ""
            respond_match = re.search(r"\[respond\]\s*", raw_answer2, re.IGNORECASE)
            if respond_match:
                raw_answer2 = raw_answer2[respond_match.end():].strip()
            ref_timestamp, answer_text = _parse_response_timestamp(raw_answer2)
            # v2 硬约束：one-time 问题的回复时间戳必须等于提问时间，不信任模型输出
            if not question.recurring:
                logger.info(f"[v2-override] Pass2 one-time {question.qid}: model said {ref_timestamp!r}, forcing to {ask_timestamp_str}")
                ref_timestamp = ask_timestamp_str

            self.question_queue.record_answer(question.qid, answer_text, video_time)
            self.recorder.record_answer(question.qid, answer_text, video_time)
            self.working_memory.add_answer(question.qid, answer_text, video_time)
            # 构建推理轨迹（含检索）
            retrieved_preview = retrieved[:300] + "..." if len(retrieved) > 300 else retrieved
            reasoning_trace = [
                {"stage": "think", "content": f"Analyzing question [{question.qid}]: {question.text}"},
                {"stage": "search", "content": query},
                {"stage": "observation", "content": retrieved_preview},
                {"stage": "respond", "content": answer_text},
            ]
            outputs.append({
                "type": "answer", "qid": question.qid,
                "answer": answer_text, "time": video_time,
                "ref_timestamp": ref_timestamp,
                "reasoning_trace": reasoning_trace,
            })
            actions_taken.append(f"RESPOND_{question.qid}")
            logger.info(f"[RESPOND_{question.qid} after SEARCH] {f'@{ref_timestamp} ' if ref_timestamp else ''}{answer_text[:100]}...")

            # 立即广播回答
            if self.event_callback:
                self.event_callback("answer_ready", {
                    "qid": question.qid,
                    "answer": answer_text,
                    "time": video_time,
                    "ref_timestamp": ref_timestamp,
                })

    return {"outputs": outputs, "actions_taken": actions_taken}


def _emit_demo_forced_reminder(
    self, video_time: float, time_span: str, kind: str,
) -> Optional[Dict]:
    """Demo 模式下强制输出固定提醒，并真实走一次 memory 检索让轨迹完整。

    Args:
        kind: "hydration"（饮水） 或 "work_unfinished"（工作未完成）
              或 "power_bank"（充电宝）。
              最终 respond 文本固定，不受检索结果影响；但检索仍会执行，
              observation 轨迹用真实检索结果填充。
    """
    import uuid
    from .prompt_templates_demo import (
        LAST_CHUNK_HYDRATION_REMINDER,
        WORK_UNFINISHED_REMINDER,
        SHOPPING_MILK_REMINDER,
        CHARGER_CLUTTER_REMINDER,
        DIVIDED_ATTENTION_REMINDER,
        HYDRATION_SEARCH_QUERY,
        WORK_UNFINISHED_SEARCH_QUERY,
        POWER_BANK_SEARCH_QUERY,
        SHOPPING_MILK_SEARCH_QUERY,
        CHARGER_CLUTTER_SEARCH_QUERY,
        DIVIDED_ATTENTION_SEARCH_QUERY,
    )

    # ---- 按 kind 选 query + 最终 respond 文本 + think 开头 ----
    if kind == "hydration":
        query = HYDRATION_SEARCH_QUERY
        content = LAST_CHUNK_HYDRATION_REMINDER
        think_text = "Long-interval check: has the user had water recently?"
    elif kind in ("work_unfinished", "phone_sustained"):
        # phone_sustained 是 LLM 判定版（caption 显示 ≥5s 持续用手机）；
        # work_unfinished 是旧的关键词硬触发版。两者 UI 呈现一致，都用
        # "工作还没完成，回去干活" 固定文案。
        query = WORK_UNFINISHED_SEARCH_QUERY
        content = WORK_UNFINISHED_REMINDER
        think_text = "User appears distracted by phone — what were they working on earlier?"
    elif kind == "power_bank":
        query = POWER_BANK_SEARCH_QUERY
        content = "The power bank is in the bedroom ahead."
        think_text = "User is searching for a power bank — recall where it was last seen."
    elif kind == "shopping_milk":
        query = SHOPPING_MILK_SEARCH_QUERY
        content = SHOPPING_MILK_REMINDER
        think_text = "User is in a supermarket picking drinks — was there a shopping plan discussed earlier?"
    elif kind == "charger_clutter":
        query = CHARGER_CLUTTER_SEARCH_QUERY
        content = CHARGER_CLUTTER_REMINDER
        think_text = "Chargers and devices look disorganized — compare with earlier tidy state."
    elif kind == "divided_attention":
        query = DIVIDED_ATTENTION_SEARCH_QUERY
        content = DIVIDED_ATTENTION_REMINDER
        think_text = "User is on phone while navigating — check recent similar split-attention incidents."
    else:
        logger.warning(f"[DEMO] unknown forced reminder kind: {kind}")
        return None

    # ---- 生成 event_id，并逐阶段广播 pipeline_stage 让前端实时渲染 ----
    event_id = f"proactive_{uuid.uuid4().hex[:8]}"
    stage_qid = event_id  # pipeline_stage 用 event_id 作为 block id（以 "proactive" 开头即被认作 proactive）

    # think_start：前端据此新建 reasoning block
    if self.event_callback:
        self.event_callback("pipeline_stage", {
            "phase": "reasoning", "stage": "think_start",
            "qid": stage_qid, "time": video_time,
        })
        self.event_callback("pipeline_stage", {
            "phase": "reasoning", "stage": "think_done",
            "qid": stage_qid, "time": video_time,
            "raw_response": think_text,
        })
        # search_start：显示检索 query
        self.event_callback("pipeline_stage", {
            "phase": "reasoning", "stage": "search_start",
            "qid": stage_qid, "time": video_time,
            "search_query": query,
        })

    # ---- 真实执行一次 memory 检索（结果不影响 respond 内容） ----
    retrieved_preview = ""
    try:
        t_ret_start = time.time()
        retrieved = self.memory.read(query, time_span) or ""
        t_ret_end = time.time()
        self._timing_retrieval.append(t_ret_end - t_ret_start)
        logger.info(
            f"[DEMO PROACTIVE / {kind}] retrieval {t_ret_end - t_ret_start:.3f}s, "
            f"got {len(retrieved)} chars"
        )
        retrieved_preview = (retrieved[:300] + "...") if len(retrieved) > 300 else retrieved
    except Exception as e:
        logger.warning(f"[DEMO PROACTIVE / {kind}] retrieval failed: {e}")
        retrieved_preview = "(retrieval unavailable)"

    if not retrieved_preview:
        retrieved_preview = "(no related memory found — issuing reminder anyway)"

    # 只标记 search 完成，不把检索结果打到面板上（避免暴露"剧本"）
    if self.event_callback:
        self.event_callback("pipeline_stage", {
            "phase": "reasoning", "stage": "search_complete_hidden",
            "qid": stage_qid, "time": video_time,
        })

    # ---- 构造 proactive 输出 ----
    # 同时更新全局（兼容）和按 kind 分桶的冷却时间
    self._last_proactive_time = video_time
    if hasattr(self, "_last_proactive_time_by_kind"):
        self._last_proactive_time_by_kind[kind] = video_time
    self._proactive_history.append({
        "time": video_time, "event_id": event_id, "content": content,
    })
    ref_timestamp = _format_day_timestamp(1, video_time)
    self.recorder.record_proactive(content, video_time, "", event_id=event_id)
    self.working_memory.add_proactive(event_id, content, video_time, "")

    proactive_trace = [
        {"stage": "think", "content": think_text},
        {"stage": "search", "content": query},
        {"stage": "observation", "content": retrieved_preview},
        {"stage": "respond", "content": content},
    ]
    outputs = [{
        "type": "proactive", "event_id": event_id,
        "content": content, "time": video_time,
        "ref_timestamp": ref_timestamp,
        "reasoning_trace": proactive_trace,
    }]
    actions_taken = ["PROACTIVE_SEARCH", "PROACTIVE"]
    logger.info(f"[DEMO PROACTIVE / {kind}] emit: {content}")

    if self.event_callback:
        self.event_callback("answer_ready", {
            "qid": event_id,
            "answer": content,
            "time": video_time,
            "is_proactive": True,
            "ref_timestamp": ref_timestamp,
        })

    return {"outputs": outputs, "actions_taken": actions_taken}


def _reason_proactive_v2(
    self, video_time: float, time_span: str,
    caption_text: str, frames_base64: List[str],
) -> Optional[Dict]:
    """v2 版本的主动服务推理。

    与原版区别：
      - 使用 v2 prompt，要求精确时间点。
      - datasets_type == "egolife" 时切换到 demo prompt（展示长期检索 + 主动
        提醒的产品愿景），并在最后一个 chunk 强制输出饮水提醒。
    """
    import uuid
    from .memory_bridge import _seconds_to_hhmmss
    from .streaming_pipeline import _parse_response_timestamp

    # --- demo 模式判定 ---
    #  is_demo = (getattr(self.config, "datasets_type", "") or "").lower() == "egolife"
    is_demo = False

    # 按 kind 检查冷却：只抑制同类型的重复提醒，不同类型互不干扰。
    def _cooldown_ok_for(kind: str) -> bool:
        last = self._last_proactive_time_by_kind.get(kind, -1e9) \
            if hasattr(self, "_last_proactive_time_by_kind") else -1e9
        return (video_time - last) >= self.config.proactive_cooldown_seconds

    # --- demo 模式下的强制剧本分支（基于关键词硬触发，只处理无歧义的 demo 剧本）---
    # 注意：单纯"玩手机"/ divided-attention / 切菜 / stove / 久坐 等所有需要
    # 语义判断的场景，统统交给 LLM（v2 prompt 自行判断），这里不做关键词硬触发。
    if is_demo:
        from .prompt_templates_demo import (
            caption_mentions_supermarket_shopping,
            caption_mentions_charger_clutter,
        )

        # 1) 第一个 chunk：强制饮水提醒（仅前端勾选 Hydrate 时触发）
        current_step = getattr(self, "_step_count", 0)
        is_first_chunk = current_step <= 1
        hydration_on = bool(getattr(self.config, "hydration_reminder_enabled", False))
        if is_first_chunk and hydration_on and _cooldown_ok_for("hydration"):
            return _emit_demo_forced_reminder(self, video_time, time_span, "hydration")

        # 2) 超市买饮料 → 买牛奶提醒
        if caption_mentions_supermarket_shopping(caption_text):
            if _cooldown_ok_for("shopping_milk"):
                return _emit_demo_forced_reminder(self, video_time, time_span, "shopping_milk")
            else:
                logger.info("[DEMO PROACTIVE / shopping_milk] suppressed by same-kind cooldown")

        # 3) 充电设备杂乱 → 整理清单提醒
        if caption_mentions_charger_clutter(caption_text):
            if _cooldown_ok_for("charger_clutter"):
                return _emit_demo_forced_reminder(self, video_time, time_span, "charger_clutter")
            else:
                logger.info("[DEMO PROACTIVE / charger_clutter] suppressed by same-kind cooldown")

        # 其余所有场景（手机、divided-attention、切菜、stove、久坐等）
        # 都落到下面的 LLM 正式 proactive 逻辑，由 v2 prompt 自行判断。

    if is_demo:
        from .prompt_templates_demo import (
            PROACTIVE_SYSTEM_PROMPT_DEMO as _PRO_SYS,
            PROACTIVE_PROMPT_DEMO as _PRO_TPL,
            PROACTIVE_WITH_MEMORY_PROMPT_DEMO as _PRO_TPL_MEM,  # noqa: F401 (保留导出以防未来扩展)
        )
    else:
        from .prompt_templates_v2 import (
            PROACTIVE_SYSTEM_PROMPT_V2 as _PRO_SYS,
            PROACTIVE_PROMPT_V2 as _PRO_TPL,
            PROACTIVE_WITH_MEMORY_PROMPT_V2 as _PRO_TPL_MEM,  # noqa: F401
        )

    # 场景扩展规则：前端勾选按钮时把对应规则追加到 system prompt 末尾
    if bool(getattr(self.config, "circuit_breaker_scene_enabled", False)):
        from .prompt_templates_v2 import CIRCUIT_BREAKER_EXTRA_RULES
        _PRO_SYS = _PRO_SYS + CIRCUIT_BREAKER_EXTRA_RULES

    # 注意：这里不做全局冷却门。冷却按 kind 在各自触发点检查——
    #   - demo 剧本 kind 在 _emit_demo_forced_reminder 前检查
    #   - LLM 自由 respond 在最终 RESPOND 处理前用 "llm" 桶检查
    caption_time_span_display = _format_caption_time_span_display(time_span)

    # ===== 与 egolife_gpt_retrieval_._proactive_service_detection 对齐的 Pass1 拼装 =====
    # 对齐项：
    #   (1) === (1) Proactive Service History === 段（JSON dump，最近 5 条）
    #   (2)  (2) CURRENT_CAPTION 段（[time_span]: text 形式）
    #   (3) === (3) RECENT_5MIN_CAPTION === 段（如启用 multi_level 且有 minute caption）
    #   (4) Video Time Ranges by Day 段（聚合自 self.memory._captions）
    #   (5) PREVIOUS RETRIEVAL REQUESTS 段（来自 self._last_proactive_search）
    # OUTPUT FORMAT OVERRIDE 仍由 _PRO_SYS 末尾的覆盖说明负责，输出仍是
    # [silent]/[search]/[respond]，以兼容 action_router。

    # 历史 JSON dump（最近 5 条，含尽量多的字段）
    history_for_prompt: List[Dict] = []
    for r in (self._proactive_history or [])[-5:]:
        history_for_prompt.append({
            "time_span": _format_day_timestamp(1, r.get("time", 0.0)),
            "content": r.get("content", ""),
            "event_id": r.get("event_id", ""),
        })

    # Day time ranges：从 memory bridge 已有的 second-level captions 聚合
    captions_for_ranges = []
    try:
        captions_for_ranges = list(getattr(self.memory, "_captions", []) or [])
    except Exception:
        captions_for_ranges = []
    day_time_ranges_block = _format_day_time_ranges_from_captions(captions_for_ranges)

    # mid-level (RECENT_5MIN_CAPTION)：仅在启用 multi-level caption 时填
    recent_min_caption = None
    recent_min_timespan = None
    if getattr(self.config, "enable_multi_level_caption", False):
        try:
            min_caps = getattr(self.memory, "_min_captions", []) or []
            if min_caps:
                last_min = min_caps[-1]
                recent_min_caption = last_min.get("caption")
                recent_min_timespan = last_min.get("time_span")
        except Exception:
            pass

    # 上次检索探针（用于抑制重复检索 — Gate 1）
    pre_retrieval_probe = getattr(self, "_last_proactive_search", None)

    prompt = _build_egolife_pass1_prompt(
        base_system_prompt=_PRO_SYS,
        current_caption_segment=caption_text,
        current_caption_timespan=caption_time_span_display,
        recent_minute_caption=recent_min_caption,
        recent_minute_timespan=recent_min_timespan,
        proactive_history=history_for_prompt,
        pre_retrieval_probe=pre_retrieval_probe,
        day_time_ranges_block=day_time_ranges_block,
    )

    # Pass 1
    t_pro_start = time.time()
    response = self._call_reasoning_vlm(prompt, [])
    t_pro_end = time.time()
    self._timing_reasoning.append(t_pro_end - t_pro_start)
    logger.info(f"[Timing] Proactive VLM (Pass 1): {t_pro_end-t_pro_start:.3f}s")

    actions = self.action_router.parse_llm_output(response)
    if not actions:
        return None

    action = actions[0]
    action_type = action.action.upper()
    outputs = []
    actions_taken = []

    if action_type == "SILENT":
        return {"outputs": outputs, "actions_taken": actions_taken}

    if action_type == "SEARCH":
        query = action.search_query or ""

        # === DEMO 模式劫持：LLM 用 [search] 表达"我检测到持续玩手机 /
        # 玩手机+推车 等场景"，我们不让它自由组织回答，直接用固定文案替换。
        # 冷却按 kind 独立判断，不同 kind 互不压制。
        if is_demo and query:
            from .prompt_templates_demo import (
                llm_query_is_sustained_phone,
                llm_query_is_divided_attention,
            )

            def _emit_if_cooled(kind_to_fire: str):
                last = self._last_proactive_time_by_kind.get(kind_to_fire, -1e9) \
                    if hasattr(self, "_last_proactive_time_by_kind") else -1e9
                if (video_time - last) < self.config.proactive_cooldown_seconds:
                    logger.info(f"[DEMO HIJACK / {kind_to_fire}] suppressed by same-kind cooldown")
                    return None
                return _emit_demo_forced_reminder(self, video_time, time_span, kind_to_fire)

            # divided-attention 更具体，优先级比 phone-sustained 高
            if llm_query_is_divided_attention(query):
                logger.info(f"[DEMO HIJACK / divided_attention] LLM query='{query[:80]}'")
                result = _emit_if_cooled("divided_attention")
                if result is not None:
                    return result
                # 冷却中被抑制就直接 silent（本 chunk 不发任何 proactive）
                return {"outputs": [], "actions_taken": []}
            if llm_query_is_sustained_phone(query):
                logger.info(f"[DEMO HIJACK / phone_sustained] LLM query='{query[:80]}'")
                result = _emit_if_cooled("phone_sustained")
                if result is not None:
                    return result
                return {"outputs": [], "actions_taken": []}

        if query:
            # 记录探针（pre_retrieval_probe）— 下一个 chunk 的 Pass1 prompt 会读
            # 它，触发 ego prompt 的 Gate 1 cooldown 抑制重复检索
            self._last_proactive_search = {
                "time_span": caption_time_span_display,
                "suspected_service_type": "",  # OUTPUT OVERRIDE 下没有 service_sub_type
                "retrieval_query": query,
            }

            t_ret_start = time.time()
            retrieved = self.memory.read(query, time_span)
            t_ret_end = time.time()
            self._timing_retrieval.append(t_ret_end - t_ret_start)
            logger.info(f"[Timing] Proactive retrieval: {t_ret_end-t_ret_start:.3f}s")

            actions_taken.append("PROACTIVE_SEARCH")
            logger.info(f"[PROACTIVE_SEARCH] query: {query[:80]}...")

            # ===== 与 egolife_gpt_retrieval_ Pass 2 调用对齐 =====
            # 原版：mem_prompt + "\n\nRETRIEVED_MEMORY_EVIDENCE:\n\n" + json_dump
            # history_messages 装上 (first_user_prompt, first_assistant_resp)
            from .prompt_templates_ego import PROACTIVE_WITH_MEMORY_SYSTEM_PROMPT_EGO

            history = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": response}]},
            ]
            pass2_prompt = _build_egolife_pass2_prompt(
                mem_system_prompt=PROACTIVE_WITH_MEMORY_SYSTEM_PROMPT_EGO,
                suspected_service_type="",
                retrieval_query=query,
                retrieved_memory_text=retrieved,
            )
            t_pro2_start = time.time()
            response2 = self._call_reasoning_vlm(pass2_prompt, [], history_messages=history)
            t_pro2_end = time.time()
            self._timing_reasoning.append(t_pro2_end - t_pro2_start)
            logger.info(f"[Timing] Proactive VLM (Pass 2): {t_pro2_end-t_pro2_start:.3f}s")

            raw_content = response2 or ""
            respond_match = re.search(r"\[respond\]\s*", raw_content, re.IGNORECASE)
            if respond_match:
                raw_content = raw_content[respond_match.end():].strip()
            ref_timestamp, content = _parse_response_timestamp(raw_content)
            # 修复：Pass2 检索后模型可能改判为 [silent]（觉得无需提醒）。
            # 原逻辑无条件强制 action_type="RESPOND"，导致把 "[silent]" 字面量
            # 当成回复内容广播给前端（前端显示 RESPOND -> [silent]）。
            # 这里先判断 Pass2 是否其实选择了 silent：剥掉 [respond] 后仍含 silent 标记、
            # 或内容为空，都视为 silent，直接静默返回。
            pass2_action = self.action_router.parse_llm_output(response2 or "")[0].action.upper()
            content_is_silent = (
                pass2_action == "SILENT"
                or re.search(r"\[silent\]|<\|silent\|>", content, re.IGNORECASE) is not None
                or not content.strip()
            )
            if content_is_silent:
                logger.info("[PROACTIVE Pass2] model chose silent after retrieval, no reminder emitted")
                return {"outputs": outputs, "actions_taken": actions_taken}
            action_type = "RESPOND"
        else:
            return {"outputs": outputs, "actions_taken": actions_taken}
    else:
        # Direct RESPOND
        raw_content = action.answer_text or ""
        ref_timestamp, content = _parse_response_timestamp(raw_content)

    # 兜底：content 若实际是 silent 标记（[silent]/<|silent|>），不应作为提醒广播
    if content and re.search(r"\[silent\]|<\|silent\|>", content, re.IGNORECASE):
        logger.info("[PROACTIVE] content resolved to silent marker, no reminder emitted")
        return {"outputs": outputs, "actions_taken": actions_taken}

    if action_type == "RESPOND" and content:
        # LLM 自由 respond 走 "llm" 桶冷却
        _last_llm = (
            self._last_proactive_time_by_kind.get("llm", -1e9)
            if hasattr(self, "_last_proactive_time_by_kind") else -1e9
        )
        if (video_time - _last_llm) < self.config.proactive_cooldown_seconds:
            logger.info("[LLM PROACTIVE] suppressed by llm-bucket cooldown")
            return {"outputs": outputs, "actions_taken": actions_taken}

        event_id = f"proactive_{uuid.uuid4().hex[:8]}"
        self._last_proactive_time = video_time
        if hasattr(self, "_last_proactive_time_by_kind"):
            # LLM 判定的 proactive 归入 "llm" 桶
            self._last_proactive_time_by_kind["llm"] = video_time
        self._proactive_history.append({
            "time": video_time, "event_id": event_id, "content": content,
        })
        self.recorder.record_proactive(content, video_time, action.evidence, event_id=event_id)
        self.working_memory.add_proactive(event_id, content, video_time, action.evidence)
        # 构建推理轨迹
        proactive_trace = [{"stage": "think", "content": "Monitoring scene for safety/task issues..."}]
        if 'query' in dir() and query:
            retrieved_preview = retrieved[:300] + "..." if len(retrieved) > 300 else retrieved
            proactive_trace.append({"stage": "search", "content": query})
            proactive_trace.append({"stage": "observation", "content": retrieved_preview})
        proactive_trace.append({"stage": "respond", "content": content})
        outputs.append({
            "type": "proactive", "event_id": event_id,
            "content": content, "time": video_time,
            "ref_timestamp": ref_timestamp,
            "reasoning_trace": proactive_trace,
        })
        actions_taken.append("PROACTIVE")
        logger.info(f"[PROACTIVE] {f'@{ref_timestamp} ' if ref_timestamp else ''}{content[:100]}...")

        # 立即广播主动提醒
        if self.event_callback:
            self.event_callback("answer_ready", {
                "qid": event_id,
                "answer": content,
                "time": video_time,
                "is_proactive": True,
                "ref_timestamp": ref_timestamp,
            })

    return {"outputs": outputs, "actions_taken": actions_taken}


def patch_pipeline_v2():
    """将 StreamingPipeline 的推理方法替换为 v2 版本。

    修改内容：
    - _reason_for_question → _reason_for_question_v2（精确时间戳 + 检索截止时间修正）
    - _reason_proactive → _reason_proactive_v2（精确时间戳，并支持 egolife demo 模式）
    - _run_async_mode / _run_sequential_mode：在进入 loop 前记录 self._total_steps，
      供 demo 模式判断"是否是最后一个 chunk"使用。
    """
    from .streaming_pipeline import StreamingPipeline

    StreamingPipeline._reason_for_question = _reason_for_question_v2
    StreamingPipeline._reason_proactive = _reason_proactive_v2

    # wrap _run_async_mode / _run_sequential_mode 以记录总 step 数
    _orig_async = StreamingPipeline._run_async_mode
    _orig_seq = StreamingPipeline._run_sequential_mode

    def _run_async_mode_with_total(self, step_batches):
        self._total_steps = len(step_batches)
        return _orig_async(self, step_batches)

    def _run_sequential_mode_with_total(self, step_batches):
        self._total_steps = len(step_batches)
        return _orig_seq(self, step_batches)

    StreamingPipeline._run_async_mode = _run_async_mode_with_total
    StreamingPipeline._run_sequential_mode = _run_sequential_mode_with_total

    logger.info("[v2] StreamingPipeline patched with v2 reasoning (precise timestamps)")
