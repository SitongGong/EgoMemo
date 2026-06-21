"""精简版 prompt 包装层：从 VideoRAG 的 ego_prompt_.py 加载 prompt。

设计原则：
- 不做内容重写，只做 **接口适配**：让 ego_prompt_ 中的 prompt 能直接被
  egomemo 现有 pipeline 调用（占位符兼容、输出格式兼容）。
- 仅保留 rebuttal 测试需要的部分：
    * second-level caption 系统提示
    * 实体提取（建图）模板
    * proactive 推理 + post-retrieval 决定
- 保持 [silent]/[search]/[respond] 输出格式以兼容 action_router；
  ego_prompt_ 原版用 JSON 输出，这里在系统提示末尾追加格式覆盖说明。
- 不再保留 mid/hour-level caption（项目中默认未启用），
  也不再保留 QA 问答推理路径（rebuttal 仅需主动服务）。
"""

import os
import sys
from typing import Dict

# 加 VideoRAG 到路径，以加载 ego_prompt_
# 开源布局下 videorag 与 egomemo 平级，默认指向上级目录；可用 VIDEORAG_ROOT 覆盖
_VIDEORAG_ROOT = os.environ.get("VIDEORAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _VIDEORAG_ROOT not in sys.path:
    sys.path.append(_VIDEORAG_ROOT)

from videorag.ego_prompt_ import PROMPTS as _EGO_PROMPTS  # type: ignore  # noqa: E402


# ==================================================================
# 1) Second-level caption 系统提示（30s window，逐帧 + 全局 caption）
# ==================================================================
# 注意：ego_prompt_ 中 simple_second_caption_system_prompt 要求模型输出
# {"caption": "...", "frames": {"0": "...", ...}} —— 这与 egomemo 的
# memory_bridge._build_caption_dict 解析逻辑兼容（它能处理 frames+caption 两种 key）。
CAPTION_SYSTEM_PROMPT_EGO: str = _EGO_PROMPTS["simple_second_caption_system_prompt"]


# ==================================================================
# 2) 实体提取（建图阶段）
# ==================================================================
# ego_prompt_ 中的 entity_extraction 包含 {entity_types}/{tuple_delimiter}/
# {record_delimiter}/{completion_delimiter}/{input_text} 占位符；
# 由 VideoRAG 的 ego_op.batch_extract_entities 在调用时填充——
# 我们这里直接透传原文。
ENTITY_EXTRACTION_PROMPT_EGO: str = _EGO_PROMPTS["entity_extraction"]


# ==================================================================
# 工具：从 ego_prompt_["proactive_service_prompt_test"] 中按"段标题"裁剪。
# 设计目标：把 22k chars 的 system prompt 砍到 ~10k，同时保留
#   - taxonomy（决定服务质量）
#   - Inputs / Core Principles（输入约定）
#   - Step 0 Mandatory Screening（episodic/long-term 触发条件）
# 砍掉的部分（已被 OUTPUT FORMAT OVERRIDE 等价覆盖或与之冲突）：
#   - Retrieval Gating Rules（OVERRIDE 里"Do NOT request retrieval if..."已覆盖）
#   - Memory Retrieval Request Format（temporal hint 强制 → 已被 OVERRIDE 反向覆盖）
#   - In-Context Query Examples（与 OVERRIDE 的"不要 temporal hint"矛盾）
#   - FINAL OUTPUT FORMAT（JSON spec → 已被 OVERRIDE 替换为 [silent]/[search]/[respond]）
#   - Suppression Rule / CRITICAL TIME WINDOW（cooldown 已 Python 侧实现，时间约束 OVERRIDE 已说）
# 不动 ego_prompt_.py 原文件，所有裁剪在这里完成。
# ==================================================================
def _trim_ego_proactive_prompt(raw: str, drop_section_titles: list) -> str:
    """从 ego_prompt_ 的 proactive_service_prompt_test 原文中删除若干 section。

    section 由 "------\nTitle\n------" 边界标识。删除时把整段（含上方分隔
    符 + 标题 + 内容 + 下方分隔符）一起去掉，避免遗留孤立的 "----" 横线。
    Title 必须精确匹配（大小写敏感）。
    """
    import re
    out = raw
    for title in drop_section_titles:
        # 匹配：上方 ---- + 标题 + 下方 ---- + 内容（直到下一个 ---- 块或 EOF）
        # 注意 ego_prompt_ 用的是 "-" * N（N 通常是 60+），用 -{4,} 兼容
        title_re = re.escape(title)
        # 段块 = "----\n<title>\n----\n<content>" 直到下一个 "----\n<???>\n----" 或文件尾
        pattern = (
            r"-{4,}\n"          # 上分隔符
            r"\s*" + title_re + r"\s*\n"  # 标题
            r"-{4,}\n"          # 下分隔符
            r".*?"              # 内容
            r"(?=-{4,}\n[^\n]+\n-{4,}|\Z)"  # 直到下一个段标题块或 EOF
        )
        out_new = re.sub(pattern, "", out, count=1, flags=re.DOTALL)
        if out_new == out:
            import logging
            logging.getLogger(__name__).warning(
                f"[prompt_templates_ego] section not found, no-op: '{title}'"
            )
        out = out_new
    # 清理连续多余空行
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# 要砍掉的 section 标题（精确匹配 ego_prompt_ 中的标题文本）。
# 注意：FINAL OUTPUT FORMAT 段下面还有两个用 "----" 包围的 JSON 子节标题
# （"finalized_services" / "episodic_longterm_probe"），裁剪函数会逐个删。
_PASS1_DROP_SECTIONS = [
    "Suppression Rule",
    "Retrieval Gating Rules (SIMPLIFIED, HISTORY-AWARE)",
    "CRITICAL TIME WINDOW RULE",
    "Memory Retrieval Request Format (QUERY-ONLY, TIME HINT REQUIRED)",
    "In-Context Query Examples (retrieval_query with temporal hint)",
    "FINAL OUTPUT FORMAT (UNAMBIGUOUS, SINGLE-SPEC)",
    '"finalized_services" (NON-EMPTY only if services are finalized now)',
    '"episodic_longterm_probe" (NON-EMPTY only if retrieval is needed)',
    "Final Instruction",
]


# ==================================================================
# 3) Proactive 推理（Pass 1：观察 → [silent]/[search]/[respond]）
# ==================================================================
# 改造点：
# - 原 ego prompt 输出 JSON（finalized_services + episodic_longterm_probe），
#   在系统提示末尾加一段 OUTPUT FORMAT OVERRIDE，强制改成 [silent]/[search]/[respond]。
# - 时间戳规则与 egomemo v2 一致：单点 DAY{N}-HH:MM:SS。
_OUTPUT_FORMAT_OVERRIDE_PASS1 = """

============================================================
OUTPUT FORMAT OVERRIDE (MANDATORY — supersedes any earlier
JSON / finalized_services / episodic_longterm_probe spec)
============================================================

You must emit EXACTLY ONE bracket-tagged action and nothing else.
Pick one of the three actions:

- [silent]
    Use when the current caption shows no immediate hazard, no clear
    short-term issue, and no plausible cue for an episodic / long-term
    probe. This is the default when in doubt.

- [search] <one-sentence retrieval query>
    Use when you suspect an Episodic or Long-Term proactive service
    (e.g. forgotten task, prolonged behavior pattern, unresolved
    earlier state) and you need older evidence to confirm.
    The query must be a single concise English sentence describing
    WHAT evidence to retrieve (entity + action / state), e.g.:
        "the most recent scene where I drank water from a cup or bottle"
        "where I last placed my phone after using it"
        "my recent repeated behavior of checking my phone"
    Do NOT request retrieval if the same suspected service was probed
    very recently (cooldown applies).
    IMPORTANT — supersedes any earlier "temporal hint MANDATORY" rule
    or "Within the past N hours/days" example: DO NOT prefix the query
    with a temporal hint like "Within the past 2 hours" / "earlier
    today" / "in the past 1 day". Just describe the semantic target;
    the retrieval system handles temporal scoping internally.

- [respond] DAY{N}-HH:MM:SS <reminder>
    Use when you can finalize an Instant or Short-Term proactive service
    based on the current caption alone. The timestamp MUST be a single
    precise time point picked from the current caption window (NOT a
    range, NOT the full 30s window, NOT a memory timestamp).
    The reminder must be 1-2 short sentences spoken DIRECTLY to the
    user (e.g. "Careful, the stove is still on.").

Hard rules:
- Output exactly ONE of the three actions, nothing before or after.
- Do NOT output JSON, do NOT output multiple actions, do NOT output
  service_main_type / service_sub_type / finalized_services keys.
- Prefer [silent] when evidence is weak or ambiguous.
"""

PROACTIVE_SYSTEM_PROMPT_EGO: str = (
    _trim_ego_proactive_prompt(
        _EGO_PROMPTS["proactive_service_prompt_test"],
        _PASS1_DROP_SECTIONS,
    ).rstrip()
    + _OUTPUT_FORMAT_OVERRIDE_PASS1
)


# ==================================================================
# 4) Proactive 推理（Pass 2：拿到检索结果后，做最终决定）
# ==================================================================
# ego_prompt_ 的 proactive_service_prompt_with_memory_simple 原本输出
# {"decision": "suppressed", ...} 或 [{"service_sub_type": ..., ...}]；
# 这里同样覆盖为 [silent]/[respond]（Pass 2 不会再触发 [search]）。
_OUTPUT_FORMAT_OVERRIDE_PASS2 = """

============================================================
OUTPUT FORMAT OVERRIDE (MANDATORY — supersedes any earlier
"decision: suppressed" / JSON service-list spec)
============================================================

You must emit EXACTLY ONE bracket-tagged action and nothing else:

- [silent]
    Use when, after considering the retrieved memory, the proactive
    service should NOT be delivered now (insufficient evidence,
    contradicted by memory, already resolved, or within cooldown).

- [respond] DAY{N}-HH:MM:SS <reminder>
    Use when retrieved memory CONFIRMS the suspected proactive
    service AND the current caption still justifies a trigger now.
    The timestamp MUST be a single precise time point picked from
    the CURRENT caption window — NOT from the retrieved memory.
    The reminder must be 1-2 short sentences spoken DIRECTLY to the
    user.

Hard rules:
- Output exactly ONE action, nothing before or after.
- Do NOT output JSON.
- Do NOT change the suspected service type relative to Pass 1.
- The trigger timestamp comes ONLY from the current caption window,
  never from retrieved memory.
"""

PROACTIVE_WITH_MEMORY_SYSTEM_PROMPT_EGO: str = (
    _EGO_PROMPTS["proactive_service_prompt_with_memory_simple"].rstrip()
    + _OUTPUT_FORMAT_OVERRIDE_PASS2
)


# ==================================================================
# 5) Pass 1 / Pass 2 的整体 user prompt 模板
# ==================================================================
# 与 prompt_templates_v2.PROACTIVE_PROMPT_V2 / PROACTIVE_WITH_MEMORY_PROMPT_V2
# 相同的占位符结构，pipeline 不需要修改 .format(...) 调用点。
PROACTIVE_PROMPT_EGO: str = """{system_prompt}

--- CURRENT OBSERVATION ---
Timestamp: {current_timestamp}
Caption window: {caption_time_span}
{current_caption}

--- RECENT HISTORY ---
{recent_history}

--- PROACTIVE HISTORY ---
{proactive_history}

Based on the current observation and recent history, decide whether to issue
a proactive reminder. Remember the OUTPUT FORMAT OVERRIDE: emit exactly one
of [silent] / [search] <query> / [respond] DAY{{N}}-HH:MM:SS <reminder>.

Your action:"""


PROACTIVE_WITH_MEMORY_PROMPT_EGO: str = """{system_prompt}

--- CURRENT OBSERVATION ---
Timestamp: {current_timestamp}
Caption window: {caption_time_span}
{current_caption}

--- RETRIEVED MEMORY ---
[observation]
{retrieved_memory}
[/observation]

--- PROACTIVE HISTORY ---
{proactive_history}

Based on the current observation and retrieved memory, decide whether to
finalize the proactive service. Remember the OUTPUT FORMAT OVERRIDE: emit
exactly [silent] or [respond] DAY{{N}}-HH:MM:SS <reminder>.

[respond] """


__all__ = [
    "CAPTION_SYSTEM_PROMPT_EGO",
    "ENTITY_EXTRACTION_PROMPT_EGO",
    "PROACTIVE_SYSTEM_PROMPT_EGO",
    "PROACTIVE_WITH_MEMORY_SYSTEM_PROMPT_EGO",
    "PROACTIVE_PROMPT_EGO",
    "PROACTIVE_WITH_MEMORY_PROMPT_EGO",
]
