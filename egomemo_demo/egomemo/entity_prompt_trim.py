"""精简 ego_prompt_.PROMPTS["entity_extraction"]：在原文上裁剪，不改原文件。

设计原则：
- 不动 videorag/ego_prompt_.py 原文件
- 在自己的进程里 import 时 monkey-patch PROMPTS dict 的 entity_extraction key
- 保留所有"格式约束 / 输出 schema / 字段定义"，这些是 nano-graphrag 解析器要的
- 仅砍掉：
    * Proactive Service Taxonomy 整段（建图阶段不需要服务分类信息）
    * D) Completion 中第 2 个 Example（保留 Example 1 已足够 demo 输出格式）

裁剪后预计 10950 → ~7800 chars (-30%)，nano-graphrag 实体抽取 LLM call 速度可降低
约 25%（input token 与处理时间近似线性）。
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _trim_entity_extraction_prompt(raw: str) -> Optional[str]:
    """在 raw entity_extraction prompt 上做裁剪。失败返回 None（保险起见保留原版）。"""
    out = raw

    # ---- 1) 砍掉 "-Proactive Service Taxonomy-" 整段 ----
    # 该段以 "-Proactive Service Taxonomy" 开头分隔符 ----，结束于下一个 ---- 段
    m = re.search(
        r"-{4,}\n-Proactive Service Taxonomy.*?\n-{4,}\n.*?(?=-{4,}\n-Inputs-)",
        out, flags=re.DOTALL,
    )
    if not m:
        logger.warning("[entity_prompt_trim] taxonomy section pattern not matched, "
                       "skipping that cut")
    else:
        out = out[:m.start()] + out[m.end():]

    # ---- 2) 砍掉 D) Completion 段中的 "Example 2" 块（保留 Example 1）----
    # Example 2 以 "Example 2:" 开头，止于下一个 "######################\n\n#######" 之前
    # 或下一个 Example N: 之前
    m2 = re.search(
        r"######################\nExample 2:.*?(?=######################\n\n#######|\Z)",
        out, flags=re.DOTALL,
    )
    if not m2:
        logger.warning("[entity_prompt_trim] Example 2 pattern not matched, "
                       "skipping that cut")
    else:
        out = out[:m2.start()] + out[m2.end():]

    # 清理多余空行
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


def patch_ego_prompt_entity_extraction() -> None:
    """启动时调用一次：把 ego_prompt_.PROMPTS["entity_extraction"] 替换为精简版。

    注意：必须在 VideoRAG 任何会读到 PROMPTS["entity_extraction"] 的代码 *之前* 调用。
    实际上 VideoRAG 在每次 entity_extraction LLM call 时 *动态* 用 .format() 取该 key，
    所以只要 patch 比那个时刻早就行——run.py 启动时就 patch 是安全的。
    """
    try:
        from videorag.ego_prompt_ import PROMPTS
    except Exception as e:
        logger.warning(f"[entity_prompt_trim] cannot import ego_prompt_: {e}")
        return

    raw = PROMPTS.get("entity_extraction")
    if not raw:
        logger.warning("[entity_prompt_trim] PROMPTS['entity_extraction'] missing, no-op")
        return

    trimmed = _trim_entity_extraction_prompt(raw)
    if trimmed is None or len(trimmed) >= len(raw) * 0.95:
        logger.warning(
            "[entity_prompt_trim] trim ineffective "
            f"(raw={len(raw)} -> trimmed={len(trimmed) if trimmed else 'None'}); "
            "keeping raw to avoid breaking nano-graphrag parser."
        )
        return

    PROMPTS["entity_extraction"] = trimmed
    logger.info(
        f"[entity_prompt_trim] entity_extraction prompt: "
        f"{len(raw)} -> {len(trimmed)} chars "
        f"({(1 - len(trimmed)/len(raw)) * 100:.1f}% reduction)"
    )
