"""
简化版实体关系提取模块（用于 demo）。

图结构：
- 实体：I（person）、Object、Location
- 关系：事件/动作（直接连接实体，带时间戳）
  例如：I --[picking_up @ DAY1-00:01:03]--> white ceramic mug

不修改 VideoRAG 任何已有代码。
"""

import logging
import re
from typing import Dict, List, Tuple

from videorag._utils import compute_mdhash_id

from .prompt_templates import ENTITY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


def _parse_extraction_output(raw_output: str) -> Tuple[List[Dict], List[Dict]]:
    """解析 LLM 输出的实体和关系。

    格式：
        ENTITY|name|type|description
        REL|source|target|action|timestamp|description|strength
    """
    entities = []
    relationships = []

    for line in raw_output.strip().split("\n"):
        line = line.strip()
        if not line or line == "DONE":
            break

        parts = [p.strip() for p in line.split("|")]

        if parts[0] == "ENTITY" and len(parts) >= 4:
            etype = parts[2].upper()
            # 只允许 PERSON / OBJECT / LOCATION
            if etype not in ("PERSON", "OBJECT", "LOCATION"):
                continue
            # PERSON 只允许 "I"
            if etype == "PERSON" and parts[1] != "I":
                continue
            entities.append({
                "entity_name": parts[1],
                "entity_type": etype,
                "description": parts[3],
            })

        elif parts[0] == "REL" and len(parts) >= 6:
            try:
                strength = int(parts[6]) if len(parts) > 6 else 5
            except (ValueError, IndexError):
                strength = 5
            relationships.append({
                "src_id": parts[1],
                "tgt_id": parts[2],
                "action": parts[3],
                "timestamp": parts[4],
                "description": parts[5],
                "weight": min(max(strength, 1), 10),
            })

    return entities, relationships


def _validate_relationships(
    entities: List[Dict], relationships: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """验证关系两端的实体都存在，同时清理孤立实体。"""
    entity_names = {e["entity_name"] for e in entities}

    # 只保留两端都存在的关系
    valid_rels = [
        r for r in relationships
        if r["src_id"] in entity_names and r["tgt_id"] in entity_names
    ]

    # 删除不在任何关系中的非 I 实体
    entities_in_rels = set()
    for r in valid_rels:
        entities_in_rels.add(r["src_id"])
        entities_in_rels.add(r["tgt_id"])

    valid_entities = [
        e for e in entities
        if e["entity_name"] in entities_in_rels or e["entity_name"] == "I"
    ]

    return valid_entities, valid_rels


async def extract_entities_simple(
    chunk_key: str,
    chunk_content: str,
    knowledge_graph_inst,
    entity_vdb,
    global_config: dict,
) -> Tuple[List[Dict], List[Dict]]:
    """简化版实体关系提取，直接写入知识图谱。

    Args:
        chunk_key: chunk 唯一标识符
        chunk_content: caption 文本内容
        knowledge_graph_inst: 知识图谱存储实例
        entity_vdb: 实体向量数据库
        global_config: 全局配置（需要包含 llm.best_model_func）

    Returns:
        (entities, relationships)
    """
    use_llm_func = global_config["llm"]["best_model_func"]

    # 构建 prompt 并调用 LLM
    prompt = ENTITY_EXTRACTION_PROMPT.replace("{input_text}", chunk_content)
    raw_output = await use_llm_func(prompt)
    if not raw_output:
        logger.warning(f"Empty extraction result for chunk {chunk_key}")
        return [], []

    # 解析
    entities, relationships = _parse_extraction_output(raw_output)
    if not entities:
        logger.warning(f"No entities extracted for chunk {chunk_key}")
        return [], []

    # 验证和清理
    entities, relationships = _validate_relationships(entities, relationships)

    # source_id 需要与 text_chunks_db 的 key 一致（哈希值），
    # 这样检索时 _find_most_related_segments_from_entities 才能
    # 通过 source_id 在 text_chunks_db 中找到对应的 chunk。
    hashed_source_id = compute_mdhash_id(f"{chunk_key}_0", prefix="chunk-")

    # 写入知识图谱 — 实体作为节点
    for entity in entities:
        await knowledge_graph_inst.upsert_node(
            entity["entity_name"],
            node_data={
                "entity_name": entity["entity_name"],
                "entity_type": entity["entity_type"],
                "description": entity["description"],
                "source_id": hashed_source_id,
            },
        )

    # 写入知识图谱 — 动作/事件作为边
    for rel in relationships:
        await knowledge_graph_inst.upsert_edge(
            rel["src_id"],
            rel["tgt_id"],
            edge_data={
                "relation_type": rel["action"],
                "description": rel["description"],
                "weight": rel["weight"],
                "timestamp": rel["timestamp"],
                "source_id": hashed_source_id,
            },
        )

    # 更新实体向量数据库
    if entity_vdb is not None and entities:
        vdb_data = {
            compute_mdhash_id(e["entity_name"], prefix="ent-"): {
                "content": e["entity_name"] + " " + e["description"],
                "entity_name": e["entity_name"],
            }
            for e in entities
        }
        await entity_vdb.upsert(vdb_data)

    logger.info(
        f"[EntityExtraction] chunk={chunk_key} | "
        f"entities={len(entities)} | relations={len(relationships)}"
    )

    return entities, relationships
