import re
import json
import openai
import asyncio
import tiktoken
from typing import Union
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .ego_prompt_ import GRAPH_FIELD_SEP, PROMPTS
from .holoassist_prompt_ import HOLOASSIST_PROMPTS
from ._videoutil import (
    retrieved_segment_caption,
    retrieved_segment_caption_minicpm
)

def get_prompts(global_config: dict = None, datasets_type: str = None):
    """
    根据dataset_types从HOLOASSIST_PROMPTS和PROMPTS中选择正确的PROMPTS
    
    Args:
        global_config: 全局配置字典，可能包含datasets_type
        datasets_type: 数据集类型，如果提供则优先使用
        
    Returns:
        选择的PROMPTS字典
    """
    # 优先使用直接提供的datasets_type参数
    if datasets_type is None and global_config is not None:
        datasets_type = global_config.get("datasets_type")
    
    # 如果datasets_type是"HoloAssist"或"holoassist"，使用HOLOASSIST_PROMPTS
    if datasets_type and datasets_type.lower() in ["holoassist", "holo_assist"]:
        return HOLOASSIST_PROMPTS
    else:
        # 默认使用PROMPTS（egolife）
        return PROMPTS

def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_video_segments(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    max_token_size=1024,
):
    # make sure each segment is not larger than max_token_size
    for index in range(len(tokens_list)):
        if len(tokens_list[index]) > max_token_size:
            tokens_list[index] = tokens_list[index][:max_token_size]
    
    results = []
    chunk_token = []
    chunk_segment_ids = []
    chunk_order_index = 0
    for index, tokens in enumerate(tokens_list):
        
        if len(chunk_token) + len(tokens) <= max_token_size:
            # add new segment
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
        else:
            # save the current chunk
            chunk = tiktoken_model.decode(chunk_token)
            results.append(
                {
                    "tokens": len(chunk_token),
                    "content": chunk.strip(),
                    "chunk_order_index": chunk_order_index,
                    "video_segment_id": chunk_segment_ids,
                }
            )
            # new chunk with current segment as begin
            chunk_token = []
            chunk_segment_ids = []
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
            chunk_order_index += 1
    
    # save the last chunk
    if len(chunk_token) > 0:
        chunk = tiktoken_model.decode(chunk_token)
        results.append(
            {
                "tokens": len(chunk_token),
                "content": chunk.strip(),
                "chunk_order_index": chunk_order_index,
                "video_segment_id": chunk_segment_ids,
            }
        )
    
    return results
    
    
def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
    datasets_type: str = None,
):

    prompts = get_prompts(datasets_type=datasets_type)
    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in prompts["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(new_videos, chunk_func=chunking_by_video_segments, **chunk_func_params):
    inserting_chunks = {}

    new_videos_list = list(new_videos.keys())
    for video_name in new_videos_list:
        segment_id_list = list(new_videos[video_name].keys())
        docs = [new_videos[video_name][index]["content"] for index in segment_id_list]
        doc_keys = [f'{video_name}_{index}' for index in segment_id_list]

        ENCODER = tiktoken.encoding_for_model("gpt-4o")
        tokens = ENCODER.encode_batch(docs, num_threads=16)
        chunks = chunk_func(
            tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
        )

        for chunk in chunks:
            inserting_chunks.update(
                {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
            )

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    llm_max_tokens = global_config["llm"]["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompts = get_prompts(global_config=global_config)
    prompt_template = prompts["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0].strip().strip('"').strip("'").lower() != 'entity':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0].strip().strip('"').strip("'").lower() != 'relationship':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )
    return_edge_data = dict(
        src_tgt=(src_id, tgt_id),
        description=description,
        weight=weight
    )
    return return_edge_data


async def streaming_extract_entities(
    chunk_key: str,
    chunk_data: TextChunkSchema,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
    processed_count: int = 0,  # 用于统计已处理的chunk数量
) -> tuple[BaseGraphStorage, list[dict], list[dict]]:
    """
    流式提取单个chunk的实体和关系，并立即合并到知识图谱中。
    
    Args:
        chunk_key: chunk的唯一标识符
        chunk_data: chunk的数据（TextChunkSchema）
        knowledge_graph_inst: 知识图谱存储实例（全局图结构）
        entity_vdb: 实体向量数据库
        global_config: 全局配置
        processed_count: 已处理的chunk数量（用于进度显示）
        
    Returns:
        tuple: (knowledge_graph_inst, all_entities_data, all_edges_data)
    """
    use_llm_func: callable = global_config["llm"]["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    prompts = get_prompts(global_config=global_config)
    entity_extract_prompt = prompts["entity_extraction"]
    context_base = dict(
        tuple_delimiter=prompts["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=prompts["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=prompts["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(prompts["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = prompts["entiti_continue_extraction"]
    if_loop_prompt = prompts["entiti_if_loop_extraction"]
    
    content = chunk_data["content"]
    
    # 第一步：初始提取
    hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
    final_result = await use_llm_func(hint_prompt)
    
    # 第二步：迭代提取（gleaning）
    history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
    for now_glean_index in range(entity_extract_max_gleaning):
        glean_result = await use_llm_func(continue_prompt, history_messages=history)
        
        history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
        final_result += glean_result
        
        # 检查是否需要继续提取
        if now_glean_index == entity_extract_max_gleaning - 1:
            break

        if_loop_result: str = await use_llm_func(
            if_loop_prompt, history_messages=history
        )
        if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
        if if_loop_result != "yes":
            break

    # 解析提取结果
    records = split_string_by_multi_markers(
        final_result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )

    # 收集当前chunk提取的节点和边
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for record in records:
        record_match = re.search(r"\((.*)\)", record)
        if record_match is None:
            continue
        record = record_match.group(1)
        record_attributes = split_string_by_multi_markers(
            record, [context_base["tuple_delimiter"]]
        )
        # 去除每个属性的首尾引号（如果存在），但保留原始值用于类型检查
        record_attributes_cleaned = [
            attr.strip().strip('"').strip("'") if isinstance(attr, str) else str(attr).strip().strip('"').strip("'")
            for attr in record_attributes
        ]
        
        # 尝试提取实体（使用清理后的属性）
        if_entities = await _handle_single_entity_extraction(
            record_attributes_cleaned, chunk_key
        )
        if if_entities is not None:
            maybe_nodes[if_entities["entity_name"]].append(if_entities)
            continue

        # 尝试提取关系（使用清理后的属性）
        if_relation = await _handle_single_relationship_extraction(
            record_attributes_cleaned, chunk_key
        )
        if if_relation is not None:
            # 对于无向图，使用排序后的元组作为key
            edge_key = tuple(sorted([if_relation["src_id"], if_relation["tgt_id"]]))
            maybe_edges[edge_key].append(if_relation)
    
    # 将不符合要求的关系和节点从图中删除
    # 过滤掉不符合条件的node_data，如果列表为空则删除整个键
    nodes_to_remove = []
    for node_name, node_data_list in maybe_nodes.items():
        # 过滤掉不符合条件的node_data（PERSON类型且不是"I"）
        filtered_list = [
            node_data for node_data in node_data_list
            if not ((node_data["entity_type"] == "PERSON" and node_data["entity_name"] != "I") or (node_data["entity_type"] not in ["PERSON", "LOCATION", "OBJECT", "EVENT"]))
        ]
        if not filtered_list:
            # 如果过滤后列表为空，标记为删除
            nodes_to_remove.append(node_name)
        else:
            # 更新为过滤后的列表
            maybe_nodes[node_name] = filtered_list
    
    # 删除所有空的节点
    for node_name in nodes_to_remove:
        del maybe_nodes[node_name]
        
    # 将多余的关系从图中删除
    # 辅助函数：根据entity_name获取entity_type
    def get_entity_type(entity_name):
        """从maybe_nodes中获取entity_name对应的entity_type"""
        if entity_name not in maybe_nodes or not maybe_nodes[entity_name]:
            return None
        # 取第一个node_data的entity_type（通常一个实体名称只有一个类型）
        return maybe_nodes[entity_name][0].get("entity_type")
    
    edges_to_remove = []
    for edge_key, edge_data_list in maybe_edges.items():
        filtered_list = []
        for edge_data in edge_data_list:
            src_id = edge_data.get("src_id")
            tgt_id = edge_data.get("tgt_id")
            
            # 如果src_id或tgt_id为"I"，删除
            if src_id == "I" and tgt_id == "I":
                continue
            
            # 获取src和tgt的entity_type
            src_type = get_entity_type(src_id)
            tgt_type = get_entity_type(tgt_id)
            
            # 如果无法确定类型，保留该边
            if src_type is None or tgt_type is None:
                filtered_list.append(edge_data)
                continue
            
            # 转换为大写以便比较（确保大小写一致）
            src_type = src_type.upper()
            tgt_type = tgt_type.upper()
            
            # 判断是否需要删除：
            # 1. 二者都是LOCATION
            # 2. 二者都是OBJECT
            # 3. 一个是PERSON另一个是LOCATION
            should_remove = (
                (src_type == "LOCATION" and tgt_type == "LOCATION") or
                (src_type == "OBJECT" and tgt_type == "OBJECT") or
                ((src_type == "PERSON" and tgt_type == "LOCATION") or
                 (src_type == "LOCATION" and tgt_type == "PERSON"))
            )
            
            # 如果不需要删除，保留该边
            if not should_remove:
                filtered_list.append(edge_data)
        
        if not filtered_list:
            # 如果过滤后列表为空，标记为删除
            edges_to_remove.append(edge_key)
        else:
            # 更新为过滤后的列表
            maybe_edges[edge_key] = filtered_list
    
    # 删除所有空的关系
    for edge_key in edges_to_remove:
        del maybe_edges[edge_key]
    
    # 收集所有在边中出现的实体名称
    entities_in_edges = set()
    for edge_key, edge_data_list in maybe_edges.items():
        for edge_data in edge_data_list:
            src_id = edge_data.get("src_id")
            tgt_id = edge_data.get("tgt_id")
            if src_id:
                entities_in_edges.add(src_id)
            if tgt_id:
                entities_in_edges.add(tgt_id)
    
    # 删除不包含在任何关系中的节点
    nodes_to_remove_no_edge = []
    for node_name in maybe_nodes.keys():
        if node_name not in entities_in_edges:
            nodes_to_remove_no_edge.append(node_name)
    
    for node_name in nodes_to_remove_no_edge:
        del maybe_nodes[node_name]
        logger.info(f"Removed node '{node_name}' because it is not in any edge")
    
    # 删除关系中的实体不在节点中的边
    edges_to_remove_invalid = []
    for edge_key, edge_data_list in maybe_edges.items():
        filtered_list = []
        for edge_data in edge_data_list:
            src_id = edge_data.get("src_id")
            tgt_id = edge_data.get("tgt_id")
            
            # 检查src_id和tgt_id是否都在maybe_nodes中
            if src_id not in maybe_nodes or tgt_id not in maybe_nodes:
                logger.info(f"Removed edge ({src_id}, {tgt_id}) because src or tgt entity not in nodes")
                continue

            filtered_list.append(edge_data)
        
        if not filtered_list:
            # 如果过滤后列表为空，标记为删除
            edges_to_remove_invalid.append(edge_key)
        else:
            # 更新为过滤后的列表
            maybe_edges[edge_key] = filtered_list
    
    # 删除所有无效的关系
    for edge_key in edges_to_remove_invalid:
        del maybe_edges[edge_key]
    
    # 立即将节点和边合并到图中（流式处理，不需要等待所有chunks）
    # _merge_nodes_then_upsert 和 _merge_edges_then_upsert 会自动处理与已有节点/边的合并
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_edges_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    
    # 更新实体向量数据库（增量更新）
    if entity_vdb is not None and len(all_entities_data) > 0:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    
    # 进度显示（可选）
    processed_count += 1
    prompts = get_prompts(global_config=global_config)
    now_ticks = prompts["process_tickers"][
        processed_count % len(prompts["process_tickers"])
    ]
    print(
        f"{now_ticks} Processed {processed_count} chunks, extracted {len(all_entities_data)} entities, {len(all_edges_data)} relations\r",
        end="",
        flush=True,
    )

    return knowledge_graph_inst, all_entities_data, all_edges_data


async def _find_most_related_segments_from_entities(
    topk_chunks: int,
    node_datas: list[dict],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    sorted_text_units = sorted(
        all_text_units, key=lambda x: -x["relation_counts"]
    )[:topk_chunks]
    
    chunk_related_segments = set()
    for _chunk_data in sorted_text_units:
        for s_id in _chunk_data['data']['time_span']:
            chunk_related_segments.add(s_id)
    
    return chunk_related_segments

async def _refine_entity_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
    datasets_type: str = None,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    prompts = get_prompts(global_config=global_config, datasets_type=datasets_type)
    query_rewrite_prompt = prompts["query_rewrite_for_entity_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt)
    return final_result

async def _refine_visual_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
    datasets_type: str = None,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    prompts = get_prompts(global_config=global_config, datasets_type=datasets_type)
    query_rewrite_prompt = prompts["query_rewrite_for_visual_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt)
    return final_result

async def _extract_keywords_query(
    query,
    query_param: QueryParam,
    global_config: dict,
    datasets_type: str = None,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    prompts = get_prompts(global_config=global_config, datasets_type=datasets_type)
    keywords_prompt = prompts["keywords_extraction"]
    keywords_prompt = keywords_prompt.format(input_text=query)
    final_result = await use_llm_func(keywords_prompt)
    return final_result


async def streaming_videorag_query(
    query,
    time_key, 
    service_type, 
    sub_service_type, 
    datasets_type, 
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    caption_model,
    caption_tokenizer,
    query_param, 
    global_config: dict, 
    reconstruct_caption=True, 
    use_minicpm=False, 
) -> str:
    use_model_func = global_config["llm"]["best_model_func"]
    
    # naive captions  检索原始caption内容, 这里检索到的caption应该包含三种层级的，seconds, minutes, hours的caption, 按着相似度从低到高排序
    results = await chunks_vdb.query(query, top_k=1000)      # 原来是query_param.top_k
    if not len(results):
        prompts = get_prompts(global_config=global_config, datasets_type=datasets_type)
        return prompts["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)
    
    # 解析time_span: '2-10:48:07-10:48:32_0' -> day=2, start='10:48:07', end='10:48:32'
    def parse_time_span(time_span_str):
        """解析time_span格式: '2-10:48:07-10:48:32_0'"""
        # 移除索引后缀（如果有）
        time_span = time_span_str.rsplit('_', 1)[0] if '_' in time_span_str else time_span_str
        
        # 分离day和time_range
        time_span_parts = time_span.split('-', 1)
        if len(time_span_parts) < 2:
            return None, None, None
        
        day = int(time_span_parts[0])  # '2'
        time_range = time_span_parts[1]  # '10:48:07-10:48:32'
        
        # 分离start_time和end_time
        if '-' in time_range:
            start_time_str, end_time_str = time_range.split('-', 1)
        else:
            start_time_str = time_range
            end_time_str = time_range
        
        return day, start_time_str, end_time_str
    
    # 将时间字符串转换为秒数（从day开始计算）
    def time_str_to_seconds(day, time_str):
        """将时间字符串转换为秒数（从day开始计算）"""
        time_parts = time_str.split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2]) if len(time_parts) > 2 else 0
        return day * 86400 + hour * 3600 + minute * 60 + second
    
    def filter_items_by_service_type(items, time_key, service_type, key_day, 
                                     get_time_span_func=None, original_items=None):
        """
        根据service_type和time_key过滤items（可以是segments或chunks等）
        这里改为计算截止时间的差距
        
        Args:
            items: 要过滤的items列表（可以是segment IDs或chunks等）
            time_key: 时间键字符串，格式如 'Day2 10:09:03-10:09:15'
            service_type: 服务类型，如 "Short-Term Proactive Service"
            key_day: 解析后的day值（从time_key解析得到）
            get_time_span_func: 可选函数，用于从item中提取time_span字符串
                               如果为None，则假设items本身就是time_span字符串列表
            original_items: 原始items列表，用于在过滤后为空时回退
        
        Returns:
            tuple: (filtered_items, was_filtered)
                - filtered_items: 过滤后的items列表
                - was_filtered: 是否进行了过滤（True表示过滤了，False表示使用原始items）
        """
        if key_day is None:
            return items, False
        
        # 解析time_key的开始时间（用于时间点比较）
        key_day_parsed, key_start, key_end = parse_time_span(time_key)
        if key_day_parsed is None:
            return items, False
        
        key_end_sec = time_str_to_seconds(key_day_parsed, key_end)
        
        filtered_items = []
        for item in items:
            # 获取time_span字符串
            if get_time_span_func is not None:
                time_span_str = get_time_span_func(item)
                if time_span_str is None:
                    continue
            else:
                # 假设item本身就是time_span字符串
                time_span_str = item
            
            # 解析time_span
            span_day, span_start, span_end = parse_time_span(time_span_str)
            if span_day is None or span_end is None:
                continue
        
            # 计算chunk的开始时间（秒数）
            span_end_sec = time_str_to_seconds(span_day, span_end)
            
            # 要求：chunk的时间点必须在time_key之前（日期和时：分：秒都应该在time_key之前）
            if span_end_sec > key_end_sec:
                continue  # chunk的时间在time_key之后或等于，跳过
            
            # 计算时间差（chunk在time_key之前，所以是key_start_sec - span_start_sec）
            time_diff = key_end_sec - span_end_sec
            
            # 根据service_type过滤
            if service_type == "Short-Term Proactive Service":
                # 同一天10分钟内（0-600秒）
                # 检查是否在同一天
                if span_day != key_day:
                    continue  # 不在同一天，跳过
                if 0 <= time_diff <= 600:
                    filtered_items.append(item)
            elif service_type == "Episodic Proactive Service":
                # 同一天10分钟以上2.5小时以内（600-9000秒）
                # 检查是否在同一天
                if span_day != key_day:
                    continue  # 不在同一天，跳过
                if 600 < time_diff <= 9000:
                    filtered_items.append(item)
            elif service_type == "Long-Term Proactive Service":
                # 跨天或同一天2.5小时以上（>9000秒）
                # Long-Term允许跨天，所以不需要检查是否同一天
                if time_diff > 9000:
                    filtered_items.append(item)
            else:
                # 如果不存在主动服务时，则选择所有时间点在time_key之前的chunks
                if time_diff >= 0:
                    filtered_items.append(item)
        
        # 如果过滤后为空，回退到原始items
        if len(filtered_items) == 0:
            if original_items is not None:
                logger.warning(f"No items match service_type {service_type} time range, using original items")
                return original_items, False
            else:
                logger.warning(f"No items match service_type {service_type} time range, returning empty list")
                return [], True
        else:
            logger.info(f"Filtered items by service_type {service_type}: {len(filtered_items)} items remain (from {len(items)})")
            return filtered_items, True
    
    # 解析time_key
    key_day, key_start, key_end = parse_time_span(time_key)
    
    # 根据service_type过滤chunks
    def get_chunk_time_span(chunk):
        """从chunk中提取time_span字符串"""
        chunk_time_span = None
        if "time_span" in chunk:
            chunk_time_span = chunk["time_span"][0]
        elif "data" in chunk and isinstance(chunk["data"], dict) and "time_span" in chunk["data"]:
            # time_span可能是列表，取第一个
            time_span_list = chunk["data"]["time_span"]
            if isinstance(time_span_list, list) and len(time_span_list) > 0:
                chunk_time_span = time_span_list[0]
            elif isinstance(time_span_list, str):
                chunk_time_span = time_span_list
        return chunk_time_span
    
    # 首先将时间线前的captions保留
    filtered_chunks, was_filtered = filter_items_by_service_type(
        items=chunks,
        time_key=time_key,
        service_type=service_type,
        key_day=key_day,
        get_time_span_func=get_chunk_time_span,
        original_items=[], 
    )
    
    # 下一步进行多尺度时序检索
    hour_chunks = [chunk for chunk in filtered_chunks if chunk["type"] == "hour"][-query_param.hour_top_k:]     # 首先筛选出hour_block中相似度最高的部分
    minute_chunks = [chunk for chunk in filtered_chunks if chunk["type"] == "minute"]       # 针对每个hour_block中的
    second_chunks = [chunk for chunk in filtered_chunks if chunk["type"] == "second"]
    
    # 如果hour_chunks为空，直接从所有minute_chunks中选择最相似的
    if not hour_chunks:
        filtered_minute_chunks = minute_chunks[-query_param.minute_top_k:] if minute_chunks else []
    else:
        # 从minute_chunks中筛选出那些time_span在hour_chunks的sub_window_captions中的
        # 检查所有hour_chunks的sub_window_captions，而不仅仅是前两个
        hour_sub_window_captions = set()
        for hour_chunk in hour_chunks:
            if "sub_window_captions" in hour_chunk:
                hour_sub_window_captions.update(hour_chunk["sub_window_captions"][0])
        filtered_minute_chunks = [
            chunk for chunk in minute_chunks 
            if chunk["time_span"][0].split('_')[0] in hour_sub_window_captions
        ][-query_param.minute_top_k:]
    
    # 如果minute_chunks也是空，直接从所有second_chunks中选择最相似的
    if not filtered_minute_chunks:
        filtered_second_chunks = second_chunks[-query_param.second_top_k:] if second_chunks else []
    else:
        # 从second_chunks中筛选出那些time_span在filtered_minute_chunks的sub_window_captions中的
        # 检查所有filtered_minute_chunks的sub_window_captions，而不仅仅是前两个
        minute_sub_window_captions = set()
        for minute_chunk in filtered_minute_chunks:
            if "sub_window_captions" in minute_chunk:
                minute_sub_window_captions.update(minute_chunk["sub_window_captions"][0])
        filtered_second_chunks = [
            chunk for chunk in second_chunks 
            if chunk["time_span"][0].split('_')[0] in minute_sub_window_captions
        ][-query_param.second_top_k:] 
    
    chunks = list(hour_chunks + filtered_minute_chunks + filtered_second_chunks)

    # 防止cpation的token数量超过模型最大限制
    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "-----Retrieved Captions-----\n".join([c["content"] for c in maybe_trun_chunks])
    retreived_chunk_context = section
    
    # visual retrieval   在知识图谱中检索实体关系对应的视频片段
    query_for_entity_retrieval = await _refine_entity_retrieval_query(
        query,
        query_param,
        global_config,
        datasets_type=datasets_type,
    )
    
    # 检索知识图谱中的实体，这一步是必做的且整理步骤可以和VideoRAG一致
    entity_results = await entities_vdb.query(query_for_entity_retrieval, top_k=1000)     # query_param.top_k
    entity_retrieved_segments = set()
    if len(entity_results):
        node_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
        )
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_results]
        )
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(entity_results, node_datas, node_degrees)
            if n is not None
        ]
        entity_retrieved_segments = entity_retrieved_segments.union(await _find_most_related_segments_from_entities(
            global_config["retrieval_topk_chunks"], node_datas, text_chunks_db, knowledge_graph_inst
        ))
        
        # 首先根据时间间隔将根据实体检索出的文本片段进行过滤
        entity_retrieved_segments, _ = filter_items_by_service_type(
            items=list(entity_retrieved_segments),
            time_key=time_key,
            service_type=service_type,
            key_day=key_day,
            get_time_span_func=None,  # items本身就是time_span字符串
            original_items=[], 
        )
        # 再次根据预先设定的数量进行过滤
        entity_retrieved_segments = set(entity_retrieved_segments[-query_param.top_k:])
        
    # visual retrieval
    # visual retrieval  通过和视觉特征的相似度检索对应的视频片段
    query_for_visual_retrieval = await _refine_visual_retrieval_query(
        query,
        query_param,
        global_config,
        datasets_type=datasets_type,
    )
    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)      # 检索出最相似的视觉片段
    visual_retrieved_segments = set()
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_segments.add(n['__id__'])
    
    # 首先根据时间间隔将根据视觉检索出的文本片段进行过滤
    visual_retrieved_segments, _ = filter_items_by_service_type(
        items=list(visual_retrieved_segments),
        time_key=time_key,
        service_type=service_type,
        key_day=key_day,
        get_time_span_func=None,  # items本身就是time_span字符串
        original_items=[], 
    )
    # 再次根据预先设定的数量进行过滤
    visual_retrieved_segments = set(visual_retrieved_segments[-query_param.top_k:])
    # caption    这里和之前的VideoRAG不一样，因为它的text chunks会对应多个视频片段，同时视觉特征的编码是按照一整段视频的顺序去编码的，而我们的视觉特征是按照每帧进行的编码
    retrieved_segments = list(entity_retrieved_segments.union(visual_retrieved_segments))
    
    def sort_key(segment_id):
        """
        Sort key for segment IDs in format: day-time_range-index
        Example: '2-10:44:25-10:44:29_0'
        Sort by: day (int), then time range (by start time)
        """
        # Split by last '_' to separate time_span and index
        parts = segment_id.rsplit('_', 1)
        time_span = parts[0]  # '2-10:44:25-10:44:29'
        index = int(parts[1]) if len(parts) > 1 else 0
        
        # Split time_span by first '-' to get day and time_range
        time_span_parts = time_span.split('-', 1)
        day = int(time_span_parts[0])  # '2'
        time_range = time_span_parts[1] if len(time_span_parts) > 1 else ''  # '10:44:25-10:44:29'
        
        # Extract start time from time_range (before second '-')
        start_time = time_range.split('-')[0] if '-' in time_range else time_range  # '10:44:25'
        
        # Convert start_time to comparable format (HH:MM:SS can be compared as string)
        return (day, start_time, index)
    
    retrieved_segments = sorted(retrieved_segments, key=sort_key)
    
    already_processed = 0
    prompts = get_prompts(global_config=global_config, datasets_type=datasets_type)
    async def _filter_single_segment(knowledge: str, segment_key_dp: tuple[str, str]):
        nonlocal use_model_func, already_processed, prompts
        segment_key = segment_key_dp[0]
        segment_content = segment_key_dp[1]
        filter_prompt = prompts["filtering_segment"]
        filter_prompt = filter_prompt.format(caption=segment_content, knowledge=knowledge)
        result = await use_model_func(filter_prompt)
        already_processed += 1
        now_ticks = prompts["process_tickers"][
            already_processed % len(prompts["process_tickers"])
        ]
        print(
            f"{now_ticks} Checked {already_processed} segments\r",
            end="",
            flush=True,
        )
        return (segment_key, result)
    
    rough_captions = {}     # 过滤无用的视频片段
    for s_id in retrieved_segments:      # 2-10:44:25-10:44:29_0    这里和videorag中的做法又有区别，因为这里就分别对应一段视频以及caption的内容，而VideoRAG中每一个视频有很多caption
        # video_name = '_'.join(s_id.split('_')[:-1])
        # index = s_id.split('_')[-1]
        # Extract time_span from segment_id (everything before last '_')
        time_span = '_'.join(s_id.split('_')[:-1]) if '_' in s_id else s_id
        
        rough_captions[s_id] = video_segments._data[time_span]["content"]
    results = await asyncio.gather(
        *[_filter_single_segment(query, (s_id, rough_captions[s_id])) for s_id in rough_captions]
    )
    remain_segments = [x[0] for x in results if 'yes' in x[1].lower()]
    print(f"{len(remain_segments)} Video Segments remain after filtering")
    if len(remain_segments) == 0:
        print("Since no segments remain after filtering, we utilized all the retrieved segments.")
        remain_segments = retrieved_segments
    print(f"Remain segments {remain_segments}")
    
    caption_construction_prompt = get_prompts(global_config=global_config, datasets_type=datasets_type)["caption_reconstruction"]
    
    # visual retrieval 这一部分暂时先去掉，因为提取关键词并根据关键词重写cpation感觉比较耗时而且没有必要
    if reconstruct_caption:
        keywords_for_caption = await _extract_keywords_query(
            query,
            query_param,
            global_config,
        )
        print(f"Keywords: {keywords_for_caption}")
        if use_minicpm:
            caption_results = retrieved_segment_caption_minicpm(
                caption_model,
                caption_tokenizer,
                keywords_for_caption,
                remain_segments,
                video_segments,
                caption_construction_prompt,
                sub_service_type, 
        )
        else:
            caption_results = retrieved_segment_caption(
                caption_model,
                caption_tokenizer,
                keywords_for_caption,
                remain_segments,
                video_segments,
                caption_construction_prompt,
                sub_service_type, 
        )

    else:
        caption_results = [video_segments._data[key]["content"] for key in remain_segments]

    # 根据检索到的上下文生成响应
    # 整理caption_results：如果是字典，按时间戳和内容格式化输出
    if isinstance(caption_results, dict):
        formatted_captions = []
        for timestamp_key, caption_content in caption_results.items():
            formatted_captions.append(f"Timestamp: {timestamp_key}\nContent: {caption_content}\n")
        caption_results_str = "\n".join(formatted_captions)
    elif isinstance(caption_results, list):
        # 如果是列表，尝试从remain_segments获取时间戳信息
        formatted_captions = []
        for i, caption_content in enumerate(caption_results):
            if i < len(remain_segments):
                timestamp_key = remain_segments[i]
                formatted_captions.append(f"Timestamp: {timestamp_key}\nContent: {caption_content}\n")
            else:
                formatted_captions.append(f"Content: {caption_content}\n")
        caption_results_str = "\n".join(formatted_captions)
    else:
        caption_results_str = str(caption_results)
    
    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```{caption_results_str}\n```\n"
    
    return retreived_video_context, retreived_chunk_context


async def videorag_query_multiple_choice(
    query,
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_path_db,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    caption_model,
    caption_tokenizer,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """_summary_
    A copy of the videorag_query function with several updates for handling multiple-choice queries.
    """
    use_model_func = global_config["llm"]["best_model_func"]
    query = query
    
    # naive chunks
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    # NOTE: I update here, not len results can also process
    if len(results):
        chunks_ids = [r["id"] for r in results]
        chunks = await text_chunks_db.get_by_ids(chunks_ids)

        maybe_trun_chunks = truncate_list_by_token_size(
            chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.naive_max_token_for_text_unit,
        )
        logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
        section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
        retreived_chunk_context = section
    else:
        retreived_chunk_context = "No Content"
        
    # visual retrieval
    query_for_entity_retrieval = await _refine_entity_retrieval_query(
        query,
        query_param,
        global_config,
    )
    entity_results = await entities_vdb.query(query_for_entity_retrieval, top_k=query_param.top_k)
    entity_retrieved_segments = set()
    if len(entity_results):
        node_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
        )
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_results]
        )
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(entity_results, node_datas, node_degrees)
            if n is not None
        ]
        entity_retrieved_segments = entity_retrieved_segments.union(await _find_most_related_segments_from_entities(
            global_config["retrieval_topk_chunks"], node_datas, text_chunks_db, knowledge_graph_inst
        ))
    
    # visual retrieval
    query_for_visual_retrieval = await _refine_visual_retrieval_query(
        query,
        query_param,
        global_config,
    )
    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)
    visual_retrieved_segments = set()
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_segments.add(n['__id__'])
    
    # caption
    retrieved_segments = list(entity_retrieved_segments.union(visual_retrieved_segments))
    retrieved_segments = sorted(
        retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]), # video_name
            eval(x.split('_')[-1]) # index
        )
    )
    print(query_for_entity_retrieval)
    print(f"Retrieved Text Segments {entity_retrieved_segments}")
    print(query_for_visual_retrieval)
    print(f"Retrieved Visual Segments {visual_retrieved_segments}")
    
    already_processed = 0
    prompts = get_prompts(global_config=global_config)
    async def _filter_single_segment(knowledge: str, segment_key_dp: tuple[str, str]):
        nonlocal use_model_func, already_processed, prompts
        segment_key = segment_key_dp[0]
        segment_content = segment_key_dp[1]
        filter_prompt = prompts["filtering_segment"]
        filter_prompt = filter_prompt.format(caption=segment_content, knowledge=knowledge)
        result = await use_model_func(filter_prompt)
        already_processed += 1
        now_ticks = prompts["process_tickers"][
            already_processed % len(prompts["process_tickers"])
        ]
        print(
            f"{now_ticks} Checked {already_processed} segments\r",
            end="",
            flush=True,
        )
        return (segment_key, result)
    
    rough_captions = {}
    for s_id in retrieved_segments:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        rough_captions[s_id] = video_segments._data[video_name][index]["content"]
    results = await asyncio.gather(
        *[_filter_single_segment(query, (s_id, rough_captions[s_id])) for s_id in rough_captions]
    )
    remain_segments = [x[0] for x in results if 'yes' in x[1].lower()]
    print(f"{len(remain_segments)} Video Segments remain after filtering")
    if len(remain_segments) == 0:
        print("Since no segments remain after filtering, we utilized all the retrieved segments.")
        remain_segments = retrieved_segments
    print(f"Remain segments {remain_segments}")
    
    # visual retrieval
    keywords_for_caption = await _extract_keywords_query(
        query,
        query_param,
        global_config,
    )
    print(f"Keywords: {keywords_for_caption}")
    caption_results = retrieved_segment_caption(
        caption_model,
        caption_tokenizer,
        keywords_for_caption,
        remain_segments,
        video_path_db,
        video_segments,
        num_sampled_frames=global_config['fine_num_frames_per_segment']
    )

    ## data table
    text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
    for s_id in caption_results:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        start_time = eval(video_segments._data[video_name][index]["time"].split('-')[0])
        end_time = eval(video_segments._data[video_name][index]["time"].split('-')[1])
        start_time = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}"
        end_time = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}"
        text_units_section_list.append([video_name, start_time, end_time, caption_results[s_id]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"
    
    # NOTE: I update here to use a different prompt
    sys_prompt_temp = prompts["videorag_response_for_multiple_choice_question"]
        
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,
        response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        use_cache=False,
    )
    while True:
        try:
            json_response = json.loads(response)
            assert "Answer" in json_response and "Explanation" in json_response
            return json_response
        except Exception as e:
            logger.info(f"Response is not valid JSON for query {query}. Found {e}. Retrying...")
            response = await use_model_func(
                query,
                system_prompt=sys_prompt,
                use_cache=False,
            )
    