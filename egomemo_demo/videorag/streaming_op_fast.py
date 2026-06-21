"""
ego_op.py 中 streaming_videorag_query 的加速版本。

基于 ego_op.py 的检索逻辑，优化点：
1. 去掉逐个 segment 的 LLM 过滤（_filter_single_segment），直接使用检索结果
2. entity query 改写、visual query 改写、关键词提取三步并行执行（asyncio.gather）
3. 时间过滤前置，在 VDB 查询后立即过滤

不修改已有代码，仅新增本文件。调用方通过导入 streaming_videorag_query_fast 替换原函数即可。
"""

import asyncio
from typing import Optional

from ._utils import (
    logger,
    truncate_list_by_token_size,
)
from .base import QueryParam
from ._videoutil import (
    retrieved_segment_caption,
    retrieved_segment_caption_minicpm,
)
from .ego_op import (
    get_prompts,
    _refine_entity_retrieval_query,
    _refine_visual_retrieval_query,
    _extract_keywords_query,
    _find_most_related_segments_from_entities,
)


# ======================================================================
# 时间解析工具函数
# ======================================================================

def _parse_time_span(time_span_str: str):
    """解析 time_span 格式: '2-10:48:07-10:48:32_0' -> (day, start_str, end_str)"""
    time_span = time_span_str.rsplit('_', 1)[0] if '_' in time_span_str else time_span_str
    parts = time_span.split('-', 1)
    if len(parts) < 2:
        return None, None, None
    day = int(parts[0])
    time_range = parts[1]
    if '-' in time_range:
        start_str, end_str = time_range.split('-', 1)
    else:
        start_str = time_range
        end_str = time_range
    return day, start_str, end_str


def _time_str_to_seconds(day: int, time_str: str) -> int:
    """将时间字符串转换为秒数（从 day 开始计算）"""
    parts = time_str.split(':')
    h = int(parts[0])
    m = int(parts[1])
    s = int(float(parts[2])) if len(parts) > 2 else 0  # 处理小数秒
    return day * 86400 + h * 3600 + m * 60 + s


def _is_before_deadline(item_time_span: str, deadline_seconds: int) -> bool:
    """判断 item 的结束时间是否严格在 deadline 之前"""
    day, _, end_str = _parse_time_span(item_time_span)
    if day is None or end_str is None:
        return False
    return _time_str_to_seconds(day, end_str) < deadline_seconds


def _compute_deadline_seconds(time_key: str) -> Optional[int]:
    """从 time_key 解析出截止时间的秒数"""
    day, _, end_str = _parse_time_span(time_key)
    if day is None:
        return None
    return _time_str_to_seconds(day, end_str)


def _get_chunk_time_span(chunk):
    """从 chunk 中提取 time_span 字符串"""
    if "time_span" in chunk:
        return chunk["time_span"][0]
    if "data" in chunk and isinstance(chunk["data"], dict) and "time_span" in chunk["data"]:
        ts = chunk["data"]["time_span"]
        if isinstance(ts, list) and ts:
            return ts[0]
        if isinstance(ts, str):
            return ts
    return None


def _sort_key(segment_id):
    """segment ID 排序键"""
    parts = segment_id.rsplit('_', 1)
    time_span = parts[0]
    index = int(parts[1]) if len(parts) > 1 else 0
    ts_parts = time_span.split('-', 1)
    day = int(ts_parts[0])
    start_time = ts_parts[1].split('-')[0] if len(ts_parts) > 1 and '-' in ts_parts[1] else ''
    return (day, start_time, index)


def _format_time_span_display(time_span_raw: str) -> str:
    """将 '1-19:02:42-20:06:57' 转换为 'DAY1-19:02:42-20:06:57'"""
    parts = time_span_raw.split("-")
    if len(parts) >= 3:
        return f"DAY{parts[0]}-{parts[1]}-{parts[2]}"
    return time_span_raw


# ======================================================================
# 主检索函数
# ======================================================================

async def streaming_videorag_query_fast(
    query: str,
    time_key: str,
    service_type: str,
    sub_service_type: str,
    datasets_type: str,
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    caption_model,
    caption_tokenizer,
    query_param: QueryParam,
    global_config: dict,
    reconstruct_caption: bool = True,
    use_minicpm: bool = False,
    args=None,
    ori_query=None,
    similarity_top_k: int = 200,
) -> str:
    """
    ego_op.py streaming_videorag_query 的加速版本。

    与原版相比的改动：
    1. 去掉了逐个 segment 的 LLM 过滤（_filter_single_segment）
    2. entity query 改写、visual query 改写、关键词提取三步并行
    3. 时间过滤前置

    保持与 ego_op.py 一致的：
    - 多尺度检索逻辑（[:n] 取相似度最高）
    - other_second_chunks + 去重
    - 分层格式化输出（High/Mid/Low level）
    - args 控制各模块开关
    - ori_query 用于 query 改写
    """
    if ori_query is None:
        ori_query = query

    deadline_sec = _compute_deadline_seconds(time_key)

    # ------------------------------------------------------------------
    # Step 0: 并行执行 query 改写 + 关键词提取
    # 根据 args 决定哪些需要执行，不需要的用 asyncio.coroutine 占位
    # ------------------------------------------------------------------
    need_entity = args is None or getattr(args, 'entity_retrieval', True)
    need_visual = args is None or getattr(args, 'visual_retrieval', True)
    need_reconstruct = args is None or getattr(args, 'reconstruct_caption', reconstruct_caption)

    tasks = {}
    if need_entity:
        tasks['entity'] = _refine_entity_retrieval_query(
            ori_query, query_param, global_config, datasets_type=datasets_type,
        )
    if need_visual:
        tasks['visual'] = _refine_visual_retrieval_query(
            ori_query, query_param, global_config, datasets_type=datasets_type,
        )
    if need_reconstruct:
        tasks['keywords'] = _extract_keywords_query(
            ori_query, query_param, global_config, datasets_type=datasets_type,
        )

    # 并行执行所有需要的 LLM 调用
    task_keys = list(tasks.keys())
    task_coros = [tasks[k] for k in task_keys]
    if task_coros:
        task_results = await asyncio.gather(*task_coros)
        task_map = dict(zip(task_keys, task_results))
    else:
        task_map = {}

    query_for_entity = task_map.get('entity', '')
    query_for_visual = task_map.get('visual', '')
    keywords_for_caption = task_map.get('keywords', '')

    # ------------------------------------------------------------------
    # Step 1: 文本相似度检索（caption chunks）
    # ------------------------------------------------------------------
    need_caption = args is None or getattr(args, 'caption_retrieval', True)
    use_multiscale = args is None or getattr(args, 'multiscale', True)
    retreived_chunk_context = ""

    if need_caption:
        results = await chunks_vdb.query(query, top_k=similarity_top_k)
        if not results:
            prompts = get_prompts(global_config=global_config, datasets_type=datasets_type)
            return prompts["fail_response"]

        chunks_ids = [r["id"] for r in results]
        chunks = await text_chunks_db.get_by_ids(chunks_ids)

        # 时间过滤：只保留 time_key 之前的 chunks
        if deadline_sec is not None:
            chunks = [
                c for c in chunks
                if (ts := _get_chunk_time_span(c)) is not None and _is_before_deadline(ts, deadline_sec)
            ]

        # 多尺度时序检索（与 ego_op.py 一致，[:n] 取相似度最高）
        hour_top_k = getattr(query_param, 'hour_top_k', 1)
        minute_top_k = getattr(query_param, 'minute_top_k', 1)
        second_top_k = getattr(query_param, 'second_top_k', 2)

        hour_chunks = [c for c in chunks if c.get("type") == "hour"][:hour_top_k]
        minute_chunks = [c for c in chunks if c.get("type") == "minute"]
        second_chunks = [c for c in chunks if c.get("type") == "second"]

        if not hour_chunks:
            filtered_minute_chunks = minute_chunks[:minute_top_k]
        else:
            hour_sub = set()
            for hc in hour_chunks:
                if "sub_window_captions" in hc:
                    hour_sub.update(hc["sub_window_captions"][0])
            filtered_minute_chunks = [
                c for c in minute_chunks
                if c["time_span"][0].split('_')[0] in hour_sub
            ][:minute_top_k]

        if not filtered_minute_chunks:
            filtered_second_chunks = second_chunks[:second_top_k]
        else:
            min_sub = set()
            for mc in filtered_minute_chunks:
                if "sub_window_captions" in mc:
                    min_sub.update(mc["sub_window_captions"][0])
            filtered_second_chunks = [
                c for c in second_chunks
                if c["time_span"][0].split('_')[0] in min_sub
            ][:second_top_k]

        # 额外保留直接相似度最高的 second_chunks + 去重（与 ego_op.py 一致）
        other_second_chunks = second_chunks[:second_top_k]
        if use_multiscale:
            all_second_chunks = list(filtered_second_chunks + other_second_chunks)
        else:
            all_second_chunks = list(other_second_chunks)

        # 去重
        seen_time_spans = set()
        deduplicated_second_chunks = []
        for chunk in all_second_chunks:
            ts = chunk.get("time_span", [])
            if isinstance(ts, list) and len(ts) > 0:
                ts_key = tuple(ts)
            elif isinstance(ts, str):
                ts_key = ts
            else:
                ts_key = str(ts)
            if ts_key not in seen_time_spans:
                seen_time_spans.add(ts_key)
                deduplicated_second_chunks.append(chunk)

        logger.info(f"[fast] Deduplicated second chunks: {len(all_second_chunks)} -> {len(deduplicated_second_chunks)}")

        # 分层格式化输出（与 ego_op.py 一致）
        high_level_chunks = hour_chunks
        mid_level_chunks = filtered_minute_chunks
        low_level_chunks = deduplicated_second_chunks

        sections = []
        if datasets_type != "egoschema" and use_multiscale:
            if high_level_chunks:
                maybe_trun_high = truncate_list_by_token_size(
                    high_level_chunks,
                    key=lambda x: x["content"],
                    max_token_size=query_param.naive_max_token_for_text_unit // 4,
                )
                if maybe_trun_high:
                    high_content_list = []
                    for c in maybe_trun_high:
                        ts_display = _format_time_span_display(c["time_span"][0].split("_")[0])
                        high_content_list.append(f"{ts_display}: {c['content']}")
                    sections.append(f"-----Retrieved High-level Captions-----\n" + "\n".join(high_content_list) + "\n")
                    logger.info(f"[fast] High-level: {len(high_level_chunks)} -> {len(maybe_trun_high)} chunks")

            if mid_level_chunks:
                maybe_trun_mid = truncate_list_by_token_size(
                    mid_level_chunks,
                    key=lambda x: x["content"],
                    max_token_size=query_param.naive_max_token_for_text_unit // 4,
                )
                if maybe_trun_mid:
                    mid_content_list = []
                    for c in maybe_trun_mid:
                        ts_display = _format_time_span_display(c["time_span"][0].split("_")[0])
                        mid_content_list.append(f"{ts_display}: {c['content']}")
                    sections.append(f"-----Retrieved Mid-level Captions-----\n" + "\n".join(mid_content_list))
                    logger.info(f"[fast] Mid-level: {len(mid_level_chunks)} -> {len(maybe_trun_mid)} chunks")

        if low_level_chunks:
            maybe_trun_low = truncate_list_by_token_size(
                low_level_chunks,
                key=lambda x: x["content"],
                max_token_size=query_param.naive_max_token_for_text_unit,
            )
            if maybe_trun_low:
                low_content_list = []
                for c in maybe_trun_low:
                    ts_display = _format_time_span_display(c["time_span"][0].split("_")[0])
                    low_content_list.append(f"{ts_display}: {c['content']}")
                sections.append(f"-----Retrieved Low-level Captions-----\n" + "\n".join(low_content_list) + "\n")
                logger.info(f"[fast] Low-level: {len(low_level_chunks)} -> {len(maybe_trun_low)} chunks")

        retreived_chunk_context = "\n\n".join(sections) if sections else ""

    # ------------------------------------------------------------------
    # Step 2: 实体检索（KG）
    # ------------------------------------------------------------------
    entity_retrieved_segments = set()
    if need_entity:
        entity_results = await entities_vdb.query(query_for_entity, top_k=similarity_top_k)
        if entity_results:
            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
            )
            node_degrees = await asyncio.gather(
                *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_results]
            )
            node_datas = [
                {**n, "entity_name": k["entity_name"], "rank": d}
                for k, n, d in zip(entity_results, node_datas, node_degrees)
                if n is not None
            ]
            if not all(n is not None for n in node_datas):
                logger.warning("Some nodes are missing, maybe the storage is damaged")
            try:
                entity_retrieved_segments = await _find_most_related_segments_from_entities(
                    global_config["retrieval_topk_chunks"], node_datas, text_chunks_db, knowledge_graph_inst
                )
            except (TypeError, KeyError) as e:
                # _find_most_related_segments_from_entities 内部
                # text_chunks_db.get_by_id 可能返回 None，导致下标访问报错
                logger.warning(f"Entity segment retrieval failed: {e}, skipping entity retrieval")
                entity_retrieved_segments = set()
            # 时间过滤
            if deadline_sec is not None:
                entity_retrieved_segments = {
                    s for s in entity_retrieved_segments if _is_before_deadline(s, deadline_sec)
                }
            # 数量截断（[:n] 与 ego_op.py 一致）
            entity_retrieved_segments = set(list(entity_retrieved_segments)[:query_param.top_k])

    # ------------------------------------------------------------------
    # Step 3: 视觉检索
    # ------------------------------------------------------------------
    visual_retrieved_segments = set()
    if need_visual:
        segment_results = await video_segment_feature_vdb.query(query_for_visual)
        if segment_results:
            for n in segment_results:
                visual_retrieved_segments.add(n['__id__'])
            # 时间过滤
            if deadline_sec is not None:
                visual_retrieved_segments = {
                    s for s in visual_retrieved_segments if _is_before_deadline(s, deadline_sec)
                }
            visual_retrieved_segments = set(list(visual_retrieved_segments)[:query_param.top_k])

    # ------------------------------------------------------------------
    # Step 4: 合并 segments，跳过 LLM 逐个过滤
    # ------------------------------------------------------------------
    retrieved_segments = list(entity_retrieved_segments.union(visual_retrieved_segments))
    retreived_video_context = ""

    if retrieved_segments:
        retrieved_segments = sorted(retrieved_segments, key=_sort_key)
        remain_segments = retrieved_segments  # 不再做 LLM 过滤
        logger.info(f"[fast] {len(remain_segments)} segments (no LLM filtering)")

        # ------------------------------------------------------------------
        # Step 5: Caption 重构
        # ------------------------------------------------------------------
        if need_reconstruct and remain_segments:
            caption_construction_prompt = get_prompts(
                global_config=global_config, datasets_type=datasets_type,
            )["caption_reconstruction"]

            logger.info(f"[fast] Keywords: {keywords_for_caption}")
            if use_minicpm:
                caption_results = retrieved_segment_caption_minicpm(
                    caption_model, caption_tokenizer,
                    keywords_for_caption, remain_segments,
                    video_segments, caption_construction_prompt,
                    sub_service_type,
                )
            else:
                caption_results = retrieved_segment_caption(
                    caption_model, caption_tokenizer,
                    keywords_for_caption, remain_segments,
                    video_segments, caption_construction_prompt,
                    sub_service_type,
                )
        else:
            caption_results = []
            for key in remain_segments:
                ts_key = key.split("_")[0] if "_" in key else key
                if ts_key in video_segments._data:
                    caption_results.append(video_segments._data[ts_key]["content"])
                else:
                    caption_results.append("(no caption)")

        # ------------------------------------------------------------------
        # Step 6: 格式化输出
        # ------------------------------------------------------------------
        if isinstance(caption_results, dict):
            formatted = []
            for ts_key, cap in caption_results.items():
                formatted.append(f"Timestamp: {ts_key.split('_')[0]}\nContent: {cap}\n")
            caption_results_str = "\n".join(formatted)
        elif isinstance(caption_results, list):
            formatted = []
            for i, cap in enumerate(caption_results):
                if i < len(remain_segments):
                    formatted.append(f"Timestamp: {remain_segments[i].split('_')[0]}\nContent: {cap}\n")
                else:
                    formatted.append(f"Content: {cap}\n")
            caption_results_str = "\n".join(formatted)
        else:
            caption_results_str = str(caption_results)

        retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```{caption_results_str}\n```\n"

    return retreived_video_context, retreived_chunk_context
