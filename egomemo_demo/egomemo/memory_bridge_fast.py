"""
MemoryBridge 的加速版 read 方法。

通过 monkey-patch 方式替换 MemoryBridge.read，
使用 streaming_videorag_query_fast 代替原版 streaming_videorag_query。

用法:
    from egomemo.memory_bridge import MemoryBridge
    from egomemo.memory_bridge_fast import patch_memory_bridge_fast_read
    patch_memory_bridge_fast_read()  # 之后所有 MemoryBridge 实例的 read() 都会使用加速版

优化点:
    1. 去掉逐个 segment 的 LLM 过滤
    2. entity/visual query 改写 + 关键词提取三步并行
    3. VDB 查询 top_k 从 1000 降到 200，减少后续处理量
"""

import logging
from dataclasses import asdict

from .memory_bridge import MemoryBridge, _run_async

logger = logging.getLogger(__name__)


def _fast_read(self, query: str, current_time_span: str, similarity_top_k: int = 200) -> str:
    """加速版的 memory read，替换 MemoryBridge.read。

    与原版区别：
    - 使用 streaming_videorag_query_fast
    - 三个 LLM query 改写并行执行
    - 不做逐 segment 的 LLM 过滤
    """
    from videorag.streaming_op_fast import streaming_videorag_query_fast
    from videorag.base import QueryParam

    query_param = QueryParam(mode="global", top_k=20)

    # ⚠️ VideoRAG streaming_op_fast.py:162 有 bug：
    #   need_reconstruct = args is None or getattr(args, 'reconstruct_caption', reconstruct_caption)
    # 因为 `args is None` 短路成 True，传 reconstruct_caption=False + args=None 时
    # 还是会触发 caption_reconstruction（每个 search 多花 ~50s）。
    # 解决：构造一个最小 args 对象，让 args is None 这条短路失效，从而真正读到
    # reconstruct_caption=False 的属性值。
    class _Args:
        reconstruct_caption = False
        entity_retrieval = True
        visual_retrieval = True
    fake_args = _Args()

    try:
        result = _run_async(
            streaming_videorag_query_fast(
                query=query,
                time_key=current_time_span,
                service_type="",
                sub_service_type="",
                datasets_type=self._datasets_type,
                entities_vdb=self._vg.entities_vdb,
                text_chunks_db=self._vg.text_chunks,
                chunks_vdb=self._vg.chunks_vdb,
                video_segments=self._vg.video_segments,
                video_segment_feature_vdb=self._vg.video_segment_feature_vdb,
                knowledge_graph_inst=self._vg.chunk_entity_relation_graph,
                caption_model=getattr(self._vg, "caption_model", None),
                caption_tokenizer=getattr(self._vg, "caption_processor", None)
                or getattr(self._vg, "processor", None),
                query_param=query_param,
                global_config=asdict(self._vg),
                reconstruct_caption=False,
                use_minicpm=False,
                args=fake_args,
                ori_query=query,
                similarity_top_k=similarity_top_k,
            )
        )

        if isinstance(result, tuple):
            return "\n".join(str(r) for r in result if r)
        return str(result) if result else "(no memory found)"

    except Exception as e:
        import traceback
        logger.error(f"[fast] Memory retrieval failed: {e}\n{traceback.format_exc()}")
        return f"(memory retrieval error: {e})"


def _format_recent_history_before(self, current_time_span: str, n: int = 3) -> str:
    """返回当前 caption 之前的最近 n 条 caption，排除当前 caption 本身。

    原版 format_recent_history 直接取 self._captions[-n:]，
    在 sequential 模式下所有 caption 已预先构建完毕，
    导致 recent_history 可能包含当前甚至之后的 caption。

    此方法根据 current_time_span 做严格过滤：
    只返回 time_span 在 current_time_span 之前的 caption。
    """
    if not self._captions:
        return "(no prior observations)"

    prior = [
        entry for entry in self._captions
        if entry.get("time_span", "") < current_time_span
    ]

    recent = prior[-n:] if prior else []
    if not recent:
        return "(no prior observations)"

    parts = []
    for entry in recent:
        ts = entry.get("time_span", "?")
        cap = entry.get("caption", "")
        parts.append(f"[{ts}] {cap}")
    return "\n\n".join(parts)


def _patch_streaming_op_fast_none_safe():
    """修复 streaming_op_fast 在 race condition 下报 NoneType 的问题。

    根因：VideoRAG ainsert_streaming_caption 里
        chunks_vdb.upsert(...)        # 1. VDB 先写
        await streaming_extract_entities(...)  # 2. ~10s 的 LLM call
        text_chunks.upsert(...)       # 3. KV 后写
    在 1 和 3 之间的窗口内，主线程的 search 会让 chunks_vdb.query 命中刚 upsert
    的 vdb id，然后 text_chunks.get_by_ids 在 KV 里找不到 → 返回 None。

    streaming_op_fast.py:209-212 对 chunks 做时间过滤时会调
    `_get_chunk_time_span(c)`，c=None → `if "time_span" in chunk` → TypeError。

    本 patch 把 `_get_chunk_time_span` monkey-patch 成 None-safe 版本：
    chunk is None 时直接返回 None，让外层列表推导式自然过滤掉。
    其它 chunk 正常处理。
    """
    try:
        from videorag import streaming_op_fast as _sof
    except Exception as e:
        logger.warning(f"[fast] cannot import streaming_op_fast: {e}")
        return

    _orig = getattr(_sof, "_get_chunk_time_span", None)
    if _orig is None:
        logger.warning("[fast] streaming_op_fast._get_chunk_time_span not found")
        return
    if getattr(_orig, "_none_safe_patched", False):
        return  # 已 patch，避免重复

    def _none_safe(chunk):
        if chunk is None:
            return None
        return _orig(chunk)

    _none_safe._none_safe_patched = True
    _sof._get_chunk_time_span = _none_safe
    logger.info(
        "[fast] streaming_op_fast._get_chunk_time_span patched: None-safe "
        "(skips chunks where vdb hit but KV upsert hasn't completed yet)"
    )

    # 保险：deadline_sec 为 None 时上游不会过滤，line 219 c.get("type") 仍会炸。
    # 包一层 streaming_videorag_query_fast 在调用它之前对 text_chunks_db.get_by_ids
    # 的返回值做 None 过滤。但更轻量的做法：替换 text_chunks_db.get_by_ids 自身
    # 让 None 在源头被丢掉。考虑到 get_by_ids 在 VideoRAG 内部还有别的调用方
    # （依赖 None 占位以保持 ids/results 的位置对齐），这里不动 get_by_ids，
    # 只处理 streaming_op_fast 这一条路径——通过 _get_chunk_time_span 过滤已经
    # 覆盖 deadline_sec is not None 的常见路径，对 deadline_sec=None 的退化路径
    # 不再加额外保护，留给上游 try/except 兜底。


def patch_memory_bridge_fast_read():
    """将 MemoryBridge.read 替换为加速版本，
    并添加 format_recent_history_before 方法。"""
    MemoryBridge.read = _fast_read
    MemoryBridge.format_recent_history_before = _format_recent_history_before
    _patch_streaming_op_fast_none_safe()
    logger.info("[fast] MemoryBridge.read patched with fast version")
    logger.info("[fast] MemoryBridge.format_recent_history_before added")
