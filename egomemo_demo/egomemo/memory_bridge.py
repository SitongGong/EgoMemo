"""
Module 5: External Memory Bridge.

Adapter that wraps the existing VideoGraphSeparated infrastructure,
providing simplified interfaces for the streaming pipeline:
  - build_step(): Generate caption + insert into all 3 memory stores (single step)
  - build_step_caption_only() + flush_pending_memory(): Batch-optimized build
  - read(): Retrieve from long-term memory (text + graph + visual)
  - get_recent_captions(): Short-term context

NOTE: This module only handles VIDEO CONTENT memory (captions, KG, visual embeddings).
Interaction history (question answers, proactive records) is managed by WorkingMemory.

Uses only single-level captions (10-second windows),
simplified from the original 3-level (hour/min/sec) hierarchy.
"""

import asyncio
import json
import logging
import os
import sys
import time as _time
import tiktoken
from dataclasses import asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TimingStats:
    """累积计时统计器"""
    def __init__(self):
        self.caption_times: List[float] = []      # 每次 caption 生成耗时
        self.kg_insert_times: List[float] = []     # 每次 KG 插入耗时（ainsert_streaming_caption）
        self.visual_embed_times: List[float] = []  # 每次 visual embedding 耗时
        self.store_frames_times: List[float] = []  # 每次 store frames 耗时
        self.batch_entity_times: List[float] = []  # batch entity extraction 耗时
        self.batch_visual_times: List[float] = []  # batch visual embedding 耗时

    def summary(self) -> Dict:
        def _stat(arr):
            if not arr:
                return {"count": 0, "total": 0, "avg": 0, "min": 0, "max": 0}
            return {
                "count": len(arr),
                "total": round(sum(arr), 3),
                "avg": round(sum(arr) / len(arr), 3),
                "min": round(min(arr), 3),
                "max": round(max(arr), 3),
            }
        return {
            "caption_generation": _stat(self.caption_times),
            "kg_insert": _stat(self.kg_insert_times),
            "visual_embedding": _stat(self.visual_embed_times),
            "store_frames": _stat(self.store_frames_times),
            "batch_entity_extraction": _stat(self.batch_entity_times),
            "batch_visual_embedding": _stat(self.batch_visual_times),
        }

    def log_summary(self):
        s = self.summary()
        logger.info("=" * 60)
        logger.info("MEMORY BRIDGE TIMING SUMMARY")
        logger.info("=" * 60)
        for name, stat in s.items():
            if stat["count"] > 0:
                logger.info(f"  {name:30s} | count={stat['count']:3d} | total={stat['total']:7.2f}s | avg={stat['avg']:6.3f}s | min={stat['min']:6.3f}s | max={stat['max']:6.3f}s")
        logger.info("=" * 60)

# Add VideoRAG to sys.path (append, not insert — keep EgoServe-RL's egomemo/ higher priority)
# 开源布局下 videorag 与 egomemo 平级，默认指向上级目录；可用 VIDEORAG_ROOT 覆盖
_VIDEORAG_ROOT = os.environ.get("VIDEORAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _VIDEORAG_ROOT not in sys.path:
    sys.path.append(_VIDEORAG_ROOT)

from videorag.egograph_retrieval_optimize_ import VideoGraphSeparated
from videorag.streaming_op import streaming_videorag_query
from videorag.ego_op import batch_extract_entities
from videorag.base import QueryParam
from videorag._llm import LLMConfig
from videorag._utils import compute_mdhash_id

from .prompt_templates import CAPTION_SYSTEM_PROMPT, MIN_CAPTION_SYSTEM_PROMPT, HOUR_CAPTION_SYSTEM_PROMPT
from .entity_extraction import extract_entities_simple


def _run_async(coro):
    """Run an async coroutine from synchronous code, handling the case
    where an event loop may or may not already be running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside a running loop (e.g. uvicorn) — use nest_asyncio
        import nest_asyncio
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    else:
        # No running loop — safe to use asyncio.run or create one
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # Fallback: create a new loop
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()


def _seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extract_last_caption_json(text: str) -> Optional[str]:
    """从含 reasoning + 多段草稿 JSON 的模型输出中，提取**最后一个**
    同时含 "caption" 和 "frames" 键的平衡 {...} 块。

    背景：ego_prompt_ 的 simple_second_caption 较复杂，Qwen3.5-4B 经常先输出
    长段无 <think> 标签的思考（其中可能写一段草稿 / 示例 JSON），最后才给真正
    合规的 JSON。直接 json.loads 第一个匹配的 {...} 会把草稿后面的合法 JSON
    误读为 "Extra data"。
    """
    if not text:
        return None
    candidates = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append((start, i + 1))
                    start = -1
    for s, e in reversed(candidates):
        block = text[s:e]
        if '"caption"' in block and '"frames"' in block:
            return block
    return None


def _make_time_span(day: int, start_seconds: float, end_seconds: float) -> str:
    """Create a time_span string in the format: day-HH:MM:SS-HH:MM:SS.

    This matches the format expected by streaming_videorag_query's parse_time_span:
        day-START_HH:MM:SS-END_HH:MM:SS
    e.g. "1-00:00:05-00:00:15" for day 1, 5s to 15s.
    """
    start_str = _seconds_to_hhmmss(start_seconds)
    end_str = _seconds_to_hhmmss(end_seconds)
    return f"{day}-{start_str}-{end_str}"


class MemoryBridge:
    """Adapter over VideoGraphSeparated for the streaming pipeline.

    Supports 3-level caption hierarchy (second / minute / hour),
    matching egograph's processing structure.
    """

    def __init__(
        self,
        videorag_instance: VideoGraphSeparated,
        llm_config: LLMConfig,
        datasets_type: str = "egomemo",
        caption_window_seconds: int = 10,
        enable_multi_level_caption: bool = False,
        caption_window_minutes: float = 5.0,
        caption_window_hours: float = 1.0,
        gap_threshold_seconds: float = 60.0,
        kg_extraction_mode: str = "simple",
    ):
        self._vg = videorag_instance
        self._llm_config = llm_config
        self._datasets_type = datasets_type

        # Config
        self._kg_extraction_mode = kg_extraction_mode  # "simple" or "full"
        self._enable_multi_level = enable_multi_level_caption
        self._window_seconds = caption_window_seconds
        self._window_minutes = caption_window_minutes
        self._window_hours = caption_window_hours
        self._gap_threshold = gap_threshold_seconds

        # Accumulated captions (all levels)
        self._captions: List[Dict] = []          # second level
        self._min_captions: List[Dict] = []      # minute level
        self._hour_captions: List[Dict] = []     # hour level

        # Minute window: accumulate second captions
        self._min_window_second_captions: List[Dict] = []
        self._min_window_start_timestamp: Optional[float] = None

        # Hour window: accumulate minute captions
        self._hour_window_min_captions: List[Dict] = []
        self._hour_window_start_timestamp: Optional[float] = None

        # Pending segments for batch entity extraction (sequential mode)
        self._pending_segments: List[Dict] = []
        self._pending_visual_data: List[Dict] = []

        # Track current day (for time_span format)
        self._current_day: int = 1

        # 计时统计
        self.timing = TimingStats()

    @property
    def videorag(self) -> VideoGraphSeparated:
        return self._vg

    # ==================================================================
    # Single-step build (used in async pipeline mode)
    # ==================================================================
    def build_step(
        self,
        frames_base64: List[str],
        time_span: str,
        timestamp: float,
        frame_times: Optional[List[Dict]] = None,
    ) -> Dict:
        """Process one step's frames through the memory pipeline.

        1. Generate caption using the loaded caption model
        2. Insert caption into text chunks + knowledge graph (via ainsert_streaming_caption)
        3. Insert visual embeddings (via upsert_video_segment)

        Args:
            frames_base64: List of base64-encoded frame images
            time_span: Time span string (e.g., "1-00:00:05-00:00:15")
            timestamp: Video timestamp in seconds
            frame_times: Per-frame time ranges [{"start": float, "end": float}, ...]

        Returns:
            Dict with caption info
        """
        # Step 1: Generate caption (raw model output)
        t0 = _time.time()
        caption_text = self._generate_caption(frames_base64)
        t1 = _time.time()
        self.timing.caption_times.append(t1 - t0)
        logger.info(f"[Timing] Caption generation: {t1-t0:.3f}s")

        # Step 2: 转换为 egograph 标准 caption_dict 格式
        caption_dict = self._build_caption_dict(caption_text, frame_times)
        content_str = json.dumps(caption_dict)

        # Step 3: Store frames in video_segments（与 egograph 格式一致）
        t2 = _time.time()
        self._store_frames(time_span, caption_dict, frames_base64)
        t3 = _time.time()
        self.timing.store_frames_times.append(t3 - t2)

        # Step 4 / Step 5: 写入 text_chunks + chunks_vdb + 实体提取 + 知识图谱
        # 注意：simple 与 full 模式对 text_chunks/chunks_vdb 的处理不同——
        #   - simple: extract_entities_simple 不动 text_chunks，所以这里必须先写。
        #   - full:   ainsert_streaming_caption 自己会 upsert text_chunks/chunks_vdb，
        #             如果这里先写了，filter_keys 会把它判定为"已有"，然后整个函数
        #             提前 return（"All chunks are already in the storage"），
        #             连 entity_extraction 都不会跑。所以 full 模式必须跳过预写。
        chunk_key = compute_mdhash_id(f"{time_span}_0", prefix="chunk-")
        chunk_data = {
            "content": content_str,
            "chunk_order_index": 0,
            "time_span": [f"{time_span}_0"],
            "type": "second",
        }

        t4 = _time.time()
        if self._kg_extraction_mode == "simple":
            # simple 路径：先把 chunk 写进文本/向量库，再做实体提取
            try:
                _run_async(self._vg.text_chunks.upsert({chunk_key: chunk_data}))
                if self._vg.enable_naive_rag and self._vg.chunks_vdb is not None:
                    _run_async(self._vg.chunks_vdb.upsert({chunk_key: chunk_data}))
            except Exception as e:
                logger.warning(f"Failed to upsert text chunks: {e}")
            try:
                _run_async(
                    extract_entities_simple(
                        chunk_key=time_span,
                        chunk_content=content_str,
                        knowledge_graph_inst=self._vg.chunk_entity_relation_graph,
                        entity_vdb=self._vg.entities_vdb,
                        global_config=asdict(self._vg),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed simple entity extraction: {e}")
        else:
            # full 路径：交给 VideoRAG 的 ainsert_streaming_caption 一站处理
            # （它会 filter_keys 去重 → 实体提取 → 最后 upsert text_chunks/chunks_vdb）
            segment_data = {
                time_span: {
                    "content": content_str,
                    "type": "second",
                }
            }
            try:
                _run_async(
                    self._vg.ainsert_streaming_caption(
                        segment_data,
                        datasets_type=self._datasets_type,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to insert streaming caption: {e}")
        t5 = _time.time()
        self.timing.kg_insert_times.append(t5 - t4)
        logger.info(f"[Timing] KG extraction ({self._kg_extraction_mode}): {t5-t4:.3f}s")

        # Step 5: Insert visual embeddings
        t6 = _time.time()
        self._insert_visual_embedding(time_span, frames_base64)
        t7 = _time.time()
        self.timing.visual_embed_times.append(t7 - t6)
        logger.info(f"[Timing] Visual embedding: {t7-t6:.3f}s")

        # Record second-level caption
        caption_entry = {
            "time_span": time_span,
            "caption": content_str,
            "timestamp": timestamp,
        }
        self._captions.append(caption_entry)

        # Step 6: 多级 caption 累积与触发（minute / hour）
        end_time = timestamp
        if frame_times:
            end_time = frame_times[-1]["end"]
        if self._enable_multi_level:
            self._accumulate_multi_level(time_span, content_str, timestamp, end_time)

        return caption_entry

    # ==================================================================
    # Batch-optimized build (used in sequential pipeline mode)
    # ==================================================================
    def build_step_caption_only(
        self,
        frames_base64: List[str],
        time_span: str,
        timestamp: float,
        frame_times: Optional[List[Dict]] = None,
    ) -> Dict:
        """Phase 1 of batch build: generate caption + store frames.

        Does NOT do entity extraction or visual embedding yet.
        Those are deferred to flush_pending_memory() for batch processing.

        Args:
            frames_base64: List of base64-encoded frame images
            time_span: Time span string
            timestamp: Video timestamp in seconds
            frame_times: Per-frame time ranges [{"start": float, "end": float}, ...]

        Returns:
            Dict with caption info
        """
        # Generate caption (raw model output)
        t0 = _time.time()
        caption_text = self._generate_caption(frames_base64)
        t1 = _time.time()
        self.timing.caption_times.append(t1 - t0)
        logger.info(f"[Timing] Caption generation: {t1-t0:.3f}s")

        # 转换为 egograph 标准 caption_dict 格式
        caption_dict = self._build_caption_dict(caption_text, frame_times)
        content_str = json.dumps(caption_dict)

        # Store frames in video_segments（与 egograph 格式一致）
        t2 = _time.time()
        self._store_frames(time_span, caption_dict, frames_base64)
        t3 = _time.time()
        self.timing.store_frames_times.append(t3 - t2)

        # Accumulate for batch processing
        segment_data = {
            time_span: {
                "content": content_str,
                "type": "second",
            }
        }
        self._pending_segments.append(segment_data)
        self._pending_visual_data.append({
            "time_span": time_span,
            "frames": frames_base64,
        })

        # Record second-level caption
        caption_entry = {
            "time_span": time_span,
            "caption": content_str,
            "timestamp": timestamp,
        }
        self._captions.append(caption_entry)

        # 多级 caption 累积与触发（minute / hour）
        end_time = timestamp
        if frame_times:
            end_time = frame_times[-1]["end"]
        if self._enable_multi_level:
            self._accumulate_multi_level(time_span, content_str, timestamp, end_time)

        return caption_entry

    def flush_pending_memory(self) -> None:
        """Phase 2 of batch build: batch entity extraction + visual embedding.

        Processes all pending segments accumulated via build_step_caption_only().
        Uses batch_extract_entities for parallel KG construction, mirroring
        the pattern in egograph_retrieval_optimize_.py's batch entity extraction.
        """
        if not self._pending_segments:
            logger.info("No pending segments to flush")
            return

        n = len(self._pending_segments)
        logger.info(f"[Batch Memory Build] Flushing {n} pending segments...")

        # --- Phase 2a: Batch entity extraction (parallel) ---
        t0 = _time.time()
        self._batch_entity_extraction()
        t1 = _time.time()
        self.timing.batch_entity_times.append(t1 - t0)
        logger.info(f"[Timing] Batch entity extraction: {t1-t0:.3f}s ({n} segments)")

        # --- Phase 2b: Batch visual embedding ---
        t2 = _time.time()
        self._batch_visual_embedding()
        t3 = _time.time()
        self.timing.batch_visual_times.append(t3 - t2)
        logger.info(f"[Timing] Batch visual embedding: {t3-t2:.3f}s ({n} segments)")

        # Clear pending
        self._pending_segments.clear()
        self._pending_visual_data.clear()
        logger.info(f"[Batch Memory Build] Completed for {n} segments")

    def _batch_entity_extraction(self) -> None:
        """Batch extract entities from all pending segments.

        Supports two modes via self._kg_extraction_mode:
        - "simple": 简化版 demo prompt，逐个 chunk 调用 extract_entities_simple
        - "full": 原始 VideoRAG 完整 batch 提取（batch_extract_entities）
        """
        if self._kg_extraction_mode == "simple":
            self._batch_entity_extraction_simple()
        else:
            self._batch_entity_extraction_full()

    def _batch_entity_extraction_simple(self) -> None:
        """简化版：逐个 chunk 调用 extract_entities_simple。"""
        for segment_data in self._pending_segments:
            for time_span, seg in segment_data.items():
                content_str = seg["content"].strip()
                try:
                    _run_async(
                        extract_entities_simple(
                            chunk_key=time_span,
                            chunk_content=content_str,
                            knowledge_graph_inst=self._vg.chunk_entity_relation_graph,
                            entity_vdb=self._vg.entities_vdb,
                            global_config=asdict(self._vg),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Simple entity extraction failed for {time_span}: {e}")

                # 同时写入 text_chunks 和 chunks_vdb（用于文本检索）
                chunk_key = compute_mdhash_id(f"{time_span}_0", prefix="chunk-")
                chunk_data = {
                    "content": content_str,
                    "chunk_order_index": 0,
                    "time_span": [f"{time_span}_0"],
                    "type": seg.get("type", "second"),
                }
                try:
                    _run_async(self._vg.text_chunks.upsert({chunk_key: chunk_data}))
                    if self._vg.enable_naive_rag and self._vg.chunks_vdb is not None:
                        _run_async(self._vg.chunks_vdb.upsert({chunk_key: chunk_data}))
                except Exception as e:
                    logger.warning(f"text_chunks upsert failed for {time_span}: {e}")

    def _batch_entity_extraction_full(self) -> None:
        """原始 VideoRAG 完整 batch 提取。"""
        try:
            _run_async(self._vg._insert_start())
        except Exception as e:
            logger.warning(f"_insert_start failed: {e}")

        all_inserting_chunks = {}
        ENCODER = tiktoken.encoding_for_model("gpt-4o")

        for segment_data in self._pending_segments:
            for time_span, seg in segment_data.items():
                content_str = seg["content"].strip()
                tokens = ENCODER.encode_batch([content_str], num_threads=16)[0]

                caption_dict = {
                    "tokens": len(tokens),
                    "content": content_str,
                    "chunk_order_index": 0,
                    "time_span": [f"{time_span}_0"],
                    "type": seg.get("type", "second"),
                }

                chunk_key = compute_mdhash_id(caption_dict["time_span"][0], prefix="chunk-")
                all_inserting_chunks[chunk_key] = caption_dict

        if not all_inserting_chunks:
            logger.info("No chunks to insert")
            self._try_insert_done()
            return

        # Filter out already-existing chunks
        try:
            add_chunk_keys = _run_async(
                self._vg.text_chunks.filter_keys(list(all_inserting_chunks.keys()))
            )
            all_inserting_chunks = {
                k: v for k, v in all_inserting_chunks.items() if k in add_chunk_keys
            }
        except Exception as e:
            logger.warning(f"filter_keys failed: {e}")

        if not all_inserting_chunks:
            logger.info("All chunks already exist in storage")
            self._try_insert_done()
            return

        logger.info(f"[Batch Entity Extraction] Inserting {len(all_inserting_chunks)} new chunks")

        # Upsert to chunks_vdb (naive RAG)
        if self._vg.enable_naive_rag and self._vg.chunks_vdb is not None:
            try:
                _run_async(self._vg.chunks_vdb.upsert(all_inserting_chunks))
            except Exception as e:
                logger.warning(f"chunks_vdb upsert failed: {e}")

        # Batch parallel entity extraction
        try:
            maybe_new_kg, _, _ = _run_async(
                batch_extract_entities(
                    all_inserting_chunks,
                    self._vg.chunk_entity_relation_graph,
                    self._vg.entities_vdb,
                    asdict(self._vg),
                    datasets_type=self._datasets_type,
                )
            )
            if maybe_new_kg is not None:
                self._vg.chunk_entity_relation_graph = maybe_new_kg
            else:
                logger.warning("No new entities found in batch extraction")
        except Exception as e:
            logger.warning(f"batch_extract_entities failed: {e}")

        # Upsert to text_chunks
        try:
            _run_async(self._vg.text_chunks.upsert(all_inserting_chunks))
        except Exception as e:
            logger.warning(f"text_chunks upsert failed: {e}")

        self._try_insert_done()

    def _batch_visual_embedding(self) -> None:
        """Batch insert visual embeddings for all pending segments."""
        if self._vg.video_segment_feature_vdb is None:
            return

        segments_data = []
        for item in self._pending_visual_data:
            if item["frames"]:
                segments_data.append((item["time_span"], item["frames"]))

        if not segments_data:
            return

        logger.info(f"[Batch Visual Embedding] Processing {len(segments_data)} segments")
        try:
            _run_async(
                self._vg.video_segment_feature_vdb.upsert_video_segment_batch(
                    segments_data,
                    encode_mode="frame",
                )
            )
        except Exception as e:
            logger.warning(f"Batch visual embedding failed: {e}")

    # ==================================================================
    # Memory read / write
    # ==================================================================
    def read(self, query: str, current_time_span: str) -> str:
        """Retrieve from memory using the existing streaming_videorag_query.

        Combines text similarity, knowledge graph, and visual retrieval.
        """
        query_param = QueryParam(mode="global", top_k=20)

        try:
            result = _run_async(
                streaming_videorag_query(
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
                    # rebuttal 测试：见 memory_bridge_fast.read 注释
                    reconstruct_caption=False,
                    use_minicpm=False,
                )
            )

            if isinstance(result, tuple):
                return "\n".join(str(r) for r in result if r)
            return str(result) if result else "(no memory found)"

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return f"(memory retrieval error: {e})"

    # ==================================================================
    # Caption accessors
    # ==================================================================
    def get_recent_captions(self, n: int = 3) -> List[Dict]:
        """Return last n captions for short-term context in the prompt."""
        return self._captions[-n:] if self._captions else []

    def get_all_captions(self) -> List[Dict]:
        """Return all accumulated captions."""
        return list(self._captions)

    def format_recent_history(self, n: int = 3) -> str:
        """Format recent captions as a readable string for prompt injection."""
        recent = self.get_recent_captions(n)
        if not recent:
            return "(no prior observations)"

        parts = []
        for entry in recent:
            ts = entry.get("time_span", "?")
            cap = entry.get("caption", "")
            parts.append(f"[{ts}] {cap}")
        return "\n\n".join(parts)

    # ==================================================================
    # Internal helpers
    # ==================================================================
    @staticmethod
    def _resize_frames(frames_base64: List[str], max_size: int = 448) -> List[str]:
        """将 base64 帧 resize 到 max_size x max_size，减少 vision token 加速推理。"""
        import io as _io
        import base64 as _b64
        from PIL import Image

        resized = []
        for fb64 in frames_base64:
            try:
                img = Image.open(_io.BytesIO(_b64.b64decode(fb64)))
                # 等比缩放到 max_size
                img.thumbnail((max_size, max_size), Image.LANCZOS)
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                resized.append(_b64.b64encode(buf.getvalue()).decode())
            except Exception:
                resized.append(fb64)  # resize 失败则用原图
        return resized

    def _generate_caption(self, frames_base64: List[str]) -> str:
        """Generate a caption for frames using the loaded MLLM model.

        Returns the raw JSON string (for storage), after validating it
        contains the expected {"caption": "...", "frames": {...}} structure.
        Retries up to 3 times if the output is not valid JSON.
        """
        if not frames_base64:
            return "(no frames available)"

        # Resize 帧到 448x448 加速 caption 生成
        resized_frames = self._resize_frames(frames_base64, max_size=448)

        # 在 prompt 末尾要求直接输出 JSON，跳过思考过程以加速
        caption_prompt = CAPTION_SYSTEM_PROMPT + "\nIMPORTANT: Output the JSON directly. Do NOT include any thinking, reasoning, or explanation. Just the JSON object."

        max_retries = 3
        text = ""
        for attempt in range(max_retries):
            raw = ""
            try:
                raw = self._vg.mllm_response(
                    self._vg.video_llm,
                    self._vg.processor,
                    caption_prompt,
                    None,
                    base64_frames=resized_frames,
                    max_new_tokens=2048,
                    has_image=True,
                )
                if not raw:
                    continue

                # ego_prompt_ 的 simple_second_caption 较复杂，Qwen3.5-4B 常在
                # 输出最终 JSON 之前先生成大段 reasoning（无 <think> 标签），
                # 中间还可能写一段示例 / 草稿 JSON。直接拿最前面那个 {...} 会
                # 撞到草稿 + 后面真正的 JSON 拼在一起，触发 "Extra data" 错误。
                # 所以：扫描所有平衡 {...} 块，取**最后一个**含 "caption" +
                # "frames" 的块作为最终 JSON 候选。
                text = _extract_last_caption_json(raw)
                if text is None:
                    logger.warning(
                        f"Caption attempt {attempt+1}/{max_retries}: "
                        f"no balanced JSON block with caption+frames found: "
                        f"{raw.strip()[:200]}..."
                    )
                    continue

                # Validate JSON structure
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "frames" in parsed and isinstance(parsed["frames"], dict):
                    return text
                else:
                    logger.warning(
                        f"Caption attempt {attempt+1}/{max_retries}: "
                        f"JSON missing 'frames' key"
                    )
                    continue

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Caption attempt {attempt+1}/{max_retries}: "
                    f"JSON parse error: {e}"
                )
                continue
            except Exception as e:
                logger.error(f"Caption generation failed: {e}")
                return f"(caption generation error: {e})"

        logger.error(f"Caption failed after {max_retries} attempts")
        return '{"caption": "(caption generation failed)", "frames": {}}'

    def _build_caption_dict(
        self, caption_text: str, frame_times: Optional[List[Dict]] = None,
    ) -> Dict:
        """将模型输出的 caption JSON 转换为 egograph 标准的 caption_dict 格式。

        输入格式（模型输出）:
            {"caption": "全局描述", "frames": {"0": "帧0描述", "1": "帧1描述", ...}}

        输出格式（egograph 标准）:
            {"dense_caption": {"DAY1-HH:MM:SS-HH:MM:SS": "帧caption", ...}, "description": "全局caption"}

        每帧 caption 对应一个独立的 time_span，例如:
            整体 time_span 为 1-00:00:00-00:00:10，则第一帧为 DAY1-00:00:00-00:00:02
        """
        # 解析模型输出
        parsed = {}
        try:
            parsed = json.loads(caption_text) if isinstance(caption_text, str) else {}
        except (json.JSONDecodeError, ValueError):
            pass

        global_caption = parsed.get("caption", caption_text if isinstance(caption_text, str) else "")
        frame_captions = parsed.get("frames", {})

        # 构建 dense_caption
        dense_caption = {}
        if frame_times:
            for i, ft in enumerate(frame_times):
                frame_cap = frame_captions.get(str(i), "")
                if frame_cap:
                    start_str = _seconds_to_hhmmss(ft["start"])
                    end_str = _seconds_to_hhmmss(ft["end"])
                    caption_key = f"DAY{self._current_day}-{start_str}-{end_str}"
                    dense_caption[caption_key] = frame_cap

        return {
            "dense_caption": dense_caption,
            "description": global_caption,
        }

    def _store_frames(
        self, time_span: str, caption_dict: Dict, frames_base64: List[str],
    ) -> None:
        """Store frames in video_segments, 与 egograph 格式完全一致。

        egograph 格式:
            video_segments.upsert({time_span: {
                "content": json.dumps(caption_dict),
                "video_frames": frames_list,
                "type": "second"
            }})
        """
        try:
            _run_async(
                self._vg.video_segments.upsert(
                    {
                        time_span: {
                            "content": json.dumps(caption_dict),
                            "video_frames": frames_base64,
                            "type": "second",
                        }
                    }
                )
            )
        except Exception as e:
            logger.warning(f"Failed to store frames in video_segments: {e}")

    def _insert_visual_embedding(self, time_span: str, frames_base64: List[str]) -> None:
        """Insert visual embeddings for a single step."""
        try:
            if self._vg.video_segment_feature_vdb is not None and frames_base64:
                _run_async(
                    self._vg.video_segment_feature_vdb.upsert_video_segment(
                        time_span=time_span,
                        video_frames=frames_base64,
                        encode_mode="joint",
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to insert video segment embeddings: {e}")

    def _try_insert_done(self) -> None:
        """Call _insert_done to save graph storage."""
        try:
            _run_async(self._vg._insert_done())
        except Exception as e:
            logger.warning(f"_insert_done failed: {e}")

    # ==================================================================
    # Multi-level caption (minute / hour)
    # ==================================================================
    def _accumulate_multi_level(
        self, time_span: str, content_str: str,
        start_timestamp: float, end_timestamp: float,
    ) -> None:
        """累积 second caption 到 minute 窗口，minute caption 到 hour 窗口。

        仅在 enable_multi_level_caption=True 时由外部 config 控制是否调用。
        但此方法本身始终执行累积逻辑，由调用方决定是否调用。
        """
        # 累积到 minute 窗口
        if self._min_window_start_timestamp is None:
            self._min_window_start_timestamp = start_timestamp

        self._min_window_second_captions.append({
            "time_span": time_span,
            "caption": content_str,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
        })

        # 检查 minute 窗口是否满
        min_duration = end_timestamp - self._min_window_start_timestamp
        if min_duration >= self._window_minutes * 60:
            self._flush_minute_window(end_timestamp)

    def _flush_minute_window(self, last_timestamp: float) -> None:
        """生成 minute-level caption 并存储，与 egograph 格式一致。"""
        if not self._min_window_second_captions:
            return

        start_ts = self._min_window_start_timestamp
        end_ts = self._min_window_second_captions[-1]["end_timestamp"]
        min_time_span = _make_time_span(self._current_day, start_ts, end_ts)

        # 拼接所有 second caption 作为输入
        caption_text = "\n".join([
            f"[{item['time_span']}]: {item['caption']}"
            for item in self._min_window_second_captions
        ])

        # 生成 minute caption（纯文本，不需要图片）
        prompt = MIN_CAPTION_SYSTEM_PROMPT.format(
            window_seconds=self._window_seconds,
            window_minutes=self._window_minutes,
        ) + f"\n\nCaptions:\n{caption_text}"

        try:
            min_caption = self._vg.mllm_response(
                self._vg.video_llm,
                self._vg.processor,
                prompt,
                None,
                base64_frames=None,
                max_new_tokens=2048,
                has_image=False,
            )
        except Exception as e:
            logger.warning(f"Minute caption generation failed: {e}")
            min_caption = "(minute caption generation failed)"

        logger.info(f"Generated minute caption: {min_time_span} "
                     f"({len(self._min_window_second_captions)} second captions)")

        # 存入 video_segments（与 egograph 一致）
        sub_windows = [item["time_span"] for item in self._min_window_second_captions]
        try:
            _run_async(self._vg.video_segments.upsert({
                min_time_span: {
                    "content": min_caption,
                    "sub_window_captions": sub_windows,
                    "type": "minute",
                }
            }))
        except Exception as e:
            logger.warning(f"Failed to store minute caption: {e}")

        # 插入 text chunks + KG
        try:
            _run_async(self._vg.ainsert_streaming_caption(
                {min_time_span: {
                    "content": min_caption,
                    "sub_window_captions": sub_windows,
                    "type": "minute",
                }},
                datasets_type=self._datasets_type,
            ))
        except Exception as e:
            logger.warning(f"Failed to insert minute caption: {e}")

        self._min_captions.append({
            "time_span": min_time_span,
            "caption": min_caption,
            "timestamp": last_timestamp,
        })

        # 累积到 hour 窗口
        if self._hour_window_start_timestamp is None:
            self._hour_window_start_timestamp = start_ts

        self._hour_window_min_captions.append({
            "time_span": min_time_span,
            "caption": min_caption,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
        })

        # 检查 hour 窗口是否满
        hour_duration = end_ts - self._hour_window_start_timestamp
        if hour_duration >= self._window_hours * 3600:
            self._flush_hour_window(last_timestamp)

        # 重置 minute 窗口
        self._min_window_second_captions = []
        self._min_window_start_timestamp = None

    def _flush_hour_window(self, last_timestamp: float) -> None:
        """生成 hour-level caption 并存储，与 egograph 格式一致。"""
        if not self._hour_window_min_captions:
            return

        start_ts = self._hour_window_start_timestamp
        end_ts = self._hour_window_min_captions[-1]["end_timestamp"]
        hour_time_span = _make_time_span(self._current_day, start_ts, end_ts)

        # 拼接所有 minute caption 作为输入
        min_caption_text = "\n".join([
            f"[{item['time_span']}]: {item['caption']}"
            for item in self._hour_window_min_captions
        ])

        prompt = HOUR_CAPTION_SYSTEM_PROMPT.format(
            window_minutes=self._window_minutes,
            window_hours=self._window_hours,
        ) + f"\n\nCaptions:\n{min_caption_text}"

        try:
            hour_caption = self._vg.mllm_response(
                self._vg.video_llm,
                self._vg.processor,
                prompt,
                None,
                base64_frames=None,
                max_new_tokens=2048,
                has_image=False,
            )
        except Exception as e:
            logger.warning(f"Hour caption generation failed: {e}")
            hour_caption = "(hour caption generation failed)"

        logger.info(f"Generated hour caption: {hour_time_span} "
                     f"({len(self._hour_window_min_captions)} minute captions)")

        # 存入 video_segments
        sub_windows = [item["time_span"] for item in self._hour_window_min_captions]
        try:
            _run_async(self._vg.video_segments.upsert({
                hour_time_span: {
                    "content": hour_caption,
                    "sub_window_captions": sub_windows,
                    "type": "hour",
                }
            }))
        except Exception as e:
            logger.warning(f"Failed to store hour caption: {e}")

        # 插入 text chunks + KG
        try:
            _run_async(self._vg.ainsert_streaming_caption(
                {hour_time_span: {
                    "content": hour_caption,
                    "sub_window_captions": sub_windows,
                    "type": "hour",
                }},
                datasets_type=self._datasets_type,
            ))
        except Exception as e:
            logger.warning(f"Failed to insert hour caption: {e}")

        self._hour_captions.append({
            "time_span": hour_time_span,
            "caption": hour_caption,
            "timestamp": last_timestamp,
        })

        # 重置 hour 窗口
        self._hour_window_min_captions = []
        self._hour_window_start_timestamp = None

    def flush_remaining_multi_level(self) -> None:
        """处理结束时，刷新所有未满的 minute/hour 窗口中剩余的 caption。"""
        if self._min_window_second_captions:
            end_ts = self._min_window_second_captions[-1]["end_timestamp"]
            self._flush_minute_window(end_ts)
        if self._hour_window_min_captions:
            end_ts = self._hour_window_min_captions[-1]["end_timestamp"]
            self._flush_hour_window(end_ts)
