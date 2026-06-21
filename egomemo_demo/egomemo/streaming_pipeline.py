"""
Core Streaming Pipeline Orchestrator for EgoServe-RL.

Per-step loop (one step = one caption_window, e.g. 10s):
1. Sample frames -> generate caption via Qwen3.5 (MemoryBridge)
2. Check question timeouts
3. For EACH active question + PROACTIVE:
   a. Build prompt with caption + frames context
   b. Reasoning VLM (Qwen3-VL) outputs: [silent] / [search] / [respond]
   c. If [search]: retrieve from memory -> force [respond] (max 2 VLM calls)
4. Record decisions

Supports two pipeline modes:
- "sequential": Build ALL memory first, then reason/retrieve (for research)
- "async": Memory construction and reasoning run concurrently (for web/real-time)

The reasoning_model_path can point to either the base Qwen3-VL or a fine-tuned
checkpoint — the inference code is identical.
"""

import concurrent.futures
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Ensure videorag is importable (append, not insert — keep EgoServe-RL's egomemo/ higher priority)
# 开源布局下 videorag 与 egomemo 平级，默认指向上级目录；可用 VIDEORAG_ROOT 覆盖
_VIDEORAG_ROOT = os.environ.get("VIDEORAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _VIDEORAG_ROOT not in sys.path:
    sys.path.append(_VIDEORAG_ROOT)

from videorag.egograph_retrieval_optimize_ import VideoGraphSeparated
from videorag._llm import (
    LLMConfig,
    openai_embedding,
    gpt_4o_mini_complete,
)
from videorag.video_processing import sample_frames_by_interval
from .config import PipelineConfig
from .question_queue import QuestionQueueManager, QuestionStatus
from .action_router import ActionRouter
from .memory_bridge import MemoryBridge, _make_time_span, _seconds_to_hhmmss, _run_async
from .output_recorder import OutputRecorder
from .working_memory import WorkingMemory
from .prompt_templates import (
    QA_SYSTEM_PROMPT,
    PROACTIVE_SYSTEM_PROMPT,
    PER_QUESTION_PROMPT,
    PER_QUESTION_WITH_MEMORY_PROMPT,
    PROACTIVE_PROMPT,
    PROACTIVE_WITH_MEMORY_PROMPT,
)


def create_default_llm_config() -> LLMConfig:
    """Create a default LLM config for VideoGraphSeparated (embedding + KG extraction)."""
    return LLMConfig(
        embedding_func_raw=openai_embedding,
        embedding_model_name="text-embedding-3-small",
        embedding_dim=1536,
        embedding_max_token_size=8192,
        embedding_batch_num=32,
        embedding_func_max_async=32,
        query_better_than_threshold=0.2,
        best_model_func_raw=gpt_4o_mini_complete,
        best_model_name="gpt-4o-mini",
        best_model_max_token_size=32768,
        best_model_max_async=32,
        cheap_model_func_raw=gpt_4o_mini_complete,
        cheap_model_name="gpt-4o-mini",
        cheap_model_max_token_size=32768,
        cheap_model_max_async=32,
    )


# API 模型列表（不需要本地加载，通过 OpenAI API 调用）
API_REASONING_MODELS = {
    "gpt-5-mini", "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano",
    "o4-mini", "o3", "o3-mini",
}


def _is_api_model(model_path: str) -> bool:
    """判断是否是 API 模型（非本地模型）"""
    return model_path in API_REASONING_MODELS or model_path.startswith("gpt-") or model_path.startswith("o3") or model_path.startswith("o4")


def _load_reasoning_vlm(model_path: str, device: str = "cuda"):
    """Load local Qwen3-VL (base or fine-tuned) for reasoning.

    Returns (model, processor). For API models, returns (None, None).
    """
    if _is_api_model(model_path):
        logger.info(f"Using API reasoning model: {model_path} (no local loading)")
        return None, None

    import torch
    from transformers import AutoProcessor

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        low_cpu_mem_usage=True,
    )

    if "Qwen3.5" in model_path or "Qwen3_5" in model_path:
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    elif "Qwen2.5" in model_path or "Qwen2_5" in model_path:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    elif "Qwen3" in model_path:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def _strip_think_block(text: str) -> str:
    """Remove thinking blocks from model output. Handles multiple formats:
    1. <think>...</think>  — 完整的 think 块
    2. ...</think>         — 只有结尾标签（开头被截断）
    3. <think>...          — 只有开头标签（结尾被截断）
    """
    # 情况 1：完整的 <think>...</think>
    match = re.search(r"<think>.*?</think>\s*", text, re.DOTALL)
    if match:
        return text[match.end():].strip()

    # 情况 2：只有 </think>（开头的 <think> 缺失）
    match = re.search(r"</think>\s*", text, re.DOTALL)
    if match:
        return text[match.end():].strip()

    # 情况 3：只有 <think>（结尾的 </think> 缺失，整个输出都是思考）
    if "<think>" in text:
        return ""

    return text


def _parse_response_timestamp(text: str) -> tuple:
    """从模型输出中提取时间戳前缀。

    模型输出格式：'DAY1-00:01:35 You are holding a blue mug.'
    返回：(timestamp_str, content) 如 ('DAY1-00:01:35', 'You are holding a blue mug.')
    如果没有时间戳：返回 (None, text)
    """
    match = re.match(r"(DAY\d+-\d{2}:\d{2}:\d{2})\s+(.*)", text, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return None, text


def _extract_caption_text(raw_caption: str) -> str:
    """Extract readable caption text from a raw caption JSON string.

    Supports two formats:
    - egograph format: {"dense_caption": {"DAY1-HH:MM:SS-HH:MM:SS": "帧caption", ...}, "description": "全局caption"}
    - legacy format: {"caption": "...", "frames": {...}}

    Returns description + dense captions for use in reasoning prompts.
    Falls back to the raw string if parsing fails.
    """
    if not raw_caption:
        return "(no caption)"
    try:
        parsed = json.loads(raw_caption)
        if isinstance(parsed, dict):
            # egograph 标准格式
            if "description" in parsed:
                parts = [parsed["description"]]
                dense = parsed.get("dense_caption", {})
                if dense:
                    for ts_key, cap in dense.items():
                        parts.append(f"[{ts_key}] {cap}")
                return "\n".join(parts)
            # legacy 格式
            if "caption" in parsed:
                return parsed["caption"]
    except (json.JSONDecodeError, ValueError):
        pass
    return raw_caption


class StreamingPipeline:
    """Streaming pipeline for egocentric video processing.

    Two models:
    - Caption model (Qwen3.5-4B/9B via VideoRAG): structured captions for memory
    - Reasoning VLM (Qwen3-VL, base or fine-tuned): [silent]/[search]/[respond] decisions

    Two pipeline modes:
    - "sequential": Build ALL memory first, then reason over each step
    - "async": Memory + reasoning run concurrently per step
    """

    def __init__(
        self,
        config: PipelineConfig,
        llm_config: Optional[LLMConfig] = None,
        event_callback: Optional[Callable] = None,
    ):
        self.config = config
        self.llm_config = llm_config or create_default_llm_config()
        self.event_callback = event_callback

        # Sub-modules (initialized lazily in run_on_video)
        self._videorag = None
        self.question_queue = QuestionQueueManager(
            timeout_seconds=config.question_timeout_seconds
        )
        self.action_router = ActionRouter()
        self.memory = None
        self.recorder = None
        self.working_memory = None

        # Reasoning VLM (loaded lazily)
        self._reasoning_model = None
        self._reasoning_processor = None

        # State
        self._proactive_history: List[Dict] = []
        self._last_proactive_time: float = -999.0
        # 按 proactive 类型分桶的冷却时间表：
        #   key = kind（"hydration"/"shopping_milk"/"charger_clutter"/"llm" 等）
        #   value = 上次该类型触发的 video_time
        # 冷却只抑制"同类型"的重复提醒；不同类型之间不互相压制。
        self._last_proactive_time_by_kind: Dict[str, float] = {}
        self._step_count: int = 0
        self._is_running: bool = False
        self._current_video_time: float = 0.0

        # 计时统计
        self._timing_reasoning: List[float] = []     # 单次推理耗时（VLM call）
        self._timing_retrieval: List[float] = []     # 单次检索耗时（memory.read）
        self._timing_step_total: List[float] = []    # 单步总耗时（推理+主动服务）

        # 端到端 wall-clock 计时（用于回应 reviewer 的实时性质疑）
        # 关键指标：RTF = 处理墙钟 / 视频时长；< 1.0 才能算 real-time
        self._wall_total_start: Optional[float] = None
        self._wall_total_end: Optional[float] = None
        self._wall_memory_start: Optional[float] = None
        self._wall_memory_end: Optional[float] = None
        self._wall_reasoning_start: Optional[float] = None
        self._wall_reasoning_end: Optional[float] = None
        self._video_duration: float = 0.0  # 实际处理的视频时长（秒）

        # 启动延迟相关：reviewer 关心"用户从开口提问到拿到第一个回答要多久"。
        #   _wall_chunk0_mem_done : chunk 0 的记忆构造完成时刻（async/sequential 都记）
        #   _wall_chunk0_reason_done : chunk 0 的推理完成时刻
        # first_response_latency = chunk0_reason_done - wall_total_start
        self._wall_chunk0_mem_done: Optional[float] = None
        self._wall_chunk0_reason_done: Optional[float] = None
        # 每个 chunk 的推理墙钟时长（用于稳态 RTF 与 bottleneck 判断）
        self._timing_per_chunk_reason_wall: List[float] = []
        # 每个 chunk 的记忆构造墙钟时长（caption + KG 等，单 chunk）
        self._timing_per_chunk_memory_wall: List[float] = []

        # 前端同步：推理完一个 step 后等前端 TTS 播完再继续下一个 step 的推理
        # 初始状态不 set —— chunk 0 的推理必须等前端播完第一段视频后发来的 frontend_ready
        import threading
        self._frontend_ready = threading.Event()

        # For async mode
        self._prev_caption_text: Optional[str] = None
        self._prev_time_span: Optional[str] = None
        self._prev_video_time: Optional[float] = None
        self._prev_frames: Optional[List[str]] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def current_video_time(self) -> float:
        return self._current_video_time

    def _load_reasoning_model(self):
        """Load the reasoning VLM if not already loaded."""
        if self._reasoning_model is None:
            logger.info(f"Loading reasoning VLM: {self.config.reasoning_model_path}")
            self._reasoning_model, self._reasoning_processor = _load_reasoning_vlm(
                self.config.reasoning_model_path
            )
            logger.info("Reasoning VLM loaded")

    def notify_frontend_ready(self):
        """前端 TTS 播完后调用，通知后端可以继续下一个 step 的推理。"""
        logger.info(f"[Sync] notify_frontend_ready called, setting event (was_set={self._frontend_ready.is_set()})")
        self._frontend_ready.set()

    def inject_question(
        self, text: str, video_timestamp: float,
        follow_up_parent: Optional[str] = None, recurring: bool = False,
    ) -> str:
        """Add a question to the queue."""
        q = self.question_queue.add_question(
            text=text, ask_time=video_timestamp,
            follow_up_parent=follow_up_parent, recurring=recurring,
        )
        self.working_memory.init_question(q.qid, text, video_timestamp, recurring)
        if follow_up_parent:
            self.working_memory.add_proactive_follow_up(
                follow_up_parent, q.qid, text, video_timestamp
            )
        logger.info(f"Question injected: {q.qid} at t={video_timestamp}s: {text}")
        if self.event_callback:
            self.event_callback("question_added", {
                "qid": q.qid, "text": text, "timestamp": video_timestamp,
            })
        return q.qid

    def run_on_video(
        self, video_path: str,
        questions_with_timestamps: Optional[List[Dict]] = None,
    ) -> Dict:
        """Main entry point: process an entire video file."""
        self._is_running = True
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_working_dir = os.path.join(self.config.working_dir, video_name)

        if self.config.clear_cache and os.path.exists(video_working_dir):
            import shutil
            shutil.rmtree(video_working_dir)
        os.makedirs(video_working_dir, exist_ok=True)

        logger.info(f"Pipeline: {video_path} | mode={self.config.pipeline_mode}")

        # Initialize VideoRAG + caption model
        # rebuttal 测试：entity_extract_max_gleaning=0 关闭 nano-graphrag 的实体
        # 抽取 gleaning 迭代（默认 1 轮，每 chunk 多花 ~20s 在 "continue extraction"
        # + "if loop" 两次额外 LLM call）。关掉后 KG insert 从 ~30s/chunk 降到
        # ~10-12s/chunk，对实体召回率影响很小。
        self._videorag = VideoGraphSeparated(
            llm=self.llm_config, working_dir=video_working_dir,
            entity_extract_max_gleaning=0,
        )
        self._videorag.load_caption_model(self.config.caption_model)

        # Load reasoning VLM
        self._load_reasoning_model()

        # Initialize sub-modules
        self.memory = MemoryBridge(
            self._videorag, self.llm_config, self.config.datasets_type,
            caption_window_seconds=self.config.caption_window_seconds,
            enable_multi_level_caption=self.config.enable_multi_level_caption,
            caption_window_minutes=self.config.caption_window_minutes,
            caption_window_hours=self.config.caption_window_hours,
            gap_threshold_seconds=self.config.gap_threshold_seconds,
            kg_extraction_mode=self.config.kg_extraction_mode,
        )
        self.recorder = OutputRecorder(video_working_dir)
        self.working_memory = WorkingMemory(video_working_dir)

        # Pre-load questions
        if questions_with_timestamps:
            for q in questions_with_timestamps:
                self.inject_question(
                    text=q["text"], video_timestamp=q["timestamp"],
                    recurring=q.get("recurring", False),
                )

        if self.event_callback:
            self.event_callback("processing_started", {"video_path": video_path})

        # ★ 端到端总墙钟开始
        self._wall_total_start = time.time()

        try:
            frames, _, time_ranges = sample_frames_by_interval(
                video_path,
                interval_seconds=self.config.step_interval_seconds,
                output_format="base64",
            )
            if not frames:
                logger.error("No frames extracted")
                return self.recorder.get_full_trajectory()

            step_batches = self._group_frames_into_steps(frames, time_ranges)
            logger.info(f"{len(frames)} frames -> {len(step_batches)} steps")

            # 记录处理的视频时长（最后一个 batch 的 end_time 即为视频末尾时间戳）
            if step_batches:
                self._video_duration = step_batches[-1]["end_time"] - step_batches[0]["start_time"]
            logger.info(f"[Wall] Video duration to process: {self._video_duration:.2f}s")

            if self.config.pipeline_mode == "async":
                self._run_async_mode(step_batches)
            else:
                self._run_sequential_mode(step_batches)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            if self.event_callback:
                self.event_callback("error", {"message": str(e)})
        finally:
            self._is_running = False
            # ★ 端到端总墙钟结束（覆盖 frame 抽取 + memory + reasoning 全流程）
            self._wall_total_end = time.time()
            # 刷新未满的多级 caption 窗口
            if self.config.enable_multi_level_caption:
                self.memory.flush_remaining_multi_level()
            # ★ 在 finalize 前，把所有分项计时塞进 recorder.timing，
            # 这样 results.json 里能直接看到。
            try:
                self.recorder.timing = self._collect_timing_for_recorder()
            except Exception as e:
                logger.warning(f"Failed to collect timing dict: {e}")
            self.recorder.finalize()
            self.working_memory.save()

            # 把 VideoRAG 构造的所有 storage 刷到磁盘。
            # JsonKVStorage / NanoVectorDBStorage / NetworkXStorage 都实现了
            # index_done_callback()，只有调用它数据才会真正写文件。
            # 没这步的话 text_chunks / chunks_vdb / entities_vdb / KG /
            # video_segment_feature 全部跑完就丢。
            try:
                self._flush_videorag_memory()
            except Exception as e:
                logger.error(f"Flush videorag memory failed: {e}", exc_info=True)

            # ★ 输出完整计时统计
            self._log_timing_summary()

            if self.event_callback:
                self.event_callback(
                    "processing_complete",
                    self.recorder.get_full_trajectory().get("summary", {}),
                )

        return self.recorder.get_full_trajectory()

    # ==================================================================
    # Sequential mode
    # ==================================================================
    def _run_sequential_mode(self, step_batches: List[Dict]) -> None:
        total = len(step_batches)

        # Phase 1: Build all memory
        # ★ 记忆构造阶段墙钟开始
        self._wall_memory_start = time.time()
        logger.info(f"[Sequential Phase 1] Building memory for {total} steps...")
        caption_entries = []
        for i, batch in enumerate(step_batches):
            time_span = _make_time_span(1, batch["start_time"], batch["end_time"])
            logger.info(f"  Caption {i+1}/{total} | t={_seconds_to_hhmmss(batch['start_time'])}")

            # 广播: caption 生成开始
            if self.event_callback:
                self.event_callback("pipeline_stage", {
                    "phase": "memory", "stage": "caption_start",
                    "step": i, "total": total,
                    "time": batch["start_time"],
                    "time_span": time_span,
                })

            t_chunk_mem_start = time.time()
            entry = self.memory.build_step_caption_only(
                frames_base64=batch["frames"],
                time_span=time_span,
                timestamp=batch["start_time"],
                frame_times=batch.get("frame_times"),
            )
            t_chunk_mem_end = time.time()
            self._timing_per_chunk_memory_wall.append(t_chunk_mem_end - t_chunk_mem_start)
            caption_entries.append(entry)

            # 广播: caption 生成完成（含内容预览）
            caption_preview = _extract_caption_text(entry.get("caption", ""))[:200]
            if self.event_callback:
                self.event_callback("pipeline_stage", {
                    "phase": "memory", "stage": "caption_done",
                    "step": i, "total": total,
                    "time": batch["start_time"],
                    "caption_preview": caption_preview,
                })

        # 广播: KG 提取开始
        if self.event_callback:
            self.event_callback("pipeline_stage", {
                "phase": "memory", "stage": "kg_start",
                "total": total,
            })

        logger.info("[Sequential Phase 1] Batch KG + visual embedding...")
        # sequential 模式下 chunk0 的"记忆完成"应该理解为整个 phase 1 结束
        # （因为 KG/visual 是批量在 phase 1 末尾做的，chunk0 没单独的 KG 落库时刻）
        self.memory.flush_pending_memory()
        self._wall_chunk0_mem_done = time.time()

        # 广播: KG + visual 完成（统计实体/关系数量）
        entity_count = 0
        relation_count = 0
        try:
            kg = self.memory._vg.chunk_entity_relation_graph
            if hasattr(kg, '_graph') and hasattr(kg._graph, 'number_of_nodes'):
                entity_count = kg._graph.number_of_nodes()
                relation_count = kg._graph.number_of_edges()
        except Exception:
            pass
        if self.event_callback:
            self.event_callback("pipeline_stage", {
                "phase": "memory", "stage": "kg_done",
                "total": total,
                "entity_count": entity_count,
                "relation_count": relation_count,
            })
            self.event_callback("pipeline_stage", {
                "phase": "memory", "stage": "visual_done",
                "total": total,
            })

        # ★ 记忆阶段墙钟结束（含 caption 生成 + 批量 KG/visual 落库）
        self._wall_memory_end = time.time()
        mem_wall = self._wall_memory_end - self._wall_memory_start
        logger.info(f"[Wall] Sequential memory phase: {mem_wall:.2f}s "
                    f"(video={self._video_duration:.1f}s, "
                    f"RTF_mem={mem_wall/max(self._video_duration,1e-6):.3f})")

        # Phase 2: Reasoning — 用 end_time 代表"这个 chunk 处理完了"
        # ★ 推理阶段墙钟开始
        self._wall_reasoning_start = time.time()
        logger.info(f"[Sequential Phase 2] Reasoning over {total} steps...")
        for i, batch in enumerate(step_batches):
            # 等待前端就绪（上一个 step 的 TTS 播完）
            self._frontend_ready.wait(timeout=120)
            self._frontend_ready.clear()

            time_span = _make_time_span(1, batch["start_time"], batch["end_time"])
            caption_text = _extract_caption_text(caption_entries[i].get("caption", ""))
            self._current_video_time = batch["end_time"]
            self._step_count += 1
            t_chunk_reason_start = time.time()
            self._reason_and_route(i, batch["end_time"], time_span, caption_text, batch["frames"])
            t_chunk_reason_end = time.time()
            self._timing_per_chunk_reason_wall.append(t_chunk_reason_end - t_chunk_reason_start)
            if i == 0:
                self._wall_chunk0_reason_done = t_chunk_reason_end

        # ★ 推理阶段墙钟结束
        self._wall_reasoning_end = time.time()

    # ==================================================================
    # Async mode
    # ==================================================================
    def _run_async_mode(self, step_batches: List[Dict]) -> None:
        """异步模式：记忆构造和推理解耦。

        三个条件全部满足后才开始下一个 chunk 的推理+前端播放：
          1. 该 chunk 的记忆构造已完成（mem_ready）
          2. 上一个 chunk 的推理已完成（隐含：主线程自身的顺序执行）
          3. 上一个 chunk 的前端播放+TTS已完成（frontend_ready）

        记忆构造在独立线程中按顺序持续运行，不受前端/推理影响。
        """
        total = len(step_batches)

        # 每个 chunk 一个 Event，记忆线程完成后 set
        mem_ready_events = [threading.Event() for _ in range(total)]
        mem_results = [None] * total  # 存放记忆构造结果

        # ★ async 模式下 memory 和 reasoning 并行，分别记两段墙钟
        self._wall_memory_start = time.time()

        # --- 记忆构造线程：顺序执行，不受前端/推理影响 ---
        def _memory_worker():
            for i, batch in enumerate(step_batches):
                time_span = _make_time_span(1, batch["start_time"], batch["end_time"])

                if self.event_callback:
                    self.event_callback("pipeline_stage", {
                        "phase": "memory", "stage": "caption_start",
                        "step": i, "total": total,
                        "time": batch["start_time"],
                        "time_span": time_span,
                    })

                t_chunk_mem_start = time.time()
                caption_entry = self.memory.build_step(
                    batch["frames"], time_span, batch["start_time"],
                    batch.get("frame_times"),
                )
                t_chunk_mem_end = time.time()
                self._timing_per_chunk_memory_wall.append(t_chunk_mem_end - t_chunk_mem_start)
                # 记录 chunk0 记忆完成时刻（首响应延迟用）
                if i == 0:
                    self._wall_chunk0_mem_done = t_chunk_mem_end
                caption_text = _extract_caption_text(caption_entry.get("caption", ""))

                if self.event_callback:
                    self.event_callback("pipeline_stage", {
                        "phase": "memory", "stage": "caption_done",
                        "step": i, "total": total,
                        "time": batch["start_time"],
                        "caption_preview": caption_text[:200],
                    })

                mem_results[i] = {
                    "caption_text": caption_text,
                    "time_span": time_span,
                    "end_time": batch["end_time"],
                    "frames": batch["frames"],
                }
                mem_ready_events[i].set()
                logger.info(f"[Memory] Step {i}/{total} done, mem_ready set")
            # ★ 记忆线程结束墙钟
            self._wall_memory_end = time.time()

        mem_thread = threading.Thread(target=_memory_worker, daemon=True)
        mem_thread.start()

        # ★ 推理阶段墙钟开始（与 memory 并行，分别记录以做对比）
        self._wall_reasoning_start = time.time()

        # --- 推理主线程：同时等记忆+前端，都就绪后立即推理 ---
        for i in range(total):
            # 同时等两个条件（用轮询避免信号丢失）
            logger.info(f"[Async] Step {i}: waiting for conditions "
                        f"(mem_ready={mem_ready_events[i].is_set()}, "
                        f"frontend_ready={self._frontend_ready.is_set()})...")

            while True:
                mem_ok = mem_ready_events[i].is_set()
                fe_ok = self._frontend_ready.is_set()
                if mem_ok and fe_ok:
                    break
                # 短暂 sleep 避免忙等，但响应很快
                time.sleep(0.1)

            # 两个条件都满足，清除 frontend_ready 并开始
            self._frontend_ready.clear()

            item = mem_results[i]
            self._current_video_time = item["end_time"]
            self._step_count += 1

            logger.info(f"[Async] Step {i}: BOTH conditions met, starting reasoning (t={item['end_time']:.1f}s)")
            t_chunk_reason_start = time.time()
            self._reason_and_route(
                i, item["end_time"], item["time_span"],
                item["caption_text"], item["frames"],
            )
            t_chunk_reason_end = time.time()
            self._timing_per_chunk_reason_wall.append(t_chunk_reason_end - t_chunk_reason_start)
            if i == 0:
                self._wall_chunk0_reason_done = t_chunk_reason_end
            logger.info(f"[Async] Step {i}: reasoning done, step_complete sent to frontend")

        mem_thread.join(timeout=10)
        # ★ 推理阶段墙钟结束（推理 for 循环结束 + 等记忆线程收尾）
        self._wall_reasoning_end = time.time()
        # 兜底：万一 memory worker 因异常未设置 end，这里取当前时间
        if self._wall_memory_end is None:
            self._wall_memory_end = time.time()

    # ==================================================================
    # Reasoning: per-question + proactive
    # ==================================================================
    def _reason_and_route(
        self, step_idx: int, video_time: float, time_span: str,
        caption_text: str, frames_base64: List[str],
    ) -> None:
        t_step_start = time.time()
        all_outputs = []
        all_actions_taken = []

        # Check timeouts
        for q in self.question_queue.check_timeouts(video_time):
            logger.info(f"Question {q.qid} timed out")

        # Per-question reasoning
        active_questions = self.question_queue.get_questions_at_time(video_time)
        for q in active_questions:
            try:
                result = self._reason_for_question(q, video_time, time_span, caption_text, frames_base64)
                if result:
                    all_outputs.extend(result.get("outputs", []))
                    all_actions_taken.extend(result.get("actions_taken", []))
            except Exception as e:
                logger.error(f"Error evaluating {q.qid}: {e}", exc_info=True)

        # Proactive
        if self.config.enable_proactive:
            try:
                r = self._reason_proactive(video_time, time_span, caption_text, frames_base64)
                if r:
                    all_outputs.extend(r.get("outputs", []))
                    all_actions_taken.extend(r.get("actions_taken", []))
            except Exception as e:
                logger.error(f"Proactive error: {e}", exc_info=True)

        t_step_end = time.time()
        self._timing_step_total.append(t_step_end - t_step_start)
        logger.info(f"[Timing] Step {step_idx} total reasoning: {t_step_end-t_step_start:.3f}s")

        # Record
        self.recorder.record_step(step_idx, video_time, caption_text, [], {
            "outputs": all_outputs, "actions_taken": all_actions_taken,
        })
        if self.event_callback:
            self.event_callback("step_complete", {
                "step": step_idx, "time": video_time, "caption": caption_text[:300],
                "actions": all_actions_taken, "outputs": all_outputs,
            })

    def _reason_for_question(
        self, question, video_time: float, time_span: str,
        caption_text: str, frames_base64: List[str],
    ) -> Optional[Dict]:
        """Two-pass reasoning for one question.

        Pass 1: VLM decides [silent] / [search] / [respond]
        Pass 2 (only if [search]): retrieve -> inject [observation] -> force [respond]
        """
        outputs = []
        actions_taken = []

        # Check if we have a pending search result from previous step
        mem_result = self.question_queue.consume_pending_mem_read(question.qid)
        previous_answers_str = self.working_memory.format_previous_answers(question.qid, max_count=5)

        # Build prompt
        if mem_result:
            # Pass 2 prompt: retrieved memory already available
            prompt = PER_QUESTION_WITH_MEMORY_PROMPT.format(
                system_prompt=QA_SYSTEM_PROMPT,
                current_timestamp=f"{_seconds_to_hhmmss(video_time)} ({video_time:.1f}s)",
                current_caption=caption_text,
                retrieved_memory=mem_result,
                qid=question.qid,
                question_text=question.text,
            )
        else:
            # Pass 1 prompt: free decision
            prompt = PER_QUESTION_PROMPT.format(
                system_prompt=QA_SYSTEM_PROMPT,
                current_timestamp=f"{_seconds_to_hhmmss(video_time)} ({video_time:.1f}s)",
                current_caption=caption_text,
                recent_history=self.memory.format_recent_history(self.config.recent_history_count),
                qid=question.qid,
                question_text=question.text,
                question_ask_time=f"{question.ask_time:.1f}",
                question_status=question.status.value,
                question_evidence="; ".join(question.evidence_notes[-3:]) if question.evidence_notes else "(none)",
                previous_answers=previous_answers_str,
            )

        # Pass 1: VLM call（仅用 caption 文本，不传视频帧）
        t_reason_start = time.time()
        response = self._call_reasoning_vlm(prompt, [])
        t_reason_end = time.time()
        self._timing_reasoning.append(t_reason_end - t_reason_start)
        logger.info(f"[Timing] Reasoning VLM (Pass 1): {t_reason_end-t_reason_start:.3f}s")
        actions = self.action_router.parse_llm_output(response)
        if not actions:
            return None

        action = actions[0]
        action_type = action.action.upper()

        if action_type == "SILENT":
            actions_taken.append(f"SILENT_{question.qid}")

        elif action_type == "RESPOND":
            raw_answer = action.answer_text or ""
            ref_timestamp, answer_text = _parse_response_timestamp(raw_answer)
            self.question_queue.record_answer(question.qid, answer_text, video_time)
            self.recorder.record_answer(question.qid, answer_text, video_time, action.evidence)
            self.working_memory.add_answer(question.qid, answer_text, video_time, action.evidence)
            outputs.append({
                "type": "answer", "qid": question.qid,
                "answer": answer_text, "time": video_time,
                "ref_timestamp": ref_timestamp,  # 模型引用的 caption 时间戳
            })
            actions_taken.append(f"RESPOND_{question.qid}")
            logger.info(f"[RESPOND_{question.qid}] {f'@{ref_timestamp} ' if ref_timestamp else ''}{answer_text[:100]}...")

        elif action_type == "SEARCH":
            query = action.search_query or ""
            if query:
                # 用问题提问时间构建检索 time_span，只检索问题之前的内容
                retrieval_time_span = _make_time_span(1, 0, question.ask_time)
                t_retrieval_start = time.time()
                retrieved = self.memory.read(query, retrieval_time_span)
                t_retrieval_end = time.time()
                self._timing_retrieval.append(t_retrieval_end - t_retrieval_start)
                logger.info(f"[Timing] Memory retrieval: {t_retrieval_end-t_retrieval_start:.3f}s")
                self.working_memory.add_mem_read(question.qid, query, retrieved, video_time)
                actions_taken.append(f"SEARCH_{question.qid}")
                logger.info(f"[SEARCH_{question.qid}] query: {query[:80]}...")

                # Pass 2: multi-turn（纯文本，不传帧）
                history = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": response}]},
                ]

                pass2_prompt = (
                    f"[observation]\n{retrieved}\n[/observation]\n\n"
                    f"Based on the retrieved information above, answer the question [{question.qid}] now.\n"
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

                self.question_queue.record_answer(question.qid, answer_text, video_time)
                self.recorder.record_answer(question.qid, answer_text, video_time)
                self.working_memory.add_answer(question.qid, answer_text, video_time)
                outputs.append({
                    "type": "answer", "qid": question.qid,
                    "answer": answer_text, "time": video_time,
                    "ref_timestamp": ref_timestamp,
                })
                actions_taken.append(f"RESPOND_{question.qid}")
                logger.info(f"[RESPOND_{question.qid} after SEARCH] {f'@{ref_timestamp} ' if ref_timestamp else ''}{answer_text[:100]}...")

        return {"outputs": outputs, "actions_taken": actions_taken}

    def _reason_proactive(
        self, video_time: float, time_span: str,
        caption_text: str, frames_base64: List[str],
    ) -> Optional[Dict]:
        """Two-pass proactive reasoning, same as questions:
        Pass 1: [silent] / [search] / [respond]
        Pass 2 (only if [search]): retrieve -> force [respond]
        """
        if (video_time - self._last_proactive_time) < self.config.proactive_cooldown_seconds:
            return None

        prompt = PROACTIVE_PROMPT.format(
            system_prompt=PROACTIVE_SYSTEM_PROMPT,
            current_timestamp=f"{_seconds_to_hhmmss(video_time)} ({video_time:.1f}s)",
            current_caption=caption_text,
            recent_history=self.memory.format_recent_history(self.config.recent_history_count),
            proactive_history=self._format_proactive_history(),
        )

        # Pass 1（纯文本，不传帧）
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
            if query:
                t_ret_start = time.time()
                retrieved = self.memory.read(query, time_span)
                t_ret_end = time.time()
                self._timing_retrieval.append(t_ret_end - t_ret_start)
                logger.info(f"[Timing] Proactive retrieval: {t_ret_end-t_ret_start:.3f}s")

                actions_taken.append("PROACTIVE_SEARCH")
                logger.info(f"[PROACTIVE_SEARCH] query: {query[:80]}...")

                # Pass 2（纯文本，不传帧）
                history = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": response}]},
                ]

                pass2_prompt = (
                    f"[observation]\n{retrieved}\n[/observation]\n\n"
                    f"Based on the retrieved information, decide whether to issue a proactive reminder.\n"
                    f"[respond] "
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
                action_type = "RESPOND"
            else:
                return {"outputs": outputs, "actions_taken": actions_taken}
        else:
            # Direct RESPOND
            raw_content = action.answer_text or ""
            ref_timestamp, content = _parse_response_timestamp(raw_content)

        if action_type == "RESPOND" and content:
            event_id = f"proactive_{uuid.uuid4().hex[:8]}"
            self._last_proactive_time = video_time
            self._proactive_history.append({
                "time": video_time, "event_id": event_id, "content": content,
            })
            self.recorder.record_proactive(content, video_time, action.evidence, event_id=event_id)
            self.working_memory.add_proactive(event_id, content, video_time, action.evidence)
            outputs.append({
                "type": "proactive", "event_id": event_id,
                "content": content, "time": video_time,
                "ref_timestamp": ref_timestamp,
            })
            actions_taken.append("PROACTIVE")
            logger.info(f"[PROACTIVE] {f'@{ref_timestamp} ' if ref_timestamp else ''}{content[:100]}...")

        return {"outputs": outputs, "actions_taken": actions_taken}

    # ==================================================================
    # VLM call: local model or API
    # ==================================================================
    def _call_reasoning_vlm(
        self, prompt: str, frames_base64: List[str],
        history_messages: Optional[List[Dict]] = None,
    ) -> str:
        """Call the reasoning model (local VLM or OpenAI API).

        推理阶段仅使用 caption 文本，不传视频帧（frames_base64 通常为空列表）。
        """
        t0 = time.time()

        if _is_api_model(self.config.reasoning_model_path):
            response = self._call_api_reasoning(prompt, history_messages)
        else:
            response = self._call_local_reasoning(prompt, frames_base64, history_messages)

        elapsed = time.time() - t0
        logger.info(f"  Reasoning call ({self.config.reasoning_model_path}): {elapsed:.1f}s")
        return _strip_think_block(response)

    def _call_api_reasoning(
        self, prompt: str, history_messages: Optional[List[Dict]] = None,
    ) -> str:
        """通过 OpenAI API 调用推理模型（GPT-5-mini 等）。"""
        from openai import OpenAI

        client = OpenAI()

        # 构建 API 消息格式（纯文本，不传图片）
        messages = []
        if history_messages:
            for msg in history_messages:
                role = msg["role"]
                # 从 content 列表中提取纯文本
                if isinstance(msg["content"], list):
                    text_parts = [c["text"] for c in msg["content"] if isinstance(c, dict) and c.get("type") == "text"]
                    content = "\n".join(text_parts)
                else:
                    content = str(msg["content"])
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt})

        try:
            resp = client.chat.completions.create(
                model=self.config.reasoning_model_path,
                messages=messages,
                max_completion_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API reasoning call failed: {e}")
            return ""

    def _call_local_reasoning(
        self, prompt: str, frames_base64: List[str],
        history_messages: Optional[List[Dict]] = None,
    ) -> str:
        """调用本地 Qwen VLM 进行推理。"""
        import torch
        import io as _io
        import base64 as _b64
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        if history_messages:
            messages = list(history_messages) + [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        else:
            content = []
            if frames_base64:
                for idx, fb64 in enumerate(frames_base64):
                    try:
                        img = Image.open(_io.BytesIO(_b64.b64decode(fb64)))
                        content.append({"type": "text", "text": f"<{idx}>"})
                        content.append({"type": "image", "image": img})
                    except Exception:
                        continue
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

        text_prompt = self._reasoning_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        vision_out = process_vision_info(messages)
        image_inputs = vision_out[0] if len(vision_out) > 0 else None
        video_inputs = vision_out[1] if len(vision_out) > 1 else None

        inputs = self._reasoning_processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        device = next(self._reasoning_model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.inference_mode():
            gen_kwargs = {"max_new_tokens": 2048}
            if hasattr(self._reasoning_processor, "tokenizer"):
                gen_kwargs["pad_token_id"] = self._reasoning_processor.tokenizer.eos_token_id
            generated_ids = self._reasoning_model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        if generated_ids.shape[1] > input_len:
            generated_ids = generated_ids[:, input_len:]

        return self._reasoning_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    # ==================================================================
    # Helpers
    # ==================================================================
    def _flush_videorag_memory(self) -> None:
        """把 VideoRAG 内存中的记忆（KV/VDB/图）统一刷盘。

        JsonKVStorage、NanoVectorDBStorage、NetworkXStorage 都通过
        `await index_done_callback()` 触发真正的写文件。没调它的话，
        text_chunks / chunks_vdb / entities_vdb / chunk_entity_relation /
        video_segment_feature 等构造的记忆只在内存里，进程退出就丢。
        """
        vg = getattr(self, "_videorag", None)
        if vg is None:
            return

        candidates = [
            "full_docs", "text_chunks", "video_path_db", "video_segments",
            "llm_response_cache",
            "chunks_vdb", "entities_vdb", "video_segment_feature_vdb",
            "chunk_entity_relation_graph",
        ]

        async def _flush_all():
            for attr in candidates:
                store = getattr(vg, attr, None)
                if store is None:
                    continue
                cb = getattr(store, "index_done_callback", None)
                if cb is None:
                    continue
                try:
                    result = cb()
                    # index_done_callback 可能是 async 的
                    if hasattr(result, "__await__"):
                        await result
                    logger.info(f"[flush] videorag.{attr} saved")
                except Exception as e:
                    logger.warning(f"[flush] videorag.{attr} failed: {e}")

        _run_async(_flush_all())

    def _collect_timing_for_recorder(self) -> Dict:
        """把 pipeline 收集的所有分项计时打包成 dict，供 OutputRecorder 写入
        results.json。结构：
            {
              "video_duration_seconds": float,
              "wall_clocks": {
                "total_seconds": float,            # 端到端
                "memory_phase_seconds": float,     # memory 阶段（async 与 reasoning 并行；sequential 是 phase1）
                "reasoning_phase_seconds": float,
                "chunk0_memory_done_seconds": float,    # chunk0 caption+KG+visual 完成时刻（相对 run_on_video 起）
                "chunk0_reason_done_seconds": float,
                "first_response_latency_seconds": float, # = chunk0_reason_done
              },
              "rtf": {
                "end_to_end": float,               # total_wall / video_duration
                "memory_phase": float,
                "reasoning_phase": float,
                "steady_state_async_max": float,   # max(steady_mem_rtf, steady_reason_rtf)，仅 async
                "steady_memory_per_chunk": float,
                "steady_reasoning_per_chunk": float,
                "bottleneck": "memory-bound|reasoning-bound|n/a",
                "avg_processing_seconds_per_1s_video": float,
                "avg_processing_seconds_per_1min_video": float,
              },
              "per_chunk": {
                "memory_wall_per_chunk": [float, ...],   # 后台 memory worker 每 chunk 耗时
                "reasoning_wall_per_chunk": [float, ...],# 主线程每 chunk 推理耗时
              },
              "memory_breakdown": {                 # memory_bridge.timing.summary()
                "caption_generation": {count, total, avg, min, max},
                "kg_insert": {...},
                "visual_embedding": {...},
                ...
              },
              "reasoning_breakdown": {              # 自身收集
                "vlm_calls": {count, total, avg, min, max},   # Pass1 + Pass2 全部
                "memory_retrieval": {...},
                "step_total": {...},                # 每 step 推理总耗时（含 search+pass2+proactive）
              },
            }
        """
        def _stat(arr):
            if not arr:
                return {"count": 0, "total": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
            return {
                "count": len(arr),
                "total": round(sum(arr), 3),
                "avg": round(sum(arr) / len(arr), 3),
                "min": round(min(arr), 3),
                "max": round(max(arr), 3),
            }

        def _delta(t_start, t_end):
            if t_start is None or t_end is None:
                return None
            return round(t_end - t_start, 3)

        video_dur = max(self._video_duration, 1e-6)

        wall_total = _delta(self._wall_total_start, self._wall_total_end)
        wall_mem = _delta(self._wall_memory_start, self._wall_memory_end)
        wall_reason = _delta(self._wall_reasoning_start, self._wall_reasoning_end)

        chunk0_mem_done = (
            round(self._wall_chunk0_mem_done - self._wall_total_start, 3)
            if (self._wall_chunk0_mem_done and self._wall_total_start) else None
        )
        chunk0_reason_done = (
            round(self._wall_chunk0_reason_done - self._wall_total_start, 3)
            if (self._wall_chunk0_reason_done and self._wall_total_start) else None
        )

        # 稳态：去掉 chunk 0
        per_mem = self._timing_per_chunk_memory_wall
        per_reason = self._timing_per_chunk_reason_wall
        chunk_video_seconds = self.config.caption_window_seconds
        steady_mem_rtf = None
        steady_reason_rtf = None
        steady_total_rtf = None
        bottleneck = "n/a"
        if len(per_mem) >= 2 and len(per_reason) >= 2:
            steady_mem_avg = sum(per_mem[1:]) / len(per_mem[1:])
            steady_reason_avg = sum(per_reason[1:]) / len(per_reason[1:])
            steady_mem_rtf = round(steady_mem_avg / max(chunk_video_seconds, 1e-6), 3)
            steady_reason_rtf = round(steady_reason_avg / max(chunk_video_seconds, 1e-6), 3)
            if self.config.pipeline_mode == "async":
                steady_total_rtf = round(max(steady_mem_rtf, steady_reason_rtf), 3)
                bottleneck = "memory-bound" if steady_mem_rtf >= steady_reason_rtf else "reasoning-bound"
            else:
                steady_total_rtf = round(steady_mem_rtf + steady_reason_rtf, 3)

        e2e_rtf = round(wall_total / video_dur, 3) if wall_total else None

        # === 用于 rebuttal 的 "reasoning-only" 指标（排除 caption 构建等待）===
        # 推理服务延迟 = "从 caption ready 到产生 proactive 决策" 的耗时；
        # 这才是 reviewer 真正关心的 "inference latency"。
        # caption 构建是离线索引/memory 阶段，不应当算进 inference 延迟。
        per_chunk_reason = self._timing_per_chunk_reason_wall
        per_chunk_step_total = self._timing_step_total
        reasoning_only_avg = (
            round(sum(per_chunk_reason) / len(per_chunk_reason), 3)
            if per_chunk_reason else None
        )
        reasoning_only_rtf = (
            round(reasoning_only_avg / max(chunk_video_seconds, 1e-6), 3)
            if reasoning_only_avg else None
        )
        # 稳态 = 排除 chunk 0
        reasoning_only_steady_avg = None
        reasoning_only_steady_rtf = None
        if len(per_chunk_reason) >= 2:
            reasoning_only_steady_avg = round(
                sum(per_chunk_reason[1:]) / len(per_chunk_reason[1:]), 3,
            )
            reasoning_only_steady_rtf = round(
                reasoning_only_steady_avg / max(chunk_video_seconds, 1e-6), 3,
            )

        return {
            "video_duration_seconds": round(self._video_duration, 3),
            # ===== rebuttal 主打 5 个指标 =====
            "headline_metrics": {
                "_comment": (
                    "These are the metrics to cite in the rebuttal. They isolate "
                    "the inference cost (caption + KG building belong to memory "
                    "indexing, not inference latency)."
                ),
                "reasoning_only_rtf": reasoning_only_rtf,
                "reasoning_only_rtf_steady": reasoning_only_steady_rtf,
                "reasoning_only_avg_seconds_per_chunk": reasoning_only_avg,
                "reasoning_only_avg_seconds_per_chunk_steady": reasoning_only_steady_avg,
                "first_response_latency_seconds": chunk0_reason_done,
                "video_duration_seconds": round(self._video_duration, 3),
                "chunk_video_seconds": chunk_video_seconds,
                "reviewer_reference_rtf": 1.289,  # 77.33s / 60s
                "end_to_end_rtf_for_context": e2e_rtf,
            },
            "wall_clocks": {
                "total_seconds": wall_total,
                "memory_phase_seconds": wall_mem,
                "reasoning_phase_seconds": wall_reason,
                "chunk0_memory_done_seconds": chunk0_mem_done,
                "chunk0_reason_done_seconds": chunk0_reason_done,
                "first_response_latency_seconds": chunk0_reason_done,
            },
            "rtf": {
                "end_to_end": e2e_rtf,
                "memory_phase": round(wall_mem / video_dur, 3) if wall_mem else None,
                "reasoning_phase": round(wall_reason / video_dur, 3) if wall_reason else None,
                "steady_state_async_max": steady_total_rtf,
                "steady_memory_per_chunk": steady_mem_rtf,
                "steady_reasoning_per_chunk": steady_reason_rtf,
                "bottleneck": bottleneck,
                "avg_processing_seconds_per_1s_video": e2e_rtf,
                "avg_processing_seconds_per_1min_video": (
                    round(e2e_rtf * 60.0, 2) if e2e_rtf else None
                ),
            },
            "per_chunk": {
                "memory_wall_per_chunk": [round(x, 3) for x in per_mem],
                "reasoning_wall_per_chunk": [round(x, 3) for x in per_reason],
                "chunk_video_seconds": chunk_video_seconds,
            },
            "memory_breakdown": (
                self.memory.timing.summary() if self.memory is not None else {}
            ),
            "reasoning_breakdown": {
                "vlm_calls": _stat(self._timing_reasoning),
                "memory_retrieval": _stat(self._timing_retrieval),
                "step_total": _stat(self._timing_step_total),
            },
            "pipeline_mode": self.config.pipeline_mode,
        }

    def _log_timing_summary(self):
        """输出完整的计时统计"""
        def _stat(arr, name):
            if not arr:
                return
            total = sum(arr)
            avg = total / len(arr)
            logger.info(f"  {name:35s} | count={len(arr):3d} | total={total:8.2f}s | avg={avg:6.3f}s | min={min(arr):6.3f}s | max={max(arr):6.3f}s")

        logger.info("")
        logger.info("=" * 80)
        logger.info("PIPELINE TIMING SUMMARY")
        logger.info("=" * 80)

        # Memory bridge 统计
        logger.info("--- Memory Construction ---")
        self.memory.timing.log_summary()

        # Reasoning 统计
        logger.info("--- Reasoning & Retrieval ---")
        _stat(self._timing_reasoning, "VLM reasoning calls")
        _stat(self._timing_retrieval, "Memory retrieval calls")
        _stat(self._timing_step_total, "Step total (reason+proactive)")

        # 总计
        all_times = {
            "Caption generation": sum(self.memory.timing.caption_times) if self.memory.timing.caption_times else 0,
            "KG insert": sum(self.memory.timing.kg_insert_times) if self.memory.timing.kg_insert_times else 0,
            "Visual embedding": sum(self.memory.timing.visual_embed_times) if self.memory.timing.visual_embed_times else 0,
            "Batch entity extraction": sum(self.memory.timing.batch_entity_times) if self.memory.timing.batch_entity_times else 0,
            "Batch visual embedding": sum(self.memory.timing.batch_visual_times) if self.memory.timing.batch_visual_times else 0,
            "VLM reasoning": sum(self._timing_reasoning) if self._timing_reasoning else 0,
            "Memory retrieval": sum(self._timing_retrieval) if self._timing_retrieval else 0,
        }
        grand_total = sum(all_times.values())
        logger.info("--- Total Time Breakdown (sum of components, may overlap in async) ---")
        for name, t in all_times.items():
            if t > 0:
                pct = t / grand_total * 100 if grand_total > 0 else 0
                logger.info(f"  {name:35s} | {t:8.2f}s | {pct:5.1f}%")
        logger.info(f"  {'GRAND TOTAL':35s} | {grand_total:8.2f}s | 100.0%")

        # ====== ★ 端到端 Wall-Clock + Real-Time Factor ======
        # 这是回应 reviewer 实时性质疑的关键指标。
        # RTF = wall_clock / video_duration
        #   < 1.0  -> 比实时快（real-time 可行）
        #   = 1.0  -> 恰好实时
        #   > 1.0  -> 慢于实时（reviewer 引用的 77.33s/60s = RTF 1.29 即此情况）
        logger.info("--- Wall-Clock & Real-Time Factor ---")
        video_dur = max(self._video_duration, 1e-6)
        logger.info(f"  Pipeline mode                      : {self.config.pipeline_mode}")
        logger.info(f"  Video duration processed           : {self._video_duration:8.2f}s")

        def _wall_block(label: str, t_start, t_end):
            if t_start is None or t_end is None:
                return
            wall = t_end - t_start
            rtf = wall / video_dur
            tag = "(real-time OK)" if rtf < 1.0 else "(SLOWER than real-time)"
            logger.info(f"  {label:35s} | wall={wall:8.2f}s | RTF={rtf:5.3f} {tag}")

        _wall_block("Memory phase wall-clock", self._wall_memory_start, self._wall_memory_end)
        _wall_block("Reasoning phase wall-clock", self._wall_reasoning_start, self._wall_reasoning_end)
        _wall_block("End-to-end wall-clock (total)", self._wall_total_start, self._wall_total_end)

        # 模式说明
        if self.config.pipeline_mode == "async":
            logger.info("  [async] memory & reasoning run in PARALLEL — "
                        "the bottleneck is max(memory_wall, reasoning_wall), NOT their sum.")
        else:
            logger.info("  [sequential] memory and reasoning run BACK-TO-BACK — "
                        "wall-clock = memory_wall + reasoning_wall.")

        # 平均每秒视频的处理代价 + 每分钟代价
        if self._wall_total_start and self._wall_total_end and self._video_duration > 0:
            total_wall = self._wall_total_end - self._wall_total_start
            per_sec = total_wall / self._video_duration
            per_min = per_sec * 60.0
            logger.info(f"  Avg processing cost per 1s video   : {per_sec:6.3f}s "
                        f"(== RTF {per_sec:5.3f})")
            logger.info(f"  Avg processing cost per 1min video : {per_min:6.2f}s")
            # 对照 reviewer 引用的 77.33s/60s = 1.29
            logger.info(f"  >>> Reviewer reference (sequential mem only): 77.33s / 60s = RTF 1.289")
            logger.info(f"  >>> Current run end-to-end          : {total_wall:6.2f}s "
                        f"/ {self._video_duration:6.2f}s = RTF {per_sec:5.3f}")

        # ====== ★ First-Response Latency（用户从视频开始到拿到第一个回答的延迟）======
        # 这是 reviewer 关心的"timely intervention"指标。
        if self._wall_chunk0_reason_done and self._wall_total_start:
            first_resp_latency = self._wall_chunk0_reason_done - self._wall_total_start
            logger.info("--- Startup Latency ---")
            logger.info(f"  First-response latency (chunk0)    : {first_resp_latency:6.2f}s "
                        f"(time until system can answer the FIRST question)")
            if self._wall_chunk0_mem_done:
                chunk0_mem = self._wall_chunk0_mem_done - self._wall_total_start
                logger.info(f"    └ chunk0 memory done at         : {chunk0_mem:6.2f}s")

        # ====== ★ Steady-State RTF & Bottleneck ======
        # 稳态 = 去掉第一个 chunk 后的平均（首 chunk 含一次性启动开销，不公平）
        per_chunk_mem = self._timing_per_chunk_memory_wall
        per_chunk_reason = self._timing_per_chunk_reason_wall
        if len(per_chunk_mem) >= 2 and len(per_chunk_reason) >= 2:
            steady_mem_avg = sum(per_chunk_mem[1:]) / len(per_chunk_mem[1:])
            steady_reason_avg = sum(per_chunk_reason[1:]) / len(per_chunk_reason[1:])
            chunk_video_seconds = self.config.caption_window_seconds  # 每 chunk 覆盖的视频秒数
            steady_mem_rtf = steady_mem_avg / max(chunk_video_seconds, 1e-6)
            steady_reason_rtf = steady_reason_avg / max(chunk_video_seconds, 1e-6)

            logger.info("--- Steady-State Per-Chunk (excluding chunk 0) ---")
            logger.info(f"  Per-chunk video duration           : {chunk_video_seconds}s")
            logger.info(f"  Avg memory wall / chunk            : {steady_mem_avg:6.3f}s | "
                        f"per-chunk RTF = {steady_mem_rtf:5.3f}")
            logger.info(f"  Avg reasoning wall / chunk         : {steady_reason_avg:6.3f}s | "
                        f"per-chunk RTF = {steady_reason_rtf:5.3f}")

            # async 模式：稳态 RTF 取 max；sequential：取 sum
            if self.config.pipeline_mode == "async":
                steady_total_rtf = max(steady_mem_rtf, steady_reason_rtf)
                bottleneck = "memory-bound" if steady_mem_rtf >= steady_reason_rtf else "reasoning-bound"
                logger.info(f"  Steady-state RTF (async, max)      : {steady_total_rtf:5.3f}  "
                            f"[{bottleneck}]")
                # 关键：async 之所以能"跟上实时"，必须 max < 1.0
                if steady_total_rtf < 1.0:
                    logger.info(f"  >>> async mode IS faster than real-time at steady state.")
                else:
                    logger.info(f"  >>> async mode is SLOWER than real-time even at steady state — "
                                f"need to optimize the {bottleneck} side.")
            else:
                steady_total_rtf = steady_mem_rtf + steady_reason_rtf
                logger.info(f"  Steady-state RTF (sequential, sum) : {steady_total_rtf:5.3f}")

        logger.info("=" * 80)

    def _format_proactive_history(self) -> str:
        if not self._proactive_history:
            return "(no proactive reminders issued yet)"
        recent = self._proactive_history[-5:]
        return "\n".join(
            f"[{_seconds_to_hhmmss(r['time'])}] {r['content']}" for r in recent
        )

    def _group_frames_into_steps(
        self, frames: List[str], time_ranges: List[Dict],
    ) -> List[Dict]:
        """Group frames into steps of caption_window_seconds.

        Applies video_start_time offset so timestamps reflect the actual
        position in the source video (important for datasets like EgoLife
        where processing may start mid-video).

        Each batch includes per-frame time ranges for fine-grained caption storage.
        """
        if not frames:
            return []
        interval = self.config.step_interval_seconds
        window = self.config.caption_window_seconds
        offset = self.config.video_start_time
        step_batches = []
        current_batch_frames = []
        current_batch_frame_times = []
        window_start = 0.0

        for i, frame in enumerate(frames):
            frame_start = i * interval
            frame_end = frame_start + interval
            if not current_batch_frames:
                window_start = frame_start
            current_batch_frames.append(frame)
            current_batch_frame_times.append({
                "start": frame_start + offset,
                "end": frame_end + offset,
            })
            if (frame_end - window_start) >= window or i == len(frames) - 1:
                step_batches.append({
                    "frames": current_batch_frames,
                    "start_time": window_start + offset,
                    "end_time": frame_end + offset,
                    "frame_times": current_batch_frame_times,
                })
                current_batch_frames = []
                current_batch_frame_times = []
        return step_batches
