import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    # Frame sampling
    step_interval_seconds: float = 5.0
    frames_per_step: int = 2

    # Video start time offset (for datasets like EgoLife where processing starts mid-video)
    video_start_time: float = 0.0

    # Caption hierarchy
    caption_window_seconds: int = 10       # second level 窗口大小（秒）
    enable_multi_level_caption: bool = False  # 是否启用多层 caption（minute + hour），False 则只用 second level
    caption_window_minutes: float = 1.0    # minute level 窗口大小（分钟），累积 second captions
    caption_window_hours: float = 5/60     # hour level 窗口大小（小时），累积 minute captions
    gap_threshold_seconds: float = 60.0    # gap 检测阈值（秒），超过此间隔则截断当前窗口

    # 知识图谱实体提取方式
    #   "simple": 简化版 demo prompt（快，实体=I/Object/Location，事件作为关系）
    #   "full":   原始 VideoRAG 完整提取，由 VideoRAG 直接调用 ego_prompt_["entity_extraction"]
    #            （慢，实体含 event 类型，经过 gleaning 迭代）
    # rebuttal 测试默认走 full —— 与 reviewer 引用的论文版本一致。
    kg_extraction_mode: str = "full"

    # Reasoning VLM (local, receives video frames + caption context)
    reasoning_model_path: str = os.environ.get("QWEN3VL_8B_PATH", "./pretrained_weights/Qwen3-VL-8B-Instruct")

    # Caption model (local VLM for memory construction, Qwen3.5-4B or 9B)
    caption_model: str = "qwenvl_3_5_4b"

    # Working directory for cache and checkpoints
    working_dir: str = "./egomemo_cache"

    # Pipeline execution mode:
    #   "sequential" - build memory first, then reason/retrieve (for research/benchmarking)
    #   "async"      - memory construction and reasoning/retrieval run concurrently
    pipeline_mode: str = "sequential"

    # Question queue
    question_timeout_seconds: float = 300.0
    max_concurrent_questions: int = 10

    # Proactive service
    enable_proactive: bool = True
    proactive_cooldown_seconds: float = 30.0

    # Memory retrieval
    retrieval_top_k: int = 5

    # Recent history window (number of past captions to include in prompt)
    recent_history_count: int = 3

    # Web UI
    host: str = "0.0.0.0"
    port: int = 8765

    # Upload directory for videos
    upload_dir: str = "./egomemo_uploads"

    # Dataset type identifier for prompt selection
    datasets_type: str = "egomemo"

    # Whether to clear the per-video cache folder before processing
    #   True  = fresh start, rebuild all memory from scratch
    #   False = reuse existing memory (for resume / incremental)
    clear_cache: bool = True
