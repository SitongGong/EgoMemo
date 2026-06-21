#!/usr/bin/env python3
"""
CLI entry point for the EgoMemo streaming pipeline.

Usage:
    # Process a video with pre-scheduled questions (CLI mode)
    python -m egomemo.run --video_path /path/to/video.mp4 \
        --questions '[{"text": "What is the person doing?", "timestamp": 10.0}]'

    # Launch the web UI
    python -m egomemo.run --web --port 8765

    # Process with specific models
    python -m egomemo.run --video_path /path/to/video.mp4 \
        --caption_model gemini_api --reasoning_model gpt-4o
"""

import argparse
import json
import logging
import os
import sys

# Ensure EgoServe-RL root is FIRST on path (before VideoRAG, which has its own egomemo/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 开源布局下 videorag 与 egomemo 平级，默认指向仓库根目录；可用 VIDEORAG_ROOT 覆盖
_VIDEORAG_ROOT = os.environ.get("VIDEORAG_ROOT", _ROOT)
# VideoRAG first (lower priority), then EgoServe-RL (higher priority, inserted at 0)
if _VIDEORAG_ROOT not in sys.path:
    sys.path.insert(0, _VIDEORAG_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from egomemo.config import PipelineConfig
from egomemo.streaming_pipeline import StreamingPipeline, create_default_llm_config

from egomemo.pipeline_v2_patch import patch_pipeline_v2
patch_pipeline_v2()

# 如果同时要用加速版检索：
from egomemo.memory_bridge_fast import patch_memory_bridge_fast_read
patch_memory_bridge_fast_read()

# rebuttal 测试：精简 ego_prompt_["entity_extraction"]，砍 ~30% 长度，
# 加速 nano-graphrag entity LLM 调用（建图阶段瓶颈）。
from egomemo.entity_prompt_trim import patch_ego_prompt_entity_extraction
patch_ego_prompt_entity_extraction()

# 请通过环境变量设置 OPENAI_API_KEY（不再硬编码）：export OPENAI_API_KEY=...
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("egomemo")


def cli_event_callback(event_type: str, payload: dict):
    """Simple CLI callback that prints events to stdout."""
    if event_type == "step_complete":
        step = payload.get("step", "?")
        t = payload.get("time", 0)
        actions = payload.get("actions", [])
        print(f"  [Step {step}] t={t:.1f}s | actions: {actions}")
        for output in payload.get("outputs", []):
            if output.get("type") == "answer":
                print(f"    >> ANSWER {output['qid']}: {output['answer']}")
            elif output.get("type") == "proactive":
                print(f"    >> PROACTIVE: {output['content']}")
    elif event_type == "question_added":
        print(f"  [+] Question {payload['qid']} added at t={payload['timestamp']}s: {payload['text']}")
    elif event_type == "question_timeout":
        print(f"  [!] Question {payload['qid']} timed out")
    elif event_type == "processing_started":
        print(f"\n{'='*60}")
        print(f"  Processing: {payload['video_path']}")
        print(f"{'='*60}")
    elif event_type == "processing_complete":
        print(f"\n{'='*60}")
        print(f"  Processing complete!")
        print(f"  Steps: {payload.get('total_steps', '?')}")
        print(f"  Answers: {payload.get('total_answers', '?')}")
        print(f"  Proactive: {payload.get('total_proactive', '?')}")
        print(f"{'='*60}")
    elif event_type == "error":
        print(f"  [ERROR] {payload.get('message', 'Unknown error')}")


def run_cli(args):
    """Run the pipeline in CLI mode on a single video."""
    config = PipelineConfig(
        working_dir=args.working_dir,
        caption_model=args.caption_model,
        reasoning_model_path=args.reasoning_model_path,
        video_start_time=args.video_start_time,
        step_interval_seconds=args.step_interval,
        caption_window_seconds=args.caption_window,
        enable_multi_level_caption=args.multi_level_caption,
        caption_window_minutes=args.caption_window_minutes,
        caption_window_hours=args.caption_window_hours,
        question_timeout_seconds=args.question_timeout,
        enable_proactive=args.enable_proactive and not args.disable_proactive,
        proactive_cooldown_seconds=args.proactive_cooldown,
        recent_history_count=args.recent_history_count,
        datasets_type=args.datasets_type,
        pipeline_mode=args.pipeline_mode,
        clear_cache=args.clear_cache and not args.no_clear_cache,
    )

    llm_config = create_default_llm_config()

    pipeline = StreamingPipeline(
        config=config,
        llm_config=llm_config,
        event_callback=cli_event_callback,
    )

    # CLI 模式没有前端 TTS 播放回环，frontend_ready 必须始终为 set，否则
    # async/sequential 推理循环（streaming_pipeline 中 wait/while 等待 frontend_ready）
    # 会永久阻塞。把 Event 替换成一个"永远 set"的代理对象。
    class _AlwaysSetEvent:
        def is_set(self): return True
        def set(self): pass
        def clear(self): pass  # 不能真清除，否则会卡死
        def wait(self, timeout=None): return True
    pipeline._frontend_ready = _AlwaysSetEvent()

    # Parse questions JSON
    questions = None
    if args.questions:
        try:
            questions = json.loads(args.questions)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid questions JSON: {e}")
            sys.exit(1)

    # Run pipeline
    trajectory = pipeline.run_on_video(args.video_path, questions)

    # Save results (in per-video subdirectory)
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    video_dir = os.path.join(config.working_dir, video_name)
    output_path = os.path.join(video_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def run_web(args):
    """Launch the web UI."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required for the web UI. Install it with:")
        print("  pip install uvicorn fastapi python-multipart")
        sys.exit(1)

    from egomemo.web.app import create_app

    app = create_app(
        default_config=PipelineConfig(
            working_dir=args.working_dir,
            caption_model=args.caption_model,
            reasoning_model_path=args.reasoning_model_path,
            upload_dir=args.upload_dir,
            datasets_type=args.datasets_type,
            pipeline_mode="async",
        ),
        tts_enabled=args.tts,
        tts_backend=args.tts_backend,
        tts_voice=args.tts_voice,
        preset_config_path=args.preset_config,
    )

    print(f"\nEgoMemo Web UI starting at http://{args.host}:{args.port}")
    if args.tts:
        print(f"TTS enabled: backend={args.tts_backend}, voice={args.tts_voice or 'default'}")
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=300)


def main():
    parser = argparse.ArgumentParser(
        description="EgoMemo: Training-Free Streaming Pipeline for Egocentric Video"
    )

    # Mode selection
    parser.add_argument(
        "--web", default=False, action="store_true", help="Launch web UI instead of CLI mode"
    )

    # Video / questions
    parser.add_argument(
        "--video_path", type=str, default="./sample_video.mp4", help="Path to video file (CLI mode)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="[]",
        help='JSON array of questions: [{"text": "...", "timestamp": 10.0, "recurring": false}]. '
             'Set "recurring": true for ongoing questions that need answers at every relevant step. '
             'rebuttal 测试默认空 —— 仅评估 proactive 主动服务能力。',
    )

    # Model configuration
    parser.add_argument(
        "--caption_model",
        type=str,
        default="qwen_vllm",
        help="Caption model name: qwen_vllm (本地 vLLM 服务，默认) / qwenvl_3_8b_instruct (本地权重) / qwen3_api 等",
    )
    parser.add_argument(
        "--reasoning_model_path",
        type=str,
        default="gpt-5-mini",  # 或本地权重路径，如 "./pretrained_weights/Qwen3-VL-8B-Instruct"
        help="Path to reasoning VLM (base Qwen3-VL or fine-tuned checkpoint)",
    )
    parser.add_argument(
        "--datasets_type",
        type=str,
        default="egolife",
        help="Dataset type: egolife, holoassist, eyewo, proassist, etc. (default: egolife)",
    )

    # Pipeline parameters
    parser.add_argument(
        "--video_start_time",
        type=float,
        default=0.0,
        help="Video start time offset in seconds (default: 0.0). "
             "For datasets like EgoLife where processing starts mid-video.",
    )
    parser.add_argument(
        "--step_interval",
        type=float,
        default=5.0,
        help="Frame sampling interval in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--caption_window",
        type=int,
        default=30,
        help="Caption window size in seconds (default: 10)",
    )
    parser.add_argument(
        "--multi_level_caption",
        action="store_true",
        default=False,
        help="Enable multi-level caption hierarchy (second + minute + hour). "
             "When disabled, only second-level captions are generated.",
    )
    parser.add_argument(
        "--caption_window_minutes",
        type=float,
        default=5.0,
        help="Minute-level caption window in minutes (default: 1.0)",
    )
    parser.add_argument(
        "--caption_window_hours",
        type=float,
        default=1,
        help="Hour-level caption window in hours (default: 5/60 = 5 min)",
    )
    parser.add_argument(
        "--question_timeout",
        type=float,
        default=300.0,
        help="Question timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--gap_threshold",
        type=float,
        default=60.0,
        help="Time gap threshold in seconds to reset caption window (default: 60)",
    )
    parser.add_argument(
        "--proactive_cooldown",
        type=float,
        default=5.0,
        help="Minimum seconds between proactive reminders (default: 30)",
    )
    parser.add_argument(
        "--enable_proactive",
        action="store_true",
        default=True,
        help="Enable proactive service (default: True)",
    )
    parser.add_argument(
        "--disable_proactive",
        action="store_true",
        default=False,
        help="Disable proactive service",
    )
    parser.add_argument(
        "--recent_history_count",
        type=int,
        default=1,
        help="Number of recent captions to include in prompt context (default: 3)",
    )
    parser.add_argument(
        "--pipeline_mode",
        type=str,
        default="async",
        choices=["sequential", "async"],
        help="Pipeline mode: 'sequential' (memory then reason, for research) "
             "or 'async' (parallel, for real-time). Default: sequential",
    )

    # Cache control
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        default=True,
        help="Clear per-video cache before processing (default: True, fresh start)",
    )
    parser.add_argument(
        "--no_clear_cache",
        action="store_true",
        default=False,
        help="Keep existing cache and resume/reuse memory",
    )

    # Directories
    parser.add_argument(
        "--working_dir",
        type=str,
        default="./egomemo_cache",
        help="Working directory for cache (default: ./egomemo_cache)",
    )
    parser.add_argument(
        "--upload_dir",
        type=str,
        default="./egomemo_uploads",
        help="Upload directory for web UI (default: ./egomemo_uploads)",
    )

    # Web UI
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Web UI host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8090, help="Web UI port (default: 8090；避开 vLLM 的 8000/8001)"
    )

    # TTS (web mode only — audio is played in the browser)
    parser.add_argument(
        "--tts", action="store_true", default=True,
        help="Enable TTS in web mode: synthesize audio for answers/reminders",
    )
    parser.add_argument(
        "--tts_backend", type=str, default="edge",
        choices=["edge", "openai"],
        help="TTS backend: 'edge' (free) or 'openai' (best quality, needs API key)",
    )
    parser.add_argument(
        "--tts_voice", type=str, default=None,
        help="TTS voice. Edge: en-US-AriaNeural, en-US-GuyNeural. OpenAI: nova, alloy, shimmer",
    )

    # Preset video (web UI 点击按钮直接加载服务器上的视频，避免慢上传)
    parser.add_argument(
        "--preset_config", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "preset_video.yaml"),
        help="预设视频 YAML 配置路径。web 界面会渲染一个按钮，点击直接加载此视频。",
    )

    args = parser.parse_args()

    if args.web:
        run_web(args)
    else:
        if not args.video_path:
            parser.error("--video_path is required in CLI mode")
        if not os.path.exists(args.video_path):
            parser.error(f"Video file not found: {args.video_path}")
        run_cli(args)


if __name__ == "__main__":
    main()
