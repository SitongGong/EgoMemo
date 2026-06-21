"""
FastAPI backend for the EgoMemo web UI.

Endpoints:
- POST /api/upload_video     - Upload a video file
- POST /api/start_processing - Start pipeline processing
- POST /api/ask_question     - Inject a question mid-processing
- POST /api/follow_up        - Follow up on a proactive service
- GET  /api/trajectory       - Get full decision trajectory
- GET  /api/status           - Get pipeline status
- WS   /ws                   - WebSocket for real-time events
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import threading
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure EgoServe-RL root is first on path (before VideoRAG which has its own egomemo/)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 开源布局下 videorag 与 egomemo 平级，默认指向仓库根目录；可用 VIDEORAG_ROOT 覆盖
_VIDEORAG_ROOT = os.environ.get("VIDEORAG_ROOT", _ROOT)
if _VIDEORAG_ROOT not in sys.path:
    sys.path.append(_VIDEORAG_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from egomemo.config import PipelineConfig
from egomemo.streaming_pipeline import StreamingPipeline, create_default_llm_config

logger = logging.getLogger(__name__)

# Global state
_pipeline: Optional[StreamingPipeline] = None
_connected_websockets: List[WebSocket] = []
_ws_lock = threading.Lock()
_tts_enabled: bool = False
_tts_backend: str = "edge"
_tts_voice: Optional[str] = None
_tts_audio_dir: Optional[str] = None

# 缓存 TTS URL：key=(qid_or_event_id, time)，value=url 或 "" 表示失败
_TTS_URL_CACHE: Dict[str, str] = {}


def _generate_tts_audio(text: str, voice: Optional[str] = None) -> Optional[str]:
    """Generate TTS audio file and return its URL path. Returns None on failure.

    Args:
        text: 要合成的文本
        voice: 覆盖默认的 TTS 声音；传 None 则用启动时指定的 _tts_voice。
               对 edge 后端：如 "en-US-GuyNeural"（男）、"en-US-AriaNeural"（女）。
               对 openai 后端：如 "onyx"/"echo"（男）、"nova"/"shimmer"（女）。
    """
    if not _tts_enabled or not text or not _tts_audio_dir:
        return None
    if text.startswith("(") and text.endswith(")"):
        return None

    try:
        filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(_tts_audio_dir, filename)
        effective_voice = voice or _tts_voice

        if _tts_backend == "openai":
            from openai import OpenAI
            client = OpenAI()
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice=effective_voice or "nova",
                input=text,
            )
            response.stream_to_file(filepath)
        else:
            # Edge TTS (free) — run in a new thread to avoid event loop conflicts
            import edge_tts
            import concurrent.futures

            def _run_edge_tts():
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(
                        edge_tts.Communicate(text, effective_voice or "en-US-AnaNeural").save(filepath)
                    )
                finally:
                    loop.close()

            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(_run_edge_tts).result(timeout=30)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return f"/static/tts/{filename}"
    except Exception as e:
        logger.warning(f"TTS generation failed: {e}")
    return None


def _broadcast_event(event_type: str, payload: dict):
    """Send event to all connected WebSocket clients (thread-safe).

    同一个答案在 answer_ready 和 step_complete 会被广播两次；
    按 (qid_or_event_id, time) 缓存 TTS URL，确保前端两次收到的 url 一致，
    且只真正调 edge-tts 一次。
    """

    def _tts_cache_key(qid_or_eid, time_v):
        if qid_or_eid is None: qid_or_eid = ""
        return f"{qid_or_eid}@{time_v}"

    def _get_or_make_tts(qid_or_eid, time_v, text):
        if not text:
            return None
        key = _tts_cache_key(qid_or_eid, time_v)
        cached = _TTS_URL_CACHE.get(key)
        if cached is not None:
            # sentinel "" 表示之前尝试过但失败，直接返回 None 不再重试
            return cached or None
        url = _generate_tts_audio(text) or ""
        _TTS_URL_CACHE[key] = url
        # 简单防无限增长
        if len(_TTS_URL_CACHE) > 2000:
            for k in list(_TTS_URL_CACHE.keys())[:1000]:
                _TTS_URL_CACHE.pop(k, None)
        return url or None

    if _tts_enabled and event_type == "step_complete":
        for output in payload.get("outputs", []):
            if output.get("type") == "answer":
                text = output.get("answer")
                key_id = output.get("qid")
            elif output.get("type") == "proactive":
                text = output.get("content")
                key_id = output.get("event_id")
            else:
                continue
            url = _get_or_make_tts(key_id, output.get("time"), text)
            if url:
                output["tts_audio_url"] = url

    if _tts_enabled and event_type == "answer_ready":
        # answer_ready 里 proactive 的 qid 字段其实是 event_id（见 pipeline_v2_patch）
        text = payload.get("answer")
        key_id = payload.get("qid")
        url = _get_or_make_tts(key_id, payload.get("time"), text)
        if url:
            payload["tts_audio_url"] = url

    message = json.dumps({"type": event_type, **payload}, ensure_ascii=False, default=str)
    with _ws_lock:
        ws_list = list(_connected_websockets)

    for ws in ws_list:
        try:
            asyncio.run_coroutine_threadsafe(
                ws.send_text(message),
                _get_event_loop(),
            )
        except Exception:
            pass


_event_loop = None


def _get_event_loop():
    global _event_loop
    if _event_loop is None:
        _event_loop = asyncio.get_event_loop()
    return _event_loop


def _load_preset_video(preset_config_path: Optional[str]) -> Optional[dict]:
    """读取预设视频 YAML 配置。失败返回 None（非致命）。"""
    if not preset_config_path:
        return None
    if not os.path.exists(preset_config_path):
        logger.warning(f"preset_config not found: {preset_config_path}")
        return None
    try:
        import yaml
        with open(preset_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"failed to load preset_config: {e}")
        return None

    path = cfg.get("path")
    if not path or not os.path.exists(path):
        logger.warning(f"preset video path invalid or missing: {path}")
        return None
    label = cfg.get("label") or f"Load: {os.path.basename(path)}"
    return {"path": os.path.abspath(path), "label": label, "filename": os.path.basename(path)}


def create_app(
    default_config: Optional[PipelineConfig] = None,
    tts_enabled: bool = False,
    tts_backend: str = "edge",
    tts_voice: Optional[str] = None,
    preset_config_path: Optional[str] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _tts_enabled, _tts_backend, _tts_voice, _tts_audio_dir

    app = FastAPI(title="EgoMemo", description="Egocentric Video Streaming Pipeline")

    config = default_config or PipelineConfig()
    os.makedirs(config.upload_dir, exist_ok=True)

    # 预设视频（可选）：用户在 YAML 里配一个服务器上的视频路径，
    # 前端按按钮直接加载，避免慢速上传链路。
    preset_video = _load_preset_video(preset_config_path)
    if preset_video:
        logger.info(f"preset video ready: {preset_video['path']}")

    # TTS setup
    _tts_enabled = tts_enabled
    _tts_backend = tts_backend
    _tts_voice = tts_voice

    @app.on_event("startup")
    async def on_startup():
        global _event_loop
        _event_loop = asyncio.get_event_loop()

    # --- Static files ---
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)

    # TTS audio directory (served as /static/tts/*.mp3)
    tts_dir = os.path.join(static_dir, "tts")
    os.makedirs(tts_dir, exist_ok=True)
    _tts_audio_dir = tts_dir

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        return JSONResponse({"error": "Frontend not found"}, status_code=404)

    @app.get("/test_params")
    async def test_params_page():
        path = os.path.join(static_dir, "test_params.html")
        if os.path.exists(path):
            return FileResponse(path)
        return JSONResponse({"error": "Test page not found"}, status_code=404)

    # --- API Endpoints ---

    @app.get("/api/preset_video")
    async def get_preset_video():
        """返回预设视频的元数据；前端用它渲染按钮并获取 stream URL。"""
        if not preset_video:
            return {"available": False}
        return {
            "available": True,
            "label": preset_video["label"],
            "filename": preset_video["filename"],
            "path": preset_video["path"],          # 后端 pipeline 直接用这个绝对路径
            "stream_url": "/preset_video_stream",  # <video> 通过 HTTP range 播放
        }

    @app.api_route("/preset_video_stream", methods=["GET", "HEAD"])
    async def stream_preset_video():
        """把预设视频当静态文件吐给浏览器（FastAPI FileResponse 自带 Range 支持）。
        需要显式声明 HEAD：浏览器 <video> 元素在加载前通常先发 HEAD 探测。
        """
        if not preset_video:
            return JSONResponse({"error": "no preset video configured"}, status_code=404)
        return FileResponse(preset_video["path"], media_type="video/mp4")

    @app.post("/api/test_avatar")
    async def test_avatar(
        text: str = Form("Hello, I can see you are making coffee."),
        voice: Optional[str] = Form(None),
    ):
        """Test endpoint: generate TTS audio and return URL for Live2D lip sync.

        可选参数 voice 覆盖默认声音，例如 "en-US-GuyNeural"（edge 男声）。
        """
        if not _tts_enabled:
            return JSONResponse({"error": "TTS not enabled. Start with --tts flag."}, status_code=400)
        audio_url = _generate_tts_audio(text, voice=voice)
        if audio_url:
            return {"status": "ok", "text": text, "voice": voice, "tts_audio_url": audio_url}
        return JSONResponse({"error": "TTS generation failed"}, status_code=500)

    @app.get("/api/test_avatar")
    async def test_avatar_get(
        text: str = "Hello, I can see you are making coffee.",
        voice: Optional[str] = None,
    ):
        """GET version for easy browser testing."""
        if not _tts_enabled:
            return JSONResponse({"error": "TTS not enabled. Start with --tts flag."}, status_code=400)
        audio_url = _generate_tts_audio(text, voice=voice)
        if audio_url:
            return {"status": "ok", "text": text, "voice": voice, "tts_audio_url": audio_url}
        return JSONResponse({"error": "TTS generation failed"}, status_code=500)

    @app.post("/api/upload_video")
    async def upload_video(file: UploadFile = File(...)):
        """Upload a video file and return its saved path.

        关键点：把整个"从 spooled tempfile 读 + 写 CPFS"的同步循环
        一次性扔到线程池里跑。不要在 async 函数里反复 await 切换，
        否则大视频会把事件循环拖死，WebSocket 超时断连。
        """
        filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        save_path = os.path.join(config.upload_dir, filename)
        os.makedirs(config.upload_dir, exist_ok=True)

        logger.info(f"upload_video: start receiving {file.filename} -> {save_path}")
        src = file.file  # 底层 SpooledTemporaryFile（阻塞 IO）
        CHUNK = 4 * 1024 * 1024  # 4MB

        def _copy_sync() -> int:
            total = 0
            with open(save_path, "wb") as dst:
                while True:
                    buf = src.read(CHUNK)
                    if not buf:
                        break
                    dst.write(buf)
                    total += len(buf)
            return total

        try:
            total = await asyncio.get_event_loop().run_in_executor(None, _copy_sync)
        except Exception as e:
            logger.exception("upload_video failed")
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
            except Exception:
                pass
            return JSONResponse({"error": f"upload failed: {e}"}, status_code=500)

        logger.info(f"upload_video: saved {total/1024/1024:.1f} MB -> {save_path}")
        return {"status": "ok", "video_path": save_path, "filename": filename, "size": total}

    @app.post("/api/start_processing")
    async def start_processing(
        video_path: str = Form(...),
        questions: str = Form("[]"),
        params: str = Form("{}"),
    ):
        """Start pipeline processing on a video."""
        global _pipeline

        if _pipeline and _pipeline.is_running:
            return JSONResponse(
                {"error": "Pipeline is already running"}, status_code=409
            )

        if not os.path.exists(video_path):
            return JSONResponse(
                {"error": f"Video not found: {video_path}"}, status_code=404
            )

        # 新一轮 run 开始，清空 TTS URL 缓存，避免跨 run 误命中
        _TTS_URL_CACHE.clear()

        # Parse questions
        try:
            questions_parsed = json.loads(questions)
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid questions JSON"}, status_code=400)

        # Parse pipeline parameters from frontend
        try:
            user_params = json.loads(params)
        except json.JSONDecodeError:
            user_params = {}

        # Create pipeline config, merging frontend params with server defaults
        # 前端可以通过 datasets_type 切换 demo 模式（egolife 会启用 demo proactive prompt）
        pipeline_config = PipelineConfig(
            working_dir=config.working_dir,
            caption_model=config.caption_model,
            reasoning_model_path=config.reasoning_model_path,
            video_start_time=config.video_start_time,
            upload_dir=config.upload_dir,
            datasets_type=str(user_params.get("datasets_type") or config.datasets_type),
            pipeline_mode="async",
            step_interval_seconds=float(user_params.get("step_interval", 2)),
            caption_window_seconds=int(user_params.get("caption_window", 10)),
            proactive_cooldown_seconds=float(user_params.get("proactive_cooldown", 30)),
            clear_cache=bool(user_params.get("clear_cache", True)),
        )

        # demo 专用 feature flag：视频最后一个 chunk 是否强制发饮水提醒。
        # 以属性方式挂到 config 对象上，dataclass 允许动态加字段。
        pipeline_config.hydration_reminder_enabled = bool(
            user_params.get("hydration_reminder_enabled", False)
        )
        # 场景扩展 flag：勾选后在 proactive system prompt 末尾追加 circuit-breaker 场景规则
        pipeline_config.circuit_breaker_scene_enabled = bool(
            user_params.get("circuit_breaker_scene_enabled", False)
        )
        # 场景扩展 flag：勾选后在 per-question system prompt 末尾追加鸡蛋羹做法引导规则
        pipeline_config.egg_recipe_guidance_enabled = bool(
            user_params.get("egg_recipe_guidance_enabled", False)
        )

        llm_config = create_default_llm_config()
        _pipeline = StreamingPipeline(
            config=pipeline_config,
            llm_config=llm_config,
            event_callback=_broadcast_event,
        )

        # Run in background thread
        def _run():
            _pipeline.run_on_video(video_path, questions_parsed or None)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        return {
            "status": "started",
            "video_path": video_path,
            "questions_count": len(questions_parsed),
        }

    @app.post("/api/ask_question")
    async def ask_question(
        text: str = Form(...),
        timestamp: float = Form(...),
        follow_up_parent: Optional[str] = Form(None),
    ):
        """Inject a question into the running pipeline."""
        if _pipeline is None:
            return JSONResponse({"error": "Pipeline not started"}, status_code=400)

        qid = _pipeline.inject_question(text, timestamp, follow_up_parent)
        return {"status": "ok", "qid": qid}

    @app.post("/api/follow_up")
    async def follow_up(
        text: str = Form(...),
        proactive_event_id: str = Form(...),
        timestamp: float = Form(0.0),
    ):
        """Follow up on a proactive service event."""
        if _pipeline is None:
            return JSONResponse({"error": "Pipeline not started"}, status_code=400)

        # Use current video time if not specified
        ts = timestamp if timestamp > 0 else _pipeline.current_video_time
        qid = _pipeline.inject_question(text, ts, follow_up_parent=proactive_event_id)
        return {"status": "ok", "qid": qid, "follow_up_parent": proactive_event_id}

    @app.get("/api/trajectory")
    async def get_trajectory():
        """Get the full decision trajectory."""
        if _pipeline:
            return _pipeline.recorder.get_full_trajectory()
        return {"error": "No pipeline running"}

    @app.get("/api/status")
    async def get_status():
        """Get current pipeline status."""
        if _pipeline is None:
            return {"status": "idle"}

        return {
            "status": "running" if _pipeline.is_running else "completed",
            "current_time": _pipeline.current_video_time,
            "total_questions": _pipeline.question_queue.total_count,
            "active_questions": _pipeline.question_queue.active_count,
            "steps_processed": _pipeline._step_count,
        }

    @app.get("/api/questions")
    async def get_questions():
        """Get all questions and their statuses."""
        if _pipeline is None:
            return {"questions": []}

        questions = _pipeline.question_queue.get_all_questions()
        return {
            "questions": [q.to_dict() for q in questions],
        }

    # --- WebSocket ---

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time event streaming."""
        await websocket.accept()
        with _ws_lock:
            _connected_websockets.append(websocket)
        logger.info("WebSocket client connected")

        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "ask_question" and _pipeline:
                    qid = _pipeline.inject_question(
                        msg["text"],
                        msg.get("timestamp", _pipeline.current_video_time),
                        msg.get("follow_up_parent"),
                        recurring=msg.get("recurring", False),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "question_added", "qid": qid})
                    )

                elif msg_type == "follow_up" and _pipeline:
                    ts = msg.get("timestamp", _pipeline.current_video_time)
                    qid = _pipeline.inject_question(
                        msg["text"], ts, msg.get("proactive_event_id"),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "question_added", "qid": qid, "is_follow_up": True})
                    )

                elif msg_type == "frontend_ready" and _pipeline:
                    # 前端 TTS 播完，通知后端可以继续下一个 step 的推理
                    _pipeline.notify_frontend_ready()
                    logger.info("[Sync] Frontend ready, backend proceeding to next step")

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            with _ws_lock:
                if websocket in _connected_websockets:
                    _connected_websockets.remove(websocket)
            logger.info("WebSocket client disconnected")

    return app
