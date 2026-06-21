# EgoMemo — Streaming Egocentric Video Assistant

**English | [简体中文](README.zh-CN.md)**

EgoMemo is a **training-free** streaming system for first-person (egocentric) video understanding and proactive assistance. As video streams in time-slice by time-slice, the system transcribes the scene into searchable long-term memory (captions + knowledge graph + vector store) while, in parallel, **deciding on its own whether to proactively alert the user** and answering user questions in real time. It runs in both CLI and Web modes; the Web UI shows a live reasoning stream and speaks answers via TTS.

---

## 1. Features

- **Streaming processing**: video is processed incrementally in fixed time-slices (steps); memory construction and reasoning run in parallel, faster than real-time end-to-end.
- **Long-term memory**: captions are written into a knowledge graph plus text/visual vector stores, enabling cross-time retrieval.
- **Proactive service**: without any user question, the system decides whether the current scene warrants a proactive reminder (e.g. safety hazards, unfinished actions).
- **Real-time Q&A**: users can ask anytime; the system retrieves memory and answers. Both one-time and recurring questions are supported.
- **Multiple caption backends**: local vLLM service, local weights, or various cloud APIs — switchable on demand.
- **Web UI**: live "thought stream" (silent / search / respond), answer bubbles, an avatar, and voice playback.

---

## 2. How It Works

### 2.1 Modules

| Module | File / Class | Responsibility |
|---|---|---|
| Entry / Orchestration | `egomemo/run.py` → `StreamingPipeline` (`streaming_pipeline.py`) | CLI / Web entry points; drives the main loop step by step |
| Memory bridge | `MemoryBridge` (`memory_bridge.py` / `memory_bridge_fast.py`) | Writes captions into the knowledge graph / vector store and exposes retrieval |
| Underlying memory engine | `videorag.egograph_retrieval_optimize_.VideoGraphSeparated` | Multi-store backend: text chunks (KV), vector stores, knowledge graph (NetworkX), visual feature store |
| Reasoning / Proactive | `pipeline_v2_patch.py` | Two-stage (Pass1 / Pass2) reasoning that decides silent / search / respond |
| Action router | `ActionRouter` (`action_router.py`) | Parses the model's `[silent] / [search] / [respond]` output |
| Question queue | `QuestionQueueManager` (`question_queue.py`) | State machine for user questions (one-time / recurring) |
| Working memory + Output | `WorkingMemory`, `OutputRecorder` | Dialogue history / evidence; writes trajectory to `results.json` |

### 2.2 Data Flow

```
Video file
  │  sample frames every step_interval (base64), group into "steps" by caption_window
  ▼
Each step
  ├─[ Memory construction ]──────────────────────────────────────
  │   1. Caption: frames → caption VLM (local vLLM Qwen3-VL by default)
  │   2. Knowledge graph: caption → entity extraction + text embedding
  │   3. Visual embedding: video segment → ImageBind feature store
  │
  └─[ Reasoning / Service ]──────────────────────────────────────
      Pass1: from "current caption + history", decide silent / search / respond
        ├ silent  → stay quiet
        ├ respond → emit a proactive reminder directly
        └ search  → retrieve long-term memory → Pass2 decides whether to respond
      (the reasoning model reads only caption text, not raw frames)
  ▼
Output events → CLI print / Web WebSocket pushes "thought stream + answer" + TTS audio
  ▼
Persist: results.json + working_memory.json + store indexes
```

### 2.3 Key Design

- **Reasoning reads text, not images**: video understanding happens entirely at the caption stage; reasoning consumes only caption text, so reasoning latency is very low.
- **Memory and reasoning in parallel** (async mode): a background thread builds memory while the main thread reasons; the bottleneck is `max` of the two, not their `sum`.
- **Two triggers**: ① proactive — the system decides whether to speak up; ② user questions — queued and answered step by step.

---

## 3. External Services / Models

| Dependency | Used for | Configuration |
|---|---|---|
| **Local vLLM service** (Qwen3-VL-8B, OpenAI-compatible) | Caption generation (when `caption_model` contains `vllm`) | env var `VLLM_BASE_URL`, default `http://localhost:8000/v1` |
| **OpenAI API** (gpt-5-mini / text-embedding-3-small / gpt-4o-mini) | Reasoning, text/entity embedding, KG entity extraction | env var `OPENAI_API_KEY` |
| **ImageBind** (local weights) | Visual embedding (optional; degrades gracefully if missing, text retrieval still works) | place `imagebind_huge.pth` under `egomemo/.checkpoints/` |
| **TTS** (Web only): edge-tts (free, default) / OpenAI tts | Voice synthesis for answers | `--tts_backend` |

> Caption and reasoning backends can also be swapped (see §6). All API keys are supplied via environment variables; the repository contains no hard-coded secrets.

---

## 4. Setup

> A dedicated conda environment is recommended to avoid polluting other projects.

```bash
conda create -n egomemo python=3.10 -y
conda activate egomemo

# vLLM server side (must match your local NVIDIA driver; e.g. use a cu12 build for CUDA 12.x)
pip install "vllm==0.11.0" "transformers==4.57.1"

# Remaining dependencies
pip install -r requirements.txt

# ImageBind (visual embedding, optional): download imagebind_huge.pth to
#   egomemo/.checkpoints/imagebind_huge.pth
#   https://github.com/facebookresearch/ImageBind
```

Make `videorag` and `egomemo` importable (the two packages sit side by side under the repo root):

```bash
# run from the repo root
export PYTHONPATH="$PWD:$PWD/egomemo:$PYTHONPATH"
```

Configure API keys and (optionally) model paths:

```bash
export OPENAI_API_KEY="sk-..."                 # required for reasoning + embedding
export VLLM_BASE_URL="http://localhost:8000/v1"
# If using local weights for caption/reasoning, set the path:
export QWEN3VL_8B_PATH="./pretrained_weights/Qwen3-VL-8B-Instruct"
```

> **Version note**: vLLM/torch must match your local NVIDIA driver. For example, with a CUDA 12.8 driver, use the cu12 build of vLLM 0.11.0 (torch 2.8.0+cu128); newer drivers allow newer versions.

---

## 5. Running

### 5.1 Start the vLLM service used for captioning

```bash
# Download Qwen3-VL-8B-Instruct weights to ./pretrained_weights/ or point MODEL_DIR to them
cd scripts
MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct GPU_ID=0 PORT=8000 \
    bash launch_vllm_qwen3vl_v023.sh
# Wait for "Application startup complete"; confirm curl http://localhost:8000/v1/models returns 200
```

### 5.2 Web mode (recommended)

```bash
export OPENAI_API_KEY="sk-..."
export VLLM_BASE_URL="http://localhost:8000/v1"

python -m egomemo.run --web        # defaults: caption=qwen_vllm, reasoning=gpt-5-mini, port=8090
```

Open `http://<host>:8090`: upload a video → (optionally uncheck Cache to rebuild memory) → Start Processing.
The "REASONING PROCESS" panel on the right shows the proactive thought stream in real time; you can ask the assistant questions in the input box, and answers are spoken aloud.

### 5.3 CLI mode

```bash
python -m egomemo.run \
    --video_path /path/to/video.mp4 \
    --caption_model qwen_vllm \
    --reasoning_model_path gpt-5-mini \
    --pipeline_mode async \
    --working_dir ./egomemo_cache
```

Results are written to `./egomemo_cache/<video_name>/results.json` (answer trajectory and timings).

Preset questions (optional):
```bash
--questions '[{"text": "What am I doing now?", "timestamp": 10.0, "recurring": false}]'
```

---

## 6. Configuration

| Argument (CLI) | Default | Description |
|---|---|---|
| `--caption_model` | `qwen_vllm` | Caption backend, see table below |
| `--reasoning_model_path` | `gpt-5-mini` | Reasoning model; a local Qwen3-VL path runs local inference |
| `--pipeline_mode` | `async` | `async` = memory/reasoning in parallel; `sequential` = build full memory first, then reason |
| `--datasets_type` | `egolife` | Dataset type, affects prompt style |
| `--port` | `8090` | Web UI port (avoids vLLM's 8000/8001) |
| `--tts` / `--tts_backend` | on / `edge` | Web voice playback backend |

Other options in `config.py`: `step_interval_seconds` (sampling interval), `caption_window_seconds` (step length), `proactive_cooldown_seconds` (proactive reminder cooldown), `clear_cache` (whether to clear old cache), `kg_extraction_mode` (`full` / `simple`).

**Caption backend options** (`load_caption_model` dispatches by keyword in `model_name`):

| `model_name` contains | Backend |
|---|---|
| `vllm` | Local vLLM service (OpenAI-compatible HTTP, no in-process weight loading) |
| `qwen3` + `api` | DashScope Qwen3-VL API (needs `DASHSCOPE_API_KEY`) |
| `gpt_4o` / `gpt4o` | GPT-4o API |
| `gemini` + `api` | Gemini API (needs `GEMINI_API_KEY`) |
| otherwise (matches MODEL_MAP) | Load local weights via transformers (e.g. `qwenvl_3_8b_instruct`) |

---

## 7. Directory Layout

```
.
├── egomemo/              # Main program (pipeline, reasoning, Web)
│   ├── run.py              # Entry point (CLI / Web)
│   ├── streaming_pipeline.py
│   ├── memory_bridge.py / memory_bridge_fast.py
│   ├── pipeline_v2_patch.py
│   ├── action_router.py / question_queue.py / working_memory.py
│   ├── prompt_templates*.py
│   └── web/                # FastAPI server + frontend static assets
├── videorag/               # Underlying memory engine (knowledge graph / vector store / caption backends)
├── scripts/
│   └── launch_vllm_qwen3vl_v023.sh   # Launch the local vLLM caption service
├── requirements.txt
├── README.md
└── README.zh-CN.md
```

> At runtime the system creates `egomemo_cache/`, `egomemo_uploads/`, `web/static/tts/`, etc., plus the user-downloaded `egomemo/.checkpoints/imagebind_huge.pth` and `pretrained_weights/`. These are all excluded via `.gitignore`.

---

## 8. FAQ

- **Reasoning panel is all SILENT / "Nothing relevant"**: usually `OPENAI_API_KEY` is unset or invalid — when reasoning calls fail they fall back to silent, and failed embeddings also prevent captions from being stored. Check the run logs for `401 / invalid_api_key`.
- **Caption generation reports a connection error**: the vLLM service is not running or `VLLM_BASE_URL` is wrong. Confirm `curl http://localhost:8000/v1/models` returns 200.
- **`No module named 'videorag'`**: `PYTHONPATH` is not set, see §4.
- **`moviepy.editor` missing**: moviepy 2.x removed this module; install `pip install "moviepy<2.0"` (pinned in requirements).
- **Memory not persisted**: the process must exit normally to flush indexes to disk; force-killing loses data.
