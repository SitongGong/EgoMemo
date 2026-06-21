# EgoMemo — 流式自我中心视频助手 (Streaming Egocentric Video Assistant)

**[English](README.md) | 简体中文**

EgoMemo 是一个**训练无关 (training-free)** 的流式第一人称视频理解与主动服务系统。视频按时间片持续流入，系统一边把画面转写为可检索的长期记忆（caption + 知识图谱 + 向量库），一边基于记忆**主动判断是否该提醒用户**、并实时**回答用户提出的问题**。提供命令行 (CLI) 与 Web 两种运行方式，Web 端带实时思考流展示与语音 (TTS) 播报。

---

## 1. 功能特性

- **流式处理**：视频按固定时间片 (step) 增量处理，记忆构建与推理并行，端到端快于实时。
- **长期记忆**：caption 自动写入知识图谱 + 文本/视觉向量库，支持跨时间检索。
- **主动服务 (Proactive)**：无需用户提问，系统自行判断当前场景是否需要主动提醒（如安全隐患、未完成动作）。
- **实时问答**：用户可随时提问，系统检索记忆后作答；支持一次性问题与持续型问题。
- **多种 Caption 后端**：本地 vLLM 服务、本地权重、或各类云端 API，可按需切换。
- **Web 界面**：实时显示"思考流 (silent / search / respond)"、回答气泡、虚拟形象与语音播报。

---

## 2. 运行原理

### 2.1 模块组成

| 模块 | 文件 / 类 | 职责 |
|---|---|---|
| 入口 / 编排 | `egomemo/run.py` → `StreamingPipeline` (`streaming_pipeline.py`) | CLI / Web 两种入口，逐时间步驱动主循环 |
| 记忆桥接 | `MemoryBridge` (`memory_bridge.py` / `memory_bridge_fast.py`) | 把 caption 写入知识图谱/向量库，并提供检索接口 |
| 底层记忆引擎 | `videorag.egograph_retrieval_optimize_.VideoGraphSeparated` | 多存储后端：文本块(KV)、向量库、知识图谱(NetworkX)、视觉特征库 |
| 推理 / 主动服务 | `pipeline_v2_patch.py` | 两阶段 (Pass1 / Pass2) 推理，决定 silent / search / respond |
| 动作路由 | `ActionRouter` (`action_router.py`) | 解析模型输出的 `[silent] / [search] / [respond]` |
| 问题队列 | `QuestionQueueManager` (`question_queue.py`) | 用户提问的状态机（一次性 / 持续型） |
| 工作记忆 + 输出 | `WorkingMemory`, `OutputRecorder` | 对话历史 / 证据记录，轨迹落盘 `results.json` |

### 2.2 数据流

```
视频文件
  │  按 step_interval 采样帧 (base64)，按 caption_window 分桶为「时间步 step」
  ▼
每个 step
  ├─[ 记忆构建链 ]────────────────────────────────────────────────
  │   1. Caption 生成：帧 → caption VLM（默认本地 vLLM 的 Qwen3-VL）
  │   2. 知识图谱：caption → 实体抽取 + 文本向量化 (embedding)
  │   3. 视觉 embedding：视频片段 → ImageBind 特征库
  │
  └─[ 推理 / 服务链 ]───────────────────────────────────────────
      Pass1：基于「当前 caption + 历史」判断 silent / search / respond
        ├ silent  → 不开口
        ├ respond → 直接主动提醒
        └ search  → 检索长期记忆 → Pass2 决定是否回应
      (reasoning 模型只读 caption 文本，不读原始帧)
  ▼
输出事件 → CLI 打印 / Web 端 WebSocket 实时推送「思考流 + 回答」+ TTS 语音
  ▼
落盘：results.json + working_memory.json + 各存储索引
```

### 2.3 关键设计

- **推理不看图，只看文字**：视频理解完全发生在 caption 阶段；推理阶段只消费 caption 文本，因此推理延迟极低。
- **记忆与推理并行**（async 模式）：后台线程建记忆，主线程做推理，瓶颈是两者的 `max` 而非 `sum`。
- **两类触发**：① 主动服务 — 系统自行判断是否开口；② 用户提问 — 进队列后逐步被回答。

---

## 3. 依赖的外部服务 / 模型

| 依赖 | 用途 | 配置方式 |
|---|---|---|
| **本地 vLLM 服务** (Qwen3-VL-8B, OpenAI 兼容) | Caption 生成（当 `caption_model` 含 `vllm`） | 环境变量 `VLLM_BASE_URL`，默认 `http://localhost:8000/v1` |
| **OpenAI API** (gpt-5-mini / text-embedding-3-small / gpt-4o-mini) | Reasoning、文本/实体向量化、知识图谱实体抽取 | 环境变量 `OPENAI_API_KEY` |
| **ImageBind**（本地权重） | 视觉 embedding（可选；缺失则视觉检索降级，文本检索仍可用） | 权重 `imagebind_huge.pth` 放入 `egomemo/.checkpoints/` |
| **TTS**（仅 Web）：edge-tts（免费默认）/ OpenAI tts | 回答语音合成 | `--tts_backend` |

> Caption 与 reasoning 也可改用其它后端（见 §6）。所有 API key 均通过环境变量提供，仓库中不含任何硬编码密钥。

---

## 4. 环境配置

> 推荐独立 conda 环境，避免污染其它项目。

```bash
conda create -n egomemo python=3.10 -y
conda activate egomemo

# 本地 vLLM 服务端（须与本机 NVIDIA 驱动匹配；例如驱动为 CUDA 12.x 时用 cu12 构建）
pip install "vllm==0.11.0" "transformers==4.57.1"

# 其余依赖
pip install -r requirements.txt

# ImageBind（视觉 embedding，可选）：权重 imagebind_huge.pth 自行下载放到
#   egomemo/.checkpoints/imagebind_huge.pth
#   https://github.com/facebookresearch/ImageBind
```

让 Python 找到 `videorag` 与 `egomemo` 两个包（仓库根目录下二者平级）：

```bash
# 在仓库根目录执行
export PYTHONPATH="$PWD:$PWD/egomemo:$PYTHONPATH"
```

配置 API Key 与（可选）模型路径：

```bash
export OPENAI_API_KEY="sk-..."                 # reasoning + embedding 必填
export VLLM_BASE_URL="http://localhost:8000/v1"
# 若使用本地权重做 caption/reasoning，可指定路径：
export QWEN3VL_8B_PATH="./pretrained_weights/Qwen3-VL-8B-Instruct"
```

> **版本说明**：vLLM/torch 必须与本机 NVIDIA 驱动匹配。例如驱动为 CUDA 12.8 时，选 cu12 构建的 vLLM 0.11.0（torch 2.8.0+cu128）；驱动更新可用更高版本。

---

## 5. 运行

### 5.1 先启动 caption 用的 vLLM 服务

```bash
# 先把 Qwen3-VL-8B-Instruct 权重下载到 ./pretrained_weights/ 或用 MODEL_DIR 指定
cd scripts
MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct GPU_ID=0 PORT=8000 \
    bash launch_vllm_qwen3vl_v023.sh
# 等待出现 "Application startup complete"，确认 curl http://localhost:8000/v1/models 返回 200
```

### 5.2 Web 模式（推荐）

```bash
export OPENAI_API_KEY="sk-..."
export VLLM_BASE_URL="http://localhost:8000/v1"

python -m egomemo.run --web        # 默认 caption=qwen_vllm, reasoning=gpt-5-mini, port=8090
```

浏览器打开 `http://<host>:8090`：上传视频 →（可取消 Cache 复选框以重建记忆）→ Start Processing。
右侧「REASONING PROCESS」实时显示主动服务的思考流，可在输入框向助手提问；AI 回答时会语音播报。

### 5.3 CLI 模式

```bash
python -m egomemo.run \
    --video_path /path/to/video.mp4 \
    --caption_model qwen_vllm \
    --reasoning_model_path gpt-5-mini \
    --pipeline_mode async \
    --working_dir ./egomemo_cache
```

结果写入 `./egomemo_cache/<video_name>/results.json`（含回答轨迹与计时）。

预设提问（可选）：
```bash
--questions '[{"text": "What am I doing now?", "timestamp": 10.0, "recurring": false}]'
```

---

## 6. 配置项

| 参数 (CLI) | 默认 | 说明 |
|---|---|---|
| `--caption_model` | `qwen_vllm` | Caption 后端，见下表 |
| `--reasoning_model_path` | `gpt-5-mini` | Reasoning 模型；填本地 Qwen3-VL 路径则走本地推理 |
| `--pipeline_mode` | `async` | `async`=记忆/推理并行；`sequential`=先建全量记忆再推理 |
| `--datasets_type` | `egolife` | 数据集类型，影响 prompt 风格 |
| `--port` | `8090` | Web UI 端口（避开 vLLM 的 8000/8001）|
| `--tts` / `--tts_backend` | 开 / `edge` | Web 语音播报后端 |

`config.py` 中的其它配置：`step_interval_seconds`（采样间隔）、`caption_window_seconds`（每 step 时长）、`proactive_cooldown_seconds`（主动提醒冷却）、`clear_cache`（是否清空旧缓存）、`kg_extraction_mode`（`full` 完整 / `simple` 快速）。

**Caption 后端选项**（`load_caption_model` 按 `model_name` 关键字分派）：

| `model_name` 含 | 走向 |
|---|---|
| `vllm` | 本地 vLLM 服务（OpenAI 兼容 HTTP，无需在本进程加载权重）|
| `qwen3` + `api` | DashScope Qwen3-VL API（需 `DASHSCOPE_API_KEY`）|
| `gpt_4o` / `gpt4o` | GPT-4o API |
| `gemini` + `api` | Gemini API（需 `GEMINI_API_KEY`）|
| 其它（命中 MODEL_MAP）| 本地 transformers 加载权重（如 `qwenvl_3_8b_instruct`）|

---

## 7. 目录结构

```
.
├── egomemo/              # 主程序（流水线、推理、Web）
│   ├── run.py              # 入口（CLI / Web）
│   ├── streaming_pipeline.py
│   ├── memory_bridge.py / memory_bridge_fast.py
│   ├── pipeline_v2_patch.py
│   ├── action_router.py / question_queue.py / working_memory.py
│   ├── prompt_templates*.py
│   └── web/                # FastAPI 服务 + 前端静态资源
├── videorag/               # 底层记忆引擎（知识图谱 / 向量库 / caption 后端）
├── scripts/
│   └── launch_vllm_qwen3vl_v023.sh   # 启动本地 vLLM caption 服务
├── requirements.txt
└── README.md
```

> 运行时会生成 `egomemo_cache/`、`egomemo_uploads/`、`web/static/tts/` 等目录，以及需自行下载的 `egomemo/.checkpoints/imagebind_huge.pth` 与 `pretrained_weights/`，这些均已在 `.gitignore` 中排除。

---

## 8. 常见问题

- **右侧推理全是 SILENT / "Nothing relevant"**：通常是 `OPENAI_API_KEY` 未设置或失效——reasoning 调用失败后会回退为 silent，embedding 失败还会导致 caption 无法入库。检查运行日志中是否有 `401 / invalid_api_key`。
- **Caption 生成报连接错误**：vLLM 服务未启动或 `VLLM_BASE_URL` 不对，先确认 `curl http://localhost:8000/v1/models` 返回 200。
- **`No module named 'videorag'`**：未设置 `PYTHONPATH`，见 §4。
- **`moviepy.editor` 缺失**：moviepy 2.x 删除了该模块，需 `pip install "moviepy<2.0"`（requirements 已固定）。
- **记忆没有落盘**：进程需正常结束以触发索引写盘，强制终止会丢失数据。
```
