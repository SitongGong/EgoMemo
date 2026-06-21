#!/usr/bin/env bash
# 适配 vllm 0.23 的 Qwen3-VL-8B-Instruct 启动脚本。
# 与原 launch_vllm_qwen3vl.sh 的区别（原脚本针对 vllm 0.10 编写，保留供旧环境使用）：
#   1. vllm 0.23 已将 --disable-log-requests 改名为 --no-enable-log-requests
#   2. 默认 MODEL_DIR / PYBIN 改为本机实际存在的路径
#      (默认权重路径改为相对路径占位符，可用 MODEL_DIR 环境变量覆盖；
#       conda 环境为新建的 egoserve)
#
# 用法:
#   bash scripts/launch_vllm_qwen3vl_v023.sh           # GPU 0, port 8000
#   GPU_ID=1 PORT=8001 bash scripts/launch_vllm_qwen3vl_v023.sh

set -euo pipefail

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-./pretrained_weights/Qwen3-VL-8B-Instruct}"
SERVED_NAME="${SERVED_NAME:-Qwen3-VL-8B-Instruct}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# 共用机器上其他用户进程可能挤占同卡显存，0.6 留余量降低被挤 OOM 的概率
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 8, \"video\": 2}}"
DTYPE="${DTYPE:-bfloat16}"

# ----------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[ERROR] MODEL_DIR not found: $MODEL_DIR" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not in PATH; cannot verify GPU." >&2
fi

PYBIN="${PYBIN:-/root/miniconda3/envs/egoserve/bin/python}"
if ! "$PYBIN" -c "import vllm" >/dev/null 2>&1; then
    echo "[ERROR] vllm not installed in $PYBIN" >&2
    echo "[HINT]  conda create -n egoserve python=3.10 && pip install -U vllm" >&2
    exit 1
fi

VLLM_VER=$("$PYBIN" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "[launch_vllm_qwen3vl_v023]"
echo "  MODEL_DIR        = $MODEL_DIR"
echo "  SERVED_NAME      = $SERVED_NAME"
echo "  GPU_ID           = $GPU_ID"
echo "  PORT             = $PORT"
echo "  MAX_MODEL_LEN    = $MAX_MODEL_LEN"
echo "  GPU_MEM_UTIL     = $GPU_MEM_UTIL"
echo "  MAX_NUM_SEQS     = $MAX_NUM_SEQS"
echo "  LIMIT_MM         = $LIMIT_MM_PER_PROMPT"
echo "  DTYPE            = $DTYPE"
echo "  vllm version     = $VLLM_VER"
echo "  python           = $PYBIN"
echo ""

# ----------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# vllm 0.11 默认用 flashinfer 采样器，会 JIT 现编译 CUDA kernel；
# 本机当前驱动 12.8 + nvcc 12.9，JIT 链路不稳定。禁用 flashinfer 采样，
# 改用 PyTorch 原生 top-k/top-p 采样（无需编译，结果一致）。
export VLLM_USE_FLASHINFER_SAMPLER=0
# 缓解显存碎片（vllm OOM 日志自身也建议），共用卡场景更稳
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 兜底：确保 ninja 在 PATH、CUDA_HOME 指向系统 nvcc（若仍触发任何 JIT）
export PATH="$(dirname "$PYBIN"):${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"

exec "$PYBIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --enable-prefix-caching \
    --no-enable-log-requests
