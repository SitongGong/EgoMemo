#!/usr/bin/env bash
# 单轮跑指定 caption_model；用法：bash run_one_caption.sh <tag> <model>
# 例：bash run_one_caption.sh qwen3_api qwen3_api
set -u

TAG=${1:?usage: run_one_caption.sh <tag> <model>}
MODEL=${2:?usage: run_one_caption.sh <tag> <model>}
# 注意：caption_window / step_interval 不在脚本里覆盖，
# 完全沿用 run.py 里的 argparse default，避免和你手动调过的默认值冲突。

VIDEO=${VIDEO:-./sample_video.mp4}
ROOT=${ROOT:-./egomemo}
LOG_DIR=${ROOT}/bench_logs
WORK=${ROOT}/bench_${TAG}
LOG=${LOG_DIR}/${TAG}.log
PIDFILE=${LOG_DIR}/${TAG}.pid

mkdir -p "${LOG_DIR}" "${WORK}"

# 按需启用 conda 环境（请改成你自己的 conda 路径与环境名）
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate egoserve
cd "$(dirname "$0")/.."

echo "$$" > "${PIDFILE}"
started=$(date '+%Y-%m-%d %H:%M:%S')
echo "[driver-${TAG}] === START ${TAG} model=${MODEL} at ${started} ===" \
    | tee -a "${LOG_DIR}/driver.log"

python -u -m egomemo.run \
    --video_path "${VIDEO}" \
    --pipeline_mode async \
    --caption_model "${MODEL}" \
    --reasoning_model_path gpt-5-mini \
    --working_dir "${WORK}" \
    > "${LOG}" 2>&1
rc=$?

ended=$(date '+%Y-%m-%d %H:%M:%S')
echo "[driver-${TAG}] === END ${TAG} rc=${rc} at ${ended} ===" \
    | tee -a "${LOG_DIR}/driver.log"
results="${WORK}/3_46_360p/results.json"
if [ -f "${results}" ]; then
    echo "[driver-${TAG}] results: ${results}" | tee -a "${LOG_DIR}/driver.log"
else
    echo "[driver-${TAG}] WARNING: ${results} not found" | tee -a "${LOG_DIR}/driver.log"
fi
rm -f "${PIDFILE}"
exit ${rc}
