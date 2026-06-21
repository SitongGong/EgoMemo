#!/usr/bin/env bash
# 串行跑三个 caption_model 实验，比较 wall-clock。
# 顺序：qwenvl_3_8b_instruct → qwenvl_3_5_4b → qwen3_api
# 每轮独立 working_dir 和 log，跑完后 results.json 在
# ${ROOT}/bench_<tag>/<video_name>/results.json
set -u

VIDEO=${VIDEO:-./sample_video.mp4}
ROOT=${ROOT:-./egomemo}
LOG_DIR=${ROOT}/bench_logs
mkdir -p "${LOG_DIR}"

# 按需启用 conda 环境（请改成你自己的 conda 路径与环境名）
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate egoserve
cd "$(dirname "$0")/.."

run_one() {
    local tag=$1            # 短标签，给 working_dir / log
    local model=$2          # --caption_model 字符串
    local work=${ROOT}/bench_${tag}
    local log=${LOG_DIR}/${tag}.log
    local started=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[driver] === START ${tag} (model=${model}) at ${started} ===" | tee -a "${LOG_DIR}/driver.log"
    mkdir -p "${work}"
    # 注意：caption_window / step_interval 沿用 run.py default，
    # 不在脚本里硬编码覆盖。
    python -u -m egomemo.run \
        --video_path "${VIDEO}" \
        --pipeline_mode async \
        --caption_model "${model}" \
        --reasoning_model_path gpt-5-mini \
        --working_dir "${work}" \
        > "${log}" 2>&1
    local rc=$?
    local ended=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[driver] === END ${tag} rc=${rc} at ${ended} ===" | tee -a "${LOG_DIR}/driver.log"
    # 结果路径提示
    local results="${work}/3_46_360p/results.json"
    if [ -f "${results}" ]; then
        echo "[driver] results: ${results}" | tee -a "${LOG_DIR}/driver.log"
    else
        echo "[driver] WARNING: ${results} not found" | tee -a "${LOG_DIR}/driver.log"
    fi
}

run_one qwenvl_3_8b qwenvl_3_8b_instruct
run_one qwenvl_3_5_4b qwenvl_3_5_4b
run_one qwen3_api qwen3_api

echo "[driver] all three runs finished at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_DIR}/driver.log"
