# Baseline Inference (GPT / Qwen)

Scripts to generate proactive-service predictions from baseline LLMs
(GPT, Qwen3-VL-Plus, Gemini) on the three EgoServe datasets. These outputs
are then scored by the evaluation scripts in the parent directory.

## Files
- `inference_parallel.py` — EgoLife parallel inference (DAY1–DAY5).
- `inference_parallel_gpt.py` — GPT-driven parallel inference for EgoLife.
- `inference_video.py` — inference for HoloAssist / CaptainCook4D (video-based).
- `prompts.py` — proactive-service prompts.
- `models/` — model wrappers (GPT, Gemini, QWen2VL, QWen3VL).
- `utils/` — dataset loaders (egolife_bench, video_bench).

## API keys
Pass keys via command-line args (`--gpt_api`, `--qwen_api_key`) or set
`OPENAI_API_KEY` / `DASHSCOPE_API_KEY` in your environment. Never hardcode keys.

## Paths
Data/output paths default to `./data/...` and `./outputs/...`; override with
CLI args. Set `EGOSERVE_ENV_FILE` to point at your `.env` if needed.
