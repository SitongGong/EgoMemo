# EgoServe Evaluation

Evaluation code for the **EgoServe** proactive-service benchmark across three
egocentric datasets (EgoLife, HoloAssist, CaptainCook4D).

## Layout

```
evaluation/
  paths.py                                   # central path / key config
  egolife_proactive_evaluation_subtype.py    # EgoLife eval module (GT extract, match, P/R/F1)
  holoassist_proactive_evaluation_subtype.py # HoloAssist eval module
  captioncook4d_proactive_evaluation_subtype.py
  combined_proactive_evaluation.py           # main entry: run all 8 configs -> result_*.json
  combined_proactive_evaluation_release.py   # same, but reads the released HuggingFace annotations
  make_table_s4.py / make_table_s8.py        # generate paper tables (F1 / LLM-judge)
  llm_score.py                               # LLM-as-judge rationality/effectiveness scoring
  baselines/                                 # GPT / Qwen inference scripts (predictions)
```

## Setup

1. Download the released annotations from
   [HuggingFace `SitongGong/EgoServe`](https://huggingface.co/datasets/SitongGong/EgoServe)
   into `$EGOSERVE_DATA/EgoServe_release/`.
2. Put model predictions under `$EGOSERVE_DATA/predictions/` (see `paths.py` for the
   expected sub-directories).
3. Configure the data root:
   ```bash
   export EGOSERVE_DATA=/path/to/your/data
   export OPENAI_API_KEY=sk-...       # only needed for LLM-judge scoring
   export DEEPSEEK_API_KEY=sk-...     # optional, second judge
   ```
   All paths are resolved in `paths.py`; edit it directly if you prefer.

## Run

```bash
# Main evaluation (all 8 configs: 5 ablations + full + 2 baselines) -> result_*.json
python combined_proactive_evaluation.py --no_llm_scoring

# Evaluate directly against the released annotations
python combined_proactive_evaluation_release.py

# Generate paper tables
python make_table_s4.py       # per-subtype F1 table
python make_table_s8.py       # LLM-judge (GPT-4o / Deepseek) R/E table
```

## Metrics
- **P / R / F1** per sub-type, matched greedily within each video/person under a
  time tolerance (EgoLife 60 s, HoloAssist 10 s, CaptainCook4D 25 s).
- **Overall** = macro-average of per-sub-type F1 over active sub-types.
- `personal_progressive` and CaptainCook4D `Episodic` are excluded from the benchmark.

## Taxonomy
| Main | Sub-types |
|------|-----------|
| Instant | safety, tool_use |
| Short-Term | next_step_guidance, error_recovery, resource_reminder |
| Episodic | memory_recall, task_reminder *(EgoLife only)* |
| Long-Term | habit_coaching, memory_link_contextual, routine_optimization *(EgoLife only)* |
