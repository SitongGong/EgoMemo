# EgoMemo

Official repository of **Vinci2: Providing Proactive Assistance in Continuous Egocentric Videos**.

EgoMemo builds a three-level temporal memory graph from continuous egocentric video
and retrieves over it to provide **proactive services**. This repo contains:

- the memory-graph construction + retrieval pipeline (`videorag/`, per-dataset scripts),
- the **EgoServe** proactive-service benchmark and its evaluation suite (`evaluation/`),
- dataset annotation & cleaning code (`dataset_annotation/`),
- a streaming demo (`egomemo_demo/`).

---

## 1. Installation

```bash
git clone https://github.com/SitongGong/EgoMemo.git
cd EgoMemo
conda create -n egomemo python=3.10 -y
conda activate egomemo
pip install -r egomemo_demo/requirements.txt
```

Copy the environment template and fill in your keys / paths:

```bash
cp .env.example .env
# edit .env: OPENAI_API_KEY, DASHSCOPE_API_KEY, GEMINI_API_KEY, ... and dataset paths
```

All scripts read keys and paths from `.env` (via `python-dotenv`) or from CLI args.
**No keys are hardcoded.**

---

## 2. Download datasets

### 2.1 EgoServe benchmark annotations (ours)

Our proactive-service annotations are hosted on HuggingFace:

```bash
huggingface-cli download SitongGong/EgoServe --repo-type dataset \
  --local-dir ./data/EgoServe_release
```

Layout after download:

```
data/EgoServe_release/
  EgoLife/{A1_JAKE,A4_LUCIA,A5_KATRINA}/{instant,short_term,long_term,episodic}.json
  HoloAssist/holoassist_service_annotations.json
  CaptainCook4D/captaincook4d_service_annotations.json
  dataset_statistics.json
```

### 2.2 Source videos

The annotations reference videos from the source datasets — download them from the
official sites and set the paths in `.env`:

| Dataset | Link | `.env` variable |
|---------|------|-----------------|
| EgoLife | https://egolife-ai.github.io/ | (see EgoLife scripts) |
| HoloAssist | https://holoassist.github.io/ | — |
| CaptainCook4D | https://captaincook4d.github.io/ | `CAPTIONCOOK4D_DATA_PATH` |
| EyeWo / ESTP-Bench | (project page) | `EYEWO_DATA_PATH`, `EYEWO_VIDEO_PATH` |
| OVO-Bench | https://github.com/JoeLeelyf/OVO-Bench | `OVOBENCH_DATA_PATH`, `OVOBENCH_VIDEO_PATH` |

---

## 3. Memory-graph construction & retrieval

Each benchmark has two stages: **(a) processing** — extract captions and build the
three-level memory graph; **(b) retrieval** — retrieve over the graph and generate the
model response. Ablations are toggled with the retrieval flags described in §3.4.

### 3.1 EgoServe (CaptainCook4D)

```bash
# (a) build memory graph
python captioncook4d_processing_parallel.py \
    --data_path $CAPTIONCOOK4D_DATA_PATH \
    --caption_model_name qwenvl_3_8b_instruct \
    --datasets_type val_test --num_processes 5

# (b) retrieval + proactive-service generation
python captioncook4d_online_retrieval_parallel_.py \
    --caption_model_name qwenvl_3_8b_instruct \
    --response_name proactive_response_gpt_5 \
    --caption_retrieval True --visual_retrieval True --entity_retrieval True \
    --multiscale True --reconstruct_caption True --need_retrieval True
```
(EgoLife / HoloAssist follow the same processing → retrieval pattern; see the
corresponding scripts.)

### 3.2 EyeWo / ESTP-Bench

```bash
# (a) processing
python eyewo_processing.py \
    --data_path $EYEWO_DATA_PATH --video_path $EYEWO_VIDEO_PATH \
    --caption_model_name qwenvl_3_8b_instruct \
    --interval_seconds 2 --window_seconds 60 --output_dir ./results/eyewo

# (b) retrieval
python eyewo_retrieval_processing_.py \
    --data_path $EYEWO_DATA_PATH --video_path $EYEWO_VIDEO_PATH \
    --caption_model_name qwenvl_3_8b_instruct \
    --caption_retrieval True --visual_retrieval True --entity_retrieval True \
    --need_retrieval True --max_rounds 5
```

### 3.3 OVO-Bench

```bash
python ovobench_retrieval.py \
    --data_path $OVOBENCH_DATA_PATH --video_path $OVOBENCH_VIDEO_PATH \
    --caption_model_name qwenvl_3_8b_instruct \
    --caption_retrieval True --visual_retrieval True --entity_retrieval True \
    --multiscale True --reconstruct_caption True --need_retrieval True --max_rounds 5
```

### 3.4 Ablation flags

Set a flag to `False` to reproduce the corresponding ablation in the paper:

| Flag | Ablation | Meaning |
|------|----------|---------|
| `--multiscale False` | **w/o MS** | replace the three-level temporal memory with a single clip-level store |
| `--reconstruct_caption False` | **w/o Recons.** | remove the VLM-based caption reconstruction |
| `--visual_retrieval False` | **w/o VSR** | disable visual retrieval |
| `--entity_retrieval False` | **w/o GSR** | disable entity / graph retrieval |
| `--caption_retrieval False` | **w/o MTR** | disable caption retrieval |

---

## 4. Evaluating on the EgoServe benchmark

The full evaluation suite lives in [`evaluation/`](evaluation/) (see its README for
details). Point it at the downloaded annotations and your model predictions:

```bash
export EGOSERVE_DATA=./data           # root holding EgoServe_release/ and predictions/
export OPENAI_API_KEY=sk-...          # only for the LLM-as-judge step

cd evaluation

# Main evaluation over all configs (5 ablations + full + 2 baselines) -> result_clean_*.json
python combined_proactive_evaluation.py --no_llm_scoring

# Evaluate directly against the released HuggingFace annotations
python combined_proactive_evaluation_release.py

# Reproduce the paper tables
python make_table_s4.py      # per-subtype P/R/F1
python make_table_s8.py      # LLM-as-judge (GPT-4o / Deepseek) rationality & effectiveness
```

Metrics: per-sub-type **P / R / F1** matched within each video/person under a time
tolerance (EgoLife 60 s, HoloAssist 10 s, CaptainCook4D 25 s); **Overall** is the
macro-average of active sub-type F1. `personal_progressive` and CaptainCook4D
`Episodic` are excluded from the benchmark.

Baseline (GPT / Qwen) prediction scripts are under
[`evaluation/baselines/`](evaluation/baselines/).

---

## 5. Dataset annotation & cleaning

Code used to generate and clean the annotations is in
[`dataset_annotation/`](dataset_annotation/):
`data_generation/` (Gemini-based generation) and `cleaning/` (consolidation,
validation and LLM-based refinement for HoloAssist / CaptainCook4D).

---

## 6. Demo

A streaming egocentric assistant demo is in [`egomemo_demo/`](egomemo_demo/) — see its
own README for setup.

---

## Citation

```bibtex
@inproceedings{gong2026vinci2,
  title     = {Vinci2: Providing Proactive Assistance in Continuous Egocentric Videos},
  author    = {Gong, Sitong and Yan, Tianyu and Kang, Caixin and Zheng, Bo and
               Ruan, Xiang and Lu, Huchuan and Zhang, Kaipeng and Sato, Yoichi and Huang, Yifei},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License
See [LICENSE](LICENSE).
