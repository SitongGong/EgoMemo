# Annotation Cleaning & Validation

Scripts that clean, validate and refine the raw proactive-service annotations for
**HoloAssist** and **CaptainCook4D** into the final benchmark form.

## HoloAssist (`holoassist/`)

Run in this order to reproduce the released `holoassist_service_annotations.json`:

1. **`consolidate_service_annotations.py`** — merge the 5 per-type annotation
   directories into one JSON keyed by video; dedup tool_use↔error_recovery,
   resolve cross-type conflicts, merge time-adjacent same-type segments.
2. **`refine_rebuttal_with_llm.py`** — self-contained single-direction pipeline
   (merging is always last): enforce every safety/tool_use/error_recovery maps to a
   real *Wrong Action*, LLM-rejudge cross-type conflicts, recover missed Wrong
   Actions, and LLM-judge whether adjacent same-type events describe the same error
   before merging.
3. **`fix_next_step.py`** — keep only `next_step_guidance` events that correspond to
   a `Conversation` event in the source annotations; LLM re-extract missed ones;
   merge consecutive ones under a duration cap.
4. **`extract_rebuttal_subset.py`** — extract the subset of videos that have model
   predictions (the evaluation set).

## CaptainCook4D (`captaincook4d/`)

1. **`consolidate_error_dialogue.py`** — merge multiple errors on the same step into
   one event with a unified dialogue; LLM re-check `service_type` consistency; drop
   Episodic types and time-less *Missing Step* errors.
2. **`recheck_suspect_types.py`** — LLM (gpt-5.2) re-judges the events whose type was
   changed during consolidation (e.g. Error-Recovery → Next-Step / Resource).

## Configuration

- **API key**: read from `$OPENAI_API_KEY` (or an `.env` file pointed at by
  `$EGOSERVE_ENV_FILE`). No keys are hardcoded.
- **Data paths**: default to `./data/...`; override via CLI args or the
  `HOLOASSIST_DIR` / `CC4D_CONSOLIDATED` environment variables. Reports are written
  under `./outputs/`.
- **LLM**: defaults to `gpt-5.2` for the LLM-judged steps.
