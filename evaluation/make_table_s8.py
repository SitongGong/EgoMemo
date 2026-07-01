#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table S8: 对 8 个模型(2 baseline + 5 消融 + EgoMemo) × 2 个评判器(GPT-4o / Deepseek-R1)
打 rationality(R) / effectiveness(E) 分(1-5), Overall = (R+E)/2。
匹配对来自各 result_*_matched_pairs_for_human_eval.json。
复用 egolife 评测模块的打分 prompt。并行调用。
"""
import sys, json, re, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import egolife_proactive_evaluation_subtype as E
from concurrent.futures import ThreadPoolExecutor
import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
WORKERS = 16

# 8 个模型: (显示名, result标签)
MODELS = [
    ("Qwen3-VL-Plus", "Qwen"),
    ("GPT-5-mini",    "GPT"),
    ("w/o MS",        "wo_multiscale"),
    ("w/o Recons",    "wo_recons"),
    ("w/o VSR",       "wo_visual"),
    ("w/o GSR",       "wo_entity"),
    ("w/o MTR",       "wo_caption"),
    ("EgoMemo",       "full"),
]
# 2 个评判器: (显示名, api模型, base_url, key)
JUDGES = [
    ("GPT-4o",      "gpt-4o",             None,                        paths.openai_api_key()),
    ("Deepseek-V3", "deepseek-chat",      "https://api.deepseek.com",  paths.deepseek_api_key()),
    ("Deepseek-R1", "deepseek-reasoner",  "https://api.deepseek.com",  paths.deepseek_api_key()),
]

from openai import OpenAI

def parse_score(text, key):
    m = re.search(rf'{key}:\s*(\d+)', text or "", re.IGNORECASE)
    return min(int(m.group(1)), 5) if m else None

def score_pair(client, model, pair):
    gt = pair["gt"]; pred = pair["pred"]
    gt_p = (gt.get("user_prompt") or "").strip(); pr_p = (pred.get("user_prompt") or "").strip()
    if not gt_p or not pr_p:
        return None
    gt_t = f"{gt.get('main_type','')} / {gt.get('sub_type','')}"
    pr_t = f"{pred.get('main_type','')} / {pred.get('sub_type','')}"
    out = {}
    for key, PROMPT in [("Rationality", E.LLM_RATIONALITY_PROMPT), ("Effectiveness", E.LLM_EFFECTIVENESS_PROMPT)]:
        prompt = PROMPT.format(gt_service_type=gt_t, gt_user_prompt=gt_p,
                               pred_service_type=pr_t, pred_user_prompt=pr_p)
        for _ in range(3):
            try:
                r = client.chat.completions.create(model=model,
                    messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2048)
                v = parse_score(r.choices[0].message.content, key)
                if v is not None: out[key.lower()] = v
                break
            except Exception:
                continue
    if "rationality" in out and "effectiveness" in out:
        return out
    return None

def score_model(client, model, pairs):
    fn = lambda p: score_pair(client, model, p)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(fn, pairs))
    valid = [r for r in res if r]
    if not valid:
        return None
    R = np.mean([r["rationality"] for r in valid])
    Eff = np.mean([r["effectiveness"] for r in valid])
    return {"R": round(float(R), 2), "E": round(float(Eff), 2),
            "Overall": round(float((R + Eff) / 2), 2), "n": len(valid)}

def main():
    table = {}  # judge -> model_display -> {R,E,Overall}
    for jname, jmodel, jbase, jkey in JUDGES:
        client = OpenAI(api_key=jkey, base_url=jbase) if jbase else OpenAI(api_key=jkey)
        table[jname] = {}
        for disp, tag in MODELS:
            pf = os.path.join(EVAL_DIR, f"result_{tag}_matched_pairs_for_human_eval.json")
            pairs = json.load(open(pf))["pairs"]
            sc = score_model(client, jmodel, pairs)
            table[jname][disp] = sc
            print(f"[{jname}] {disp}: {sc}", flush=True)
    os.makedirs(paths.OUTPUT_ROOT, exist_ok=True)
    json.dump(table, open(os.path.join(paths.OUTPUT_ROOT, "table_s8.json"), "w"), ensure_ascii=False, indent=2)
    # 打印 Table S8 风格
    cols = [d for d, _ in MODELS]
    print("\n" + "=" * 90)
    print(f"{'Eval LLM':14s}{'Metric':8s}" + "".join(f"{c:>15s}" for c in cols))
    print("=" * 90)
    for jname in [j[0] for j in JUDGES]:
        for met in ["R", "E", "Overall"]:
            row = f"{jname if met=='R' else '':14s}{met:8s}"
            for c in cols:
                v = table[jname][c]
                row += f"{(v[met] if v else 0):>15.1f}"
            print(row)
    print("\n已存 table_s8.json")

if __name__ == "__main__":
    main()
