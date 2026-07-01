#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 gpt-4o 对 EgoMemo(full) 的 strict 匹配对打 rationality / effectiveness 分(1-5)。
复用 egolife 评测模块里的打分 prompt。匹配对来自 result_full_matched_pairs_for_human_eval.json。
并行调用。输出全局 + per-dataset 平均分。

deepseek 评判待 key 提供后再补(同脚本换 base_url/model 即可)。
"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import egolife_proactive_evaluation_subtype as E
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np

PAIRS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_full_matched_pairs_for_human_eval.json")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"
WORKERS = 16

from openai import OpenAI
# deepseek 用 OpenAI 兼容端点; 其余走默认 OpenAI
if MODEL.startswith("deepseek"):
    client = OpenAI(api_key=paths.deepseek_api_key(),
                    base_url="https://api.deepseek.com")
else:
    client = OpenAI(api_key=paths.openai_api_key())

def parse_score(text, key):
    m = re.search(rf'{key}:\s*(\d+)', text or "", re.IGNORECASE)
    return min(int(m.group(1)), 5) if m else None

def score_one(pair):
    gt = pair["gt"]; pred = pair["pred"]
    gt_p = (gt.get("user_prompt") or "").strip()
    pr_p = (pred.get("user_prompt") or "").strip()
    if not gt_p or not pr_p:
        return None
    gt_type = f"{gt.get('main_type','')} / {gt.get('sub_type','')}"
    pr_type = f"{pred.get('main_type','')} / {pred.get('sub_type','')}"
    out = {}
    for key, PROMPT in [("Rationality", E.LLM_RATIONALITY_PROMPT), ("Effectiveness", E.LLM_EFFECTIVENESS_PROMPT)]:
        prompt = PROMPT.format(gt_service_type=gt_type, gt_user_prompt=gt_p,
                               pred_service_type=pr_type, pred_user_prompt=pr_p)
        for _ in range(3):
            try:
                r = client.chat.completions.create(model=MODEL,
                    messages=[{"role": "user", "content": prompt}], temperature=0.0)
                v = parse_score(r.choices[0].message.content, key)
                if v is not None:
                    out[key.lower()] = v
                break
            except Exception:
                continue
    if "rationality" in out and "effectiveness" in out:
        return {"dataset": pair["dataset"], **out}
    return None

def main():
    data = json.load(open(PAIRS_FILE))
    pairs = data["pairs"]
    print(f"模型={MODEL}, 匹配对={len(pairs)}, 打分中...")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(score_one, pairs))
    valid = [r for r in results if r]
    # 全局
    gr = np.mean([r["rationality"] for r in valid])
    ge = np.mean([r["effectiveness"] for r in valid])
    print(f"\n=== {MODEL} 评判分 (n={len(valid)}/{len(pairs)}) ===")
    print(f"  Global  Rationality={gr:.3f}  Effectiveness={ge:.3f}")
    # per-dataset
    by = defaultdict(list)
    for r in valid:
        by[r["dataset"]].append(r)
    print("  per-dataset:")
    for ds in ["EgoLife", "HoloAssist", "CaptionCook4D"]:
        rs = by.get(ds, [])
        if rs:
            print(f"    {ds:14s} R={np.mean([x['rationality'] for x in rs]):.3f}  "
                  f"E={np.mean([x['effectiveness'] for x in rs]):.3f}  (n={len(rs)})")
    os.makedirs(paths.OUTPUT_ROOT, exist_ok=True)
    json.dump({"model": MODEL, "n": len(valid),
               "global": {"rationality": round(float(gr),3), "effectiveness": round(float(ge),3)},
               "per_dataset": {ds: {"rationality": round(float(np.mean([x['rationality'] for x in by[ds]])),3),
                                    "effectiveness": round(float(np.mean([x['effectiveness'] for x in by[ds]])),3),
                                    "n": len(by[ds])} for ds in by}},
              open(os.path.join(paths.OUTPUT_ROOT, f"llm_score_{MODEL.replace('/','_')}.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\n已存 llm_score_{MODEL.replace('/','_')}.json")

if __name__ == "__main__":
    main()
