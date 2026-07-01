#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CaptionCook4D —— 对"存疑类型重判"的事件用 gpt-5.2 再判一次。

背景：consolidate_error_dialogue.py 把同一 step 的多错误合并并重判 service_type 时，
有一批原本是 Error-Recovery 的被改成了 Next-Step Guidance / Resource Reminder，
其中 Order Error / 已发生的错误 被判成 Next-Step 语义上存疑（Next-Step 是"步骤已做对、
提示下一步"，而"顺序错/已做错"应是需要纠正=Error-Recovery）。

本脚本：
  1) 读最终产物，挑出"类型相对原标注变化 且 最终落到 Next-Step Guidance / Resource Reminder"的事件；
  2) 用 gpt-5.2 在完整 5 类里重新判定（带更明确的判别指引）；
  3) 把复判结果写回（仅改这些存疑事件的 service_type，其余不动；备份原文件）。

判别指引（关键）：
  - "已经做错的动作/步骤"（顺序错、用量错、漏做、装错）→ Error-Recovery（需纠正/补做）。
  - Next-Step Guidance 仅当"当前步骤已正确完成、只是引导下一步"，不针对错误。
  - 操作技巧/时间/温度等可当场调整、不需回退 → Tool Use。
  - 涉及人身危险 → Safety。
  - 遗留未收尾（没关/没盖/忘拿）→ Resource Reminder。

用法:
  python recheck_suspect_types.py            # 在线 gpt-5.2
  python recheck_suspect_types.py --dry_run  # 不调模型，只统计存疑数
"""
import os, json, argparse
from concurrent.futures import ThreadPoolExecutor

FP = os.environ.get("CC4D_CONSOLIDATED", "./data/CaptainCook4D/error_to_dialogue_results_consolidated.json")
ENV = os.environ.get("EGOSERVE_ENV_FILE", ".env")
SUB_TO_MAIN = {
    "Safety": "Instant Proactive Service", "Tool Use": "Instant Proactive Service",
    "Error-Recovery": "Short-Term Proactive Service", "Next-Step Guidance": "Short-Term Proactive Service",
    "Resource Reminder": "Short-Term Proactive Service",
}
FIVE = list(SUB_TO_MAIN.keys())
SUSPECT_FINAL = {"Next-Step Guidance", "Resource Reminder"}

PROMPT = """You are auditing a proactive-service label for a cooking-error event.

Decide the SINGLE most appropriate service sub-type (choose exactly one of the five):
- Error-Recovery: the user has ALREADY done something wrong (wrong order, wrong amount,
  wrong/forgotten step, wrong part) and it must be corrected / redone. THIS IS THE DEFAULT
  for any already-committed mistake (e.g. an "Order Error" means the order is already wrong).
- Tool Use: the step choice is correct but the technique/timing/temperature can be adjusted
  on the spot WITHOUT redoing anything.
- Safety: the action poses a bodily hazard (burn, cut, etc.).
- Resource Reminder: something was left unhandled (not closed/covered, forgotten, left on).
- Next-Step Guidance: the current step is COMPLETED CORRECTLY and you are only guiding the
  next action. Do NOT use this for an action that is itself wrong.

Event:
{payload}

Return STRICT JSON: {{"sub_type":"<one of the five>","reason":"<short>"}}"""


def load_key():
    for l in open(ENV):
        if l.strip().startswith("OPENAI_API_KEY"):
            return l.split("=",1)[1].strip().strip('"').strip("'")
    return os.getenv("OPENAI_API_KEY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--report", default="./outputs/cc4d_recheck_report.json")
    args = ap.parse_args()

    data = json.load(open(FP, encoding="utf-8"))
    # 收集存疑事件的引用
    jobs = []  # (rec_idx, item_idx, payload)
    for ri, rec in enumerate(data):
        for ii, it in enumerate(rec["dialogue"]["items"]):
            oset = set(it.get("original_sub_types") or [])
            cur = it["service_type"]["sub"]
            changed = not (len(oset) == 1 and list(oset)[0] == cur)
            if changed and cur in SUSPECT_FINAL:
                assist = next((d["utterance"] for d in it.get("dialogue", []) if d.get("role")=="assistant"), "")
                payload = json.dumps({
                    "error_tags": it.get("merged_from_error_tags") or [it.get("error_tag")],
                    "step_description": it.get("description",""),
                    "current_label": cur,
                    "original_labels": sorted(oset),
                    "observation": it.get("observation","") or (it.get("merged_observations") or [""])[0],
                    "assistant_message": assist,
                }, ensure_ascii=False, indent=2)
                jobs.append((ri, ii, payload))
    print(f"存疑事件(类型变化且->Next-Step/Resource): {len(jobs)}")
    if args.dry_run:
        print("dry-run: 不调模型"); return

    from openai import OpenAI
    client = OpenAI(api_key=load_key())
    def judge(payload):
        for _ in range(4):
            try:
                r = client.chat.completions.create(model=args.model,
                    messages=[{"role":"user","content":PROMPT.format(payload=payload)}],
                    response_format={"type":"json_object"})
                res = json.loads(r.choices[0].message.content)
                if res.get("sub_type") in FIVE: return res
            except Exception:
                pass
        return None
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda j: judge(j[2]), jobs))

    from collections import Counter
    changed_back = Counter(); kept = 0; failed = 0; transitions = Counter()
    report = []
    for (ri, ii, payload), res in zip(jobs, results):
        it = data[ri]["dialogue"]["items"][ii]
        old = it["service_type"]["sub"]
        if res is None:
            failed += 1; continue
        new_sub = res["sub_type"]
        transitions[(old, new_sub)] += 1
        report.append({"recording": data[ri]["recording_id"], "step": it["step_id"],
                       "old": old, "new": new_sub, "reason": res.get("reason","")[:120],
                       "error_tags": it.get("merged_from_error_tags") or [it.get("error_tag")]})
        if new_sub != old:
            it["service_type"] = {"main": SUB_TO_MAIN[new_sub], "sub": new_sub}
            it["recheck_changed"] = True
            changed_back[new_sub] += 1
        else:
            kept += 1

    # 备份 + 写回
    bak = FP.replace(".json", ".bak_beforeRecheck.json")
    if not os.path.exists(bak):
        json.dump(json.load(open(FP)), open(bak,"w"), ensure_ascii=False, indent=2)
    json.dump(data, open(FP,"w"), ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    json.dump(report, open(args.report,"w"), ensure_ascii=False, indent=2)

    print(f"\n复判完成: 改判 {sum(changed_back.values())}, 维持原判 {kept}, LLM失败 {failed}")
    print("改判去向:", dict(changed_back))
    print("主要 old->new 流向:")
    for (o,n),c in transitions.most_common(10):
        mark = "(改)" if o!=n else "(留)"
        print(f"  {o:20s} -> {n:20s}: {c} {mark}")
    print(f"备份: {bak}\n报告: {args.report}")


if __name__ == "__main__":
    main()
