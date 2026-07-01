#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EgoLife ProactiveBench — GPT 并行推理驱动（不改原脚本）。

为什么需要：原 inference.py 的 eval() 逐段串行调用 GPT，~7s/段，7566 段要 15+ 小时，
且全部跑完才落盘（中途零产出、无法续跑）。本驱动：
  - 复用 utils/egolife_bench 的段发现/帧采样 与 models.GPT.EvalGPT 的 build_prompt/inference/parse_response；
  - 用线程池并发调用 API（每段独立，天然可并行）；
  - 按 (person, day) 落盘，跑完一天存一天，支持 --skip_existing 断点续跑；
  - 输出格式与原脚本完全一致（{person}_{day}_proactive.json: List[segment_result]），
    evaluate.py 可直接读取。

用法：
    python inference_parallel_gpt.py \
        --data_dir /mnt/workspace/gst/EgoLife/egolife \
        --result_dir .../GPT_results_A2A4 \
        --persons A2_ALICE A4_LUCIA --days DAY1 DAY2 DAY3 DAY4 DAY5 \
        --gpt_model gpt-5-mini --num_frames 8 --workers 16
key 默认从 egomemo_demo/.env 的 OPENAI_API_KEY 读。
"""

import os
import sys
import json
import types
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.egolife_bench import (
    discover_video_segments, get_video_time_window,
    sample_uniform_frames, filename_to_time_number,
)
from models.GPT import EvalGPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ENV_FILE = os.environ.get("EGOSERVE_ENV_FILE", ".env")


def load_key(explicit):
    if explicit:
        return explicit
    if os.path.exists(ENV_FILE):
        for line in open(ENV_FILE):
            if line.strip().startswith("OPENAI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.getenv("OPENAI_API_KEY")


def process_segment(model, seg, person, day, num_frames):
    """处理单段：采样帧 -> GPT 推理 -> 解析。返回 segment_result(与原脚本同构)。"""
    video_path = seg["video_path"]
    try:
        day_str, time_window = get_video_time_window(video_path, seg["annotation_path"])
        frames, timestamps, duration = sample_uniform_frames(
            video_path, num_frames=num_frames, output_format='pil')
        if not frames:
            return None
        prompt = model.build_prompt(seg, time_window, duration, timestamps)
        response = model.inference(frames, prompt)
        parsed = model.parse_response(response, filename_to_time_number(seg["segment_id"]))
        return {
            "segment_id": seg["segment_id"], "person": person, "day": day,
            "time_window": time_window, "video_path": video_path,
            "num_frames_sampled": len(frames), "response": parsed,
            "raw_response": response,
        }
    except Exception as e:
        logger.error(f"段 {seg.get('segment_id')} 失败: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/mnt/workspace/gst/EgoLife/egolife")
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--persons", nargs="+", default=["A2_ALICE", "A4_LUCIA"])
    ap.add_argument("--days", nargs="+", default=["DAY1", "DAY2", "DAY3", "DAY4", "DAY5"])
    ap.add_argument("--gpt_api", default="")
    ap.add_argument("--gpt_model", default="gpt-5-mini")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    key = load_key(args.gpt_api)
    assert key, "未找到 OPENAI_API_KEY"
    # EvalGPT.__init__ 会读 args 的多个属性（data_dir/result_dir/persons/days/num_frames），补全
    model = EvalGPT(types.SimpleNamespace(
        gpt_api=key, num_frames=args.num_frames, data_dir=args.data_dir,
        result_dir=args.result_dir, persons=args.persons, days=args.days,
        model="GPT",
    ), model=args.gpt_model)

    out_dir = os.path.join(args.result_dir, "GPT")
    os.makedirs(out_dir, exist_ok=True)

    for person in args.persons:
        for day in args.days:
            out_path = os.path.join(out_dir, f"{person}_{day}_proactive.json")
            if args.skip_existing and os.path.exists(out_path):
                logger.info(f"跳过已存在: {person}/{day}")
                continue
            segs = discover_video_segments(args.data_dir, person, day)
            if not segs:
                logger.warning(f"无段: {person}/{day}")
                continue
            logger.info(f"开始 {person}/{day}: {len(segs)} 段, workers={args.workers}")

            results = [None] * len(segs)
            done = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                fut2idx = {ex.submit(process_segment, model, s, person, day, args.num_frames): i
                           for i, s in enumerate(segs)}
                for fut in as_completed(fut2idx):
                    results[fut2idx[fut]] = fut.result()
                    done += 1
                    if done % 100 == 0:
                        logger.info(f"  {person}/{day}: {done}/{len(segs)}")
            results = [r for r in results if r is not None]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"已落盘 {out_path}: {len(results)} 段")

    logger.info("全部完成")


if __name__ == "__main__":
    main()
