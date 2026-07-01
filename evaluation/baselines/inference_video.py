"""
Proactive Service Benchmark — Inference for HoloAssist / CaptionCook4D.

Splits long-form videos into fixed-length windows (default 30s), uniformly
samples frames per window, and runs model inference for proactive service
detection.

Usage:
    # HoloAssist with Gemini
    python inference_video.py \
        --dataset holoassist \
        --model Gemini \
        --gemini_project <api_key>

    # CaptionCook4D with QWen3VL
    python inference_video.py \
        --dataset captioncook4d \
        --model QWen3VL \
        --qwen_api_key <api_key>

    # HoloAssist with GPT-4o
    python inference_video.py \
        --dataset holoassist \
        --model GPT \
        --gpt_api <api_key>

    # CaptionCook4D with QWen2-VL local model
    python inference_video.py \
        --dataset captioncook4d \
        --model QWen2VL_7B \
        --model_path /path/to/Qwen2-VL-7B-Instruct
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Default paths
# ============================================================================

HOLOASSIST_DEFAULTS = {
    "video_dir": "./data/HoloAssist/videos",
    "anno_path": "./data/HoloAssist/annotations.json",
}

CAPTIONCOOK4D_DEFAULTS = {
    "video_dir": "./data/CaptainCook4D/videos",
    "data_splits_path": "./data/CaptainCook4D/data_splits.json",
    "error_anno_path": "./data/CaptainCook4D/error_annotations.json",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Proactive Service Benchmark — HoloAssist / CaptionCook4D Inference"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset", type=str, default="captioncook4d",
        choices=["holoassist", "captioncook4d"],
        help="Dataset to evaluate on"
    )

    # Data paths (auto-filled from defaults based on --dataset)
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Directory containing video files. "
             "Default: auto-set per dataset"
    )
    parser.add_argument(
        "--result_dir", type=str,
        default="./outputs/gpt_captioncook4d_results",
        help="Root directory for saving results"
    )

    # HoloAssist-specific
    parser.add_argument(
        "--anno_path", type=str, default=None,
        help="Path to HoloAssist val_service_annotations.json"
    )

    # CaptionCook4D-specific
    parser.add_argument(
        "--data_splits_path", type=str, default=None,
        help="Path to CaptionCook4D recordings_data_split_combined.json"
    )
    parser.add_argument(
        "--error_anno_path", type=str, default=None,
        help="Path to CaptionCook4D error_annotations.json"
    )
    parser.add_argument(
        "--splits", type=str, nargs='+', default=["val", "test"],
        help="CaptionCook4D splits to evaluate (default: val test)"
    )

    # Video processing
    parser.add_argument(
        "--window_seconds", type=int, default=60,
        help="Length of each video window in seconds (default: 30)"
    )
    parser.add_argument(
        "--num_frames", type=int, default=16,
        help="Number of frames to uniformly sample per window (default: 8)"
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip videos whose results already exist (default: True)"
    )
    parser.add_argument(
        "--no_skip_existing", dest="skip_existing", action="store_false",
        help="Re-process all videos even if results exist"
    )

    # Model selection
    parser.add_argument(
        "--model", type=str, default="GPT",
        choices=["Gemini", "GPT", "QWen2VL_7B", "QWen2VL_72B", "QWen3VL"],
        help="Model to use for evaluation"
    )

    # Gemini
    parser.add_argument(
        "--gemini_project", type=str, default="AIzaSyCOzZCneNLi8Aoj4odDSHuZskan2Pa1fVE",
        help="Google API key for Gemini"
    )
    parser.add_argument(
        "--gemini_model", type=str, default="gemini-2.5-flash",
        help="Gemini model name"
    )

    # GPT
    parser.add_argument("--gpt_api", type=str, default=os.environ.get("OPENAI_API_KEY",""), help="OpenAI API key")
    parser.add_argument("--gpt_model", type=str, default="gpt-5-mini", help="GPT model name")

    # QWen2-VL (local)
    parser.add_argument("--model_path", type=str, default=None, help="Path to local model weights")

    # QWen3-VL (API)
    parser.add_argument(
        "--qwen_api_key", type=str, default=os.environ.get("DASHSCOPE_API_KEY",""),
        help="DashScope API key for QWen3-VL"
    )
    parser.add_argument(
        "--qwen_model", type=str, default="qwen2.5-72b-instruct",
        help="QWen3-VL model name"
    )

    # 可选:视频列表分片(用于多进程并行,不传则跑全部,行为不变)
    parser.add_argument(
        "--shard_idx", type=int, default=0,
        help="本进程负责的分片下标 (0-based);配合 --shard_total 使用"
    )
    parser.add_argument(
        "--shard_total", type=int, default=1,
        help="总分片数;默认1表示不分片,跑全部视频"
    )
    parser.add_argument(
        "--qwen_base_url", type=str,
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        help="QWen3-VL API base URL"
    )

    args = parser.parse_args()

    # Fill in default paths based on dataset
    if args.dataset == "holoassist":
        if args.video_dir is None:
            args.video_dir = HOLOASSIST_DEFAULTS["video_dir"]
        if args.anno_path is None:
            args.anno_path = HOLOASSIST_DEFAULTS["anno_path"]
    elif args.dataset == "captioncook4d":
        if args.video_dir is None:
            args.video_dir = CAPTIONCOOK4D_DEFAULTS["video_dir"]
        if args.data_splits_path is None:
            args.data_splits_path = CAPTIONCOOK4D_DEFAULTS["data_splits_path"]
        if args.error_anno_path is None:
            args.error_anno_path = CAPTIONCOOK4D_DEFAULTS["error_anno_path"]

    return args


def _create_model(args):
    """Instantiate the appropriate model for VideoBench evaluation."""
    from utils.video_bench import VideoBench

    if args.model == "Gemini":
        from google import genai
        from google.genai import types
        import io

        class _GeminiVideo(VideoBench):
            def __init__(self, a):
                self.gemini_model_name = getattr(a, 'gemini_model', 'gemini-2.5-flash')
                self.api_key = a.gemini_project
                super().__init__(a)

            def _init_model(self):
                self.client = genai.Client(api_key=self.api_key)

            def inference(self, frames, prompt):
                parts = []
                for frame in frames:
                    buffered = io.BytesIO()
                    frame.save(buffered, format="JPEG")
                    parts.append(types.Part.from_bytes(
                        data=buffered.getvalue(), mime_type="image/jpeg"))
                parts.append(types.Part.from_text(text=prompt))
                try:
                    resp = self.client.models.generate_content(
                        model=self.gemini_model_name,
                        contents=parts,
                        config=types.GenerateContentConfig(
                            temperature=0, max_output_tokens=8192),
                    )
                    return resp.text
                except Exception as e:
                    print(f"Gemini inference error: {e}")
                    return None

        return _GeminiVideo(args)

    elif args.model == "GPT":
        import io
        import base64
        import time
        from openai import OpenAI

        class _GPTVideo(VideoBench):
            def __init__(self, a):
                self.gpt_model = getattr(a, 'gpt_model', 'gpt-5-mini')
                self.api_key = a.gpt_api
                super().__init__(a)

            def _init_model(self):
                self.client = OpenAI(api_key=self.api_key)

            def inference(self, frames, prompt):
                content = []
                for frame in frames:
                    buffered = io.BytesIO()
                    frame.save(buffered, format="JPEG")
                    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                for attempt in range(5):
                    try:
                        resp = self.client.chat.completions.create(
                            model=self.gpt_model,
                            messages=messages,
                            max_completion_tokens=4096,
                            # temperature=0,
                        )
                        return resp.choices[0].message.content
                    except Exception as e:
                        if attempt < 4:
                            print(f"GPT API error ({attempt+1}/5), retrying: {e}")
                            time.sleep(2)
                        else:
                            print(f"GPT API failed after 5 attempts: {e}")
                            return None

        assert args.gpt_api is not None, "--gpt_api is required for GPT model"
        return _GPTVideo(args)

    elif args.model in ("QWen2VL_7B", "QWen2VL_72B"):
        import io
        import base64
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        class _QWen2VLVideo(VideoBench):
            def __init__(self, a):
                self.model_path_str = a.model_path
                super().__init__(a)

            def _init_model(self):
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_path_str,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
                self.processor = AutoProcessor.from_pretrained(self.model_path_str)

            def inference(self, frames, prompt):
                content = []
                for frame in frames:
                    buffered = io.BytesIO()
                    frame.save(buffered, format="JPEG")
                    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    content.append({
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{b64}",
                        "min_pixels": 256 * 28 * 28,
                        "max_pixels": 360 * 420,
                    })
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=1024)
                trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, output_ids)]
                return self.processor.batch_decode(
                    trimmed, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0]

        assert args.model_path and os.path.exists(args.model_path), \
            f"--model_path must point to a valid directory, got: {args.model_path}"
        return _QWen2VLVideo(args)

    elif args.model == "QWen3VL":
        import io
        import base64
        import time
        from openai import OpenAI

        class _QWen3VLVideo(VideoBench):
            def __init__(self, a):
                self.qwen_model = getattr(a, 'qwen_model', 'qwen3-vl-plus')
                self.api_key = a.qwen_api_key
                self.base_url = getattr(
                    a, 'qwen_base_url',
                    'https://dashscope-intl.aliyuncs.com/compatible-mode/v1')
                super().__init__(a)

            def _init_model(self):
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            def inference(self, frames, prompt):
                content = []
                for frame in frames:
                    buffered = io.BytesIO()
                    frame.save(buffered, format="JPEG")
                    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                for attempt in range(5):
                    try:
                        resp = self.client.chat.completions.create(
                            model=self.qwen_model,
                            messages=messages,
                            max_tokens=1024,
                            temperature=0,
                        )
                        return resp.choices[0].message.content
                    except Exception as e:
                        if attempt < 4:
                            print(f"QWen3-VL API error ({attempt+1}/5), retrying: {e}")
                            time.sleep(2)
                        else:
                            print(f"QWen3-VL API failed after 5 attempts: {e}")
                            return None

        assert args.qwen_api_key is not None, "--qwen_api_key is required for QWen3VL"
        return _QWen3VLVideo(args)

    else:
        raise ValueError(f"Unsupported model: {args.model}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Video dir: {args.video_dir}")
    logger.info(f"Window: {args.window_seconds}s, Frames/window: {args.num_frames}")
    logger.info(f"Result dir: {args.result_dir}")
    logger.info(f"Skip existing: {args.skip_existing}")

    model = _create_model(args)

    results, summary = model.eval()

    logger.info(
        "\nDone! Results saved to: %s",
        os.path.join(args.result_dir, args.model, args.dataset)
    )


if __name__ == "__main__":
    main()
