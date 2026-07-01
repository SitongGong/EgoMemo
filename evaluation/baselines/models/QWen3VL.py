"""
Qwen3-VL Proactive Service Evaluation for EgoLife.

Uses Alibaba DashScope API (OpenAI-compatible) to detect proactive services
from uniformly sampled egocentric video frames.

Model: qwen3-vl-plus / qwen3-vl-235b-a22b-instruct / etc.
API docs: https://www.alibabacloud.com/help/en/model-studio/qwen-vl-compatible-with-openai

Usage:
    python inference.py --model QWen3VL \
        --qwen_api_key <dashscope_api_key> \
        --qwen_model qwen3-vl-plus ...
"""

import os
import io
import base64
import time
from openai import OpenAI

from utils.egolife_bench import EgoLifeProactiveBench


class EvalQWen3VL(EgoLifeProactiveBench):

    def __init__(self, args):
        self.qwen_model = getattr(args, 'qwen_model', 'qwen3-vl-plus')
        self.api_key = args.qwen_api_key
        self.base_url = getattr(
            args, 'qwen_base_url',
            'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
        )
        super().__init__(args)

    def _init_model(self):
        """Initialize OpenAI-compatible client for DashScope."""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _pil_to_base64_url(self, pil_image):
        """Convert PIL Image to base64 data URL."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    def inference(self, frames, prompt):
        """
        Run Qwen3-VL inference on sampled frames via DashScope API.

        Args:
            frames: List of PIL Image objects
            prompt: Text prompt string

        Returns:
            str: Model response text
        """
        content = []
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._pil_to_base64_url(frame)},
            })
        content.append({
            "type": "text",
            "text": prompt,
        })

        messages = [{"role": "user", "content": content}]

        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.qwen_model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < 4:
                    print(f"QWen3-VL API error ({attempt+1}/5), retrying: {e}")
                    time.sleep(2)
                else:
                    print(f"QWen3-VL API failed after 5 attempts: {e}")
                    return None
