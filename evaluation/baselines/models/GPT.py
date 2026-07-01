"""
GPT-4o Proactive Service Evaluation for EgoLife.

Uses OpenAI GPT-4o API to detect proactive services
from uniformly sampled egocentric video frames.

Usage:
    python inference.py --model GPT --gpt_api <api_key> ...
"""

import os
import io
import base64
import time
from openai import OpenAI

from utils.egolife_bench import EgoLifeProactiveBench


class EvalGPT(EgoLifeProactiveBench):

    def __init__(self, args, model="gpt-5-mini"):
        self.gpt_model = model
        self.api_key = args.gpt_api
        super().__init__(args)

    def _init_model(self):
        """Initialize OpenAI client."""
        self._proxy_on()
        self.client = OpenAI(
            # base_url="https://api.openai.com/v1",
            api_key=self.api_key,
            timeout=300.0,
        )

    def _proxy_on(self):
        # Proxy disabled — direct connection to OpenAI works from this network
        for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            os.environ.pop(key, None)

    def _pil_to_base64_url(self, pil_image):
        """Convert PIL Image to base64 data URL."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    def _call_api(self, messages, retries=5, wait_time=5):
        """Call OpenAI API with retry logic."""
        for i in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=messages,
                    max_completion_tokens=4096,
                    # temperature=0,
                )
                return response.choices[0].message.content
            except Exception as e:
                if i < retries - 1:
                    sleep_time = wait_time * (2 ** i)
                    print(f"API call failed ({i+1}/{retries}), retrying in {sleep_time}s: {e}")
                    time.sleep(sleep_time)
                else:
                    print(f"API call failed after {retries} attempts: {e}")
                    return None

    def inference(self, frames, prompt):
        """
        Run GPT-4o inference on sampled frames.

        Args:
            frames: List of PIL Image objects
            prompt: Text prompt string

        Returns:
            str: Model response text
        """
        # Build multimodal message
        content = []
        for frame in frames:
            url = self._pil_to_base64_url(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "low"},
            })

        content.append({
            "type": "text",
            "text": prompt,
        })

        messages = [{"role": "user", "content": content}]

        return self._call_api(messages)
