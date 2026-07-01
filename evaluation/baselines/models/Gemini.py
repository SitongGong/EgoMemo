"""
Gemini Proactive Service Evaluation for EgoLife.

Uses Google GenAI SDK (API Key auth) to detect proactive services
from uniformly sampled egocentric video frames.

Usage:
    python inference.py --model Gemini --gemini_project <api_key> ...
"""

import os
import io
from google import genai
from google.genai import types

from utils.egolife_bench import EgoLifeProactiveBench


class EvalGemini(EgoLifeProactiveBench):

    def __init__(self, args):
        self.model_name = getattr(args, 'gemini_model', 'gemini-2.0-flash')
        self.api_key = args.gemini_project
        super().__init__(args)

    def _init_model(self):
        """Initialize Google GenAI client with API key."""
        # self._proxy_on()
        self.client = genai.Client(api_key=self.api_key)

    def _proxy_on(self):
        os.environ['http_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['https_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['HTTP_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['HTTPS_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'

    def _pil_to_part(self, pil_image):
        """Convert PIL Image to google.genai Part."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    def inference(self, frames, prompt):
        """
        Run Gemini inference on sampled frames.

        Args:
            frames: List of PIL Image objects
            prompt: Text prompt string

        Returns:
            str: Model response text
        """
        parts = []
        for frame in frames:
            parts.append(self._pil_to_part(frame))
        parts.append(types.Part.from_text(text=prompt))

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8192,
                ),
            )
            return response.text
        except Exception as e:
            print(f"Gemini inference error: {e}")
            return None
