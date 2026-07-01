"""
Qwen2-VL Proactive Service Evaluation for EgoLife.

Uses Qwen2-VL model (local HuggingFace) to detect proactive services
from uniformly sampled egocentric video frames.

Weight from:
  - https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
  - https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct

Inference Platform:
  - 7B: 4*A100 80GB
  - 72B: 8*A100 80GB

Usage:
    python inference.py --model QWen2VL_7B --model_path /path/to/Qwen2-VL-7B-Instruct ...
"""

import io
import base64
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.egolife_bench import EgoLifeProactiveBench


class EvalQWen2VL(EgoLifeProactiveBench):

    def __init__(self, args):
        super().__init__(args)

    def _init_model(self):
        """Initialize Qwen2-VL model and processor."""
        model_path = self.args.model_path
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def _pil_to_base64(self, pil_image):
        """Convert PIL Image to base64 data URL."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    def inference(self, frames, prompt):
        """
        Run Qwen2-VL inference on sampled frames.

        Args:
            frames: List of PIL Image objects
            prompt: Text prompt string

        Returns:
            str: Model response text
        """
        # Build message content with multiple images
        content = []
        for frame in frames:
            content.append({
                "type": "image",
                "image": self._pil_to_base64(frame),
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 360 * 420,
            })

        content.append({
            "type": "text",
            "text": prompt,
        })

        messages = [{"role": "user", "content": content}]

        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0] if output_text else None
