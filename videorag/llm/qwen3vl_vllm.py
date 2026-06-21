"""Qwen3-VL local vLLM backend (OpenAI-compatible).

This is a drop-in replacement for ``videorag.llm.qwen3vl_api.mllm_response`` that
talks to a vLLM serve endpoint running on the SAME machine (default
``http://localhost:8000/v1``) instead of DashScope.

Spin up the server with:
    bash scripts/launch_vllm_qwen3vl.sh

Then load via HyperVideoGraph:
    hyper_vg.load_caption_model(
        "qwen_vllm",
        vllm_base_url="http://localhost:8000/v1",
        vllm_model_name="Qwen3-VL-8B-Instruct",
    )

Same signature as ``qwen3vl_api.mllm_response`` so existing pipelines work
without code changes.
"""

import base64
import io
import json
import re
from typing import List, Optional

from PIL import Image
from openai import OpenAI


def mllm_response(
    video_llm,  # Unused, kept for interface compatibility
    processor,  # Unused, kept for interface compatibility
    user_prompt: str,
    system_prompt: Optional[str] = None,
    base64_frames: Optional[List[str]] = None,
    max_new_tokens: int = 2048,
    has_image: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    **kwargs,
) -> str:
    """Run Qwen3-VL caption inference against a local vLLM OpenAI-compatible
    server.

    Extra kwargs:
        base_url:   vLLM endpoint URL. Default ``http://localhost:8000/v1``.
        api_key:    Token expected by vLLM (vLLM accepts any non-empty string).
                    Default ``"EMPTY"``.
        model_name: Name used at ``vllm serve --served-model-name <name>``.
                    Default ``"Qwen3-VL-8B-Instruct"``.
    """
    base_url = kwargs.get("base_url") or "http://localhost:8000/v1"
    api_key = kwargs.get("api_key") or "EMPTY"
    model_name = kwargs.get("model_name", "Qwen3-VL-8B-Instruct")

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content = []
    if user_prompt:
        content.append({"type": "text", "text": user_prompt})

    if has_image and base64_frames:
        for idx, frame_b64 in enumerate(base64_frames):
            try:
                # Validate frame
                Image.open(io.BytesIO(base64.b64decode(frame_b64)))
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                    },
                })
            except Exception as e:
                print(f"Warning: Failed to decode base64 frame {idx}: {e}")
                continue

    messages.append({"role": "user", "content": content})

    gen_params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_new_tokens,
    }
    if temperature is not None:
        gen_params["temperature"] = temperature
    if top_p is not None:
        gen_params["top_p"] = top_p

    try:
        response = client.chat.completions.create(**gen_params)
        response_text = response.choices[0].message.content.strip()

        # Strip optional ```json``` fences just like the API backend does, so
        # downstream JSON parsing logic in ovobench_hyper_processing._safe_json_parse
        # sees the same string format.
        m = re.search(r'```(?:json)?\s*\n(.*?)\n?```',
                      response_text, re.DOTALL | re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            try:
                json.loads(inner)
                return inner
            except json.JSONDecodeError:
                pass

        m = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
        if m:
            inner = m.group(1).strip()
            try:
                json.loads(inner)
                return inner
            except json.JSONDecodeError:
                pass

        return response_text
    except Exception as e:
        raise RuntimeError(f"Qwen vLLM error: {e}") from e
