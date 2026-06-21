"""
GPT-4o API wrapper for caption generation with multiple images.
Compatible with mllm_response function interface.
"""

import os
import base64
import io
import re
import json
from typing import Optional, List
from PIL import Image
from openai import OpenAI


def mllm_response(
    video_llm,  # Not used for API, kept for compatibility
    processor,  # Not used for API, kept for compatibility
    user_prompt: str,
    system_prompt: Optional[str] = None,
    base64_frames: Optional[List[str]] = None,
    max_new_tokens: int = 512,
    has_image: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    **kwargs
) -> str:
    """
    Generate response using GPT-4o API with base64-encoded images.
    
    Args:
        video_llm: Not used (kept for compatibility with mllm_response interface)
        processor: Not used (kept for compatibility with mllm_response interface)
        user_prompt: Text prompt for caption generation
        system_prompt: Optional system prompt
        base64_frames: List of base64-encoded frame strings (JPEG format)
        max_new_tokens: Maximum number of tokens to generate
        has_image: Whether the input contains images
        temperature: Sampling temperature (0.0-2.0)
        top_p: Top-p sampling parameter (0.0-1.0)
        **kwargs: Additional arguments (e.g., api_key, model_name, base_url)
    
    Returns:
        Generated text response
    """
    # Get API key from kwargs or environment variable
    api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key in kwargs.")
    
    # Get model name from kwargs or use default
    model_name = kwargs.get("model_name", "gpt-4o")
    
    # Get base_url from kwargs (for custom endpoints like Azure OpenAI)
    base_url = kwargs.get("base_url", None)
    
    # Initialize OpenAI client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    
    # Build messages
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Build user message content
    content = []
    
    # Add text prompt
    if user_prompt:
        content.append({
            "type": "text",
            "text": user_prompt
        })
    
    # Add images if provided
    if has_image and base64_frames is not None and len(base64_frames) > 0:
        for idx, frame_base64 in enumerate(base64_frames):
            try:
                # Validate base64 image
                frame_bytes = base64.b64decode(frame_base64)
                Image.open(io.BytesIO(frame_bytes))  # Validate it's a valid image
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_base64}"
                    }
                })
            except Exception as e:
                print(f"Warning: Failed to decode base64 frame {idx}: {e}")
                continue
    
    messages.append({
        "role": "user",
        "content": content
    })
    
    # Prepare generation parameters
    generation_params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_new_tokens,
    }
    
    if temperature is not None:
        generation_params["temperature"] = temperature
    if top_p is not None:
        generation_params["top_p"] = top_p
    
    # Generate response
    try:
        response = client.chat.completions.create(**generation_params)
        response_text = response.choices[0].message.content.strip()
        
        # 提取 JSON 内容（如果响应包含 ```json``` 代码块）
        # Pattern: ```json\n...\n``` 或 ```\n...\n``` (处理有无 json 标记的情况)
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n?```', response_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1).strip()
            # 验证是否为有效的 JSON
            try:
                json.loads(json_str)  # 验证 JSON 格式
                return json_str  # 返回纯 JSON 字符串
            except json.JSONDecodeError:
                # 如果不是有效 JSON，继续尝试其他方式
                pass
        
        # 尝试直接提取 JSON 对象或数组（如果响应本身就是 JSON）
        # 查找 { ... } 或 [ ... ] 模式
        json_obj_match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
        if json_obj_match:
            json_str = json_obj_match.group(1).strip()
            try:
                json.loads(json_str)  # 验证 JSON 格式
                return json_str
            except json.JSONDecodeError:
                pass
        
        # 如果没有找到 JSON，返回原始响应
        return response_text
    except Exception as e:
        raise RuntimeError(f"GPT-4o API error: {e}") from e

