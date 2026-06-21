from transformers import AutoProcessor
import torch
import base64
import io
import os
from PIL import Image
from qwen_vl_utils import process_vision_info
    

def mllm_response(video_llm, processor, user_prompt, system_prompt=None, base64_frames=None, max_new_tokens=512, has_image=True, temperature=None, top_p=None):
    """
    Generate response using Qwen3VL model with base64-encoded video frames or pure text.
    Optimized for speed with greedy decoding and inference mode.
    
    Args:
        video_llm: The Qwen3VL model instance
        processor: AutoProcessor for Qwen3VL
        user_prompt: Text prompt
        system_prompt: Optional system prompt
        base64_frames: List of base64-encoded frame strings (JPEG format). If None and has_image=False, pure text mode.
        max_new_tokens: Maximum number of tokens to generate
        has_image: Whether the input contains images. If False, process as pure text.
        temperature: Sampling temperature (if None, uses greedy decoding for speed)
        top_p: Top-p sampling parameter (if None, uses greedy decoding for speed)
    
    Returns:
        Generated text response
    """
    # Build messages in Qwen3VL format
    content = [{"type": "text", "text": user_prompt}]
    
    # Handle image/video frames if provided
    if has_image and base64_frames is not None and len(base64_frames) > 0:
        # Convert all frames to images (multiple images instead of video)
        # Qwen VL models can handle multiple images in the content
        # Add index markers before each frame: <0>, <1>, <2>, etc.
        for idx, frame_base64 in enumerate(base64_frames):
            try:
                # Add index marker before each frame
                content.append({"type": "text", "text": f"<{idx}>"})
                
                # Decode base64 to bytes
                frame_bytes = base64.b64decode(frame_base64)
                # Convert bytes to PIL Image
                frame_image = Image.open(io.BytesIO(frame_bytes))
                # Add each frame as a separate image
                content.append({"type": "image", "image": frame_image})
            except Exception as e:
                print(f"Warning: Failed to decode base64 frame: {e}")
                continue
    
    if system_prompt:
        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": system_prompt}]
            }
        ]
        messages.append({
            "role": "user",
            "content": content
        })
    else:
        messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Apply chat template
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info (images) using qwen_vl_utils
    # For pure text input, image_inputs and video_inputs will be None
    vision_outputs = process_vision_info(messages)
    image_inputs = vision_outputs[0] if len(vision_outputs) > 0 else None
    video_inputs = vision_outputs[1] if len(vision_outputs) > 1 else None
    
    # Prepare inputs for processor
    # For pure text, images and videos will be None
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 从环境变量CUDA_VISIBLE_DEVICES获取设备，固定在该设备上运行
    # 如果设置了CUDA_VISIBLE_DEVICES，使用cuda:0（环境变量会重新映射设备）
    # 如果没有设置，使用默认的cuda:0或cpu
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible and torch.cuda.is_available():
        # 如果设置了CUDA_VISIBLE_DEVICES，使用cuda:0（这是重新映射后的设备）
        target_device = torch.device("cuda:0")
        # 设置当前CUDA设备为cuda:0，确保所有操作都在这个设备上执行
        torch.cuda.set_device(0)
    elif torch.cuda.is_available():
        # 如果没有设置CUDA_VISIBLE_DEVICES，使用默认的cuda:0
        target_device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        target_device = torch.device("cpu")
    
    # 将输入数据移动到目标设备（单卡）
    if isinstance(inputs, dict):
        # Move all tensor values in the dict to the target device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    else:
        # If inputs is not a dict, move it directly
        inputs = inputs.to(target_device)
    
    # Generate response with optimizations for speed
    # Use inference_mode to disable gradient computation (faster)
    with torch.inference_mode():
        # Prepare generation kwargs - 使用模型默认的生成参数
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            # "return_dict_in_generate": True,
            # "use_cache": True,  # Enable KV cache for faster generation
        }
        
        # Set pad_token_id if available
        if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'eos_token_id'):
            generation_kwargs["pad_token_id"] = processor.tokenizer.eos_token_id
        
        # 不设置temperature、top_p、top_k等参数，使用模型默认值
        # 如果需要自定义这些参数，可以在调用时通过generation_config传递
        
        # 确保推理在单卡上运行
        # 由于已经设置了torch.cuda.set_device(0)，所有CUDA操作都会在cuda:0上执行
        generated_ids = video_llm.generate(**inputs, **generation_kwargs)
    
    # Extract generated tokens (remove input tokens)
    # generated_ids 现在直接是tensor，不是字典
    input_ids = inputs.input_ids
    # generated_ids 包含输入+生成的token，需要移除输入部分
    if generated_ids.shape[0] == input_ids.shape[0] and generated_ids.shape[1] > input_ids.shape[1]:
        # 如果batch size匹配且生成的token长度大于输入，直接截取生成的部分
        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
    elif generated_ids.shape[0] == input_ids.shape[0]:
        # 如果长度相同，说明可能只生成了很少的token，或者模型返回格式不同
        # 尝试直接使用，但记录警告
        import warnings
        warnings.warn(f"Generated tokens shape {generated_ids.shape} matches input shape {input_ids.shape}, using full output")
        generated_ids_trimmed = generated_ids
    else:
        # 如果batch size不匹配，可能是单样本情况，尝试处理
        if len(generated_ids.shape) == 2 and len(input_ids.shape) == 2:
            # 都是2D tensor，尝试截取
            if generated_ids.shape[1] > input_ids.shape[1]:
                generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
            else:
                generated_ids_trimmed = generated_ids
        else:
            # 形状不匹配，直接使用（可能是特殊情况）
            generated_ids_trimmed = generated_ids
    
    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    result = output_text[0].strip() if output_text else ""
    
    # 清理输出：移除可能的markdown代码块标记，提取纯JSON
    # 这有助于解决JSON解析错误（如未终止的字符串）
    import re
    import json
    
    # 尝试提取JSON内容（如果响应包含 ```json``` 代码块）
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n?```', result, re.DOTALL | re.IGNORECASE)
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
    json_obj_match = re.search(r'(\{.*\}|\[.*\])', result, re.DOTALL)
    if json_obj_match:
        json_str = json_obj_match.group(1).strip()
        try:
            json.loads(json_str)  # 验证 JSON 格式
            return json_str
        except json.JSONDecodeError:
            pass
    
    # 如果没有找到 JSON，返回原始响应
    # 但尝试修复常见的JSON问题：未转义的换行符和引号
    # 注意：这是一个简单的修复，可能无法处理所有情况
    if result.startswith('{') or result.startswith('['):
        # 尝试修复未转义的换行符（在字符串值中）
        # 这是一个保守的修复，只处理明显的问题
        try:
            # 先尝试直接解析
            json.loads(result)
            return result
        except json.JSONDecodeError:
            # 如果失败，尝试修复常见的未转义换行符问题
            # 将字符串值中的换行符转义（但这是危险的，可能会破坏合法的换行符）
            # 更安全的做法是让调用者处理，或者使用更智能的JSON修复库
            pass
    
    return result
