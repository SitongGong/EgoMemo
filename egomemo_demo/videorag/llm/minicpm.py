from transformers import AutoModel, AutoTokenizer
import torch
import base64
import io
import os
from PIL import Image

def load_video(video_path, args):
    """
    Load video function (placeholder for compatibility).
    This function may need to be implemented based on your specific requirements.
    """
    # TODO: Implement video loading if needed
    raise NotImplementedError("load_video function needs to be implemented")

def _load_minicpm_model(self, model_path: str, **kwargs):
    """Load MiniCPM model using transformers.
    
    The device is determined by CUDA_VISIBLE_DEVICES environment variable.
    If CUDA_VISIBLE_DEVICES is set, the model will be loaded on the first visible device (cuda:0).
    """
    # 从环境变量获取CUDA_VISIBLE_DEVICES，确定使用的设备
    # 如果设置了CUDA_VISIBLE_DEVICES，则使用cuda:0（因为环境变量会重新映射设备）
    # 如果没有设置，使用auto让系统自动分配
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible and torch.cuda.is_available():
        # 如果设置了CUDA_VISIBLE_DEVICES，固定使用cuda:0（这是重新映射后的设备）
        target_device = "cuda:0"
        device_map = kwargs.get("device_map", target_device)
    else:
        # 如果没有设置，使用auto
        device_map = kwargs.get("device_map", "auto")
    
    # Load MiniCPM model and tokenizer
    self.video_llm = AutoModel.from_pretrained(
        "/root/models/MiniCPM-o-4_5", # model_path,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation="sdpa", # sdpa or flash_attention_2
        torch_dtype=torch.bfloat16,
        # init_vision=True,
        # init_audio=True,
        # init_tts=True, 
    )
    self.processor = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    self.image_processor = self.processor  # For compatibility
    
    # Set model to eval mode
    self.video_llm.eval()
    
    # 如果使用device_map，模型已经加载到指定设备
    # 如果device_map为None，手动移动到设备
    if device_map is None or (isinstance(device_map, str) and device_map != "auto"):
        if cuda_visible and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.video_llm.to(device)
        
    return self.video_llm, self.processor, self.image_processor
    

def mllm_response(video_llm, tokenizer, user_prompt, system_prompt=None, base64_frames=None, max_new_tokens=512, has_image=True, temperature=None, top_p=None):
    """
    Generate response using MiniCPM model with base64-encoded video frames or pure text.
    Mimics the usage pattern from caption.py.
    
    Args:
        video_llm: The MiniCPM model instance (AutoModel)
        tokenizer: AutoTokenizer for MiniCPM
        user_prompt: Text prompt
        system_prompt: Optional system prompt (not used in MiniCPM chat format)
        base64_frames: List of base64-encoded frame strings (JPEG format). If None and has_image=False, pure text mode.
        max_new_tokens: Maximum number of tokens to generate
        has_image: Whether the input contains images. If False, process as pure text.
        temperature: Sampling temperature (if None, uses default model settings)
        top_p: Top-p sampling parameter (if None, uses default model settings)
    
    Returns:
        Generated text response
    """
    # Convert base64 frames to PIL Images if provided
    video_frames = []
    if has_image and base64_frames is not None and len(base64_frames) > 0:
        for idx, frame_base64 in enumerate(base64_frames):
            try:
                # Decode base64 to bytes
                frame_bytes = base64.b64decode(frame_base64)
                # Convert bytes to PIL Image
                frame_image = Image.open(io.BytesIO(frame_bytes)).resize((1280, 720))
                video_frames.append((idx, frame_image))
            except Exception as e:
                print(f"Warning: Failed to decode base64 frame: {e}")
                continue
    
    # Build messages in MiniCPM format: [{'role': 'user', 'content': [<0>, image0, <1>, image1, ..., query]}]
    # The content is a list where each frame is preceded by its index marker, then the text query
    content = []
    if video_frames:
        # Add index markers and images in alternating order
        for idx, frame_image in video_frames:
            content.append(f"<{idx}>")  # Add index marker before each frame
            content.append(frame_image)  # Add the frame image
    content.append(user_prompt)  # Add the text query at the end
    
    msgs = [{'role': 'user', 'content': content}]
    
    # Prepare parameters for model.chat()
    params = {}
    params["use_image_id"] = False
    params["max_slice_nums"] = 2
    
    # Add generation parameters if provided
    if max_new_tokens is not None:
        params["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    
    # Generate response using model.chat() method
    with torch.no_grad():
        response = video_llm.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
    
    # Clean up response (remove newlines and endoftext tokens)
    cleaned_response = response.replace("\n", "").replace("<|endoftext|>", "").strip()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    return cleaned_response
