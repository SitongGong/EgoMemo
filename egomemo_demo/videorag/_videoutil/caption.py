import os
import json
import torch
import numpy as np
import base64
import io
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from moviepy.video.io.VideoFileClip import VideoFileClip
from qwen_vl_utils import process_vision_info
from videorag.llm.qwen3vl_api import mllm_response

def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames
    
def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    try:
        model = AutoModel.from_pretrained('/root/githubs/VideoRAG/MiniCPM-V-2_6-int4', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('/root/githubs/VideoRAG/MiniCPM-V-2_6-int4', trust_remote_code=True)
        model.eval()
        
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                msgs = [{'role': 'user', 'content': video_frames + [query]}]
                params = {}
                params["use_image_id"] = False
                params["max_slice_nums"] = 2
                segment_caption = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    **params
                )
                caption_result[index] = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
                torch.cuda.empty_cache()
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments
        
def retrieved_segment_caption_with_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}'"
        msgs = [{'role': 'user', 'content': video_frames + [query]}]
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2
        segment_caption = caption_model.chat(
            image=None,
            msgs=msgs,
            tokenizer=caption_tokenizer,
            **params
        )
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        torch.cuda.empty_cache()
    return caption_result
        
def retrieved_segment_caption(caption_model, caption_processor, refine_knowledge, retrieved_segments, video_segments, reconstruction_prompt, task_type):
    """
    Generate captions for retrieved video segments using Qwen3-VL model.
    
    Args:
        caption_model: Qwen3-VL model instance (Qwen3VLForConditionalGeneration)
        caption_processor: AutoProcessor for Qwen3-VL
        refine_knowledge: Knowledge to extract from the video
        retrieved_segments: List of segment IDs to caption
        video_path_db: Database containing video paths
        video_segments: Database containing video segment information
        num_sampled_frames: Number of frames to sample from each segment
    
    Returns:
        Dictionary mapping segment IDs to captions
    """
    caption_result = {}
    
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        # Extract time_span from segment_id (format: time_span_index)
        time_span = '_'.join(this_segment.split('_')[:-1]) if '_' in this_segment else this_segment
        index = this_segment.split('_')[-1] if '_' in this_segment else '0'
        
        base64_frames = video_segments._data[time_span]["video_frames"]
        original_caption = video_segments._data[time_span]["content"]
        video_frames = []
        for frame_base64 in base64_frames:
            # Decode base64 to bytes
            frame_bytes = base64.b64decode(frame_base64)
            # Convert bytes to PIL Image
            frame_image = Image.open(io.BytesIO(frame_bytes))
            video_frames.append(frame_image)
        
        # segment_transcript = video_segments._data[time_span][index]["transcript"]
        # query = f"\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}"
        # 确保 refine_knowledge 是字符串类型
        if isinstance(refine_knowledge, list):
            keywords_str = ', '.join(str(k) for k in refine_knowledge)
        else:
            keywords_str = str(refine_knowledge)
        query = reconstruction_prompt + f"\n\n### Retrieval Keywords ### {keywords_str} \n### Original Fine-Grained Caption (REFERENCE ONLY) ###\n {original_caption}" # .format(keywords=keywords_str, original_caption=original_caption)
        
        # Build messages in Qwen3-VL format
        content = [{"type": "text", "text": query}]
        
        # Add video frames (list of PIL Images) as video input
        if len(video_frames) > 1:
            # Multiple frames: treat as video (more token-efficient)
            content.append({"type": "video", "video": video_frames})
        elif len(video_frames) == 1:
            # Single frame: treat as image
            content.append({"type": "image", "image": video_frames[0]})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Apply chat template
        text_prompt = caption_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info using qwen_vl_utils
        vision_outputs = process_vision_info(messages)
        image_inputs = vision_outputs[0] if len(vision_outputs) > 0 else None
        video_inputs = vision_outputs[1] if len(vision_outputs) > 1 else None
        
        # Prepare inputs for processor
        inputs = caption_processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the same device as the model
        if hasattr(caption_model, 'device'):
            device = caption_model.device
        elif hasattr(caption_model, 'model') and hasattr(caption_model.model, 'device'):
            device = caption_model.model.device
        else:
            device = next(caption_model.parameters()).device
        
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = caption_model.generate(
                **inputs,
                max_new_tokens=512,
                return_dict_in_generate=True
            )
        
        # Extract generated tokens (remove input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
        ]
        
        # Decode output
        output_text = caption_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        segment_caption = output_text[0].strip() if output_text else ""
    
        # Clean up response (remove newlines and endoftext tokens)
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "").strip()

        # Handle various output formats:
        # 1. JSON format: {"caption": "..."} -> extract the value
        # 2. Empty string "" -> skip
        # 3. Plain caption text "xxx" -> use directly
        if not this_caption or this_caption == '""' or this_caption == "''":
            continue
        try:
            parsed = json.loads(this_caption)
            if isinstance(parsed, dict) and "caption" in parsed:
                this_caption = parsed["caption"].strip()
            elif isinstance(parsed, str):
                this_caption = parsed.strip()
        except (json.JSONDecodeError, TypeError):
            pass  # not JSON, use as-is

        if not this_caption:
            continue

        caption_result[this_segment] = f"Caption:\n{this_caption}\n"
        torch.cuda.empty_cache()
    
    return caption_result

def retrieved_segment_caption_qwen3_api(caption_model, caption_processor, refine_knowledge, retrieved_segments, video_segments, reconstruction_prompt, task_type):
    """
    Generate captions for retrieved video segments using Qwen3-VL API.
    
    Args:
        caption_model: Not used for API (kept for compatibility)
        caption_processor: Not used for API (kept for compatibility)
        refine_knowledge: Knowledge to extract from the video
        retrieved_segments: List of segment IDs to caption
        video_segments: Database containing video segment information
        reconstruction_prompt: Prompt template for caption reconstruction
        task_type: Task type (not used, kept for compatibility)
    
    Returns:
        Dictionary mapping segment IDs to captions
    """
    caption_result = {}
    
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query (Qwen3 API)'):
        # Extract time_span from segment_id (format: time_span_index)
        time_span = '_'.join(this_segment.split('_')[:-1]) if '_' in this_segment else this_segment
        index = this_segment.split('_')[-1] if '_' in this_segment else '0'
        
        # Get base64 frames directly (already in string format, no need to decode)
        base64_frames = video_segments._data[time_span]["video_frames"]
        original_caption = video_segments._data[time_span]["content"]
        
        # Process refine_knowledge (support both list and string)
        if isinstance(refine_knowledge, list):
            keywords_str = ', '.join(str(k) for k in refine_knowledge)
        else:
            keywords_str = str(refine_knowledge)
        
        # Build query using reconstruction_prompt
        query = reconstruction_prompt.format(keywords=keywords_str, original_caption=original_caption)
        
        # query = reconstruction_prompt + f"\n\n### Retrieval Keywords ### {keywords_str} \n### Original Fine-Grained Caption (REFERENCE ONLY) ###\n {original_caption}" # .format(keywords=keywords_str, original_caption=original_caption)
        
        # Call Qwen3-VL API using mllm_response
        try:
            segment_caption = mllm_response(
                video_llm=caption_model,  # Not used by API, kept for compatibility
                processor=caption_processor,  # Not used by API, kept for compatibility
                user_prompt=query,
                system_prompt=None,
                base64_frames=base64_frames,
                max_new_tokens=512,
                has_image=True
            )
        except Exception as e:
            print(f"Warning: Error calling Qwen3 API for segment {this_segment}: {e}")
            segment_caption = f"Error generating caption: {str(e)}"
        
        # Clean up response (remove newlines and endoftext tokens)
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "").strip()

        # Handle various output formats:
        # 1. JSON format: {"caption": "..."} -> extract the value
        # 2. Empty string "" -> skip
        # 3. Plain caption text "xxx" -> use directly
        if not this_caption or this_caption == '""' or this_caption == "''":
            continue
        try:
            parsed = json.loads(this_caption)
            if isinstance(parsed, dict) and "caption" in parsed:
                this_caption = parsed["caption"].strip()
            elif isinstance(parsed, str):
                this_caption = parsed.strip()
        except (json.JSONDecodeError, TypeError):
            pass  # not JSON, use as-is

        if not this_caption:
            continue

        caption_result[this_segment] = f"Caption:\n{this_caption}\n"
        torch.cuda.empty_cache()
    
    return caption_result

def retrieved_segment_caption_minicpm(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_segments, reconstruction_prompt, task_type):
    """
    Generate captions for retrieved video segments using MiniCPM model.
    
    Args:
        caption_model: MiniCPM model instance (AutoModel)
        caption_tokenizer: AutoTokenizer for MiniCPM
        refine_knowledge: Knowledge to extract from the video
        retrieved_segments: List of segment IDs to caption
        video_path_db: Database containing video paths
        video_segments: Database containing video segment information
        num_sampled_frames: Number of frames to sample from each segment
    
    Returns:
        Dictionary mapping segment IDs to captions
    """
    caption_result = {}
    
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        # Extract time_span from segment_id (format: time_span_index)
        time_span = '_'.join(this_segment.split('_')[:-1]) if '_' in this_segment else this_segment
        index = this_segment.split('_')[-1] if '_' in this_segment else '0'
        
        base64_frames = video_segments._data[time_span]["video_frames"]
        original_caption = video_segments._data[time_span]["content"]
        video_frames = []
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
        
        # query = f"\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}"
        # 确保 refine_knowledge 是字符串类型
        if isinstance(refine_knowledge, list):
            keywords_str = ', '.join(str(k) for k in refine_knowledge)
        else:
            keywords_str = str(refine_knowledge)
        query = reconstruction_prompt.format(keywords=keywords_str, original_caption=original_caption)
        
        # Build messages in MiniCPM format: [{'role': 'user', 'content': [<0>, image0, <1>, image1, ..., query]}]
        # The content is a list where each frame is preceded by its index marker, then the text query
        content = []
        if video_frames:
            # Add index markers and images in alternating order
            for idx, frame_image in video_frames:
                content.append(f"<{idx}>")  # Add index marker before each frame
                content.append(frame_image)  # Add the frame image
        content.append(query)  # Add the text query at the end
        
        msgs = [{'role': 'user', 'content': content}]
        
        # Prepare parameters for model.chat()
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2
        params["max_new_tokens"] = 512
        
        # Generate response using model.chat() method
        with torch.no_grad():
            segment_caption = caption_model.chat(
                image=None,
                msgs=msgs,
                tokenizer=caption_tokenizer,
                **params
            )
        
        # Clean up response (remove newlines and endoftext tokens)
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "").strip()

        # Handle various output formats:
        # 1. JSON format: {"caption": "..."} -> extract the value
        # 2. Empty string "" -> skip
        # 3. Plain caption text "xxx" -> use directly
        if not this_caption or this_caption == '""' or this_caption == "''":
            continue
        try:
            parsed = json.loads(this_caption)
            if isinstance(parsed, dict) and "caption" in parsed:
                this_caption = parsed["caption"].strip()
            elif isinstance(parsed, str):
                this_caption = parsed.strip()
        except (json.JSONDecodeError, TypeError):
            pass  # not JSON, use as-is

        if not this_caption:
            continue

        caption_result[this_segment] = f"Caption:\n{this_caption}\n"
        torch.cuda.empty_cache()
    
    return caption_result