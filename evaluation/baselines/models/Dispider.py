"""
Dispider Proactive Service Evaluation for EgoLife / HoloAssist / CaptionCook4D.

Uses the Dispider model (CVPR 2025) for proactive service detection from
egocentric video frames.

Dispider processes video as a whole (not individual frames), so we reconstruct
a short video clip from the sampled PIL frames and feed it through the model's
native video understanding pipeline.

Weight from:
  - https://huggingface.co/Mar2Ding/Dispider

Usage:
    # EgoLife benchmark
    python inference.py \
        --model Dispider \
        --model_path /path/to/Dispider/checkpoint \
        --data_dir /mnt/tokyo-ai/gst/EgoLife/egolife \
        --persons A1_JAKE \
        --days DAY1

    # HoloAssist / CaptionCook4D benchmark
    python inference_video.py \
        --dataset holoassist \
        --model Dispider \
        --model_path /path/to/Dispider/checkpoint
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

from transformers import StoppingCriteria, StoppingCriteriaList

# Add Dispider repo to path so we can import its modules
DISPIDER_REPO = os.environ.get(
    "DISPIDER_REPO", "./Dispider"
)
if DISPIDER_REPO not in sys.path:
    sys.path.insert(0, DISPIDER_REPO)

from dispider.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_ANS_TOKEN,
    DEFAULT_TODO_TOKEN,
)
from dispider.conversation import conv_templates
from dispider.model.builder import load_pretrained_model
from dispider.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
)


# ============================================================================
# Stopping criteria (same as Dispider inference.py)
# ============================================================================

class _StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


# ============================================================================
# Video processing utilities (adapted from Dispider inference.py)
# ============================================================================

def _get_seq_frames(total_num_frames, desired_num_frames):
    """Uniformly sample frame indices (median of each segment)."""
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq


def _preprocess_time_from_duration(duration_seconds, num_clip, num_frm, tokenizer):
    """
    Build temporal token sequences from known duration.
    Since we have PIL frames (not a video file), we synthesize time info
    based on the total duration and number of clips.
    """
    clip_duration = duration_seconds / num_clip
    seq = []
    for i in range(num_clip):
        start = int(np.round(i * clip_duration))
        end = int(np.round((i + 1) * clip_duration))
        sentence = 'This contains a clip sampled in %d to %d seconds' % (start, end) + DEFAULT_IMAGE_TOKEN
        sentence = tokenizer_image_token(sentence, tokenizer, return_tensors='pt')
        seq.append(sentence)
    return seq


def _preprocess_question(question_text, tokenizer):
    """Tokenize question with <to_do> token appended."""
    sentence = tokenizer_image_token(
        question_text + DEFAULT_TODO_TOKEN, tokenizer, return_tensors='pt'
    )
    return [sentence]


def _prepare_frames_for_dispider(
    pil_frames,
    duration_seconds,
    model_config,
    tokenizer,
    image_processor,
    image_processor_large,
    time_tokenizer,
    question_text,
    num_frm=16,
):
    """
    Convert a list of PIL Image frames + text prompt into Dispider model inputs.

    Dispider expects video organized as (num_clips, num_frm, C, H, W).
    We treat the input frames as a flat sequence and reshape accordingly.

    Args:
        pil_frames: list of PIL.Image — the sampled frames
        duration_seconds: float — total duration of the video segment
        model_config: model.config object
        tokenizer: main tokenizer
        image_processor: CLIP image processor
        image_processor_large: CLIP image processor (large variant)
        time_tokenizer: temporal tokenizer
        question_text: str — the prompt/question
        num_frm: int — frames per clip (Dispider default: 16)

    Returns:
        input_ids, image_tensor, image_tensor_large, seqs, compress_mask, qs, qs_mask
    """
    total_frames = len(pil_frames)

    # Determine number of clips
    num_clip = max(1, total_frames // num_frm)
    # Limit to a reasonable max
    num_clip = min(num_clip, 4)
    total_num_frm = num_frm * num_clip

    # If we have fewer frames than needed, resample
    if total_frames < total_num_frm:
        # Resample with replacement to fill
        indices = _get_seq_frames(total_frames, total_num_frm)
        frames = [pil_frames[min(idx, total_frames - 1)] for idx in indices]
    elif total_frames > total_num_frm:
        # Subsample
        indices = _get_seq_frames(total_frames, total_num_frm)
        frames = [pil_frames[idx] for idx in indices]
    else:
        frames = list(pil_frames)

    # Make frames square (match Dispider's load_video behavior)
    processed_frames = []
    for frame in frames:
        w, h = frame.size
        if w != h:
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            frame = frame.crop((left, top, left + size, top + size))
        processed_frames.append(frame)

    # Build prompt
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_text
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question_text

    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Process frames through CLIP
    video = image_processor.preprocess(processed_frames, return_tensors='pt')['pixel_values']
    video = video.view(num_clip, num_frm, *video.shape[1:])

    video_large = image_processor_large.preprocess(processed_frames, return_tensors='pt')['pixel_values']
    video_large = video_large.view(num_clip, num_frm, *video_large.shape[1:])[:, :1].contiguous()

    # Temporal tokens
    seqs = _preprocess_time_from_duration(duration_seconds, num_clip, num_frm, time_tokenizer)
    seqs = torch.nn.utils.rnn.pad_sequence(
        seqs,
        batch_first=True,
        padding_value=time_tokenizer.pad_token_id,
    )
    compress_mask = seqs.ne(time_tokenizer.pad_token_id)

    # Question tokens
    question_tokens = _preprocess_question(question_text, time_tokenizer)
    question_tokens = torch.nn.utils.rnn.pad_sequence(
        question_tokens,
        batch_first=True,
        padding_value=time_tokenizer.pad_token_id,
    )
    qs_mask = question_tokens.ne(time_tokenizer.pad_token_id)

    # Input IDs
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    return input_ids, video, video_large, seqs, compress_mask, question_tokens, qs_mask


# ============================================================================
# Dispider Model Wrapper
# ============================================================================

class DispiderModel:
    """
    Wraps Dispider model for use in the proactive service benchmark.

    Provides a simple `inference(frames, prompt)` interface that:
    1. Converts PIL frames into Dispider's expected video tensor format
    2. Runs the model's generate method
    3. Returns the text output
    """

    def __init__(self, model_path, device="cuda", num_frm=16):
        """
        Initialize Dispider model.

        Args:
            model_path: Path to Dispider model checkpoint
            device: CUDA device
            num_frm: Frames per clip (default 16, Dispider's default)
        """
        self.device = device
        self.num_frm = num_frm

        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, image_processor, self.context_len = \
            load_pretrained_model(model_path, None, model_name)

        self.image_processor, self.time_tokenizer = image_processor
        self.image_processor_large = self.image_processor

        if self.time_tokenizer.pad_token is None:
            self.time_tokenizer.pad_token = '<pad>'

        # Stopping criteria
        stop_words_ids = [
            torch.tensor(self.tokenizer('<|im_end|>').input_ids).cuda(),
        ]
        self.stopping_criteria = StoppingCriteriaList(
            [_StoppingCriteriaSub(stops=stop_words_ids)]
        )

    def inference(self, frames, prompt, duration_seconds=30.0, max_new_tokens=1024):
        """
        Run Dispider inference on a list of PIL frames with a text prompt.

        Args:
            frames: list of PIL.Image objects
            prompt: str — text prompt
            duration_seconds: float — estimated duration of the video segment
            max_new_tokens: int — max tokens to generate

        Returns:
            str: model's text response
        """
        if not frames:
            return None

        input_ids, image_tensor, image_tensor_large, seqs, compress_mask, qs, qs_mask = \
            _prepare_frames_for_dispider(
                pil_frames=frames,
                duration_seconds=duration_seconds,
                model_config=self.model.config,
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                image_processor_large=self.image_processor_large,
                time_tokenizer=self.time_tokenizer,
                question_text=prompt,
                num_frm=self.num_frm,
            )

        input_ids = input_ids.unsqueeze(0).to(device=self.device, non_blocking=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device=self.device, non_blocking=True),
                images_large=image_tensor_large.to(dtype=torch.float16, device=self.device, non_blocking=True),
                seqs=seqs.to(device=self.device, non_blocking=True),
                compress_mask=compress_mask.to(device=self.device, non_blocking=True),
                qs=qs.to(device=self.device, non_blocking=True),
                qs_mask=qs_mask.to(device=self.device, non_blocking=True),
                ans_token=self.time_tokenizer(
                    DEFAULT_ANS_TOKEN, return_tensors="pt"
                ).input_ids.to(device=self.device, non_blocking=True),
                todo_token=self.time_tokenizer(
                    DEFAULT_TODO_TOKEN, return_tensors="pt"
                ).input_ids.to(device=self.device, non_blocking=True),
                q_id=None,
                insert_position=0,
                ans_position=[],
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.stopping_criteria,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        return outputs


# ============================================================================
# EgoLife Benchmark Integration
# ============================================================================

class EvalDispider:
    """
    Dispider evaluation class for EgoLife ProactiveBench.

    Inherits the evaluation loop from EgoLifeProactiveBench but uses
    Dispider for inference.
    """

    def __new__(cls, args):
        """Dynamically create a class that inherits from EgoLifeProactiveBench."""
        from utils.egolife_bench import EgoLifeProactiveBench

        class _DispiderEgoLife(EgoLifeProactiveBench):
            def __init__(self, a):
                self.model_path_str = a.model_path
                super().__init__(a)

            def _init_model(self):
                self.dispider = DispiderModel(self.model_path_str)

            def inference(self, frames, prompt):
                duration = len(frames) * 3.75  # ~30s / 8 frames
                return self.dispider.inference(
                    frames, prompt, duration_seconds=duration
                )

        return _DispiderEgoLife(args)


# ============================================================================
# VideoBench Integration (HoloAssist / CaptionCook4D)
# ============================================================================

class EvalDispiderVideo:
    """
    Dispider evaluation class for VideoBench (HoloAssist / CaptionCook4D).
    """

    def __new__(cls, args):
        """Dynamically create a class that inherits from VideoBench."""
        from utils.video_bench import VideoBench

        class _DispiderVideo(VideoBench):
            def __init__(self, a):
                self.model_path_str = a.model_path
                super().__init__(a)

            def _init_model(self):
                self.dispider = DispiderModel(self.model_path_str)

            def inference(self, frames, prompt):
                duration = getattr(self.args, 'window_seconds', 30)
                return self.dispider.inference(
                    frames, prompt, duration_seconds=duration
                )

        return _DispiderVideo(args)
