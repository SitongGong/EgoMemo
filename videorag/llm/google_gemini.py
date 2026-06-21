"""
Google Gemini Model Wrapper with comprehensive video and image processing capabilities.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import google.generativeai as genai
from decord import VideoReader, cpu
from PIL import Image
from tqdm.asyncio import tqdm as tqdm_asyncio

import sys
from pathlib import Path

# Add worldmm to path for importing utils
worldmm_path = Path(__file__).parent.parent.parent / "worldmm" / "llm"
if str(worldmm_path) not in sys.path:
    sys.path.insert(0, str(worldmm_path.parent.parent))

from worldmm.llm.utils import dynamic_retry_decorator

# Configure logging
logger = logging.getLogger(__name__)

# Model configuration
MODEL_DICT = {
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",
}


class GeminiModelError(Exception):
    """Custom exception for Gemini model operations."""
    pass


class GeminiModel:
    """
    Google Gemini model wrapper with video and image processing capabilities.
    """

    def __init__(
        self,
        model_name: str,
        max_retries: int = 3,
        max_size: Tuple[int, int] = (512, 512),
        max_size_video: Tuple[int, int] = (256, 256),
        quality: int = 85,
        fps: Optional[int] = None,
        nframes: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Gemini model wrapper.
        
        Args:
            model_name: Name of the Gemini model to use
            max_retries: Maximum number of retry attempts
            max_size: Maximum size for image thumbnails
            max_size_video: Maximum size for video frames
            quality: JPEG quality for encoding (1-100)
            fps: Frames per second for video sampling
            nframes: Number of frames to sample from video
            api_key: Google API key (uses env var if not provided)
            **kwargs: Additional arguments passed to Gemini API
            
        Raises:
            GeminiModelError: If model initialization fails
            ValueError: If both fps and nframes are provided
        """
        # Validate parameters
        if fps is not None and nframes is not None:
            raise ValueError("Cannot provide both 'fps' and 'nframes'. Please choose one for video sampling.")
            
        if model_name not in MODEL_DICT:
            raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_DICT.keys())}")

        # Initialize API key
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise GeminiModelError("Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable or pass api_key parameter.")

        # Configure Gemini
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(MODEL_DICT[model_name])
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise GeminiModelError("Failed to initialize Gemini client") from e

        # Set instance attributes
        self.model_name = MODEL_DICT[model_name]
        self.max_retries = max(1, max_retries)
        self.max_size = max_size
        self.max_size_video = max_size_video
        self.quality = max(1, min(100, quality))  # Clamp quality between 1-100
        self.fps = fps
        self.nframes = nframes
        self.kwargs = kwargs

        logger.info(f"Initialized GeminiModel with {self.model_name}")

    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and convert file path to Path object."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path

    def _image_cache_identifier(self, img: Image.Image) -> str:
        """Return a stable identifier for a PIL Image for caching."""
        filename = getattr(img, "filename", None)
        if filename:
            return str(filename)
        
        try:
            buf = io.BytesIO()
            img_rgb = img.convert("RGB")
            img_rgb.save(buf, format="PNG", optimize=True)
            import hashlib
            return hashlib.md5(buf.getvalue()).hexdigest()
        except Exception:
            return f"pil-id-{id(img)}"

    def encode_image(self, image: Union[str, Path, Image.Image]) -> bytes:
        """
        Encode image to bytes with optimization.
        
        Args:
            image: Path to the image file or a PIL Image object

        Returns:
            Image bytes (JPEG format)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            GeminiModelError: If image processing fails
        """
        try:
            if isinstance(image, Image.Image):
                img = image
            else:
                path = self._validate_file_path(image)
                img = Image.open(path)

            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize maintaining aspect ratio
            img.thumbnail(self.max_size, Image.Resampling.LANCZOS)

            # Encode to JPEG
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=self.quality, optimize=True)
            image_bytes = buffered.getvalue()

            # If we opened the image from path, close the file-handle
            if not isinstance(image, Image.Image):
                try:
                    img.close()
                except Exception:
                    pass
            
            return image_bytes
            
        except Exception as e:
            logger.error(f"Failed to encode image {image}: {e}")
            raise GeminiModelError(f"Failed to encode image: {e}") from e

    def encode_video(self, video_path: Union[str, Path]) -> List[bytes]:
        """
        Encode video frames to bytes with intelligent sampling.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of frame bytes (JPEG format)
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            GeminiModelError: If video processing fails
        """
        video_path = self._validate_file_path(video_path)
        
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames == 0:
                raise GeminiModelError(f"Video file appears to be empty or corrupted: {video_path}")

            # Determine sampling strategy
            sample_indices = self._calculate_sample_indices(vr, total_frames)
            
            # Extract and encode frames
            frame_bytes_list = []
            for idx in sample_indices:
                try:
                    frame = vr[idx].asnumpy()  # RGB format from decord
                    
                    # Convert to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Resize frame
                    frame_resized = cv2.resize(frame_bgr, self.max_size_video, interpolation=cv2.INTER_LANCZOS4)
                    
                    # Encode to JPEG
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                    success, buffer = cv2.imencode(".jpg", frame_resized, encode_param)
                    
                    if not success:
                        logger.warning(f"Failed to encode frame {idx} from {video_path}")
                        continue
                    
                    frame_bytes_list.append(buffer.tobytes())
                    
                except Exception as e:
                    logger.warning(f"Failed to process frame {idx} from {video_path}: {e}")
                    continue

            if not frame_bytes_list:
                raise GeminiModelError(f"No frames could be extracted from video: {video_path}")

            logger.info(f"Encoded {len(frame_bytes_list)} frames from video: {video_path}")
            return frame_bytes_list
            
        except Exception as e:
            logger.error(f"Failed to encode video {video_path}: {e}")
            raise GeminiModelError(f"Failed to encode video: {e}") from e

    def _calculate_sample_indices(self, vr: VideoReader, total_frames: int) -> List[int]:
        """Calculate which frames to sample from the video."""
        sample_indices = []
        
        if self.fps is not None:
            # Sample at specified FPS
            video_fps = vr.get_avg_fps()
            if video_fps <= 0:
                raise GeminiModelError("Cannot determine video FPS")
                
            frame_interval = max(1, int(video_fps / self.fps))
            sample_indices = list(range(0, total_frames, frame_interval))
            
        elif self.nframes is not None:
            # Sample fixed number of frames
            if self.nframes <= 0:
                raise ValueError("nframes must be a positive integer")
                
            if self.nframes >= total_frames:
                sample_indices = list(range(total_frames))
            else:
                # Evenly distribute frames across video duration
                indices = [int(i * (total_frames - 1) / (self.nframes - 1)) for i in range(self.nframes)]
                sample_indices = sorted(list(set(indices)))
                
        else:
            # Default: sample at 1 FPS
            logger.warning("No fps or nframes specified, defaulting to 1 FPS sampling")
            video_fps = vr.get_avg_fps()
            frame_interval = max(1, int(video_fps)) if video_fps > 0 else 30
            sample_indices = list(range(0, total_frames, frame_interval))

        # Ensure we have at least the first and last frame
        if sample_indices and sample_indices[0] != 0:
            sample_indices.insert(0, 0)
        if sample_indices and sample_indices[-1] != total_frames - 1:
            sample_indices.append(total_frames - 1)
            
        # Remove duplicates and sort
        sample_indices = sorted(list(set(sample_indices)))
        
        logger.debug(f"Sampling {len(sample_indices)} frames from {total_frames} total frames")
        return sample_indices

    def _process_content(self, content: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Process content by converting image and video references to Gemini format."""
        if isinstance(content, str):
            return [{"text": content}]

        if isinstance(content, dict):
            content = [content]

        if not isinstance(content, list):
            raise ValueError("content must be a str, dict, or list")

        # Process content items
        processed_parts = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError(f"Content item must be a dict, got {type(item)}")

            t = item.get("type")
            if t == "text" and "text" in item:
                processed_parts.append({"text": item["text"]})
            elif t == "image" and "image" in item:
                # Encode image
                image_bytes = self.encode_image(item["image"])
                processed_parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(image_bytes).decode("utf-8")
                    }
                })
            elif t == "video" and "video" in item:
                # Encode video frames
                video_frames = self.encode_video(item["video"])
                for frame_bytes in video_frames:
                    processed_parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(frame_bytes).decode("utf-8")
                        }
                    })
            else:
                raise ValueError(f"Unsupported media item: {item}")

        return processed_parts

    def _preprocess_prompt(self, prompt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess prompt by handling images and videos in parallel.
        
        Converts OpenAI-style messages (with role/content) to Gemini format (list of parts).
        """
        prompt_copy = copy.deepcopy(prompt)
        
        # Collect items with content to process in parallel
        content_items = [(i, item) for i, item in enumerate(prompt_copy) if "content" in item]
        
        if not content_items:
            # Convert to Gemini format if no content processing needed
            parts = []
            for item in prompt_copy:
                if isinstance(item, dict) and "text" in item:
                    parts.append({"text": item["text"]})
                elif isinstance(item, str):
                    parts.append({"text": item})
            return parts if parts else [{"text": ""}]
        
        # Process content items in parallel using ThreadPoolExecutor
        max_workers = min(len(content_items), (os.cpu_count() or 1) + 4)
        processed_contents = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_content, item["content"]): i 
                for i, item in content_items
            }
            
            # Wait for all content processing to complete
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    processed_contents[i] = future.result()
                except Exception as e:
                    logger.error(f"Failed to process content item: {e}")
                    processed_contents[i] = [{"text": ""}]
        
        # Build Gemini format parts
        parts = []
        for i, item in enumerate(prompt_copy):
            if i in processed_contents:
                parts.extend(processed_contents[i])
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append({"text": item["text"]})
            elif isinstance(item, str):
                parts.append({"text": item})
        
        return parts if parts else [{"text": ""}]

    def _normalize_prompt(self, prompt: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Coerce string prompts into a single user message."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    @dynamic_retry_decorator
    def generate(self, prompt: Union[str, List[Dict[str, Any]]], text_format: Optional[type] = None, **kwargs) -> Any:
        """
        Generate completion for a single prompt.
        
        Args:
            prompt: Conversation prompt (list of role/content dicts or raw string)
            text_format (optional): Pydantic model or other structure for parsing the response
            **kwargs: Additional arguments passed to Gemini API

        Returns:
            Generated response string
        """
        prompt_copy = copy.deepcopy(self._normalize_prompt(prompt))
        
        # Convert to Gemini format
        # Gemini uses a single list of parts (text + images)
        parts = []
        for item in prompt_copy:
            if item.get("role") == "system":
                # System messages are typically prepended to user messages
                if "text" in item.get("content", {}):
                    parts.insert(0, {"text": item["content"]["text"]})
            elif item.get("role") == "user":
                processed = self._process_content(item.get("content", ""))
                parts.extend(processed)
            elif item.get("role") == "assistant":
                # For multi-turn conversations, we might need to handle assistant messages
                # For now, we'll skip them in the initial request
                pass
        
        try:
            # Generate content
            generation_config = {**self.kwargs, **kwargs}
            response = self.model.generate_content(
                parts,
                generation_config=genai.types.GenerationConfig(**generation_config) if generation_config else None
            )
            
            # Extract text from response
            if text_format is not None:
                # Try to parse response as structured format
                import json
                try:
                    response_dict = json.loads(response.text)
                    if isinstance(response_dict, dict):
                        return text_format(**response_dict)
                    return response_dict
                except json.JSONDecodeError:
                    logger.warning("Failed to parse response as JSON, returning raw text")
                    return response.text.strip()
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise GeminiModelError(f"Failed to get completion: {e}") from e

    def generate_batch(self, batch_prompts: List[Union[str, List[Dict[str, Any]]]], text_format: Optional[type] = None) -> List[Any]:
        """
        Process multiple prompts in batch with async generation and preprocessing.
        
        Args:
            batch_prompts: List of conversation prompts
            text_format (optional): Pydantic model or other structure for parsing the response

        Returns:
            List of response strings
        """
        # Normalize prompts first
        batch_prompts_copy = [copy.deepcopy(self._normalize_prompt(prompt)) for prompt in batch_prompts]

        if not batch_prompts_copy:
            return []
        
        # Process prompts in parallel using ThreadPoolExecutor
        max_workers = min(len(batch_prompts_copy), (os.cpu_count() or 1) + 4)
        processed_prompts = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_prompts = list(executor.map(self._preprocess_prompt, batch_prompts_copy))
        
        # Use async generation for batch processing
        try:
            return asyncio.run(self.async_generation(processed_prompts, text_format=text_format))
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise GeminiModelError(f"Batch generation failed: {e}") from e

    @dynamic_retry_decorator
    async def _generate_single_prompt(self, prompt: List[Dict[str, Any]], text_format: Optional[type] = None) -> Any:
        """Generate completion for a single prompt asynchronously."""
        # Note: Gemini Python SDK doesn't have native async support,
        # so we'll run it in a thread pool
        loop = asyncio.get_event_loop()
        
        def _sync_generate():
            parts = self._preprocess_prompt(prompt)
            generation_config = self.kwargs
            response = self.model.generate_content(
                parts,
                generation_config=genai.types.GenerationConfig(**generation_config) if generation_config else None
            )
            
            if text_format is not None:
                import json
                try:
                    response_dict = json.loads(response.text)
                    if isinstance(response_dict, dict):
                        return text_format(**response_dict)
                    return response_dict
                except json.JSONDecodeError:
                    logger.warning("Failed to parse response as JSON, returning raw text")
                    return response.text.strip()
            
            return response.text.strip()
        
        return await loop.run_in_executor(None, _sync_generate)

    async def async_generation(self, batch_prompts: List[List[Dict[str, Any]]], chunk_size: int = 50, text_format: Optional[type] = None) -> List[Any]:
        """
        Generate completions for multiple prompts asynchronously with chunking.
        
        Args:
            batch_prompts: List of conversation prompts (already preprocessed)
            chunk_size: Number of concurrent requests per chunk
            text_format (optional): Pydantic model or other structure for parsing the response

        Returns:
            List of response strings
        """
        responses = []
        total_chunks = (len(batch_prompts) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(batch_prompts), chunk_size):
            chunk_num = (i // chunk_size) + 1
            batch = batch_prompts[i:i + chunk_size]
            
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(batch)} prompts)")
            
            tasks = [self._generate_single_prompt(prompt, text_format) for prompt in batch]
            try:
                batch_responses = await tqdm_asyncio.gather(*tasks, desc=f"Chunk {chunk_num}")
                responses.extend(batch_responses)
            except Exception as e:
                logger.error(f"Error in chunk {chunk_num}: {e}")
                raise

        return responses

    def __repr__(self) -> str:
        """String representation of the model instance."""
        return (f"GeminiModel(model_name='{self.model_name}', "
                f"kwargs={self.kwargs})")


# Convenience function for backward compatibility
def create_gemini_model(model_name: str, **kwargs) -> GeminiModel:
    """Create a Gemini model instance with convenient defaults."""
    return GeminiModel(model_name=model_name, **kwargs)

