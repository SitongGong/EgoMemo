import asyncio
import os
import sys
import warnings
import base64
import io
from PIL import Image
import torch
import threading
from torchvision import transforms
from torchvision.transforms import InterpolationMode
# Lazy imagebind import: see videorag/_videoutil/feature.py for context.
# Retrieval-only paths don't need imagebind / pytorchvideo at all.
try:
    from imagebind import data
    from imagebind.models.imagebind_model import ImageBindModel, ModalityType
    from pytorchvideo import transforms as pv_transforms
    from imagebind.data import NormalizeVideo, SpatialCrop
    _IMAGEBIND_OK = True
except Exception as _e:
    import logging as _lg
    _lg.getLogger(__name__).warning(
        f"imagebind import failed in vdb_nanovectordb ({_e}); video-segment storage will be unavailable, "
        "retrieval pipeline OK."
    )
    data = None
    ImageBindModel = type("ImageBindModel", (), {})
    class _ModalityTypeStub:
        VISION = "vision"; TEXT = "text"
    ModalityType = _ModalityTypeStub()
    pv_transforms = None
    NormalizeVideo = None
    SpatialCrop = None
    _IMAGEBIND_OK = False

# Suppress cuDNN warnings BEFORE importing torch
os.environ['CUDNN_LOGINFO_DBG'] = '0'
os.environ['CUDNN_LOGDEST_DBG'] = '/dev/null'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fix cuDNN library loading issue by prioritizing nvidia-cudnn-cu12 package libraries
# This prevents loading incompatible system-level cuDNN libraries
# Use ctypes to set library path before any CUDA libraries are loaded
try:
    import ctypes
    import ctypes.util
    
    # Try to find nvidia-cudnn-cu12 library path
    cudnn_lib_path = None
    try:
        import nvidia.cudnn.lib as cudnn_lib
        if hasattr(cudnn_lib, '__path__'):
            for path in cudnn_lib.__path__:
                if os.path.exists(path) and os.path.isdir(path):
                    cudnn_lib_path = path
                    break
    except (ImportError, AttributeError):
        pass
    
    if not cudnn_lib_path:
        # Fallback: try to find manually
        import site
        for site_pkg in site.getsitepackages():
            potential_path = os.path.join(site_pkg, 'nvidia', 'cudnn', 'lib')
            if os.path.exists(potential_path):
                cudnn_lib_path = potential_path
                break
    
    if cudnn_lib_path:
        # Set LD_LIBRARY_PATH environment variable with highest priority
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_paths = [cudnn_lib_path]
        
        # Also add cublas path if available
        cublas_lib_path = None
        try:
            import nvidia.cublas.lib as cublas_lib
            if hasattr(cublas_lib, '__path__'):
                for path in cublas_lib.__path__:
                    if os.path.exists(path) and os.path.isdir(path):
                        cublas_lib_path = path
                        new_paths.append(path)
                        break
        except (ImportError, AttributeError):
            # Try to find cublas manually
            import site
            for site_pkg in site.getsitepackages():
                potential_cublas_path = os.path.join(site_pkg, 'nvidia', 'cublas', 'lib')
                if os.path.exists(potential_cublas_path):
                    cublas_lib_path = potential_cublas_path
                    new_paths.append(potential_cublas_path)
                    break
        
        # Add current LD_LIBRARY_PATH after our paths (lower priority)
        if current_ld_path:
            new_paths.append(current_ld_path)
        
        os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths)
        
        # Preload cuDNN libraries using ctypes to ensure they're loaded before system libraries
        try:
            libdl = ctypes.CDLL('libdl.so.2')
            libdl.dlopen.restype = ctypes.c_void_p
            libdl.dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]
            
            # RTLD_GLOBAL flag to make symbols available to other libraries
            RTLD_GLOBAL = 0x00100
            RTLD_NOW = 0x00002
            
            # Preload libcudnn_graph.so.9 from Python package to override system version
            cudnn_graph_path = os.path.join(cudnn_lib_path, 'libcudnn_graph.so.9')
            if os.path.exists(cudnn_graph_path):
                libdl.dlopen(cudnn_graph_path.encode('utf-8'), RTLD_GLOBAL | RTLD_NOW)
            
            # Also preload main cuDNN library
            cudnn_main_path = os.path.join(cudnn_lib_path, 'libcudnn.so.9')
            if os.path.exists(cudnn_main_path):
                libdl.dlopen(cudnn_main_path.encode('utf-8'), RTLD_GLOBAL | RTLD_NOW)
        except (OSError, AttributeError, Exception):
            # If preloading fails, LD_LIBRARY_PATH should still work
            pass
except Exception:
    # If anything fails, just continue - the error filtering will handle warnings
    pass

# Filter stderr to suppress cuDNN warnings
class FilteredStderr:
    """Filter stderr to suppress cuDNN-related warnings while preserving other errors"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.cudnn_keywords = ['cudnnGetLibConfig', 'undefined symbol', 'libcudnn_graph', 'Could not load symbol']
    
    def write(self, message):
        if any(keyword.lower() in message.lower() for keyword in self.cudnn_keywords):
            return
        self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()

# Set stderr filter before importing torch
_original_stderr = sys.stderr
sys.stderr = FilteredStderr(_original_stderr)

import torch
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm import tqdm
try:
    from imagebind.models import imagebind_model
except Exception:
    imagebind_model = None

# Filter warnings
warnings.filterwarnings('ignore', message='.*cudnnGetLibConfig.*')
warnings.filterwarnings('ignore', message='.*undefined symbol.*')
warnings.filterwarnings('ignore', message='.*Could not load symbol.*')

from .._utils import logger
from ..base import BaseVectorStorage
from .._videoutil import encode_video_segments, encode_string_query


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2
    
    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["llm"]["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NanoVectorDBVideoSegmentStorage(BaseVectorStorage):
    embedding_func = None
    segment_retrieval_top_k: float = 2
    
    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["video_embedding_batch_num"]
        self._client = NanoVectorDB(
            self.global_config["video_embedding_dim"], storage_file=self._client_file_name
        )
        self.top_k = self.global_config.get(
            "segment_retrieval_top_k", self.segment_retrieval_top_k
        )
        # 缓存 embedder 模型，避免每次调用都重新加载
        self._embedder = None
        self._embedder_lock = threading.Lock()
    
    def _get_embedder(self):
        """
        获取或初始化 embedder 模型（懒加载，线程安全）。
        模型只加载一次，后续调用复用缓存的模型。
        
        Returns:
            ImageBindModel: 初始化的 embedder 模型
        """
        if self._embedder is None:
            with self._embedder_lock:
                # 双重检查，避免多线程环境下重复加载
                if self._embedder is None:
                    # Suppress cuDNN warnings during model loading
                    # 注意：CUDA_VISIBLE_DEVICES 已在脚本入口设置，物理卡已映射为 cuda:0
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self._embedder = imagebind_model.imagebind_huge(pretrained=True).to(device)
                    self._embedder.eval()
        return self._embedder
    
    def _load_and_transform_base64_images(self, base64_images, device):
        """
        Load and transform base64-encoded images for ImageBind.
        
        Args:
            base64_images: List of base64-encoded image strings
            device: torch device
            
        Returns:
            torch.Tensor: Stacked and transformed images
        """
        image_outputs = []
        
        data_transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        
        for base64_str in base64_images:
            # Decode base64 string to bytes
            image_bytes = base64.b64decode(base64_str)
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Apply transforms and move to device
            image = data_transform(image).to(device)
            image_outputs.append(image)
        
        return torch.stack(image_outputs, dim=0)
    
    def _load_and_transform_base64_images_as_video(self, base64_images, device):
        """
        Load and transform base64-encoded images as video using imagebind's video preprocessing.
        This mimics the video processing pipeline used in load_and_transform_video_data.
        Note: The frames are already sampled, so no temporal subsampling is applied.
        
        Args:
            base64_images: List of base64-encoded image strings (already sampled frames)
            device: torch device
            
        Returns:
            torch.Tensor: Video tensor with shape [1, num_spatial_crops, C, T, H, W]
                compatible with imagebind's video encoding format
        """
        # Step 1: Load images and convert to tensor format
        # Match the exact format from load_and_transform_video_data
        # clip["video"] from EncodedVideo is in (T, C, H, W) format
        images = []
        for base64_str in base64_images:
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append(image)
        
        # Convert PIL images to tensors
        # Each image will be (C, H, W), we stack them to get (T, C, H, W)
        to_tensor = transforms.ToTensor()
        image_tensors = [to_tensor(img) for img in images]
        video_tensor = torch.stack(image_tensors, dim=0)  # (T, C, H, W)
        
        # Convert to (C, T, H, W) format (required by pytorchvideo transforms)
        # This matches clip["video"] format after potential conversion
        # video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        # Step 2: Normalize to [0, 1] range (matching clip["video"] / 255.0)
        # Note: ToTensor already gives [0, 1], but we ensure consistency
        # In load_and_transform_video_data: video_clip = video_clip / 255.0
        # Our ToTensor already gives [0, 1], so this is already done
        
        # Step 3: Apply video transforms (same as load_and_transform_video_data)
        # all_video = [video_transform(clip) for clip in all_video]
        video_transform = transforms.Compose([
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        
        # Apply transforms to get (C, T, H, W)
        video_tensor = [video_transform(clip.unsqueeze(1)) for clip in video_tensor]  # list:[(C, T, H, W)]
        
        # Step 4: Apply spatial cropping (same as SpatialCrop in load_and_transform_video_data)
        # all_video = SpatialCrop(224, num_crops=3)(all_video)
        # SpatialCrop expects a list of (C, T, H, W) videos and returns a list
        # For a single clip, we create a list with one element
        spatial_crop = SpatialCrop(224, num_crops=3)
        video_clips = spatial_crop(video_tensor)  # Returns list of (C, T, 224, 224)
        
        # Step 5: Stack spatial crops (same as load_and_transform_video_data)
        # all_video = torch.stack(all_video, dim=0)
        # For a single clip with 3 spatial crops, this gives (3, C, T, H, W)
        video_tensor = torch.stack(video_clips, dim=0)  # (num_spatial_crops, C, T, H, W)
        
        # Step 6: Add batch dimension to match load_and_transform_video_data output format
        # video_outputs.append(all_video) then torch.stack(video_outputs, dim=0)
        # For a single video segment, this gives [1, num_spatial_crops, C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0).to(device)    # 1, num_crops, 3, 1, H, W
        
        return video_tensor
    
    ### 新增的将视频帧转换为embedding并储存的函数
    async def upsert_video_segment(self, time_span, video_frames, encode_mode="joint"):
        """
        Encode video frames and store embeddings.
        
        Args:
            time_span: Time span identifier for the video segment
            video_frames: List of base64-encoded image strings
            encode_mode: Encoding mode, either "frame" or "joint"
                - "frame": Encode each frame separately (default)
                - "joint": Encode all frames together as a single video segment embedding
        """
        # 使用缓存的 embedder，避免每次调用都重新加载模型
        embedder = self._get_embedder()
        device = next(embedder.parameters()).device
        
        if encode_mode == "joint":
            # Joint encoding: encode all frames together as a video using imagebind's video preprocessing
            # This mimics the video processing pipeline used in load_and_transform_video_data
            # Note: video_frames are already sampled, so no temporal subsampling is applied
            video_tensor = self._load_and_transform_base64_images_as_video(
                video_frames, device
            )
            
            # Encode video using imagebind (same as encode_video_segments)
            # Note: video_tensor shape is [1, num_spatial_crops, C, T, H, W]
            # If different video segments have different frame counts (T dimension),
            # we need to process each spatial crop separately to avoid dimension mismatch
            with torch.no_grad():
                inputs = {
                    ModalityType.VISION: video_tensor,
                }
                video_embeddings = embedder(inputs)[ModalityType.VISION]  # [1, 1, embedding_dim]
                # crop_embeddings.append(crop_embedding.squeeze(1))  # [1, embedding_dim]
                
                # Stack all crop embeddings: [1, num_spatial_crops, embedding_dim]
                # video_embeddings = torch.stack(crop_embeddings, dim=1)
            
            # Flatten all spatial crops into a single embedding vector
            # video_embeddings shape: [1, num_spatial_crops, embedding_dim]
            # Flatten to [num_spatial_crops * embedding_dim] for storage
            segment_embedding = video_embeddings.cpu().numpy()  # [num_spatial_crops * embedding_dim]
            
            # Prepare list_data for upsert - single embedding for the entire segment
            list_data = [{
                "__id__": f"{time_span}_0",
                "__time_span__": time_span,        # time_span所起的所用相当于原来的video_name
                "__vector__": segment_embedding[0],
                "__index__": 0,
                # "__frames__": video_frames, 
            }]
            
        else:  # encode_mode == "frame" (default)
            # Frame-by-frame encoding: encode each frame separately
            batches = [
                video_frames[i: i + self._max_batch_size] 
                for i in range(0, len(video_frames), self._max_batch_size)
            ]
            embeddings = []
            for batch_base64_images in tqdm(batches, desc=f"Encoding Video Segments {time_span}"):
                # Use custom function to process base64-encoded images  
                inputs = {
                    ModalityType.VISION: self._load_and_transform_base64_images(batch_base64_images, device),
                }
                # Suppress cuDNN warnings during inference
                with torch.no_grad():
                    embedding = embedder(inputs)[ModalityType.VISION]
                embeddings.append(embedding.cpu())
            embeddings = torch.concat(embeddings, dim=0).numpy()     
            
            # Prepare list_data for upsert
            list_data = []
            for i, _ in enumerate(video_frames):
                list_data.append({
                    "__id__": f"{time_span}_{i}",
                    "__time_span__": time_span, 
                    "__vector__": embeddings[i],
                    "__index__": i, 
                })
        
        results = self._client.upsert(datas=list_data)
        return results
    
    async def upsert_video_segment_batch(self, segments_data: list[tuple[str, list]], encode_mode="joint"):
        """
        批量编码多个视频段并存储 embeddings，提高处理效率。
        
        Args:
            segments_data: List of tuples (time_span, video_frames)
                - time_span: Time span identifier for the video segment
                - video_frames: List of base64-encoded image strings
            encode_mode: Encoding mode, either "frame" or "joint"
                - "frame": Encode each frame separately
                - "joint": Encode all frames together as a single video segment embedding
        
        Returns:
            List of results from upsert operations
        """
        if not segments_data:
            logger.warning("No segments to process")
            return []
        
        # 使用缓存的 embedder，避免每次调用都重新加载模型
        embedder = self._get_embedder()
        device = next(embedder.parameters()).device
        
        logger.info(f"[Batch Video Encoding] Processing {len(segments_data)} segments in batch...")
        
        if encode_mode == "joint":
            # Joint encoding: 批量处理所有 segments
            all_list_data = []
            
            # 批量处理所有 segments
            for time_span, video_frames in tqdm(segments_data, desc="Encoding Video Segments (Batch)"):
                # 为每个 segment 准备视频 tensor
                video_tensor = self._load_and_transform_base64_images_as_video(
                    video_frames, device
                )
                
                # 编码视频
                with torch.no_grad():
                    inputs = {
                        ModalityType.VISION: video_tensor,
                    }
                    video_embeddings = embedder(inputs)[ModalityType.VISION]  # [1, 1, embedding_dim]
                
                # 转换为 numpy
                segment_embedding = video_embeddings.cpu().numpy()  # [1, embedding_dim]
                
                # 准备数据
                all_list_data.append({
                    "__id__": f"{time_span}_0",
                    "__time_span__": time_span,
                    "__vector__": segment_embedding[0],
                    "__index__": 0,
                })
            
            # 批量 upsert 所有数据
            results = self._client.upsert(datas=all_list_data)
            logger.info(f"[Batch Video Encoding] Completed processing {len(segments_data)} segments")
            return results
            
        else:  # encode_mode == "frame"
            # Frame-by-frame encoding: 批量处理所有 frames
            all_list_data = []
            
            # 收集所有 frames 和对应的 time_span/index 信息
            all_frames = []
            frame_metadata = []  # (time_span, frame_index, segment_index)
            
            for seg_idx, (time_span, video_frames) in enumerate(segments_data):
                for frame_idx, _ in enumerate(video_frames):
                    all_frames.append((time_span, frame_idx, seg_idx))
            
            # 批量编码所有 frames
            batches = [
                [(segments_data[seg_idx][1][frame_idx] if seg_idx < len(segments_data) and frame_idx < len(segments_data[seg_idx][1]) else None) 
                 for time_span, frame_idx, seg_idx in all_frames[i: i + self._max_batch_size]]
                for i in range(0, len(all_frames), self._max_batch_size)
            ]
            
            # 重新组织：按 segment 分组 frames
            for seg_idx, (time_span, video_frames) in enumerate(tqdm(segments_data, desc="Encoding Video Segments (Batch)")):
                # 批量编码当前 segment 的 frames
                frame_batches = [
                    video_frames[i: i + self._max_batch_size] 
                    for i in range(0, len(video_frames), self._max_batch_size)
                ]
                embeddings = []
                for batch_base64_images in frame_batches:
                    inputs = {
                        ModalityType.VISION: self._load_and_transform_base64_images(batch_base64_images, device),
                    }
                    with torch.no_grad():
                        embedding = embedder(inputs)[ModalityType.VISION]
                    embeddings.append(embedding.cpu())
                embeddings = torch.concat(embeddings, dim=0).numpy()
                
                # 准备数据
                for i, _ in enumerate(video_frames):
                    all_list_data.append({
                        "__id__": f"{time_span}_{i}",
                        "__time_span__": time_span,
                        "__vector__": embeddings[i],
                        "__index__": i,
                    })
            
            # 批量 upsert 所有数据
            results = self._client.upsert(datas=all_list_data)
            logger.info(f"[Batch Video Encoding] Completed processing {len(segments_data)} segments")
            return results
        
    
    async def upsert(self, video_name, segment_index2name, video_output_format):
        # 使用缓存的 embedder，避免每次调用都重新加载模型
        embedder = self._get_embedder()
        
        logger.info(f"Inserting {len(segment_index2name)} segments to {self.namespace}")
        if not len(segment_index2name):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data, video_paths = [], []
        cache_path = os.path.join(self.global_config["working_dir"], '_cache', video_name)
        index_list = list(segment_index2name.keys())
        for index in index_list:
            list_data.append({
                "__id__": f"{video_name}_{index}",
                "__video_name__": video_name,
                "__index__": index,
            })
            segment_name = segment_index2name[index]
            video_file = os.path.join(cache_path, f"{segment_name}.{video_output_format}")
            video_paths.append(video_file)
        batches = [
            video_paths[i: i + self._max_batch_size]
            for i in range(0, len(video_paths), self._max_batch_size)
        ]
        embeddings = []
        for _batch in tqdm(batches, desc=f"Encoding Video Segments {video_name}"):
            batch_embeddings = encode_video_segments(_batch, embedder)
            embeddings.append(batch_embeddings)
        embeddings = torch.concat(embeddings, dim=0)
        embeddings = embeddings.numpy()
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results
    
    async def query(self, query: str):
        # 使用缓存的 embedder，避免每次调用都重新加载模型
        embedder = self._get_embedder()
        
        embedding = encode_string_query(query, embedder)
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=self.top_k,
            better_than_threshold=-1,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results
    
    async def index_done_callback(self):
        self._client.save()
