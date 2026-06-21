import os
import sys
import contextlib
import warnings

# MUST set environment variables BEFORE importing torch or any CUDA libraries
os.environ['CUDNN_LOGINFO_DBG'] = '0'
os.environ['CUDNN_LOGDEST_DBG'] = '/dev/null'
# Suppress cuDNN warnings at the system level
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

# Filter stderr BEFORE importing torch to catch early warnings
class FilteredStderr:
    """Filter stderr to suppress cuDNN-related warnings while preserving other errors"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.cudnn_keywords = [
            'cudnnGetLibConfig', 
            'undefined symbol', 
            'libcudnn_graph', 
            'Could not load symbol',
            'Error: /lib/x86_64-linux-gnu/libcudnn'
        ]
        self.buffer = ''  # Buffer for multi-line messages
    
    def write(self, message):
        # Handle both string and bytes
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='ignore')
        
        # Add to buffer and check line by line
        self.buffer += message
        lines = self.buffer.split('\n')
        # Keep the last incomafanmplete line in buffer
        self.buffer = lines[-1]
        
        # Process complete lines
        for line in lines[:-1]:
            line_lower = line.lower()
            # Only suppress lines containing cuDNN-related keywords
            if any(keyword.lower() in line_lower for keyword in self.cudnn_keywords):
                continue
            self.original_stderr.write(line + '\n')
    
    def flush(self):
        # Flush any remaining buffer
        if self.buffer:
            line_lower = self.buffer.lower()
            if not any(keyword.lower() in line_lower for keyword in self.cudnn_keywords):
                self.original_stderr.write(self.buffer)
            self.buffer = ''
        self.original_stderr.flush()

# Replace stderr BEFORE any imports that might trigger cuDNN warnings
_original_stderr = sys.stderr
sys.stderr = FilteredStderr(_original_stderr)

# Now import torch and other libraries (warnings will be filtered)
import torch
import pickle
from tqdm import tqdm
# Lazy imagebind import: 在某些 torch/torchaudio 二进制不兼容的环境里 (e.g. retrieval-only
# evaluator 不需要 video encoder), 顶层 import 会因为 libtorchaudio.so undefined symbol
# 直接挂掉. encode_video_segments / encode_string_query 只在建图时被调用, 不影响 retrieval.
try:
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ImageBindModel, ModalityType
    _IMAGEBIND_OK = True
except Exception as _e:
    import logging as _lg
    _lg.getLogger(__name__).warning(
        f"imagebind import failed ({_e}); video-segment encoding will be unavailable, "
        "but retrieval pipeline will still work."
    )
    data = None
    imagebind_model = None
    ImageBindModel = type("ImageBindModel", (), {})
    class _ModalityTypeStub:
        VISION = "vision"; TEXT = "text"
    ModalityType = _ModalityTypeStub()
    _IMAGEBIND_OK = False

# Filter out cuDNN-related warnings
warnings.filterwarnings('ignore', message='.*cudnnGetLibConfig.*')
warnings.filterwarnings('ignore', message='.*undefined symbol.*')
warnings.filterwarnings('ignore', message='.*Could not load symbol.*')

@contextlib.contextmanager
def suppress_cudnn_warnings():
    """Context manager to suppress cuDNN warnings while preserving other stderr output"""
    # stderr is already filtered globally, but we can add extra suppression here if needed
    yield


def encode_video_segments(video_paths, embedder: ImageBindModel):
    device = next(embedder.parameters()).device
    inputs = {
        ModalityType.VISION: data.load_and_transform_video_data(video_paths, device, clip_duration=4, clips_per_video=1,),
    }
    # Suppress cuDNN warnings during inference
    with torch.no_grad(), warnings.catch_warnings(), suppress_cudnn_warnings():
        warnings.simplefilter("ignore")
        # Set cudnn to deterministic mode to avoid some warnings
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        embeddings = embedder(inputs)[ModalityType.VISION]
    embeddings = embeddings.cpu()
    return embeddings

def encode_string_query(query:str, embedder: ImageBindModel):
    device = next(embedder.parameters()).device
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text([query], device),
    }
    # Suppress cuDNN warnings during inference
    with torch.no_grad(), warnings.catch_warnings(), suppress_cudnn_warnings():
        warnings.simplefilter("ignore")
        # Set cudnn to deterministic mode to avoid some warnings
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        embeddings = embedder(inputs)[ModalityType.TEXT]
    embeddings = embeddings.cpu()
    return embeddings