import os
import sys
import warnings

# Global cuDNN warning suppression - set BEFORE any torch imports
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

# Global stderr filter for cuDNN warnings
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
        # Keep the last incomplete line in buffer
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

# Apply global stderr filter if not already set
if not isinstance(sys.stderr, FilteredStderr):
    _original_stderr = sys.stderr
    sys.stderr = FilteredStderr(_original_stderr)

# Filter warnings
warnings.filterwarnings('ignore', message='.*cudnnGetLibConfig.*')
warnings.filterwarnings('ignore', message='.*undefined symbol.*')
warnings.filterwarnings('ignore', message='.*Could not load symbol.*')

from .videorag import VideoRAG, QueryParam