import numpy as np

from openai import (
    AsyncOpenAI, 
    AsyncAzureOpenAI, 
    APIConnectionError, 
    RateLimitError,
    APITimeoutError,
    APIError,
    InternalServerError
)
from ollama import AsyncClient
from dataclasses import asdict, dataclass, field
import asyncio
import time
from threading import Lock

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    wait_random,
)
import os
import logging

logger = logging.getLogger(__name__)

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc

global_openai_async_client = None
global_azure_openai_async_client = None
global_ollama_client = None
global_gemini_client = None

# 全局API调用限流机制 - 防止多个进程同时调用导致限流
# 使用信号量控制并发API调用数量
_global_api_semaphore = None
_global_api_semaphore_lock = Lock()
_global_api_semaphore_max_workers = int(os.environ.get("VIDEORAG_API_MAX_WORKERS", "10"))

def get_global_api_semaphore():
    """获取全局API调用信号量，用于限制并发API调用"""
    global _global_api_semaphore, _global_api_semaphore_max_workers
    if _global_api_semaphore is None:
        with _global_api_semaphore_lock:
            if _global_api_semaphore is None:
                _global_api_semaphore = asyncio.Semaphore(_global_api_semaphore_max_workers)
                logger.info(f"初始化全局API调用信号量，最大并发数: {_global_api_semaphore_max_workers}")
    return _global_api_semaphore

# 全局API调用速率限制 - 使用令牌桶算法（异步锁保护）
_global_api_rate_limiter = None
_global_api_rate_limiter_lock = None
_global_api_min_interval = float(os.environ.get("VIDEORAG_API_MIN_INTERVAL", "0.1"))  # 最小调用间隔（秒）
_last_api_call_time = 0

def get_api_rate_limit_lock():
    """获取异步锁用于速率限制"""
    global _global_api_rate_limiter_lock
    if _global_api_rate_limiter_lock is None:
        _global_api_rate_limiter_lock = asyncio.Lock()
    return _global_api_rate_limiter_lock

async def wait_for_api_rate_limit():
    """等待API调用速率限制（线程安全版本）"""
    global _last_api_call_time, _global_api_min_interval
    lock = get_api_rate_limit_lock()
    async with lock:
        current_time = time.time()
        time_since_last_call = current_time - _last_api_call_time
        if time_since_last_call < _global_api_min_interval:
            wait_time = _global_api_min_interval - time_since_last_call
            await asyncio.sleep(wait_time)
        _last_api_call_time = time.time()

def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        # 配置超时和重试参数
        import httpx
        timeout = httpx.Timeout(
            timeout=180.0,  # 总超时时间（增加到3分钟）
            connect=30.0,   # 连接超时（增加到30秒）
            read=150.0,     # 读取超时（增加到2.5分钟）
        )
        # 配置连接池限制
        limits = httpx.Limits(
            max_keepalive_connections=20,  # 最大保持连接数
            max_connections=50,             # 最大总连接数
            keepalive_expiry=30.0          # 连接保持时间
        )
        # 创建 httpx 客户端，支持代理（通过环境变量 HTTP_PROXY/HTTPS_PROXY）
        http_client = httpx.AsyncClient(limits=limits, timeout=timeout)

        global_openai_async_client = AsyncOpenAI(
            timeout=timeout,
            max_retries=3,  # OpenAI SDK内置重试（增加到3次）
            http_client=http_client
        )
        logger.info("初始化 OpenAI 客户端，超时配置: 总180s, 连接30s, 读取150s，支持环境变量代理")
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client

def get_ollama_async_client_instance():
    global global_ollama_client
    if global_ollama_client is None:
        # set OLLAMA_HOST or pass in host="http://127.0.0.1:11434"
        global_ollama_client = AsyncClient()  # Adjust base URL if necessary        
    return global_ollama_client

def get_gemini_client_instance():
    global global_gemini_client
    if global_gemini_client is None:
        try:
            from google import genai
            api_key = os.environ.get('GOOGLE_API_KEY')
            if api_key:
                global_gemini_client = genai.Client(api_key=api_key)
            else:
                global_gemini_client = None
        except ImportError:
            global_gemini_client = None
    return global_gemini_client

# Setup LLM Configuration.
@dataclass
class LLMConfig:
    # To be set
    embedding_func_raw: callable
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int    
    embedding_func_max_async: int 
    query_better_than_threshold: float
    
    best_model_func_raw: callable
    best_model_name: str    
    best_model_max_token_size: int
    best_model_max_async: int
    
    cheap_model_func_raw: callable
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int

    # Assigned in post init
    embedding_func: EmbeddingFunc  = None    
    best_model_func: callable = None    
    cheap_model_func: callable = None
    

    def __post_init__(self):
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim = self.embedding_dim,
            max_token_size = self.embedding_max_token_size,
            model_name = self.embedding_model_name)
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)
        self.best_model_func = lambda prompt, *args, **kwargs: self.best_model_func_raw(
            self.best_model_name, prompt, *args, **kwargs
        )

        self.cheap_model_func = lambda prompt, *args, **kwargs: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *args, **kwargs
        )

##### OpenAI Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10), #  + wait_random(0, 2),  # 增加等待时间和随机抖动
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # # 使用全局信号量限制并发API调用
    # semaphore = get_global_api_semaphore()
    
    # async with semaphore:
    #     # 等待API调用速率限制
    #     await wait_for_api_rate_limit()
        
        openai_async_client = get_openai_async_client_instance()
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        use_cache = kwargs.pop("use_cache", True)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # 先检查缓存（这是省钱的缓存调用方式）
        if hashing_kv is not None and use_cache:
            args_hash = compute_args_hash(model, messages)
            try:
                if_cache_return = await hashing_kv.get_by_id(args_hash)
                # NOTE: I update here to avoid the if_cache_return["return"] is None
                if if_cache_return is not None and if_cache_return.get("return") is not None:
                    logger.debug(f"使用缓存结果，hash: {args_hash[:16]}...")
                    return if_cache_return["return"]
            except Exception as e:
                logger.warning(f"读取缓存失败，继续调用API: {e}")

        # Handle models that require max_completion_tokens instead of max_tokens
        # Models like o1-preview, o1-mini, o3-mini, etc. require max_completion_tokens
        api_kwargs = kwargs.copy()
        if "max_tokens" in api_kwargs:
            # Check if model requires max_completion_tokens
            model_lower = model.lower()
            # if any(prefix in model_lower for prefix in ["o1-", "o3-"]):
                # Convert max_tokens to max_completion_tokens for o1/o3 models
            api_kwargs.pop("max_tokens")  # Remove max_tokens
            api_kwargs["max_completion_tokens"] = 2048  # Set to fixed value 2048
            # For other models, keep max_tokens as is
        # print(api_kwargs)
        # 调用API
        try:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **api_kwargs
            )
            result = response.choices[0].message.content
        except (RateLimitError, APIConnectionError) as e:
            logger.warning(f"API调用失败: {e}，将重试...")
            raise  # 让retry装饰器处理重试
        except Exception as e:
            logger.error(f"API调用出现未知错误: {e}")
            raise

        # 保存到缓存
        if hashing_kv is not None and use_cache:
            try:
                await hashing_kv.upsert(
                    {args_hash: {"return": result, "model": model}}
                )
                await hashing_kv.index_done_callback()
            except Exception as e:
                logger.warning(f"保存缓存失败: {e}，但不影响返回结果")
        
        return result
    
    # openai_async_client = get_openai_async_client_instance()
    # hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    # use_cache = kwargs.pop("use_cache", True)

    # messages = []
    # if system_prompt:
    #     messages.append({"role": "system", "content": system_prompt})
    # messages.extend(history_messages)
    # messages.append({"role": "user", "content": prompt})

    # if hashing_kv is not None and use_cache:
    #     args_hash = compute_args_hash(model, messages)
    #     if_cache_return = await hashing_kv.get_by_id(args_hash)
    #     # NOTE: I update here to avoid the if_cache_return["return"] is None
    #     if if_cache_return is not None and if_cache_return["return"] is not None:
    #         return if_cache_return["return"]

    # # 使用信号量控制并发，避免同时发起过多请求导致连接问题
    # semaphore = get_global_api_semaphore()
    # async with semaphore:
    #     response = await openai_async_client.chat.completions.create(
    #         model=model, messages=messages, **kwargs
    #     )

    # if hashing_kv is not None and use_cache:
    #     await hashing_kv.upsert(
    #         {args_hash: {"return": response.choices[0].message.content, "model": model}}
    #     )
    #     await hashing_kv.index_done_callback()
    # return response.choices[0].message.content


async def gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((
        RateLimitError, 
        APIConnectionError, 
        APITimeoutError,
        InternalServerError
    )),
)
async def openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    """
    OpenAI Embedding API 调用，带并发控制和速率限制
    
    Args:
        model_name: 模型名称（如 text-embedding-3-small）
        texts: 待编码的文本列表
    
    Returns:
        embeddings 的 numpy 数组
    """
    # 1. 获取信号量进行并发控制
    # semaphore = get_global_api_semaphore()
    
    # async with semaphore:
        # 2. 速率限制（已注释，提升性能）
        # await wait_for_api_rate_limit()
        
        # 3. 调用 API
    try:
        openai_async_client = get_openai_async_client_instance()
        response = await openai_async_client.embeddings.create(
            model=model_name, 
            input=texts, 
            encoding_format="float"
        )
        result = np.array([dp.embedding for dp in response.data])
        logger.debug(f"成功获取 {len(texts)} 个文本的 embeddings")
        return result
        
    except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
        # 可重试的错误
        logger.warning(f"OpenAI Embedding API 调用失败（可重试）: {type(e).__name__}: {e}")
        raise  # 让 retry 装饰器处理重试
        
    except APIError as e:
        # 其他 API 错误
        logger.error(f"OpenAI Embedding API 错误: {e.status_code} - {e.message}")
        raise
        
    except Exception as e:
        # 未知错误
        logger.error(f"OpenAI Embedding 未知错误: {type(e).__name__}: {e}")
        raise

openai_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

openai_4o_mini_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_mini_complete,
    best_model_name = "gpt-4o-mini",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

###### Azure OpenAI Configuration
@retry(
    stop=stop_after_attempt(3),  # 增加重试次数
    wait=wait_exponential(multiplier=1, min=4, max=10),  # + wait_random(0, 2),  # 增加等待时间和随机抖动
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # # 使用全局信号量限制并发API调用
    # semaphore = get_global_api_semaphore()
    
    # async with semaphore:
        # 等待API调用速率限制
        # await wait_for_api_rate_limit()
        
        azure_openai_client = get_azure_openai_async_client_instance()
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        use_cache = kwargs.pop("use_cache", True)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # 先检查缓存（这是省钱的缓存调用方式）
        if hashing_kv is not None and use_cache:
            args_hash = compute_args_hash(deployment_name, messages)
            try:
                if_cache_return = await hashing_kv.get_by_id(args_hash)
                # NOTE: I update here to avoid the if_cache_return["return"] is None
                if if_cache_return is not None and if_cache_return.get("return") is not None:
                    logger.debug(f"使用缓存结果，hash: {args_hash[:16]}...")
                    return if_cache_return["return"]
            except Exception as e:
                logger.warning(f"读取缓存失败，继续调用API: {e}")

        # 调用API
        try:
            response = await azure_openai_client.chat.completions.create(
                model=deployment_name, messages=messages, **kwargs
            )
            result = response.choices[0].message.content
        except (RateLimitError, APIConnectionError) as e:
            logger.warning(f"API调用失败: {e}，将重试...")
            raise  # 让retry装饰器处理重试
        except Exception as e:
            logger.error(f"API调用出现未知错误: {e}")
            raise

        # 保存到缓存
        if hashing_kv is not None and use_cache:
            try:
                await hashing_kv.upsert(
                    {
                        args_hash: {
                            "return": result,
                            "model": deployment_name,
                        }
                    }
                )
                await hashing_kv.index_done_callback()
            except Exception as e:
                logger.warning(f"保存缓存失败: {e}，但不影响返回结果")
        
        return result


async def azure_gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    调用Azure GPT-4o模型，带缓存机制（省钱）。
    
    注意：此函数会自动使用缓存，如果传入hashing_kv参数，会先检查缓存，
    如果缓存中有结果则直接返回，避免重复调用API，从而节省成本。
    
    如果多个进程同时运行，建议：
    1. 使用共享的hashing_kv缓存实例
    2. 设置环境变量VIDEORAG_API_MAX_WORKERS控制并发数
    3. 设置环境变量VIDEORAG_API_MIN_INTERVAL控制调用间隔
    """
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


azure_openai_config = LLMConfig(
    embedding_func_raw = azure_openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size = 8192,    
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    best_model_func_raw = azure_gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,

    cheap_model_func_raw  = azure_gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)


######  Ollama configuration

async def ollama_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Initialize the Ollama client
    ollama_client = get_ollama_async_client_instance()

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # Send the request to Ollama
    response = await ollama_client.chat(
        model=model,
        messages=messages
    )
    # print(messages)
    # print(response['message']['content'])

    
    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": response['message']['content'], "model": model}}
        )
        await hashing_kv.index_done_callback()

    return response['message']['content']


async def ollama_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages
    )

async def ollama_mini_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        # "deepseek-r1:latest",  # For now select your model
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def ollama_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    # Initialize the Ollama client
    ollama_client = get_ollama_async_client_instance()

    # Send the request to Ollama for embeddings
    response = await ollama_client.embed(
        model=model_name,  
        input=texts
    )

    # Extract embeddings from the response
    embeddings = response['embeddings']

    return np.array(embeddings)

ollama_config = LLMConfig(
    embedding_func_raw = ollama_embedding,
    embedding_model_name = "nomic-embed-text",
    embedding_dim = 768,
    embedding_max_token_size=8192,
    embedding_batch_num = 1,
    embedding_func_max_async = 1,
    query_better_than_threshold = 0.2,
    best_model_func_raw = ollama_complete ,
    best_model_name = "gemma2:latest", # need to be a solid instruct model
    best_model_max_token_size = 32768,
    best_model_max_async  = 1,
    cheap_model_func_raw = ollama_mini_complete,
    cheap_model_name = "olmo2",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 1
)
###### DeepSeek Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def deepseek_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # 使用DeepSeek API
    import httpx
    
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # DeepSeek API调用
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', 'sk-*******')}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 4096)
            },
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": content, "model": model}}
        )
        await hashing_kv.index_done_callback()

    return content

async def deepseek_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await deepseek_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def bge_m3_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    # 使用硅基流动的BAAI/bge-m3嵌入模型
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.siliconflow.cn/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY', 'sk-******')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "BAAI/bge-m3",
                "input": texts,
                "encoding_format": "float"
            },
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return np.array(embeddings)

# DeepSeek + BAAI/bge-m3 配置
deepseek_bge_config = LLMConfig(
    embedding_func_raw = bge_m3_embedding,
    embedding_model_name = "BAAI/bge-m3",
    embedding_dim = 1024,  # bge-m3的嵌入维度
    embedding_max_token_size = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,
    
    best_model_func_raw = deepseek_complete,
    best_model_name = "deepseek-chat",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
    
    cheap_model_func_raw = deepseek_complete,
    cheap_model_name = "deepseek-chat",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

### Gemini Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def gemini_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    client = get_gemini_client_instance()
    if client is None:
        raise ValueError("Gemini client not available. Please set GOOGLE_API_KEY environment variable.")
    
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    use_cache = kwargs.pop("use_cache", True)

    # Build messages in OpenAI-like format, then convert to Gemini format
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and use_cache:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # Convert OpenAI format to Gemini format
    # New SDK still uses contents format but with Content objects
    from google.genai import types
    
    contents = []
    system_instruction = None
    
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        else:
            # New SDK: use Content objects with role and parts
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            ))

    # Build generation config
    generation_config = types.GenerateContentConfig(
        temperature=kwargs.get("temperature", 0.7),
        max_output_tokens=kwargs.get("max_tokens", 8192),
    )
    
    # Generate content using new SDK (run in executor to make it async-compatible)
    import asyncio
    loop = asyncio.get_event_loop()
    
    # Generate content using new SDK (run in executor to make it async-compatible)
    response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generation_config,
            # system_instruction=system_instruction if system_instruction else None
        )
    content = response.text

    if hashing_kv is not None and use_cache:
        await hashing_kv.upsert(
            {args_hash: {"return": content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    
    return content


async def gemini_pro_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gemini_flash_complete(
    model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


def gemini_complete_with_image_sync(
    model, prompt, images=None, system_prompt=None, **kwargs
) -> str:
    """
    Synchronous Gemini completion function with image support.
    This is a wrapper for VideoGraph.py to use Gemini with images.
    
    Args:
        model: Model name (e.g., "gemini-1.5-pro")
        prompt: Text prompt
        images: List of PIL Image objects or base64-encoded image strings
        system_prompt: Optional system prompt
        **kwargs: Additional arguments
        
    Returns:
        Generated text response
    """
    client = get_gemini_client_instance()
    if client is None:
        raise ValueError("Gemini client not available. Please set GOOGLE_API_KEY environment variable.")
    
    import base64
    from PIL import Image
    import io
    from google.genai import types
    
    # Build parts for Gemini API - convert images to PIL Images
    processed_images = []
    if images:
        for img in images:
            if isinstance(img, str):
                # Base64 string
                try:
                    img_bytes = base64.b64decode(img)
                    img_obj = Image.open(io.BytesIO(img_bytes))
                    img_obj = img_obj.convert("RGB")
                    processed_images.append(img_obj)
                except Exception as e:
                    logger.warning(f"Failed to process base64 image: {e}")
                    continue
            elif isinstance(img, Image.Image):
                # PIL Image
                processed_images.append(img.convert("RGB"))
            else:
                logger.warning(f"Unsupported image type: {type(img)}")
                continue
    
    # Build generation config
    generation_config = types.GenerateContentConfig(
        temperature=kwargs.get("temperature", 0.7),
        max_output_tokens=kwargs.get("max_tokens", 8192),
    )
    
    # Use new Google GenAI SDK
    # Build Content object with text and images using Part objects
    try:
        parts = [types.Part.from_text(text=prompt)]
        # Add images as Part objects
        for img in processed_images:
            # Convert PIL Image to bytes
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_bytes = buffered.getvalue()
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=generation_config,
            system_instruction=system_prompt if system_prompt else None
        )
        return response.text
    except Exception as e:
        logger.warning(f"Gemini SDK call failed, falling back to REST API: {e}")
        # Fallback to REST API if SDK fails
        import httpx
        api_key = os.environ.get('GOOGLE_API_KEY')
        
        # Build parts in REST API format
        rest_parts = [{"text": prompt}]
        if images:
            for img in images:
                if isinstance(img, str):
                    try:
                        img_bytes = base64.b64decode(img)
                        img_obj = Image.open(io.BytesIO(img_bytes))
                        img_obj = img_obj.convert("RGB")
                        buffered = io.BytesIO()
                        img_obj.save(buffered, format="JPEG", quality=85)
                        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    except Exception as e:
                        logger.warning(f"Failed to process base64 image: {e}")
                        continue
                elif isinstance(img, Image.Image):
                    img = img.convert("RGB")
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                else:
                    continue
                
                rest_parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_base64
                    }
                })
        
        request_payload = {
            "contents": [{
                "role": "user",
                "parts": rest_parts
            }],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 8192),
            }
        }
        
        if system_prompt:
            request_payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }
        
        with httpx.Client(timeout=60.0) as http_client:
            response = http_client.post(
                f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}",
                json=request_payload
            )
            response.raise_for_status()
            result = response.json()
            
            if "candidates" not in result or len(result["candidates"]) == 0:
                raise ValueError(f"Gemini API returned no candidates: {result}")
            
            return result["candidates"][0]["content"]["parts"][0]["text"]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def gemini_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    client = get_gemini_client_instance()
    if client is None:
        raise ValueError("Gemini client not available. Please set GOOGLE_API_KEY environment variable.")
    
    # Remove 'models/' prefix if present (model_name might be "models/text-embedding-004")
    clean_model_name = model_name.replace("models/", "") if model_name.startswith("models/") else model_name
    
    from google.genai import types
    
    # Use new Google GenAI SDK (run in executor to make it async-compatible)
    import asyncio
    loop = asyncio.get_event_loop()
    
    embeddings = []
    try:
        for text in texts:
            response = client.models.embed_content(
                    model=clean_model_name,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
            embeddings.append(response.embeddings[0].values)
    except Exception as e:
        logger.warning(f"Gemini SDK embedding call failed, falling back to REST API: {e}")
        # Fallback to REST API if SDK fails
        import httpx
        api_key = os.environ.get('GOOGLE_API_KEY')
        
        async with httpx.AsyncClient() as http_client:
            for text in texts:
                response = await http_client.post(
                    f"https://generativelanguage.googleapis.com/v1/models/{clean_model_name}:embedContent?key={api_key}",
                    json={
                        "model": clean_model_name,
                        "content": {
                            "parts": [{"text": text}]
                        },
                        "taskType": "RETRIEVAL_DOCUMENT"
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"]["values"])
    
    return np.array(embeddings)


gemini_config = LLMConfig(
    embedding_func_raw=gemini_embedding,
    embedding_model_name="models/text-embedding-004",
    embedding_dim=768,
    embedding_max_token_size=2048,
    embedding_batch_num=32,
    embedding_func_max_async=16,
    query_better_than_threshold=0.2,

    best_model_func_raw=gemini_pro_complete,
    best_model_name="gemini-3.0-pro-preview",
    best_model_max_token_size=32768,
    best_model_max_async=16,

    cheap_model_func_raw=gemini_flash_complete,
    cheap_model_name="gemini-2.5-flash",
    cheap_model_max_token_size=32768,
    cheap_model_max_async=16
)


###### OpenAI Batch API Configuration (50% discount) ######
import json
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

# 全局同步 OpenAI 客户端（Batch API 需要同步客户端）
_global_openai_sync_client = None

def get_openai_sync_client_instance():
    """获取同步 OpenAI 客户端实例"""
    global _global_openai_sync_client
    if _global_openai_sync_client is None:
        _global_openai_sync_client = OpenAI()
        logger.info("初始化 OpenAI 同步客户端（用于 Batch API）")
    return _global_openai_sync_client


class OpenAIBatchProcessor:
    """
    OpenAI Batch API 处理器

    使用 Batch API 可以节省 50% 的费用，但需要等待最多 24 小时。
    适合大批量非实时请求。

    使用示例:
        processor = OpenAIBatchProcessor(model="gpt-4o-mini")

        # 添加请求
        processor.add_request("custom_id_1", "What is 2+2?", system_prompt="You are a math tutor.")
        processor.add_request("custom_id_2", "Explain photosynthesis.")

        # 提交并等待结果
        results = processor.submit_and_wait()

        # 获取结果
        for custom_id, result in results.items():
            print(f"{custom_id}: {result}")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        working_dir: str = "/tmp/openai_batch",
        poll_interval: int = 60,  # 轮询间隔（秒）
        max_wait_time: int = 86400,  # 最大等待时间（秒），默认24小时
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time

        self.client = get_openai_sync_client_instance()
        self.requests: List[Dict] = []
        self.batch_id: Optional[str] = None

    def add_request(
        self,
        custom_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Dict] = None,
    ) -> None:
        """
        添加一个请求到批处理队列

        Args:
            custom_id: 自定义ID，用于匹配结果
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            history_messages: 历史消息（可选）
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        }
        self.requests.append(request)

    def clear_requests(self) -> None:
        """清空请求队列"""
        self.requests = []

    def _create_batch_file(self) -> str:
        """创建 JSONL 批处理文件并上传"""
        if not self.requests:
            raise ValueError("没有请求需要处理")

        # 创建临时 JSONL 文件
        batch_file_path = self.working_dir / f"batch_input_{uuid.uuid4().hex[:8]}.jsonl"
        with open(batch_file_path, 'w', encoding='utf-8') as f:
            for request in self.requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')

        logger.info(f"创建批处理文件: {batch_file_path}，包含 {len(self.requests)} 个请求")

        # 上传文件
        with open(batch_file_path, 'rb') as f:
            file_response = self.client.files.create(file=f, purpose="batch")

        logger.info(f"文件上传成功，file_id: {file_response.id}")
        return file_response.id

    def submit_batch(self) -> str:
        """
        提交批处理任务

        Returns:
            batch_id: 批处理任务ID
        """
        file_id = self._create_batch_file()

        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"VideoRAG batch processing - {len(self.requests)} requests"
            }
        )

        self.batch_id = batch.id
        logger.info(f"批处理任务已提交，batch_id: {self.batch_id}")
        return self.batch_id

    def check_status(self) -> Tuple[str, Optional[Dict]]:
        """
        检查批处理状态

        Returns:
            (status, batch_info): 状态和批处理信息
        """
        if not self.batch_id:
            raise ValueError("没有活跃的批处理任务")

        batch = self.client.batches.retrieve(self.batch_id)
        return batch.status, {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "failed_at": batch.failed_at,
            "request_counts": {
                "total": batch.request_counts.total,
                "completed": batch.request_counts.completed,
                "failed": batch.request_counts.failed,
            },
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }

    def wait_for_completion(self) -> Dict[str, str]:
        """
        等待批处理完成并返回结果

        Returns:
            results: {custom_id: response_content}
        """
        if not self.batch_id:
            raise ValueError("没有活跃的批处理任务")

        start_time = time.time()

        while True:
            status, info = self.check_status()
            elapsed = time.time() - start_time

            logger.info(f"批处理状态: {status}, 已完成: {info['request_counts']['completed']}/{info['request_counts']['total']}, 耗时: {elapsed/60:.1f}分钟")

            if status == "completed":
                logger.info(f"批处理完成！总耗时: {elapsed/60:.1f}分钟")
                return self._download_results(info['output_file_id'])

            elif status in ["failed", "expired", "cancelled"]:
                error_msg = f"批处理失败，状态: {status}"
                if info.get('error_file_id'):
                    errors = self._download_errors(info['error_file_id'])
                    error_msg += f", 错误: {errors}"
                raise RuntimeError(error_msg)

            elif elapsed > self.max_wait_time:
                raise TimeoutError(f"批处理超时，已等待 {elapsed/3600:.1f} 小时")

            time.sleep(self.poll_interval)

    def _download_results(self, output_file_id: str) -> Dict[str, str]:
        """下载并解析结果"""
        content = self.client.files.content(output_file_id)
        results = {}

        for line in content.text.strip().split('\n'):
            if not line:
                continue
            result = json.loads(line)
            custom_id = result['custom_id']

            if result.get('response') and result['response'].get('body'):
                body = result['response']['body']
                if body.get('choices') and len(body['choices']) > 0:
                    content_text = body['choices'][0]['message']['content']
                    results[custom_id] = content_text
                else:
                    results[custom_id] = None
            else:
                results[custom_id] = None

        logger.info(f"成功解析 {len(results)} 个结果")
        return results

    def _download_errors(self, error_file_id: str) -> List[Dict]:
        """下载错误信息"""
        content = self.client.files.content(error_file_id)
        errors = []
        for line in content.text.strip().split('\n'):
            if line:
                errors.append(json.loads(line))
        return errors

    def submit_and_wait(self) -> Dict[str, str]:
        """
        提交批处理并等待完成

        Returns:
            results: {custom_id: response_content}
        """
        self.submit_batch()
        return self.wait_for_completion()


def batch_process_prompts(
    prompts: List[Tuple[str, str, Optional[str]]],  # [(custom_id, prompt, system_prompt), ...]
    model: str = "gpt-4o-mini",
    max_tokens: int = 2048,
    poll_interval: int = 60,
) -> Dict[str, str]:
    """
    批量处理多个提示，使用 Batch API 节省 50% 费用

    Args:
        prompts: [(custom_id, prompt, system_prompt), ...] 列表
        model: 模型名称
        max_tokens: 最大输出 token 数
        poll_interval: 轮询间隔（秒）

    Returns:
        {custom_id: response_content} 字典

    使用示例:
        prompts = [
            ("q1", "What is AI?", "You are a helpful assistant."),
            ("q2", "Explain quantum computing.", None),
        ]
        results = batch_process_prompts(prompts, model="gpt-4o-mini")
    """
    processor = OpenAIBatchProcessor(
        model=model,
        max_tokens=max_tokens,
        poll_interval=poll_interval,
    )

    for custom_id, prompt, system_prompt in prompts:
        processor.add_request(custom_id, prompt, system_prompt)

    return processor.submit_and_wait()


# 便捷函数：将现有的异步调用转换为批处理
async def batch_gpt_complete(
    requests: List[Dict],  # [{"id": str, "prompt": str, "system_prompt": str}, ...]
    model: str = "gpt-4o-mini",
    max_tokens: int = 2048,
) -> Dict[str, str]:
    """
    将多个 GPT 请求打包为批处理（异步接口）

    注意：虽然是异步函数，但实际上会阻塞等待批处理完成（最多24小时）

    Args:
        requests: [{"id": str, "prompt": str, "system_prompt": str (optional)}, ...]
        model: 模型名称
        max_tokens: 最大输出 token 数

    Returns:
        {id: response_content} 字典
    """
    import asyncio

    prompts = [
        (req["id"], req["prompt"], req.get("system_prompt"))
        for req in requests
    ]

    # 在线程池中运行同步的批处理
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: batch_process_prompts(prompts, model=model, max_tokens=max_tokens)
    )

    return results
