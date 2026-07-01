"""
Gemini API 数据生成框架
用于为不同数据集生成不同类型的数据
"""

import os
import re
import json
import time
import argparse
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from google import genai as genai_batch
from google.genai import types
import requests
from holoassist_prompt import PROMPTS
from pydantic import BaseModel, Field


High_Level_Services = ["short_term"] # "short_term"
Instant_Services = ["safety", "tool_use"]  # "safety" "tool_use"
Short_Term_Services = ["next_step_guidance", "resource_reminder"]


def extract_json_from_markdown(text: str) -> str:
    """
    从 Markdown 代码块中提取 JSON 内容
    处理 ```json ... ``` 或 ``` ... ``` 格式
    """
    # 移除 Markdown 代码块标记
    # 匹配 ```json ... ``` 或 ``` ... ```
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # 如果有代码块，返回最后一个（通常是最完整的）
        return matches[-1].strip()
    
    # 如果没有代码块，尝试查找 JSON 对象（以 { 开头，以 } 结尾）
    json_pattern = r'\{.*\}'
    json_match = re.search(json_pattern, text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # 如果都没有，返回原始文本
    return text.strip()


class GeminiDataGenerator:
    """使用Gemini API生成数据的生成器类"""
    
    def __init__(
        self,
        high_level_category: str = "egolife_long_term",
        low_level_category: str = "long_term_preference", 
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0, 
        input_mode: float = "inline", 
    ):
        """
        初始化Gemini数据生成器
        
        Args:
            api_key: Gemini API密钥，如果为None则从环境变量GEMINI_API_KEY读取
            model_name: 使用的模型名称，默认为gemini-1.5-pro
            temperature: 生成温度，控制随机性
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API密钥未提供，请设置api_key参数或GEMINI_API_KEY环境变量")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.input_mode = input_mode
        self.high_level_category = high_level_category
        self.low_level_category = low_level_category
        
        # 配置API密钥
        genai.configure(api_key=self.api_key)
        
        # 初始化 batch API 客户端
        self.batch_client = genai_batch.Client(api_key=self.api_key)
    
    
    def generate_batch_api(
        self,
        user_prompts: List[str],
        generation_config: Optional[Dict[str, Any]] = None,
        display_name: str = "batch-generation-job",
        task_type: str = "safety", 
        check_interval: int = 30,
        output_schema: Optional[BaseModel] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        使用 Batch API 批量生成数据（支持多模态输入：视频+文本）
        
        Args:
            user_prompts: 用户提示词列表
            video_paths: 视频文件路径列表，长度应与user_prompts相同
            generation_config: 额外的生成配置参数
            display_name: Batch job 的显示名称
            check_interval: 检查 batch job 状态的间隔（秒）
            **kwargs: 其他传递给API的参数
        
        Returns:
            包含生成结果和元数据的字典列表
        """
        
        print(f"准备创建 batch job，共 {len(user_prompts)} 个请求...")
        
        # 构建生成配置
        generation_config_params = {
            "temperature": self.temperature,
            **(generation_config or {})
        }
        
        # 第二步：构建 batch 请求列表
        print("构建 batch 请求...")
        inline_requests = []
        file_requests = []
        # 用于追踪请求索引的映射（用于 inline 模式）
        request_index_map = {}  # {request_id: original_index}
        
        for idx, user_prompt in enumerate(user_prompts):
            # 为每个请求生成唯一标识符（添加到 prompt 中作为隐藏标记）
            # 使用一个不太可能出现在正常 prompt 中的特殊标记
            request_id = f"__BATCH_REQUEST_ID_{idx}__"
            # 在 prompt 末尾添加请求 ID（作为注释，不影响模型输出）
            # 注意：由于我们使用 JSON schema，模型不会输出这个 ID，但我们可以通过其他方式追踪
            
            # 构建请求配置，根据官网示例使用 'config' 而不是 'generation_config'
            # 直接使用 Pydantic 模型类型作为 response_schema
            request = {
                'contents': [{
                    'parts': [
                        {'text': user_prompt}  # 文本提示
                    ],
                    'role': 'user'
                }],
                'config': {
                    'response_mime_type': 'application/json',
                    # 'response_schema': output_schema,  # 直接使用 Pydantic 模型类型
                    # **(generation_config_params or {})  # 合并其他配置参数（如 temperature）
                }
            }
            # 存储请求索引（用于后续匹配）
            request_index_map[id(request)] = idx
            
            file_request = {
                "key": f"request-{idx}",  # file 模式使用 key 来追踪
                "request":{
                    'contents': [{
                        'parts': [
                            {'text': user_prompt}  # 文本提示
                        ],
                    }],
                },
                'generation_config': {
                    'response_mime_type': 'application/json',
                    'response_schema': output_schema,  # 使用 JSON Schema 字典
                    **generation_config_params  # 合并其他配置参数（如 temperature）
                }
            }
            
            inline_requests.append(request)
            file_requests.append(file_request)
            
        # 第三步：创建 batch job   这里使用两种方式
        with open("my-batch-requests.jsonl", "w", encoding="utf-8") as f:
            for request_item in file_requests:
                f.write(json.dumps(request_item, ensure_ascii=False) + "\n")
                
        # Upload the file to the File API
        uploaded_file = self.batch_client.files.upload(
            file='my-batch-requests.jsonl',
            config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='jsonl')
        )
                
        print(f"创建 batch job: {display_name}...")
        try:
            if self.input_mode == "inline":
                batch_job = self.batch_client.batches.create(
                    model=f"models/{self.model_name}",
                    src=inline_requests,
                    config={
                        'display_name': display_name,
                    },
                )
            elif self.input_mode == "file":
                batch_job = self.batch_client.batches.create(
                    model=f"models/{self.model_name}",
                    src=uploaded_file.name,
                    config={
                        'display_name': display_name,
                    },
                )
            print(f"Batch job 已创建: {batch_job.name}")
            print(f"状态: {batch_job.state}")
        except Exception as e:
            # 清理已上传的文件
            print(f"创建 batch job 失败，清理已上传的文件...")
            raise RuntimeError(f"创建 batch job 失败: {str(e)}")
        
        # 第四步：等待 batch job 完成
        print("等待 batch job 完成...")
        batch_job_name = batch_job.name  # 保存 batch job 名称
        
        incompleted_states = set([
                                'JOB_STATE_FAILED',
                                'JOB_STATE_CANCELLED',
                                'JOB_STATE_EXPIRED',
                            ])
        
        while True:
            batch_job = self.batch_client.batches.get(name=batch_job_name)
            print(f"当前状态: {batch_job.state}")
            
            if batch_job.state.name == "JOB_STATE_SUCCEEDED":
                print("Batch job 已完成！")
                break
            elif batch_job.state.name in incompleted_states:
                raise RuntimeError(f"Batch job 失败: {batch_job.name}")
            
            time.sleep(check_interval)
        
        # 第五步：获取结果
        print("获取 batch job 结果...")
        results = []
        try:
            # 使用inline的方式输入，使用inline的方式取出
            if batch_job.dest and batch_job.dest.inlined_responses:
                # Gemini Batch API 官方文档说明：inline 模式的输出顺序应该和输入顺序一致
                # 如果出现顺序混乱，可能是由于：
                # 1. JSON 解析错误导致某些结果被跳过
                # 2. 响应格式不符合 schema 导致解析失败
                # 3. 错误处理不当导致索引不匹配
                
                expected_count = len(user_prompts)
                actual_count = len(batch_job.dest.inlined_responses)
                
                print(f"收到 {actual_count} 个响应，期望 {expected_count} 个")
                if expected_count != actual_count:
                    print(f"警告: 响应数量不匹配！这可能导致结果与输入不对应")
                
                # 确保所有响应都被处理，保持顺序一致
                # 使用列表而不是字典，确保顺序
                results = [None] * expected_count  # 预先分配，确保长度一致
                
                for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                    # 检查索引是否超出范围
                    if i >= expected_count:
                        print(f"错误: 响应索引 {i} 超出期望范围 [0, {expected_count-1}]")
                        continue
                    
                    if inline_response.response:
                        try:
                            # 尝试获取响应文本
                            response_text = inline_response.response.text
                            
                            # 尝试解析 JSON
                            try:
                                result_data = json.loads(response_text)
                                results[i] = result_data
                            except json.JSONDecodeError as e:
                                # JSON 解析失败 - 可能是格式不符合 schema
                                print(f"错误: 响应 {i} 的 JSON 解析失败")
                                print(f"  错误信息: {e}")
                                print(f"  响应内容前500字符: {response_text[:500] if response_text else 'None'}")
                                # 尝试从 markdown 代码块中提取 JSON
                                try:
                                    extracted_json = extract_json_from_markdown(response_text)
                                    result_data = json.loads(extracted_json)
                                    results[i] = result_data
                                    print(f"  成功从 markdown 中提取 JSON")
                                except Exception as e2:
                                    print(f"  从 markdown 提取也失败: {e2}")
                                    results[i] = None
                        except AttributeError as e:
                            # 无法访问 response.text
                            print(f"错误: 无法访问响应 {i} 的 text 属性")
                            print(f"  错误信息: {e}")
                            print(f"  response 对象: {inline_response.response}")
                            results[i] = None
                        except Exception as e:
                            # 其他未预期的错误
                            print(f"错误: 处理响应 {i} 时发生未预期的错误: {e}")
                            import traceback
                            traceback.print_exc()
                            results[i] = None
                    elif inline_response.error:
                        # API 返回了错误
                        print(f"错误: 响应 {i} 包含错误: {inline_response.error}")
                        results[i] = None
                    else:
                        # 既没有 response 也没有 error
                        print(f"警告: 响应 {i} 既没有 response 也没有 error")
                        print(f"  响应对象: {inline_response}")
                        results[i] = None
                
                # 检查是否有 None 值（表示处理失败）
                failed_count = sum(1 for r in results if r is None)
                if failed_count > 0:
                    print(f"警告: {failed_count}/{expected_count} 个响应处理失败，这些位置的结果为 None")
            elif batch_job.dest and batch_job.dest.file_name:
                # Results are in a file (usually JSONL format)
                result_file_name = batch_job.dest.file_name
                print(f"Results are in file: {result_file_name}")

                print("Downloading result file content...")
                file_content = self.batch_client.files.download(file=result_file_name)
                
                # 处理文件内容：可能是 bytes 或字符串
                if isinstance(file_content, bytes):
                    file_content = file_content.decode('utf-8')
                
                # Batch API 结果文件通常是 JSONL 格式（每行一个 JSON 对象）
                # 注意：file 模式的结果顺序可能与输入顺序不一致
                # 需要使用 'key' 字段来匹配原始请求
                indexed_results = {}  # {key: result} 用于存储结果
                
                # 逐行解析
                for line in file_content.strip().split('\n'):
                    if line.strip():  # 跳过空行
                        try:
                            result_item = json.loads(line)
                            # 提取 key（用于匹配原始请求）
                            result_key = None
                            if isinstance(result_item, dict) and 'key' in result_item:
                                result_key = result_item['key']
                            
                            # 提取实际的响应内容
                            if isinstance(result_item, dict):
                                # 检查是否有 response 字段
                                if 'response' in result_item:
                                    response_obj = result_item['response']
                                    parsed_result = None
                                    # response_obj 可能是字典或对象
                                    if isinstance(response_obj, dict):
                                        # 如果是字典，尝试提取文本
                                        if 'candidates' in response_obj and response_obj['candidates']:
                                            candidate = response_obj['candidates'][0]
                                            if 'content' in candidate and 'parts' in candidate['content']:
                                                text_parts = [part.get('text', '') for part in candidate['content']['parts'] if isinstance(part, dict) and 'text' in part]
                                                if text_parts:
                                                    parsed_result = json.loads(extract_json_from_markdown(''.join(text_parts)))
                                                else:
                                                    parsed_result = json.dumps(response_obj, ensure_ascii=False)
                                            else:
                                                parsed_result = json.dumps(response_obj, ensure_ascii=False)
                                        elif 'text' in response_obj:
                                            parsed_result = response_obj['text']
                                        else:
                                            parsed_result = json.dumps(response_obj, ensure_ascii=False)
                                    else:
                                        # 如果是对象，使用 hasattr 检查
                                        if hasattr(response_obj, 'candidates') and response_obj.candidates:
                                            candidate = response_obj.candidates[0]
                                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                                text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                                                if text_parts:
                                                    parsed_result = json.loads(extract_json_from_markdown(''.join(text_parts)))
                                                else:
                                                    parsed_result = str(response_obj)
                                            else:
                                                parsed_result = str(response_obj)
                                        elif hasattr(response_obj, 'text'):
                                            parsed_result = response_obj.text
                                        else:
                                            parsed_result = str(response_obj)
                                    
                                    # 使用 key 存储结果
                                    if result_key:
                                        indexed_results[result_key] = parsed_result
                                    else:
                                        # 如果没有 key，按顺序添加（可能不准确）
                                        indexed_results[len(indexed_results)] = parsed_result
                                else:
                                    # 如果没有 response 字段，直接使用整个对象
                                    if result_key:
                                        indexed_results[result_key] = result_item
                                    else:
                                        indexed_results[len(indexed_results)] = result_item
                            else:
                                if result_key:
                                    indexed_results[result_key] = result_item
                                else:
                                    indexed_results[len(indexed_results)] = result_item
                        except json.JSONDecodeError as e:
                            print(f"解析 JSON 行时出错: {e}, 行内容: {line[:100]}...")
                            continue
                        except Exception as e:
                            print(f"处理结果行时出错: {e}, 行内容: {line[:100]}...")
                            import traceback
                            traceback.print_exc()
                            continue
                
                # 按原始请求顺序构建结果列表（使用 key 匹配）
                for idx in range(len(user_prompts)):
                    key = f"request-{idx}"
                    if key in indexed_results:
                        results.append(indexed_results[key])
                    else:
                        print(f"警告: 缺少 key '{key}' 的结果，使用 None 占位")
                        results.append(None)
            else:
                print("警告: batch job 没有输出文件")
            
        except Exception as e:
            print(f"获取结果时出错: {e}")
            import traceback
            traceback.print_exc()
        
        return results


def main(args):
    
    """示例用法"""
    # 初始化生成器
    generator = GeminiDataGenerator(
        high_level_category=args.high_level_service_type,
        low_level_category=args.low_level_service_type,
        api_key=args.api_key,  # 在这里填入你的API密钥，或设置环境变量GEMINI_API_KEY
        model_name=args.model_name,
        temperature=0.7,
        input_mode=args.input_mode, 
    )
    
    video_annotation_list = [os.path.join(args.holoassist_path, name) for name in sorted(os.listdir(args.holoassist_path))]
    
    # 生成数据
    try:
        # 使用 Batch API 批量生成
        print("使用 Batch API 模式...")
        
        # 对于每一种服务类型，分别标注数据
        for high_level_service_type in High_Level_Services:
            if high_level_service_type == "instant":
                low_level_service_types = Instant_Services
            else:
                low_level_service_types = Short_Term_Services
            
            for low_level_service_type in low_level_service_types:
                
                if high_level_service_type == "instant":
                    if low_level_service_type == "safety":
                        from holoassist_output_schema.instant_service import SafetyInstantServiceOutput
                        output_schema = SafetyInstantServiceOutput.model_json_schema()
                    elif low_level_service_type == "tool_use":
                        from holoassist_output_schema.instant_service import ToolUseInstantServiceOutput
                        output_schema = ToolUseInstantServiceOutput.model_json_schema()
                    else:
                        raise ValueError(f"不支持的低级服务类型: {low_level_service_type}")
                    
                elif high_level_service_type == "short_term":
                    if low_level_service_type == "next_step_guidance":
                        from holoassist_output_schema.short_term_service import NextStepServiceOutput
                        output_schema = NextStepServiceOutput.model_json_schema()
                    elif low_level_service_type == "error_recovery":
                        from holoassist_output_schema.short_term_service import ErrorRecoveryServiceOutput
                        output_schema = ErrorRecoveryServiceOutput.model_json_schema()
                    elif low_level_service_type == "resource_reminder":
                        from holoassist_output_schema.short_term_service import ResourceReminderServiceOutput
                        output_schema = ResourceReminderServiceOutput.model_json_schema()
                    else:
                        raise ValueError(f"不支持的低级服务类型: {low_level_service_type}")
                
                
                # 获取提示词
                system_prompt = PROMPTS[high_level_service_type][low_level_service_type]
                    
                # 将人工标注读取出来然后和system_prompt拼接在一起
                user_prompts = []
                output_paths = []
                for idx, annotation_path in enumerate(video_annotation_list):
                    # if idx < 500:
                    #     continue
                    
                    # 读取注释文件
                    with open(annotation_path, "r", encoding="utf-8") as f:
                        annotation_data = json.load(f)
                        
                    output_path = os.path.join(args.output_path, high_level_service_type, low_level_service_type, annotation_path.split("/")[-1].split(".")[0] + ".json")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # 构建用户提示词
                    annotation_str = json.dumps(annotation_data, ensure_ascii=False, indent=2)  # 或者 indent=2
                    user_prompt = system_prompt + "\n\n" + PROMPTS["user_prompt"].format(annotation_data=annotation_str, clip_id=annotation_data["video_name"])
                    user_prompts.append(user_prompt)
                    output_paths.append(output_path)
        
                # 使用 Batch API 生成
                # 将请求分批处理，每 128 个为一组
                batch_size = 128
                batch_results = []
                total_batches = (len(user_prompts) + batch_size - 1) // batch_size
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(user_prompts))
                    batch_prompts = user_prompts[start_idx:end_idx]
                    
                    print(f"处理第 {batch_idx + 1}/{total_batches} 批请求 (索引 {start_idx} 到 {end_idx - 1})...")
                    batch_result = generator.generate_batch_api(
                        user_prompts=batch_prompts,
                        display_name=f"egolife-summarize-batch-{batch_idx + 1}",
                        check_interval=args.batch_check_interval, 
                        task_type=low_level_service_type,
                        output_schema=output_schema, 
                    )
                    batch_results.extend(batch_result)
            
                # 整理结果格式
                for idx, (annotation_path, output_path) in enumerate(zip(batch_results, output_paths)):
                    result_item = batch_results[idx]
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(result_item, f, ensure_ascii=False, indent=2)
                    print(f"结果已保存到: {output_path}")      
        
    except Exception as e:
        print(f"生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument("--output_path", type=str, default="./data/HoloAssist_hyf/holoassist_service_annotations_rec_")
    # 对于holoassist数据集的配置参数
    parse.add_argument("--holoassist_path", type=str, default="./data/holoassist_annotations_")
    # 对于模型的配置参数
    parse.add_argument("--model_name", type=str, default="gemini-3-pro-preview")
    parse.add_argument("--save_format", type=str, default="json")
    parse.add_argument("--temperature", type=float, default=0.7)
    parse.add_argument("--max_retries", type=int, default=3)
    parse.add_argument("--retry_delay", type=float, default=1.0)
    parse.add_argument("--api_key", type=str, default=os.environ.get("GEMINI_API_KEY", ""))
    parse.add_argument("--use_batch_api", type=bool, default=True, help="使用 Batch API 进行批量生成")
    parse.add_argument("--batch_check_interval", type=int, default=30, help="检查 batch job 状态的间隔（秒）")
    parse.add_argument("--input_mode", type=str, default="file", choices=["inline", "file"])
    # 对于服务类型prompt的选择
    parse.add_argument("--high_level_service_type", type=str, nargs='+', default=["instant"], choices=["instant", "short_term"])
    parse.add_argument("--low_level_service_type", type=str, nargs='+', default=["safety", "tool_use"])
    
    args = parse.parse_args()
    
    main(args)

