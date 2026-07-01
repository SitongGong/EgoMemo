"""
Gemini API 数据生成框架
用于为不同数据集生成不同类型的数据
"""

import os
import json
import time
import argparse
import re
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from google import genai as genai_batch
from google.genai import types
import requests
from egolife_prompt import PROMPTS
from egolife_output_schema.output_schema import EgoLifeOutput


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
        model_name: str = "gemini-1.5-pro",
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
    
    
    def generate_proactive_service(
        self,
        result_files: List[str],
        generation_config: Optional[Dict[str, Any]] = None,
        display_name: str = "proactive-service-job",
        check_interval: int = 30,
        retrieval_interval: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        使用 Gemini API 递归生成数据
        将多个 JSON 文件内容分组（每10个一组），每组分别调用 API 生成数据
        然后使用所有结果进行第二次 API 调用生成对话数据
        
        Args:
            result_files: JSON 文件路径列表
            generation_config: 额外的生成配置参数
            display_name: 显示名称（保留用于兼容性，不再使用）
            check_interval: 检查间隔（保留用于兼容性，不再使用）
            retrieval_interval: 读取文件的间隔（用于控制读取哪些文件）
        
        Returns:
            包含 'first_results' 和 'conversation_results' 的字典
        """
        print(f"准备处理 {len(result_files)} 个 JSON 文件...")
        
        # 选择对应服务类型的输出schema
        if self.high_level_category == "egolife_long_term":
            if self.low_level_category == "long_term_preference":
                from egolife_output_schema.long_term_service import LongTermPreferenceOutput, LongTermPreferenceConv
                output_schema = LongTermPreferenceOutput
                conv_schema = LongTermPreferenceConv
            elif self.low_level_category == "habit_coaching":
                from egolife_output_schema.long_term_service import HabitCoachingOutput, HabitCoachingConv
                output_schema = HabitCoachingOutput
                conv_schema = HabitCoachingConv
            elif self.low_level_category == "routine_optimization":
                from egolife_output_schema.long_term_service import RoutineOptimizationOutput, RoutineOptimizationConv
                output_schema = RoutineOptimizationOutput
                conv_schema = RoutineOptimizationConv
            elif self.low_level_category == "personal_progressive":
                from egolife_output_schema.long_term_service import PersonalProgresssiveOutput, PersonalProgresssiveConv
                output_schema = PersonalProgresssiveOutput
                conv_schema = PersonalProgresssiveConv
            else:
                raise ValueError(f"不支持的输出schema: {self.high_level_category}_{self.low_level_category}")
        else:
            # 对于非长期服务类型，需要根据具体类型导入对应的schema
            # 这里暂时抛出错误，如果后续需要支持其他类型，可以在这里添加
            raise ValueError(f"不支持的输出schema: {self.high_level_category}_{self.low_level_category}。当前仅支持 egolife_long_term 类型。")
        
        
        # 1. 读取所有 JSON 文件内容并转换为字符串，每10个组成一个子列表
        json_contents = []
        current_group = []
        for idx, result_file in enumerate(result_files):
            if not os.path.exists(result_file):
                raise FileNotFoundError(f"JSON 文件不存在: {result_file}")
            
            print(f"读取 JSON 文件 {idx}/{len(result_files)}: {os.path.basename(result_file)}")
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    json_data = {result_file.split("/")[-1].split(".")[0]: json_data}
                    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)          # 将 JSON 转换为格式化的字符串
                    current_group.append(json_str)
                    print(f"  已读取 JSON 文件: {os.path.basename(result_file)}, 大小: {len(json_str)} 字符")
                    
                    # 每10个组成一个子列表
                    if len(current_group) == retrieval_interval:
                        json_contents.append(current_group)
                        current_group = []
            except Exception as e:
                raise RuntimeError(f"读取 JSON 文件失败: {result_file}, 错误: {e}")
        
        # 如果还有剩余的元素，也组成一个子列表
        if current_group:
            json_contents.append(current_group)
        
        # 2. 构建系统提示词
        if self.low_level_category not in PROMPTS:
            raise ValueError(f"未找到服务类型 {self.low_level_category} 的提示词配置")
        if "timestamp" not in PROMPTS[self.low_level_category]:
            raise ValueError(f"服务类型 {self.low_level_category} 缺少 'timestamp' 提示词配置")
        system_prompt = PROMPTS[self.low_level_category]["timestamp"]
        
        # 3. 将 Pydantic 模型转换为 JSON Schema 并移除 default 字段
        # 兼容 Pydantic v1 和 v2
        try:
            # Pydantic v2
            json_schema = output_schema.model_json_schema()
        except AttributeError:
            # Pydantic v1
            json_schema = output_schema.schema()
        
        # 3. 构建生成配置
        generation_config_params = {
            # "temperature": self.temperature,
            "response_mime_type": "application/json",
            "response_json_schema": json_schema,
            **(generation_config or {})
        }
        
        # 4. 开始递归生成数据
        if self.high_level_category == "egolife_long_term":
            accumulated_results = []
            for group_idx, json_group in enumerate(json_contents, 1):
                print(f"\n处理第 {group_idx}/{len(json_contents)} 组数据...")
                
                # 将所有 JSON 内容与 system_prompt 拼接
                # 将每个 JSON 文件内容格式化为带标题的文本
                annotations_text = "\n\n".join([
                    f"## Annotation {i+1}:\n{content}" 
                    for i, content in enumerate(json_group)
                ])
                json_content_text = f"\n\n ### Current Batch: Segment-level Episodic Annotations ###\n\n{annotations_text}"
            
                # 根据服务类型拼接完整的提示词
                if accumulated_results:
                    json_content_text = json_content_text + f"\n\n ### Historical preference summary JSON ### \n\n {json.dumps(accumulated_results, ensure_ascii=False, indent=2)}"
                else:
                    json_content_text = json_content_text + f"\n\n ### Historical preference summary JSON ### \n\n No historical preference summary available"
                
                full_prompt = f"{system_prompt}{json_content_text}"
                
                print(f"完整提示词长度: {len(full_prompt)} 字符")

                # 5. 调用 Gemini API 生成内容
                last_error = None
                for attempt in range(self.max_retries):
                    try:
                        print(f"调用 Gemini API (尝试 {attempt + 1}/{self.max_retries})...")
                        response = self.batch_client.models.generate_content(
                            model=f"models/{self.model_name}",
                            contents=full_prompt,
                            config=generation_config_params
                        )
                        
                        # 提取生成的文本
                        if hasattr(response, 'text'):
                            result_text = response.text
                        elif hasattr(response, 'candidates') and response.candidates:
                            if hasattr(response.candidates[0], 'content'):
                                result_text = response.candidates[0].content.parts[0].text
                            elif hasattr(response.candidates[0], 'text'):
                                result_text = response.candidates[0].text
                            else:
                                raise ValueError("无法从响应中提取文本内容")
                        else:
                            raise ValueError("无法从响应中提取文本内容")
                        
                        print(f"第 {group_idx} 组数据生成成功，结果长度: {len(result_text)} 字符")
                        break
                        
                    except Exception as e:
                        last_error = e
                        if attempt < self.max_retries - 1:
                            wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                            print(f"生成失败，{wait_time:.1f}秒后重试 (尝试 {attempt + 1}/{self.max_retries})...")
                            time.sleep(wait_time)
                        else:
                            raise RuntimeError(f"生成失败，已重试{self.max_retries}次: {str(last_error)}")

                # 清理 Markdown 代码块标记（如果有）
                result_dict = json.loads(result_text)
                if accumulated_results:
                    accumulated_results["candidate_events"].extend(result_dict["candidate_events"])
                else:
                    accumulated_results = result_dict
        # 如果是其他几种类型的服务，换成inline请求
        else:
            # 初始化结果为字典格式，与后续代码期望的数据结构一致
            accumulated_results = {"candidate_events": []}
            inline_requests = []
            for group_idx, json_group in enumerate(json_contents, 1):
                annotations_text = "\n\n".join([
                    f"## Annotation {i+1}:\n{content}" 
                    for i, content in enumerate(json_group)
                ])
                json_content_text = f"\n\n ### Current Batch: Segment-level Episodic Annotations ###\n\n{annotations_text}"
                full_prompt = f"{system_prompt}{json_content_text}"
                
                request = {
                    'contents': [{
                        'parts': [
                            {'text': full_prompt}  # 文本提示
                        ],
                        'role': 'user'
                    }],
                    'config': {
                        'response_mime_type': 'application/json',
                        'response_schema': json_schema,  # 直接使用 Pydantic 模型类型
                        # **(generation_config_params or {})  # 合并其他配置参数（如 temperature）
                    }
                }
                inline_requests.append(request)
            
            try:
                batch_job = self.batch_client.batches.create(
                    model=f"models/{self.model_name}",
                    src=inline_requests,
                    config={
                        'display_name': display_name,
                    },
                )
            except Exception as e:
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
            try:
                # 使用inline的方式输入，使用inline的方式取出
                if batch_job.dest is None:
                    raise RuntimeError("Batch job 没有输出目标 (dest is None)")
                if not hasattr(batch_job.dest, 'inlined_responses'):
                    raise RuntimeError(f"Batch job 输出目标不支持 inlined_responses: {type(batch_job.dest)}")
                if batch_job.dest.inlined_responses is None:
                    raise RuntimeError("Batch job 的 inlined_responses 为 None")
                
                for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                    if inline_response.response:
                        # Accessing response, structure may vary.
                        try:
                            result_dict = json.loads(inline_response.response.text)
                            # 合并结果到 accumulated_results
                            if isinstance(result_dict, dict) and "candidate_events" in result_dict:
                                accumulated_results["candidate_events"].extend(result_dict["candidate_events"])
                            else:
                                # 如果结果格式不符合预期，尝试包装
                                print(f"警告: 结果格式不符合预期，尝试包装: {type(result_dict)}")
                                if isinstance(result_dict, list):
                                    accumulated_results["candidate_events"].extend(result_dict)
                                else:
                                    accumulated_results["candidate_events"].append(result_dict)
                        except (AttributeError, json.JSONDecodeError) as e:
                            print(f"解析响应时出错 (请求 {i}): {e}")
                            print(f"响应内容: {inline_response.response}")
                    elif inline_response.error:
                        print(f"Error in request {i}: {inline_response.error}")
            except Exception as e:
                print(f"获取结果时出错: {e}")
                import traceback
                traceback.print_exc()
                raise
                
        # 6. 重新生成对话数据信息（第二次 API 调用）
        print("\n开始第二次 API 调用：生成对话数据...")
        if "conversation" not in PROMPTS[self.low_level_category]:
            raise ValueError(f"服务类型 {self.low_level_category} 缺少 'conversation' 提示词配置")
        conv_prompt = PROMPTS[self.low_level_category]["conversation"]
        
        # 对生成的服务时间进行整理
        event_dict = {}
        # 检查 accumulated_results 的数据结构
        if not isinstance(accumulated_results, dict):
            raise ValueError(f"accumulated_results 应该是字典类型，但得到: {type(accumulated_results)}")
        if "candidate_events" not in accumulated_results:
            raise ValueError(f"accumulated_results 缺少 'candidate_events' 键。可用键: {list(accumulated_results.keys())}")
        
        for candidate_event in accumulated_results["candidate_events"]:
            event_id = candidate_event["event_id"]
            preference_key = candidate_event["preference_key"]
            
            # 创建候选事件的副本，去掉 event_id 和 preference_key
            event_data = {k: v for k, v in candidate_event.items() 
                         if k not in ["event_id", "preference_key"]}
            
            if event_id not in event_dict:
                event_dict[event_id] = {
                    "preference_key": preference_key,
                    "batches": [event_data]
                }
            else:
                # 验证 preference_key 是否一致
                # assert preference_key == event_dict[event_id]["preference_key"], \
                #     f"preference_key 不一致: {preference_key} vs {event_dict[event_id]['preference_key']}"
                event_dict[event_id]["batches"].append(event_data)
        
        aggregated_results = {
            "service_main_type": "Long-Term Proactive Service",
            "service_sub_type": "Long-Term Preference Proactive Service",
            "proactive_service_events": event_dict, 
        }
        
        try:
            # 将第一次的结果转换为字符串
            result_content = json.dumps(aggregated_results, ensure_ascii=False, indent=2)
            
            print(f"第一次调用结果长度: {len(result_content)} 字符")
            
            # 拼接完整的提示词
            full_conv_prompt = f"{conv_prompt}\n\n{result_content}"
            
            print(f"第二次调用完整提示词长度: {len(full_conv_prompt)} 字符")
            
            # 将 Pydantic 模型转换为 JSON Schema 并移除 default 字段
            try:
                # Pydantic v2
                conv_json_schema = conv_schema.model_json_schema()
            except AttributeError:
                # Pydantic v1
                conv_json_schema = conv_schema.schema()
            
            # 移除 default 字段（Gemini API 不支持）
            # conv_json_schema = remove_default_from_schema(conv_json_schema)
            
            # 构建第二次调用的生成配置
            conv_generation_config = {
                # "temperature": self.temperature,
                "response_mime_type": "application/json",
                "response_schema": conv_json_schema,
                **(generation_config or {})
            }
            
            # 调用 Gemini API 生成对话内容
            print("调用 Gemini API 生成对话数据...")
            last_error = None
            conv_result_text = None
            for attempt in range(self.max_retries):
                try:
                    print(f"调用 Gemini API (尝试 {attempt + 1}/{self.max_retries})...")
                    conv_response = self.batch_client.models.generate_content(
                        full_conv_prompt,
                        generation_config=conv_generation_config
                    )
                    
                    # 提取生成的文本
                    if hasattr(conv_response, 'text'):
                        conv_result_text = conv_response.text
                    elif hasattr(conv_response, 'candidates') and conv_response.candidates:
                        if hasattr(conv_response.candidates[0], 'content'):
                            conv_result_text = conv_response.candidates[0].content.parts[0].text
                        elif hasattr(conv_response.candidates[0], 'text'):
                            conv_result_text = conv_response.candidates[0].text
                        else:
                            raise ValueError("无法从响应中提取文本内容")
                    else:
                        raise ValueError("无法从响应中提取文本内容")
                    
                    print(f"第二次调用成功，结果长度: {len(conv_result_text)} 字符")
                    break

                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                        print(f"生成失败，{wait_time:.1f}秒后重试 (尝试 {attempt + 1}/{self.max_retries})...")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"生成失败，已重试{self.max_retries}次: {str(last_error)}")
            
            # 返回两次调用的结果
            conv_results = []
            if conv_result_text:
                # 清理 Markdown 代码块标记（如果有）
                conv_results = json.loads(conv_results)
            
            return {
                'first_results': accumulated_results,
                'conversation_results': conv_results
            }
            
        except Exception as e:
            print(f"第二次 API 调用失败: {e}")
            import traceback
            traceback.print_exc()
            # 即使第二次调用失败，也返回第一次的结果
            return {
                'first_results': accumulated_results,
                'conversation_results': []
            }
    
    
    def generate_batch_api(
        self,
        user_prompts: List[str],
        generation_config: Optional[Dict[str, Any]] = None,
        display_name: str = "batch-generation-job",
        check_interval: int = 30,
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
        
        # 将 Pydantic 模型转换为 JSON Schema（用于文件序列化）
        # 兼容 Pydantic v1 和 v2
        try:
            # Pydantic v2
            json_schema = EgoLifeOutput.model_json_schema()
        except AttributeError:
            # Pydantic v1
            json_schema = EgoLifeOutput.schema()
        
        for idx, user_prompt in enumerate(user_prompts):
            
            # 构建请求配置，根据官网示例使用 'config' 而不是 'generation_config'
            # 对于 inline 请求，直接使用 Pydantic 模型类型
            inline_request = {
                'contents': [{
                    'parts': [
                        {'text': user_prompt}  # 文本提示
                    ],
                    'role': 'user'
                }],
                'config': {
                    'response_mime_type': 'application/json',
                    'response_schema': EgoLifeOutput,  # 直接使用 Pydantic 模型类型
                    # **(generation_config_params or {})  # 合并其他配置参数（如 temperature）
                }
            }
            
            # 对于文件请求，使用 JSON Schema（可序列化）
            # Gemini Batch API 使用 generation_config 而不是 config
            file_request = {
                "key": f"request-{idx}",
                "request":{
                    'contents': [{
                        'parts': [
                            {'text': user_prompt}  # 文本提示
                        ],
                    }],
                },
                'generation_config': {
                    'response_mime_type': 'application/json',
                    'response_schema': json_schema,  # 使用 JSON Schema 字典
                    **generation_config_params  # 合并其他配置参数（如 temperature）
                }
            }
            
            inline_requests.append(inline_request)
            # 对于文件请求，直接使用请求对象，不需要包装
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
                for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                    if inline_response.response:
                    # Accessing response, structure may vary.
                        try:
                            results.append(json.loads(inline_response.response.text))
                        except AttributeError:
                            print(inline_response.response) # Fallback
                    elif inline_response.error:
                        print(f"Error: {inline_response.error}")
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
                # 逐行解析
                for line in file_content.strip().split('\n'):
                    if line.strip():  # 跳过空行
                        try:
                            result_item = json.loads(line)
                            # 提取实际的响应内容
                            if isinstance(result_item, dict):
                                # 检查是否有 response 字段
                                if 'response' in result_item:
                                    response_obj = result_item['response']
                                    # response_obj 可能是字典或对象
                                    if isinstance(response_obj, dict):
                                        # 如果是字典，尝试提取文本
                                        if 'candidates' in response_obj and response_obj['candidates']:
                                            candidate = response_obj['candidates'][0]
                                            if 'content' in candidate and 'parts' in candidate['content']:
                                                text_parts = [part.get('text', '') for part in candidate['content']['parts'] if isinstance(part, dict) and 'text' in part]
                                                if text_parts:
                                                    results.append(json.loads(extract_json_from_markdown(''.join(text_parts))))
                                                else:
                                                    results.append(json.dumps(response_obj, ensure_ascii=False))
                                            else:
                                                results.append(json.dumps(response_obj, ensure_ascii=False))
                                        elif 'text' in response_obj:
                                            results.append(response_obj['text'])
                                        else:
                                            results.append(json.dumps(response_obj, ensure_ascii=False))
                                    else:
                                        # 如果是对象，使用 hasattr 检查
                                        if hasattr(response_obj, 'candidates') and response_obj.candidates:
                                            candidate = response_obj.candidates[0]
                                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                                text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                                                if text_parts:
                                                    results.append(json.loads(extract_json_from_markdown(''.join(text_parts))))
                                                else:
                                                    results.append(str(response_obj))
                                            else:
                                                results.append(str(response_obj))
                                        elif hasattr(response_obj, 'text'):
                                            results.append(response_obj.text)
                                        else:
                                            results.append(str(response_obj))
                                else:
                                    # 如果没有 response 字段，直接使用整个对象
                                    results.append(result_item)
                            else:
                                results.append(result_item)
                        except json.JSONDecodeError as e:
                            print(f"解析 JSON 行时出错: {e}, 行内容: {line[:100]}...")
                            continue
                        except Exception as e:
                            print(f"处理结果行时出错: {e}, 行内容: {line[:100]}...")
                            import traceback
                            traceback.print_exc()
                            continue
            else:
                print("警告：batch job 没有输出文件")
            
        except Exception as e:
            print(f"获取结果时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 清理已上传的文件
        print("清理已上传的视频文件...")
        try:
            genai.delete_file(uploaded_file.name)
        except Exception as e:
                print(f"警告：删除上传的batch请求文件失败: {e}")
        
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
        input_mode=args.input_mode
    )
    
    # 创建新的需要保存的文件夹
    egolife_rag_annotation_list = []
    for activity in args.activity_list:
        for day in args.days_list:
            egolife_annotation_path = os.path.join(args.ego_life_path, activity, day)
            egolife_annotation_list = sorted(os.listdir(egolife_annotation_path))
            os.makedirs(os.path.join(args.output_path, activity, day), exist_ok=True)
            for annotation_file in egolife_annotation_list:
                egolife_rag_annotation_list.append((os.path.join(egolife_annotation_path, annotation_file), os.path.join(args.output_path, activity, day, annotation_file)))            
    print(f"共生成 {len(egolife_rag_annotation_list)} 条数据")
    
    
    # 获取提示词
    system_prompt = PROMPTS["egolife_summarize_system_prompt"]
    
    # 生成数据
    try:
        # 使用 Batch API 批量生成
        print("使用 Batch API 模式...")
        
        # 准备所有提示词和视频路径
        user_prompts = []
        video_paths = []
        output_paths = []
        for idx, (annotation_path, output_path) in enumerate(egolife_rag_annotation_list):
            if idx < 64:
                continue
            
            # 读取注释文件
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation_data = json.load(f)
            annotation_data = {output_path.split("/")[-1]: annotation_data}
            
            # 构建用户提示词
            annotation_str = json.dumps(annotation_data, ensure_ascii=False, indent=2)  # 或者 indent=2
            user_prompt = system_prompt + PROMPTS["egolife_summarize_user_prompt"].format(annotation_data=annotation_str)
            user_prompts.append(user_prompt)
            output_paths.append((annotation_path, output_path))
        
        # 使用 Batch API 生成
        if args.record_stage:
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
                    check_interval=args.batch_check_interval
                )
                batch_results.extend(batch_result)
            
            # 整理结果格式
            result_path_list = []
            for idx, (annotation_path, output_path) in enumerate(output_paths):
                result_item = batch_results[idx]
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result_item, f, ensure_ascii=False, indent=2)
                print(f"结果已保存到: {output_path}")
                result_path_list.append(output_path)
        else:
            result_path_list = []
            for annotation_path, output_path in output_paths:
                if not os.path.exists(output_path):
                    raise FileNotFoundError(f"输出文件不存在: {output_path}")
                result_path_list.append(output_path)
        
        # 采用递归方式生成所有潜在主动服务timestamp
        if args.conversation_stage:
            proactive_service_results = generator.generate_proactive_service(result_files=result_path_list,
                                                generation_config=None,
                                                display_name="proactive-service-job")
        
            first_results = proactive_service_results["first_results"]
            conversation_results = proactive_service_results["conversation_results"]
            
            with open(os.path.join(args.output_path, "first_results.json"), "w", encoding="utf-8") as f:
                json.dump(first_results, f, ensure_ascii=False, indent=2)
            with open(os.path.join(args.output_path, "conversation_results.json"), "w", encoding="utf-8") as f:
                json.dump(conversation_results, f, ensure_ascii=False, indent=2)
        
        
    except Exception as e:
        print(f"生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument("--task", type=str, default="long_preference")
    parse.add_argument("--output_path", type=str, default="./data/egolife/gemini_annotation_segments")
    # 对于egolife数据集的配置参数
    parse.add_argument("--ego_life_path", type=str, default="./data/egolife/rag_annotation_segments")
    parse.add_argument("--activity_list", type=list, default=["A1_JAKE"])
    parse.add_argument("--days_list", type=list, default=["DAY1", "DAY2", "DAY3", "DAY4", "DAY5", "DAY6", "DAY7"])
    # 对于模型的配置参数
    parse.add_argument("--model_name", type=str, default="gemini-2.5-pro")
    parse.add_argument("--save_format", type=str, default="json")
    parse.add_argument("--temperature", type=float, default=0.7)
    parse.add_argument("--max_retries", type=int, default=3)
    parse.add_argument("--retry_delay", type=float, default=1.0)
    parse.add_argument("--api_key", type=str, default=os.environ.get("GEMINI_API_KEY", ""))
    parse.add_argument("--use_batch_api", type=bool, default=True, help="使用 Batch API 进行批量生成")
    parse.add_argument("--batch_check_interval", type=int, default=30, help="检查 batch job 状态的间隔（秒）")
    parse.add_argument("--input_mode", type=str, default="inline", choices=["inline", "file"])
    # 对于服务类型prompt的选择
    parse.add_argument("--high_level_service_type", type=str, default="egolife_long_term")
    parse.add_argument("--low_level_service_type", type=str, default="long_term_preference")
    # 对于阶段的选择，一共两个阶段，一个是生成记录数据，另一个是直接生成对话数据
    parse.add_argument("--record_stage", type=bool, default=True)
    parse.add_argument("--conversation_stage", type=bool, default=False)
    
    args = parse.parse_args()
    
    main(args)

