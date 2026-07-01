"""
Gemini API 数据生成框架
用于为不同数据集生成不同类型的数据
"""

import os
import json
import time
import argparse
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from google import genai as genai_batch
from google.genai import types
import requests
from prompt import PROMPTS


class GeminiDataGenerator:
    """使用Gemini API生成数据的生成器类"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0
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
        
        # 配置API密钥
        genai.configure(api_key=self.api_key)
        
        # 初始化 batch API 客户端
        self.batch_client = genai_batch.Client(api_key=self.api_key)
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        video_path: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        生成数据
        
        Args:
            system_prompt: 系统提示词（角色定义和任务说明）
            user_prompt: 用户提示词（具体的数据生成请求）
            video_path: 视频文件路径（可选），支持MP4、MPEG、MOV、AVI等格式
            generation_config: 额外的生成配置参数
            **kwargs: 其他传递给API的参数
        
        Returns:
            生成的文本内容
        """
        if not system_prompt:
            raise ValueError("system_prompt不能为空")
        if not user_prompt:
            raise ValueError("user_prompt不能为空")
        
        # 构建生成配置
        generation_config_params = {
            "temperature": self.temperature,
            **(generation_config or {})
        }
        
        # 初始化模型
        model = genai.GenerativeModel(model_name=self.model_name)
        
        # 构建完整提示词（Gemini API将system_prompt和user_prompt合并）
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # 准备内容：如果有视频，需要同时传递文本和视频
        video_file = None
        if video_path:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
            # 上传视频文件
            print(f"正在上传视频文件: {video_path}")
            video_file = genai.upload_file(path=video_path)
            
            # 等待文件处理完成
            print("等待视频文件处理完成...")
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise RuntimeError(f"视频文件处理失败: {video_file.state.name}")
            
            print(f"视频文件已就绪: {video_path}")
            
            # 构建包含视频和文本的内容
            content = [video_file, full_prompt]
        else:
            # 仅文本内容
            content = full_prompt
        
        # 重试机制
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # 将生成配置作为参数传递
                response = model.generate_content(
                    content,
                    generation_config=generation_config_params,
                    **kwargs
                )
                
                # 如果使用了视频文件，在完成后删除上传的文件
                if video_path:
                    try:
                        genai.delete_file(video_file.name)
                    except Exception as e:
                        print(f"警告：删除上传的视频文件失败: {e}")
                
                # 提取生成的文本
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    if hasattr(response.candidates[0], 'content'):
                        return response.candidates[0].content.parts[0].text
                    elif hasattr(response.candidates[0], 'text'):
                        return response.candidates[0].text
                else:
                    raise ValueError("无法从响应中提取文本内容")
                    
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                    print(f"生成失败，{wait_time:.1f}秒后重试 (尝试 {attempt + 1}/{self.max_retries})...")
                    time.sleep(wait_time)
                else:
                    # 如果失败，确保清理上传的文件
                    if video_path and 'video_file' in locals():
                        try:
                            genai.delete_file(video_file.name)
                        except Exception:
                            pass
                    raise RuntimeError(f"生成失败，已重试{self.max_retries}次: {str(last_error)}")
    
    def generate_batch(
        self,
        system_prompt: str,
        user_prompts: List[str],
        video_paths: Optional[List[str]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        delay_between_requests: float = 0.5,
        **kwargs
    ) -> List[str]:
        """
        批量生成数据
        
        Args:
            system_prompt: 系统提示词
            user_prompts: 用户提示词列表
            video_paths: 视频文件路径列表（可选），如果提供，长度应与user_prompts相同
            generation_config: 额外的生成配置参数
            delay_between_requests: 请求之间的延迟（秒），避免API限流
            **kwargs: 其他传递给API的参数
        
        Returns:
            生成的文本内容列表
        """
        results = []
        total = len(user_prompts)
        
        if video_paths and len(video_paths) != total:
            raise ValueError(f"video_paths长度({len(video_paths)})必须与user_prompts长度({total})相同")
        
        for idx, user_prompt in enumerate(user_prompts, 1):
            print(f"正在生成 {idx}/{total}...")
            video_path = video_paths[idx - 1] if video_paths else None
            result = self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                video_path=video_path,
                generation_config=generation_config,
                **kwargs
            )
            results.append(result)
            
            # 请求间隔，避免API限流
            if idx < total and delay_between_requests > 0:
                time.sleep(delay_between_requests)
        
        return results
    
    def generate_and_save(
        self,
        system_prompt: str,
        user_prompt: str,
        output_path: str,
        video_path: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        save_format: str = "json",
        **kwargs
    ) -> str:
        """
        生成数据并保存到文件
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            output_path: 输出文件路径
            video_path: 视频文件路径（可选）
            generation_config: 额外的生成配置参数
            save_format: 保存格式，支持'json', 'txt', 'jsonl'
            **kwargs: 其他传递给API的参数
        
        Returns:
            生成的文本内容
        """
        result = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_path=video_path,
            generation_config=generation_config,
            **kwargs
        )
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        if save_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"generated_text": result}, f, ensure_ascii=False, indent=2)
        elif save_format == "jsonl":
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump({"generated_text": result}, f, ensure_ascii=False)
                f.write("\n")
        else:  # txt
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
        
        print(f"结果已保存到: {output_path}")
        return result
    
    def generate_batch_api(
        self,
        user_prompts: List[str],
        video_paths: List[str],
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
        if len(user_prompts) != len(video_paths):
            raise ValueError(f"user_prompts长度({len(user_prompts)})必须与video_paths长度({len(video_paths)})相同")
        
        print(f"准备创建 batch job，共 {len(user_prompts)} 个请求...")
        
        # 构建生成配置
        generation_config_params = {
            "temperature": self.temperature,
            **(generation_config or {})
        }
        
        # 第一步：上传所有视频文件
        print("正在上传视频文件...")
        uploaded_video_files = []
        for idx, video_path in enumerate(video_paths, 1):
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
            print(f"上传视频 {idx}/{len(video_paths)}: {os.path.basename(video_path)}")
            try:
                # 上传视频文件
                video_file = genai.upload_file(path=video_path)
                
                # 等待文件处理完成
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)
                
                if video_file.state.name == "FAILED":
                    raise RuntimeError(f"视频文件处理失败: {video_path}")
                
                uploaded_video_files.append(video_file)
                print(f"视频文件已就绪: {os.path.basename(video_path)}")
            except Exception as e:
                print(f"上传视频文件失败: {video_path}, 错误: {e}")
                # 清理已上传的文件
                for uploaded_file in uploaded_video_files:
                    try:
                        genai.delete_file(uploaded_file.name)
                    except:
                        pass
                raise
        
        # 第二步：构建 batch 请求列表
        print("构建 batch 请求...")
        inline_requests = []
        for user_prompt, video_file in zip(user_prompts, uploaded_video_files):
            # 构建完整提示词
            full_prompt = f"{user_prompt}"
            
            # 构建请求内容（多模态：视频+文本）
            # 对于 batch API，使用文件 URI 格式引用上传的视频文件
            file_uri = getattr(video_file, 'uri', None) or getattr(video_file, 'name', None)
            mime_type = getattr(video_file, 'mime_type', 'video/mp4')
            
            request = {
                'contents': [{
                    'parts': [
                        {
                            'file_data': {
                                'file_uri': file_uri,
                                'mime_type': mime_type
                            }
                        },
                        {'text': full_prompt}  # 文本提示
                    ],
                    'role': 'user'
                }]
            }
            
            # 如果有生成配置，添加到请求中
            if generation_config_params:
                request['generation_config'] = generation_config_params
            
            inline_requests.append(request)
        
        # 第三步：创建 batch job
        print(f"创建 batch job: {display_name}...")
        try:
            batch_job = self.batch_client.batches.create(
                model=f"models/{self.model_name}",
                src=inline_requests,
                config={
                    'display_name': display_name,
                },
            )
            print(f"Batch job 已创建: {batch_job.name}")
            print(f"状态: {batch_job.state}")
        except Exception as e:
            # 清理已上传的文件
            print(f"创建 batch job 失败，清理已上传的文件...")
            for uploaded_file in uploaded_video_files:
                try:
                    genai.delete_file(uploaded_file.name)
                except:
                    pass
            raise RuntimeError(f"创建 batch job 失败: {str(e)}")
        
        # 第四步：等待 batch job 完成
        print("等待 batch job 完成...")
        while True:
            batch_job = self.batch_client.batches.get(batch_job.name)
            print(f"当前状态: {batch_job.state}")
            
            if batch_job.state == "STATE_SUCCEEDED":
                print("Batch job 已完成！")
                break
            elif batch_job.state == "STATE_FAILED":
                # 清理已上传的文件
                for uploaded_file in uploaded_video_files:
                    try:
                        genai.delete_file(uploaded_file.name)
                    except:
                        pass
                raise RuntimeError(f"Batch job 失败: {batch_job.name}")
            
            time.sleep(check_interval)
        
        # 第五步：获取结果
        print("获取 batch job 结果...")
        results = []
        try:
            # 获取 batch job 的输出文件
            if hasattr(batch_job, 'output') and batch_job.output:
                # 下载并解析结果文件
                output_file = self.batch_client.files.get(batch_job.output)
                print(f"输出文件: {output_file.name}")
                
                # 下载文件内容
                if hasattr(output_file, 'uri') or hasattr(output_file, 'download_uri'):
                    # 使用文件 URI 下载内容
                    download_uri = getattr(output_file, 'download_uri', None) or getattr(output_file, 'uri', None)
                    if download_uri:
                        response = requests.get(download_uri, headers={'Authorization': f'Bearer {self.api_key}'})
                        if response.status_code == 200:
                            # 解析 JSONL 格式的结果
                            for line in response.text.strip().split('\n'):
                                if line.strip():
                                    result_data = json.loads(line)
                                    results.append(result_data)
                            print(f"成功解析 {len(results)} 条结果")
                        else:
                            print(f"下载结果文件失败，状态码: {response.status_code}")
                else:
                    # 尝试直接读取文件内容（如果 API 支持）
                    print("警告：无法获取输出文件的下载链接，请手动检查 batch job 结果")
            else:
                print("警告：batch job 没有输出文件")
        except Exception as e:
            print(f"获取结果时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 清理已上传的文件
        print("清理已上传的视频文件...")
        for uploaded_file in uploaded_video_files:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception as e:
                print(f"警告：删除上传的视频文件失败: {e}")
        
        return results


def main(args):
    
    
    """示例用法"""
    # 初始化生成器
    generator = GeminiDataGenerator(
        api_key=args.api_key,  # 在这里填入你的API密钥，或设置环境变量GEMINI_API_KEY
        model_name=args.model_name,
        temperature=0.7
    )
    
    video_annotation_list = []
    for activity in args.activity_list:
        for day in args.days_list:
            egolife_path = os.path.join(args.ego_life_path, activity, day)
            egolife_annotation_path = os.path.join(args.ego_life_path, "annotation_segments", activity, day)
            egolife_video_list = sorted(os.listdir(egolife_path))
            egolife_annotation_list = sorted(os.listdir(egolife_annotation_path))
            for video_file, annotation_file in zip(egolife_video_list, egolife_annotation_list):
                assert video_file.split(".")[0] == annotation_file.split(".")[0]
                video_path = os.path.join(egolife_path, video_file)
                annotation_path = os.path.join(egolife_annotation_path, annotation_file)
                video_annotation_list.append((video_path, annotation_path))
    print(f"共生成 {len(video_annotation_list)} 条数据")
    
    
    # 获取提示词
    system_prompt = PROMPTS["egolife_summarize"]
    
    # 生成数据
    try:
        if args.use_batch_api:
            # 使用 Batch API 批量生成
            print("使用 Batch API 模式...")
            
            # 准备所有提示词和视频路径
            user_prompts = []
            video_paths = []
            
            for video_path, annotation_path in video_annotation_list:
                # 读取注释文件
                with open(annotation_path, "r", encoding="utf-8") as f:
                    annotation_data = json.load(f)
                
                # 构建用户提示词
                annotation_str = json.dumps(annotation_data, ensure_ascii=False, indent=2)  # 或者 indent=2
                user_prompt = system_prompt.format(annotation_data=annotation_str)
                user_prompts.append(user_prompt)
                video_paths.append(video_path)
            
            # 使用 Batch API 生成
            batch_results = generator.generate_batch_api(
                user_prompts=user_prompts,
                video_paths=video_paths,
                display_name=f"egolife-summarize-batch",
                check_interval=args.batch_check_interval
            )
            
            # 整理结果格式
            results = []
            for idx, (video_path, annotation_path) in enumerate(video_annotation_list):
                result_item = {
                    "video_path": video_path,
                    "annotation_path": annotation_path,
                }
                if idx < len(batch_results):
                    # 根据实际返回格式提取生成文本
                    batch_result = batch_results[idx]
                    if isinstance(batch_result, dict):
                        if 'response' in batch_result:
                            response = batch_result['response']
                            if 'candidates' in response and response['candidates']:
                                candidate = response['candidates'][0]
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    text_parts = [p.get('text', '') for p in candidate['content']['parts'] if 'text' in p]
                                    result_item["generated_text"] = ''.join(text_parts)
                                elif 'text' in candidate:
                                    result_item["generated_text"] = candidate['text']
                        elif 'generated_text' in batch_result:
                            result_item["generated_text"] = batch_result['generated_text']
                    elif isinstance(batch_result, str):
                        result_item["generated_text"] = batch_result
                results.append(result_item)
            
            # 保存结果
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"所有结果已保存到: {args.output_path}")
            
        else:
            # 使用传统的逐个生成方式
            print("使用传统 API 模式...")
            results = []
            for idx, (video_path, annotation_path) in enumerate(video_annotation_list, 1):
                print(f"处理 {idx}/{len(video_annotation_list)}: {os.path.basename(video_path)}")
                
                # 读取注释文件（根据实际格式调整）
                with open(annotation_path, "r", encoding="utf-8") as f:
                    annotation_data = json.load(f)  # 或根据实际格式解析
                
                # 构建用户提示词（根据实际需求调整）
                user_prompt = f"请分析以下视频和注释数据：\n{json.dumps(annotation_data, ensure_ascii=False, indent=2)}"
                
                # 生成数据（包含视频）
                result = generator.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    video_path=video_path
                )
                results.append({
                    "video_path": video_path,
                    "annotation_path": annotation_path,
                    "generated_text": result
                })
                
                # 每处理一定数量后保存一次（可选）
                if idx % 10 == 0:
                    print(f"已处理 {idx} 个，保存中间结果...")
                    with open(args.output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存最终结果
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"所有结果已保存到: {args.output_path}")
        
    except Exception as e:
        print(f"生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument("--task", type=str, default="long_preference")
    parse.add_argument("--output_path", type=str, default="output/generated_data.json")
    # 对于egolife数据集的配置参数
    parse.add_argument("--ego_life_path", type=str, default="./data/egolife")
    parse.add_argument("--activity_list", type=list, default=["A1_JAKE"])
    parse.add_argument("--days_list", type=list, default=["DAY1", "DAY2", "DAY3", "DAY4", "DAY5", "DAY6", "DAY7"])
    parse.add_argument("--model_name", type=str, default="gemini-2.5-pro")
    parse.add_argument("--save_format", type=str, default="json")
    parse.add_argument("--temperature", type=float, default=0.7)
    parse.add_argument("--max_retries", type=int, default=3)
    parse.add_argument("--retry_delay", type=float, default=1.0)
    parse.add_argument("--api_key", type=str, default="")
    parse.add_argument("--use_batch_api", action="store_true", help="使用 Batch API 进行批量生成")
    parse.add_argument("--batch_check_interval", type=int, default=30, help="检查 batch job 状态的间隔（秒）")
    args = parse.parse_args()
    
    main(args)

