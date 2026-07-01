"""
将 CaptionCook4D 错误标注转换为对话的代码
使用 Gemini Batch API 处理错误标注数据
"""

import os
import json
import time
import argparse
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
import google.generativeai as genai
from google import genai as genai_batch


# Pydantic models for error dialogue output schema
class DialogueUtterance(BaseModel):
    """Dialogue utterance for error dialogue"""
    role: Literal["assistant", "user"] = Field(..., description="Role of the speaker")
    utterance: str = Field(..., description="The spoken message (natural, respectful, concise)")


class ServiceType(BaseModel):
    """Service type classification"""
    main: Literal["Instant Proactive Service", "Short-Term Proactive Service", "Episodic Proactive Service"] = Field(
        ..., 
        description="Main category of the proactive service"
    )
    sub: Literal[
        "Safety",
        "Tool Use",
        "Error-Recovery",
        "Next-Step Guidance",
        "Resource Reminder",
        "Episodic Task Reminder",
        "Episodic Memory Recall"
    ] = Field(..., description="Sub-category of the proactive service")


class ErrorDialogueItem(BaseModel):
    """Single error dialogue item"""
    step_id: float = Field(..., description="The step ID from the annotation")
    description: str = Field(..., description="The original or modified step description from annotation")
    start_time: float = Field(..., description="Start time of the step in seconds")
    end_time: float = Field(..., description="End time of the step in seconds")
    error_tag: str = Field(..., description="Error tag from the annotation")
    service_type: ServiceType = Field(..., description="Service type classification")
    dialogue: List[DialogueUtterance] = Field(
        ..., 
        min_length=1,
        max_length=2,
        description="Proactive dialogue between assistant and user (1-2 turns)"
    )


class ErrorDialogueOutput(BaseModel):
    """Error dialogue output schema"""
    items: List[ErrorDialogueItem] = Field(
        ..., 
        description="List of error dialogue items"
    )


# Use the wrapper class
OUTPUT_SCHEMA = ErrorDialogueOutput


SYSTEM_PROMPT = """
You are a proactive service annotation assistant for egocentric instructional videos.

You will be given structured step annotations from a task video.
Each step may include:
- a step description,
- start and end times,
- and one or more annotated errors (already identified by humans).

IMPORTANT:
You MUST NOT discover new errors.
You MUST ONLY generate proactive services for the errors that are explicitly annotated.

Your task is to:
For EACH annotated error,
(1) determine the correct proactive service type,
(2) generate a short proactive dialogue (1-2 turns),
based strictly on the provided annotations.

The dialogue should be:
- natural and respectful,
- supportive rather than commanding,
- concise and task-focused,
- appropriate for an instructional / hands-on scenario
  (e.g., cooking, assembling furniture, laboratory experiments).

------------------------------------------------------------
Proactive Service Types (MUST choose exactly ONE per error)
------------------------------------------------------------

### Instant Proactive Services (second-level, immediate, less than 10 seconds)

1. Safety  
Use ONLY if the annotated error could plausibly cause bodily harm
or an immediate accident.

Examples:
- risk of burns, cuts, electric shock,
- unsafe proximity to moving parts,
- unstable heavy objects,
- hazardous spills or heat.

RULE:
If an error could endanger the user's body,
it MUST be classified as Safety,
even if it also involves a tool or technique.

2. Tool Use  
Use when the error reflects:
- improper tool operation,
- unstable or unfamiliar handling,
- incorrect technique that does NOT yet require rollback.

Examples:
- incorrect pouring technique,
- unstable grip,
- poor control,
- lack of precision or consistency.

------------------------------------------------------------

### Short-Horizon Proactive Services (10 seconds to 10 minutes)

3. Error-Recovery  
Use ONLY if:
- the user has already completed a wrong step,
- the task must be corrected or redone,
- the error typically spans more than ~10 seconds
  and affects task correctness.

Examples:
- wrong measurement already added,
- wrong component assembled,
- incorrect configuration applied.

4. Next-Step Guidance  
Use if:
- the step itself is completed correctly,
- the user is transitioning to the next stage,
- guidance helps move the workflow forward.

5. Resource Reminder  
Use if:
- something is left unfinished or unresolved
  while transitioning to another step.

Examples:
- leftover material,
- missing cleanup,
- container not closed,
- power or heat not turned off.

------------------------------------------------------------
### Episodic Proactive Services (short-horizon memory within the same day + more than 10 minutes + less than 2.5 hours)

6. Episodic Task Reminder
Use if:
- the current error is best explained by an earlier unfinished commitment or step,
- the user previously started or committed to something earlier in the same session,
- there is no evidence of completion in the provided annotations,
- and the user is now moving on without resolving it.

Examples:
- earlier calibration/setup step was started but not finished,
- a required part-prep step was deferred and is now causing trouble,
- a previously mentioned prerequisite remains incomplete.

7. Episodic Memory Recall
Use if:
- the current error can be corrected by recalling something from earlier in the same session,
- such as where an item was placed, a prior measurement/setting, or an earlier instruction,
- and the annotation/context explicitly contains that earlier information.

Examples:
- recall where a tool/part was placed earlier,
- recall a previously stated measurement/setting,
- recall an earlier instruction that resolves the current confusion.

NOTE:
Use Episodic services ONLY when the provided annotations/context explicitly support
a same-session dependency. Do NOT speculate or invent prior events.

------------------------------------------------------------
STRICT Classification Rules
------------------------------------------------------------

- Any error that may cause bodily injury → Safety.
- Unstable or unskilled operation → Tool Use.
- Errors that require rollback or correction → Error-Recovery.
- Do NOT downgrade Safety to Tool Use.
- Do NOT use Error-Recovery unless correction is necessary.

------------------------------------------------------------
Dialogue Generation Rules
------------------------------------------------------------

For EACH error:
- Generate a short proactive dialogue of 1-2 turns.
- The assistant speaks first.
- The tone must be polite, calm, and respectful.
- Do NOT scold, command, or sound judgmental.
- Do NOT mention "error", "mistake", or "annotation".
- The dialogue should naturally fit into the task flow.

------------------------------------------------------------
Output Format (STRICT JSON)
------------------------------------------------------------

For each error, output one JSON object:
[
  {
    "step_id": <number>,

    "description": "<the original or modified step description from annotation>",

    "start_time": <float>,
    "end_time": <float>,

    "error_tag": "<tag from annotation>",

    "service_type": {
    "main": "Instant Proactive Service"
        | "Short-Term Proactive Service"
        | "Episodic Proactive Service",
    "sub": "Safety"
        | "Tool Use"
        | "Error-Recovery"
        | "Next-Step Guidance"
        | "Resource Reminder"
        | "Episodic Task Reminder"
        | "Episodic Memory Recall"
    },

    "dialogue": [
    {
        "role": "assistant",
        "utterance": "<first proactive message (natural, respectful, concise)>"
    },
    {
        "role": "user",
        "utterance": "<optional short user reply>"
    }
    ]
  }
]

------------------------------------------------------------
Input
------------------------------------------------------------

You will receive step annotations in JSON format.

Each step may contain:
- description
- step_id
- errors (with tag and description)
- start_time, end_time
- modified_description (optional)

ONLY generate outputs for steps that contain errors.
Do NOT generate output for correct steps.
"""


class ErrorToDialogueTransformer:
    """将错误标注转换为对话的转换器类"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        check_interval: int = 30,
    ):
        """
        初始化转换器
        
        Args:
            api_key: Gemini API密钥，如果为None则从环境变量GEMINI_API_KEY读取
            model_name: 使用的模型名称
            temperature: 生成温度，控制随机性
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            check_interval: 检查batch job状态的间隔（秒）
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API密钥未提供，请设置api_key参数或GEMINI_API_KEY环境变量")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.check_interval = check_interval
        
        # 配置API密钥
        genai.configure(api_key=self.api_key)
        
        # 初始化 batch API 客户端
        self.batch_client = genai_batch.Client(api_key=self.api_key)
    
    def load_error_annotations(self, json_path: str) -> List[Dict[str, Any]]:
        """
        加载错误标注文件，提取所有 is_error=True 的条目
        
        Args:
            json_path: error_annotations.json 文件路径
            
        Returns:
            包含错误标注的列表，每个元素包含 recording_id, activity_id, step_annotations
        """
        print(f"正在加载错误标注文件: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            all_annotations = json.load(f)
        
        # 筛选出所有 is_error=True 的条目
        error_annotations = [
            {
                "recording_id": item["recording_id"],
                "activity_id": item["activity_id"],
                "step_annotations": item["step_annotations"]
            }
            for item in all_annotations
            if item.get("is_error", False)
        ]
        
        print(f"找到 {len(error_annotations)} 个错误标注条目")
        return error_annotations
    
    def format_step_annotations(self, step_annotations: List[Dict[str, Any]]) -> str:
        """
        将 step_annotations 格式化为文本，用于构建 prompt
        
        Args:
            step_annotations: 步骤标注列表
            
        Returns:
            格式化后的文本字符串
        """
        formatted_text = "步骤标注信息：\n"
        for idx, step in enumerate(step_annotations, 1):
            formatted_text += f"\n步骤 {idx}:\n"
            formatted_text += f"  - 描述: {step.get('description', 'N/A')}\n"
            formatted_text += f"  - 步骤ID: {step.get('step_id', 'N/A')}\n"
            formatted_text += f"  - 开始时间: {step.get('start_time', 'N/A')} 秒\n"
            formatted_text += f"  - 结束时间: {step.get('end_time', 'N/A')} 秒\n"
        
        return formatted_text
    
    def build_prompt(self, step_annotations: List[Dict[str, Any]]) -> str:
        """
        构建发送给 Gemini 的 prompt
        
        Args:
            step_annotations: 步骤标注列表
            recording_id: 录制ID
            activity_id: 活动ID
            
        Returns:
            完整的 prompt 字符串
        """
        # 格式化步骤标注
        # formatted_steps = self.format_step_annotations(step_annotations)
        system_prompt = SYSTEM_PROMPT + "\n\nInput Annotations:\n" + json.dumps(step_annotations, ensure_ascii=False, indent=2)
        
        return system_prompt
    
    def transform_to_dialogue(
        self,
        error_annotations: List[Dict[str, Any]],
        batch_size: int = 10,
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用 Gemini Batch API 将错误标注转换为对话
        
        Args:
            error_annotations: 错误标注列表
            batch_size: 每批处理的条目数量
            output_path: 输出文件路径，如果为None则不保存
            
        Returns:
            转换后的对话列表
        """
        print(f"开始处理 {len(error_annotations)} 个错误标注条目...")
        
        all_results = []
        
        # 将错误标注分批处理
        for batch_idx in range(0, len(error_annotations), batch_size):
            batch = error_annotations[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(error_annotations) + batch_size - 1) // batch_size
            
            print(f"\n处理第 {batch_num}/{total_batches} 批（{len(batch)} 个条目）...")
            
            # 为每个条目构建 prompt 和 batch 请求
            batch_requests = []
            for item in batch:
                prompt = self.build_prompt(item["step_annotations"])
                
                batch_request = {
                    'contents': [{
                        'parts': [
                            {'text': prompt}
                        ],
                        'role': 'user'
                    }],
                    'config': {
                        'response_mime_type': 'application/json',
                        'response_schema': OUTPUT_SCHEMA.model_json_schema(),
                        # 'temperature': self.temperature,
                    }
                }
                batch_requests.append(batch_request)
            
            # 调用 Batch API
            batch_results = self._call_batch_api(
                batch_requests,
                display_name=f"error-to-dialogue-batch-{batch_num}"
            )
            
            # 将结果与原始数据关联
            for item, result in zip(batch, batch_results):
                result_item = {
                    "recording_id": item["recording_id"],
                    "activity_id": item["activity_id"],
                    "step_annotations": item["step_annotations"],
                    "dialogue": result
                }
                all_results.append(result_item)
            
            print(f"第 {batch_num} 批处理完成")
        
        # 保存结果
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_path}")
        
        return all_results
    
    def _call_batch_api(
        self,
        batch_requests: List[Dict[str, Any]],
        display_name: str = "error-to-dialogue-job"
    ) -> List[Any]:
        """
        调用 Gemini Batch API
        
        Args:
            batch_requests: batch 请求列表
            display_name: 显示名称
            
        Returns:
            API 响应列表
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"调用 Gemini Batch API (尝试 {attempt + 1}/{self.max_retries})...")
                
                # 创建 batch job
                batch_job = self.batch_client.batches.create(
                    model=f"models/{self.model_name}",
                    src=batch_requests,
                    config={
                        'display_name': display_name,
                    },
                )
                print(f"Batch job 已创建: {batch_job.name}")
                
                # 等待 batch job 完成
                print("等待 batch job 完成...")
                batch_job_name = batch_job.name
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
                    
                    time.sleep(self.check_interval)
                
                # 获取结果
                if batch_job.dest is None:
                    raise RuntimeError("Batch job 没有输出目标 (dest is None)")
                if not hasattr(batch_job.dest, 'inlined_responses'):
                    raise RuntimeError(f"Batch job 输出目标不支持 inlined_responses: {type(batch_job.dest)}")
                if batch_job.dest.inlined_responses is None or len(batch_job.dest.inlined_responses) == 0:
                    raise RuntimeError("Batch job 的 inlined_responses 为空")
                
                # 提取所有响应
                results = []
                for inline_response in batch_job.dest.inlined_responses:
                    if inline_response.error:
                        raise RuntimeError(f"Batch job 响应包含错误: {inline_response.error}")
                    
                    if not inline_response.response:
                        raise RuntimeError("Batch job 响应为空")
                    
                    # 提取文本内容并解析JSON
                    if inline_response.response.candidates:
                        candidate = inline_response.response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            text = candidate.content.parts[0].text
                            # 解析JSON响应（因为使用了response_schema和response_mime_type='application/json'）
                            try:
                                parsed_result = json.loads(text)
                                results.append(parsed_result)
                            except json.JSONDecodeError as e:
                                print(f"警告: JSON解析失败，原始文本: {text[:200]}...")
                                raise RuntimeError(f"无法解析JSON响应: {e}")
                        else:
                            raise RuntimeError("Batch job 响应内容为空")
                    else:
                        raise RuntimeError("Batch job 响应没有候选结果")
                
                return results
                
            except Exception as e:
                last_error = e
                print(f"尝试 {attempt + 1} 失败: {e}")
                if attempt < self.max_retries - 1:
                    print(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"所有重试都失败，最后错误: {last_error}")
        
        raise RuntimeError(f"无法完成 batch API 调用: {last_error}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将 CaptionCook4D 错误标注转换为对话")
    parser.add_argument(
        "--input_path",
        type=str,
        default="./data/captioncook4d/annotations/annotation_json/error_annotations.json",
        help="输入的错误标注JSON文件路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/gemini_generation/error_to_dialogue_results.json",
        help="输出的对话结果JSON文件路径"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API密钥（如果不提供，将从环境变量GEMINI_API_KEY读取）"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-3-pro-preview",
        help="使用的Gemini模型名称"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="每批处理的条目数量"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度"
    )
    
    args = parser.parse_args()
    
    # 创建转换器
    transformer = ErrorToDialogueTransformer(
        api_key=args.api_key,
        model_name=args.model_name,
        temperature=args.temperature,
    )
    
    # 加载错误标注
    error_annotations = transformer.load_error_annotations(args.input_path)
    
    if len(error_annotations) == 0:
        print("没有找到错误标注，退出")
        return
    
    # 转换为对话
    results = transformer.transform_to_dialogue(
        error_annotations,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )
    
    print(f"\n处理完成！共生成 {len(results)} 个对话结果")


if __name__ == "__main__":
    main()


