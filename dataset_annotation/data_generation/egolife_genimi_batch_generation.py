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
from egolife_prompt_total import PROMPTS
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
        Days: list = None,
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
        
        self.days = Days
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
        output_path: Optional[str] = None,
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
            if self.low_level_category == "memory_link_contextual":
                from egolife_output_schema.long_term_service import LongHorizonMemoryLinkProactiveServiceOutput
                output_schema = LongHorizonMemoryLinkProactiveServiceOutput
            elif self.low_level_category == "habit_coaching":
                from egolife_output_schema.long_term_service import HabitCoachingProactiveServiceOutput
                output_schema = HabitCoachingProactiveServiceOutput
            elif self.low_level_category == "routine_optimization":
                from egolife_output_schema.long_term_service import RoutineOptimizationProactiveServiceOutput
                output_schema = RoutineOptimizationProactiveServiceOutput
            elif self.low_level_category == "personal_progressive":
                from egolife_output_schema.long_term_service import PersonalProgressFeedbackProactiveServiceOutput
                output_schema = PersonalProgressFeedbackProactiveServiceOutput
            else:
                raise ValueError(f"不支持的输出schema: {self.high_level_category}_{self.low_level_category}")
            
        elif self.high_level_category == "egolife_episodic":
            if self.low_level_category == "memory_recall":
                from egolife_output_schema.episodic_service import EpisodicMemoryRecallProactiveServiceOutput
                output_schema = EpisodicMemoryRecallProactiveServiceOutput
            elif self.low_level_category == "task_reminder":
                from egolife_output_schema.episodic_service import EpisodicTaskReminderProactiveServiceOutput
                output_schema = EpisodicTaskReminderProactiveServiceOutput
            else:
                raise ValueError(f"不支持的输出schema: {self.high_level_category}_{self.low_level_category}")
            
        elif self.high_level_category == "egolife_short_term":
            if self.low_level_category == "error_recovery":
                from egolife_output_schema.short_term_service import ErrorRecoveryProactiveServiceOutput
                output_schema = ErrorRecoveryProactiveServiceOutput
            elif self.low_level_category == "next_step_guidance":
                from egolife_output_schema.short_term_service import NextStepGuidanceProactiveServiceOutput
                output_schema = NextStepGuidanceProactiveServiceOutput
            elif self.low_level_category == "resource_reminder":
                from egolife_output_schema.short_term_service import ShortTermResourceReminderProactiveServiceOutput
                output_schema = ShortTermResourceReminderProactiveServiceOutput
            else:
                raise ValueError(f"不支持的输出schema: {self.high_level_category}_{self.low_level_category}")
            
        elif self.high_level_category == "egolife_instant":
            if self.low_level_category == "safety":
                from egolife_output_schema.instant_service import SafetyProactiveServiceOutput
                output_schema = SafetyProactiveServiceOutput
            elif self.low_level_category == "tool_use":
                from egolife_output_schema.instant_service import ToolUseProactiveServiceOutput
                output_schema = ToolUseProactiveServiceOutput
            else:
                raise ValueError(f"不支持的输出schema: {self.high_level_category}_{self.low_level_category}")   
            
        else:
            # 对于非长期服务类型，需要根据具体类型导入对应的schema
            # 这里暂时抛出错误，如果后续需要支持其他类型，可以在这里添加
            raise ValueError(f"不支持的输出schema: {self.high_level_category}类型。")
        
        # 1. 读取所有 JSON 文件内容并转换为字符串，将每2h的文件组成一个列表
        json_contents = {}  # 字典，key 为 "DAY1_11000000-12000000"，value 为 JSON 字符串列表
        
        def timestamp_to_minutes(timestamp):
            """将时间戳(HHMMSSMM格式)转换为总分钟数"""
            hour = timestamp // 1000000
            minute = (timestamp // 10000) % 100
            return hour * 60 + minute
        
        # 先按照日期划分
        days_dict = {}
        for day in self.days:
            days_dict[day] = []
        for result_file, timestamp_tuple in result_files:
            date = result_file.split("/")[-2]
            days_dict[date].append((result_file, timestamp_tuple))
        
        # 对每天的文件按时间戳分组（每2小时一组）
        for day, day_files in days_dict.items():
            # 按 start_time 排序
            day_files.sort(key=lambda x: x[1][0])
            
            current_group = []
            group_start_time = None  # 当前组的起始时间（分钟）
            group_end_time = None    # 当前组的结束时间（分钟）
            group_start_timestamp = None  # 当前组的起始时间戳（原始格式）
            group_end_timestamp = None    # 当前组的结束时间戳（原始格式）
            
            # 使用索引遍历，以便知道是否是最后一个文件和下一个文件的信息
            for file_idx, (result_file, timestamp_tuple) in enumerate(day_files):
                start_time, end_time = timestamp_tuple
                start_minutes = timestamp_to_minutes(start_time)
                end_minutes = timestamp_to_minutes(end_time)
                
                # 判断是否是最后一个文件
                is_last_file = (file_idx == len(day_files) - 1)
                # 获取下一个文件的信息（如果存在）
                next_file_info = None
                next_start_minutes = None
                time_gap = None  # 当前文件结束时间和下一个文件开始时间的间隔（分钟）
                if not is_last_file:
                    next_file_info = day_files[file_idx + 1]
                    next_start_time, _ = next_file_info[1]
                    next_start_minutes = timestamp_to_minutes(next_start_time)
                    time_gap = next_start_minutes - end_minutes
                
                # 先读取当前文件
                if not os.path.exists(result_file):
                    raise FileNotFoundError(f"JSON 文件不存在: {result_file}")
                
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                        json_data = {result_file.split("/")[-1].split(".")[0]: json_data}
                        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                except Exception as e:
                    raise RuntimeError(f"读取 JSON 文件失败: {result_file}, 错误: {e}")
                
                # 如果当前组为空，直接开始新组
                if group_start_time is None:
                    group_start_time = start_minutes
                    group_end_time = end_minutes
                    group_start_timestamp = start_time
                    group_end_timestamp = end_time
                    current_group.append(json_str)
                    # print(f"  已读取 JSON 文件: {os.path.basename(result_file)}, 时间跨度: {group_start_time//60}:{group_start_time%60:02d} - {group_end_time//60}:{group_end_time%60:02d}, 文件数: {len(current_group)}")
                    continue
                
                # 检查时间是否断开（前一个的 end_time < 后一个的 start_time）
                time_disconnected = (group_end_time < start_minutes)
                
                # 计算加上当前文件后的时间跨度（从组开始到当前文件结束）
                new_group_duration = end_minutes - group_start_time
                
                # 判断是否可以加入当前组
                should_save_group = False
                can_add_to_current_group = False
                
                if time_disconnected:
                    # 时间断开，应该先保存当前组
                    # 但如果是最后一个文件，或者时间间隔很大，可以加入当前组
                    if is_last_file:
                        # 最后一个文件，允许加入（即使超过2小时）
                        can_add_to_current_group = True
                    elif time_gap is not None and time_gap > 0:
                        # 时间断开且间隔较大，允许加入（即使超过2小时）
                        can_add_to_current_group = True
                    else:
                        # 时间断开且间隔不大，保存当前组，开始新组
                        should_save_group = True
                else:
                    # 时间连续或重叠
                    if new_group_duration <= 120:
                        # 不超过2小时，可以加入当前组
                        can_add_to_current_group = True
                    else:
                        # 超过2小时，检查是否允许
                        if is_last_file:
                            # 最后一个文件，允许加入
                            can_add_to_current_group = True
                        elif time_gap is not None and time_gap > 0:
                            # 下一个文件时间间隔很大，允许加入
                            can_add_to_current_group = True
                        else:
                            # 不允许超过2小时，保存当前组并开始新组
                            should_save_group = True
                
                if can_add_to_current_group:
                    # 加入当前组
                    group_end_time = max(group_end_time, end_minutes)
                    group_end_timestamp = max(group_end_timestamp, end_time)
                    current_group.append(json_str)
                    group_duration = group_end_time - group_start_time
                    if group_duration > 120:
                        print(f"  已读取 JSON 文件: {os.path.basename(result_file)}, 时间跨度: {group_start_time//60}:{group_start_time%60:02d} - {group_end_time//60}:{group_end_time%60:02d}, 文件数: {len(current_group)} (超过2h但允许)")
                    else:
                        print(f"  已读取 JSON 文件: {os.path.basename(result_file)}, 时间跨度: {group_start_time//60}:{group_start_time%60:02d} - {group_end_time//60}:{group_end_time%60:02d}, 文件数: {len(current_group)}")
                    continue
                
                # 如果需要保存组
                if should_save_group:
                    group_key = f"{day}_{group_start_timestamp}-{group_end_timestamp}"
                    json_contents[group_key] = current_group
                    current_group = []
                    group_start_time = start_minutes
                    group_end_time = end_minutes
                    group_start_timestamp = start_time
                    group_end_timestamp = end_time
                    # 将当前文件加入新组
                    current_group.append(json_str)
                    print(f"  已读取 JSON 文件: {os.path.basename(result_file)}, 时间跨度: {group_start_time//60}:{group_start_time%60:02d} - {group_end_time//60}:{group_end_time%60:02d}, 文件数: {len(current_group)} (新组)")
            
            # 处理最后一组（如果存在）
            if current_group:
                group_key = f"{day}_{group_start_timestamp}-{group_end_timestamp}"
                json_contents[group_key] = current_group
        
        # 2. 构建系统提示词
        if self.high_level_category == "egolife_long_term":
            system_prompt = PROMPTS["long_term"][self.low_level_category]
        elif self.high_level_category == "egolife_episodic":
            system_prompt = PROMPTS["episodic"][self.low_level_category]
        elif self.high_level_category == "egolife_short_term":
            system_prompt = PROMPTS["short_term"][self.low_level_category]
        elif self.high_level_category == "egolife_instant":
            system_prompt = PROMPTS["instant"][self.low_level_category]
        else:
            raise ValueError(f"不支持的输出schema: {self.high_level_category}_{self.low_level_category}")
        
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
            "temperature": 1.0,  # 设置为0以确保生成结果的一致性
            "response_mime_type": "application/json",
            "response_json_schema": json_schema,
            # **(generation_config or {})
        }
        
        # 4. 开始递归生成数据
        if self.high_level_category == "egolife_long_term":
            
            # 检查是否存在已保存的部分结果文件，如果存在则加载
            accumulated_results = None
            conversation_results = None
            
            # 用于记录需要跳过的最后一个group
            last_processed_group_key = None
            
            if output_path:
                accumulated_partial_path = os.path.join(output_path, "accumulated_results_partial.json")
                conversation_partial_path = os.path.join(output_path, "conversation_results_partial.json")
                
                if os.path.exists(accumulated_partial_path) and os.path.exists(conversation_partial_path):
                    print(f"\n检测到已保存的部分结果文件，正在加载作为记忆...")
                    try:
                        with open(accumulated_partial_path, "r", encoding="utf-8") as f:
                            accumulated_results = json.load(f)
                        with open(conversation_partial_path, "r", encoding="utf-8") as f:
                            conversation_results = json.load(f)
                        print(f"✓ 成功加载已保存的结果文件:")
                        print(f"  - {accumulated_partial_path}")
                        print(f"  - {conversation_partial_path}")
                        print(f"将使用这些已保存的结果作为历史记忆继续处理...\n")
                        
                        # 提取所有已处理的segment_id
                        processed_segment_ids = set()
                        if self.low_level_category == "habit_coaching" and "habit_events" in accumulated_results:
                            for event in accumulated_results.get("habit_events", []):
                                for occurrence in event.get("occurrences", []):
                                    if "segment_id" in occurrence:
                                        processed_segment_ids.add(occurrence["segment_id"])
                        elif self.low_level_category == "memory_link_contextual" and "memory_events" in accumulated_results:
                            for event in accumulated_results.get("memory_events", []):
                                for occurrence in event.get("occurrences", []):
                                    if "segment_id" in occurrence:
                                        processed_segment_ids.add(occurrence["segment_id"])
                        elif self.low_level_category == "routine_optimization" and "routine_events" in accumulated_results:
                            for event in accumulated_results.get("routine_events", []):
                                for occurrence in event.get("occurrences", []):
                                    if "segment_id" in occurrence:
                                        processed_segment_ids.add(occurrence["segment_id"])
                        elif self.low_level_category == "personal_progressive" and "progress_events" in accumulated_results:
                            for event in accumulated_results.get("progress_events", []):
                                for occurrence in event.get("occurrences", []):
                                    if "segment_id" in occurrence:
                                        processed_segment_ids.add(occurrence["segment_id"])
                        
                        # 找到最后一个处理过的group
                        # 从processed_segment_ids中提取day和time，找到最大的那个segment_id，然后找到它属于哪个group
                        if processed_segment_ids:
                            print(f"已处理的segment数量: {len(processed_segment_ids)}")
                            
                            # 解析segment_id，提取day和timestamp，找到最大的
                            # segment_id格式: "A1_JAKE_DAY1_11090000"
                            def parse_segment_id(segment_id):
                                """解析segment_id，返回(day_num, timestamp)"""
                                try:
                                    # 格式: "A1_JAKE_DAY1_11090000"
                                    parts = segment_id.split("_")
                                    if len(parts) >= 4:
                                        day_str = parts[2]  # "DAY1"
                                        day_num = int(day_str.replace("DAY", ""))
                                        timestamp = int(parts[3])  # "11090000"
                                        return (day_num, timestamp)
                                except Exception as e:
                                    print(f"警告: 解析segment_id失败: {segment_id}, 错误: {e}")
                                return None
                            
                            # 找到最大的segment_id（先比较day，再比较timestamp）
                            max_segment_id = None
                            max_day = None
                            max_timestamp = None
                            
                            for segment_id in processed_segment_ids:
                                parsed = parse_segment_id(segment_id)
                                if parsed:
                                    day_num, timestamp = parsed
                                    if max_segment_id is None:
                                        max_segment_id = segment_id
                                        max_day = day_num
                                        max_timestamp = timestamp
                                    else:
                                        # 如果day更大，或者day相同但timestamp更大，则更新
                                        if day_num > max_day or (day_num == max_day and timestamp > max_timestamp):
                                            max_segment_id = segment_id
                                            max_day = day_num
                                            max_timestamp = timestamp
                            
                            # 找到max_segment_id属于哪个group
                            if max_segment_id:
                                print(f"最后一个处理的segment: {max_segment_id} (DAY{max_day}, timestamp: {max_timestamp})")
                                
                                # 遍历所有groups，找到包含max_segment_id的group
                                for group_key in sorted(json_contents.keys()):
                                    json_group = json_contents[group_key]
                                    # 检查这个group中是否包含max_segment_id
                                    for json_str in json_group:
                                        try:
                                            json_data = json.loads(json_str)
                                            # 获取第一个key作为segment_id
                                            segment_id = list(json_data.keys())[0] if json_data else None
                                            if segment_id == max_segment_id:
                                                last_processed_group_key = group_key
                                                break
                                        except Exception as e:
                                            continue
                                    
                                    # 如果找到了，就不需要继续查找
                                    if last_processed_group_key == group_key:
                                        break
                                
                                if last_processed_group_key:
                                    print(f"✓ 检测到上次处理到: {last_processed_group_key}")
                                    print(f"将从下一个group开始继续处理...\n")
                                else:
                                    print(f"⚠ 无法找到segment {max_segment_id} 所属的group，将从第一个group开始处理\n")
                            else:
                                print(f"⚠ 无法解析已处理的segment_id，将从第一个group开始处理\n")
                        else:
                            print(f"⚠ 未找到已处理的segment_id，将从第一个group开始处理\n")
                            
                    except Exception as e:
                        print(f"警告: 加载已保存的结果文件时出错: {e}")
                        print(f"将使用新的初始化结果")
                        accumulated_results = None
                        conversation_results = None
            
            # 如果没有加载到已保存的结果，则初始化新的结果字典
            if accumulated_results is None or conversation_results is None:
                # 为不同子服务类型构造字典，用于存储结果
                if self.low_level_category == "habit_coaching":
                    accumulated_results = {"service_main_type": "Long-Term Proactive Service", 
                                           "service_sub_type": "Habit-Coaching Proactive Service", 
                                           "habit_events": [],
                                           "historical_habit_triggers": []}
                    conversation_results = {"service_main_type": "Long-Term Proactive Service", 
                                           "service_sub_type": "Habit-Coaching Proactive Service", 
                                           "habit_triggers": []}
                elif self.low_level_category == "memory_link_contextual":
                    accumulated_results = {"service_main_type": "Long-Term Proactive Service", 
                                           "service_sub_type": "Memory Link Contextual Proactive Service", 
                                           "memory_events": [],
                                           "historical_realized_memory_links": []}
                    conversation_results = {"service_main_type": "Long-Term Proactive Service", 
                                           "service_sub_type": "Memory Link Contextual Proactive Service", 
                                           "realized_memory_links": []}
                elif self.low_level_category == "routine_optimization":
                    accumulated_results = {"service_main_type": "Long-Term Proactive Service", 
                                           "service_sub_type": "Routine Optimization Proactive Service", 
                                           "routine_events": [],
                                           "historical_optimization_triggers": []}
                    conversation_results = {"service_main_type": "Long-Term Proactive Service", 
                                           "service_sub_type": "Routine Optimization Proactive Service", 
                                           "optimization_triggers": []}
                elif self.low_level_category == "personal_progressive":
                    accumulated_results = {"service_main_type": "Long-Term Proactive Service", 
                                           "service_sub_type": "Personal Progressive Feedback Proactive Service", 
                                           "progress_events": [],
                                           "historical_feedback_triggers": []}
                    conversation_results = {"service_main_type": "Long-Term Proactive Service", 
                                           "service_sub_type": "Personal Progressive Feedback Proactive Service", 
                                           "feedback_triggers": []}
                       
                       
            # 开始处理groups，如果找到了上次处理的位置，则跳过已处理的groups
            skip_until_found = (last_processed_group_key is not None)
            for group_idx, (group_key, json_group) in enumerate(json_contents.items(), 1):
                # 如果需要跳过，检查是否到达了上次处理的位置
                if skip_until_found:
                    if group_key == last_processed_group_key:
                        print(f"\n跳过已处理的group: {group_key} (第 {group_idx}/{len(json_contents)} 组)")
                        skip_until_found = False  # 找到后，下一个group开始处理
                        continue
                    else:
                        print(f"\n跳过已处理的group: {group_key} (第 {group_idx}/{len(json_contents)} 组)")
                        continue
                
                print(f"\n处理第 {group_idx}/{len(json_contents)} 组数据 ({group_key})...")
                
                # 将所有 JSON 内容与 system_prompt 拼接
                # 将每个 JSON 文件内容格式化为带标题的文本
                annotations_text = "\n\n".join([
                    f"\n{content}" 
                    for i, content in enumerate(json_group)
                ])
                json_content_text = f"\n\n ### Segment-level Episodic Annotations Batch: {group_key} ###\n\n{annotations_text}"
            
                # 根据服务类型拼接完整的提示词
                json_content_text = json_content_text + f"\n\n ### Historical Habit Summary JSON ### \n\n {json.dumps(accumulated_results, ensure_ascii=False, indent=2)}"
                                
                full_prompt = f"{system_prompt}{json_content_text}"
                
                print(f"完整提示词长度: {len(full_prompt)} 字符")

                # 5. 使用 Batch API 生成内容（即使只有一个请求也使用batch API）
                last_error = None
                for attempt in range(self.max_retries):
                    try:
                        print(f"调用 Gemini Batch API (尝试 {attempt + 1}/{self.max_retries})...")
                        
                        # 构建 batch 请求（即使只有一个请求）
                        batch_request = {
                            'contents': [{
                                'parts': [
                                    {'text': full_prompt}
                                ],
                                'role': 'user'
                            }],
                            'config': {
                                'response_mime_type': 'application/json',
                                'response_schema': json_schema,  # 使用之前定义的 json_schema
                                'temperature': generation_config_params.get('temperature', 1.0),
                            }
                        }
                        
                        # 创建 batch job
                        batch_job = self.batch_client.batches.create(
                            model=f"models/{self.model_name}",
                            src=[batch_request],  # 即使只有一个请求也使用列表
                            config={
                                'display_name': f"{display_name}-group-{group_idx}",
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
                            
                            time.sleep(check_interval)
                        
                        # 获取结果
                        if batch_job.dest is None:
                            raise RuntimeError("Batch job 没有输出目标 (dest is None)")
                        if not hasattr(batch_job.dest, 'inlined_responses'):
                            raise RuntimeError(f"Batch job 输出目标不支持 inlined_responses: {type(batch_job.dest)}")
                        if batch_job.dest.inlined_responses is None or len(batch_job.dest.inlined_responses) == 0:
                            raise RuntimeError("Batch job 的 inlined_responses 为空")
                        
                        # 提取生成的文本（只有一个响应）
                        inline_response = batch_job.dest.inlined_responses[0]
                        if inline_response.error:
                            raise RuntimeError(f"Batch job 响应包含错误: {inline_response.error}")
                        
                        if not inline_response.response:
                            raise RuntimeError("Batch job 响应为空")
                        
                        # 提取文本内容
                        if hasattr(inline_response.response, 'text'):
                            result_text = inline_response.response.text
                        else:
                            raise ValueError("无法从响应中提取文本内容")
                        
                        print(f"第 {group_idx} 组数据生成成功，结果长度: {len(result_text)} 字符")
                        break
                        
                    except Exception as e:
                        last_error = e
                        error_str = str(e)
                        is_503_error = "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower()
                        is_429_error = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower() or "resource has been exhausted" in error_str.lower()
                        
                        if attempt < self.max_retries - 1:
                            # 对于 429 错误（配额用尽），使用最长的延迟时间
                            if is_429_error:
                                # 429 错误使用更长的延迟：60秒、120秒、240秒（配额问题需要等待更久）
                                wait_time = 60 * (2 ** attempt)
                                print(f"API配额用尽 (429错误)，等待 {wait_time:.1f}秒后重试 (尝试 {attempt + 1}/{self.max_retries})...")
                                print(f"提示: 如果配额已完全用尽，可能需要等待配额重置或增加配额")
                            # 对于 503 错误（模型过载），使用较长的延迟时间
                            elif is_503_error:
                                # 503 错误使用更长的延迟：30秒、60秒、120秒
                                wait_time = 30 * (2 ** attempt)
                                print(f"模型过载 (503错误)，等待 {wait_time:.1f}秒后重试 (尝试 {attempt + 1}/{self.max_retries})...")
                            else:
                                # 其他错误使用正常的指数退避
                                wait_time = self.retry_delay * (2 ** attempt)
                                print(f"生成失败，{wait_time:.1f}秒后重试 (尝试 {attempt + 1}/{self.max_retries})...")
                            time.sleep(wait_time)
                        else:
                            # 所有错误在重试次数用尽时都保存当前结果
                            if is_429_error:
                                error_type = "API配额用尽 (429错误)"
                            elif is_503_error:
                                error_type = "模型过载 (503错误)"
                            else:
                                error_type = "生成错误"
                            print(f"\n警告: {error_type}，已重试{self.max_retries}次仍失败。保存当前已处理的结果...")
                            if is_429_error:
                                print(f"提示: API配额可能已用尽，请检查配额限制或等待配额重置后再继续")
                            if output_path:
                                os.makedirs(output_path, exist_ok=True)
                                if accumulated_results:
                                    save_path = os.path.join(output_path, "accumulated_results_partial.json")
                                    with open(save_path, "w", encoding="utf-8") as f:
                                        json.dump(accumulated_results, f, ensure_ascii=False, indent=2)
                                    print(f"已保存部分结果到: {save_path}")
                                if conversation_results:
                                    save_path = os.path.join(output_path, "conversation_results_partial.json")
                                    with open(save_path, "w", encoding="utf-8") as f:
                                        json.dump(conversation_results, f, ensure_ascii=False, indent=2)
                                    print(f"已保存部分结果到: {save_path}")
                            print(f"返回已处理的结果，共处理了 {group_idx - 1} 组数据")
                            print(f"错误详情: {str(last_error)}")
                            return accumulated_results, conversation_results

                # 清理 Markdown 代码块标记（如果有）
                result_dict = json.loads(result_text)
                if self.low_level_category == "habit_coaching":
                    if accumulated_results["habit_events"]:
                        accumulated_results["habit_events"].extend(result_dict["habit_state_updates"])
                        accumulated_results["historical_habit_triggers"].extend(result_dict["habit_triggers"])
                    else:
                        accumulated_results["habit_events"] = result_dict["habit_state_updates"]
                        accumulated_results["historical_habit_triggers"] = result_dict["habit_triggers"]
                    # conversation_results 使用 historical_habit_triggers 的内容，避免重复
                    conversation_results["habit_triggers"] = accumulated_results["historical_habit_triggers"].copy()
                elif self.low_level_category == "memory_link_contextual":
                    if accumulated_results["memory_events"]:
                        accumulated_results["memory_events"].extend(result_dict["new_memory_candidates"])
                        accumulated_results["historical_realized_memory_links"].extend(result_dict["realized_memory_links"])
                    else:
                        accumulated_results["memory_events"] = result_dict["new_memory_candidates"]
                        accumulated_results["historical_realized_memory_links"] = result_dict["realized_memory_links"]
                    # conversation_results 使用 historical_realized_memory_links 的内容，避免重复
                    conversation_results["realized_memory_links"] = accumulated_results["historical_realized_memory_links"].copy()
                elif self.low_level_category == "routine_optimization":
                    if accumulated_results["routine_events"]:
                        accumulated_results["routine_events"].extend(result_dict["routine_state_updates"])
                        accumulated_results["historical_optimization_triggers"].extend(result_dict["optimization_triggers"])
                    else:
                        accumulated_results["routine_events"] = result_dict["routine_state_updates"]
                        accumulated_results["historical_optimization_triggers"] = result_dict["optimization_triggers"]
                    # conversation_results 使用 historical_optimization_triggers 的内容，避免重复
                    conversation_results["optimization_triggers"] = accumulated_results["historical_optimization_triggers"].copy()
                elif self.low_level_category == "personal_progressive":
                    if accumulated_results["progress_events"]:
                        accumulated_results["progress_events"].extend(result_dict["progress_state_updates"])
                        accumulated_results["historical_feedback_triggers"].extend(result_dict["feedback_triggers"])
                    else:
                        accumulated_results["progress_events"] = result_dict["progress_state_updates"]
                        accumulated_results["historical_feedback_triggers"] = result_dict["feedback_triggers"]
                    # conversation_results 使用 historical_feedback_triggers 的内容，避免重复
                    conversation_results["feedback_triggers"] = accumulated_results["historical_feedback_triggers"].copy()
                        
            return accumulated_results, conversation_results
        
        # 如果是其他几种类型的服务，换成inline请求
        else:
            # 初始化结果为字典格式，与后续代码期望的数据结构一致
            conversation_results = []
            inline_requests = []
            for group_idx, (group_key, json_group) in enumerate(json_contents.items(), 1):
                annotations_text = "\n\n".join([
                    f"\n{content}" 
                    for i, content in enumerate(json_group)
                ])
                json_content_text = f"\n\n ### Segment-level Episodic Annotations Batch: {group_key} ###\n\n{annotations_text}"
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
            
            # 将请求分批处理，每15个一批
            batch_size = 15
            total_batches = (len(inline_requests) + batch_size - 1) // batch_size
            incompleted_states = set([
                'JOB_STATE_FAILED',
                'JOB_STATE_CANCELLED',
                'JOB_STATE_EXPIRED',
            ])
            
            # 第五步：获取结果
            print("开始分批处理请求...")
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(inline_requests))
                batch_requests = inline_requests[start_idx: end_idx]
                
                print(f"\n处理第 {batch_idx + 1}/{total_batches} 批请求 (索引 {start_idx} 到 {end_idx - 1})...")
                
                # 创建 batch job
                try:
                    batch_job = self.batch_client.batches.create(
                        model=f"models/{self.model_name}",
                        src=batch_requests,
                        config={
                            'display_name': f"{display_name}-batch-{batch_idx + 1}",
                        },
                    )
                    print(f"Batch job 已创建: {batch_job.name}")
                except Exception as e:
                    raise RuntimeError(f"创建 batch job 失败: {str(e)}")
                
                # 等待 batch job 完成
                print("等待 batch job 完成...")
                batch_job_name = batch_job.name
                
                while True:
                    batch_job = self.batch_client.batches.get(name=batch_job_name)
                    print(f"当前状态: {batch_job.state}")
                    
                    if batch_job.state.name == "JOB_STATE_SUCCEEDED":
                        print("Batch job 已完成！")
                        break
                    elif batch_job.state.name in incompleted_states:
                        raise RuntimeError(f"Batch job 失败: {batch_job.name}")
                    
                    time.sleep(check_interval)
                
                # 获取当前批的结果
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
                                    
                                # 合并结果到 conversation_results
                                if self.low_level_category == "safety":
                                    if isinstance(result_dict, dict) and "safety_events" in result_dict:
                                        conversation_results.extend(result_dict["safety_events"])
                                    else:
                                        # 如果结果格式不符合预期，尝试包装
                                        print(f"警告: 结果格式不符合预期，尝试包装: {type(result_dict)}")
                                        if isinstance(result_dict, list):
                                            conversation_results.extend(result_dict["safety_events"])
                                        else:
                                            conversation_results.append(result_dict)
                                elif self.low_level_category == "tool_use":
                                    if isinstance(result_dict, dict) and "tool_use_events" in result_dict:
                                        conversation_results.extend(result_dict["tool_use_events"])
                                    else:
                                        # 如果结果格式不符合预期，尝试包装
                                        print(f"警告: 结果格式不符合预期，尝试包装: {type(result_dict)}")
                                        if isinstance(result_dict, list):
                                            conversation_results.extend(result_dict["tool_use_events"])
                                        else:
                                            conversation_results.append(result_dict["tool_use_events"])
                                
                                elif self.low_level_category == "error_recovery":
                                    if isinstance(result_dict, dict) and "error_recovery_events" in result_dict:
                                        conversation_results.extend(result_dict["error_recovery_events"])
                                    else:
                                        # 如果结果格式不符合预期，尝试包装
                                        print(f"警告: 结果格式不符合预期，尝试包装: {type(result_dict)}")
                                        if isinstance(result_dict, list):
                                            conversation_results.extend(result_dict["error_recovery_events"])
                                        else:
                                            conversation_results.append(result_dict["error_recovery_events"])
                                elif self.low_level_category == "resource_reminder":
                                    if isinstance(result_dict, dict) and "resource_events" in result_dict:
                                        conversation_results.extend(result_dict["resource_events"])
                                    else:
                                        # 如果结果格式不符合预期，尝试包装
                                        print(f"警告: 结果格式不符合预期，尝试包装: {type(result_dict)}")
                                        if isinstance(result_dict, list):
                                            conversation_results.extend(result_dict["resource_events"])
                                        else:
                                            conversation_results.append(result_dict["resource_events"])
                                elif self.low_level_category == "next_step_guidance":
                                    if isinstance(result_dict, dict) and "next_step_events" in result_dict:
                                        conversation_results.extend(result_dict["next_step_events"])
                                    else:
                                        # 如果结果格式不符合预期，尝试包装
                                        print(f"警告: 结果格式不符合预期，尝试包装: {type(result_dict)}")
                                        if isinstance(result_dict, list):
                                            conversation_results.extend(result_dict["next_step_events"])
                                        else:
                                            conversation_results.append(result_dict["next_step_events"])
                                            
                                elif self.low_level_category == "task_reminder":
                                    if isinstance(result_dict, dict) and "task_reminders" in result_dict:
                                        conversation_results.extend(result_dict["task_reminders"])
                                    else:
                                        # 如果结果格式不符合预期，尝试包装
                                        print(f"警告: 结果格式不符合预期，尝试包装: {type(result_dict)}")
                                        if isinstance(result_dict, list):
                                            conversation_results.extend(result_dict["task_reminders"])
                                        else:
                                            conversation_results.append(result_dict["task_reminders"])
                                elif self.low_level_category == "memory_recall":
                                    if isinstance(result_dict, dict) and "recall_dialogues" in result_dict:
                                        conversation_results.extend(result_dict["recall_dialogues"])
                                    else:
                                        # 如果结果格式不符合预期，尝试包装
                                        print(f"警告: 结果格式不符合预期，尝试包装: {type(result_dict)}")
                                        if isinstance(result_dict, list):
                                            conversation_results.extend(result_dict["recall_dialogues"])
                                        else:
                                            conversation_results.append(result_dict["recall_dialogues"])
                                    
                            except (AttributeError, json.JSONDecodeError) as e:
                                print(f"解析响应时出错 (请求 {i}): {e}")
                                print(f"响应内容: {inline_response.response}")
                                    
                        elif inline_response.error:
                            print(f"Error in request {i}: {inline_response.error}")
                            
                except Exception as e:
                    print(f"获取第 {batch_idx + 1} 批结果时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            return None, conversation_results
                
    
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
                            parsed_result = None
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
                                else:
                                    # 如果没有 response 字段，直接使用整个对象
                                    parsed_result = result_item
                            else:
                                parsed_result = result_item
                            
                            # 使用 key 存储结果
                            if result_key:
                                indexed_results[result_key] = parsed_result
                            else:
                                # 如果没有 key，按顺序添加（可能不准确）
                                indexed_results[len(indexed_results)] = parsed_result
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
        input_mode=args.input_mode,
        Days = args.days_list
    )
    
    # 创建新的需要保存的文件夹
    egolife_rag_annotation_list = []
    total_timestamp_list = []
    for activity in args.activity_list:
        
        timestamp_path = os.path.join(args.total_timestamp_path, activity, activity + "_10min_rectify_total.json")
        with open(timestamp_path, "r", encoding="utf-8") as f:
            timestamp_list = json.load(f)
        
        for idx, day in enumerate(args.days_list):
            
            day_timestamp_list = []
            for timestamp in timestamp_list:
                if idx + 1 == timestamp["date"]:
                    day_timestamp_list.append(timestamp)
            
            egolife_annotation_path = os.path.join(args.ego_life_path, activity, day)
            # 按照文件名中的时间戳数字进行排序（例如：A1_JAKE_DAY6_12000000.json 中的 12000000）
            egolife_annotation_list = sorted(
                os.listdir(egolife_annotation_path),
                key=lambda x: int(x.split("_")[-1].split(".")[0]) if x.split("_")[-1].split(".")[0].isdigit() else 0
            )
            
            assert len(day_timestamp_list) == len(egolife_annotation_list)
            
            os.makedirs(os.path.join(args.output_path, activity, day), exist_ok=True)
            for annotation_file, timestamp in zip(egolife_annotation_list, day_timestamp_list):
                assert timestamp["start_time"] == int(annotation_file.split("_")[-1].split(".")[0])
                egolife_rag_annotation_list.append((os.path.join(egolife_annotation_path, annotation_file), os.path.join(args.output_path, activity, day, annotation_file), (timestamp["start_time"], timestamp["end_time"])))            


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
        for idx, (annotation_path, output_path, timestamp_tuple) in enumerate(egolife_rag_annotation_list):
            # if idx < 64:
            #     continue
            
            # 读取注释文件
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation_data = json.load(f)
            annotation_data = {output_path.split("/")[-1]: annotation_data}
            
            # 构建用户提示词
            annotation_str = json.dumps(annotation_data, ensure_ascii=False, indent=2)  # 或者 indent=2
            user_prompt = system_prompt + PROMPTS["egolife_summarize_user_prompt"].format(annotation_data=annotation_str)
            user_prompts.append(user_prompt)
            output_paths.append((annotation_path, output_path, timestamp_tuple))
        
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
            for idx, (annotation_path, output_path, timestamp_tuple) in enumerate(output_paths):
                result_item = batch_results[idx]
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result_item, f, ensure_ascii=False, indent=2)
                print(f"结果已保存到: {output_path}")
                result_path_list.append(output_path)
        else:
            result_path_list = []
            for annotation_path, output_path, timestamp_tuple in output_paths:
                if not os.path.exists(output_path):
                    raise FileNotFoundError(f"输出文件不存在: {output_path}")
                result_path_list.append((output_path, timestamp_tuple))
        
        # 采用递归方式生成所有潜在主动服务timestamp
        if args.conversation_stage:
            output_path = os.path.join(args.output_path, args.activity_list[0], args.high_level_service_type, args.low_level_service_type)
            os.makedirs(output_path, exist_ok=True)
            
            accumulated_results, conversation_results = generator.generate_proactive_service(result_files=result_path_list,
                                                generation_config=None,
                                                display_name="proactive-service-job",
                                                output_path=output_path)
        
            if accumulated_results:
                with open(os.path.join(output_path, "accumulated_results.json"), "w", encoding="utf-8") as f:
                    json.dump(accumulated_results, f, ensure_ascii=False, indent=2)
            if conversation_results:
                with open(os.path.join(output_path, "conversation_results.json"), "w", encoding="utf-8") as f:
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
    parse.add_argument("--activity_list", type=list, default=["A3_TASHA"])
    parse.add_argument("--days_list", type=list, default=["DAY1", "DAY2", "DAY3", "DAY4", "DAY5", "DAY6", "DAY7"])
    parse.add_argument("--total_timestamp_path", type=str, default="./data/egolife/rag_annotation/egolife")
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
    parse.add_argument("--high_level_service_type", type=str, default="egolife_short_term")
    parse.add_argument("--low_level_service_type", type=str, default="resource_reminder")
    # 对于阶段的选择，一共两个阶段，一个是生成记录数据，另一个是直接生成对话数据
    parse.add_argument("--record_stage", type=bool, default=False)
    parse.add_argument("--conversation_stage", type=bool, default=True)
    
    args = parse.parse_args()
    
    main(args)

