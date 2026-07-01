from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class ProactiveDialogueUtterance(BaseModel):
    """Proactive dialogue utterance"""
    role: Literal["assistant", "user"] = Field(..., description="Role of the utterance")
    utterance: str = Field(
        ..., 
        description="Assistant proactive reminder or user reply (>= 12 words for user, optionally 1-2 more turns alternating assistant ↔ user)"
    )


class RecallDialogue(BaseModel):
    """Recall dialogue entry for episodic memory recall"""
    recall_id: str = Field(..., description="Recall identifier (e.g., erecall_001)")
    recall_type: Literal[
        "forgotten_item",
        "unfinished_plan",
        "context_cue",
        "check_status"
    ] = Field(
        ..., 
        description="Type of recall"
    )
    current_segment_id: str = Field(..., description="Segment id where the assistant speaks")
    current_time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    current_supporting_source: Literal["i_do_steps", "speakers_say"] = Field(
        ..., 
        description="Source of current supporting evidence"
    )
    current_observation: str = Field(..., description="What is happening now that makes the past episode relevant")
    current_local_context: str = Field(..., description="Short description of current scene / situation")
    past_segment_id: str = Field(..., description="Segment id of the past episode")
    past_time_window: str = Field(..., description="Time window of the past episode in format: DAY# HH:MM:SS–HH:MM:SS")
    past_supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say"
    ] = Field(
        ..., 
        description="Source of past supporting evidence"
    )
    past_observation: str = Field(..., description="What happened earlier that is now useful")
    time_gap_seconds: float = Field(
        0.0, 
        description="Time gap in seconds (can be approximate; used to reflect the short-horizon nature, e.g., 600 for 10 minutes)"
    )
    link_reason: str = Field(..., description="Why this past episode is helpful to recall now within ~2 hours")
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    proactive_dialogue: List[ProactiveDialogueUtterance] = Field(
        ..., 
        min_length=2,
        description="Proactive dialogue sequence (assistant -> user, optionally 1-2 more turns alternating assistant ↔ user). User reply should be >= 12 words."
    )


class EpisodicMemoryRecallProactiveServiceOutput(BaseModel):
    """EgoLife Episodic Memory Recall Proactive Service output schema"""
    service_main_type: str = Field(
        default="Long-Term Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Episodic Memory Recall Proactive Service", 
        description="Sub service type"
    )
    recall_dialogues: List[RecallDialogue] = Field(
        default_factory=list,
        description="List of recall dialogues. May contain 0, 1, or multiple entries depending on how many good episodic recall opportunities exist in this batch."
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no short-horizon episodic memory recall opportunities were detected in this batch"
    )


# episodic task reminder proactive service
class TaskReminder(BaseModel):
    """Task reminder entry for episodic task reminder"""
    reminder_id: str = Field(..., description="Reminder identifier (e.g., etask_001)")
    reminder_type: Literal[
        "skipped_step",
        "unfinished_task",
        "pending_check"
    ] = Field(
        ..., 
        description="Type of reminder"
    )
    task_thread_summary: str = Field(..., description="Short abstract description of the task this reminder belongs to")
    current_segment_id: str = Field(..., description="Segment id where the assistant speaks")
    current_time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    current_supporting_source: Literal["i_do_steps", "speakers_say"] = Field(
        ..., 
        description="Source of current supporting evidence"
    )
    current_observation: str = Field(..., description="What is happening now that shows the user is moving away")
    current_local_context: str = Field(..., description="Short description of current scene / situation")
    past_segment_id: str = Field(..., description="Segment id of the pending-task episode")
    past_time_window: str = Field(..., description="Time window of the pending-task episode in format: DAY# HH:MM:SS–HH:MM:SS")
    past_supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say"
    ] = Field(
        ..., 
        description="Source of past supporting evidence"
    )
    past_observation: str = Field(..., description="What happened earlier that defines the unfinished task or step")
    time_gap_seconds: float = Field(
        0.0, 
        description="Time gap in seconds (can be approximate; only needs to reflect that this is a short within-episode gap, e.g., 120 for 2 minutes, 900 for 15 minutes)"
    )
    transition_reason: str = Field(..., description="Why this looks like a switch away from the task")
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    proactive_dialogue: List[ProactiveDialogueUtterance] = Field(
        ..., 
        min_length=2,
        description="Proactive dialogue sequence (assistant -> user, optionally 1-2 more turns alternating assistant ↔ user). User reply should be >= 12 words."
    )


class EpisodicTaskReminderProactiveServiceOutput(BaseModel):
    """EgoLife Episodic Task Reminder Proactive Service output schema"""
    service_main_type: str = Field(
        default="Long-Term Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Episodic Task Reminder Proactive Service", 
        description="Sub service type"
    )
    task_reminders: List[TaskReminder] = Field(
        default_factory=list,
        description="List of task reminders. May contain 0, 1, or multiple entries depending on how many good within-episode pending-task opportunities exist in this batch."
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no within-episode pending tasks with clear transition-away behavior were detected in this batch"
    )

