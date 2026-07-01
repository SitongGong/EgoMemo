from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class ProactiveDialogueUtterance(BaseModel):
    """Proactive dialogue utterance"""
    role: Literal["assistant", "user"] = Field(..., description="Role of the utterance")
    utterance: str = Field(
        ..., 
        description="Assistant proactive guidance or user reply (>= 12 words for user, optionally 1-2 more turns alternating assistant ↔ user)"
    )


# error-recovery proactive service
class ErrorRecoveryEvent(BaseModel):
    """Error recovery event entry for instant error-recovery proactive service"""
    event_id: str = Field(..., description="Event identifier (e.g., erecovery_001)")
    current_segment_id: str = Field(..., description="Segment id where the assistant speaks")
    current_time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say",
        "interaction_record"
    ] = Field(..., description="Source of supporting evidence")
    scene_description: str = Field(..., description="Short neutral description of what the user is doing around this moment")
    trigger_reason: str = Field(..., description="Why this moment is an Error-Recovery opportunity, grounded in the annotations")
    error_key: str = Field(..., description="Abstract description of the workflow error, e.g., 'wrong container', 'incorrect orientation', 'wrong port', 'invalid step order'")
    error_summary: str = Field(..., description="Short explanation of what is wrong and why rollback is needed")
    rollback_required: bool = Field(
        default=True,
        description="Whether rollback is required (always true for Error-Recovery)"
    )
    rollback_reason: str = Field(..., description="Why the user needs to undo and redo the step instead of only adjusting technique")
    risk_type: Literal["workflow_error"] = Field(
        default="workflow_error",
        description="Type of risk (always workflow_error for Error-Recovery)"
    )
    risk_immediacy: Literal["immediate"] = Field(
        default="immediate",
        description="Risk immediacy (always immediate for Error-Recovery)"
    )
    potential_downstream_issue: Optional[str] = Field(
        None,
        description="Optional: what could happen if the user continues without fixing it (invalid result, wasted effort, etc.). Should stay consistent with annotations (e.g., 'measurement result may be invalid', 'mixture may not follow the intended recipe')."
    )
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    proactive_dialogue: List[ProactiveDialogueUtterance] = Field(
        ..., 
        min_length=2,
        description="Proactive dialogue sequence (assistant -> user, optionally 1-2 more turns alternating assistant ↔ user). User reply should be >= 12 words."
    )


class ErrorRecoveryProactiveServiceOutput(BaseModel):
    """EgoLife Error-Recovery Proactive Service output schema"""
    service_main_type: str = Field(
        default="Instant Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Error-Recovery Proactive Service", 
        description="Sub service type"
    )
    error_recovery_events: List[ErrorRecoveryEvent] = Field(
        default_factory=list,
        description="List of error recovery events. May contain 0, 1, or multiple entries depending on how many Instant Error-Recovery opportunities exist in this batch."
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no Instant Error-Recovery Proactive Service opportunities were detected in the current batch"
    )


# next-step guidance proactive service
class NextStepEvent(BaseModel):
    """Next step event entry for next-step guidance proactive service"""
    event_id: str = Field(..., description="Event identifier (e.g., nextstep_001)")
    current_segment_id: str = Field(..., description="Segment id")
    current_time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say",
        "interaction_record"
    ] = Field(..., description="Source of supporting evidence")
    scene_description: str = Field(..., description="Neutral description of what step the user just completed")
    trigger_reason: str = Field(..., description="Why this moment qualifies as Next-Step Guidance")
    next_step_key: str = Field(..., description="Short tag: 'heat_pan', 'tighten_next_screw', 'select_mode', etc.")
    next_step_summary: str = Field(..., description="The next action the user can take")
    workflow_continuity: Literal["sequential"] = Field(
        default="sequential",
        description="Workflow continuity (always sequential for Next-Step Guidance)"
    )
    risk_immediacy: Literal["none"] = Field(
        default="none",
        description="Risk immediacy (always none for Next-Step Guidance)"
    )
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    proactive_dialogue: List[ProactiveDialogueUtterance] = Field(
        ..., 
        min_length=2,
        description="Proactive dialogue sequence (assistant -> user). User reply should be >= 12 words."
    )


class NextStepGuidanceProactiveServiceOutput(BaseModel):
    """EgoLife Next-Step Guidance Proactive Service output schema"""
    service_main_type: str = Field(
        default="Instant Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Next-Step Guidance Proactive Service", 
        description="Sub service type"
    )
    next_step_events: List[NextStepEvent] = Field(
        default_factory=list,
        description="List of next step events. May contain 0, 1, or multiple entries depending on how many Next-Step Guidance opportunities exist in this batch."
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no Next-Step Guidance opportunities were detected in this batch"
    )


# short-term resource reminder proactive service
class ResourceEvent(BaseModel):
    """Resource event entry for short-term resource reminder proactive service"""
    event_id: str = Field(..., description="Event identifier (e.g., resource_001)")
    current_segment_id: str = Field(..., description="Segment id")
    current_time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say",
        "interaction_record"
    ] = Field(..., description="Source of supporting evidence")
    scene_description: str = Field(..., description="What the user was doing before walking away")
    trigger_reason: str = Field(..., description="Why this is a short-term resource reminder")
    resource_key: str = Field(..., description="Resource key, e.g., 'stove_left_on', 'water_running', 'file_unsaved', 'door_ajar'")
    resource_summary: str = Field(..., description="Short explanation of what closure step was missed")
    potential_consequence: str = Field(..., description="Potential consequence: waste, mess, data loss, inconvenience")
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    proactive_dialogue: List[ProactiveDialogueUtterance] = Field(
        ..., 
        min_length=2,
        description="Proactive dialogue sequence (assistant -> user, optional turns 3-4). User reply should be >= 12 words."
    )


class ShortTermResourceReminderProactiveServiceOutput(BaseModel):
    """EgoLife Short-Term Resource Reminder Proactive Service output schema"""
    service_main_type: str = Field(
        default="Instant Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Short-Term Resource Reminder Proactive Service", 
        description="Sub service type"
    )
    resource_events: List[ResourceEvent] = Field(
        default_factory=list,
        description="List of resource events. May contain 0, 1, or multiple entries depending on how many Short-Term Resource Reminder opportunities exist in this batch."
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no Short-Term Resource Reminder opportunities were detected in this batch"
    )

