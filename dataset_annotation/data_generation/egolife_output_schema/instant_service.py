from pydantic import BaseModel, Field
from typing import Literal, Optional, List


class ProactiveDialogueUtterance(BaseModel):
    """Proactive dialogue utterance"""
    role: Literal["assistant", "user"] = Field(..., description="Role of the utterance")
    utterance: str = Field(
        ..., 
        description="Assistant proactive warning or user reply (>= 12 words for user, optionally 1-2 more turns alternating assistant ↔ user)"
    )


class SafetyEvent(BaseModel):
    """Safety event entry for instant safety proactive service"""
    event_id: str = Field(..., description="Event identifier (e.g., safety_001)")
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
    trigger_reason: str = Field(..., description="Why this moment is an Instant Safety opportunity, grounded in the annotations")
    hazard_key: str = Field(..., description="Abstract description of the hazard, e.g., 'knife near fingers', 'spilled liquid underfoot'")
    hazard_summary: str = Field(..., description="Short explanation of what is dangerous and why now")
    risk_type: Literal[
        "cut",
        "burn",
        "slip",
        "collision",
        "electric_shock",
        "other"
    ] = Field(
        ..., 
        description="Type of risk"
    )
    risk_immediacy: Literal["immediate"] = Field(
        default="immediate",
        description="Risk immediacy (always immediate for Instant Safety)"
    )
    potential_consequence: Optional[str] = Field(
        None,
        description="Optional: what could happen (scald, cut, fall, etc.). May approximate severity but must remain consistent with annotations."
    )
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    proactive_dialogue: List[ProactiveDialogueUtterance] = Field(
        ..., 
        min_length=2,
        description="Proactive dialogue sequence (assistant -> user, optionally 1-2 more turns alternating assistant ↔ user). User reply should be >= 12 words."
    )


class SafetyProactiveServiceOutput(BaseModel):
    """EgoLife Safety Proactive Service output schema"""
    service_main_type: str = Field(
        default="Instant Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Safety Proactive Service", 
        description="Sub service type"
    )
    safety_events: List[SafetyEvent] = Field(
        default_factory=list,
        description="List of safety events. May contain 0, 1, or multiple entries depending on how many Instant Safety opportunities exist in this batch."
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no Instant Safety Proactive Service opportunities were detected in the current batch"
    )


# tool use proactive service
class DialogueUtterance(BaseModel):
    """Dialogue utterance for tool use"""
    role: Literal["assistant", "user"] = Field(..., description="Role of the utterance")
    utterance: str = Field(
        ..., 
        description="Concise technique advice (assistant) or natural reply (>= 12 words for user)"
    )


class ToolUseDialogueEvent(BaseModel):
    """Tool use dialogue event entry"""
    event_id: str = Field(..., description="Event identifier (e.g., tooluse_001)")
    segment_id: str = Field(..., description="Segment id")
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say",
        "interaction_record"
    ] = Field(..., description="Source of supporting evidence")
    scene_description: str = Field(..., description="Neutral description of what the user is doing")
    trigger_reason: str = Field(..., description="Why this is a Tool Use technique issue (not Safety / not Error-Recovery)")
    tool_name: str = Field(..., description="Tool name (e.g., knife, screwdriver, brush, etc.)")
    technique_issue: str = Field(..., description="Specific suboptimal technique")
    why_suboptimal: str = Field(..., description="Why this technique reduces stability/efficiency/quality")
    risk_level: Literal["low"] = Field(
        default="low",
        description="Risk level (always low for Tool Use)"
    )
    requires_rollback: bool = Field(
        default=False,
        description="Whether rollback is required (always false for Tool Use)"
    )
    is_safe: bool = Field(
        default=True,
        description="Whether the situation is safe (always true for Tool Use)"
    )
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    dialogue: List[DialogueUtterance] = Field(
        ..., 
        min_length=2,
        description="Dialogue sequence (assistant -> user). User reply should be >= 12 words."
    )


class ToolUseProactiveServiceOutput(BaseModel):
    """EgoLife Tool Use Proactive Service output schema"""
    service_main_type: str = Field(
        default="Instant Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Tool Use Proactive Service", 
        description="Sub service type"
    )
    tool_use_events: List[ToolUseDialogueEvent] = Field(
        default_factory=list,
        description="List of dialogue events. May contain 0, 1, or multiple entries depending on how many Tool Use opportunities exist in this batch."
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no Tool Use technique issues were detected in the current batch"
    )

