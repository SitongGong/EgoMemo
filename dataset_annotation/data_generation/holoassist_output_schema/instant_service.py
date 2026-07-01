from pydantic import BaseModel, Field
from typing import Literal, Optional, List


# Safety Instant Event Schema
class SafetyInstantEvent(BaseModel):
    """Safety instant event"""
    clip_id: str = Field(..., description="<from annotations.clip_id>")
    segment_id: str = Field(..., description="<from annotations.segment_id or 'unknown'>")
    time_window: str = Field(..., description="Time window in format: HH:MM:SS.mmm-HH:MM:SS.mmm")
    risk_type: Literal[
        "sharp_blade_near_hand",
        "spill_slip_risk",
        "electric_shock_risk",
        "open_flame_near_cloth",
        "hot_surface_burn_risk",
        "unguarded_rotating_part",
        "falling_object_risk",
        "vehicle_close_pass",
        "other"
    ] = Field(..., description="Type of risk")
    observation: str = Field(..., description="<concise, objective evidence (no speculation)>")
    source: str = Field(default="manual_annotation", description="Source of the annotation")
    confidence: float = Field(default=0.0, description="Confidence score")


# Tool Use Instant Event Schema
class ToolUseInstantEvent(BaseModel):
    """Tool use instant event"""
    clip_id: str = Field(..., description="<from annotations.clip_id>")
    segment_id: str = Field(..., description="<from annotations.segment_id or 'unknown'>")
    time_window: str = Field(..., description="Time window in format: HH:MM:SS.mmm-HH:MM:SS.mmm")
    risk_type: Literal[
        "improper_grip",
        "unstable_handle",
        "device_not_off",
        "unsafe_posture",
        "wrong_orientation",
        "missing_guard",
        "loose_attachment",
        "other"
    ] = Field(..., description="Type of risk")
    observation: str = Field(..., description="<concise, objective evidence (no speculation)>")
    source: str = Field(default="manual_annotation", description="Source of the annotation")
    confidence: float = Field(default=0.0, description="Confidence score")


# Dialog Utterance Schema
class DialogUtterance(BaseModel):
    """Single utterance in a dialogue"""
    role: Literal["assistant", "user"] = Field(..., description="Role of the speaker")
    utterance: str = Field(..., description="The spoken text")


# Safety Dialog Schema
class SafetyDialog(BaseModel):
    """Dialog associated with a safety instant event"""
    clip_id: str = Field(..., description="<same as paired event>")
    segment_id: str = Field(..., description="<same as paired event>")
    time_window: str = Field(..., description="Time window in format: HH:MM:SS.mmm-HH:MM:SS.mmm")
    risk_type: str = Field(..., description="<same as paired event>")
    dialogue: List[DialogUtterance] = Field(
        ...,
        description="Dialogue turns: Turn 1 (assistant): empty, Turn 2 (user): non-trivial response (>=12 words), Turn 3 (assistant): short immediate mitigation + confirmation, Turn 4 (user): optional"
    )


# Tool Use Dialog Schema
class ToolUseDialog(BaseModel):
    """Dialog associated with a tool use instant event"""
    clip_id: str = Field(..., description="<same as paired event>")
    segment_id: str = Field(..., description="<same as paired event>")
    time_window: str = Field(..., description="Time window in format: HH:MM:SS.mmm-HH:MM:SS.mmm")
    risk_type: str = Field(..., description="<same as paired event>")
    dialogue: List[DialogUtterance] = Field(
        ...,
        description="Dialogue turns: Turn 1 (assistant): empty, Turn 2 (user): non-trivial response (>=12 words), Turn 3 (assistant): two-step corrective action + confirmation, Turn 4 (user): optional"
    )


# Top-level Output Schema for Safety Instant Service
class SafetyInstantServiceOutput(BaseModel):
    """Safety instant service output schema for safety events and dialogs"""
    safety_instant_events: List[SafetyInstantEvent] = Field(
        default_factory=list,
        description="List of safety instant events (can be empty if no events found)"
    )
    dialogs: List[SafetyDialog] = Field(
        default_factory=list,
        description="List of dialogs associated with the events (can be empty if no events found)"
    )
    # note: Optional[str] = Field(
    #     default=None,
    #     description="Optional note explaining why lists are empty (e.g., 'No Safety risk segments were present in the provided manual annotations.')"
    # )


# Top-level Output Schema for Tool Use Instant Service
class ToolUseInstantServiceOutput(BaseModel):
    """Tool use instant service output schema for tool use events and dialogs"""
    tool_use_instant_events: List[ToolUseInstantEvent] = Field(
        default_factory=list,
        description="List of tool use instant events (can be empty if no events found)"
    )
    dialogs: List[ToolUseDialog] = Field(
        default_factory=list,
        description="List of dialogs associated with the events (can be empty if no events found)"
    )
