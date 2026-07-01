from pydantic import BaseModel, Field
from typing import Literal, Optional, List


# Error Recovery Event Schema
class ErrorRecoveryEvent(BaseModel):
    """Error recovery event"""
    clip_id: str = Field(default="", description="Clip ID")
    segment_id: str = Field(..., description="<seg or 'unknown'>")
    time_window: str = Field(..., description="Time window in format: HH:MM:SS.mmm-HH:MM:SS.mmm")
    error_type: Literal[
        "wrong_order",
        "missing_component",
        "wrong_part_type",
        "wrong_dose",
        "misconfiguration_param",
        "forgot_step_required",
        "other"
    ] = Field(..., description="Type of error")
    observation: str = Field(default="", description="Observation")
    source: str = Field(default="manual_annotation", description="Source of the annotation")
    confidence: float = Field(default=0.0, description="Confidence score")


# Next Step Event Schema
class NextStepEvent(BaseModel):
    """Next step event"""
    clip_id: str = Field(default="", description="Clip ID")
    segment_id: str = Field(..., description="<seg or 'unknown'>")
    time_window: str = Field(..., description="Time window in format: HH:MM:SS.mmm-HH:MM:SS.mmm")
    guidance_type: Literal[
        "next_step_mixing",
        "next_step_install",
        "next_step_measure",
        "next_step_cleanup",
        "next_step_save_export",
        "other"
    ] = Field(..., description="Type of guidance")
    observation: str = Field(..., description="<objective cue of what's completed and what's next>")
    source: str = Field(default="manual_annotation", description="Source of the annotation")
    confidence: float = Field(default=0.0, description="Confidence score")


# Resource Reminder Event Schema
class ResourceReminderEvent(BaseModel):
    """Resource reminder event"""
    clip_id: str = Field(default="", description="Clip ID")
    segment_id: str = Field(..., description="<seg or 'unknown'>")
    time_window: str = Field(..., description="Time window in format: HH:MM:SS.mmm-HH:MM:SS.mmm")
    reminder_type: Literal[
        "stove_left_on",
        "unsaved_data",
        "cap_loose",
        "door_unlocked",
        "valve_open",
        "power_not_off",
        "forgot_item_left",
        "low_supply_needs_refill",
        "other"
    ] = Field(..., description="Type of reminder")
    observation: str = Field(default="", description="Observation")
    source: str = Field(default="manual_annotation", description="Source of the annotation")
    confidence: float = Field(default=0.0, description="Confidence score")


# Dialog Utterance Schema
class DialogUtterance(BaseModel):
    """Single utterance in a dialogue"""
    role: Literal["assistant", "user"] = Field(..., description="Role of the speaker")
    utterance: str = Field(..., description="The spoken text")


# Error Recovery Dialog Schema
class ErrorRecoveryDialog(BaseModel):
    """Dialog associated with an error recovery event"""
    clip_id: str = Field(default="", description="Clip ID (empty string)")
    segment_id: str = Field(default="", description="Segment ID (empty string)")
    time_window: str = Field(..., description="Time window in format: HH:MM:SS.mmm-HH:MM:SS.mmm")
    error_type: str = Field(default="", description="Error type (empty string)")
    dialogue: List[DialogUtterance] = Field(
        ...,
        description="Dialogue turns: Turn 1 (assistant): gentle pinpoint of the error + propose rollback/fix, Turn 2 (user): non-trivial reply (>=12 words), Turn 3 (assistant): guide through rollback/replace/retune + confirm back-on-track, Turn 4 (user): optional"
    )


# Next Step Dialog Schema
class NextStepDialog(BaseModel):
    """Dialog associated with a next step event"""
    clip_id: str = Field(default="", description="Clip ID (empty string)")
    segment_id: str = Field(default="", description="Segment ID (empty string)")
    time_window: str = Field(..., description="Time window in format: HH:MM:SS.mmm-HH:MM:SS.mmm")
    guidance_type: str = Field(default="", description="Guidance type (empty string)")
    dialogue: List[DialogUtterance] = Field(
        ...,
        description="Dialogue turns: Turn 1 (assistant): acknowledge completed step(s) + offer 1-2 next options, Turn 2 (user): non-trivial reply (>=12 words), choose/ask/modify, Turn 3 (assistant): confirm choice + mini step card / timer / checklist, Turn 4 (user): optional"
    )


# Resource Reminder Dialog Schema
class ResourceReminderDialog(BaseModel):
    """Dialog associated with a resource reminder event"""
    clip_id: str = Field(default="", description="Clip ID (empty string)")
    segment_id: str = Field(default="", description="Segment ID (empty string)")
    time_window: str = Field(..., description="Time window in format: HH:MM:SS.mmm-HH:MM:SS.mmm")
    reminder_type: str = Field(default="", description="Reminder type (empty string)")
    dialogue: List[DialogUtterance] = Field(
        ...,
        description="Dialogue turns: Turn 1 (assistant): polite cue of the unclosed/unhandled state + quick options, Turn 2 (user): non-trivial reply (>=12 words), Turn 3 (assistant): apply close/save/lock/take/refill or set reminder + confirm, Turn 4 (user): optional"
    )


# Top-level Output Schema for Error Recovery Service
class ErrorRecoveryServiceOutput(BaseModel):
    """Error recovery service output schema for error recovery events and dialogs"""
    error_recovery_events: List[ErrorRecoveryEvent] = Field(
        default_factory=list,
        description="List of error recovery events (can be empty if no events found)"
    )
    dialogs: List[ErrorRecoveryDialog] = Field(
        default_factory=list,
        description="List of dialogs associated with the events (can be empty if no events found)"
    )


# Top-level Output Schema for Next Step Service
class NextStepServiceOutput(BaseModel):
    """Next step service output schema for next step events and dialogs"""
    next_step_events: List[NextStepEvent] = Field(
        default_factory=list,
        description="List of next step events (can be empty if no events found)"
    )
    dialogs: List[NextStepDialog] = Field(
        default_factory=list,
        description="List of dialogs associated with the events (can be empty if no events found)"
    )


# Top-level Output Schema for Resource Reminder Service
class ResourceReminderServiceOutput(BaseModel):
    """Resource reminder service output schema for resource reminder events and dialogs"""
    resource_reminder_events: List[ResourceReminderEvent] = Field(
        default_factory=list,
        description="List of resource reminder events (can be empty if no events found)"
    )
    dialogs: List[ResourceReminderDialog] = Field(
        default_factory=list,
        description="List of dialogs associated with the events (can be empty if no events found)"
    )
