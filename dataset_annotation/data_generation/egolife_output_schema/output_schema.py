from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union


# Pydantic models for EgoLife output schema
class IDoStep(BaseModel):
    """First-person atomic action step"""
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS-HH:MM:SS")
    action: str = Field(..., description="Concise first-person action phrase")
    role: Literal["do", "participate", "observe"] = Field(..., description="Role of the action")
    targets: List[str] = Field(default_factory=list, description="Object/person names if applicable")


class SpeakerSay(BaseModel):
    """Utterance from speaker (I or other person)"""
    speaker: str = Field(..., description="I | <Other Person Name>")
    utterance: str = Field(..., description="Verbatim utterance from transcript")
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS-HH:MM:SS")


class InteractionWithObject(BaseModel):
    """Interaction with an object"""
    object: str = Field(..., description="Object name")
    action: str = Field(..., description="Manipulation description for this specific phase (e.g., 'picked up', 'held while looking', 'placed back on surface')")
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS-HH:MM:SS")
    relation_to_me: Literal[
        "manipulate", "place", "pick_up", "adjust", 
        "point_to", "show", "receive", "hand_over", "inspect", "hold"
    ] = Field(..., description="Relation to the camera wearer")


class InteractionWithPeople(BaseModel):
    """Interaction with people"""
    with_person: str = Field(..., description="Person Name", alias="with")
    type: Literal[
        "conversation", "hand_over", "gaze", 
        "assistance", "physical_contact"
    ] = Field(..., description="Type of interaction")
    topic: str = Field(..., description="Short phrase about what this interaction is about")
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS-HH:MM:SS")
    
    class Config:
        populate_by_name = True


class EnvironmentState(BaseModel):
    """Environment state information"""
    location: Literal[
        "kitchen", "living room", "office", 
        "dining table", "hallway", "bedroom", "outdoors", "other"
    ] = Field(..., description="Location")
    lighting: Literal["natural", "artificial", "dim", "bright", "mixed"] = Field(..., description="Lighting condition")
    devices_state: str = Field(..., description="States of relevant visible devices (e.g., 'screens on', 'items on table', 'boxes open')")
    notable_changes: str = Field(..., description="Anything in the scene that changes and matters for understanding my actions")
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS-HH:MM:SS")


class TaskTransition(BaseModel):
    """Task transition information"""
    from_task: str = Field(..., description="Prior micro-task (first-person)", alias="from")
    to: str = Field(..., description="Next micro-task (first-person)")
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS-HH:MM:SS")
    
    class Config:
        populate_by_name = True


class InteractionRecord(BaseModel):
    """Single interaction record"""
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS-HH:MM:SS")
    description: str = Field(..., description="2-4 sentences summarizing what I do, who/what I interact with, and the micro-context within this record")
    i_do_steps: List[IDoStep] = Field(default_factory=list, description="All first-person atomic actions")
    speakers_say: List[SpeakerSay] = Field(default_factory=list, description="Utterances from speakers (I or other people)")
    interactions_with_objects: List[InteractionWithObject] = Field(
        default_factory=list, 
        description="Interactions with objects"
    )
    interactions_with_people: List[InteractionWithPeople] = Field(
        default_factory=list, 
        description="Interactions with people"
    )
    environment_state: List[EnvironmentState] = Field(
        default_factory=list, 
        description="Environment state information"
    )
    task_transition: List[TaskTransition] = Field(
        default_factory=list, 
        description="Task transition steps"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")


class EgoLifeOutput(BaseModel):
    """EgoLife video analysis output schema"""
    description: str = Field(
        ..., 
        description="Comprehensive, factual, first-person summary of everything I do and what happens around me in this video segment"
    )
    interaction_records: List[InteractionRecord] = Field(
        ..., 
        description="List of interaction records in chronological order"
    )
    
    