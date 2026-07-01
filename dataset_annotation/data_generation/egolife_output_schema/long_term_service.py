from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union


# habit coaching - new schema based on prompt
class HabitMetrics(BaseModel):
    """Metrics for habit occurrence"""
    sitting_minutes: float = Field(0.0, description="Sitting minutes")
    screen_minutes: float = Field(0.0, description="Screen time minutes")
    no_drink_minutes: float = Field(0.0, description="No drink minutes")


class HabitOccurrence(BaseModel):
    """Habit occurrence in a specific segment"""
    segment_id: str = Field(..., description="Source segment id")
    day_id: int = Field(..., description="Day identifier")
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal[
        "i_do_steps", 
        "interactions_with_objects", 
        "interactions_with_people", 
        "speakers_say", 
        "interaction_record"
    ] = Field(..., description="Source of supporting evidence")
    observation: str = Field(..., description="Specific evidence of this habit in this batch")
    local_context: str = Field(..., description="What the user is doing / environment")
    metrics: HabitMetrics = Field(..., description="Metrics for this occurrence")
    historical_context: str = Field(..., description="How this occurrence relates to earlier ones under this habit_id")
    inferred_role: Literal["sub_threshold", "threshold_cross", "followup_check"] = Field(
        ..., 
        description="Role of this occurrence"
    )
    workflow_position: Literal["start", "middle", "end", "standalone"] = Field(
        ..., 
        description="Position in workflow"
    )
    social_dynamics: Literal["self-initiated", "reacting_to_others", "jointly_decided"] = Field(
        ..., 
        description="Social dynamics of the occurrence"
    )
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score for this occurrence (0.0-1.0)")


class HabitStateUpdate(BaseModel):
    """Habit state update entry"""
    habit_id: str = Field(..., description="Habit identifier (e.g., habit_001)")
    habit_key: str = Field(..., description="One unhealthy habit pattern")
    habit_type: Literal["intra_day_health", "cross_day_lifestyle", "mixed"] = Field(
        ..., 
        description="Type of habit"
    )
    threshold_description: str = Field(..., description="Natural-language threshold")
    habit_summary: str = Field(
        ..., 
        description="How this habit looks so far including this batch. For first batch: 'initial hypothesis of an unhealthy habit pattern in this batch.'"
    )
    batch_consistency_level: Literal["high", "medium", "low"] = Field(
        ..., 
        description="Consistency level for this batch"
    )
    batch_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score for this batch (0.0-1.0)")
    occurrences: List[HabitOccurrence] = Field(..., description="List of occurrences for this habit")


class ProactiveDialogueUtterance(BaseModel):
    """Proactive dialogue utterance"""
    role: Literal["assistant", "user"] = Field(..., description="Role of the utterance")
    utterance: str = Field(
        ..., 
        description="Assistant coaching reminder (like a health band) or user reply (>= 10-12 words, may accept/decline/adjust) or follow-up"
    )


class HabitTrigger(BaseModel):
    """Habit trigger entry for coaching moment"""
    trigger_id: str = Field(..., description="Trigger identifier (e.g., htrig_001)")
    habit_id: str = Field(..., description="Habit identifier (e.g., habit_001)")
    habit_key: str = Field(..., description="Copied from habit_state_updates")
    trigger_type: Literal["intra_day_threshold", "cross_day_pattern"] = Field(
        ..., 
        description="Type of trigger"
    )
    current_segment_id: str = Field(..., description="Segment id where reminder is triggered")
    current_time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say",
        "environment_state",
        "interaction_record"
    ] = Field(..., description="Source of supporting evidence")
    current_observation: str = Field(..., description="What just happened that makes this a good coaching moment")
    current_local_context: str = Field(..., description="Short description of current scene")
    trigger_reason: str = Field(..., description="Why threshold is considered exceeded or cross-day pattern confirmed")
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    proactive_dialogue: List[ProactiveDialogueUtterance] = Field(
        ..., 
        description="Proactive dialogue sequence (assistant -> user -> assistant)"
    )


class HabitCoachingProactiveServiceOutput(BaseModel):
    """EgoLife Habit-Coaching Proactive Service output schema"""
    service_main_type: str = Field(
        default="Long-Term Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Habit-Coaching Proactive Service", 
        description="Sub service type"
    )
    habit_state_updates: List[HabitStateUpdate] = Field(
        default_factory=list,
        description="List of habit state updates (only new or updated habits/occurrences from this batch)"
    )
    habit_triggers: List[HabitTrigger] = Field(
        default_factory=list,
        description="List of habit triggers (each entry corresponds to one concrete coaching moment in this batch)"
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no unhealthy accumulation or cross-day lifestyle habit requiring coaching was detected"
    )


# long-horizon memory link proactive service
class MemoryOccurrence(BaseModel):
    """Memory occurrence in a specific segment"""
    segment_id: str = Field(..., description="Source segment file name")
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say",
        "interaction_record"
    ] = Field(..., description="Source of supporting evidence")
    observation: str = Field(..., description="Specific action or speech that creates a future-impact memory hook")
    local_context: Optional[str] = Field(
        None, 
        description="What is around / scene state (optional but recommended)"
    )
    historical_context: str = Field(
        ..., 
        description="How this occurrence relates to previous hooks of the same memory_key, if any. For first batch: 'no prior records; first detected in this batch.'"
    )
    inferred_link_role: Literal["initial_hook", "unresolved_plan"] = Field(
        ..., 
        description="Role of this occurrence in the memory link"
    )
    link_target_ids: List[str] = Field(
        default_factory=list,
        description="List of target event IDs this occurrence links to"
    )
    workflow_position: Literal["start", "middle", "end", "standalone"] = Field(
        ..., 
        description="Position in workflow"
    )
    social_dynamics: Literal["self-initiated", "reacting_to_others", "jointly_decided"] = Field(
        ..., 
        description="Social dynamics of the occurrence"
    )
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score for this occurrence (0.0-1.0)")


class NewMemoryCandidate(BaseModel):
    """New memory candidate entry"""
    event_id: str = Field(..., description="Event identifier (e.g., mem_00X)")
    memory_key: str = Field(..., description="Abstract but concrete memory pattern")
    memory_summary: str = Field(
        ..., 
        description="Summary of this memory pattern within the current batch, optionally referencing earlier days. For first batch: 'initial hypothesis of a possible long-horizon memory pattern in this batch.'"
    )
    memory_type: Literal["future_hook"] = Field(
        default="future_hook",
        description="Type of memory"
    )
    occurrences: List[MemoryOccurrence] = Field(..., description="List of occurrences for this memory")
    batch_consistency_level: Literal["high", "medium", "low"] = Field(
        ..., 
        description="Consistency level for this batch"
    )
    batch_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score for this batch (0.0-1.0)")


class LinkedPastEvent(BaseModel):
    """Linked past event in a realized memory link"""
    event_id: str = Field(..., description="Event identifier (e.g., mem_00A), copied from history")
    memory_key: str = Field(..., description="Memory key copied from historical JSON")
    past_segment_id: str = Field(..., description="Segment id of the past occurrence, copied from historical JSON")
    past_time_window: str = Field(..., description="Time window of the past occurrence in format: DAY# HH:MM:SS–HH:MM:SS, copied from historical JSON")
    link_reason: str = Field(..., description="Why this current event is a follow-up to that past event")


class RealizedMemoryLink(BaseModel):
    """Realized memory link entry"""
    link_id: str = Field(..., description="Link identifier (e.g., mlink_00Y)")
    current_segment_id: str = Field(..., description="Current segment id")
    current_time_window: str = Field(
        ..., 
        description="Time window in format: DAY# HH:MM:SS–HH:MM:SS. MUST be copied from i_do_steps[].time_window or speakers_say[].time_window only. Must be the earliest time_window where the user first starts the memory-dependent action or verbal reference."
    )
    supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say",
        "interaction_record"
    ] = Field(..., description="Source of supporting evidence")
    current_observation: str = Field(..., description="What happens now that uses/depends on a past event")
    current_local_context: str = Field(..., description="Current scene / situation")
    inferred_link_role: Literal["followup_use", "reminder", "check_status", "other"] = Field(
        ..., 
        description="Role of this link"
    )
    linked_past_events: List[LinkedPastEvent] = Field(
        ..., 
        min_length=1,
        description="List of linked past events. MUST contain at least one entry, each with event_id, past_segment_id, and past_time_window copied from the historical JSON."
    )
    workflow_position: Literal["start", "middle", "end", "standalone"] = Field(
        ..., 
        description="Position in workflow"
    )
    social_dynamics: Literal["self-initiated", "reacting_to_others", "jointly_decided"] = Field(
        ..., 
        description="Social dynamics of the occurrence"
    )
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    proactive_dialogue: List[ProactiveDialogueUtterance] = Field(
        ..., 
        min_length=3,
        description="Multi-turn dialogue (≥ 3 turns) showing how the assistant could use the remembered past event to help the user in the current situation. Must sound like a natural real-time assistant (do NOT mention 'video', 'annotations', or 'model')."
    )


class LongHorizonMemoryLinkProactiveServiceOutput(BaseModel):
    """EgoLife Long-Horizon Memory Link Proactive Service output schema"""
    service_main_type: str = Field(
        default="Long-Term Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Long-Horizon Memory Link Proactive Service", 
        description="Sub service type"
    )
    new_memory_candidates: List[NewMemoryCandidate] = Field(
        default_factory=list,
        description="List of new memory candidates (hooks detected in this batch)"
    )
    realized_memory_links: List[RealizedMemoryLink] = Field(
        default_factory=list,
        description="List of realized memory links. If no historical JSON is provided, MUST be empty list."
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no long-horizon memory-related hooks or realized links were detected in this batch"
    )


# personal progress feedback proactive service
class ProgressMetrics(BaseModel):
    """Metrics for progress occurrence"""
    practice_minutes: float = Field(0.0, description="Practice minutes")
    attempt_count: int = Field(0, description="Number of attempts")
    quality_trend: Literal["struggling", "partial_success", "clear_success", "unclear"] = Field(
        ..., 
        description="Quality trend"
    )


class ProgressOccurrence(BaseModel):
    """Progress occurrence in a specific segment"""
    segment_id: str = Field(..., description="Source segment id")
    day_id: int = Field(..., description="Day identifier")
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say",
        "interaction_record"
    ] = Field(..., description="Source of supporting evidence")
    observation: str = Field(..., description="Specific evidence of practice/execution in this batch")
    local_context: str = Field(..., description="What the user is doing / environment")
    metrics: ProgressMetrics = Field(..., description="Metrics for this occurrence")
    historical_context: str = Field(
        ..., 
        description="How this relates to earlier occurrences under this progress_id. For first batch: 'no prior records; first detected in this batch.'"
    )
    effort_type: Literal["practice", "execution", "review", "planning"] = Field(
        ..., 
        description="Type of effort"
    )
    outcome_quality: Literal["improved", "similar", "worse", "unclear"] = Field(
        ..., 
        description="Outcome quality compared to previous occurrences"
    )
    workflow_position: Literal["start", "middle", "end", "standalone"] = Field(
        ..., 
        description="Position in workflow"
    )
    social_dynamics: Literal["self-initiated", "reacting_to_others", "jointly_decided"] = Field(
        ..., 
        description="Social dynamics of the occurrence"
    )
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score for this occurrence (0.0-1.0)")


class ProgressStateUpdate(BaseModel):
    """Progress state update entry"""
    progress_id: str = Field(..., description="Progress identifier (e.g., pprog_001)")
    progress_key: str = Field(..., description="One skill/goal/performance pattern")
    progress_type: Literal[
        "skill_learning",
        "personal_goal",
        "performance",
        "habitual_task"
    ] = Field(
        ..., 
        description="Type of progress"
    )
    progress_summary: str = Field(
        ..., 
        description="How this pattern looks so far including this batch. For first batch: 'initial hypothesis of a progress-related pattern in this batch.'"
    )
    batch_consistency_level: Literal["high", "medium", "low"] = Field(
        ..., 
        description="Consistency level for this batch"
    )
    batch_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score for this batch (0.0-1.0)")
    occurrences: List[ProgressOccurrence] = Field(..., description="List of occurrences for this progress")


class FeedbackTrigger(BaseModel):
    """Feedback trigger entry"""
    trigger_id: str = Field(..., description="Trigger identifier (e.g., pfeed_001)")
    progress_id: str = Field(..., description="Progress identifier (e.g., pprog_001)")
    progress_key: str = Field(..., description="Copied from progress_state_updates")
    trigger_type: Literal[
        "cross_session_progress",
        "sustained_effort",
        "milestone_completion"
    ] = Field(
        ..., 
        description="Type of trigger"
    )
    current_segment_id: str = Field(..., description="Segment id where feedback is triggered")
    current_time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal["i_do_steps", "speakers_say"] = Field(
        ..., 
        description="Source of supporting evidence"
    )
    current_observation: str = Field(..., description="What just happened that makes this a good feedback moment")
    current_local_context: str = Field(..., description="Short description of current scene")
    comparison_basis: str = Field(..., description="How this compares to earlier runs / sessions")
    trigger_reason: str = Field(..., description="Why feedback is considered helpful now")
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    proactive_dialogue: List[ProactiveDialogueUtterance] = Field(
        ..., 
        min_length=2,
        description="Proactive dialogue sequence (assistant -> user, optionally 1-2 more turns alternating assistant ↔ user). User reply should be >= 12 words."
    )


class PersonalProgressFeedbackProactiveServiceOutput(BaseModel):
    """EgoLife Personal Progress Feedback Proactive Service output schema"""
    service_main_type: str = Field(
        default="Long-Term Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Personal Progress Feedback Proactive Service", 
        description="Sub service type"
    )
    progress_state_updates: List[ProgressStateUpdate] = Field(
        default_factory=list,
        description="List of progress state updates (only new or updated progress patterns from this batch, not all history)"
    )
    feedback_triggers: List[FeedbackTrigger] = Field(
        default_factory=list,
        description="List of feedback triggers. For first batch, will usually be empty because longitudinal comparison is missing."
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no personal-progress patterns or feedback-worthy cross-session moments were detected in this batch"
    )


# routine optimization proactive service
class RoutineOccurrence(BaseModel):
    """Routine occurrence in a specific segment"""
    segment_id: str = Field(..., description="Source segment id")
    day_id: int = Field(..., description="Day identifier")
    time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal[
        "i_do_steps",
        "interactions_with_objects",
        "interactions_with_people",
        "speakers_say",
        "interaction_record"
    ] = Field(..., description="Source of supporting evidence")
    observation: str = Field(..., description="Specific evidence of this pattern in this batch")
    local_context: str = Field(..., description="What the user is doing / environment")
    historical_context: str = Field(
        ..., 
        description="How this relates to earlier occurrences under this routine_id. For first batch: 'no prior records; first detected in this batch.'"
    )
    inferred_motivation: str = Field(..., description="Behavior-grounded guess")
    workflow_position: Literal["start", "middle", "end", "standalone"] = Field(
        ..., 
        description="Position in workflow"
    )
    social_dynamics: Literal["self-initiated", "reacting_to_others", "jointly_decided"] = Field(
        ..., 
        description="Social dynamics of the occurrence"
    )
    implicit_avoidance: Optional[str] = Field(
        None,
        description="Optional: what the user seems to avoid relative to this routine"
    )
    routine_evolution: Optional[str] = Field(
        None,
        description="Optional: how this occurrence strengthens/changes/refines the routine"
    )
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score for this occurrence (0.0-1.0)")


class RoutineStateUpdate(BaseModel):
    """Routine state update entry"""
    routine_id: str = Field(..., description="Routine identifier (e.g., rtopt_001)")
    routine_key: str = Field(..., description="One routine/configuration pattern")
    routine_type: Literal[
        "configuration_preference",
        "time_structured_routine",
        "task_structured_routine"
    ] = Field(
        ..., 
        description="Type of routine"
    )
    routine_summary: str = Field(
        ..., 
        description="How this pattern looks so far including this batch. For first batch: 'initial hypothesis of a routine/configuration pattern in this batch.'"
    )
    batch_consistency_level: Literal["high", "medium", "low"] = Field(
        ..., 
        description="Consistency level for this batch"
    )
    batch_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score for this batch (0.0-1.0)")
    occurrences: List[RoutineOccurrence] = Field(..., description="List of occurrences for this routine")


class OptimizationTrigger(BaseModel):
    """Optimization trigger entry"""
    trigger_id: str = Field(..., description="Trigger identifier (e.g., ropt_001)")
    routine_id: str = Field(..., description="Routine identifier (e.g., rtopt_001)")
    routine_key: str = Field(..., description="Copied from routine_state_updates")
    routine_type: Literal[
        "configuration_preference",
        "time_structured_routine",
        "task_structured_routine"
    ] = Field(
        ..., 
        description="Type of routine"
    )
    trigger_type: Literal[
        "routine_block_optimization",
        "configuration_default_suggestion",
        "missed_routine_reminder"
    ] = Field(
        ..., 
        description="Type of trigger"
    )
    current_segment_id: str = Field(..., description="Segment id where optimization/reminder is triggered")
    current_time_window: str = Field(..., description="Time window in format: DAY# HH:MM:SS–HH:MM:SS")
    supporting_source: Literal["i_do_steps", "speakers_say"] = Field(
        ..., 
        description="Source of supporting evidence"
    )
    current_observation: str = Field(..., description="What just happened that makes this a good optimization or reminder moment")
    current_local_context: str = Field(..., description="Short description of current scene")
    activation_type: Literal["cross_day_pattern", "expected_pattern_missing"] = Field(
        ..., 
        description="Type of activation"
    )
    comparison_basis: str = Field(..., description="How this day's pattern compares to earlier days / sessions")
    activation_reason: str = Field(..., description="Why optimization or reminder is considered helpful now")
    occurrence_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    proactive_dialogue: List[ProactiveDialogueUtterance] = Field(
        ..., 
        min_length=2,
        description="Proactive dialogue sequence (assistant -> user, optionally 1-2 more turns alternating assistant ↔ user). User reply should be >= 12 words."
    )


class RoutineOptimizationProactiveServiceOutput(BaseModel):
    """EgoLife Routine Optimization Proactive Service output schema"""
    service_main_type: str = Field(
        default="Long-Term Proactive Service", 
        description="Main service type"
    )
    service_sub_type: str = Field(
        default="Routine Optimization Proactive Service", 
        description="Sub service type"
    )
    routine_state_updates: List[RoutineStateUpdate] = Field(
        default_factory=list,
        description="List of routine state updates (only new or updated patterns from this batch, not the whole historical store)"
    )
    optimization_triggers: List[OptimizationTrigger] = Field(
        default_factory=list,
        description="List of optimization triggers. For first batch, will usually be empty because cross-day evidence is missing."
    )
    note: Optional[str] = Field(
        None,
        description="Optional note when no stable routine/configuration patterns or optimization-worthy cross-day moments were detected in this batch"
    )


