"""
EgoMemo: Training-Free Streaming Pipeline for Egocentric Video Processing.

Implements the Phase 1 (Training-Free) architecture with:
- Action Router (rule-based dispatcher)
- Question Queue Manager
- External Memory (via VideoGraphSeparated)
- Unified per-step LLM reasoning
"""

from .config import PipelineConfig
from .question_queue import QuestionQueueManager, Question, QuestionStatus
from .action_router import ActionRouter, ParsedAction
from .memory_bridge import MemoryBridge
from .output_recorder import OutputRecorder
from .working_memory import WorkingMemory
from .streaming_pipeline import StreamingPipeline
