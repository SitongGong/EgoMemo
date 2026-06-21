"""
Module 4: Question Queue Manager (Pure Rules).

Maintains a List[Question] structure. Each question tracks:
qid, text, ask_time, status, evidence_notes, priority, recurring.

Key concepts:
- recurring=False (default): One-time question, marked ANSWERED after first answer.
  e.g., "What did I just do?"
- recurring=True: Ongoing question, stays active after each answer.
  e.g., "Tell me what I'm holding at all times" / "Guide me through making juice"
  These accumulate multiple answers over time.

Rules:
- New question -> append + status = PENDING
- LLM adds evidence -> modify evidence_notes, status = EVIDENCE_GATHERING
- LLM answers:
    - recurring=False -> status = ANSWERED, removed from active
    - recurring=True  -> record answer in answers list, stays PENDING
- Timeout -> status = TIMED_OUT + notify user
- Each step: active questions queried independently per-question
"""

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class QuestionStatus(Enum):
    PENDING = "pending"
    EVIDENCE_GATHERING = "evidence_gathering"
    ANSWERED = "answered"
    TIMED_OUT = "timed_out"


@dataclass
class Question:
    qid: str
    text: str
    ask_time: float  # video timestamp (seconds) when the question was asked
    wall_clock_received: float  # wall-clock time when received
    status: QuestionStatus = QuestionStatus.PENDING
    evidence_notes: List[str] = field(default_factory=list)
    priority: int = 1  # higher = more urgent
    answer: Optional[str] = None  # latest answer (for one-time questions)
    answer_time: Optional[float] = None
    answers: List[Dict] = field(default_factory=list)  # all answers (for recurring)
    recurring: bool = False  # if True, question stays active after each answer
    follow_up_parent: Optional[str] = None
    # Per-question MEM_READ pending result (for next step)
    pending_mem_read: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "qid": self.qid,
            "text": self.text,
            "ask_time": self.ask_time,
            "status": self.status.value,
            "evidence_notes": self.evidence_notes,
            "priority": self.priority,
            "recurring": self.recurring,
            "follow_up_parent": self.follow_up_parent,
            "answer": self.answer,
            "answers_count": len(self.answers),
        }


class QuestionQueueManager:
    """Thread-safe question queue with pure rule-based logic."""

    def __init__(self, timeout_seconds: float = 300.0):
        self._questions: List[Question] = []
        self._next_qid: int = 1
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()

    def add_question(
        self,
        text: str,
        ask_time: float,
        priority: int = 1,
        recurring: bool = False,
        follow_up_parent: Optional[str] = None,
    ) -> Question:
        """Add a new question to the queue."""
        with self._lock:
            qid = f"Q{self._next_qid}"
            self._next_qid += 1
            q = Question(
                qid=qid,
                text=text,
                ask_time=ask_time,
                wall_clock_received=time.time(),
                priority=priority,
                recurring=recurring,
                follow_up_parent=follow_up_parent,
            )
            self._questions.append(q)
            return q

    def record_answer(self, qid: str, answer: str, answer_time: float) -> bool:
        """Record an answer for a question.

        - recurring=False: marks ANSWERED (removed from active)
        - recurring=True: appends to answers list, stays PENDING
        """
        with self._lock:
            for q in self._questions:
                if q.qid == qid:
                    q.answer = answer
                    q.answer_time = answer_time
                    q.answers.append({
                        "answer": answer,
                        "time": answer_time,
                    })
                    if not q.recurring:
                        q.status = QuestionStatus.ANSWERED
                    # recurring questions stay PENDING/EVIDENCE_GATHERING
                    return True
            return False

    def add_evidence(self, qid: str, note: str) -> bool:
        """Add evidence to a question, transitioning to EVIDENCE_GATHERING."""
        with self._lock:
            for q in self._questions:
                if q.qid == qid and q.status in (
                    QuestionStatus.PENDING,
                    QuestionStatus.EVIDENCE_GATHERING,
                ):
                    q.evidence_notes.append(note)
                    q.status = QuestionStatus.EVIDENCE_GATHERING
                    return True
            return False

    def set_pending_mem_read(self, qid: str, result: str) -> None:
        """Store a MEM_READ result for a specific question (consumed next step)."""
        with self._lock:
            for q in self._questions:
                if q.qid == qid:
                    q.pending_mem_read = result
                    return

    def consume_pending_mem_read(self, qid: str) -> Optional[str]:
        """Consume and return the pending MEM_READ result for a question."""
        with self._lock:
            for q in self._questions:
                if q.qid == qid and q.pending_mem_read is not None:
                    result = q.pending_mem_read
                    q.pending_mem_read = None
                    return result
            return None

    def check_timeouts(self, current_video_time: float) -> List[Question]:
        """Check for timed-out questions. Returns list of newly timed-out ones.
        Note: recurring questions do NOT time out."""
        timed_out = []
        with self._lock:
            for q in self._questions:
                if q.recurring:
                    continue
                if q.status in (
                    QuestionStatus.PENDING,
                    QuestionStatus.EVIDENCE_GATHERING,
                ):
                    if (current_video_time - q.ask_time) > self._timeout_seconds:
                        q.status = QuestionStatus.TIMED_OUT
                        timed_out.append(q)
        return timed_out

    def get_active_questions(self) -> List[Question]:
        """Return PENDING + EVIDENCE_GATHERING questions, sorted by priority desc."""
        with self._lock:
            active = [
                q for q in self._questions
                if q.status in (QuestionStatus.PENDING, QuestionStatus.EVIDENCE_GATHERING)
            ]
            active.sort(key=lambda q: q.priority, reverse=True)
            return active

    def get_questions_at_time(self, current_video_time: float) -> List[Question]:
        """Get active questions whose ask_time < current_video_time.

        chunk [0,10) 处理完后传入 end_time=10：
        - ask_time=0 的问题 → 0 < 10 → 包含 ✓
        - ask_time=8 的问题 → 8 < 10 → 包含 ✓
        - ask_time=10 的问题 → 10 < 10 → 不包含（等下一个 chunk）✓
        - recurring 问题（状态仍为 PENDING）→ 每个 chunk 都会重新评估 ✓
        """
        with self._lock:
            return [
                q for q in self._questions
                if q.status in (QuestionStatus.PENDING, QuestionStatus.EVIDENCE_GATHERING)
                and q.ask_time < current_video_time
            ]

    def get_all_questions(self) -> List[Question]:
        with self._lock:
            return list(self._questions)

    def get_question_by_qid(self, qid: str) -> Optional[Question]:
        with self._lock:
            for q in self._questions:
                if q.qid == qid:
                    return q
            return None

    @property
    def total_count(self) -> int:
        with self._lock:
            return len(self._questions)

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(
                1 for q in self._questions
                if q.status in (QuestionStatus.PENDING, QuestionStatus.EVIDENCE_GATHERING)
            )
