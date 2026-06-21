"""
Working Memory: per-question dialogue history + proactive service records.

This is SEPARATE from the long-term memory (caption/KG/visual embedding).
Long-term memory stores video content for retrieval.
Working memory stores the model's own interaction history:
  - Per-question: all answers, evidence notes, MEM_READ queries/results
  - Per-proactive: the reminder content, user follow-ups

Saved as a structured JSON file, updated after each step.
Previous answers for a question are injected into its per-question prompt
so the model knows what it already said.
"""

import json
import logging
import os
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Manages per-question and per-proactive interaction history."""

    def __init__(self, working_dir: str):
        self._working_dir = working_dir
        self._lock = threading.Lock()
        os.makedirs(working_dir, exist_ok=True)
        self._file_path = os.path.join(working_dir, "working_memory.json")

        # Per-question history: qid -> list of interaction records
        self._questions: Dict[str, Dict] = {}
        # Proactive history: event_id -> record
        self._proactive: Dict[str, Dict] = {}

        # Load from disk if exists (for checkpoint resume)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._questions = data.get("questions", {})
                self._proactive = data.get("proactive", {})
                logger.info(
                    f"Loaded working memory: {len(self._questions)} questions, "
                    f"{len(self._proactive)} proactive events"
                )
            except Exception as e:
                logger.warning(f"Failed to load working memory: {e}")

    def save(self) -> None:
        """Persist working memory to disk."""
        with self._lock:
            data = {
                "questions": self._questions,
                "proactive": self._proactive,
            }
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save working memory: {e}")

    # ---- Question history ----

    def init_question(self, qid: str, text: str, ask_time: float, recurring: bool) -> None:
        """Register a new question."""
        with self._lock:
            if qid not in self._questions:
                self._questions[qid] = {
                    "qid": qid,
                    "text": text,
                    "ask_time": ask_time,
                    "recurring": recurring,
                    "interactions": [],
                }

    def add_answer(
        self, qid: str, answer: str, video_time: float,
        evidence: Optional[str] = None, reasoning: Optional[str] = None,
    ) -> None:
        """Record an answer for a question."""
        with self._lock:
            if qid not in self._questions:
                return
            self._questions[qid]["interactions"].append({
                "type": "answer",
                "answer": answer,
                "evidence": evidence,
                "reasoning": reasoning,
                "video_time": video_time,
            })
        self.save()

    def add_evidence(self, qid: str, note: str, video_time: float) -> None:
        """Record partial evidence for a question."""
        with self._lock:
            if qid not in self._questions:
                return
            self._questions[qid]["interactions"].append({
                "type": "evidence",
                "note": note,
                "video_time": video_time,
            })
        self.save()

    def add_mem_read(self, qid: str, query: str, result: str, video_time: float) -> None:
        """Record a MEM_READ query and its result for a question."""
        with self._lock:
            if qid not in self._questions:
                return
            self._questions[qid]["interactions"].append({
                "type": "mem_read",
                "query": query,
                "result": result[:500],  # truncate to avoid bloat
                "video_time": video_time,
            })
        self.save()

    def get_question_history(self, qid: str) -> Optional[Dict]:
        """Get full history for a question."""
        with self._lock:
            return self._questions.get(qid)

    def get_previous_answers(self, qid: str, max_count: int = 5) -> List[Dict]:
        """Get previous answers for a question (for recurring questions)."""
        with self._lock:
            q = self._questions.get(qid)
            if not q:
                return []
            answers = [
                i for i in q["interactions"] if i["type"] == "answer"
            ]
            return answers[-max_count:]

    def format_previous_answers(self, qid: str, max_count: int = 5) -> str:
        """Format previous answers as a string for prompt injection."""
        answers = self.get_previous_answers(qid, max_count)
        if not answers:
            return "(none)"
        parts = []
        for a in answers:
            t = a.get("video_time", 0)
            parts.append(f"[t={t:.0f}s] {a['answer']}")
        return "\n".join(parts)

    # ---- Proactive history ----

    def add_proactive(
        self, event_id: str, content: str, video_time: float,
        evidence: Optional[str] = None,
    ) -> None:
        """Record a proactive service event."""
        with self._lock:
            self._proactive[event_id] = {
                "event_id": event_id,
                "content": content,
                "video_time": video_time,
                "evidence": evidence,
                "follow_ups": [],
            }
        self.save()

    def add_proactive_follow_up(
        self, event_id: str, qid: str, text: str, video_time: float,
    ) -> None:
        """Record a user follow-up on a proactive event."""
        with self._lock:
            if event_id in self._proactive:
                self._proactive[event_id]["follow_ups"].append({
                    "qid": qid,
                    "text": text,
                    "video_time": video_time,
                })
        self.save()

    def get_proactive(self, event_id: str) -> Optional[Dict]:
        """Get a proactive event record."""
        with self._lock:
            return self._proactive.get(event_id)

    def get_all_data(self) -> Dict:
        """Get all working memory data."""
        with self._lock:
            return {
                "questions": dict(self._questions),
                "proactive": dict(self._proactive),
            }
