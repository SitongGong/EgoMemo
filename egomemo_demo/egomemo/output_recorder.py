"""
Module 6: Output Layer / Decision Trajectory Recorder.

Records the complete decision trajectory for:
- Answers to user questions
- Proactive reminders
- Silence decisions
- Memory operations
- Full per-step state
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OutputRecorder:
    """Records all pipeline decisions and outputs for analysis and replay."""

    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.trajectory: List[Dict] = []
        self.answers: List[Dict] = []
        self.proactive_reminders: List[Dict] = []
        self.silence_steps: List[int] = []
        self._start_time = time.time()
        # 由 StreamingPipeline.run_on_video 在 finalize 前注入：
        #   {
        #     "video_duration_seconds": float,
        #     "wall_clocks": {...},      # memory_wall / reasoning_wall / total / chunk0 / ...
        #     "rtf": {...},              # 各种 RTF 指标
        #     "per_chunk": {             # 每 chunk 分项耗时
        #         "memory_wall_per_chunk": [...],
        #         "reasoning_wall_per_chunk": [...],
        #     },
        #     "memory_breakdown": {...}, # memory_bridge.timing 的细粒度统计
        #     "reasoning_breakdown": {...},
        #   }
        self.timing: Dict = {}

    def record_step(
        self,
        step_idx: int,
        video_time: float,
        caption: str,
        actions: List,
        route_result: Dict,
    ) -> None:
        """Record a complete step in the trajectory."""
        entry = {
            "step": step_idx,
            "video_time": video_time,
            "caption": caption[:500],
            "actions_taken": route_result.get("actions_taken", []),
            "outputs": route_result.get("outputs", []),
            "has_retrieved_memory": route_result.get("retrieved_memory") is not None,
            "wall_clock": time.time() - self._start_time,
        }
        self.trajectory.append(entry)

    def record_answer(
        self,
        qid: str,
        answer: str,
        video_time: float,
        evidence: Optional[str] = None,
    ) -> None:
        """Record an answer event."""
        entry = {
            "qid": qid,
            "answer": answer,
            "video_time": video_time,
            "evidence": evidence,
            "wall_clock": time.time() - self._start_time,
        }
        self.answers.append(entry)
        logger.info(f"Recorded answer for {qid} at t={video_time:.1f}s")

    def record_proactive(
        self,
        content: str,
        video_time: float,
        evidence: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> None:
        """Record a proactive reminder event."""
        entry = {
            "event_id": event_id,
            "content": content,
            "video_time": video_time,
            "evidence": evidence,
            "wall_clock": time.time() - self._start_time,
        }
        self.proactive_reminders.append(entry)
        logger.info(f"Recorded proactive reminder at t={video_time:.1f}s")

    def record_silence(self, step_idx: int, video_time: float) -> None:
        """Record a silence step."""
        self.silence_steps.append(step_idx)

    def get_proactive_by_id(self, event_id: str) -> Optional[Dict]:
        """Retrieve a proactive reminder by its event ID."""
        for r in self.proactive_reminders:
            if r.get("event_id") == event_id:
                return r
        return None

    def finalize(self) -> None:
        """Save the full trajectory to disk."""
        os.makedirs(self.working_dir, exist_ok=True)
        output_path = os.path.join(self.working_dir, "decision_trajectory.json")
        data = self.get_full_trajectory()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Decision trajectory saved to {output_path}")

    def get_full_trajectory(self) -> Dict:
        """Return the complete decision trajectory as a dict."""
        summary = {
            "total_steps": len(self.trajectory),
            "total_answers": len(self.answers),
            "total_proactive": len(self.proactive_reminders),
            "total_silence": len(self.silence_steps),
            "wall_clock_seconds": time.time() - self._start_time,
        }
        # 把外部注入的 timing 平铺到 summary 顶层（便于直接看），
        # 完整结构同时挂在 "timing" 顶级字段
        if self.timing:
            for key in (
                "video_duration_seconds",
                "wall_clocks",
                "rtf",
                "per_chunk",
                "memory_breakdown",
                "reasoning_breakdown",
            ):
                if key in self.timing:
                    summary[key] = self.timing[key]
        return {
            "summary": summary,
            "answers": self.answers,
            "proactive_reminders": self.proactive_reminders,
            "trajectory": self.trajectory,
            "timing": self.timing,
        }

    def get_recent_outputs(self, n: int = 10) -> List[Dict]:
        """Return the last n trajectory entries."""
        return self.trajectory[-n:]
