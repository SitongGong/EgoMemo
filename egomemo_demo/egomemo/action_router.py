"""
Action Router — parse action tokens from VLM output.

Supports two token formats:
  Text tokens (base model):     [silent]  [search]  [respond]
  Special tokens (RL fine-tuned): <|silent|>  <|search|>  <|respond|>

Both formats are tried in order; first match wins.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedAction:
    action: str  # SILENT, SEARCH, RESPOND
    answer_text: Optional[str] = None      # content after [respond]
    search_query: Optional[str] = None     # content after [search]
    evidence: Optional[str] = None
    reasoning: Optional[str] = None
    raw_output: str = ""


# Compiled patterns for token detection
# Special tokens (RL fine-tuned) checked first, then text tokens (base model)
_TOKEN_PATTERNS = [
    # RL fine-tuned special tokens
    (re.compile(r"<\|silent\|>"), "SILENT"),
    (re.compile(r"<\|search\|>"), "SEARCH"),
    (re.compile(r"<\|respond\|>"), "RESPOND"),
    (re.compile(r"<\|proactive\|>"), "RESPOND"),
    # Base model text tokens
    (re.compile(r"\[silent\]", re.IGNORECASE), "SILENT"),
    (re.compile(r"\[search\]", re.IGNORECASE), "SEARCH"),
    (re.compile(r"\[respond\]", re.IGNORECASE), "RESPOND"),
]


class ActionRouter:
    """Parse VLM output into structured actions based on [token] markers."""

    def parse_llm_output(self, raw_response: str) -> List[ParsedAction]:
        """Parse VLM output into a list of ParsedAction objects.

        Looks for [silent], [search], or [respond] tokens in the output.
        Returns a single-element list for compatibility with existing pipeline.
        """
        if not raw_response:
            return [ParsedAction(action="SILENT", raw_output="")]

        text = raw_response.strip()

        # Strip <think>...</think> if present (Qwen3.5/Qwen3-VL thinking mode)
        think_match = re.search(r"<think>.*?</think>\s*", text, re.DOTALL)
        if think_match:
            text = text[think_match.end():].strip()

        # Try to match action tokens
        for pattern, action_name in _TOKEN_PATTERNS:
            match = pattern.search(text)
            if match:
                remainder = text[match.end():].strip()
                if action_name == "SEARCH":
                    return [ParsedAction(
                        action="SEARCH",
                        search_query=remainder if remainder else None,
                        raw_output=raw_response,
                    )]
                elif action_name == "RESPOND":
                    return [ParsedAction(
                        action="RESPOND",
                        answer_text=remainder if remainder else None,
                        raw_output=raw_response,
                    )]
                else:  # SILENT
                    return [ParsedAction(action="SILENT", raw_output=raw_response)]

        # Fallback heuristics
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["no relevant", "nothing to report", "skip", "cannot answer"]):
            return [ParsedAction(action="SILENT", raw_output=raw_response)]

        # If substantial content, treat as respond
        if len(text) > 20:
            return [ParsedAction(action="RESPOND", answer_text=text, raw_output=raw_response)]

        return [ParsedAction(action="SILENT", raw_output=raw_response)]
