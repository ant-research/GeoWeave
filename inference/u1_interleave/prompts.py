"""Prompts for GeoWeave interleave inference on the geo-aux bench."""
from __future__ import annotations

import re
_QUERY_IMAGE_TAG_RE = re.compile(r"\s*<image\d+>\s*", re.IGNORECASE)
_LEADING_IMAGE_PLACEHOLDER_RE = re.compile(r"^(?:\s*<image>\s*)+", re.IGNORECASE)
_TRAINFMT_PROBLEM_RE = re.compile(
    r"(?:^|\n)\s*Problem\s*[:：]\s*\n?(.*?)"
    r"(?=\n\s*(?:Auxiliary construction\(s\)|Reason step by step|Answer the problem)|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def problem_from_item(item: dict) -> str:
    """Extract the raw question from regular or train-format benchmark rows.

    Train-format rows often omit ``query`` and retain an old chain-of-thought
    wrapper in the human conversation. Only the ``Problem:`` section is reused.
    """
    query = _QUERY_IMAGE_TAG_RE.sub("", str(item.get("query", ""))).strip()
    if query:
        return query

    human = ""
    for turn in item.get("conversations", []):
        if turn.get("from") == "human":
            human = str(turn.get("value", ""))
            break

    match = _TRAINFMT_PROBLEM_RE.search(human)
    if match:
        return match.group(1).strip()
    return _LEADING_IMAGE_PLACEHOLDER_RE.sub("", human).strip()



def build_user_prompt(problem: str) -> str:
    """Build the uniform interleave SFT input format for every model."""
    problem = (problem or "").strip()
    return "<image>" if not problem else f"<image>{problem}"
