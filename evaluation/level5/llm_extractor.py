"""LLM-based answer extractor for Level 3.

Used as a fallback when the rule-based extractor (`extract_pred`) returns
nothing or returns a multi-sentence blob. The text judge re-reads the model
response and emits a short answer string.

Generic (no question-type metadata): geo-aux-bench answers are mixed —
option letters ("B"), numerics ("24"), symbolic ("2000\\pi"),
angles ("70°"). The prompt is written to handle all of them.
"""
from __future__ import annotations

import re
from typing import Optional

from evaluation.common.judge_cache import JudgeCache, make_key
from evaluation.common.judge_client import JudgeClient


_EXTRACT_SYS = (
    "You extract the FINAL answer from a model's response to a geometry "
    "problem. The answer may be an option letter (A/B/C/D), an integer, a "
    "decimal, a symbolic expression (e.g. 2000\\pi, \\sqrt{3}), or an angle "
    "(e.g. 70°). Return ONLY the answer string itself — no explanation, no "
    "units words like 'degrees', no surrounding quotes, no trailing period. "
    "If the response does not contain a final answer, return the empty string."
)

_EXTRACT_USER_TEMPLATE = (
    "Examples:\n"
    "---\n"
    "Question: Find the value of x.\n"
    "Model response: After solving the equation, we get x = 14.\n"
    "Extracted answer: 14\n"
    "---\n"
    "Question: In rectangle ABCD ... What is PE + PF?\\nChoices: A: 50  B: 24  C: 30  D: 25\n"
    "Model response: ... Therefore PE + PF = 24, which corresponds to option B.\n"
    "Extracted answer: B\n"
    "---\n"
    "Question: Compute the lateral surface area of the cone.\n"
    "Model response: The lateral surface area is \\boxed{{2000\\pi}} square units.\n"
    "Extracted answer: 2000\\pi\n"
    "---\n"
    "Question: 求∠A的度数。\n"
    "Model response: 由三角形内角和得 ∠A = 70°。\n"
    "Extracted answer: 70°\n"
    "---\n\n"
    "Now extract the final answer from this response:\n\n"
    "Question: {question}\n\n"
    "Model response: {response}\n\n"
    "Extracted answer:"
)


# Heuristic: when the rule-based pred is "obviously a sentence" rather than
# a short token, escalate to LLM. Tweak thresholds if needed.
_SENTENCE_RE = re.compile(r"[.。!?！？]")


def looks_like_sentence(pred: str) -> bool:
    if not pred:
        return False
    if len(pred) > 80:
        return True
    if _SENTENCE_RE.search(pred) and len(pred) > 20:
        return True
    if pred.count(" ") > 8:
        return True
    return False


def llm_extract(
    *,
    question: str,
    response: str,
    text_judge: JudgeClient,
    cache: Optional[JudgeCache] = None,
    max_tokens: int = 1024,
) -> str:
    """Ask the text judge to extract a short final answer. Cached.

    Returns the extracted string (may be empty). Never raises on judge errors
    — falls back to empty string so the scorer can still mark it incorrect
    cleanly instead of crashing the whole run.
    """
    sys_msg = _EXTRACT_SYS
    # Cap inputs to keep cache keys and token usage bounded.
    user_msg = _EXTRACT_USER_TEMPLATE.format(
        question=(question or "").strip()[:1500],
        response=(response or "").strip()[:4000],
    )
    if cache is not None:
        key = make_key(model=text_judge.cfg.model, system=sys_msg, user=user_msg)
        cached = cache.get(key)
        if cached is not None:
            return str(cached.get("extracted", ""))
    try:
        raw = text_judge.chat(sys_msg, user_msg, max_tokens=max_tokens)
    except Exception:
        return ""
    extracted = (raw or "").strip()
    # Strip surrounding quotes / leading "Extracted answer:" if the model echoed.
    extracted = extracted.removeprefix("Extracted answer:").strip()
    extracted = extracted.strip("`").strip('"').strip("'").strip()
    # First non-empty line only.
    for line in extracted.splitlines():
        line = line.strip()
        if line:
            extracted = line
            break
    if cache is not None:
        cache.put(key, {"extracted": extracted})
    return extracted
