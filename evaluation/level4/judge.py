"""Visual judge for factual errors in a geometry solution process."""
from __future__ import annotations

from evaluation.common.judge_cache import JudgeCache, make_key
from evaluation.common.judge_client import JudgeClient

_SYSTEM = (
    "You are a strict geometry reasoning auditor. Check ONLY for explicit factual "
    "errors in the intermediate solution process: invalid geometric relations, "
    "missing theorem premises, wrong correspondence, invalid deductions, and "
    "arithmetic/algebra/numerical calculation errors in intermediate steps. "
    "Do not judge whether the final answer is correct. Do not penalize an incomplete "
    "proof, an uncommon but valid method, style, or omission unless there is an "
    "explicit false claim. Claims about auxiliary constructions should be checked "
    "against the supplied images and the process. If a claim is ambiguous or cannot "
    "be established from the question and images, do not mark it as an error."
)

_USER = """Question:
{question}

Model's complete solution process:
-----
{reasoning}
-----

Check the process for factual geometry/logic errors and intermediate calculation
errors. The final result is explicitly out of scope. Do not treat an unverified or
ambiguous statement as an error.

Return ONLY JSON:
{{
  "reasoning_correct": 0 or 1,
  "errors": [
    {{"claim": "<verbatim or concise claim>", "error_type": "<type>", "reason": "<why it is factually wrong>"}}
  ],
  "ignored_issues": [
    {{"claim": "<issue not counted>", "reason": "<e.g. final answer or incomplete proof>"}}
  ],
  "reason": "<short summary>"
}}
"""


def judge_reasoning(*, images: list[str], question: str, reasoning: str,
                    vision_judge: JudgeClient, cache: JudgeCache) -> dict:
    user = _USER.format(question=question, reasoning=reasoning)
    key = make_key(model=vision_judge.cfg.model, system=_SYSTEM, user=user,
                   image_paths=images)
    cached = cache.get(key)
    if cached is not None:
        return cached
    raw = vision_judge.chat_json(_SYSTEM, user, images=images, max_tokens=4096)
    errors = raw.get("errors", [])
    if not isinstance(errors, list):
        errors = []
    ignored = raw.get("ignored_issues", [])
    if not isinstance(ignored, list):
        ignored = []
    raw_value = raw.get("reasoning_correct", 0)
    if isinstance(raw_value, str):
        raw_value = raw_value.strip().lower() in {"1", "true", "yes"}
    out = {
        "reasoning_correct": 1 if bool(raw_value) else 0,
        "errors": [x if isinstance(x, dict) else {"claim": str(x)} for x in errors][:30],
        "ignored_issues": [x if isinstance(x, dict) else {"claim": str(x)} for x in ignored][:30],
        "reason": str(raw.get("reason", ""))[:1000],
    }
    # A binary error verdict must be supported by evidence. Conversely, any
    # returned error is sufficient for a zero under the metric definition.
    if out["errors"]:
        out["reasoning_correct"] = 0
    elif out["reasoning_correct"] == 0:
        out["reasoning_correct"] = 1
        out["reason"] = (out["reason"] + "; invalid zero verdict without evidence").strip("; ")
    cache.put(key, out)
    return out
