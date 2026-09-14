"""Visual judge for errors in reading the original geometry diagram."""
from __future__ import annotations

from evaluation.common.judge_cache import JudgeCache, make_key
from evaluation.common.judge_client import JudgeClient

_SYSTEM = (
    "You are a strict geometry perception auditor. Check ONLY whether the model "
    "correctly describes facts visible in the ORIGINAL question diagram. "
    "Do not judge theorem use, reasoning validity, arithmetic, completeness, "
    "or the final answer. IMPORTANT: the model may describe auxiliary lines or "
    "points that it constructed in its reasoning. Those self-constructed elements "
    "are outside this audit: do NOT mark an error merely because they are absent "
    "from the original diagram. Treat claims about model-constructed auxiliary "
    "elements as correct by default and ignore them. Only mark a clear conflict "
    "about an element that is actually present in the original diagram. If the "
    "original image cannot clearly verify a claim, do not mark it as an error."
)

_USER = """Original question text:
{question}

Model's complete solution process:
-----
{reasoning}
-----

Audit only original-diagram perception. Ignore all claims whose subject is a
self-constructed auxiliary line, point, label, angle, or relation; do not mark
such a claim wrong because the element is absent from the original image. A claim
that cannot be clearly verified from the original image is NOT an error.

Return ONLY JSON:
{{
  "perception_correct": 0 or 1,
  "errors": [
    {{"claim": "<verbatim or concise claim>", "error_type": "<type>", "reason": "<why it conflicts with the original image>"}}
  ],
  "uncertain_claims": ["<claim that cannot be verified; these do not lower the score>"],
  "reason": "<short summary>"
}}
"""


def judge_perception(*, original_image: str, question: str, reasoning: str,
                      vision_judge: JudgeClient, cache: JudgeCache) -> dict:
    user = _USER.format(question=question, reasoning=reasoning)
    key = make_key(model=vision_judge.cfg.model, system=_SYSTEM, user=user,
                   image_paths=[original_image])
    cached = cache.get(key)
    if cached is not None:
        return cached
    raw = vision_judge.chat_json(_SYSTEM, user, images=[original_image], max_tokens=3072)
    errors = raw.get("errors", [])
    if not isinstance(errors, list):
        errors = []
    uncertain = raw.get("uncertain_claims", [])
    if not isinstance(uncertain, list):
        uncertain = []
    raw_value = raw.get("perception_correct", 0)
    if isinstance(raw_value, str):
        raw_value = raw_value.strip().lower() in {"1", "true", "yes"}
    out = {
        "perception_correct": 1 if bool(raw_value) else 0,
        "errors": [x if isinstance(x, dict) else {"claim": str(x)} for x in errors][:30],
        "uncertain_claims": [str(x)[:500] for x in uncertain][:30],
        "reason": str(raw.get("reason", ""))[:1000],
    }
    # A binary error verdict must be supported by evidence. Conversely, any
    # returned error is sufficient for a zero under the metric definition.
    if out["errors"]:
        out["perception_correct"] = 0
    elif out["perception_correct"] == 0:
        out["perception_correct"] = 1
        out["reason"] = (out["reason"] + "; invalid zero verdict without evidence").strip("; ")
    cache.put(key, out)
    return out
