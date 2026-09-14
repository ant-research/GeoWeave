"""VLM judge that scores an auxiliary-line image on two axes."""
from __future__ import annotations

from evaluation.common.judge_cache import JudgeCache, make_key
from evaluation.common.judge_client import JudgeClient


_SYS = (
    "You are a strict visual judge for geometry-diagram quality. You will be "
    "shown two images and a construction instruction. The FIRST image is the "
    "original question diagram. The SECOND image is the model-generated "
    "auxiliary-line diagram that should add the auxiliary construction on top "
    "of the original diagram. Score it on two axes."
)

_USER_TEMPLATE = (
    "Construction instruction:\n  {instruction}\n\n"
    "Scoring rubric (output integers in [0,2]).\n\n"
    "element_preservation: are ALL geometric elements from the original "
    "(image 1) present in the aux image (image 2) at the same positions?\n"
    "  2 — every original point/line/label is present and in the correct place.\n"
    "  1 — minor drift or label change, but all required elements are still "
    "      recognizable and usable.\n"
    "  0 — at least one required element is missing, severely displaced, or "
    "      corrupted such that downstream reasoning would be misled.\n\n"
    "instruction_consistency: does the aux image render the new construction "
    "exactly as described in the instruction, clearly and legibly?\n"
    "  2 — the new line/point/marking is drawn exactly as instructed AND is "
    "      clearly visible and well-labeled.\n"
    "  1 — the construction is roughly correct but has a noticeable error "
    "      (wrong endpoint, slightly off perpendicular, blurry label).\n"
    "  0 — the instructed construction is missing, wrong type (e.g. asked for "
    "      perpendicular but drew arbitrary line), or unreadable.\n\n"
    "Respond ONLY with JSON:\n"
    "{{\n"
    "  \"element_preservation\": 0|1|2,\n"
    "  \"element_preservation_reason\": \"<short>\",\n"
    "  \"instruction_consistency\": 0|1|2,\n"
    "  \"instruction_consistency_reason\": \"<short>\"\n"
    "}}"
)


def judge_aux_image(
    *,
    original_image: str,
    aux_image: str,
    instruction: str,
    vision_judge: JudgeClient,
    cache: JudgeCache,
) -> dict:
    user = _USER_TEMPLATE.format(instruction=instruction or "(not extracted)")
    key = make_key(
        model=vision_judge.cfg.model,
        system=_SYS,
        user=user,
        image_paths=[original_image, aux_image],
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    raw = vision_judge.chat_json(
        _SYS,
        user,
        images=[original_image, aux_image],
        max_tokens=4096,
    )

    def _clip(v):
        try:
            v = int(v)
        except Exception:
            return 0
        return max(0, min(2, v))

    out = {
        "element_preservation": _clip(raw.get("element_preservation", 0)),
        "element_preservation_reason": str(raw.get("element_preservation_reason", ""))[:500],
        "instruction_consistency": _clip(raw.get("instruction_consistency", 0)),
        "instruction_consistency_reason": str(raw.get("instruction_consistency_reason", ""))[:500],
    }
    cache.put(key, out)
    return out
