"""VLM judge for auxiliary-line usage in reasoning."""
from __future__ import annotations

from evaluation.common.judge_cache import JudgeCache, make_key
from evaluation.common.judge_client import JudgeClient


_SYS = (
    "You are a strict judge evaluating whether a model's reasoning genuinely "
    "USES the auxiliary-line images it generated. You will see the original "
    "diagram, one or more generated auxiliary diagrams, construction "
    "instructions when they were successfully extracted, and the model's "
    "reasoning text. Prioritize the extracted construction instruction; use "
    "direct original/auxiliary image comparison as a fallback only when the "
    "instruction for an image is unavailable."
)

_USER_TEMPLATE = (
    "The FIRST image is the original input diagram. Images 2..N are the "
    "model-generated auxiliary diagrams in generation order. The following "
    "instruction list is aligned with images 2..N; an empty string means that "
    "instruction extraction failed for that image:\n"
    "----- CONSTRUCTION INSTRUCTIONS -----\n"
    "{instructions}\n"
    "----- END CONSTRUCTION INSTRUCTIONS -----\n\n"
    "Evaluation rule (important): for an image with a non-empty extracted "
    "instruction, treat that instruction as the primary description of the "
    "auxiliary construction. Check its use in the reasoning and whether the "
    "image is compatible with it. Do not replace a valid instruction with a "
    "different construction inferred from the image. For an image whose "
    "instruction is empty, infer the newly added elements by comparing the "
    "original image with that auxiliary image, and use this visual inference "
    "as the fallback.\n\n"
    "Model reasoning (with <imageK> tags stripped):\n"
    "----- REASONING -----\n"
    "{reasoning}\n"
    "----- END REASONING -----\n\n"
    "Score on two 0/1/2 axes.\n\n"
    "element_reference: does the reasoning explicitly identify or refer to "
    "the construction/new geometric elements associated with an auxiliary "
    "image? Prefer the extracted instruction when present; otherwise use the "
    "visually inferred new elements.\n"
    "  2 — it clearly names or describes the relevant new point/line/angle/"
    "      triangle/relation.\n"
    "  1 — it vaguely refers to the construction or element, but the reference "
    "is incomplete or ambiguous.\n"
    "  0 — it does not refer to the relevant construction/element, or the "
    "claim is contradicted by the available instruction/image evidence.\n\n"
    "reasoning_integration: does the reasoning use the relevant construction "
    "or new elements to advance the solution consistently?\n"
    "  2 — the construction/new element is used in a meaningful geometric step.\n"
    "  1 — it is mentioned or lightly used, but the step is weak or incomplete.\n"
    "  0 — it is not used, or the claimed use contradicts the instruction/image.\n\n"
    "For non-empty instructions, do not invent a substitute instruction from "
    "the image. For empty instructions, do not invent a textual instruction; "
    "only report visual additions supported by the image comparison.\n\n"
    "Also list only the new elements/constructions supported by the instruction "
    "when present, or by image comparison when instruction is absent.\n\n"
    "Respond ONLY with JSON:\n"
    "{{\n"
    "  \"element_reference\": 0|1|2,\n"
    "  \"element_reference_reason\": \"<short>\",\n"
    "  \"reasoning_integration\": 0|1|2,\n"
    "  \"reasoning_integration_reason\": \"<short>\",\n"
    "  \"new_elements_referenced\": [\"...\", \"...\"],\n"
    "  \"reason\": \"<short>\"\n"
    "}}"
)



def judge_aux_usage(
    *,
    original_image: str,
    aux_images: list[str],
    instructions: list[str],
    reasoning: str,
    vision_judge: JudgeClient,
    cache: JudgeCache,
) -> dict:
    instr_block = "\n".join(
        f"  image{i + 1}: {instruction or '(not extracted; use visual fallback)'}"
        for i, instruction in enumerate(instructions)
    ) or "  (none)"
    user = _USER_TEMPLATE.format(instructions=instr_block, reasoning=reasoning)
    images = [original_image] + list(aux_images)
    key = make_key(
        model=vision_judge.cfg.model,
        system=_SYS,
        user=user,
        image_paths=images,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    raw = vision_judge.chat_json(
        _SYS,
        user,
        images=images,
        max_tokens=2048,
    )

    def _clip(v):
        try:
            v = int(v)
        except Exception:
            return 0
        return max(0, min(2, v))

    elements = raw.get("new_elements_referenced", [])
    if not isinstance(elements, list):
        elements = []
    out = {
        "element_reference": _clip(raw.get("element_reference", raw.get("aux_usage", 0))),
        "element_reference_reason": str(raw.get("element_reference_reason", ""))[:500],
        "reasoning_integration": _clip(raw.get("reasoning_integration", raw.get("aux_usage", 0))),
        "reasoning_integration_reason": str(raw.get("reasoning_integration_reason", ""))[:500],
        "new_elements_referenced": [str(x)[:80] for x in elements][:20],
        "reason": str(raw.get("reason", ""))[:500],
    }
    out["aux_usage"] = (out["element_reference"] + out["reasoning_integration"]) / 2.0
    out["analysis_basis"] = (
        "construction_instruction"
        if instructions and all(instructions)
        else "construction_instruction_with_visual_fallback"
        if any(instructions)
        else "input_output_image_comparison_fallback"
    )
    cache.put(key, out)
    return out
