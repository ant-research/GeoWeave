"""Extract the auxiliary-line construction instruction for each <imageK> in <think>.

Uses the text judge to handle cases where the construction sentence is not the
sentence immediately preceding the tag. Output is cached.
"""
from __future__ import annotations

from evaluation.common.judge_cache import JudgeCache, make_key
from evaluation.common.judge_client import JudgeClient
from evaluation.common.parsing import context_around_tag


_SYS = (
    "You are an expert geometry tutor. Given the reasoning text of a math "
    "problem that contains <image1>, <image2>, ... tags marking auxiliary-line "
    "images, your job is to identify, for a target tag <imageK>, the construction "
    "instruction that produces the new diagram at that tag."
)

_USER_TEMPLATE = (
    "Below is an excerpt of the reasoning around the target tag <image{k}>.\n"
    "----- EXCERPT -----\n"
    "{excerpt}\n"
    "----- END EXCERPT -----\n\n"
    "Find ONLY the explicit visual construction or diagram-editing action that "
    "produces the diagram at <image{k}>. The instruction is usually a short verb "
    "phrase like 'Connect OB and OC', 'Draw a perpendicular from A to BC at foot M', "
    "or 'Extend segment AB to point E'. Do not return theorem statements, equations, "
    "calculations, assumptions, or general reasoning merely because they are near "
    "the tag. Keep instruction within 40 Chinese characters (or 80 ASCII characters). "
    "If no explicit drawing/editing action exists, return an empty instruction and "
    "found=false; do not copy nearby reasoning sentences.\n\n"
    "Respond ONLY with a JSON object:\n"
    "{{\"instruction\": \"<short construction action or empty string>\", "
    "\"found\": true/false}}"
)


def extract_instruction(
    *,
    think: str,
    tag_index: int,
    text_judge: JudgeClient,
    cache: JudgeCache,
) -> dict:
    excerpt = context_around_tag(think, tag_index, window_chars=600)
    if not excerpt:
        return {"instruction": "", "found": False}
    user = _USER_TEMPLATE.format(k=tag_index, excerpt=excerpt)
    key = make_key(model=text_judge.cfg.model, system=_SYS, user=user)
    cached = cache.get(key)
    if cached is not None:
        return cached
    raw = text_judge.chat_json(_SYS, user, max_tokens=2048)
    out = {
        "instruction": str(raw.get("instruction", ""))[:500],
        "found": bool(raw.get("found", False)),
    }
    cache.put(key, out)
    return out
