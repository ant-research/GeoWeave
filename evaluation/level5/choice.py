"""Multiple-choice parsing helpers for Level 3 evaluation.

Some benchmark answers store only an option label (``A``-``D``), while models
may emit either that label or the corresponding option text/value.  These
helpers recover the option table from the original question so both forms can
be scored against the same schema.
"""
from __future__ import annotations

import re


_GT_LABEL_RE = re.compile(
    r"^\s*(?:answer\s*[:：]\s*)?([A-D])(?:[.。])?\s*$",
    re.IGNORECASE,
)
_PRED_LABEL_PATTERNS = (
    _GT_LABEL_RE,
    re.compile(
        r"^\s*(?:答案(?:是|为)?|选择|选|故选|choice|option|answer(?:\s+is)?)"
        r"\s*[:：]?\s*(?:\\boxed\s*\{\s*)?([A-D])\s*\}?\s*[.。]?\s*$",
        re.IGNORECASE,
    ),
)
# Supports ``A.``, ``A:``, ``（A）``, markdown ``**A.**`` / ``**A**.`` and
# inline Chinese option lists immediately following ``（　）``.
_OPTION_MARKER_RE = re.compile(
    r"(?<![A-Za-z0-9])[-*]?\s*(?:\*\*)?[（(]?([A-D])[）)]?"
    r"(?:\*\*)?\s*[.．、:：]\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)
_MATH_SPAN_RE = re.compile(r"\$+([^$]+?)\$+")


def parse_choice_label(text: str, *, prediction: bool = False) -> str | None:
    """Return a canonical ``A``-``D`` label when *text* explicitly selects one."""
    patterns = _PRED_LABEL_PATTERNS if prediction else (_GT_LABEL_RE,)
    for pattern in patterns:
        match = pattern.fullmatch(text or "")
        if match:
            return match.group(1).upper()
    return None


def parse_choice_options(question: str) -> dict[str, str]:
    """Parse a four-option A-D table from a benchmark question.

    We select a contiguous A/B/C/D marker sequence rather than treating every
    ``A.`` in prose as an option.  The last complete sequence is preferred,
    since actual choices normally occur after the question body.
    """
    matches = list(_OPTION_MARKER_RE.finditer(question or ""))
    selected = None
    for index in range(max(0, len(matches) - 3)):
        group = matches[index:index + 4]
        if [m.group(1).upper() for m in group] == list("ABCD"):
            selected = group
    if selected is None:
        return {}

    options: dict[str, str] = {}
    for index, marker in enumerate(selected):
        end = selected[index + 1].start() if index + 1 < len(selected) else len(question)
        value = question[marker.end():end].strip()
        value = re.sub(r"\s*[-*]+\s*$", "", value).strip()
        options[marker.group(1).upper()] = value
    return options


def option_answer_candidates(option_text: str) -> list[str]:
    """Return conservative answer candidates represented by one option.

    The complete option text is retained.  Standalone LaTeX spans are also
    exposed so ``$80^\\circ$`` matches ``80^\\circ``.  Ambiguity is handled by
    matching against *all* options before accepting a label.
    """
    text = (option_text or "").strip()
    candidates = [text]
    candidates.extend(m.group(1).strip() for m in _MATH_SPAN_RE.finditer(text))
    candidates.extend(
        re.sub(r"[*`]", "", candidate).strip()
        for candidate in list(candidates)
    )
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))
