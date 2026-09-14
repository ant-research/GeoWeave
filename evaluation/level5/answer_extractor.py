"""Level 3 answer extraction and normalization.

Level 3 answers are not always a single LaTeX span.  They may mix plain text
and math (``6$\\text{cm}$``), contain tuples/multiple blanks, or be a short
proof/conclusion.  Therefore this module deliberately preserves the complete
final-answer candidate instead of selecting the last ``$...$`` span.
"""
from __future__ import annotations

import re

from evaluation.common.parsing import _extract_boxed


_TAG_ONLY_RE = re.compile(r"^</?(?:image\d*|think|aux\d*|img\d*)>$", re.IGNORECASE)
_FENCE_ONLY_RE = re.compile(r"^\$+\s*$")
_ANSWER_MARKER_RE = re.compile(
    r"(?i)(?:"
    r"(?:thus|therefore|hence|so)[,\s]*(?:the\s+)?answers?\s*(?:is|are)?"
    r"|(?:the\s+)?answers?\s*(?:is|are)?"
    r"|(?:故|因此|所以)?答案\s*(?:是|为)?"
    r")\s*[:：]?\s*(.*)$"
)


def _clean_candidate(s: str) -> str:
    """Remove presentation wrappers without destroying answer semantics.

    In particular, a leading ``-`` is retained because it may be a numeric
    sign (the old common cleaner turned ``-6`` into ``6``).
    """
    s = (s or "").strip()
    s = s.removeprefix("Extracted answer:").strip()
    s = s.strip("` \t")
    s = re.sub(r"^(?:is|are|是|为)\s*[:：]?\s*", "", s, flags=re.IGNORECASE)
    # Markdown bullets are removed only when the marker is followed by
    # whitespace.  Thus "- answer" is a bullet, while "-6" remains negative.
    s = re.sub(r"^[-*+]\s+", "", s)
    s = s.strip("` \t")
    return s.rstrip(".,;。；，．")


def _next_answer_line(lines: list[str], start: int) -> str:
    for line in lines[start:]:
        line = line.strip()
        if not line or _TAG_ONLY_RE.fullmatch(line) or _FENCE_ONLY_RE.fullmatch(line):
            continue
        return _clean_candidate(line)
    return ""


def extract_pred(text: str) -> str:
    """Extract a complete Level 3 final answer.

    Priority is ``\\boxed`` > explicit answer marker > last non-empty line.
    Unlike the generic extractor, this does *not* replace a mixed/textual
    candidate with its last math span.  This preserves units, coefficients,
    tuple components, multiple answers and propositions.
    """
    text = text or ""
    boxed = _extract_boxed(text)
    if boxed is not None:
        return _clean_candidate(boxed)

    lines = text.splitlines()
    marked: list[str] = []
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or _TAG_ONLY_RE.fullmatch(line):
            continue
        m = _ANSWER_MARKER_RE.search(line.strip("*_` \t"))
        if not m:
            continue
        candidate = _clean_candidate(m.group(1))
        if not candidate:
            candidate = _next_answer_line(lines, i + 1)
        if candidate:
            marked.append(candidate)
    if marked:
        return marked[-1]

    for raw_line in reversed(lines):
        line = raw_line.strip()
        if line and not _TAG_ONLY_RE.fullmatch(line) and not _FENCE_ONLY_RE.fullmatch(line):
            return _clean_candidate(line)
    return ""


_NORM_REPLACEMENTS = [
    (re.compile(r"\\degree(?![a-zA-Z])"), ""),
    (re.compile(r"\bdegrees?\b", re.IGNORECASE), ""),
    (re.compile(r"度"), ""),
    (re.compile(r"°"), ""),
    (re.compile(r"\\cdot"), "*"),
    (re.compile(r"\\times"), "*"),
    (re.compile(r"\\pi"), "pi"),
]
_LATEX_TEXT_RE = re.compile(r"\\(?:text|mathrm|textrm|operatorname)\s*\{([^{}]*)\}")


def normalize_answer(s: str) -> str:
    """Normalize ASCII, LaTeX and mixed ASCII/LaTeX answer formatting.

    Dollar delimiters are formatting rather than answer content, so all
    unescaped delimiters are removed.  Text/unit commands are unwrapped, which
    makes ``6$\\text{cm}$``, ``$6\\,\\mathrm{cm}$`` and ``6cm`` comparable
    while retaining the complete expression.
    """
    s = (s or "").strip()
    s = re.sub(r"\\(?:left|right)(?=[\[\](){}|.])", "", s)
    s = re.sub(r"(?<!\\)\$+", "", s)
    s = s.replace(r"\(", "").replace(r"\)", "")
    s = s.replace(r"\[", "").replace(r"\]", "")
    # Repeated substitution supports simple nested formatting wrappers.
    previous = None
    while previous != s:
        previous = s
        s = _LATEX_TEXT_RE.sub(r"\1", s)
    s = re.sub(r"\\(?:,|;|:|!|quad|qquad)", "", s)
    s = re.sub(r"(?<!\\)\\\s+", "", s)  # Single LaTeX control-space: ``\ cm``.
    for pat, repl in _NORM_REPLACEMENTS:
        s = pat.sub(repl, s)
    # Preserve the previous evaluator's handling of count answers while still
    # keeping richer textual/proof answers intact.
    if re.fullmatch(r"(?:有且只有)?一条", s.strip()):
        return "1条"
    return s.strip()
