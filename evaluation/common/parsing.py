"""Parsing helpers for model outputs.

Conventions:
  - <image0> in the prompt is the question diagram; <image1>..<imageN> in <think>
    are auxiliary-line images. images[k-1] corresponds to <imageK> (k >= 1).
  - The <think>...</think> block holds the reasoning trace. If absent, the full
    text is treated as the reasoning trace.
  - The final answer is extracted with `extract_answer` (boxed > Answer: > last line).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_IMAGE_TAG_RE = re.compile(r"<image(\d*)>")
_ANSWER_MARKER_RE = re.compile(
    r"(?i)(?:"
    r"(?:thus|therefore|hence|so)[,\s]*(?:the\s+)?answers?\s*(?:is|are)?"
    r"|(?:the\s+)?answers?\s*(?:is|are)?"
    r"|(?:故|因此|所以)?答案\s*(?:是|为)?"
    r")\s*[:：]?\s*(.*)$"
)
_MATH_SPAN_RE = re.compile(r"\$+([^$]+?)\$+")


@dataclass
class ParsedOutput:
    think: str                       # contents inside <think>...</think>, or full text
    final: str                       # text after </think>, or "" if no closing tag
    aux_indices: list[int]           # ordered, unique <imageK> indices in <think> (K >= 1)


def parse_output(text: str) -> ParsedOutput:
    m = _THINK_RE.search(text)
    if m:
        think = m.group(1)
        final = text[m.end():]
    else:
        think = text
        final = ""
    seen: set[int] = set()
    indices: list[int] = []
    next_bare_idx = 1
    for tm in _IMAGE_TAG_RE.finditer(think):
        raw = tm.group(1)
        if raw:
            k = int(raw)
            next_bare_idx = max(next_bare_idx, k + 1)
        else:
            while next_bare_idx in seen:
                next_bare_idx += 1
            k = next_bare_idx
            next_bare_idx += 1
        if k >= 1 and k not in seen:
            seen.add(k)
            indices.append(k)
    return ParsedOutput(think=think, final=final, aux_indices=indices)


def context_around_tag(think: str, tag_index: int, window_chars: int = 600) -> str:
    """Slice reasoning around an auxiliary tag, padded by window_chars.

    The SFT/eval format uses <image1>, <image2>, ...; upstream U1 interleave
    inference may emit bare <image> tags. Multiple bare tags are interpreted
    in order as <image1>, <image2>, ...
    """
    seen: set[int] = set()
    next_bare_idx = 1
    target = None
    for m in _IMAGE_TAG_RE.finditer(think):
        raw = m.group(1)
        if raw:
            k = int(raw)
            next_bare_idx = max(next_bare_idx, k + 1)
        else:
            while next_bare_idx in seen:
                next_bare_idx += 1
            k = next_bare_idx
            next_bare_idx += 1
        if k >= 1:
            seen.add(k)
        if k == tag_index:
            target = m
            break
    if target is None:
        return ""
    lo = max(0, target.start() - window_chars)
    hi = min(len(think), target.end() + window_chars)
    return think[lo:hi]


def strip_image_tags(text: str) -> str:
    """Remove <imageK> tags from text (used when feeding reasoning to judges)."""
    return _IMAGE_TAG_RE.sub("", text)


def aux_image_path(images: list[str], k: int) -> Optional[str]:
    """Return the file path for <image{k}> (k >= 1) given the model's `images` list."""
    if k < 1 or k > len(images):
        return None
    return images[k - 1]


_TAG_ONLY_RE = re.compile(r"^</?(?:image\d*|think|aux\d*|img\d*)>$", re.IGNORECASE)
_FENCE_ONLY_RE = re.compile(r"^\$+\s*$")


def _clean_answer_candidate(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*[-*+]\s*", "", s)
    s = s.strip("*_` \t")
    s = s.removeprefix("Extracted answer:").strip()
    s = s.strip("*_` \t")
    s = re.sub(r"^(?:is|are|是|为)\s*[:：]?\s*", "", s, flags=re.IGNORECASE)
    if re.search(r"(?i)\b(?:exactly|only)\s+one\b|有且只有一条|只有一条|一条", s):
        return "1"
    spans = [m.group(1).strip() for m in _MATH_SPAN_RE.finditer(s) if m.group(1).strip()]
    if spans:
        last = spans[-1].strip()
        # Use math spans for numeric/symbolic answers, but avoid object labels
        # such as $AB$ in textual answers like "parallel to $AB$".
        if re.search(r"\d|\\(?:frac|dfrac|sqrt|pi|sin|cos|tan|arcsin|arccos|arctan|degree|text)|[\^_=+\-*/]", last):
            return last.rstrip(".,;。；，")
    return s.strip().rstrip(".,;。；，")


def _next_answer_line(lines: list[str], start: int) -> str:
    for line in lines[start:]:
        line = line.strip()
        if not line or _TAG_ONLY_RE.match(line) or _FENCE_ONLY_RE.match(line):
            continue
        return _clean_answer_candidate(line)
    return ""


def extract_answer(text: str) -> str:
    """boxed > Answer: > last non-empty line (skipping bare tags)."""
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed.strip()

    lines = [line.strip() for line in text.splitlines()]
    marked: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if not line or _TAG_ONLY_RE.match(line):
            continue
        m = _ANSWER_MARKER_RE.search(line.strip("*_` \t"))
        if not m:
            continue
        candidate = _clean_answer_candidate(m.group(1))
        if not candidate:
            candidate = _next_answer_line(lines, i + 1)
        if candidate:
            marked.append((i, candidate))
    if marked:
        return marked[-1][1]

    for line in reversed(lines):
        line = line.strip()
        if line and not _TAG_ONLY_RE.match(line) and not _FENCE_ONLY_RE.match(line):
            return _clean_answer_candidate(line)
    return ""


def _extract_boxed(s: str) -> Optional[str]:
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    if i >= len(s):
        return None
    if s[i] != "{":
        return s[i:].split("$")[0].strip()
    stack = 1
    out = []
    j = i + 1
    while j < len(s) and stack > 0:
        c = s[j]
        if c == "{":
            stack += 1
            out.append(c)
        elif c == "}":
            stack -= 1
            if stack == 0:
                break
            out.append(c)
        else:
            out.append(c)
        j += 1
    return "".join(out)
