"""Helpers for the offline ground-truth auxiliary instructions used by Mode C.

Mode C is the guided ideal upper bound for Mode A.  It consumes only the
minimal diagram-editing instructions extracted offline from ``thinking``. The
canonical text field is already merged offline into one paragraph; the full
chain of thought must never be inserted into an inference prompt.
"""
from __future__ import annotations

import re
from typing import Any


GT_AUX_INSTRUCTIONS_FIELD = "gt_aux_instructions"
GT_AUX_INSTRUCTIONS_TEXT_FIELD = "gt_aux_instructions_text"
_IMAGE_MARKER_RE = re.compile(r"<image(\d+)>", re.IGNORECASE)


def image_marker_indices(text: str) -> list[int]:
    return [int(m.group(1)) for m in _IMAGE_MARKER_RE.finditer(text or "")]


def format_gt_aux_instructions(entries: list[dict[str, Any]]) -> str:
    """Merge ordered instructions into the canonical prompt-ready paragraph."""
    return " ".join(str(entry["instruction"]).strip() for entry in entries)


def normalized_gt_aux_instructions(item: dict, *, strict: bool = True) -> list[dict[str, Any]]:
    """Return and validate the structured offline instructions for one item.

    Under strict Mode-C semantics there must be exactly one non-empty entry for
    every GT auxiliary image / ``<imageN>`` marker, in consecutive order.
    """
    raw = item.get(GT_AUX_INSTRUCTIONS_FIELD)
    if not isinstance(raw, list):
        if strict:
            raise ValueError(f"missing {GT_AUX_INSTRUCTIONS_FIELD}")
        return []

    entries: list[dict[str, Any]] = []
    for pos, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"{GT_AUX_INSTRUCTIONS_FIELD}[{pos - 1}] must be an object")
        try:
            idx = int(entry.get("image_index"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid image_index at instruction {pos}") from exc
        instruction = str(entry.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"empty instruction for <image{idx}>")
        entries.append({"image_index": idx, "instruction": instruction})

    expected_from_images = max(0, len(item.get("images") or []) - 1)
    marker_indices = image_marker_indices(str(item.get("thinking", "")))
    expected_indices = list(range(1, expected_from_images + 1))
    actual_indices = [entry["image_index"] for entry in entries]

    if actual_indices != expected_indices:
        raise ValueError(
            f"instruction indices {actual_indices} do not match GT images {expected_indices}"
        )
    if marker_indices and marker_indices != expected_indices:
        raise ValueError(
            f"thinking markers {marker_indices} do not match GT images {expected_indices}"
        )
    return entries


def get_gt_aux_instructions_text(item: dict, *, strict: bool = True) -> str:
    """Read Mode-C instructions, preferring and validating the structured field."""
    entries = normalized_gt_aux_instructions(item, strict=strict)
    if entries:
        canonical = format_gt_aux_instructions(entries)
        stored = str(item.get(GT_AUX_INSTRUCTIONS_TEXT_FIELD, "")).strip()
        if strict and not stored:
            raise ValueError(f"missing {GT_AUX_INSTRUCTIONS_TEXT_FIELD}")
        if strict and stored != canonical:
            raise ValueError(
                f"{GT_AUX_INSTRUCTIONS_TEXT_FIELD} is inconsistent with "
                f"{GT_AUX_INSTRUCTIONS_FIELD}"
            )
        return canonical

    if not strict:
        return str(item.get(GT_AUX_INSTRUCTIONS_TEXT_FIELD, "")).strip()
    raise ValueError("no GT auxiliary instructions")
