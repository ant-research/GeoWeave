"""Utilities for aligning generated token IDs with decoded text characters."""

from __future__ import annotations

from typing import Any, Iterable, Sequence, Tuple

import torch


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def decode_with_token_char_spans(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    stop_markers: Iterable[str] = ("<|im_end|>", "</s>"),
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Decode IDs and return one conservative character span per token.

    Prefix decodes are compared with the final decode instead of independently
    decoding each token. This preserves context-sensitive whitespace and safely
    handles byte-fallback pieces: incomplete byte pieces receive an empty span,
    while the completing token receives the resulting characters.
    """
    ids = [int(token_id) for token_id in token_ids]
    if not ids:
        empty = torch.empty(0, dtype=torch.long)
        return "", empty, empty.clone()

    full_text = str(tokenizer.decode(ids, skip_special_tokens=False))
    visible_end = len(full_text)
    for marker in stop_markers:
        marker_index = full_text.find(marker)
        if marker_index >= 0:
            visible_end = min(visible_end, marker_index)
    visible_text = full_text[:visible_end]

    boundaries = [0]
    previous = 0
    for token_count in range(1, len(ids) + 1):
        prefix = str(tokenizer.decode(ids[:token_count], skip_special_tokens=False))
        boundary = min(visible_end, _common_prefix_length(prefix, full_text))
        # Some tokenizers rewrite a trailing replacement character when the next
        # byte arrives. Keep packed spans monotonic and assign the completed text
        # to the later token rather than producing overlapping/backward offsets.
        boundary = max(previous, boundary)
        boundaries.append(boundary)
        previous = boundary

    starts = torch.tensor(boundaries[:-1], dtype=torch.long)
    ends = torch.tensor(boundaries[1:], dtype=torch.long)
    return visible_text, starts, ends


__all__ = ["decode_with_token_char_spans"]
