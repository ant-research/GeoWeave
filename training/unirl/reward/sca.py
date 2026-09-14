"""SCA token-credit construction and prompt-group normalization."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import torch

from unirl.distributed.tensor.ref import hydrate
from unirl.types.rollout_resp import RolloutTrack, _track_with_field
from unirl.types.segments import TextSegment


def _annotation_step_weights(
    annotation: Dict[str, Any],
) -> List[tuple[int, int, float]]:
    steps = list(annotation.get("steps") or [])
    weights = list(annotation.get("step_error_weights") or [])
    result: List[tuple[int, int, float]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        try:
            start = int(step["char_start"])
            end = int(step["char_end"])
            weight = float(weights[index]) if index < len(weights) else 0.0
        except (KeyError, TypeError, ValueError):
            continue
        if end > start and weight > 0.0:
            result.append((start, end, weight))
    return result


def compute_sca_token_advantages(
    track: RolloutTrack,
    *,
    outcome_correct_credit: float = 2.0,
    process_error_penalty: float = 1.0,
    eps: float = 1e-6,
) -> RolloutTrack:
    """Attach packed SCA rewards and group-relative per-token advantages.

    Normalization is performed independently for every prompt group over all
    valid decoded tokens from all sibling rollouts in that group. Invalid judge
    rows, status=0 rows, and stripped special/end-marker tokens are excluded.
    """
    segment = track.segment
    annotations = track.process_annotations
    if not isinstance(segment, TextSegment) or segment.tokens is None or segment.cu_seqlens is None:
        raise ValueError("SCA requires a packed TextSegment with tokens")
    if segment.decoded_char_starts is None or segment.decoded_char_ends is None:
        raise ValueError("SCA requires decoded token character spans from the rollout pipeline")
    if annotations is None or len(annotations) != track.batch_size:
        raise ValueError(
            "SCA process_annotations must be sample-aligned: "
            f"annotations={None if annotations is None else len(annotations)}, "
            f"batch={track.batch_size}"
        )

    starts = hydrate(segment.decoded_char_starts).to(torch.long).cpu()
    ends = hydrate(segment.decoded_char_ends).to(torch.long).cpu()
    total_tokens = int(starts.numel())
    eligibility = torch.ones(total_tokens, dtype=torch.float32)
    if segment.loss_mask is not None:
        eligibility = hydrate(segment.loss_mask).to(torch.float32).cpu().reshape(-1)
        if eligibility.numel() != total_tokens:
            raise ValueError("SCA loss_mask must align 1:1 with packed tokens")
        eligibility = (eligibility > 0.5).to(torch.float32)
    if ends.numel() != total_tokens:
        raise ValueError("SCA decoded character spans must align 1:1 with packed tokens")

    cu = hydrate(segment.cu_seqlens).to(torch.long).cpu()
    if int(cu[-1].item()) != total_tokens:
        raise ValueError("SCA decoded character spans must align 1:1 with packed tokens")
    status = None
    if track.status is not None:
        status = hydrate(track.status).to(torch.float32).cpu().reshape(-1)

    process_error = torch.zeros(total_tokens, dtype=torch.float32)
    raw_credit = torch.zeros(total_tokens, dtype=torch.float32)
    token_mask = torch.zeros(total_tokens, dtype=torch.float32)
    sample_valid = torch.zeros(track.batch_size, dtype=torch.float32)

    for sample_index, annotation in enumerate(annotations):
        begin, finish = int(cu[sample_index]), int(cu[sample_index + 1])
        sample_starts = starts[begin:finish]
        sample_ends = ends[begin:finish]
        visible = sample_ends > sample_starts
        valid_annotation = isinstance(annotation, dict) and bool(annotation.get("valid", False))
        valid_status = status is None or bool(status[sample_index].item() > 0.5)
        if not valid_annotation or not valid_status:
            continue
        sample_valid[sample_index] = 1.0

        local_error = torch.zeros(finish - begin, dtype=torch.float32)
        for char_start, char_end, weight in _annotation_step_weights(annotation):
            overlap = (sample_ends > char_start) & (sample_starts < char_end)
            local_error = torch.maximum(
                local_error,
                overlap.to(torch.float32) * float(weight),
            )
        answer_correct = float(bool(annotation.get("answer_correct", False)))
        process_error[begin:finish] = local_error
        raw_credit[begin:finish] = (
            float(outcome_correct_credit) * answer_correct
            - float(process_error_penalty) * local_error
        )
        token_mask[begin:finish] = (
            visible.to(torch.float32) * eligibility[begin:finish]
        )

    token_advantages = torch.zeros_like(raw_credit)
    group_to_token_indices: Dict[str, List[torch.Tensor]] = {}
    for sample_index, group_id in enumerate(track.group_ids):
        begin, finish = int(cu[sample_index]), int(cu[sample_index + 1])
        local_indices = torch.arange(begin, finish, dtype=torch.long)
        local_indices = local_indices[token_mask[begin:finish] > 0.5]
        if local_indices.numel():
            group_to_token_indices.setdefault(str(group_id), []).append(local_indices)

    for chunks in group_to_token_indices.values():
        indices = torch.cat(chunks)
        values = raw_credit[indices]
        mean = values.mean()
        variance = values.var(unbiased=False)
        token_advantages[indices] = (values - mean) / torch.sqrt(variance + float(eps))

    updated_segment = copy.copy(segment)
    updated_segment.process_error_mask = process_error
    updated_segment.raw_token_credit = raw_credit
    updated_segment.sca_token_mask = token_mask
    updated_segment.token_advantages = token_advantages

    updated = _track_with_field(track, "segment", updated_segment)
    updated = _track_with_field(updated, "status", sample_valid)
    # The training stack requires a sample-level field. SCA's actual signal is
    # packed on the segment and GRPO consumes it directly.
    placeholder = torch.zeros(track.batch_size, dtype=torch.float32)
    return _track_with_field(updated, "advantages", placeholder)


__all__ = ["compute_sca_token_advantages"]
