"""GeoWeave AR condition container.

Mirrors :mod:`unirl.models.bagel.conditions`.

Key design points:

- **AR conditions** store raw prompt queries (strings) rather than pre-tokenized
  splits, because GeoWeave uses standard batched attention (not navit packed
  sequences). Both rollout and replay tokenize on-the-fly.

- For **interleave** mode, the AR conditions also store:
  - ``generated_images``: per-sample list of generated image tensors (used
    during replay to re-encode through the understanding ViT)
  - ``text_segment_boundaries``: per-sample list of (start, end) tuples marking
    where each text segment falls in the packed token sequence
  - ``latent_segments``: per-sample list of :class:`LatentSegment` from the
    interleave diffusion (one per generated image). Consumed by
    :class:`InterleaveRL` for inline velocity-MSE.

There is no separate diffusion-side condition class: GeoWeave's MoT
architecture invalidates :class:`FlowMSE` (see history.md 2026-07-14 续), and
:class:`InterleaveRL` reads its diffusion inputs directly from
:class:`SensenovaU1ARConditions.latent_segments` plus a live KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple

import torch

from unirl.distributed.tensor.batch import concat_field
from unirl.types.conditions.base import Condition, Modality

if TYPE_CHECKING:
    from unirl.types.segments.latent import LatentSegment


@dataclass
class SensenovaU1ARConditions(Condition):
    """Per-sample prompt material for the SensenovaU1 AR (text-out) stage.

    For standard (non-interleave) mode, ``prompt_queries`` is all that is needed.
    For interleave mode, ``generated_images`` and ``text_segment_boundaries`` are
    populated during rollout and consumed during replay.
    """

    modality: ClassVar[Modality] = Modality.TEXT

    # Chat-template-formatted prompt strings, one per sample.
    prompt_queries: List[str] = concat_field(default_factory=list)

    # Stable rollout identity used by batching and performance metrics.
    rollout_ids: List[str] = concat_field(default_factory=list)
    prompt_ids: List[str] = concat_field(default_factory=list)
    rewrite_ids: List[int] = concat_field(default_factory=list)

    # Optional input images for image-understanding tasks (VQA, editing).
    pixel_values: List[Optional[torch.Tensor]] = concat_field(default_factory=list)

    # ViT grid sizes for input images, one per sample.
    grid_hws: List[Optional[torch.Tensor]] = concat_field(default_factory=list)

    # --- Interleave-specific fields ---

    # Per-sample list of generated image tensors [3, H, W] from interleave rollout.
    # Used during replay to re-encode through the understanding ViT.
    generated_images: List[List[torch.Tensor]] = concat_field(default_factory=list)

    # Per-sample list of (start, end) tuples marking each text segment's span
    # in the packed token sequence. Used during replay to score each segment.
    text_segment_boundaries: List[List[Tuple[int, int]]] = concat_field(
        default_factory=list
    )

    # Per-sample list of LatentSegments from interleave diffusion (one per
    # generated image). Used by InterleaveRL for inline velocity-MSE.
    latent_segments: List[List["LatentSegment"]] = concat_field(default_factory=list)

    # Per-sample image generation size (H, W) from smart_resize.
    # Falls back to ARParams.image_size when empty.
    image_sizes: List[Tuple[int, int]] = concat_field(default_factory=list)

    # Per-sample rollout termination reason: eos, max_new_tokens, or max_images.
    stop_reasons: List[str] = concat_field(default_factory=list)

    # Number of generated-image understanding tokens appended to the KV cache,
    # including each image's closing </img> token. These tokens are not part of
    # TextSegment.tokens and therefore do not consume max_new_tokens.
    image_context_token_counts: List[int] = concat_field(default_factory=list)

    # Lightweight per-sample rollout profile. Dynamic trainside scheduling uses
    # this metadata to aggregate phase timings on the driver without collectives.
    rollout_phase_metrics: List[Dict[str, Any]] = concat_field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.prompt_queries)

    @classmethod
    def for_sample(
        cls,
        *,
        query: str,
        pixel_values: Optional[torch.Tensor] = None,
        grid_hw: Optional[torch.Tensor] = None,
        generated_images: Optional[List[torch.Tensor]] = None,
        text_segment_boundaries: Optional[List[Tuple[int, int]]] = None,
        latent_segments: Optional[List["LatentSegment"]] = None,
        stop_reason: str = "unknown",
        image_context_token_count: int = 0,
    ) -> "SensenovaU1ARConditions":
        return cls(
            prompt_queries=[query],
            rollout_ids=["sample-0"],
            prompt_ids=["prompt-0"],
            rewrite_ids=[0],
            pixel_values=[pixel_values],
            grid_hws=[grid_hw],
            generated_images=[generated_images or []],
            text_segment_boundaries=[text_segment_boundaries or []],
            latent_segments=[latent_segments or []],
            stop_reasons=[stop_reason],
            image_context_token_counts=[int(image_context_token_count)],
            rollout_phase_metrics=[{}],
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SensenovaU1ARConditions":
        val = d.get("sensenova_u1_ar")
        if isinstance(val, cls):
            return val
        raise ValueError(
            "SensenovaU1ARConditions.from_dict: expected a 'sensenova_u1_ar' key "
            f"holding a SensenovaU1ARConditions instance; got keys {sorted(d.keys())}."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"sensenova_u1_ar": self}


__all__ = ["SensenovaU1ARConditions"]
