"""SensenovaU1Bundle — loads the NEOChatModel and exposes it for RL training.

Mirrors :class:`unirl.models.bagel.bundle.BagelBundle`.

The bundle owns one NEOChatModel (Qwen3 MoT backbone + fm_modules + ViT),
one tokenizer, and geometry constants. The trainable surface is
``model.language_model`` (the Qwen3 MoT), aliased as ``self.transformer``
so recipes can set ``backend.trainable_attr: transformer``.

Freezing logic:
    1. Everything starts frozen (``model.requires_grad_(False)``).
    2. Under LoRA (``use_lora=True``): adapters are injected later by the
       backend; nothing is unfrozen here.
    3. Under full fine-tuning (``use_lora=False``): understanding and/or
       generation branch params are selectively unfrozen based on
       ``freeze_und`` / ``freeze_gen`` / ``freeze_fm_modules``.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import (
    SensenovaU1PipelineConfig,
    _GEN_LAYER_PARAM_NAMES,
    _UND_LAYER_PARAM_NAMES,
)

logger = logging.getLogger(__name__)

SENSENOVA_U1_FSDP_BLOCK_CLASS = "Qwen3DecoderLayer"


def _restore_vision_rope_buffers(model: nn.Module) -> None:
    """Rebuild non-persistent 2D-RoPE caches zeroed by Transformers 5 loader.

    ``NEOVisionEmbeddings`` registers cos/sin_cached_{x,y} with
    ``persistent=False``. Transformers 5.6 overwrites them with
    ``torch.empty_like`` after ``from_pretrained`` — the four buffers become
    garbage. Rebuild them from the same ``precompute_rope_freqs_sincos`` the
    vendor constructor uses, for BOTH understanding and generation Vision.
    """
    from .vendor.modeling_neo_vit import precompute_rope_freqs_sincos

    vision_modules = []
    if hasattr(model, "vision_model"):
        vision_modules.append(("und", model.vision_model))
    if hasattr(model, "fm_modules") and "vision_model_mot_gen" in model.fm_modules:
        vision_modules.append(("gen", model.fm_modules["vision_model_mot_gen"]))

    for tag, vm in vision_modules:
        emb = vm.embeddings if hasattr(vm, "embeddings") else None
        if emb is None:
            continue
        cfg = emb.config
        dim = emb.rope_dim_part
        max_pos = cfg.max_position_embeddings_vision
        theta = cfg.rope_theta_vision
        cos_x, sin_x = precompute_rope_freqs_sincos(dim, max_pos, base=theta)
        cos_y, sin_y = precompute_rope_freqs_sincos(dim, max_pos, base=theta)
        emb.cos_cached_x = cos_x.to(device=emb.cos_cached_x.device, dtype=emb.cos_cached_x.dtype)
        emb.sin_cached_x = sin_x.to(device=emb.sin_cached_x.device, dtype=emb.sin_cached_x.dtype)
        emb.cos_cached_y = cos_y.to(device=emb.cos_cached_y.device, dtype=emb.cos_cached_y.dtype)
        emb.sin_cached_y = sin_y.to(device=emb.sin_cached_y.device, dtype=emb.sin_cached_y.dtype)
        logger.info("Restored Vision 2D-RoPE buffers for %s ViT (dim=%d, max_pos=%d)", tag, dim, max_pos)


def _get_nested_attr(module: nn.Module, name: str) -> nn.Module:
    """Resolve a dot-separated attribute path on *module*."""
    for part in name.split("."):
        module = getattr(module, part)
    return module


class SensenovaU1Bundle(Bundle):
    """Collection of modules for the GeoWeave (NEO-Unify MoT) model."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        dtype: torch.dtype,
        device: Any,
        config: SensenovaU1PipelineConfig,
    ) -> None:
        self.model = model
        self.transformer = model.language_model
        self.fm_modules = model.fm_modules
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.config = config
        self.pretrained_path = config.pretrained_model_ckpt_path

        self.patch_size = model.patch_size
        self.downsample_ratio = model.downsample_ratio
        self.merge_size = int(1 / self.downsample_ratio)

        self.img_start_token_id = model.img_start_token_id
        self.img_context_token_id = model.img_context_token_id
        self.conv_template = model.conv_template

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: SensenovaU1PipelineConfig) -> "SensenovaU1Bundle":
        from .vendor.modeling_neo_chat import NEOChatModel

        device = config.device or "cuda"
        dtype = parse_torch_dtype(config.model_precision)

        logger.info(
            "Loading GeoWeave from %s (dtype=%s, device=%s)",
            config.pretrained_model_ckpt_path,
            dtype,
            device,
        )

        model = NEOChatModel.from_pretrained(
            config.pretrained_model_ckpt_path,
            torch_dtype=dtype,
        )

        _restore_vision_rope_buffers(model)

        model = model.to(device).eval()

        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_ckpt_path,
            trust_remote_code=True,
        )

        # Set up special token IDs
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        model.img_context_token_id = img_context_token_id

        # ----- Freeze everything, then selectively unfreeze -----
        model.requires_grad_(False)

        if not config.use_lora:
            cls._apply_freeze_config(model, config)

        n_total = sum(1 for _ in model.parameters())
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(
            "GeoWeave parameter count: %d total, %d trainable (%.1f%%)",
            n_total,
            n_trainable,
            100.0 * n_trainable / max(n_total, 1),
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            config=config,
        )

    @classmethod
    def _apply_freeze_config(
        cls,
        model: nn.Module,
        config: SensenovaU1PipelineConfig,
    ) -> None:
        """Selectively unfreeze branches based on freeze flags."""
        from .vendor.modeling_qwen3 import Qwen3DecoderLayer

        lm = model.language_model

        # Unfreeze understanding branch
        if not config.freeze_und:
            unfrozen_und = 0
            for layer in lm.model.layers:
                if isinstance(layer, Qwen3DecoderLayer):
                    for name in _UND_LAYER_PARAM_NAMES:
                        try:
                            sub = _get_nested_attr(layer, name)
                            sub.requires_grad_(True)
                            unfrozen_und += 1
                        except AttributeError:
                            pass
            logger.info("Unfroze %d understanding sub-modules", unfrozen_und)

        # Unfreeze generation branch
        if not config.freeze_gen:
            unfrozen_gen = 0
            for layer in lm.model.layers:
                if isinstance(layer, Qwen3DecoderLayer):
                    for name in _GEN_LAYER_PARAM_NAMES:
                        try:
                            sub = _get_nested_attr(layer, name)
                            sub.requires_grad_(True)
                            unfrozen_gen += 1
                        except AttributeError:
                            pass
            logger.info("Unfroze %d generation sub-modules", unfrozen_gen)

        # Unfreeze fm_modules (timestep_embedder, fm_head, vision_model_mot_gen)
        if not config.freeze_fm_modules:
            model.fm_modules.requires_grad_(True)
            logger.info("Unfroze fm_modules")

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def trainable_module(self) -> nn.Module:
        """The module that FSDP wraps / LoRA targets / the optimizer trains.

        When ``freeze_fm_modules=False``, returns the full ``NEOChatModel`` so
        that ``fm_modules`` (timestep_embedder, fm_head, vision_model_mot_gen)
        are included in the FSDP wrap and optimizer. Otherwise returns just
        ``model.language_model`` (the Qwen3 MoT backbone).
        """
        if not self.config.freeze_fm_modules:
            return self.model
        return self.transformer


__all__ = ["SensenovaU1Bundle", "SENSENOVA_U1_FSDP_BLOCK_CLASS"]
