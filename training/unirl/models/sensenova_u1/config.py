"""Construction config for the GeoWeave (NEO-Unify MoT) pipeline.

Mirrors :class:`unirl.models.bagel.config.BagelPipelineConfig`.

GeoWeave is a Qwen3-based MoT model that operates in pixel space (no VAE).
Image generation uses flow matching with a gen-branch ViT + fm_head.

Per-request sampling knobs (CFG, noise, steps) live in
``SensenovaU1DiffusionParams`` and ``SensenovaU1ARParams``, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

from unirl.config.validation import validate_precision_type

# ---------------------------------------------------------------------------
# LoRA targets
# ---------------------------------------------------------------------------

# Generation-branch LoRA targets: the MoT ``_mot_gen`` projections.
# Train these for image-generation RL (gen experts only, und stays frozen).
SENSENOVA_U1_MOT_GEN_LORA_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj_mot_gen",
    "self_attn.k_proj_mot_gen",
    "self_attn.v_proj_mot_gen",
    "self_attn.o_proj_mot_gen",
    "mlp_mot_gen.gate_proj",
    "mlp_mot_gen.up_proj",
    "mlp_mot_gen.down_proj",
)

# Understanding-branch LoRA targets: the base (non-``_mot_gen``) projections.
# Train these for Interleave-RL (und experts only, gen stays frozen).
SENSENOVA_U1_UND_LORA_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

# FSDP wraps each ``Qwen3DecoderLayer`` (or ``Qwen3MoeDecoderLayer`` for the
# A3B variant) as a sharding unit.
SENSENOVA_U1_FSDP_BLOCK_CLASS = "Qwen3DecoderLayer"

# ---------------------------------------------------------------------------
# Per-decoder-layer parameter name groups (for selective freeze/unfreeze)
# ---------------------------------------------------------------------------

# Understanding branch params inside each Qwen3DecoderLayer
_UND_LAYER_PARAM_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "self_attn.q_norm_hw",
    "self_attn.k_norm_hw",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm",
)

# Generation branch params inside each Qwen3DecoderLayer
_GEN_LAYER_PARAM_NAMES = (
    "self_attn.q_proj_mot_gen",
    "self_attn.k_proj_mot_gen",
    "self_attn.v_proj_mot_gen",
    "self_attn.o_proj_mot_gen",
    "self_attn.q_norm_mot_gen",
    "self_attn.k_norm_mot_gen",
    "self_attn.q_norm_hw_mot_gen",
    "self_attn.k_norm_hw_mot_gen",
    "mlp_mot_gen",
    "input_layernorm_mot_gen",
    "post_attention_layernorm_mot_gen",
)


@dataclass
class SensenovaU1PipelineConfig:
    """Construction args for ``SensenovaU1Bundle.from_config``.

    GeoWeave is a Qwen3-based MoT model with two branches:
    - Understanding (und): text/VQA tokens, standard projections
    - Generation (gen): image-generation tokens, ``_mot_gen`` projections

    Image generation operates in **pixel space** (no VAE). The gen-branch ViT
    (``vision_model_mot_gen``) encodes noised image patches, the LLM gen branch
    predicts hidden states, and ``fm_head`` maps them back to pixel predictions.
    """

    pretrained_model_ckpt_path: str
    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    trajectory_precision: str = "bf16"
    logprob_precision: str = "fp32"

    # -- Pixel-space geometry (no VAE) --
    # patch_size: from the model config (typically 16 for the gen ViT)
    # merge_size: int(1 / downsample_ratio), typically 2
    patch_size: int = 16
    merge_size: int = 2

    # -- Noise / flow-matching schedule --
    noise_scale: float = 1.0
    noise_scale_mode: str = "resolution"
    t_eps: float = 0.02
    time_shift_type: str = "dynamic"
    base_shift: float = 0.5
    max_shift: float = 1.15

    # -- ViT for understanding (input images) --
    enable_vit: bool = True

    # -- Weight sync --
    weight_sync_param_name_prefix: str = "language_model."

    # -- LoRA --
    use_lora: bool = False
    lora_target_modules: Tuple[str, ...] = SENSENOVA_U1_UND_LORA_TARGETS

    # -- Freeze controls --
    # Selectively freeze/unfreeze the MoT understanding and generation branches.
    # For Interleave-RL: freeze_und=False, freeze_gen=True, freeze_fm_modules=True
    freeze_und: bool = False
    freeze_gen: bool = True
    freeze_fm_modules: bool = True

    def __post_init__(self) -> None:
        validate_precision_type(
            self.model_precision,
            field="SensenovaU1PipelineConfig.model_precision",
        )
        if not isinstance(self.lora_target_modules, tuple):
            self.lora_target_modules = tuple(self.lora_target_modules)


__all__ = [
    "SENSENOVA_U1_MOT_GEN_LORA_TARGETS",
    "SENSENOVA_U1_UND_LORA_TARGETS",
    "SENSENOVA_U1_FSDP_BLOCK_CLASS",
    "SensenovaU1PipelineConfig",
]
