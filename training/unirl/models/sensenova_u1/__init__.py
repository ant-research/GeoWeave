"""SensenovaU1 (NEO-Unify MoT) model integration for UniRL.

``import unirl.models.sensenova_u1`` MUST NOT pull the vendored modeling code
(which hard-imports ``flash_attn``). ``SensenovaU1Bundle`` is intentionally NOT
re-exported — import it explicitly::

    from unirl.models.sensenova_u1.bundle import SensenovaU1Bundle
"""

from .ar import SensenovaU1ARParams, SensenovaU1ARStage, SensenovaU1ARStep
from .conditions import SensenovaU1ARConditions
from .config import (
    SENSENOVA_U1_FSDP_BLOCK_CLASS,
    SENSENOVA_U1_MOT_GEN_LORA_TARGETS,
    SENSENOVA_U1_UND_LORA_TARGETS,
    SensenovaU1PipelineConfig,
)
from .diffusion import (
    SensenovaU1DiffusionParams,
    SensenovaU1DiffusionStage,
    SensenovaU1DiffusionStep,
)
from .pipeline import SensenovaU1Pipeline, SensenovaU1UniPipeline

__all__ = [
    "SensenovaU1ARConditions",
    "SensenovaU1ARParams",
    "SensenovaU1ARStage",
    "SensenovaU1ARStep",
    "SensenovaU1DiffusionParams",
    "SensenovaU1DiffusionStage",
    "SensenovaU1DiffusionStep",
    "SensenovaU1Pipeline",
    "SensenovaU1UniPipeline",
    "SensenovaU1PipelineConfig",
    "SENSENOVA_U1_FSDP_BLOCK_CLASS",
    "SENSENOVA_U1_MOT_GEN_LORA_TARGETS",
    "SENSENOVA_U1_UND_LORA_TARGETS",
]
