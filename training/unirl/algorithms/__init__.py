"""unirl stage-driven algorithms.

Public surface for the ``models`` training contract.
"""

from __future__ import annotations

from .bagel_flow_unigrpo import BagelFlowUniGRPO
from .base import AlgorithmStepResult, StageAlgorithm
from .cppo import CPPO, CPPOConfig
from .diffusionnft import DiffusionNFT, DiffusionNFTConfig
from .dppo import DPPO, DPPOConfig
from .drpo import DRPO, DRPOConfig
from .flow_mse import FlowMSE
from .flowdppo import FlowDPPO, FlowDPPOConfig
from .flowgrpo import FlowGRPO, FlowGRPOConfig
from .grpo import GRPO, GRPOConfig
from .gspo import GSPO, GSPOConfig
from .interleave_rl import InterleaveRL

__all__ = [
    "GRPO",
    "GRPOConfig",
    "GSPO",
    "GSPOConfig",
    "CPPO",
    "CPPOConfig",
    "DPPO",
    "DPPOConfig",
    "DRPO",
    "DRPOConfig",
    "AlgorithmStepResult",
    "BagelFlowUniGRPO",
    "FlowGRPO",
    "FlowGRPOConfig",
    "FlowMSE",
    "InterleaveRL",
    "DiffusionNFT",
    "DiffusionNFTConfig",
    "FlowDPPO",
    "FlowDPPOConfig",
    "StageAlgorithm",
]
