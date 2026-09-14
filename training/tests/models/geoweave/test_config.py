import pytest

from unirl.models.sensenova_u1.config import (
    SENSENOVA_U1_UND_LORA_TARGETS,
    SensenovaU1PipelineConfig,
)


def test_understanding_defaults_freeze_non_target_branches():
    cfg = SensenovaU1PipelineConfig(pretrained_model_ckpt_path="/tmp/model")

    assert cfg.freeze_und is False
    assert cfg.freeze_gen is True
    assert cfg.freeze_fm_modules is True
    assert cfg.lora_target_modules == SENSENOVA_U1_UND_LORA_TARGETS


def test_config_rejects_invalid_precision():
    with pytest.raises((TypeError, ValueError)):
        SensenovaU1PipelineConfig(pretrained_model_ckpt_path="/tmp/model", model_precision="not-a-dtype")
