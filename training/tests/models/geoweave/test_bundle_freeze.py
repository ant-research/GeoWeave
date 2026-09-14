import os

import pytest
import torch
from torch import nn

from unirl.models.sensenova_u1.bundle import SensenovaU1Bundle
from unirl.models.sensenova_u1.config import SensenovaU1PipelineConfig
from unirl.models.sensenova_u1.vendor.configuration_neo_vit import NEOVisionConfig
from unirl.models.sensenova_u1.vendor.modeling_neo_vit import (
    NEOVisionEmbeddings,
    precompute_rope_freqs_sincos,
)
from unirl.models.sensenova_u1.vendor.modeling_qwen3 import Qwen3DecoderLayer


def _leaf():
    return nn.Linear(1, 1, bias=False)


def _fake_model():
    layer = Qwen3DecoderLayer.__new__(Qwen3DecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = nn.Module()
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "q_norm_hw",
        "k_norm_hw",
        "q_proj_mot_gen",
        "k_proj_mot_gen",
        "v_proj_mot_gen",
        "o_proj_mot_gen",
        "q_norm_mot_gen",
        "k_norm_mot_gen",
        "q_norm_hw_mot_gen",
        "k_norm_hw_mot_gen",
    ):
        setattr(layer.self_attn, name, _leaf())
    for name in (
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        "mlp_mot_gen",
        "input_layernorm_mot_gen",
        "post_attention_layernorm_mot_gen",
    ):
        setattr(layer, name, _leaf())

    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = nn.ModuleList([layer])
    model.language_model.model.norm = _leaf()
    model.language_model.model.norm_mot_gen = _leaf()
    model.fm_modules = _leaf()
    model.requires_grad_(False)
    return model


def test_restore_vision_rope_buffers_repairs_both_vision_paths():
    vision_config = NEOVisionConfig(
        num_channels=3,
        patch_size=2,
        hidden_size=8,
        llm_hidden_size=16,
        downsample_ratio=0.5,
        max_position_embeddings_vision=8,
    )
    model = nn.Module()
    model.vision_model = nn.Module()
    model.vision_model.embeddings = NEOVisionEmbeddings(vision_config)
    model.fm_modules = nn.ModuleDict({"vision_model_mot_gen": nn.Module()})
    model.fm_modules["vision_model_mot_gen"].embeddings = NEOVisionEmbeddings(vision_config)

    for module in model.modules():
        if isinstance(module, NEOVisionEmbeddings):
            for name, buffer in module.named_buffers():
                if name.startswith(("cos_cached_", "sin_cached_")):
                    buffer.zero_()

    restored = SensenovaU1Bundle._restore_vision_rope_buffers(model)

    assert restored == (
        "vision_model.embeddings",
        "fm_modules.vision_model_mot_gen.embeddings",
    )
    expected_cos, expected_sin = precompute_rope_freqs_sincos(
        vision_config.hidden_size // 2,
        vision_config.max_position_embeddings_vision,
        base=vision_config.rope_theta_vision,
        device=None,
    )
    for module in model.modules():
        if isinstance(module, NEOVisionEmbeddings):
            assert torch.equal(module.cos_cached_x, expected_cos)
            assert torch.equal(module.sin_cached_x, expected_sin)
            assert torch.equal(module.cos_cached_y, expected_cos)
            assert torch.equal(module.sin_cached_y, expected_sin)


@pytest.mark.parametrize(
    ("freeze_und", "freeze_gen", "expected_kind"),
    [(False, True, "und"), (True, False, "gen"), (True, True, "none")],
)
def test_freeze_matrix_selects_only_requested_branch(freeze_und, freeze_gen, expected_kind):
    model = _fake_model()
    cfg = SensenovaU1PipelineConfig(
        pretrained_model_ckpt_path="/tmp/model",
        freeze_und=freeze_und,
        freeze_gen=freeze_gen,
        freeze_fm_modules=True,
    )

    SensenovaU1Bundle._apply_freeze_config(model, cfg)

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if expected_kind == "und":
        assert trainable and all("mot_gen" not in name for name in trainable)
    elif expected_kind == "gen":
        assert trainable and all("mot_gen" in name for name in trainable)
    else:
        assert trainable == []
    assert not any(name.startswith("fm_modules.") for name in trainable)
    assert all(name.startswith("language_model.model.layers.") for name in trainable)
    assert model.language_model.model.norm.weight.requires_grad is False
    assert model.language_model.model.norm_mot_gen.weight.requires_grad is False


def test_optimizer_step_changes_only_understanding_parameters():
    model = _fake_model()
    cfg = SensenovaU1PipelineConfig(
        pretrained_model_ckpt_path="/tmp/model",
        freeze_und=False,
        freeze_gen=True,
        freeze_fm_modules=True,
    )
    SensenovaU1Bundle._apply_freeze_config(model, cfg)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.1)

    sum(param.sum() for param in model.parameters() if param.requires_grad).backward()
    optimizer.step()

    for name, param in model.named_parameters():
        changed = not torch.equal(param.detach(), before[name])
        assert changed == param.requires_grad, name
        if changed:
            assert "mot_gen" not in name
            assert not name.startswith("fm_modules.")


@pytest.mark.gpu
@pytest.mark.ckpt
@pytest.mark.slow
def test_checkpoint_loads_with_understanding_only_trainable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    path = os.environ.get("SENSENOVA_U1_CKPT")
    if not path:
        pytest.skip("set SENSENOVA_U1_CKPT to enable the checkpoint smoke test")

    bundle = SensenovaU1Bundle.from_config(
        SensenovaU1PipelineConfig(
            pretrained_model_ckpt_path=path,
            device="cuda",
            freeze_und=False,
            freeze_gen=True,
            freeze_fm_modules=True,
        )
    )
    trainable = [name for name, param in bundle.model.named_parameters() if param.requires_grad]
    assert trainable
    assert not any("mot_gen" in name or name.startswith("fm_modules.") for name in trainable)
    assert bundle.model.language_model.model.norm.weight.requires_grad is False
