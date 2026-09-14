from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_understanding_recipe_composes_and_resolves_checkpoint(monkeypatch):
    checkpoint = "/tmp/geoweave-checkpoint"
    monkeypatch.setenv("SENSENOVA_U1_PATH", checkpoint)
    examples = Path(__file__).resolve().parents[2] / "examples"

    with initialize_config_dir(config_dir=str(examples), version_base=None):
        cfg = compose(config_name="unified_model/geoweave_interleave_rl")
    OmegaConf.resolve(cfg)

    assert cfg.bundle.config.pretrained_model_ckpt_path == checkpoint
    assert cfg.pipeline._target_.endswith("SensenovaU1UnderstandingPipeline")
    assert cfg.reward_track == "ar"
    assert cfg.rollout.stage_attrs == ["ar"]
    assert cfg.algorithm.image is None
    assert "diffusion" not in cfg.sampling


def test_understanding_lora_recipe_targets_only_und_branch(monkeypatch):
    monkeypatch.setenv("SENSENOVA_U1_PATH", "/tmp/geoweave-checkpoint")
    examples = Path(__file__).resolve().parents[2] / "examples"

    with initialize_config_dir(config_dir=str(examples), version_base=None):
        cfg = compose(config_name="unified_model/sensenova_u1_understanding_lora_grpo")
    OmegaConf.resolve(cfg)

    targets = list(cfg.backend.lora_cfg.target_modules)
    assert cfg.bundle.config.use_lora is True
    assert targets
    assert all("mot_gen" not in target for target in targets)
    assert cfg.backend.optimizer_cfg.learning_rate == 3.0e-5


def test_phase_c_recipe_is_ar_only_interleave_without_cfg_or_sde(monkeypatch):
    monkeypatch.setenv("SENSENOVA_U1_PATH", "/tmp/geoweave-checkpoint")
    examples = Path(__file__).resolve().parents[2] / "examples"

    with initialize_config_dir(config_dir=str(examples), version_base=None):
        cfg = compose(config_name="unified_model/sensenova_u1_interleave_grpo")
    OmegaConf.resolve(cfg)

    assert cfg.stage_config.system_message == ""
    assert cfg.pipeline._target_.endswith("SensenovaU1UniPipeline")
    assert cfg.reward_track == "ar"
    assert cfg.algorithm.image is None
    assert cfg.rollout.stage_attrs == ["ar"]
    assert cfg.backend.fsdp_cfg.reshard_after_forward is False
    assert cfg.sampling.diffusion.samples_per_prompt == 1
    assert cfg.sampling.diffusion.num_inference_steps == 30
    assert cfg.sampling.diffusion.guidance_scale == 1.0
    assert cfg.sampling.diffusion.max_images == 4
    assert cfg.sampling.diffusion.min_pixels is None
    assert cfg.sampling.diffusion.target_pixels == 512 * 512
    assert cfg.sampling.diffusion.timestep_shift == 3.0
    assert cfg.sampling.diffusion.eta == 0.0
    assert cfg.sampling.diffusion.sde_indices == []
    assert cfg.bundle.config.trajectory_precision == "bf16"
    assert cfg.sampling.diffusion.trajectory_precision == "bf16"
    assert cfg.sampling.diffusion.logprob_precision == "fp32"
