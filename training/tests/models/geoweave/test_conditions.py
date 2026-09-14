import pytest
import torch

from unirl.models.sensenova_u1.conditions import (
    SensenovaU1ARConditions,
    SensenovaU1DiffusionConditions,
)


def test_ar_conditions_round_trip_preserves_multimodal_payload():
    conditions = SensenovaU1ARConditions.for_sample(
        query="formatted prompt",
        pixel_values=torch.ones(2, 3),
        grid_hw=torch.tensor([[2, 3]]),
    )

    assert SensenovaU1ARConditions.from_dict(conditions.to_dict()) is conditions
    assert conditions.batch_size == 1


def test_ar_conditions_reject_missing_or_wrong_slot():
    with pytest.raises(ValueError, match="sensenova_u1_ar"):
        SensenovaU1ARConditions.from_dict({})
    with pytest.raises(ValueError, match="sensenova_u1_ar"):
        SensenovaU1ARConditions.from_dict({"sensenova_u1_ar": object()})


def test_diffusion_conditions_round_trip_and_single():
    cache = object()
    conditions = SensenovaU1DiffusionConditions.for_sample(past_kv_cond=cache, image_shape=(64, 96), text_len=17)

    assert SensenovaU1DiffusionConditions.from_dict(conditions.to_dict()) is conditions
    assert conditions.single() == (cache, None, 17, None, (64, 96))


def test_diffusion_conditions_preserve_cfg_cache_and_length():
    cond_cache = object()
    uncond_cache = object()
    conditions = SensenovaU1DiffusionConditions.for_sample(
        past_kv_cond=cond_cache,
        past_kv_uncond=uncond_cache,
        image_shape=(64, 96),
        text_len=17,
        text_len_uncond=9,
    )

    assert conditions.single() == (cond_cache, uncond_cache, 17, 9, (64, 96))
