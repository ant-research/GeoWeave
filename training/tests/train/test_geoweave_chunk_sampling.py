import pytest
import torch

from unirl.models.sensenova_u1.ar import SensenovaU1ARStep


def _kernel(seeds):
    generators = {}
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        generators[index] = generator
    return SensenovaU1ARStep(generators=generators)


def test_per_candidate_generators_are_independent_of_batch_row_order():
    logits = torch.zeros(2, 32)
    batched = _kernel([17, 29])
    batched_first, _ = batched.step(logits, [0, 1])
    batched_second, _ = batched.step(logits, [0, 1])

    reordered = _kernel([17, 29])
    row1_first, _ = reordered.step(logits[1:2], [1])
    row0_first, _ = reordered.step(logits[0:1], [0])
    row0_second, _ = reordered.step(logits[0:1], [0])
    row1_second, _ = reordered.step(logits[1:2], [1])

    assert batched_first.tolist() == [row0_first.item(), row1_first.item()]
    assert batched_second.tolist() == [row0_second.item(), row1_second.item()]


def test_per_candidate_sampling_requires_aligned_indices():
    kernel = _kernel([17, 29])
    with pytest.raises(ValueError, match="one sample index"):
        kernel.step(torch.zeros(2, 8), [0])
