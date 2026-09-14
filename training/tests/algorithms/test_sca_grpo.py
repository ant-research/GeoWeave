import pytest
import torch

from unirl.algorithms.grpo import GRPO
from unirl.types.segments import TextSegment


def _algorithm(mode: str = "seq-mean-token-mean") -> GRPO:
    algorithm = object.__new__(GRPO)
    algorithm.loss_agg_mode = mode
    algorithm.horizon = 8
    return algorithm


def test_grpo_prefers_packed_sca_advantages():
    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 2]), torch.tensor([3])],
        log_probs=[torch.zeros(2), torch.zeros(1)],
        token_advantages=[torch.tensor([0.5, -0.5]), torch.tensor([2.0])],
    )

    resolved = GRPO._resolve_token_advantages(
        torch.tensor([99.0, 99.0]),
        segment,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert resolved.tolist() == [0.5, -0.5, 2.0]


def test_seq_mean_token_mean_ignores_unmapped_and_invalid_sca_rows():
    values = torch.tensor([1.0, 3.0, 100.0], requires_grad=True)
    loss = _algorithm()._reduce_token_loss(
        values,
        torch.tensor([2, 1]),
        sample_mask=torch.tensor([1.0, 0.0]),
        token_mask=torch.tensor([1.0, 1.0, 0.0]),
    )

    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    assert values.grad.tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_token_mean_uses_only_sca_mapped_tokens():
    values = torch.tensor([1.0, 3.0, 100.0], requires_grad=True)
    loss = _algorithm("token-mean")._reduce_token_loss(
        values,
        torch.tensor([2, 1]),
        token_mask=torch.tensor([1.0, 1.0, 0.0]),
    )

    assert loss.item() == pytest.approx(2.0)
