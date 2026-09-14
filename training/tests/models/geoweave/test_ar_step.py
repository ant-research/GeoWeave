import torch

from unirl.models.sensenova_u1.ar import SensenovaU1ARStep


def test_greedy_step_returns_full_softmax_logprob():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    token, logp = SensenovaU1ARStep(temperature=0).step(logits)

    assert token.tolist() == [1]
    torch.testing.assert_close(logp, torch.log_softmax(logits, dim=-1)[:, 1])


def test_top_k_sampling_reports_pre_filter_logprob():
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 1.0, 4.0]])
    token, logp = SensenovaU1ARStep(temperature=1.0, top_k=1).step(logits)

    assert token.tolist() == [2]
    torch.testing.assert_close(logp, torch.log_softmax(logits, dim=-1)[:, 2])


def test_ar_step_uses_explicit_per_trajectory_generator():
    logits = torch.zeros(1, 8)
    first = SensenovaU1ARStep(
        temperature=1.0,
        generator=torch.Generator().manual_seed(77),
    ).step(logits)[0]
    second = SensenovaU1ARStep(
        temperature=1.0,
        generator=torch.Generator().manual_seed(77),
    ).step(logits)[0]
    torch.testing.assert_close(first, second)
