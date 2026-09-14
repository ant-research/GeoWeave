from types import SimpleNamespace

import pytest
import torch
from torch import nn

from unirl.models.sensenova_u1 import rl_ops
from unirl.models.sensenova_u1.ar import SensenovaU1ARStage
from unirl.models.sensenova_u1.conditions import SensenovaU1ARConditions
from unirl.types.segments import TextSegment


class _EmbeddingLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 4)

    def get_input_embeddings(self):
        return self.embed


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.language_model = _EmbeddingLM()


def test_image_replay_passes_image_prefix_embeddings_to_scorer(monkeypatch):
    model = _Model()
    bundle = SimpleNamespace(model=model, tokenizer=object(), transformer=model.language_model)
    stage = SensenovaU1ARStage(model=bundle, autocast_precision="fp32")
    image_prefix = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    prefix_indexes = torch.tensor([[0, 1, 1, 1, 2], [0, 0, 1, 2, 0], [0, 0, 1, 2, 0]], dtype=torch.long)
    monkeypatch.setattr(
        rl_ops,
        "build_it2i_inputs",
        lambda *_args: (image_prefix, prefix_indexes, {"full_attention": torch.zeros(1)}),
    )
    captured = {}

    def fake_score(_model, **kwargs):
        captured.update(kwargs)
        return torch.zeros(len(kwargs["response_ids"]))

    monkeypatch.setattr(rl_ops, "score_response", fake_score)
    conditions = SensenovaU1ARConditions.for_sample(query="q", pixel_values=torch.ones(1), grid_hw=torch.ones(1, 2))
    segment = TextSegment.pack(tokens=[torch.tensor([7, 8, 9])], log_probs=[torch.zeros(3)])

    replay = stage.replay(conditions, segment=segment)

    assert replay.shape == (3,)
    assert "input_ids" not in captured
    torch.testing.assert_close(captured["input_embeds"][:, :5], image_prefix)
    assert captured["input_embeds"].shape == (1, 7, 4)
    torch.testing.assert_close(captured["indexes"][:, :5], prefix_indexes)
    assert captured["indexes"][0].tolist() == [0, 1, 1, 1, 2, 3, 4]


def test_score_response_requires_exactly_one_input_representation():
    with torch.no_grad(), pytest.raises(ValueError, match="exactly one"):
        rl_ops.score_response(
            object(),
            response_ids=torch.tensor([1]),
            indexes=torch.zeros(3, 1, dtype=torch.long),
            attention_mask={},
            device=torch.device("cpu"),
        )


class _CumulativeCore(nn.Module):
    def forward(self, *, input_ids=None, inputs_embeds=None, **_kwargs):
        del input_ids
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))


class _ParityLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _CumulativeCore()
        self.embed = nn.Embedding(16, 5)
        self.lm_head = nn.Linear(5, 16, bias=False)

    def get_input_embeddings(self):
        return self.embed


@pytest.mark.parametrize("with_image_prefix", [False, True])
def test_teacher_forced_ratio_is_one_for_text_and_image_prefix(with_image_prefix):
    torch.manual_seed(4)
    language_model = _ParityLanguageModel()
    model = SimpleNamespace(language_model=language_model)
    response = torch.tensor([3, 5, 7])
    if with_image_prefix:
        prefix = torch.randn(1, 4, 5)
    else:
        prefix = language_model.embed(torch.tensor([[1, 2, 4, 6]]))
    response_inputs = language_model.embed(response[:-1].unsqueeze(0))
    full_embeds = torch.cat([prefix, response_inputs], dim=1)

    replay_logp = rl_ops.score_response(
        model,
        input_embeds=full_embeds,
        response_ids=response,
        indexes=torch.zeros(3, full_embeds.shape[1], dtype=torch.long),
        attention_mask={},
        device=torch.device("cpu"),
    )

    hidden = prefix.sum(dim=1).squeeze(0)
    rollout_logps = []
    for token in response:
        logits = language_model.lm_head(hidden).float()
        rollout_logps.append(torch.log_softmax(logits, dim=-1)[token])
        hidden = hidden + language_model.embed(token)
    rollout_logp = torch.stack(rollout_logps)

    torch.testing.assert_close(replay_logp, rollout_logp)
    torch.testing.assert_close(torch.exp(replay_logp - rollout_logp), torch.ones(3))


def test_packed_logps_checkpoint_covers_full_fp32_logprob(monkeypatch):
    torch.manual_seed(9)
    lm_head = nn.Linear(7, 19, bias=False)
    hidden = torch.randn(5, 7, requires_grad=True)
    targets = torch.tensor([1, 3, 5, 7, 9])
    checkpointed_functions = []

    def run_checkpoint(function, *args, **kwargs):
        assert kwargs == {"use_reentrant": False}
        checkpointed_functions.append(function.__name__)
        return function(*args)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", run_checkpoint)
    result = rl_ops.packed_logps_from_hidden(
        lm_head,
        hidden,
        targets,
        temperature=0.75,
        logprob_chunk=2,
    )
    direct_logits = lm_head(hidden).float() / 0.75
    expected = torch.log_softmax(direct_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)

    torch.testing.assert_close(result, expected)
    assert checkpointed_functions == ["chunk_logps", "chunk_logps", "chunk_logps"]

    result.sum().backward()
    assert hidden.grad is not None
    assert lm_head.weight.grad is not None
