from types import SimpleNamespace

import torch

from unirl.models.sensenova_u1 import rl_ops
from unirl.models.sensenova_u1.pipeline import SensenovaU1UnderstandingPipeline
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(map(str, ids)) + "<|im_end|>ignored"


class _AR:
    def __init__(self):
        self.conditions = None
        self.params = None

    def autoregress(self, conditions, *, sampling_params, params):
        del sampling_params
        self.conditions = conditions
        self.params = params
        return TextSegment.pack(
            tokens=[torch.tensor([10 + i, 20 + i]) for i in range(conditions.batch_size)],
            log_probs=[torch.zeros(2) for _ in range(conditions.batch_size)],
        )


def test_understanding_pipeline_returns_only_text_track(monkeypatch):
    pipeline = SensenovaU1UnderstandingPipeline.__new__(SensenovaU1UnderstandingPipeline)
    pipeline.bundle = SimpleNamespace(model=object(), tokenizer=_Tokenizer(), downsample_ratio=0.5)
    pipeline.ar = _AR()
    monkeypatch.setattr(Images, "to_pils", lambda self: [object() for _ in range(len(self))])
    monkeypatch.setattr(rl_ops, "prepare_input_image", lambda *_args: (torch.ones(2, 3), torch.tensor([[2, 2]])))
    monkeypatch.setattr(
        rl_ops,
        "build_query",
        lambda _model, prompt, **_kwargs: f"QUERY:{prompt}",
    )
    req = RolloutReq(
        sample_ids=["p0", "p1"],
        group_ids=["g0", "g1"],
        primitives={
            "text": Texts(texts=["first", "second"]),
            "image": Images(pixels=torch.zeros(2, 3, 4, 4)),
        },
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=2, max_new_tokens=4)},
        stage_config={"system_message": "system"},
    )

    resp = pipeline.generate(req)

    assert list(resp.tracks) == ["ar"]
    track = resp.tracks["ar"]
    assert track.sample_ids == ["p0/a0", "p0/a1", "p1/a0", "p1/a1"]
    assert track.parent_ids == ["p0", "p0", "p1", "p1"]
    assert track.decoded.texts == ["10 20", "11 21", "12 22", "13 23"]
    assert pipeline.ar.params.interleave is False
    assert pipeline.ar.conditions.prompt_queries == [
        "QUERY:<img><IMG_CONTEXT></img>\nfirst",
        "QUERY:<img><IMG_CONTEXT></img>\nfirst",
        "QUERY:<img><IMG_CONTEXT></img>\nsecond",
        "QUERY:<img><IMG_CONTEXT></img>\nsecond",
    ]
