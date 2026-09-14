from types import SimpleNamespace

import pytest
import torch
from torch import nn

from unirl.models.sensenova_u1 import rl_ops
from unirl.models.sensenova_u1.diffusion import SensenovaU1DiffusionParams
from unirl.models.sensenova_u1.pipeline import fit_image_size_to_pixel_budget


class _DecodeLanguageModel:
    def __init__(self):
        self.model = SimpleNamespace(current_index=0)

    def __call__(self, *, past_key_values, **_kwargs):
        return SimpleNamespace(
            logits=torch.zeros(1, 1, 32),
            past_key_values=past_key_values,
        )


def test_interleave_decode_generates_four_images_and_omits_unpaired_marker(monkeypatch):
    image_token = 9
    # Four complete image events, followed by ordinary text and a fifth marker.
    sampled_tokens = iter([1, image_token, 2, image_token, 3, image_token, 4, image_token, 5, image_token])

    def sample_fn(_logits):
        return torch.tensor([next(sampled_tokens)]), torch.tensor([-0.25])

    generated_sizes = []

    def diffuse_fn(cache, _text_len, uncond_cache, uncond_text_len, image_size):
        assert uncond_cache is None
        assert uncond_text_len is None
        generated_sizes.append(image_size)
        return torch.zeros(3, *image_size), None

    monkeypatch.setattr(
        rl_ops,
        "append_image_to_cache",
        lambda _model, _tokenizer, cache, t_idx, _image, *, device: (
            cache,
            t_idx + 1,
            torch.zeros(1, 1, 32, device=device),
        ),
    )
    model = SimpleNamespace(language_model=_DecodeLanguageModel())

    tokens, logps, images, boundaries, _cache, _t_idx = rl_ops.interleave_decode(
        model,
        tokenizer=object(),
        past_key_values=object(),
        t_idx=3,
        start_logits=torch.zeros(1, 32),
        sample_fn=sample_fn,
        max_new_tokens=32,
        max_images=4,
        stop_ids=[],
        img_start_token_id=image_token,
        diffuse_fn=diffuse_fn,
        image_size=(384, 672),
        device=torch.device("cpu"),
    )

    assert tokens == [1, 9, 2, 9, 3, 9, 4, 9, 5]
    assert len(logps) == len(tokens)
    assert boundaries == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 9)]
    assert len(images) == 4
    assert generated_sizes == [(384, 672)] * 4
    assert sum(tokens[i - 1] == image_token for _, i in boundaries) == len(images)


class _ReplayCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(16))

    def forward(self, *, inputs_embeds, **_kwargs):
        return SimpleNamespace(last_hidden_state=self.proj(inputs_embeds))


class _ReplayLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ReplayCore()
        self.embed = nn.Embedding(16, 16)
        self.lm_head = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.eye(16))
            self.lm_head.weight.copy_(torch.eye(16))

    def get_input_embeddings(self):
        return self.embed


def test_interleaved_replay_scores_every_token_across_two_images(monkeypatch):
    image_token = 9
    language_model = _ReplayLanguageModel()
    model = SimpleNamespace(language_model=language_model, img_start_token_id=image_token)
    image_logits = []
    for target in (2, 3):
        logits = torch.zeros(1, 1, 16)
        logits[0, 0, target] = 4.0
        image_logits.append(logits)

    monkeypatch.setattr(
        rl_ops,
        "build_text_inputs",
        lambda *_args: (
            torch.tensor([[7, 1]]),
            torch.tensor([[0, 1], [0, 0], [0, 0]], dtype=torch.long),
            {},
        ),
    )
    reencode_calls = []

    def build_image(_model, _tokenizer, t_idx, image, *, device):
        del device
        reencode_calls.append(image)
        embeds = image_logits[len(reencode_calls) - 1]
        indexes = torch.tensor([[t_idx + 2], [0], [0]], dtype=torch.long)
        return embeds, indexes, t_idx + 2

    monkeypatch.setattr(rl_ops, "build_generated_image_inputs", build_image)

    tokens = [1, image_token, 2, image_token, 3, 4]
    result = rl_ops.score_interleaved_response(
        model,
        tokenizer=object(),
        query="q",
        all_tokens=tokens,
        text_boundaries=[(0, 2), (2, 4), (4, 6)],
        generated_images=[torch.zeros(3, 32, 32), torch.ones(3, 32, 32)],
        device=torch.device("cpu"),
    )

    expected_logits = [
        torch.nn.functional.one_hot(torch.tensor(1), 16).float(),
        torch.nn.functional.one_hot(torch.tensor(1), 16).float(),
        image_logits[0][0, 0],
        torch.nn.functional.one_hot(torch.tensor(2), 16).float(),
        image_logits[1][0, 0],
        torch.nn.functional.one_hot(torch.tensor(3), 16).float(),
    ]
    expected = torch.stack([torch.log_softmax(logits, dim=-1)[token] for logits, token in zip(expected_logits, tokens)])
    torch.testing.assert_close(result, expected)
    assert result.shape == (len(tokens),)
    assert len(reencode_calls) == 2
    # In particular, the first token after an auxiliary image must retain a
    # gradient through the understanding transformer replay.
    result[2].backward()
    assert language_model.model.proj.weight.grad is not None
    assert torch.count_nonzero(language_model.model.proj.weight.grad) > 0


def test_interleaved_replay_rejects_image_event_mismatch(monkeypatch):
    model = SimpleNamespace(
        language_model=_ReplayLanguageModel(),
        img_start_token_id=9,
    )
    monkeypatch.setattr(
        rl_ops,
        "build_text_inputs",
        lambda *_args: (torch.tensor([[7]]), torch.zeros(3, 1, dtype=torch.long), {}),
    )

    with pytest.raises(ValueError, match="event/image mismatch"):
        rl_ops.score_interleaved_response(
            model,
            tokenizer=object(),
            query="q",
            all_tokens=[1, 9],
            text_boundaries=[(0, 2)],
            generated_images=[],
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("source_size", [(1600, 900), (900, 1600), (1024, 1024)])
def test_variable_aspect_size_preserves_ratio_and_fixed_pixel_area(source_size):
    source_w, source_h = source_size
    height, width = fit_image_size_to_pixel_budget(
        source_width=source_w,
        source_height=source_h,
        min_pixels=512 * 512,
        target_pixels=512 * 512,
        multiple=32,
    )

    assert height % 32 == 0
    assert width % 32 == 0
    assert abs(width / height - source_w / source_h) / (source_w / source_h) < 0.08
    # 32-aligned floor/ceil means the area is approximate, not exact.
    assert abs(width * height - 512 * 512) / (512 * 512) < 0.12


def test_fixed_pixel_area_downscales_large_input():
    assert fit_image_size_to_pixel_budget(
        source_width=1264,
        source_height=848,
        min_pixels=512 * 512,
        target_pixels=512 * 512,
        multiple=32,
    ) == (416, 608)


def test_phase_c_diffusion_defaults():
    params = SensenovaU1DiffusionParams()
    assert params.num_inference_steps == 30
    assert params.max_images == 4
    assert params.min_pixels is None
    assert params.target_pixels == 512 * 512
    assert params.noise_scale_mode == "resolution"


def test_generation_vit_receives_flat_native_patches_and_per_token_time():
    calls = {}

    class _Embedder(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, values):
            calls[self.name] = values.detach().clone()
            return values[:, None].expand(-1, 5)

    class _GenModel:
        patch_size = 16
        add_noise_scale_embedding = True
        noise_scale_max_value = 4.0
        fm_modules = {
            "timestep_embedder": _Embedder("time"),
            "noise_scale_embedder": _Embedder("noise"),
        }

        @staticmethod
        def patchify(_image, _patch_size, channel_first=False):
            assert channel_first
            return torch.zeros(1, 4, 3)

        @staticmethod
        def extract_feature(patches, *, gen_model, grid_hw):
            assert gen_model
            calls["patch_shape"] = tuple(patches.shape)
            calls["grid_hw"] = grid_hw.clone()
            # Simulate merge_size=2: four raw patches -> one merged token.
            return torch.zeros(1, 5)

    grid = torch.tensor([[2, 2]])
    embeds = rl_ops.build_image_embeds(
        _GenModel(),
        torch.zeros(1, 3, 32, 32),
        torch.tensor([0.25]),
        grid_hw=grid,
        noise_scale_value=2.0,
    )

    assert calls["patch_shape"] == (4, 3)
    torch.testing.assert_close(calls["grid_hw"], grid)
    torch.testing.assert_close(calls["time"], torch.tensor([0.25]))
    torch.testing.assert_close(calls["noise"], torch.tensor([0.5]))
    assert embeds.shape == (1, 1, 5)


def test_stable_trajectory_seed_is_deterministic_and_identity_sensitive():
    from unirl.models.sensenova_u1.interleave_runtime import stable_trajectory_seed

    seed = stable_trajectory_seed(42, 7, "prompt-3/a5", 0)
    assert seed == stable_trajectory_seed(42, 7, "prompt-3/a5", 0)
    assert seed != stable_trajectory_seed(42, 7, "prompt-3/a6", 0)
    assert seed != stable_trajectory_seed(42, 7, "prompt-3/a5", 1)
    assert 0 <= seed < 2**63 - 1


def test_per_trajectory_generator_replays_the_same_stream():
    from unirl.models.sensenova_u1.interleave_runtime import make_generator

    first = torch.randn(8, generator=make_generator("cpu", 1234))
    second = torch.randn(8, generator=make_generator("cpu", 1234))
    other = torch.randn(8, generator=make_generator("cpu", 1235))
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, other)


def test_trajectory_result_validates_replay_contract():
    from unirl.models.sensenova_u1.interleave_runtime import SensenovaU1TrajectoryResult

    result = SensenovaU1TrajectoryResult(
        tokens=[1, 9, 2],
        log_probs=[-0.1, -0.2, -0.3],
        generated_images=[torch.zeros(3, 8, 8)],
        text_segment_boundaries=[(0, 2), (2, 3)],
    )
    result.validate(img_start_token_id=9)

    result.generated_images = []
    with pytest.raises(ValueError, match="<img>/image mismatch"):
        result.validate(img_start_token_id=9)


def test_generate_one_trajectory_wraps_existing_interleave_runtime(monkeypatch):
    from unirl.models.sensenova_u1 import interleave_runtime

    calls = {}
    model = SimpleNamespace(language_model=SimpleNamespace(lm_head=lambda hidden: hidden))

    monkeypatch.setattr(
        interleave_runtime.rl_ops,
        "build_text_inputs",
        lambda *_args: (
            torch.tensor([[3, 4]]),
            torch.tensor([[0, 1], [0, 0], [0, 0]]),
            {},
        ),
    )
    monkeypatch.setattr(
        interleave_runtime.rl_ops,
        "prefix_forward",
        lambda *_args: ("prefix-kv", torch.zeros(1, 1, 16)),
    )

    def fake_interleave(_model, _tokenizer, past_kv, t_idx, **kwargs):
        calls.update(kwargs)
        return (
            [5, 9, 6],
            [-0.5, -0.9, -0.6],
            [torch.ones(3, 8, 8)],
            [(0, 2), (2, 3)],
            f"{past_kv}-done",
            t_idx + 4,
        )

    monkeypatch.setattr(interleave_runtime.rl_ops, "interleave_decode", fake_interleave)

    result = interleave_runtime.generate_one_trajectory(
        model=model,
        tokenizer=object(),
        query="query",
        pixel_values=None,
        grid_hw=None,
        sample_fn=lambda logits: (torch.tensor([0]), torch.tensor([0.0])),
        max_new_tokens=32,
        stop_ids=[7],
        interleave=True,
        max_images=4,
        img_start_token_id=9,
        diffuse_fn=lambda *_args: (torch.zeros(3, 8, 8), None),
        text_uncondition_query=None,
        image_size=(8, 8),
        device=torch.device("cpu"),
    )

    assert result.tokens == [5, 9, 6]
    assert result.log_probs == [-0.5, -0.9, -0.6]
    assert result.text_segment_boundaries == [(0, 2), (2, 3)]
    assert len(result.generated_images) == 1
    assert result.past_key_values == "prefix-kv-done"
    assert result.t_idx == 5
    assert result.finish_reason == "completed"
    assert result.timing["total_s"] >= 0
    assert calls["image_size"] == (8, 8)
    assert calls["past_key_values_uncond"] is None
    assert calls["t_idx_uncond"] is None


def test_interleave_decode_keeps_cfg_cache_text_unconditioned(monkeypatch):
    image_token = 9
    sampled_tokens = iter([1, image_token, 2, 7])

    def sample_fn(_logits):
        return torch.tensor([next(sampled_tokens)]), torch.tensor([-0.25])

    class _TrackingLanguageModel:
        def __init__(self):
            self.model = SimpleNamespace(current_index=0)
            self.calls = []

        def __call__(self, *, input_ids, past_key_values, **_kwargs):
            token = int(input_ids.item())
            self.calls.append((past_key_values, token))
            return SimpleNamespace(
                logits=torch.zeros(1, 1, 32),
                past_key_values=f"{past_key_values}+{token}",
            )

    lm = _TrackingLanguageModel()
    model = SimpleNamespace(language_model=lm)
    diffuse_calls = []
    append_calls = []

    def diffuse_fn(cond_cache, cond_len, uncond_cache, uncond_len, image_size):
        diffuse_calls.append((cond_cache, cond_len, uncond_cache, uncond_len, image_size))
        return torch.zeros(3, *image_size), None

    def append_image(_model, _tokenizer, cache, t_idx, _image, *, device):
        del device
        append_calls.append((cache, t_idx))
        return f"{cache}+image", t_idx + 2, torch.zeros(1, 1, 32)

    monkeypatch.setattr(rl_ops, "append_image_to_cache", append_image)

    result = rl_ops.interleave_decode(
        model,
        tokenizer=object(),
        past_key_values="cond",
        t_idx=3,
        start_logits=torch.zeros(1, 32),
        sample_fn=sample_fn,
        max_new_tokens=8,
        max_images=1,
        stop_ids=[7],
        img_start_token_id=image_token,
        diffuse_fn=diffuse_fn,
        image_size=(8, 8),
        device=torch.device("cpu"),
        past_key_values_uncond="uncond",
        t_idx_uncond=5,
    )

    assert result[0] == [1, 9, 2, 7]
    # Reasoning token 1 is appended only to the conditioned branch. Both
    # branches receive <img>, then the same generated image.
    assert lm.calls[:3] == [("cond", 1), ("cond+1", 9), ("uncond", 9)]
    assert diffuse_calls == [("cond+1+9", 5, "uncond+9", 6, (8, 8))]
    assert append_calls == [("cond+1+9", 5), ("uncond+9", 6)]


def test_diffuse_restores_t_eps_when_rollout_raises(monkeypatch):
    from unirl.models.sensenova_u1.diffusion import SensenovaU1DiffusionStage

    stage = SensenovaU1DiffusionStage.__new__(SensenovaU1DiffusionStage)
    stage.model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(t_eps=0.11)))
    monkeypatch.setattr(
        stage,
        "_diffuse_impl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        stage.diffuse(
            object(),
            schedule=torch.tensor([1.0, 0.0]),
            params=SimpleNamespace(t_eps=0.02),
        )
    assert stage.model.model.config.t_eps == 0.11


def test_flash_kv_cache_is_cleared_when_request_raises(monkeypatch):
    from unirl.models.sensenova_u1 import diffusion
    from unirl.models.sensenova_u1.vendor import modeling_neo_chat

    calls = []
    monkeypatch.setattr(
        modeling_neo_chat,
        "prepare_flash_kv_cache",
        lambda cache, *, current_len, batch_size: calls.append(("prepare", cache, current_len, batch_size)),
    )
    monkeypatch.setattr(
        modeling_neo_chat,
        "clear_flash_kv_cache",
        lambda cache: calls.append(("clear", cache)),
    )

    with pytest.raises(RuntimeError, match="boom"):
        with diffusion._prepared_flash_kv_cache("kv", current_len=17):
            raise RuntimeError("boom")

    assert calls == [("prepare", "kv", 17, 1), ("clear", "kv")]


def test_unified_pipeline_consumes_async_request_seed(monkeypatch):
    from types import SimpleNamespace

    from torch import nn

    from unirl.models.sensenova_u1.interleave_runtime import ASYNC_TRAJECTORY_SEED_KEY
    from unirl.models.sensenova_u1.pipeline import SensenovaU1UniPipeline
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import ARSamplingParams
    from unirl.types.segments import TextSegment

    captured = {}

    class _AR:
        def autoregress(self, conditions, **kwargs):
            captured["conditions"] = conditions
            captured.update(kwargs)
            return TextSegment.pack(
                tokens=[torch.tensor([7], dtype=torch.long)],
                log_probs=[torch.tensor([-0.5], dtype=torch.float32)],
            )

    model = nn.Linear(1, 1)
    model.config = SimpleNamespace()
    bundle = SimpleNamespace(
        model=model,
        tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: "answer"),
        patch_size=2,
        merge_size=2,
        downsample_ratio=1,
        img_start_token_id=99,
    )
    pipeline = SensenovaU1UniPipeline.__new__(SensenovaU1UniPipeline)
    pipeline.bundle = bundle
    pipeline.ar = _AR()
    pipeline.diffusion = object()

    monkeypatch.setattr("unirl.models.sensenova_u1.pipeline.rl_ops.build_query", lambda *_args, **_kwargs: "query")
    req = RolloutReq(
        sample_ids=["p0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=["question"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=1)},
        stage_config={ASYNC_TRAJECTORY_SEED_KEY: 1234},
    )

    resp = pipeline.generate(req)

    assert captured["trajectory_seeds"] == [1234]
    assert captured["diffuse_fn"] is None
    assert len(captured["diffuse_fns"]) == 1
    assert captured["params"].text_uncondition_queries is None
    assert resp.tracks["ar"].sample_ids == ["p0/a0"]

    captured.clear()
    cfg_req = RolloutReq(
        sample_ids=["p0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=["question"])},
        sampling_params={
            "ar": ARSamplingParams(samples_per_prompt=1),
            "diffusion": SensenovaU1DiffusionParams(guidance_scale=4.0),
        },
        stage_config={ASYNC_TRAJECTORY_SEED_KEY: 1234},
    )
    pipeline.generate(cfg_req)
    assert captured["params"].text_uncondition_queries == ["query"]
