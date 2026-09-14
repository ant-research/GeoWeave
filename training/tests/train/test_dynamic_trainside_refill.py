from types import SimpleNamespace

import pytest
import torch

from unirl.models.sensenova_u1.conditions import SensenovaU1ARConditions
from unirl.reward.base import RewardBackend
from unirl.reward.local.dashscope_llm_judge import _extract_json_object
from unirl.reward.service import RewardService
from unirl.trainer.dynamic_trainside import (
    _is_informative,
    _rollout_phase_profile,
    _score_group,
    _stop_reason,
)
from unirl.types.primitives import Texts
from unirl.types.reward import RewardResponse
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment


def _track(size: int) -> RolloutTrack:
    return RolloutTrack(
        sample_ids=[f"candidate-{i}" for i in range(size)],
        parent_ids=["prompt-0"] * size,
        conditions={},
        segment=TextSegment.pack(
            tokens=[torch.tensor([i + 1]) for i in range(size)],
            log_probs=[torch.tensor([-0.1]) for _ in range(size)],
        ),
        decoded=Texts(texts=[f"answer-{i}" for i in range(size)]),
    )


def test_stop_reason_uses_explicit_condition_metadata():
    track = _track(1)
    track.conditions = {"ar": SimpleNamespace(stop_reasons=["max_images"])}
    assert _stop_reason(track) == "max_images"


def test_rollout_phase_profile_uses_condition_metadata():
    track = _track(1)
    track.conditions = {
        "ar": SimpleNamespace(
            rollout_phase_metrics=[
                {"prefix_s": 1.0, "diffusion_s": 3.5, "generated_images": 2}
            ]
        )
    }
    assert _rollout_phase_profile(track) == {
        "prefix_s": 1.0,
        "diffusion_s": 3.5,
        "generated_images": 2.0,
    }


def test_informative_requires_reward_variation():
    assert not _is_informative(torch.tensor([0.0, 0.0]))
    assert not _is_informative(torch.tensor([1.0, 1.0]))
    assert _is_informative(torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r'{"correct": true, "reason": "由 \alpha 可知"}', True),
        (
            "```json\n"
            r'{"correct": false, "reason": "候选式中的 \cdot 不成立"}'
            "\n```",
            False,
        ),
    ],
)
def test_judge_recovers_boolean_verdict_from_invalid_latex_escape(response, expected):
    assert _extract_json_object(response)["correct"] is expected


def test_judge_malformed_fallback_still_requires_boolean_correct():
    with pytest.raises(ValueError, match="boolean field 'correct'"):
        _extract_json_object(r'{"correct": "true", "reason": "bad \alpha"}')


def test_score_group_skips_forced_truncation_rows_and_scatter_merges():
    class Reward:
        def __init__(self):
            self.seen = None

        def score_and_attach(self, *, req, track):
            self.seen = (req, track)
            return SimpleNamespace(
                rewards=torch.tensor([1.0, 0.0]),
                component_rewards={
                    "answer_correctness": torch.tensor([1.0, 0.0]),
                    "judge_failed": torch.tensor([0.0, 0.0]),
                },
                process_annotations=[{"valid": True}, {"valid": True}],
            )

    reward = Reward()
    req = RolloutReq(
        sample_ids=["prompt-0"],
        group_ids=["group-0"],
        primitives={"text": Texts(texts=["problem"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=3)},
        metadata=[{"answer": "42"}],
    )
    _, rewards, components, annotations, _, failed = _score_group(
        driver_reward=reward,
        req=req,
        prompt_index=0,
        track=_track(3),
        forced_zero=[False, True, False],
        stop_reasons=["eos", "max_images", "max_new_tokens"],
    )

    judge_req, judge_track = reward.seen
    assert judge_track.batch_size == 2
    assert judge_req.sampling_params["ar"].samples_per_prompt == 2
    assert rewards.tolist() == [1.0, 0.0, 0.0]
    assert components["forced_truncation_zero"].tolist() == [0.0, 1.0, 0.0]
    assert components["truncated"].tolist() == [0.0, 1.0, 1.0]
    assert components["image_truncated"].tolist() == [0.0, 1.0, 0.0]
    assert components["text_truncated"].tolist() == [0.0, 0.0, 1.0]
    assert components["judge_evaluated"].tolist() == [1.0, 0.0, 1.0]
    assert annotations == [{"valid": True}, None, {"valid": True}]
    assert failed is False


def test_dynamic_scheduler_retries_truncation_then_replaces_uniform_prompt(monkeypatch):
    import unirl.trainer.dynamic_trainside as dynamic

    class Ref:
        def __init__(self, value):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(refs):
            return [ref.value for ref in refs]

        @staticmethod
        def wait(refs, num_returns=1, timeout=None):
            del timeout
            return list(refs[:num_returns]), list(refs[num_returns:])

    calls = {"prompt-0": 0, "prompt-1": 0}

    def generate(req):
        prompt_id = req.sample_ids[0]
        call_index = calls[prompt_id]
        calls[prompt_id] += 1
        reason = (
            "max_new_tokens" if prompt_id == "prompt-0" and call_index == 0 else "eos"
        )
        answer = f"{prompt_id}-answer-{call_index}"
        track = RolloutTrack(
            sample_ids=["temporary"],
            parent_ids=[prompt_id],
            conditions={"ar": SimpleNamespace(stop_reasons=[reason])},
            segment=TextSegment.pack(
                tokens=[torch.tensor([call_index + 1])],
                log_probs=[torch.tensor([-0.1])],
            ),
            decoded=Texts(texts=[answer]),
        )
        return SimpleNamespace(tracks={"ar": track})

    class RemoteCall:
        @staticmethod
        def remote(role_name, method, args, kwargs, grad, context):
            del role_name, kwargs, grad, context
            assert method == "generate"
            return Ref(generate(args[0]))

    class Worker:
        call = RemoteCall()

    class Transport:
        pass

    class Handle:
        workers = [Worker()]
        role_name = "rollout"
        pool = SimpleNamespace(transport_cls=Transport)
        rank_infos = [
            SimpleNamespace(
                dp_rank=0, tp_rank=0, is_pipeline_last_stage=True, sp_rank=0
            )
        ]
        world_size = 1
        sp_size = 1
        dp_size = 1

        @staticmethod
        def _rebind_tree(result, worker, worker_local):
            del worker, worker_local
            return result

    class Reward:
        backend = SimpleNamespace(thread_safe=False)

        @staticmethod
        def score_and_attach(*, req, track):
            prompt = req.primitives["text"].texts[0]
            if prompt == "problem-0":
                # Length shaping can make final rewards non-uniform even though
                # correctness remains all-zero; filtering must use correctness.
                values = torch.tensor([0.0, -0.5])[: track.batch_size]
                correctness = torch.zeros(track.batch_size)
            else:
                values = torch.tensor([0.0, 1.0])[: track.batch_size]
                correctness = values.clone()
            return SimpleNamespace(
                rewards=values,
                component_rewards={
                    "answer_correctness": correctness,
                    "judge_failed": torch.zeros(track.batch_size),
                },
            )

    monkeypatch.setattr(dynamic, "ray", FakeRay)
    req = RolloutReq(
        sample_ids=["prompt-0", "prompt-1"],
        group_ids=["group-0", "group-1"],
        primitives={"text": Texts(texts=["problem-0", "problem-1"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=2)},
        stage_config={
            "retry_truncated_trajectories": True,
            "max_attempts_per_candidate": 2,
            "max_truncated_refills_per_prompt": 2,
            "dynamic_prompt_group_refill": True,
            "max_replacement_prompts": 1,
            "max_trajectory_multiplier": 3.0,
        },
        metadata=[{"answer": "0"}, {"answer": "1"}],
    )

    result = dynamic.run_dynamic_trainside_rollout(
        rollout_handle=Handle(),
        driver_reward=Reward(),
        req=req,
        rollout_id=7,
        target_prompt_count=1,
    )

    assert result.selected_prompt_indices == [1]
    assert result.consumed_prompt_indices == [0, 1]
    assert result.resp.tracks["ar"].rewards.tolist() == [0.0, 1.0]
    assert result.resp.tracks["ar"].status.tolist() == [1.0, 1.0]
    assert result.metrics["truncated_attempts"] == 1.0
    assert result.metrics["replacement_prompts"] == 1.0


def test_dynamic_scheduler_padding_keeps_independent_grpo_groups(monkeypatch):
    import unirl.trainer.dynamic_trainside as dynamic

    class Ref:
        def __init__(self, value):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(refs):
            return [ref.value for ref in refs]

        @staticmethod
        def wait(refs, num_returns=1, timeout=None):
            del timeout
            return list(refs[:num_returns]), list(refs[num_returns:])

    prompt_count = 8
    candidates_per_prompt = 8
    calls = {f"prompt-{i}": 0 for i in range(prompt_count)}

    def generate(req):
        prompt_id = req.sample_ids[0]
        call_index = calls[prompt_id]
        calls[prompt_id] += 1
        track = RolloutTrack(
            sample_ids=["temporary"],
            parent_ids=[prompt_id],
            conditions={"ar": SimpleNamespace(stop_reasons=["eos"])},
            segment=TextSegment.pack(
                tokens=[torch.tensor([call_index + 1])],
                log_probs=[torch.tensor([-0.1])],
            ),
            decoded=Texts(texts=[f"{prompt_id}-answer-{call_index}"]),
        )
        return SimpleNamespace(tracks={"ar": track})

    class RemoteCall:
        @staticmethod
        def remote(role_name, method, args, kwargs, grad, context):
            del role_name, kwargs, grad, context
            assert method == "generate"
            return Ref(generate(args[0]))

    class Worker:
        call = RemoteCall()

    class Transport:
        pass

    class Handle:
        workers = [Worker()]
        role_name = "rollout"
        pool = SimpleNamespace(transport_cls=Transport)
        rank_infos = [
            SimpleNamespace(
                dp_rank=0, tp_rank=0, is_pipeline_last_stage=True, sp_rank=0
            )
        ]
        world_size = 1
        sp_size = 1
        dp_size = 1

        @staticmethod
        def _rebind_tree(result, worker, worker_local):
            del worker, worker_local
            return result

    class Reward:
        backend = SimpleNamespace(thread_safe=False)

        @staticmethod
        def score_and_attach(*, req, track):
            prompt = req.primitives["text"].texts[0]
            if prompt == "problem-0":
                values = torch.zeros(track.batch_size)
                failed = torch.zeros(track.batch_size)
                failed[0] = 1.0
            else:
                values = torch.arange(track.batch_size, dtype=torch.float32) % 2
                failed = torch.zeros(track.batch_size)
            return SimpleNamespace(
                rewards=values,
                component_rewards={
                    "answer_correctness": values.clone(),
                    "judge_failed": failed,
                },
                process_annotations=(
                    None
                    if prompt == "problem-0"
                    else [
                        {"valid": True, "prompt": prompt, "sample": i}
                        for i in range(track.batch_size)
                    ]
                ),
            )

    monkeypatch.setattr(dynamic, "ray", FakeRay)
    req = RolloutReq(
        sample_ids=[f"prompt-{i}" for i in range(prompt_count)],
        group_ids=[f"group-{i}" for i in range(prompt_count)],
        primitives={"text": Texts(texts=[f"problem-{i}" for i in range(prompt_count)])},
        sampling_params={
            "ar": ARSamplingParams(samples_per_prompt=candidates_per_prompt)
        },
        stage_config={
            "retry_truncated_trajectories": False,
            "dynamic_prompt_group_refill": False,
            "max_trajectory_multiplier": 1.0,
        },
        metadata=[{"answer": str(i)} for i in range(prompt_count)],
    )

    result = dynamic.run_dynamic_trainside_rollout(
        rollout_handle=Handle(),
        driver_reward=Reward(),
        req=req,
        rollout_id=11,
        target_prompt_count=prompt_count,
    )

    track = result.resp.tracks["ar"]
    unique_parent_ids = list(dict.fromkeys(track.parent_ids))
    assert track.batch_size == 64
    assert len(unique_parent_ids) == 8
    assert [track.parent_ids.count(parent_id) for parent_id in unique_parent_ids] == [
        8
    ] * 8
    assert len(set(track.sample_ids)) == 64
    assert track.status.tolist() == [1.0] * 56 + [0.0] * 8
    assert result.metrics["padding_groups"] == 1.0
    assert len(track.process_annotations) == 64
    assert all(item is not None for item in track.process_annotations[:56])
    assert track.process_annotations[56:] == [None] * 8

    with_advantages = track.compute_advantages(normalize=True)
    assert with_advantages.advantages.shape == (64,)


def test_dynamic_scheduler_chunks_initial_candidates_and_counts_rpcs(monkeypatch):
    import unirl.trainer.dynamic_trainside as dynamic

    class Ref:
        def __init__(self, value):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(refs):
            return [ref.value for ref in refs]

        @staticmethod
        def wait(refs, num_returns=1, timeout=None):
            del timeout
            return list(refs[:num_returns]), list(refs[num_returns:])

    seen_reqs = []

    def generate(req):
        seen_reqs.append(req)
        size = int(req.sampling_params["ar"].samples_per_prompt)
        return SimpleNamespace(tracks={"ar": _track(size)})

    class RemoteCall:
        @staticmethod
        def remote(role_name, method, args, kwargs, grad, context):
            del role_name, kwargs, grad, context
            assert method == "generate"
            return Ref(generate(args[0]))

    class Worker:
        call = RemoteCall()

    class Transport:
        pass

    class Handle:
        workers = [Worker()]
        role_name = "rollout"
        pool = SimpleNamespace(transport_cls=Transport)
        rank_infos = [
            SimpleNamespace(
                dp_rank=0, tp_rank=0, is_pipeline_last_stage=True, sp_rank=0
            )
        ]

        @staticmethod
        def _rebind_tree(result, worker, worker_local):
            del worker, worker_local
            return result

    class Reward:
        backend = SimpleNamespace(thread_safe=False)

        @staticmethod
        def score_and_attach(*, req, track):
            del req
            values = torch.arange(track.batch_size, dtype=torch.float32) % 2
            return SimpleNamespace(
                rewards=values,
                component_rewards={
                    "answer_correctness": values.clone(),
                    "judge_failed": torch.zeros(track.batch_size),
                },
            )

    monkeypatch.setattr(dynamic, "ray", FakeRay)
    req = RolloutReq(
        sample_ids=["prompt-0"],
        group_ids=["group-0"],
        primitives={"text": Texts(texts=["problem-0"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=4)},
        stage_config={
            "dynamic_rollout_chunk_size": 2,
            "retry_truncated_trajectories": False,
            "dynamic_prompt_group_refill": False,
            "max_trajectory_multiplier": 1.0,
        },
        metadata=[{"answer": "0"}],
    )

    result = dynamic.run_dynamic_trainside_rollout(
        rollout_handle=Handle(),
        driver_reward=Reward(),
        req=req,
        rollout_id=13,
        target_prompt_count=1,
    )

    track = result.resp.tracks["ar"]
    assert len(seen_reqs) == 2
    assert [item.sampling_params["ar"].samples_per_prompt for item in seen_reqs] == [
        2,
        2,
    ]
    assert all(item.stage_config["rollout_text_batch_size"] == 2 for item in seen_reqs)
    assert all(len(item.stage_config["trajectory_seeds"]) == 2 for item in seen_reqs)
    assert track.sample_ids == [
        "prompt-0/a0",
        "prompt-0/a1",
        "prompt-0/a2",
        "prompt-0/a3",
    ]
    assert result.metrics["trajectory_jobs"] == 4.0
    assert result.metrics["rpc_jobs"] == 2.0
    assert result.metrics["average_rpc_batch_size"] == 2.0


@pytest.mark.parametrize(
    (
        "stop_reason",
        "expected_generate_calls",
        "expected_judge_calls",
        "expected_forced_zero",
    ),
    [
        ("max_new_tokens", 2, 1, 0.0),
        ("max_images", 4, 0, 2.0),
    ],
)
def test_soft_text_truncation_is_judged_but_image_truncation_stays_zero(
    monkeypatch,
    stop_reason,
    expected_generate_calls,
    expected_judge_calls,
    expected_forced_zero,
):
    import unirl.trainer.dynamic_trainside as dynamic

    class Ref:
        def __init__(self, value):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(refs):
            return [ref.value for ref in refs]

        @staticmethod
        def wait(refs, num_returns=1, timeout=None):
            del timeout
            return list(refs[:num_returns]), list(refs[num_returns:])

    calls = {"generate": 0, "judge": 0}

    def generate(req):
        del req
        calls["generate"] += 1
        track = RolloutTrack(
            sample_ids=["temporary"],
            parent_ids=["prompt-0"],
            conditions={"ar": SimpleNamespace(stop_reasons=[stop_reason])},
            segment=TextSegment.pack(
                tokens=[torch.tensor([1, 2])],
                log_probs=[torch.tensor([-0.1, -0.1])],
            ),
            decoded=Texts(texts=["answer"]),
        )
        return SimpleNamespace(tracks={"ar": track})

    class RemoteCall:
        @staticmethod
        def remote(role_name, method, args, kwargs, grad, context):
            del role_name, kwargs, grad, context
            assert method == "generate"
            return Ref(generate(args[0]))

    class Worker:
        call = RemoteCall()

    class Transport:
        pass

    class Handle:
        workers = [Worker()]
        role_name = "rollout"
        pool = SimpleNamespace(transport_cls=Transport)
        rank_infos = [
            SimpleNamespace(
                dp_rank=0, tp_rank=0, is_pipeline_last_stage=True, sp_rank=0
            )
        ]

        @staticmethod
        def _rebind_tree(result, worker, worker_local):
            del worker, worker_local
            return result

    class Reward:
        truncated_reward = "soft"
        backend = SimpleNamespace(thread_safe=False)

        @staticmethod
        def score_and_attach(*, req, track):
            del req
            calls["judge"] += 1
            correctness = torch.tensor([0.0, 1.0])[: track.batch_size]
            return SimpleNamespace(
                rewards=correctness.clone(),
                component_rewards={
                    "answer_correctness": correctness,
                    "judge_failed": torch.zeros(track.batch_size),
                },
            )

    monkeypatch.setattr(dynamic, "ray", FakeRay)
    req = RolloutReq(
        sample_ids=["prompt-0"],
        group_ids=["group-0"],
        primitives={"text": Texts(texts=["problem-0"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=2)},
        stage_config={
            "retry_truncated_trajectories": True,
            "max_attempts_per_candidate": 2,
            "max_truncated_refills_per_prompt": 4,
            "dynamic_prompt_group_refill": False,
            "max_trajectory_multiplier": 2.0,
        },
        metadata=[{"answer": "0"}],
    )

    result = dynamic.run_dynamic_trainside_rollout(
        rollout_handle=Handle(),
        driver_reward=Reward(),
        req=req,
        rollout_id=17,
        target_prompt_count=1,
    )

    assert calls["generate"] == expected_generate_calls
    assert calls["judge"] == expected_judge_calls
    assert result.metrics["forced_truncation_zero"] == expected_forced_zero
    track = result.resp.tracks["ar"]
    if stop_reason == "max_new_tokens":
        assert result.metrics["soft_text_truncations"] == 2.0
        assert track.component_rewards["text_truncated"].tolist() == [1.0, 1.0]
        assert track.component_rewards["judge_evaluated"].tolist() == [1.0, 1.0]
    else:
        assert result.metrics["soft_text_truncations"] == 0.0
        assert track.component_rewards["image_truncated"].tolist() == [1.0, 1.0]
        assert track.component_rewards["judge_evaluated"].tolist() == [0.0, 0.0]


class _FakeTextReward(RewardBackend):
    input_kind = "text"

    def __init__(self, rewards):
        super().__init__(model_name="fake-text")
        self.rewards = list(rewards)

    def compute_rewards(self, request):
        assert request.texts is not None
        size = request.batch_size
        return RewardResponse(
            rewards=self.rewards[:size],
            component_rewards={"answer_correctness": self.rewards[:size]},
            successes=[True] * size,
            errors=[None] * size,
        )

    def is_available(self):
        return True


def _soft_long_req(max_new_tokens=2560):
    return RolloutReq(
        sample_ids=["prompt-0"],
        group_ids=["group-0"],
        primitives={"text": Texts(texts=["problem"])},
        sampling_params={
            "ar": ARSamplingParams(
                samples_per_prompt=3,
                max_new_tokens=max_new_tokens,
            )
        },
    )


def _soft_long_track(lengths):
    return RolloutTrack(
        sample_ids=[f"candidate-{i}" for i in range(len(lengths))],
        parent_ids=["prompt-0"] * len(lengths),
        conditions={},
        segment=TextSegment.pack(
            tokens=[torch.ones(length, dtype=torch.long) for length in lengths],
            log_probs=[torch.zeros(length) for length in lengths],
        ),
        decoded=Texts(texts=[f"answer-{i}" for i in range(len(lengths))]),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"truncated_reward": "soft", "overlong_buffer_len": 0}, "must be positive"),
        (
            {"truncated_reward": "soft", "overlong_penalty_factor": -0.1},
            "must be non-negative",
        ),
    ],
)
def test_soft_long_configuration_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RewardService(_FakeTextReward([0.0]), **kwargs)


def test_soft_long_buffer_cannot_exceed_generation_cap():
    service = RewardService(
        _FakeTextReward([0.0]),
        truncated_reward="soft",
        overlong_buffer_len=513,
    )
    with pytest.raises(ValueError, match="must not exceed AR max_new_tokens"):
        service.score_and_attach(
            req=_soft_long_req(max_new_tokens=512),
            track=_soft_long_track([512]),
        )


def test_soft_long_penalty_and_components_follow_dapo_ramp():
    service = RewardService(
        _FakeTextReward([1.0, 1.0, 1.0]),
        truncated_reward="soft",
        overlong_buffer_len=512,
        overlong_penalty_factor=1.0,
    )

    scored = service.score_and_attach(
        req=_soft_long_req(max_new_tokens=2560),
        track=_soft_long_track([2048, 2304, 2560]),
    )

    assert scored.rewards.tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert scored.component_rewards["answer_correctness"].tolist() == [1.0, 1.0, 1.0]
    assert scored.component_rewards["overlong_reward"].tolist() == pytest.approx(
        [0.0, -0.5, -1.0]
    )
    assert scored.component_rewards["overlong"].tolist() == [0.0, 1.0, 1.0]
    assert scored.component_rewards["response_token_length"].tolist() == [
        2048.0,
        2304.0,
        2560.0,
    ]


def test_zero_reward_text_truncation_keeps_group_advantage_but_masks_its_update(
    monkeypatch,
):
    import unirl.trainer.dynamic_trainside as dynamic

    class Ref:
        def __init__(self, value):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(refs):
            return [ref.value for ref in refs]

        @staticmethod
        def wait(refs, num_returns=1, timeout=None):
            del timeout
            return list(refs[:num_returns]), list(refs[num_returns:])

    generate_calls = 0

    def generate(req):
        nonlocal generate_calls
        assert req.sampling_params["ar"].samples_per_prompt == 1
        truncated = generate_calls == 0
        generate_calls += 1
        tokens = torch.tensor([1, 2, 3]) if truncated else torch.tensor([4])
        track = RolloutTrack(
            sample_ids=["temporary"],
            parent_ids=["prompt-0"],
            conditions={
                "ar": SensenovaU1ARConditions.for_sample(
                    query="problem-0",
                    stop_reason="max_new_tokens" if truncated else "eos",
                )
            },
            segment=TextSegment.pack(
                tokens=[tokens],
                log_probs=[torch.full((tokens.numel(),), -0.1)],
            ),
            decoded=Texts(texts=["unfinished" if truncated else "correct answer"]),
        )
        return SimpleNamespace(tracks={"ar": track})

    class RemoteCall:
        @staticmethod
        def remote(role_name, method, args, kwargs, grad, context):
            del role_name, kwargs, grad, context
            assert method == "generate"
            return Ref(generate(args[0]))

    class Worker:
        call = RemoteCall()

    class Transport:
        pass

    class Handle:
        workers = [Worker()]
        role_name = "rollout"
        pool = SimpleNamespace(transport_cls=Transport)
        rank_infos = [
            SimpleNamespace(
                dp_rank=0, tp_rank=0, is_pipeline_last_stage=True, sp_rank=0
            )
        ]
        world_size = 1
        sp_size = 1
        dp_size = 1

        @staticmethod
        def _rebind_tree(result, worker, worker_local):
            del worker, worker_local
            return result

    class Reward:
        truncated_reward = "zero"
        backend = SimpleNamespace(thread_safe=False)

        @staticmethod
        def score_and_attach(*, req, track):
            del req
            assert track.batch_size == 1
            return SimpleNamespace(
                rewards=torch.tensor([1.0]),
                component_rewards={
                    "answer_correctness": torch.tensor([1.0]),
                    "judge_failed": torch.tensor([0.0]),
                },
            )

    monkeypatch.setattr(dynamic, "ray", FakeRay)
    req = RolloutReq(
        sample_ids=["prompt-0"],
        group_ids=["group-0"],
        primitives={"text": Texts(texts=["problem-0"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=2)},
        stage_config={
            "dynamic_rollout_chunk_size": 1,
            "retry_truncated_trajectories": False,
            "ignore_truncated_samples": True,
            "dynamic_prompt_group_refill": False,
            "max_trajectory_multiplier": 1.0,
        },
        metadata=[{"answer": "correct answer"}],
    )

    result = dynamic.run_dynamic_trainside_rollout(
        rollout_handle=Handle(),
        driver_reward=Reward(),
        req=req,
        rollout_id=19,
        target_prompt_count=1,
    )

    track = result.resp.tracks["ar"]
    assert track.rewards.tolist() == [0.0, 1.0]
    assert track.status.tolist() == [0.0, 1.0]
    assert track.component_rewards["text_truncated"].tolist() == [1.0, 0.0]
    # Status is deliberately applied only after full-group GRPO advantages.
    advantages = track.compute_advantages(normalize=True).advantages
    assert advantages[0].item() < 0.0
    assert advantages[1].item() > 0.0
    assert result.metrics["masked_truncated_samples"] == 1.0
    assert result.metrics["effective_train_samples"] == 1.0
    assert result.metrics["effective_train_tokens"] == 1.0
    assert result.metrics["groups_with_one_trainable_sample"] == 1.0


def test_group_advantage_excludes_truncated_row_from_statistics():
    track = RolloutTrack(
        sample_ids=[f"sample-{i}" for i in range(8)],
        parent_ids=["prompt-0"] * 8,
        rewards=torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        status=torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )

    full_group = track.compute_advantages(normalize=True).advantages
    valid_group = track.compute_advantages(
        normalize=True, valid_mask=track.status
    ).advantages

    # Historical mode: the zero-reward truncation depresses the 8-way mean.
    assert full_group[1].item() < 0.0
    # New mode: row 0 is absent from statistics and is represented by adv=0.
    # The seven valid rewards are [0..6], with mean=3 and population std=2.
    assert valid_group.tolist() == pytest.approx(
        [0.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], abs=1e-6
    )


def test_group_advantage_mask_handles_empty_and_singleton_valid_groups():
    track = RolloutTrack(
        sample_ids=["a0", "a1", "b0", "b1"],
        parent_ids=["a", "a", "b", "b"],
        rewards=torch.tensor([1.0, 9.0, 2.0, 8.0]),
    )
    advantages = track.compute_advantages(
        normalize=True, valid_mask=torch.tensor([0.0, 0.0, 1.0, 0.0])
    ).advantages
    assert advantages.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert torch.isfinite(advantages).all()


def test_grpo_token_mean_excludes_masked_sample_tokens_from_denominator():
    from unirl.algorithms.grpo import GRPO

    algorithm = object.__new__(GRPO)
    algorithm.loss_agg_mode = "token-mean"
    algorithm.horizon = 8
    values = torch.tensor([1.0, 3.0, 10.0, 20.0, 30.0], requires_grad=True)

    loss = algorithm._reduce_token_loss(
        values,
        torch.tensor([2, 3]),
        sample_mask=torch.tensor([1.0, 0.0]),
    )

    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    assert values.grad.tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0, 0.0])


def test_grpo_masked_policy_metrics_cover_only_effective_rows_and_tokens():
    from unirl.algorithms.grpo import GRPO

    metrics = GRPO._masked_policy_metrics(
        new_logp=torch.tensor([-0.1, -0.2, 1.0, 2.0, 3.0]),
        old_logp=torch.tensor([-0.1, -0.2, 0.0, 0.0, 0.0]),
        lengths=torch.tensor([2, 3]),
        sample_mask=torch.tensor([1.0, 0.0]),
        clip_range=0.1,
        clip_range_high=None,
    )

    assert metrics["effective_train_samples"] == 1.0
    assert metrics["effective_train_tokens"] == 2.0
    assert metrics["train_sample_fraction"] == pytest.approx(0.5)
    assert metrics["train_token_fraction"] == pytest.approx(0.4)
    assert metrics["masked_ratio_mean"] == pytest.approx(1.0)
    assert metrics["masked_clip_fraction"] == pytest.approx(0.0)


def test_repetition_truncation_recovery_keeps_complete_prefix_for_sca():
    from unirl.trainer.dynamic_trainside import (
        _recover_repetition_truncation,
        _reward_only_track,
    )

    prefix_ids = list(range(80))
    block_ids = list(range(100, 120))
    token_ids = prefix_ids + block_ids * 3
    token_count = len(token_ids)
    # The first complete SCA step ends exactly where the second block starts.
    text = "a" * 99 + "。" + "b" * (token_count - 100)
    segment = TextSegment.pack(
        tokens=[torch.tensor(token_ids)],
        log_probs=[torch.full((token_count,), -0.1)],
        decoded_char_starts=[torch.arange(token_count)],
        decoded_char_ends=[torch.arange(1, token_count + 1)],
    )
    track = RolloutTrack(
        sample_ids=["p0/a0"],
        parent_ids=["p0"],
        conditions={
            "ar": SensenovaU1ARConditions.for_sample(
                query="problem", stop_reason="max_new_tokens"
            )
        },
        segment=segment,
        decoded=Texts(texts=[text]),
    )

    recovered = _recover_repetition_truncation(
        track,
        enabled=True,
        min_block_tokens=16,
        max_block_tokens=64,
        min_repeats=3,
        min_prefix_tokens=64,
        tail_tolerance_tokens=0,
        require_complete_step=True,
    )

    assert _stop_reason(recovered) == "repetition"
    assert recovered.decoded.texts == [text[:100]]
    assert recovered.repetition_recovered.tolist() == [1.0]
    assert recovered.original_text_truncated.tolist() == [1.0]
    assert recovered.effective_text_token_count.tolist() == [100.0]
    assert recovered.removed_repetition_token_count.tolist() == [40.0]
    assert recovered.segment.loss_mask.tolist() == [1.0] * 100 + [0.0] * 40
    assert _reward_only_track(recovered).segment.lengths.tolist() == [100]


def test_repetition_truncation_recovery_preserves_existing_token_mask():
    from unirl.trainer.dynamic_trainside import _recover_repetition_truncation

    prefix_ids = list(range(80))
    block_ids = list(range(100, 120))
    token_ids = prefix_ids + block_ids * 3
    token_count = len(token_ids)
    text = "a" * 99 + "。" + "b" * (token_count - 100)
    existing_mask = torch.ones(token_count)
    existing_mask[10] = 0.0
    segment = TextSegment.pack(
        tokens=[torch.tensor(token_ids)],
        log_probs=[torch.full((token_count,), -0.1)],
        loss_mask=[existing_mask],
        decoded_char_starts=[torch.arange(token_count)],
        decoded_char_ends=[torch.arange(1, token_count + 1)],
    )
    track = RolloutTrack(
        sample_ids=["p0/a0"],
        parent_ids=["p0"],
        conditions={
            "ar": SensenovaU1ARConditions.for_sample(
                query="problem", stop_reason="max_new_tokens"
            )
        },
        segment=segment,
        decoded=Texts(texts=[text]),
    )

    recovered = _recover_repetition_truncation(
        track,
        enabled=True,
        min_block_tokens=16,
        max_block_tokens=64,
        min_repeats=3,
        min_prefix_tokens=64,
        tail_tolerance_tokens=0,
        require_complete_step=True,
    )

    assert recovered.segment.loss_mask[10].item() == 0.0
    assert recovered.segment.loss_mask[:100].sum().item() == 99.0
    assert recovered.segment.loss_mask[100:].sum().item() == 0.0
