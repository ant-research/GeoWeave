from types import SimpleNamespace

import torch

from transformers.cache_utils import DynamicCache

from unirl.models.sensenova_u1.rollout_batching import (
    RolloutPhase,
    _State,
    _TextGroup,
    _coalesce_text_groups,
    batched_interleave_decode,
)
from unirl.models.sensenova_u1.rollout_metrics import SampleRolloutMetrics
from unirl.trainer.dynamic_trainside import (
    TrajectoryJob,
    _merge_session_jobs,
    _percentile,
    _pop_live_microbundle,
    _trajectory_req,
)
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _metrics(index: int) -> SampleRolloutMetrics:
    return SampleRolloutMetrics(
        rollout_id=f"candidate-{index}",
        prompt_id="prompt-0",
        group_id="group-0",
        rewrite_id=index,
        rank=0,
        local_sample_index=index,
    )


def _state(index: int) -> _State:
    return _State(index, _metrics(index), (64, 64))


def _cache(rows: int, seq_len: int, *, offset: int = 0) -> DynamicCache:
    keys = torch.arange(
        offset,
        offset + rows * seq_len * 2,
        dtype=torch.float32,
    ).reshape(rows, 1, seq_len, 2)
    return DynamicCache(ddp_cache_data=[(keys, keys + 1000)])


def _group(
    indices,
    *,
    seq_len: int = 3,
    t_idx: int = 7,
    uncond_seq_len: int | None = None,
    offset: int = 0,
) -> _TextGroup:
    states = [_state(index) for index in indices]
    uncond_cache = (
        _cache(len(states), uncond_seq_len, offset=offset + 100)
        if uncond_seq_len is not None
        else None
    )
    return _TextGroup(
        states=states,
        cache=_cache(len(states), seq_len, offset=offset),
        t_idx=t_idx,
        uncond_cache=uncond_cache,
        uncond_t_idx=t_idx if uncond_cache is not None else None,
        next_tokens=torch.tensor(indices, dtype=torch.long),
        next_logps=torch.tensor(indices, dtype=torch.float32) * -0.1,
    )


def _request(*, continuous: bool) -> RolloutReq:
    return RolloutReq(
        sample_ids=["prompt-0"],
        group_ids=["group-0"],
        primitives={"text": Texts(texts=["problem"])},
        sampling_params={
            "ar": ARSamplingParams(samples_per_prompt=4),
            "diffusion": DiffusionSamplingParams(seed=123),
        },
        stage_config={
            "continuous_batching": continuous,
            "continuous_rollout_pool_size": 4,
            "continuous_text_batch_size": 2,
        },
        metadata=[{"answer": "42"}],
    )


def test_state_finished_property_preserves_phase_compatibility():
    state = _state(0)
    assert state.phase is RolloutPhase.READY_TEXT
    assert not state.finished

    state.finished = True
    assert state.phase is RolloutPhase.FINISHED
    assert state.finished

    state.finished = False
    assert state.phase is RolloutPhase.READY_TEXT


def test_coalesce_merges_compatible_groups_and_rechunks_in_stable_order():
    first = _group([0], offset=0)
    second = _group([1, 2], offset=20)
    original_keys = torch.cat(
        [first.cache.layers[0].keys, second.cache.layers[0].keys], dim=0
    )

    groups, stats = _coalesce_text_groups([first, second], text_batch_size=2)

    assert stats.calls == 1
    assert stats.ready_groups == 2
    assert stats.compatible_groups == 2
    assert stats.incompatible_groups == 0
    assert stats.underfilled_slots == 1
    assert stats.refilled_slots == 0
    assert stats.compactions == 1
    assert [[state.sample_index for state in group.states] for group in groups] == [
        [0, 1],
        [2],
    ]
    assert [group.next_tokens.tolist() for group in groups] == [[0, 1], [2]]
    assert torch.equal(groups[0].cache.layers[0].keys, original_keys[:2])
    assert torch.equal(groups[1].cache.layers[0].keys, original_keys[2:])


def test_coalesce_is_noop_when_compatible_groups_are_already_packed():
    first = _group([0, 1], offset=0)
    second = _group([2, 3], offset=20)

    groups, stats = _coalesce_text_groups([first, second], text_batch_size=2)

    assert groups == [first, second]
    assert stats.compatible_groups == 2
    assert stats.underfilled_slots == 0
    assert stats.refilled_slots == 0
    assert stats.compactions == 0


def test_coalesce_keeps_incompatible_kv_and_cfg_groups_separate():
    groups, stats = _coalesce_text_groups(
        [
            _group([0], seq_len=3),
            _group([1], seq_len=4),
            _group([2], seq_len=3, uncond_seq_len=3),
        ],
        text_batch_size=2,
    )

    assert stats.compatible_groups == 0
    assert stats.incompatible_groups == 3
    assert stats.underfilled_slots == 3
    assert stats.refilled_slots == 0
    assert stats.compactions == 0
    assert [[state.sample_index for state in group.states] for group in groups] == [
        [0],
        [1],
        [2],
    ]



def test_coalesce_reports_slots_eliminated_by_true_refill():
    groups, stats = _coalesce_text_groups(
        [_group([0], offset=0), _group([1], offset=20)], text_batch_size=2
    )

    assert [[state.sample_index for state in group.states] for group in groups] == [
        [0, 1]
    ]
    assert stats.underfilled_slots == 2
    assert stats.refilled_slots == 2
    assert stats.compactions == 1


def test_live_microbundle_keeps_only_adjacent_same_prompt_jobs_together():
    req = RolloutReq(
        sample_ids=["prompt-0", "prompt-1"],
        group_ids=["group-0", "group-1"],
        primitives={"text": Texts(texts=["problem-0", "problem-1"])},
        sampling_params={
            "ar": ARSamplingParams(samples_per_prompt=4),
            "diffusion": DiffusionSamplingParams(seed=123),
        },
        stage_config={"continuous_batching": True},
    )
    from unirl.trainer.dynamic_trainside import _build_jobs

    jobs, _ = _build_jobs(req, rollout_id=9)
    queue = __import__("collections").deque(jobs)

    first = _pop_live_microbundle(queue, max_size=2)
    second = _pop_live_microbundle(queue, max_size=2)
    third = _pop_live_microbundle(queue, max_size=2)

    assert [job.candidate_ids[0] for job in first] == [
        "prompt-0/a0",
        "prompt-0/a1",
    ]
    assert [job.candidate_ids[0] for job in second] == [
        "prompt-0/a2",
        "prompt-0/a3",
    ]
    assert [job.candidate_ids[0] for job in third] == [
        "prompt-1/a0",
        "prompt-1/a1",
    ]


def test_percentile_uses_linear_interpolation():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([4.0], 0.9) == 4.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.9) == 3.7

def test_trajectory_request_separates_rpc_pool_from_active_text_batch():
    req = _request(continuous=True)

    prompt_id, candidate_ids, job_req = _trajectory_req(
        req,
        prompt_index=0,
        rewrite_indices=[0, 1, 2, 3],
        attempt_indices=[0, 0, 0, 0],
        rollout_id=9,
    )

    assert prompt_id == "prompt-0"
    assert candidate_ids == (
        "prompt-0/a0",
        "prompt-0/a1",
        "prompt-0/a2",
        "prompt-0/a3",
    )
    assert job_req.sampling_params["ar"].samples_per_prompt == 4
    assert job_req.stage_config["rollout_text_batch_size"] == 2
    assert len(job_req.stage_config["trajectory_seeds"]) == 4
    assert len(set(job_req.stage_config["trajectory_seeds"])) == 4
    assert len(job_req.init_noise_group_ids) == 4


def test_trajectory_request_uses_full_rpc_batch_when_continuous_is_disabled():
    req = _request(continuous=False)

    _, _, job_req = _trajectory_req(
        req,
        prompt_index=0,
        rewrite_indices=[0, 1, 2, 3],
        attempt_indices=[0, 0, 0, 0],
        rollout_id=9,
    )

    assert job_req.stage_config["rollout_text_batch_size"] == 4


def test_dynamic_scheduler_uses_one_rpc_pool_without_changing_trajectory_count(
    monkeypatch,
):
    import unirl.trainer.dynamic_trainside as dynamic
    from unirl.types.rollout_resp import RolloutTrack
    from unirl.types.segments import TextSegment

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

    seen_requests = []

    def generate(req):
        seen_requests.append(req)
        size = req.sampling_params["ar"].samples_per_prompt
        track = RolloutTrack(
            sample_ids=[f"temporary-{i}" for i in range(size)],
            parent_ids=[req.sample_ids[0]] * size,
            conditions={},
            segment=TextSegment.pack(
                tokens=[torch.tensor([i + 1]) for i in range(size)],
                log_probs=[torch.tensor([-0.1]) for _ in range(size)],
            ),
            decoded=Texts(texts=[f"answer-{i}" for i in range(size)]),
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
            del req
            values = torch.arange(track.batch_size, dtype=torch.float32) % 2
            return SimpleNamespace(
                rewards=values,
                component_rewards={"answer_correctness": values.clone()},
            )

    monkeypatch.setattr(dynamic, "ray", FakeRay)
    req = _request(continuous=True)
    req.stage_config.update(
        {
            "retry_truncated_trajectories": False,
            "dynamic_prompt_group_refill": False,
            "max_trajectory_multiplier": 1.0,
        }
    )

    result = dynamic.run_dynamic_trainside_rollout(
        rollout_handle=Handle(),
        driver_reward=Reward(),
        req=req,
        rollout_id=9,
        target_prompt_count=1,
    )

    assert len(seen_requests) == 1
    assert seen_requests[0].sampling_params["ar"].samples_per_prompt == 4
    assert seen_requests[0].stage_config["rollout_text_batch_size"] == 2
    assert result.resp.tracks["ar"].sample_ids == [
        "prompt-0/a0",
        "prompt-0/a1",
        "prompt-0/a2",
        "prompt-0/a3",
    ]
    assert result.owner_ranks == [0, 0, 0, 0]
    assert result.metrics["trajectory_jobs"] == 4.0
    assert result.metrics["rpc_jobs"] == 1.0
    assert result.metrics["average_rpc_batch_size"] == 4.0
    assert result.metrics["rpc_duration_mean_s"] >= 0.0
    assert result.metrics["rpc_duration_p50_s"] >= 0.0
    assert result.metrics["rpc_duration_p90_s"] >= result.metrics["rpc_duration_p50_s"]
    assert result.metrics["rpc_duration_max_s"] >= result.metrics["rpc_duration_p90_s"]
    assert result.metrics["rpc_duration_cv"] >= 0.0
    assert result.metrics["rpc_seconds_per_trajectory"] >= 0.0
    assert result.metrics["continuous_text_batch_size"] == 2.0


def test_dynamic_scheduler_reuses_one_persistent_worker_session(monkeypatch):
    import queue as queue_module
    import threading

    import unirl.trainer.dynamic_trainside as dynamic
    from unirl.types.rollout_resp import RolloutTrack
    from unirl.types.segments import TextSegment

    class LocalQueue:
        def __init__(self):
            self.queue = queue_module.Queue()

        def put(self, value):
            self.queue.put(value)

        def get(self):
            return self.queue.get()

        def shutdown(self, force=False):
            del force

    class ThreadRef:
        def __init__(self, target):
            self.error = None

            def run():
                try:
                    target()
                except BaseException as exc:
                    self.error = exc

            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()

        def result(self):
            self.thread.join(timeout=5)
            assert not self.thread.is_alive()
            if self.error is not None:
                raise self.error

    class FakeRay:
        @staticmethod
        def get(refs):
            if isinstance(refs, list):
                return [ref.result() for ref in refs]
            return refs.result()

    seen_requests = []
    session_starts = []

    def generate(req):
        seen_requests.append(req)
        size = req.sampling_params["ar"].samples_per_prompt
        return SimpleNamespace(
            tracks={
                "ar": RolloutTrack(
                    sample_ids=[f"temporary-{i}" for i in range(size)],
                    parent_ids=[req.sample_ids[0]] * size,
                    conditions={},
                    segment=TextSegment.pack(
                        tokens=[torch.tensor([i + 1]) for i in range(size)],
                        log_probs=[torch.tensor([-0.1]) for _ in range(size)],
                    ),
                    decoded=Texts(texts=[f"answer-{i}" for i in range(size)]),
                )
            }
        )

    class SessionCall:
        @staticmethod
        def remote(role_name, method, input_queue, output_queue):
            session_starts.append((role_name, method))

            def serve():
                while True:
                    item = input_queue.get()
                    if item is None:
                        return
                    request_id, args, kwargs = item
                    del kwargs
                    output_queue.put(("ok", request_id, generate(args[0])))

            return ThreadRef(serve)

    class Worker:
        session_call = SessionCall()

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
            del req
            values = torch.arange(track.batch_size, dtype=torch.float32) % 2
            return SimpleNamespace(
                rewards=values,
                component_rewards={"answer_correctness": values.clone()},
            )

    monkeypatch.setattr(dynamic, "ray", FakeRay)
    monkeypatch.setattr(dynamic, "_make_session_queue", LocalQueue)
    req = RolloutReq(
        sample_ids=["prompt-0", "prompt-1"],
        group_ids=["group-0", "group-1"],
        primitives={"text": Texts(texts=["problem-0", "problem-1"])},
        sampling_params={
            "ar": ARSamplingParams(samples_per_prompt=4),
            "diffusion": DiffusionSamplingParams(seed=123),
        },
        stage_config={
            "continuous_batching": True,
            "persistent_worker_session": True,
            "continuous_rollout_pool_size": 4,
            "continuous_text_batch_size": 2,
            "retry_truncated_trajectories": False,
            "dynamic_prompt_group_refill": False,
            "max_trajectory_multiplier": 1.0,
        },
        metadata=[{"answer": "1"}, {"answer": "2"}],
    )

    result = dynamic.run_dynamic_trainside_rollout(
        rollout_handle=Handle(),
        driver_reward=Reward(),
        req=req,
        rollout_id=11,
        target_prompt_count=2,
    )

    assert session_starts == [("rollout", "generate")]
    assert len(seen_requests) == 2
    assert result.metrics["persistent_worker_session"] == 1.0
    assert result.metrics["persistent_session_count"] == 1.0
    assert result.metrics["rpc_jobs"] == 2.0
    assert result.metrics["trajectory_jobs"] == 8.0
    assert result.resp.tracks["ar"].batch_size == 8



def test_dynamic_scheduler_merges_request_window_for_row_admission(monkeypatch):
    import queue as queue_module
    import threading

    import unirl.trainer.dynamic_trainside as dynamic
    from unirl.types.rollout_resp import RolloutTrack
    from unirl.types.segments import TextSegment

    class LocalQueue:
        def __init__(self):
            self.queue = queue_module.Queue()

        def put(self, value):
            self.queue.put(value)

        def get(self):
            return self.queue.get()

        def shutdown(self, force=False):
            del force

    class ThreadRef:
        def __init__(self, target):
            self.error = None

            def run():
                try:
                    target()
                except BaseException as exc:
                    self.error = exc

            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()

        def result(self):
            self.thread.join(timeout=5)
            assert not self.thread.is_alive()
            if self.error is not None:
                raise self.error

    class FakeRay:
        @staticmethod
        def get(refs):
            if isinstance(refs, list):
                return [ref.result() for ref in refs]
            return refs.result()

    seen_requests = []
    session_starts = []

    def generate(req):
        seen_requests.append(req)
        samples_per_prompt = req.sampling_params["ar"].samples_per_prompt
        size = req.batch_size * samples_per_prompt
        parent_ids = [
            sample_id
            for sample_id in req.sample_ids
            for _ in range(samples_per_prompt)
        ]
        return SimpleNamespace(
            tracks={
                "ar": RolloutTrack(
                    sample_ids=[f"temporary-{i}" for i in range(size)],
                    parent_ids=parent_ids,
                    conditions={},
                    segment=TextSegment.pack(
                        tokens=[torch.tensor([i + 1]) for i in range(size)],
                        log_probs=[torch.tensor([-0.1]) for _ in range(size)],
                    ),
                    decoded=Texts(texts=[f"answer-{i}" for i in range(size)]),
                )
            }
        )

    class SessionCall:
        @staticmethod
        def remote(role_name, method, input_queue, output_queue):
            session_starts.append((role_name, method))

            def serve():
                while True:
                    item = input_queue.get()
                    if item is None:
                        return
                    request_id, args, kwargs = item
                    del kwargs
                    output_queue.put(("ok", request_id, generate(args[0])))

            return ThreadRef(serve)

    class Worker:
        session_call = SessionCall()

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
            del req
            values = torch.arange(track.batch_size, dtype=torch.float32) % 2
            return SimpleNamespace(
                rewards=values,
                component_rewards={"answer_correctness": values.clone()},
            )

    monkeypatch.setattr(dynamic, "ray", FakeRay)
    monkeypatch.setattr(dynamic, "_make_session_queue", LocalQueue)
    req = RolloutReq(
        sample_ids=["prompt-0", "prompt-1"],
        group_ids=["group-0", "group-1"],
        primitives={"text": Texts(texts=["problem-0", "problem-1"])},
        sampling_params={
            "ar": ARSamplingParams(samples_per_prompt=4),
            "diffusion": DiffusionSamplingParams(seed=123),
        },
        stage_config={
            "continuous_batching": True,
            "persistent_worker_session": True,
            "continuous_request_admission": True,
            "continuous_session_window_size": 8,
            "continuous_rollout_pool_size": 4,
            "continuous_text_batch_size": 2,
            "retry_truncated_trajectories": False,
            "dynamic_prompt_group_refill": False,
            "max_trajectory_multiplier": 1.0,
        },
        metadata=[{"answer": "1"}, {"answer": "2"}],
    )

    result = dynamic.run_dynamic_trainside_rollout(
        rollout_handle=Handle(),
        driver_reward=Reward(),
        req=req,
        rollout_id=13,
        target_prompt_count=2,
    )

    assert session_starts == [("rollout", "generate")]
    assert len(seen_requests) == 1
    merged = seen_requests[0]
    assert merged.batch_size == 8
    assert merged.sampling_params["ar"].samples_per_prompt == 1
    assert merged.stage_config["continuous_request_admission"] is True
    assert merged.stage_config["trajectory_candidate_ids"] == [
        "prompt-0/a0",
        "prompt-0/a1",
        "prompt-0/a2",
        "prompt-0/a3",
        "prompt-1/a0",
        "prompt-1/a1",
        "prompt-1/a2",
        "prompt-1/a3",
    ]
    assert len(merged.stage_config["trajectory_seeds"]) == 8
    assert len(set(merged.stage_config["trajectory_seeds"])) == 8
    assert len(merged.init_noise_group_ids) == 8
    assert result.metrics["persistent_worker_session"] == 1.0
    assert result.metrics["persistent_session_count"] == 1.0
    assert result.metrics["continuous_request_admission"] == 1.0
    assert result.metrics["continuous_session_window_size"] == 8.0
    assert result.metrics["continuous_rollout_pool_size"] == 4.0
    assert result.metrics["rpc_jobs"] == 1.0
    assert result.metrics["logical_rpc_jobs"] == 8.0
    assert result.metrics["trajectory_jobs"] == 8.0
    assert result.metrics["average_rpc_batch_size"] == 8.0
    assert result.resp.tracks["ar"].batch_size == 8
    assert result.resp.tracks["ar"].sample_ids == [
        "prompt-0/a0",
        "prompt-0/a1",
        "prompt-0/a2",
        "prompt-0/a3",
        "prompt-1/a0",
        "prompt-1/a1",
        "prompt-1/a2",
        "prompt-1/a3",
    ]

def test_row_level_admission_refills_active_pool_before_window_tail():
    states = []
    pending = list(range(5))
    capacities = []

    def admit(capacity, block_when_idle):
        del block_when_idle
        capacities.append(capacity)
        admitted = []
        for _ in range(min(capacity, len(pending))):
            index = pending.pop(0)
            state = _state(index)
            states.append(state)
            admitted.append(
                _TextGroup(
                    states=[state],
                    cache=_cache(1, 3, offset=index * 10),
                    t_idx=7,
                    uncond_cache=None,
                    uncond_t_idx=None,
                    next_tokens=torch.tensor([99]),
                    next_logps=torch.tensor([-0.1]),
                )
            )
        return admitted, not pending

    batched_interleave_decode(
        model=SimpleNamespace(),
        tokenizer=None,
        base_cache=None,
        t_idx=0,
        base_uncond_cache=None,
        uncond_t_idx=None,
        start_logits=torch.empty(0),
        states=states,
        sample_fn=lambda *_args: (_ for _ in ()).throw(AssertionError()),
        max_new_tokens=8,
        max_images=0,
        stop_ids=[99],
        img_start_token_id=100,
        diffuse_batch_fn=lambda *_args: (_ for _ in ()).throw(AssertionError()),
        text_batch_size=2,
        diffusion_batch_size=1,
        reencode_batch_size=1,
        num_diffusion_steps=1,
        device=torch.device("cpu"),
        continuous_batching=True,
        initial_text_groups=[],
        admit_fn=admit,
        max_active_states=2,
    )

    assert capacities == [2, 2, 2]
    assert [state.sample_index for state in states] == [0, 1, 2, 3, 4]
    assert all(state.finished for state in states)
    assert [state.metrics.stop_reason for state in states] == ["eos"] * 5


def test_merge_session_jobs_preserves_candidate_seed_and_noise_identity():
    req = _request(continuous=True)
    jobs = []
    for job_index, rewrite_index in enumerate([0, 1, 2]):
        prompt_id, candidate_ids, job_req = _trajectory_req(
            req,
            prompt_index=0,
            rewrite_indices=[rewrite_index],
            attempt_indices=[0],
            rollout_id=11,
        )
        jobs.append(
            TrajectoryJob(
                job_index=job_index,
                prompt_index=0,
                rewrite_indices=(rewrite_index,),
                attempt_indices=(0,),
                prompt_id=prompt_id,
                candidate_ids=candidate_ids,
                req=job_req,
            )
        )

    merged = _merge_session_jobs(jobs, text_batch_size=2)

    assert merged.batch_size == 3
    assert merged.sampling_params["ar"].samples_per_prompt == 1
    assert merged.stage_config["continuous_request_admission"] is True
    assert merged.stage_config["rollout_text_batch_size"] == 2
    assert merged.stage_config["trajectory_candidate_ids"] == [
        "prompt-0/a0",
        "prompt-0/a1",
        "prompt-0/a2",
    ]
    assert len(set(merged.stage_config["trajectory_seeds"])) == 3
    assert merged.init_noise_group_ids == [
        job.req.init_noise_group_ids[0] for job in jobs
    ]


def test_decode_streams_each_finished_row_once_before_next_admission():
    states = []
    pending = list(range(5))
    events = []

    def admit(capacity, block_when_idle):
        events.append(("admit", capacity, block_when_idle))
        admitted = []
        for _ in range(min(capacity, len(pending))):
            index = pending.pop(0)
            state = _state(index)
            states.append(state)
            admitted.append(
                _TextGroup(
                    states=[state],
                    cache=_cache(1, 3, offset=index * 10),
                    t_idx=7,
                    uncond_cache=None,
                    uncond_t_idx=None,
                    next_tokens=torch.tensor([99]),
                    next_logps=torch.tensor([-0.1]),
                )
            )
        return admitted, not pending

    def finish(finished):
        events.append(
            ("finish", [state.sample_index for state in finished])
        )

    batched_interleave_decode(
        model=SimpleNamespace(),
        tokenizer=None,
        base_cache=None,
        t_idx=0,
        base_uncond_cache=None,
        uncond_t_idx=None,
        start_logits=torch.empty(0),
        states=states,
        sample_fn=lambda *_args: (_ for _ in ()).throw(AssertionError()),
        max_new_tokens=8,
        max_images=0,
        stop_ids=[99],
        img_start_token_id=100,
        diffuse_batch_fn=lambda *_args: (_ for _ in ()).throw(AssertionError()),
        text_batch_size=2,
        diffusion_batch_size=1,
        reencode_batch_size=1,
        num_diffusion_steps=1,
        device=torch.device("cpu"),
        continuous_batching=True,
        initial_text_groups=[],
        admit_fn=admit,
        max_active_states=2,
        finish_fn=finish,
    )

    assert events == [
        ("admit", 2, True),
        ("finish", [0, 1]),
        ("admit", 2, True),
        ("finish", [2, 3]),
        ("admit", 2, True),
        ("finish", [4]),
    ]


def test_trainside_engine_live_session_prepares_requests_and_restores_mode(
    monkeypatch,
):
    from unirl.rollout.engine.trainside import engine as engine_module
    from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine

    model = torch.nn.Linear(1, 1)
    model.train()
    requests = [_request(continuous=True), _request(continuous=True)]
    prepared = []
    emitted = []

    def ensure_sigmas(req, policy):
        assert policy == "schedule-policy"
        assert model.training is False
        req.sigmas = torch.tensor([1.0, 0.0])
        prepared.append(req)

    monkeypatch.setattr(engine_module, "ensure_req_sigmas", ensure_sigmas)

    class Pipeline:
        def generate_live(self, *, admit_fn, emit_fn):
            for request_id in range(2):
                items, closed = admit_fn(1, True)
                assert closed is (request_id == 1)
                assert len(items) == 1
                queued_id, args, kwargs = items[0]
                assert queued_id == request_id
                assert kwargs == {}
                assert args[0].sigmas is not None
                assert torch.is_grad_enabled() is False
                emit_fn(queued_id, args[0])

    queued = [
        (0, (requests[0],), {}),
        (1, (requests[1],), {}),
    ]

    def admit(capacity, block_when_idle):
        assert capacity == 1
        assert block_when_idle is True
        item = queued.pop(0)
        return [item], not queued

    rollout = object.__new__(TrainsideRolloutEngine)
    rollout.pipeline = Pipeline()
    rollout.schedule_policy = "schedule-policy"
    rollout._models = [model]
    rollout.forward_batch_size = None

    rollout.generate_live(
        admit_fn=admit,
        emit_fn=lambda request_id, response: emitted.append(
            (request_id, response)
        ),
    )

    assert prepared == requests
    assert emitted == [(0, requests[0]), (1, requests[1])]
    assert model.training is True


def test_worker_live_session_resolves_packs_streams_and_closes():
    import queue as queue_module

    from unirl.distributed.group.worker import Worker as PhysicalWorker

    input_queue = queue_module.Queue()
    output_queue = queue_module.Queue()
    input_queue.put((3, ("first",), {}))
    input_queue.put((4, ("second",), {}))
    input_queue.put(None)

    class Role:
        @staticmethod
        def generate_live(*, admit_fn, emit_fn):
            admitted, closed = admit_fn(4, True)
            assert closed is True
            for request_id, args, kwargs in admitted:
                assert kwargs == {}
                emit_fn(request_id, args[0])

    worker = object.__new__(PhysicalWorker)
    worker._roles = {"rollout": Role()}
    worker._resolve_call_inputs = lambda args, kwargs: (
        tuple(value.upper() for value in args),
        kwargs,
    )
    worker._pack_call_result = lambda result: f"packed:{result}"

    worker.live_session_call(
        "rollout", "generate_live", input_queue, output_queue
    )

    assert output_queue.get_nowait() == ("ok", 3, "packed:FIRST")
    assert output_queue.get_nowait() == ("ok", 4, "packed:SECOND")
    assert output_queue.get_nowait() == ("closed", None, None)


def test_worker_live_session_admits_atomic_microbundle_over_initial_capacity():
    import queue as queue_module

    from unirl.distributed.group.worker import Worker as PhysicalWorker

    input_queue = queue_module.Queue()
    output_queue = queue_module.Queue()
    input_queue.put(
        [
            (3, ("first",), {}),
            (4, ("second",), {}),
        ]
    )
    input_queue.put((5, ("retry",), {}))
    input_queue.put(None)

    admissions = []

    class Role:
        @staticmethod
        def generate_live(*, admit_fn, emit_fn):
            initial, closed = admit_fn(1, True)
            admissions.append([request_id for request_id, _, _ in initial])
            assert closed is False
            for request_id, args, _ in initial:
                emit_fn(request_id, args[0])

            retry, closed = admit_fn(2, True)
            admissions.append([request_id for request_id, _, _ in retry])
            assert closed is True
            for request_id, args, _ in retry:
                emit_fn(request_id, args[0])

    worker = object.__new__(PhysicalWorker)
    worker._roles = {"rollout": Role()}
    worker._resolve_call_inputs = lambda args, kwargs: (args, kwargs)
    worker._pack_call_result = lambda result: result

    worker.live_session_call(
        "rollout", "generate_live", input_queue, output_queue
    )

    assert admissions == [[3, 4], [5]]
    assert [output_queue.get_nowait()[1] for _ in range(4)] == [3, 4, 5, None]


def test_dynamic_scheduler_live_queue_streams_retry_in_same_session(monkeypatch):
    import queue as queue_module
    import threading

    import unirl.trainer.dynamic_trainside as dynamic
    from unirl.models.sensenova_u1.conditions import SensenovaU1ARConditions
    from unirl.types.rollout_resp import RolloutTrack
    from unirl.types.segments import TextSegment

    class LocalQueue:
        def __init__(self):
            self.queue = queue_module.Queue()

        def put(self, value):
            self.queue.put(value)

        def get(self, block=True):
            return self.queue.get(block=block)

        def shutdown(self, force=False):
            del force

    class ThreadRef:
        def __init__(self, target):
            self.error = None

            def run():
                try:
                    target()
                except BaseException as exc:
                    self.error = exc

            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()

        def result(self):
            self.thread.join(timeout=5)
            assert not self.thread.is_alive()
            if self.error is not None:
                raise self.error

    class FakeRay:
        @staticmethod
        def get(refs):
            if isinstance(refs, list):
                return [ref.result() for ref in refs]
            return refs.result()

    session_starts = []
    seen_candidates = []

    class LiveSessionCall:
        @staticmethod
        def remote(role_name, method, input_queue, output_queue):
            session_starts.append((role_name, method))

            def serve():
                while True:
                    item = input_queue.get()
                    if item is None:
                        output_queue.put(("closed", None, None))
                        return
                    request_id, args, kwargs = item
                    assert kwargs == {}
                    job_req = args[0]
                    candidate = job_req.stage_config["trajectory_candidate_ids"][0]
                    seen_candidates.append(candidate)
                    first_attempt = "/attempt-" not in candidate
                    reason = (
                        "max_new_tokens"
                        if candidate.endswith("/a0") and first_attempt
                        else "eos"
                    )
                    condition = SensenovaU1ARConditions.for_sample(
                        query="query", stop_reason=reason
                    )
                    response = SimpleNamespace(
                        tracks={
                            "ar": RolloutTrack(
                                sample_ids=["temporary"],
                                parent_ids=[job_req.sample_ids[0]],
                                conditions=condition.to_dict(),
                                segment=TextSegment.pack(
                                    tokens=[torch.tensor([1])],
                                    log_probs=[torch.tensor([-0.1])],
                                ),
                                decoded=Texts(texts=[candidate]),
                            )
                        }
                    )
                    output_queue.put(("ok", request_id, response))

            return ThreadRef(serve)

    class Worker:
        live_session_call = LiveSessionCall()

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
            del req
            values = torch.arange(track.batch_size, dtype=torch.float32) % 2
            return SimpleNamespace(
                rewards=values,
                component_rewards={"answer_correctness": values.clone()},
            )

    monkeypatch.setattr(dynamic, "ray", FakeRay)
    monkeypatch.setattr(dynamic, "_make_session_queue", LocalQueue)
    req = RolloutReq(
        sample_ids=["prompt-0"],
        group_ids=["group-0"],
        primitives={"text": Texts(texts=["problem-0"])},
        sampling_params={
            "ar": ARSamplingParams(samples_per_prompt=4),
            "diffusion": DiffusionSamplingParams(seed=123),
        },
        stage_config={
            "continuous_batching": True,
            "persistent_worker_session": True,
            "continuous_request_admission": True,
            "continuous_live_admission": True,
            "continuous_rollout_pool_size": 4,
            "continuous_text_batch_size": 2,
            "retry_truncated_trajectories": True,
            "max_attempts_per_candidate": 2,
            "dynamic_prompt_group_refill": False,
            "max_trajectory_multiplier": 2.0,
        },
        metadata=[{"answer": "1"}],
    )

    result = dynamic.run_dynamic_trainside_rollout(
        rollout_handle=Handle(),
        driver_reward=Reward(),
        req=req,
        rollout_id=17,
        target_prompt_count=1,
    )

    assert session_starts == [("rollout", "generate_live")]
    assert len(seen_candidates) == 5
    assert "prompt-0/a0/attempt-1" in seen_candidates
    assert result.metrics["continuous_live_admission"] == 1.0
    assert result.metrics["live_session_count"] == 1.0
    assert result.metrics["streamed_trajectory_results"] == 5.0
    assert result.metrics["live_queue_admissions"] == 5.0
    assert result.metrics["rpc_jobs"] == 1.0
    assert result.metrics["logical_rpc_jobs"] == 5.0
    assert result.metrics["truncated_attempts"] == 1.0
    assert result.resp.tracks["ar"].sample_ids == [
        "prompt-0/a0",
        "prompt-0/a1",
        "prompt-0/a2",
        "prompt-0/a3",
    ]


def test_pipeline_generate_live_streams_one_response_per_request(monkeypatch):
    from unirl.models.sensenova_u1 import pipeline as pipeline_module
    from unirl.models.sensenova_u1.diffusion import SensenovaU1DiffusionParams
    from unirl.models.sensenova_u1.pipeline import SensenovaU1UniPipeline
    from unirl.types.segments import TextSegment

    class Tokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return " ".join(str(token) for token in token_ids)

    class FakeAR:
        last_rollout_metrics = []

        def autoregress(self, conditions, *, live_admit_fn, live_finish_fn, **kwargs):
            del conditions, kwargs
            while True:
                specs, closed = live_admit_fn(2, True)
                for spec in specs:
                    state = SimpleNamespace(
                        tokens=[spec.rewrite_id + 1],
                        logps=[-0.1],
                        images=[],
                        boundaries=[(0, 1)],
                        latent_segments=[],
                        metrics=_metrics(spec.rewrite_id),
                    )
                    state.metrics.stop_reason = "eos"
                    state.metrics.generated_text_tokens = 1
                    live_finish_fn(spec, state)
                if closed:
                    return TextSegment()

    model = torch.nn.Linear(1, 1)
    pipeline = object.__new__(SensenovaU1UniPipeline)
    pipeline.bundle = SimpleNamespace(
        model=model,
        tokenizer=Tokenizer(),
        patch_size=16,
        merge_size=2,
        downsample_ratio=2,
    )
    pipeline.ar = FakeAR()
    pipeline.diffusion = SimpleNamespace(_autocast_dtype=torch.float32)
    monkeypatch.setattr(
        pipeline_module.rl_ops,
        "build_query",
        lambda model, prompt, system_message="": f"{system_message}:{prompt}",
    )

    def request(index):
        return RolloutReq(
            sample_ids=[f"prompt-{index}"],
            group_ids=[f"group-{index}"],
            primitives={"text": Texts(texts=[f"problem-{index}"])},
            sampling_params={
                "ar": ARSamplingParams(samples_per_prompt=1),
                "diffusion": SensenovaU1DiffusionParams(
                    seed=123, max_images=0, num_inference_steps=2
                ),
            },
            stage_config={
                "disable_rollout_metrics": True,
                "continuous_rollout_pool_size": 2,
                "continuous_text_batch_size": 1,
                "trajectory_candidate_ids": [f"prompt-{index}/a{index}"],
                "trajectory_rewrite_indices": [index],
                "trajectory_seed": 100 + index,
            },
            init_noise_group_ids=[f"noise-{index}"],
        )

    queued = [
        (10, (request(0),), {}),
        (11, (request(1),), {}),
    ]

    def admit(capacity, block_when_idle):
        del block_when_idle
        items = queued[:capacity]
        del queued[:capacity]
        return items, not queued

    emitted = []
    pipeline.generate_live(
        admit_fn=admit,
        emit_fn=lambda request_id, response: emitted.append(
            (request_id, response)
        ),
    )

    assert [request_id for request_id, _ in emitted] == [10, 11]
    assert [
        response.tracks["ar"].sample_ids[0] for _, response in emitted
    ] == ["prompt-0/a0", "prompt-1/a1"]
    assert [
        response.tracks["ar"].decoded.texts[0] for _, response in emitted
    ] == ["1", "2"]
    assert [
        next(iter(response.tracks["ar"].conditions.values())).stop_reasons[0]
        for _, response in emitted
    ] == ["eos", "eos"]
