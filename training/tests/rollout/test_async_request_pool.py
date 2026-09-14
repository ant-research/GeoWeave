from __future__ import annotations

from dataclasses import dataclass

from unirl.models.sensenova_u1.diffusion import SensenovaU1DiffusionParams
from unirl.rollout.async_request_pool import (
    SelectedWorkerRolloutPool,
    build_trajectory_requests,
)
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import ARSamplingParams


@dataclass(eq=False)
class _Ref:
    value: object
    ready: bool


class _FakeRay:
    @staticmethod
    def wait(refs, *, num_returns, timeout):
        ready = [ref for ref in refs if ref.ready][:num_returns]
        if not ready and timeout is None and refs:
            refs[0].ready = True
            ready = [refs[0]]
        return ready, [ref for ref in refs if ref not in ready]

    @staticmethod
    def get(ref):
        return ref.value


class _Transport:
    @classmethod
    def localize(cls, shards, pool, device_ids, worker_ids):
        return shards


class _Call:
    def __init__(self, worker):
        self.worker = worker

    def remote(self, role_name, method_name, args, kwargs, grad_mode, call_id):
        assert role_name == "rollout" and method_name == "generate"
        req = args[0]
        self.worker.requests.append(req.sample_ids[0])
        response = RolloutResp(
            tracks={
                "ar": RolloutTrack(
                    sample_ids=[f"{req.sample_ids[0]}/a0"],
                    parent_ids=list(req.sample_ids),
                    decoded=Texts(texts=[req.sample_ids[0]]),
                )
            }
        )
        # Replica 0 is fast; replica 1's first request remains pending until
        # wait_one. The fast replica therefore pulls more work from the queue.
        return _Ref(response, ready=self.worker.index == 0)


class _Worker:
    def __init__(self, index):
        self.index = index
        self.requests = []
        self.call = _Call(self)


class _Pool:
    transport_cls = _Transport


class _Handle:
    role_name = "rollout"
    pool = _Pool()
    workers = [_Worker(0), _Worker(1)]
    device_ids = [0, 1]
    worker_ids = ["dw0", "dw1"]

    @staticmethod
    def _rebind_tree(value, worker, *, worker_local):
        return value


def _request(prompts=2, candidates=2):
    return RolloutReq(
        sample_ids=[f"prompt-{i}" for i in range(prompts)],
        group_ids=[f"group-{i}" for i in range(prompts)],
        primitives={"text": Texts(texts=[f"q{i}" for i in range(prompts)])},
        sampling_params={
            "ar": ARSamplingParams(samples_per_prompt=candidates),
            "diffusion": SensenovaU1DiffusionParams(samples_per_prompt=1),
        },
        metadata=[{"answer": str(i)} for i in range(prompts)],
    )


def test_expands_formal_batch_to_64_independent_trajectories():
    requests = build_trajectory_requests(
        _request(prompts=8, candidates=8), rollout_id=5, policy_version=2, base_seed=7
    )
    assert len(requests) == 64
    assert requests[0].sample_id == "prompt-0/a0"
    assert requests[7].sample_id == "prompt-0/a7"
    assert requests[8].sample_id == "prompt-1/a0"
    assert len({request.request_id for request in requests}) == 64
    assert len({request.seed for request in requests}) == 64
    assert all(request.rollout_req.batch_size == 1 for request in requests)
    assert all(request.rollout_req.sampling_params["ar"].samples_per_prompt == 1 for request in requests)


def test_fast_replica_pulls_more_work_and_complete_groups_are_scored():
    import torch

    from unirl.rollout.async_request_pool import (
        CPURewardDispatcher,
        CurrentBatchRolloutRewardCoordinator,
    )
    from unirl.types.rollout_resp import _track_with_field

    class _RewardService:
        def __init__(self):
            self.parents = []

        def score_and_attach(self, *, req, track):
            self.parents.append(track.parent_ids[0])
            return _track_with_field(
                track,
                "rewards",
                torch.arange(track.batch_size, dtype=torch.float32),
            )

    handle = _Handle()
    requests = build_trajectory_requests(
        _request(prompts=2, candidates=2), rollout_id=1, policy_version=1
    )
    service = _RewardService()
    dispatcher = CPURewardDispatcher(service, max_inflight_groups=2)
    coordinator = CurrentBatchRolloutRewardCoordinator(
        SelectedWorkerRolloutPool(handle, ray_api=_FakeRay),
        dispatcher,
        reward_track="ar",
    )
    try:
        response = coordinator.run(requests)
    finally:
        dispatcher.shutdown()

    assert len(handle.workers[0].requests) > len(handle.workers[1].requests)
    assert sorted(service.parents) == ["prompt-0", "prompt-1"]
    track = response.tracks["ar"]
    assert track.sample_ids == [
        "prompt-0/a0",
        "prompt-0/a1",
        "prompt-1/a0",
        "prompt-1/a1",
    ]
    assert track.parent_ids == ["prompt-0", "prompt-0", "prompt-1", "prompt-1"]
    assert track.rewards.tolist() == [0.0, 1.0, 0.0, 1.0]
