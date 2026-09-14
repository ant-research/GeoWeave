import pytest
import torch

from unirl.reward.local.mathverify import MathVerifyRewardScorer, MathVerifySpec
from unirl.reward.service import RewardService
from unirl.types.primitives import Texts
from unirl.types.reward import RewardRequest
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment


def test_mathverify_scores_generated_text_not_input_prompt():
    scorer = MathVerifyRewardScorer(config=MathVerifySpec(), base_device="cpu")
    request = RewardRequest(
        primitives={"text": Texts(texts=[r"prompt containing \\boxed{999}", "prompt"])},
        generated={"text": Texts(texts=[r"Therefore \\boxed{2}", r"Therefore \\boxed{3}"])},
        metadata=[{"answer": "2"}, {"answer": "4"}],
    )

    assert scorer.compute_rewards(request).rewards == [1.0, 0.0]


def test_truncated_correct_response_is_zeroed_by_default():
    scorer = MathVerifyRewardScorer(config=MathVerifySpec(), base_device="cpu")
    service = RewardService(backend=scorer)
    req = RolloutReq(
        sample_ids=["p0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=["question"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=1, max_new_tokens=1)},
        metadata=[{"answer": "2"}],
    )
    track = RolloutTrack(
        sample_ids=["p0/a0"],
        parent_ids=["p0"],
        segment=TextSegment.pack(tokens=[torch.tensor([1])], log_probs=[torch.zeros(1)]),
        decoded=Texts(texts=[r"\\boxed{2}"]),
    )

    scored = RewardService.score_and_attach.__wrapped__(service, req=req, track=track)

    assert scored.rewards.tolist() == [0.0]


def test_soft_overlong_reward_matches_dapo_linear_buffer():
    scorer = MathVerifyRewardScorer(config=MathVerifySpec(), base_device="cpu")
    service = RewardService(
        backend=scorer,
        truncated_reward="soft",
        overlong_buffer_len=512,
        overlong_penalty_factor=1.0,
    )
    lengths = [2048, 2304, 2560]
    req = RolloutReq(
        sample_ids=["p0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=["question"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=3, max_new_tokens=2560)},
        metadata=[{"answer": "2"}],
    )
    track = RolloutTrack(
        sample_ids=[f"p0/a{i}" for i in range(3)],
        parent_ids=["p0"] * 3,
        segment=TextSegment.pack(
            tokens=[torch.ones(length, dtype=torch.long) for length in lengths],
            log_probs=[torch.zeros(length) for length in lengths],
        ),
        decoded=Texts(texts=[r"\\boxed{2}", r"\\boxed{2}", r"\\boxed{3}"]),
    )

    scored = RewardService.score_and_attach.__wrapped__(service, req=req, track=track)

    assert scored.rewards.tolist() == [1.0, 0.5, -1.0]
    assert scored.component_rewards["overlong_reward"].tolist() == [0.0, -0.5, -1.0]
    assert scored.component_rewards["overlong"].tolist() == [0.0, 1.0, 1.0]
    assert scored.component_rewards["response_token_length"].tolist() == lengths


def test_soft_overlong_validates_buffer_and_factor():
    scorer = MathVerifyRewardScorer(config=MathVerifySpec(), base_device="cpu")
    with pytest.raises(ValueError, match="overlong_buffer_len must be positive"):
        RewardService(backend=scorer, truncated_reward="soft", overlong_buffer_len=0)
    with pytest.raises(ValueError, match="overlong_penalty_factor must be non-negative"):
        RewardService(backend=scorer, truncated_reward="soft", overlong_penalty_factor=-1.0)
