from types import SimpleNamespace

import pytest
import torch

from unirl.reward.sca import compute_sca_token_advantages
from unirl.reward.service import RewardService
from unirl.types.primitives import Texts
from unirl.types.reward import RewardResponse
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment
from unirl.utils.token_alignment import decode_with_token_char_spans


def _segment() -> TextSegment:
    return TextSegment.pack(
        tokens=[torch.tensor([10, 11]), torch.tensor([20, 21])],
        log_probs=[torch.zeros(2), torch.zeros(2)],
        decoded_char_starts=[torch.tensor([0, 1]), torch.tensor([0, 1])],
        decoded_char_ends=[torch.tensor([1, 2]), torch.tensor([1, 2])],
    )


def test_decode_with_token_char_spans_strips_marker_and_masks_its_token():
    class Tokenizer:
        pieces = {1: "A", 2: " B", 3: "<|im_end|>"}

        def decode(self, ids, skip_special_tokens=False):
            return "".join(self.pieces[int(token_id)] for token_id in ids)

    text, starts, ends = decode_with_token_char_spans(Tokenizer(), [1, 2, 3])

    assert text == "A B"
    assert starts.tolist() == [0, 1, 3]
    assert ends.tolist() == [1, 3, 3]


def test_sca_normalizes_all_sibling_tokens_together():
    track = RolloutTrack(
        sample_ids=["p0/a0", "p0/a1"],
        parent_ids=["p0", "p0"],
        segment=_segment(),
        process_annotations=[
            {
                "valid": True,
                "answer_correct": True,
                "steps": [{"char_start": 0, "char_end": 2}],
                "step_error_weights": [0.0],
            },
            {
                "valid": True,
                "answer_correct": False,
                "steps": [
                    {"char_start": 0, "char_end": 1},
                    {"char_start": 1, "char_end": 2},
                ],
                "step_error_weights": [0.0, 1.0],
            },
        ],
    )

    scored = compute_sca_token_advantages(track)

    assert scored.segment.process_error_mask.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert scored.segment.raw_token_credit.tolist() == [2.0, 2.0, 0.0, -1.0]
    selected = scored.segment.token_advantages[scored.segment.sca_token_mask > 0.5]
    assert float(selected.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(selected.var(unbiased=False)) == pytest.approx(1.0, rel=1e-5)
    assert scored.advantages.tolist() == [0.0, 0.0]


def test_sca_excludes_invalid_and_status_zero_rows():
    track = RolloutTrack(
        sample_ids=["p0/a0", "p0/a1"],
        parent_ids=["p0", "p0"],
        segment=_segment(),
        status=torch.tensor([1.0, 0.0]),
        process_annotations=[
            {"valid": True, "answer_correct": True, "steps": [], "step_error_weights": []},
            {"valid": True, "answer_correct": False, "steps": [], "step_error_weights": []},
        ],
    )

    scored = compute_sca_token_advantages(track)

    # Only one constant-credit valid row remains, so centered advantages are zero.
    assert scored.segment.sca_token_mask.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert scored.segment.token_advantages.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_reward_service_attaches_process_annotations():
    class Backend:
        preferred_input_kind = "text"

        def get_model_name(self):
            return "fake-sca"

        def compute_rewards(self, request):
            return RewardResponse(
                rewards=[1.0],
                component_rewards={"answer_correctness": [1.0]},
                process_annotations=[{"valid": True, "answer_correct": True}],
                successes=[True],
                errors=[None],
            )

    service = RewardService(backend=Backend(), truncated_reward="keep")
    req = RolloutReq(
        sample_ids=["p0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=["question"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=1)},
    )
    track = RolloutTrack(
        sample_ids=["p0/a0"],
        parent_ids=["p0"],
        segment=TextSegment.pack(tokens=[torch.tensor([1])], log_probs=[torch.zeros(1)]),
        decoded=Texts(texts=["answer"]),
    )

    scored = RewardService.score_and_attach.__wrapped__(service, req=req, track=track)

    assert scored.process_annotations == [{"valid": True, "answer_correct": True}]


def test_sca_packed_signals_follow_track_selection():
    track = RolloutTrack(
        sample_ids=["p0/a0", "p0/a1"],
        parent_ids=["p0", "p0"],
        segment=_segment(),
        process_annotations=[
            {"valid": True, "answer_correct": True, "steps": [], "step_error_weights": []},
            {
                "valid": True,
                "answer_correct": False,
                "steps": [{"char_start": 1, "char_end": 2}],
                "step_error_weights": [1.0],
            },
        ],
    )

    selected = compute_sca_token_advantages(track).select(torch.tensor([1]))

    assert selected.sample_ids == ["p0/a1"]
    assert selected.process_annotations[0]["answer_correct"] is False
    assert selected.segment.raw_token_credit.tolist() == [0.0, -1.0]
    assert selected.segment.process_error_mask.tolist() == [0.0, 1.0]
