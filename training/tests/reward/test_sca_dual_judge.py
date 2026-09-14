import time

from unirl.reward.base import RewardBackend
from unirl.reward.local.sca_dual_judge import SCADualJudgeRewardScorer
from unirl.types.primitives import Texts
from unirl.types.reward import RewardRequest, RewardResponse


class _StubBackend(RewardBackend):
    input_kind = "text"

    def __init__(self, response, delay=0.0):
        super().__init__()
        self.response = response
        self.delay = delay

    def compute_rewards(self, request):
        time.sleep(self.delay)
        return self.response

    def is_available(self):
        return True


def test_dual_judge_uses_qwen_outcome_and_gemini_process_annotation():
    outcome = _StubBackend(
        RewardResponse(
            rewards=[1.0, 0.0],
            component_rewards={"answer_correctness": [1.0, 0.0], "judge_failed": [0.0, 1.0]},
            successes=[True, True],
            errors=[None, None],
        )
    )
    process = _StubBackend(
        RewardResponse(
            rewards=[0.0, 1.0],  # Gemini outcome verdicts must be ignored.
            component_rewards={"answer_correctness": [0.0, 1.0], "sca_judge_failed": [0.0, 0.0]},
            process_annotations=[
                {"valid": True, "answer_correct": False, "step_error_weights": [1.0]},
                {"valid": True, "answer_correct": True, "step_error_weights": [0.0]},
            ],
            successes=[True, True],
            errors=[None, None],
        )
    )
    scorer = SCADualJudgeRewardScorer(outcome_backend=outcome, process_backend=process)
    request = RewardRequest(
        primitives={"text": Texts(texts=["p0", "p1"])},
        generated={"text": Texts(texts=["r0", "r1"])},
    )

    result = scorer.compute_rewards(request)

    assert result.rewards == [1.0, 0.0]
    assert result.component_rewards["answer_correctness"] == [1.0, 0.0]
    assert result.component_rewards["judge_failed"] == [0.0, 1.0]
    assert result.process_annotations[0]["answer_correct"] is True
    assert result.process_annotations[0]["process_judge_answer_correct_ignored"] is False
    assert result.process_annotations[0]["valid"] is True
    assert result.process_annotations[1]["answer_correct"] is False
    assert result.process_annotations[1]["valid"] is False
