import sys
from types import ModuleType, SimpleNamespace

from unirl.reward.local.dashscope_llm_judge import (
    DashScopeLLMJudgeRewardScorer,
    DashScopeLLMJudgeSpec,
)
from unirl.types.primitives import Texts
from unirl.types.reward import RewardRequest


def _response(content=None, *, status_code=200, message=""):
    output = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return SimpleNamespace(status_code=status_code, message=message, output=output)


def _install_fake_dashscope(monkeypatch, call):
    module = ModuleType("dashscope")
    module.api_key = None
    module.Generation = SimpleNamespace(call=call)
    monkeypatch.setitem(sys.modules, "dashscope", module)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")


def _request():
    return RewardRequest(
        primitives={"text": Texts(texts=["question 1", "question 2"])},
        generated={"text": Texts(texts=["candidate 1", "candidate 2"])},
        metadata=[{"answer": "gold 1"}, {"answer": "gold 2"}],
    )


def test_llm_judge_returns_binary_rewards_and_components(monkeypatch):
    outputs = iter(
        [
            _response('{"correct": true, "reason": "equivalent"}'),
            _response('```json\n{"correct": false, "reason": "different"}\n```'),
        ]
    )
    _install_fake_dashscope(monkeypatch, lambda **_kwargs: next(outputs))
    scorer = DashScopeLLMJudgeRewardScorer(
        config=DashScopeLLMJudgeSpec(max_workers=1, max_retries=0),
        base_device="cpu",
    )

    result = scorer.compute_rewards(_request())

    assert result.rewards == [1.0, 0.0]
    assert result.component_rewards["answer_correctness"] == [1.0, 0.0]
    assert result.component_rewards["judge_failed"] == [0.0, 0.0]
    assert result.successes == [True, True]


def test_llm_judge_api_failure_is_nonfatal_zero_reward(monkeypatch):
    calls = []

    def _fail(**_kwargs):
        calls.append(1)
        return _response(status_code=500, message="temporary failure")

    _install_fake_dashscope(monkeypatch, _fail)
    scorer = DashScopeLLMJudgeRewardScorer(
        config=DashScopeLLMJudgeSpec(
            max_workers=1,
            max_retries=2,
            retry_backoff_s=0,
        ),
        base_device="cpu",
    )

    request = _request()
    request.generated = {"text": Texts(texts=["candidate 1"])}
    request.primitives = {"text": Texts(texts=["question 1"])}
    request.metadata = [{"answer": "gold 1"}]
    result = scorer.compute_rewards(request)

    assert len(calls) == 3
    assert result.rewards == [0.0]
    assert result.component_rewards["judge_failed"] == [1.0]
    assert result.component_rewards["judge_retries"] == [2.0]
    assert result.successes == [True]


def test_llm_judge_missing_answer_is_marked_failed(monkeypatch):
    _install_fake_dashscope(monkeypatch, lambda **_kwargs: _response('{"correct": true}'))
    scorer = DashScopeLLMJudgeRewardScorer(
        config=DashScopeLLMJudgeSpec(max_workers=1, max_retries=0),
        base_device="cpu",
    )
    request = RewardRequest(
        primitives={"text": Texts(texts=["question"])},
        generated={"text": Texts(texts=["candidate"])},
        metadata=[{}],
    )

    result = scorer.compute_rewards(request)

    assert result.rewards == [0.0]
    assert result.component_rewards["judge_failed"] == [1.0]
    assert result.successes == [True]


def test_llm_judge_spec_coerces_omegaconf_env_strings():
    spec = DashScopeLLMJudgeSpec(
        temperature="0.1",
        top_p="0.8",
        max_tokens="256",
        enable_thinking="false",
        timeout="12.5",
        max_retries="3",
        retry_backoff_s="0.25",
        max_workers="4",
    )

    assert spec.temperature == 0.1
    assert spec.top_p == 0.8
    assert spec.max_tokens == 256
    assert spec.enable_thinking is False
    assert spec.timeout == 12.5
    assert spec.max_retries == 3
    assert spec.retry_backoff_s == 0.25
    assert spec.max_workers == 4
