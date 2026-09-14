import time
from types import SimpleNamespace

from unirl.reward.local.sca_genprm import (
    SCAGenPRMRewardScorer,
    SCAGenPRMSpec,
    aggregate_sca_critiques,
    parse_sca_judgment,
    split_sca_steps,
)
from unirl.types.primitives import Texts
from unirl.types.reward import RewardRequest


def test_split_sca_steps_retains_exact_character_spans():
    text = "  first step  \n\n\n second step\nline 2  "
    steps = split_sca_steps(text)

    assert [step["text"] for step in steps] == ["first step", "second step", "line 2"]
    assert [text[step["char_start"] : step["char_end"]] for step in steps] == [
        "first step",
        "second step",
        "line 2",
    ]
    assert [step["step_id"] for step in steps] == [0, 1, 2]


def test_split_sca_steps_ignores_think_tags_and_splits_sentences():
    text = "<think>\n第一步。\n第二步：\n$x=1$\n所以得到 $x=1$。\n</think>\n\n答案：1"
    steps = split_sca_steps(text)

    assert [step["text"] for step in steps] == [
        "第一步。",
        "第二步：\n$x=1$",
        "所以得到 $x=1$。",
        "答案：1",
    ]
    assert all(text[s["char_start"] : s["char_end"]].strip() == s["text"] for s in steps)


def test_parse_sca_judgment_filters_duplicate_and_out_of_range_steps():
    raw = '''```json
    {"answer_correct": false, "incorrect_steps": [
      {"step_id": "1", "error_type": "计算错误", "reason": "bad", "correction": "good"},
      {"step_id": 1, "error_type": "逻辑错误"},
      {"step_id": 99, "error_type": "其他"}
    ]}
    ```'''

    result = parse_sca_judgment(raw, num_steps=3)

    assert result["answer_correct"] is False
    assert [row["step_id"] for row in result["incorrect_steps"]] == [1]
    assert result["invalid_step_count"] == 1


def test_aggregate_sca_critiques_majority_and_average():
    critiques = [
        {"answer_correct": True, "incorrect_steps": [{"step_id": 0}, {"step_id": 2}]},
        {"answer_correct": True, "incorrect_steps": [{"step_id": 0}]},
        {"answer_correct": False, "incorrect_steps": [{"step_id": 1}]},
    ]

    majority = aggregate_sca_critiques(critiques, num_steps=3, voting="majority")
    average = aggregate_sca_critiques(critiques, num_steps=3, voting="average")

    assert majority["answer_correct"] is True
    assert majority["step_error_weights"] == [1.0, 0.0, 0.0]
    assert average["step_error_weights"] == [2 / 3, 1 / 3, 1 / 3]


def test_sca_scorer_calls_matrixllm_and_returns_structured_annotations(monkeypatch):
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": '{"answer_correct":false,"incorrect_steps":[{"step_id":1,"error_type":"计算错误","reason":"2+2不等于5","correction":"2+2=4"}]}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setenv("API_KEY_ENV", "test-key")
    monkeypatch.setattr("unirl.reward.local.sca_genprm.requests.post", fake_post)
    scorer = SCAGenPRMRewardScorer(
        config=SCAGenPRMSpec(qps=0, max_retries=0, max_workers=1),
        base_device="cpu",
    )
    request = RewardRequest(
        primitives={"text": Texts(texts=["question"])},
        generated={"text": Texts(texts=["step zero\n\nstep one"])},
        metadata=[{"answer": "reference"}],
    )

    result = scorer.compute_rewards(request)

    assert result.rewards == [0.0]
    assert result.component_rewards["sca_process_valid"] == [1.0]
    assert result.process_annotations[0]["valid"] is True
    assert result.process_annotations[0]["step_error_weights"] == [0.0, 1.0]
    assert result.process_annotations[0]["steps"][1]["text"] == "step one"
    assert calls[0][0].endswith("/v1/chat/completions")
    assert calls[0][2]["model"] == "gemini-3.5-flash"
    assert calls[0][2]["reasoning_effort"] == "low"
    assert calls[0][2]["response_format"] == {"type": "json_object"}
    assert result.component_rewards["sca_judge_queue_wait_s"][0] >= 0.0
    assert result.component_rewards["sca_judge_api_wall_s"][0] >= 0.0


def test_sca_scorer_parse_failure_is_explicit(monkeypatch):
    def fake_post(*_args, **_kwargs):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "not json"}}]},
        )

    monkeypatch.setenv("API_KEY_ENV", "test-key")
    monkeypatch.setattr("unirl.reward.local.sca_genprm.requests.post", fake_post)
    scorer = SCAGenPRMRewardScorer(
        config=SCAGenPRMSpec(qps=0, max_retries=0, max_workers=1),
        base_device="cpu",
    )
    request = RewardRequest(
        primitives={"text": Texts(texts=["question"])},
        generated={"text": Texts(texts=["one step"])},
        metadata=[{"answer": "reference"}],
    )

    result = scorer.compute_rewards(request)

    assert result.component_rewards["sca_process_valid"] == [0.0]
    assert result.component_rewards["sca_judge_failed"] == [1.0]
    assert result.process_annotations[0]["valid"] is False


def test_sca_scorer_separates_executor_queue_wait_from_execution(monkeypatch):
    def fake_post(*_args, **_kwargs):
        time.sleep(0.03)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"incorrect_steps":[]}'},
                    }
                ],
                "usage": {
                    "completion_tokens": 6,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    monkeypatch.setenv("API_KEY_ENV", "test-key")
    monkeypatch.setattr("unirl.reward.local.sca_genprm.requests.post", fake_post)
    scorer = SCAGenPRMRewardScorer(
        config=SCAGenPRMSpec(qps=0, max_retries=0, max_workers=1),
        base_device="cpu",
    )
    request = RewardRequest(
        primitives={"text": Texts(texts=["q0", "q1"])},
        generated={"text": Texts(texts=["s0", "s1"])},
        metadata=[{"answer": "a0"}, {"answer": "a1"}],
    )

    result = scorer.compute_rewards(request)
    execution = result.component_rewards["sca_judge_latency_s"]
    queue_wait = result.component_rewards["sca_judge_queue_wait_s"]
    total = result.component_rewards["sca_judge_total_latency_s"]

    assert execution[0] >= 0.025
    assert execution[1] >= 0.025
    assert queue_wait[0] < 0.02
    assert queue_wait[1] >= 0.025
    assert total[1] >= execution[1] + 0.025
    assert result.component_rewards["sca_judge_finish_length"] == [0.0, 0.0]
    assert result.component_rewards["sca_judge_completion_tokens"] == [6.0, 6.0]
