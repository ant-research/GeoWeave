"""SCA reward backend combining independent outcome and process judges."""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from unirl.reward.base import RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse


class SCADualJudgeRewardScorer(RewardBackend):
    """Use Qwen for answer correctness and Gemini for process annotations.

    The two backends score the same batch concurrently. Only the outcome
    backend's binary reward is used as ``answer_correct``; any answer verdict
    emitted by the process backend is deliberately ignored.
    """

    input_kind = "text"
    thread_safe = True

    def __init__(
        self,
        *,
        outcome_backend: RewardBackend,
        process_backend: RewardBackend,
    ) -> None:
        super().__init__(model_name="sca_dual_judge")
        self.outcome_backend = outcome_backend
        self.process_backend = process_backend
        if outcome_backend.preferred_input_kind != "text":
            raise ValueError("SCA outcome backend must consume text")
        if process_backend.preferred_input_kind != "text":
            raise ValueError("SCA process backend must consume text")

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        started = time.perf_counter()
        # These are independent external services, so overlap their batch calls.
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcome_future = executor.submit(self.outcome_backend.compute_rewards, request)
            process_future = executor.submit(self.process_backend.compute_rewards, request)
            outcome = outcome_future.result()
            process = process_future.result()

        batch_size = request.batch_size
        if len(outcome.rewards) != batch_size or len(process.rewards) != batch_size:
            raise ValueError(
                "SCA dual judges returned misaligned batches: "
                f"request={batch_size}, outcome={len(outcome.rewards)}, process={len(process.rewards)}"
            )

        outcome_failed = list((outcome.component_rewards or {}).get("judge_failed", [0.0] * batch_size))
        process_failed = list((process.component_rewards or {}).get("sca_judge_failed", [0.0] * batch_size))
        process_annotations = list(process.process_annotations or [None] * batch_size)
        if len(process_annotations) != batch_size:
            raise ValueError(
                "SCA process annotations are not sample-aligned: "
                f"annotations={len(process_annotations)}, batch={batch_size}"
            )

        annotations = []
        for idx in range(batch_size):
            raw = process_annotations[idx]
            annotation = copy.deepcopy(raw) if isinstance(raw, dict) else {"valid": False}
            process_verdict = annotation.pop("answer_correct", None)
            if process_verdict is not None:
                annotation["process_judge_answer_correct_ignored"] = bool(process_verdict)
            annotation["answer_correct"] = bool(float(outcome.rewards[idx]) > 0.5)
            annotation["outcome_judge_failed"] = bool(float(outcome_failed[idx]) > 0.5)
            annotation["process_judge_failed"] = bool(float(process_failed[idx]) > 0.5)
            annotation["valid"] = bool(
                annotation.get("valid", False)
                and not annotation["outcome_judge_failed"]
                and not annotation["process_judge_failed"]
            )
            annotations.append(annotation)

        combined_failed = [
            max(float(outcome_failed[idx]), float(process_failed[idx]))
            for idx in range(batch_size)
        ]
        components: Dict[str, List[float]] = {
            "answer_correctness": [float(value) for value in outcome.rewards],
            # Existing refill/dump logic consumes this canonical aggregate.
            "judge_failed": combined_failed,
        }
        for name, values in (outcome.component_rewards or {}).items():
            key = str(name)
            if key == "answer_correctness":
                continue
            components[f"outcome_{key}"] = list(values)
        for name, values in (process.component_rewards or {}).items():
            key = str(name)
            if key == "answer_correctness":
                continue
            components[f"process_{key}"] = list(values)

        successes = [
            bool(outcome.successes[idx]) and bool(process.successes[idx])
            for idx in range(batch_size)
        ]
        errors = [
            outcome.errors[idx] or process.errors[idx]
            for idx in range(batch_size)
        ]
        return RewardResponse(
            # Scalar logging and historical consumers use the Qwen outcome.
            rewards=[float(value) for value in outcome.rewards],
            component_rewards=components,
            process_annotations=annotations,
            successes=successes,
            errors=errors,
            compute_time=time.perf_counter() - started,
        )

    def is_available(self) -> bool:
        return self.outcome_backend.is_available() and self.process_backend.is_available()

    def offload(self) -> None:
        self.outcome_backend.offload()
        self.process_backend.offload()

    def onload(self) -> None:
        self.outcome_backend.onload()
        self.process_backend.onload()

    def dispose(self) -> None:
        self.outcome_backend.dispose()
        self.process_backend.dispose()


__all__ = ["SCADualJudgeRewardScorer"]
