from types import SimpleNamespace

import pytest
import torch

from unirl.algorithms import AlgorithmStepResult
from unirl.train.stack import TrainStepResult
from unirl.train.unified_model_stack import UnifiedModelTrainStack
from unirl.trainer.unified_model import UnifiedModelTrainer
from unirl.utils.wandb_logger import UniRLWandBLogger


class _Backend:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.model.weight)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)
        self.scheduler = None
        self.grad_sync_deferred = False
        self.zero_grad_calls = 0
        self.optimizer_step_calls = 0
        self.rollout_end_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1
        self.optimizer.zero_grad()

    def optimizer_step(self, *, max_grad_norm):
        self.optimizer_step_calls += 1
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
        return float(norm)

    def on_rollout_end(self):
        self.rollout_end_calls += 1


class _Algorithm:
    supports_multi_update = True

    def __init__(self, backend):
        self.backend = backend

    def compute_loss_and_backward(
        self, *, conditions, segment, advantages, training_progress, loss_scale
    ):
        coefficient = float(conditions["coefficient"])
        loss = self.backend.model.weight.sum() * coefficient
        (loss * float(loss_scale)).backward()
        return AlgorithmStepResult(
            loss=float(loss.detach()),
            metrics={"coefficient": coefficient},
            num_steps_or_tokens=1,
            has_backward=True,
        )


def _track(coefficient, batch_size=1):
    return SimpleNamespace(
        batch_size=batch_size,
        conditions={"coefficient": float(coefficient)},
        segment=object(),
        advantages=torch.ones(batch_size),
    )


def _stack(accumulation_steps):
    backend = _Backend()
    stack = UnifiedModelTrainStack(
        fsdp_backend=backend,
        ar_algorithm=_Algorithm(backend),
        micro_batch_size=1,
        max_grad_norm=100.0,
        num_updates_per_batch=1,
        gradient_accumulation_steps=accumulation_steps,
    )
    return stack, backend


def _accumulate(stack, coefficient, *, force=False, batch_size=1):
    track = _track(coefficient, batch_size=batch_size)
    return stack._accumulate_one_rollout(
        {"ar": track},
        {"ar": [(0, batch_size)]},
        training_progress=0.0,
        force_optimizer_step=force,
    )


def test_accumulation_averages_rollout_gradients_and_steps_once():
    stack, backend = _stack(2)

    first = _accumulate(stack, 2.0)
    assert backend.zero_grad_calls == 1
    assert backend.optimizer_step_calls == 0
    assert backend.model.weight.item() == pytest.approx(0.0)
    assert first["ar"].metrics["optimizer_stepped"] == 0.0
    assert first["ar"].metrics["gradient_accumulation_count"] == 1.0
    assert first["ar"].metrics["gradient_accumulation_window_closed"] == 0.0

    second = _accumulate(stack, 4.0)
    assert backend.zero_grad_calls == 1
    assert backend.optimizer_step_calls == 1
    assert backend.rollout_end_calls == 1
    assert backend.model.weight.item() == pytest.approx(-3.0)
    assert second["ar"].metrics["optimizer_stepped"] == 1.0
    assert second["ar"].metrics["gradient_accumulation_count"] == 2.0
    assert second["ar"].metrics["gradient_accumulation_window_closed"] == 1.0


def test_force_optimizer_step_uses_actual_short_window_divisor():
    stack, backend = _stack(4)

    _accumulate(stack, 2.0)
    result = _accumulate(stack, 4.0, force=True)

    assert backend.optimizer_step_calls == 1
    assert backend.model.weight.item() == pytest.approx(-3.0)
    assert result["ar"].metrics["gradient_accumulation_count"] == 2.0
    assert result["ar"].metrics["gradient_accumulation_window_closed"] == 1.0


def test_accumulation_rejects_invalid_configuration():
    backend = _Backend()
    algorithm = _Algorithm(backend)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        UnifiedModelTrainStack(
            fsdp_backend=backend,
            ar_algorithm=algorithm,
            micro_batch_size=1,
            max_grad_norm=1.0,
            gradient_accumulation_steps=0,
        )
    with pytest.raises(ValueError, match="num_updates_per_batch=1"):
        UnifiedModelTrainStack(
            fsdp_backend=backend,
            ar_algorithm=algorithm,
            micro_batch_size=1,
            max_grad_norm=1.0,
            num_updates_per_batch=2,
            gradient_accumulation_steps=2,
        )


def test_accumulation_requires_fixed_track_geometry_within_window():
    stack, _ = _stack(2)
    _accumulate(stack, 1.0, batch_size=1)
    with pytest.raises(ValueError, match="fixed presence and per-rank batch size"):
        _accumulate(stack, 1.0, batch_size=2)


def _result(*, optimizer_stepped=None, has_backward=True, window_closed=None):
    metrics = {}
    if optimizer_stepped is not None:
        metrics["optimizer_stepped"] = float(optimizer_stepped)
    if window_closed is not None:
        metrics["gradient_accumulation_window_closed"] = float(window_closed)
    return TrainStepResult(
        loss=1.0,
        grad_norm=2.0,
        lr=3.0,
        has_backward=has_backward,
        micros=[],
        metrics=metrics,
    )


def test_logger_only_advances_on_real_optimizer_step():
    logger = UniRLWandBLogger.__new__(UniRLWandBLogger)
    logger.enabled = True
    logger._initialized = True
    logger._optimizer_step = 0
    logged = []
    logger.log_step = lambda step, metrics: logged.append((step, metrics))

    logger._log_train({"ar": _result(optimizer_stepped=False)})
    assert logger._optimizer_step == 0
    assert logged == []

    logger._log_train({"ar": _result(optimizer_stepped=True)})
    assert logger._optimizer_step == 1
    assert len(logged) == 1


def test_trainer_accumulation_helpers_preserve_legacy_fallbacks():
    pending = {"ar": _result(optimizer_stepped=False, window_closed=False)}
    closed = {"ar": _result(optimizer_stepped=True, window_closed=True)}
    legacy = {"ar": _result(has_backward=True)}

    assert not UnifiedModelTrainer._optimizer_stepped(pending)
    assert UnifiedModelTrainer._optimizer_stepped(closed)
    assert UnifiedModelTrainer._optimizer_stepped(legacy)
    assert not UnifiedModelTrainer._gradient_accumulation_window_closed(pending)
    assert UnifiedModelTrainer._gradient_accumulation_window_closed(closed)
    assert UnifiedModelTrainer._gradient_accumulation_window_closed(legacy)
