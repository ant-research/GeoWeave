from unirl.utils.wandb_logger import UniRLWandBLogger


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, key, value, global_step):
        self.scalars.append((key, value, global_step))


def test_log_with_step_writes_tensorboard_scalars_without_wandb():
    logger = UniRLWandBLogger(enabled=False)
    logger.enabled = True
    logger._initialized = True
    logger._tensorboard_writer = _Writer()

    logger.log_rollout(3, {"reward_mean": 0.75, "ignored": "text"})

    assert logger._tensorboard_writer.scalars == [("rollout/reward_mean", 0.75, 3)]


def test_nonzero_rank_does_not_create_tensorboard_writer(tmp_path):
    logger = UniRLWandBLogger(
        enabled=False,
        rank=1,
        tensorboard_enabled=True,
        tensorboard_dir=str(tmp_path),
    )

    assert logger.enabled is False
    assert logger._tensorboard_writer is None
