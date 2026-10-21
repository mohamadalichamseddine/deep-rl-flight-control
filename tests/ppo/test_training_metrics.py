import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.logger import Logger

from rocketlander.ppo.training_metrics import (
    COLUMNS,
    TrainingMetrics,
    read_training_rounds,
)
from rocketlander.ppo.training_round import TrainingRound


def _rows(metrics_file: Path) -> list[dict[str, str]]:
    with metrics_file.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_each_round_is_a_row_and_a_validation_redraws_the_curves(tmp_path) -> None:
    metrics_file, curves_file = tmp_path / "metrics.csv", tmp_path / "training_curves.png"
    logger = Logger(str(tmp_path), [TrainingMetrics(metrics_file, curves_file, "Test run", [])])

    # Round 1: training flights only, before any update. SB3 logs NumPy numbers.
    logger.record("rollout/ep_rew_mean", np.float64(-80.5))
    logger.record("rollout/landing_rate", 0.0)
    logger.record("time/fps", 3000)  # not a column
    logger.dump(step=8192)
    assert not curves_file.exists()

    # Round 2: the losses of the first update, and a validation.
    logger.record("rollout/ep_rew_mean", np.float64(-60.0))
    logger.record("rollout/landing_rate", 0.1)
    logger.record("train/policy_gradient_loss", np.float64(-0.01))
    logger.record("train/value_loss", np.float64(250.0))
    logger.record("train/loss", np.float64(124.99))
    logger.record("validation/landing_rate", 0.2)
    logger.record("validation/reward", -40.0)
    logger.dump(step=16384)

    first, second = _rows(metrics_file)
    assert list(first) == COLUMNS
    assert first["step"] == "8192" and first["training_reward"] == "-80.5"
    assert first["actor_loss"] == "" and first["validation_landing_rate"] == ""
    assert second == {
        "step": "16384",
        "training_reward": "-60.0",
        "training_landing_rate": "0.1",
        "validation_reward": "-40.0",
        "validation_landing_rate": "0.2",
        "actor_loss": "-0.01",
        "critic_loss": "250.0",
        "total_loss": "124.99",
    }
    assert curves_file.stat().st_size > 10_000


def test_the_end_of_training_draws_the_final_curves(tmp_path) -> None:
    curves_file = tmp_path / "training_curves.png"
    logger = Logger(
        str(tmp_path), [TrainingMetrics(tmp_path / "metrics.csv", curves_file, "Test run", [])]
    )
    logger.record("rollout/ep_rew_mean", -80.0)
    logger.dump(step=8192)

    logger.close()

    assert curves_file.exists()


def test_a_continued_run_keeps_its_earlier_rounds(tmp_path) -> None:
    metrics_file = tmp_path / "metrics.csv"
    earlier = [
        TrainingRound(8192, -80.5, 0.0, None, None, None, None, None),
        TrainingRound(16384, -60.0, 0.1, -40.0, 0.2, -0.01, 250.0, 124.99),
    ]
    first_part = TrainingMetrics(metrics_file, tmp_path / "curves.png", "Test run", earlier)
    assert read_training_rounds(metrics_file) == earlier

    # Continuing: read the rounds back, start from them, and add the next round.
    logger = Logger(
        str(tmp_path),
        [
            TrainingMetrics(
                metrics_file,
                tmp_path / "curves.png",
                "Test run",
                read_training_rounds(metrics_file),
            )
        ],
    )
    logger.record("rollout/ep_rew_mean", 10.0)
    logger.dump(step=24576)

    rounds = read_training_rounds(metrics_file)
    assert rounds[:2] == earlier and [r.step for r in rounds] == [8192, 16384, 24576]
    assert first_part.training_rounds == earlier
