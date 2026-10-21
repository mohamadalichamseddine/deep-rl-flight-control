"""Saves how training is going, from what SB3 logs at the end of every round."""

import csv
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from stable_baselines3.common.logger import KVWriter

from rocketlander.ppo.training_curves import plot_training_curves
from rocketlander.ppo.training_round import TrainingRound

COLUMNS = [field.name for field in fields(TrainingRound)]


class TrainingMetrics(KVWriter):
    """Adds a ``TrainingRound`` to metrics.csv at the end of every round, and redraws the
    training curves after each round with a validation, and when training ends.

    It starts metrics.csv with ``previous_rounds``: none for a new run, and the rounds read back
    with ``read_training_rounds`` when continuing one.

    SB3's logger hands it everything logged in the round, so give it to the trainer's logger:
    ``ppo.set_logger(Logger(folder, [..., training_metrics]))``.
    """

    def __init__(
        self,
        metrics_file: Path,
        curves_file: Path,
        title: str,
        previous_rounds: list[TrainingRound],
    ) -> None:
        self.metrics_file = metrics_file
        self.curves_file = curves_file
        self.title = title
        self.training_rounds = list(previous_rounds)
        with metrics_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(asdict(training_round) for training_round in previous_rounds)

    def write(
        self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int
    ) -> None:
        """Called by SB3's logger at the end of every round, with SB3's names for the values."""
        training_round = TrainingRound(
            step=step,
            training_reward=_number(key_values, "rollout/ep_rew_mean"),
            training_landing_rate=_number(key_values, "rollout/landing_rate"),
            validation_reward=_number(key_values, "validation/reward"),
            validation_landing_rate=_number(key_values, "validation/landing_rate"),
            actor_loss=_number(key_values, "train/policy_gradient_loss"),
            critic_loss=_number(key_values, "train/value_loss"),
            total_loss=_number(key_values, "train/loss"),
        )
        self.training_rounds.append(training_round)
        with self.metrics_file.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(asdict(training_round))
        if training_round.validation_landing_rate is not None:
            self.draw_curves()

    def close(self) -> None:
        """Called through the logger when training ends: draws the final curves."""
        self.draw_curves()

    def draw_curves(self) -> None:
        plot_training_curves(self.training_rounds, self.title, self.curves_file)


def read_training_rounds(metrics_file: Path) -> list[TrainingRound]:
    """The rounds saved in a metrics.csv, to continue a run."""
    with metrics_file.open(encoding="utf-8", newline="") as f:
        return [
            TrainingRound(
                step=int(row["step"]),
                training_reward=_cell(row["training_reward"]),
                training_landing_rate=_cell(row["training_landing_rate"]),
                validation_reward=_cell(row["validation_reward"]),
                validation_landing_rate=_cell(row["validation_landing_rate"]),
                actor_loss=_cell(row["actor_loss"]),
                critic_loss=_cell(row["critic_loss"]),
                total_loss=_cell(row["total_loss"]),
            )
            for row in csv.DictReader(f)
        ]


def _cell(text: str) -> float | None:
    """A metrics.csv cell as a number, or None if it is empty (not measured that round)."""
    return None if text == "" else float(text)


def _number(key_values: dict[str, Any], key: str) -> float | None:
    """A logged value as a plain float (SB3 logs NumPy numbers), or None if it wasn't logged."""
    value = key_values.get(key)
    return None if value is None else float(value)
