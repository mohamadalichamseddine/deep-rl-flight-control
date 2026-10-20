import csv
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3.common.logger import Logger

from rocketlander.config_files import TRAIN_CONFIG_FILE
from rocketlander.environment.env_config import EnvConfig
from rocketlander.environment.outcome import Outcome
from rocketlander.ppo.ppo_training import (
    TRAINING_FLIGHTS,
    TrainingFlightsCallback,
    build_train_config,
    train,
)
from rocketlander.rocket.rocket_spec import RocketSpec


def test_default_config_loads_with_ppo_settings() -> None:
    train_config = build_train_config(TRAIN_CONFIG_FILE)
    assert train_config.parallel_rockets == 8
    assert isinstance(train_config.ppo["n_steps"], int)
    assert train_config.ppo["policy_kwargs"]["hidden_sizes"] == [64, 64]


def test_training_landing_rate_counts_the_last_finished_flights() -> None:
    logger = Logger(None, [])
    callback = TrainingFlightsCallback()
    callback.init_callback(SimpleNamespace(num_timesteps=0, logger=logger))  # the trainer
    landed, crashed = {"outcome": Outcome.LANDED}, {"outcome": Outcome.LOST_CONTROL}

    def step(infos: list[dict], dones: list[bool]) -> None:
        callback.update_locals({"infos": infos, "dones": np.array(dones)})
        callback.on_step()

    step([{}, {}], [False, False])  # two rockets, both still flying
    step([landed, crashed], [True, True])
    step([{}, landed], [False, True])
    callback.on_rollout_end()
    assert logger.name_to_value["rollout/landing_rate"] == pytest.approx(2 / 3)

    for _ in range(TRAINING_FLIGHTS):  # the crash drops out of the last 100 flights
        step([landed], [True])
    callback.on_rollout_end()
    assert logger.name_to_value["rollout/landing_rate"] == 1.0


def test_a_tiny_training_run_saves_everything(
    rocket_spec: RocketSpec, env_config: EnvConfig, tmp_path
) -> None:
    train_config = dataclasses.replace(
        build_train_config(TRAIN_CONFIG_FILE),
        total_steps=256,
        parallel_rockets=2,
        validate_every=128,
        validation_episodes=1,
        ppo={
            "n_steps": 64,
            "batch_size": 64,
            "n_epochs": 1,
            "policy_kwargs": {"hidden_sizes": [64, 64]},
        },
    )

    train(train_config, rocket_spec, env_config, level=0, run_dir=tmp_path, start_from=None)

    for name in (
        "best_network.zip",
        "final_network.zip",
        "train_config.json",
        "env_config.json",
        "rocket_spec.json",
        "training.log",
        "metrics.csv",
        "training_curves.png",
    ):
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "tensorboard").is_dir()

    # 256 steps = 2 rounds of 2 rockets x 64 steps, each ending with a validation; the
    # second has the losses of the first update.
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[1]["actor_loss"] != "" and rows[1]["validation_landing_rate"] != ""
