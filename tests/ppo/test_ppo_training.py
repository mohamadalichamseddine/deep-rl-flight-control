import csv
import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3.common.logger import Logger

from rocketlander.config_files import TRAIN_CONFIG_FILE
from rocketlander.environment.env_config import EnvConfig, build_env_config
from rocketlander.environment.outcome import Outcome
from rocketlander.ppo.ppo_training import (
    TRAINING_FLIGHTS,
    TrainingFlightsCallback,
    ValidationCallback,
    build_train_config,
    continue_training,
    train,
)
from rocketlander.rocket.rocket_spec import RocketSpec, build_rocket_spec


def test_default_config_loads_with_ppo_settings() -> None:
    train_config = build_train_config(TRAIN_CONFIG_FILE)
    assert train_config.parallel_rockets == 8
    assert isinstance(train_config.ppo["n_steps"], int)
    assert train_config.ppo["policy_kwargs"]["hidden_sizes"] == [64, 64]


def test_training_numbers_are_over_the_last_finished_flights() -> None:
    logger = Logger(None, [])
    callback = TrainingFlightsCallback()
    callback.init_callback(SimpleNamespace(num_timesteps=0, logger=logger))  # the trainer
    # The info of a flight's last step: its outcome, and its total reward from SB3's Monitor.
    landed = {"outcome": Outcome.LANDED, "episode": {"r": 150.0}}
    crashed = {"outcome": Outcome.LOST_CONTROL, "episode": {"r": -90.0}}

    def step(infos: list[dict], dones: list[bool]) -> None:
        callback.update_locals({"infos": infos, "dones": np.array(dones)})
        callback.on_step()

    assert callback.get_landing_rate() is None and callback.get_average_reward() is None
    step([{}, {}], [False, False])  # two rockets, both still flying
    step([landed, crashed], [True, True])
    step([{}, landed], [False, True])
    callback.on_rollout_end()
    assert logger.name_to_value["rollout/landing_rate"] == pytest.approx(2 / 3)
    assert callback.get_average_reward() == pytest.approx((150 - 90 + 150) / 3)

    for _ in range(TRAINING_FLIGHTS):  # the crash drops out of the last 100 flights
        step([landed], [True])
    callback.on_rollout_end()
    assert logger.name_to_value["rollout/landing_rate"] == 1.0
    assert callback.get_average_reward() == 150.0


@pytest.mark.parametrize(
    ("steps_so_far", "first_validation"),
    [(0, 250_000), (5_005_312, 5_250_000)],  # a new run; a run continued after 5M steps
)
def test_validation_starts_at_the_next_multiple_of_validate_every(
    steps_so_far: int, first_validation: int, tmp_path
) -> None:
    callback = ValidationCallback(
        build_train_config(TRAIN_CONFIG_FILE),
        tmp_path,
        None,
        TrainingFlightsCallback(),
        -1.0,
        -math.inf,
    )
    callback.init_callback(SimpleNamespace(num_timesteps=steps_so_far))  # the trainer

    callback.on_training_start({}, {})

    assert callback.next_validation_step == first_validation
    assert "validation" in (tmp_path / "training.log").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("landing_rate", "reward", "better"),
    [
        (1.0, 140.0, True),  # a higher landing rate wins, even with a lower reward
        (0.9, 160.0, True),  # the same landing rate with a higher reward
        (0.9, 150.0, False),  # a tie on both keeps the earlier network
        (0.9, 140.0, False),
        (0.8, 200.0, False),
    ],
)
def test_the_best_network_has_the_best_landing_rate_then_reward(
    landing_rate: float, reward: float, better: bool, tmp_path
) -> None:
    callback = ValidationCallback(
        build_train_config(TRAIN_CONFIG_FILE), tmp_path, None, TrainingFlightsCallback(), 0.9, 150.0
    )
    assert callback.beats_best(landing_rate, reward) == better


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

    ppo = train(train_config, rocket_spec, env_config, level=0, run_dir=tmp_path, start_from=None)

    # The exploration noise started at e^log_std_init; two small updates barely moved it.
    log_std = ppo.policy.log_std.detach().numpy()
    assert log_std == pytest.approx([train_config.log_std_init] * 3, abs=0.01)

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

    # Continuing for 128 more steps: one more round, with the step count going on from 256.
    continue_training(
        build_train_config(tmp_path / "train_config.json"),
        build_rocket_spec(tmp_path / "rocket_spec.json"),
        build_env_config(tmp_path / "env_config.json"),
        level=0,
        run_dir=tmp_path,
        extra_steps=128,
    )

    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as f:
        steps = [int(row["step"]) for row in csv.DictReader(f)]
    assert steps == [128, 256, 384]
    assert build_train_config(tmp_path / "train_config.json").total_steps == 384
    assert "continuing level 0" in (tmp_path / "training.log").read_text(encoding="utf-8")
