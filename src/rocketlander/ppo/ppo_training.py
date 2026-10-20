"""Train the network with PPO from Stable-Baselines3 (SB3)."""

import json
import statistics
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat
from stable_baselines3.common.vec_env import SubprocVecEnv

from rocketlander.environment.env_config import EnvConfig
from rocketlander.environment.outcome import Outcome
from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.evaluation.evaluation import run_episode
from rocketlander.ppo.network_controller import ScaledObservation, scale_observation
from rocketlander.ppo.rocket_network import SB3PolicyAdapter
from rocketlander.ppo.training_metrics import TrainingMetrics
from rocketlander.rocket.rocket_spec import RocketSpec

# Validation starts: used to watch training and pick the best network. Separate from both
# the training starts (small seeds) and the test set (seeds from 1,000,000).
FIRST_VALIDATION_SEED = 2_000_000

# The training landing rate is over the last TRAINING_FLIGHTS finished training flights, like
# SB3's average reward of the training flights (rollout/ep_rew_mean, its stats_window_size).
TRAINING_FLIGHTS = 100


@dataclass(frozen=True)
class TrainConfig:
    level: int
    total_steps: int
    parallel_rockets: int
    seed: int
    validate_every: int
    validation_episodes: int
    ppo: dict[str, Any]  # passed to stable_baselines3.PPO


def build_train_config(path: str | Path) -> TrainConfig:
    """Read a training config file (YAML) and build the settings from it."""
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return TrainConfig(**config)


def log(run_dir: Path, line: str) -> None:
    """Print a line and add it to the run's training.log."""
    print(line, flush=True)
    with (run_dir / "training.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


class TrainingFlightsCallback(BaseCallback):
    """At the end of every round, logs the landing rate of the last ``TRAINING_FLIGHTS``
    training flights (with exploration noise) as rollout/landing_rate, next to SB3's
    rollout/ep_rew_mean: their average reward."""

    def __init__(self) -> None:
        super().__init__()
        self.outcomes: deque[Outcome] = deque(maxlen=TRAINING_FLIGHTS)

    def _on_step(self) -> bool:
        # One info and done per rocket; the info of a flight's last step holds its outcome.
        for info, done in zip(self.locals["infos"], self.locals["dones"], strict=True):
            if done:
                self.outcomes.append(info["outcome"])
        return True

    def _on_rollout_end(self) -> None:
        if self.outcomes:
            landed = sum(outcome == Outcome.LANDED for outcome in self.outcomes)
            self.logger.record("rollout/landing_rate", landed / len(self.outcomes))


class ValidationCallback(BaseCallback):
    """Every ``train_config.validate_every`` steps, flies the current network on the validation
    starts (without exploration noise), logs the landing rate and the average reward to
    TensorBoard, metrics.csv, the console and training.log, and saves the best network so far
    as best_network.zip.

    SB3 calls ``_on_step`` after every training step. ``self.model`` is SB3's name for the PPO
    trainer, which holds the network.
    """

    def __init__(self, train_config: TrainConfig, run_dir: Path, env: RocketLandingEnv) -> None:
        super().__init__()
        self.train_config = train_config
        self.run_dir = run_dir
        self.env = env
        self.best_landing_rate = -1.0
        self._next_validation_step = train_config.validate_every
        self._start = time.monotonic()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_validation_step:
            self._next_validation_step += self.train_config.validate_every
            self._validate()
        return True

    def _validate(self) -> None:
        def controller(observation):
            return self.model.predict(scale_observation(observation), deterministic=True)[0]

        results = [
            run_episode(self.env, controller, FIRST_VALIDATION_SEED + i)
            for i in range(self.train_config.validation_episodes)
        ]
        rate = sum(r.outcome == Outcome.LANDED for r in results) / len(results)
        reward = statistics.mean(r.total_reward for r in results)
        outcomes = Counter(r.outcome for r in results)
        self.logger.record("validation/landing_rate", rate)
        self.logger.record("validation/reward", reward)
        for outcome, count in outcomes.items():
            self.logger.record(f"validation/{outcome}", count / len(results))
        if rate >= self.best_landing_rate:
            self.best_landing_rate = rate
            self.model.save(self.run_dir / "best_network")
        minutes = (time.monotonic() - self._start) / 60
        counts = ", ".join(f"{outcome} {count}" for outcome, count in outcomes.most_common())
        log(
            self.run_dir,
            f"{self.num_timesteps:>11,} steps  {minutes:5.1f} min  "
            f"landing rate {rate:4.0%}  (best {self.best_landing_rate:4.0%})  "
            f"average reward {reward:7.1f}  {counts}",
        )


def train(
    train_config: TrainConfig,
    rocket_spec: RocketSpec,
    env_config: EnvConfig,
    run_dir: Path,
    start_from: Path | None,
) -> PPO:
    """Train a network (or keep training ``start_from``) and save it in ``run_dir``.

    Saves best_network.zip (best validation landing rate), final_network.zip, checkpoints/,
    the settings used (train_config.json, env_config.json, rocket_spec.json), training.log,
    how training went (metrics.csv, one ``TrainingRound`` per round, and training_curves.png),
    and TensorBoard logs (tensorboard/). The step count starts at 0 even when continuing a
    network.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {"train_config": train_config, "env_config": env_config, "rocket_spec": rocket_spec}
    for name, value in settings.items():
        (run_dir / f"{name}.json").write_text(json.dumps(asdict(value), indent=2), encoding="utf-8")
    log(
        run_dir,
        f"training level {train_config.level}, seed {train_config.seed}, "
        f"{train_config.total_steps:,} steps, {train_config.parallel_rockets} rockets in "
        "parallel, "
        + (f"continuing from {start_from}" if start_from else "starting from random weights"),
    )
    level = train_config.level
    env = make_vec_env(
        lambda: ScaledObservation(
            RocketLandingEnv(rocket_spec, env_config, level=level, render_mode=None)
        ),
        n_envs=train_config.parallel_rockets,
        seed=train_config.seed,
        vec_env_cls=SubprocVecEnv,
    )
    if start_from is None:
        ppo = PPO(
            SB3PolicyAdapter,
            env,
            seed=train_config.seed,
            **train_config.ppo,
        )
    else:
        # Without seed=, SB3 would reuse the seed saved in the file.
        ppo = PPO.load(
            start_from,
            env=env,
            seed=train_config.seed,
            **train_config.ppo,
        )
    # Everything SB3 and our callbacks log goes to TensorBoard and to metrics.csv.
    training_metrics = TrainingMetrics(
        run_dir / "metrics.csv",
        run_dir / "training_curves.png",
        f"Training curves: level {level}, seed {train_config.seed}",
    )
    ppo.set_logger(
        Logger(
            str(run_dir),
            [TensorBoardOutputFormat(str(run_dir / "tensorboard")), training_metrics],
        )
    )
    validation_env = RocketLandingEnv(rocket_spec, env_config, level=level, render_mode=None)
    callbacks = [
        TrainingFlightsCallback(),
        ValidationCallback(train_config, run_dir, validation_env),
        CheckpointCallback(
            save_freq=max(train_config.validate_every * 4 // train_config.parallel_rockets, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="network",
        ),
    ]
    try:
        ppo.learn(train_config.total_steps, callback=callbacks)
    finally:
        ppo.save(run_dir / "final_network")
        env.close()
        ppo.logger.close()  # draws the final training curves
    log(run_dir, f"saved to {run_dir}")
    return ppo
