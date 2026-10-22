"""Train the network with PPO from Stable-Baselines3 (SB3)."""

import dataclasses
import json
import math
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from rocketlander.environment.env_config import EnvConfig
from rocketlander.environment.outcome import Outcome
from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.evaluation.evaluation import run_episode
from rocketlander.ppo.network_controller import ScaledObservation, scale_observation
from rocketlander.ppo.rocket_network import SB3PolicyAdapter
from rocketlander.ppo.training_metrics import TrainingMetrics, read_training_rounds
from rocketlander.ppo.training_round import TrainingRound
from rocketlander.rocket.rocket_spec import RocketSpec

# Validation starts: used to watch training and pick the best network. Separate from both
# the training starts (small seeds) and the test set (seeds from 1,000,000).
FIRST_VALIDATION_SEED = 2_000_000

# The training numbers are over the last TRAINING_FLIGHTS finished training flights, like SB3's
# average reward of the training flights (rollout/ep_rew_mean, its stats_window_size).
TRAINING_FLIGHTS = 100

# The validation lines of training.log: the training flights (with exploration noise) next to
# the validation flights (without), so the gap between the two shows.
LOG_HEADER = (
    f"{'':11}  {'':9}  {'training':<21}  validation\n"
    f"{'steps':>11}  {'time':>9}  {'landed':>6}  {'avg reward':>13}  "
    f"{'landed':>6}  {'(best)':>6}  {'avg reward':>10}  outcomes"
)


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int
    parallel_rockets: int
    seed: int
    validate_every: int
    validation_episodes: int
    ppo: dict[str, Any]  # passed to stable_baselines3.PPO


def build_train_config(path: str | Path) -> TrainConfig:
    """Read a training config file (YAML, or a run's saved train_config.json: JSON is valid
    YAML) and build the settings from it."""
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return TrainConfig(**config)


def log(run_dir: Path, line: str) -> None:
    """Print a line and add it to the run's training.log."""
    print(line, flush=True)
    with (run_dir / "training.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


class TrainingFlightsCallback(BaseCallback):
    """Keeps the outcome and total reward of the last ``TRAINING_FLIGHTS`` finished training
    flights (with exploration noise), and at the end of every round logs their landing rate as
    rollout/landing_rate, next to SB3's rollout/ep_rew_mean: their average reward."""

    def __init__(self) -> None:
        super().__init__()
        self.outcomes: deque[Outcome] = deque(maxlen=TRAINING_FLIGHTS)
        self.rewards: deque[float] = deque(maxlen=TRAINING_FLIGHTS)

    def get_landing_rate(self) -> float | None:
        """None until a training flight has finished."""
        if not self.outcomes:
            return None
        return sum(outcome == Outcome.LANDED for outcome in self.outcomes) / len(self.outcomes)

    def get_average_reward(self) -> float | None:
        """None until a training flight has finished."""
        return statistics.mean(self.rewards) if self.rewards else None

    def _on_step(self) -> bool:
        # One info and done per rocket. The info of a flight's last step holds its outcome,
        # and its total reward from SB3's Monitor, which wraps every training rocket.
        for info, done in zip(self.locals["infos"], self.locals["dones"], strict=True):
            if done:
                self.outcomes.append(info["outcome"])
                self.rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self) -> None:
        landing_rate = self.get_landing_rate()
        if landing_rate is not None:
            self.logger.record("rollout/landing_rate", landing_rate)


class ValidationCallback(BaseCallback):
    """Every ``train_config.validate_every`` steps, flies the current network on the validation
    starts (without exploration noise), logs the landing rate and the average reward to
    TensorBoard, metrics.csv, the console and training.log (next to those of the training
    flights), and saves the best network so far as best_network.zip.

    The best network has the highest landing rate, and among equal landing rates the highest
    average reward: faster, less fuel, gentler and more centered. ``best_landing_rate`` and
    ``best_reward`` are the ones to beat: -1 and -inf for a new run, the best so far when
    continuing one.

    SB3 calls ``_on_step`` after every training step. ``self.model`` is SB3's name for the PPO
    trainer, which holds the network.
    """

    def __init__(
        self,
        train_config: TrainConfig,
        run_dir: Path,
        env: RocketLandingEnv,
        training_flights: TrainingFlightsCallback,
        best_landing_rate: float,
        best_reward: float,
    ) -> None:
        super().__init__()
        self.train_config = train_config
        self.run_dir = run_dir
        self.env = env
        self.training_flights = training_flights
        self.best_landing_rate = best_landing_rate
        self.best_reward = best_reward
        self.next_validation_step = 0  # set when training starts
        self._start = time.monotonic()

    def beats_best(self, landing_rate: float, reward: float) -> bool:
        """Whether a validation result is better than the best so far: a higher landing rate,
        or the same one with a higher average reward."""
        return (landing_rate, reward) > (self.best_landing_rate, self.best_reward)

    def _on_training_start(self) -> None:
        # The next multiple of validate_every: the first one for a new run, and the one after
        # the step count the run stopped at when continuing it.
        every = self.train_config.validate_every
        self.next_validation_step = (self.num_timesteps // every + 1) * every
        log(self.run_dir, LOG_HEADER)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_validation_step:
            self.next_validation_step += self.train_config.validate_every
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
        if self.beats_best(rate, reward):
            self.best_landing_rate, self.best_reward = rate, reward
            self.model.save(self.run_dir / "best_network")

        minutes = (time.monotonic() - self._start) / 60
        training_rate = self.training_flights.get_landing_rate()
        training_reward = self.training_flights.get_average_reward()
        training_landed = "-" if training_rate is None else f"{training_rate:.0%}"
        training_avg = "-" if training_reward is None else f"{training_reward:+.1f}"
        best = f"({self.best_landing_rate:.0%})"
        counts = ", ".join(f"{outcome} {count}" for outcome, count in outcomes.most_common())
        log(
            self.run_dir,
            f"{self.num_timesteps:>11,}  {minutes:5.1f} min  "
            f"{training_landed:>6}  {training_avg:>13}  "
            f"{rate:>6.0%}  {best:>6}  {reward:>+10.1f}  {counts}",
        )


def train(
    train_config: TrainConfig,
    rocket_spec: RocketSpec,
    env_config: EnvConfig,
    level: int,
    run_dir: Path,
    start_from: Path | None,
) -> PPO:
    """Train a network on the starts of ``level``, from random weights or from ``start_from``
    (the best network of the level below), and save it in ``run_dir``.

    Saves best_network.zip (best validation landing rate), final_network.zip, checkpoints/,
    the settings used (train_config.json, env_config.json, rocket_spec.json), training.log,
    how training went (metrics.csv, one ``TrainingRound`` per round, and training_curves.png),
    and TensorBoard logs (tensorboard/). The step count starts at 0, also from ``start_from``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {"train_config": train_config, "env_config": env_config, "rocket_spec": rocket_spec}
    for name, value in settings.items():
        _save_json(run_dir / f"{name}.json", value)
    log(
        run_dir,
        f"training level {level}, seed {train_config.seed}, "
        f"{train_config.total_steps:,} steps, {train_config.parallel_rockets} rockets in "
        "parallel, "
        + (f"continuing from {start_from}" if start_from else "starting from random weights"),
    )
    env = _training_env(train_config, rocket_spec, env_config, level)
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
    _learn(
        ppo,
        env,
        train_config,
        rocket_spec,
        env_config,
        level,
        run_dir,
        previous_rounds=[],
        steps=train_config.total_steps,
        reset_num_timesteps=True,
    )
    return ppo


def continue_training(
    train_config: TrainConfig,
    rocket_spec: RocketSpec,
    env_config: EnvConfig,
    level: int,
    run_dir: Path,
    extra_steps: int,
) -> PPO:
    """Keep training the run in ``run_dir`` from its final_network.zip for ``extra_steps`` more
    steps. Pass the run's own saved settings, so both parts train on the same task.

    The step count goes on from where the run stopped. metrics.csv, training.log and the
    curves continue, best_network.zip is only replaced by a validation at least as good as
    the run's best so far, and train_config.json gets the new total of steps.
    """
    previous_rounds = read_training_rounds(run_dir / "metrics.csv")
    env = _training_env(train_config, rocket_spec, env_config, level)
    ppo = PPO.load(
        run_dir / "final_network.zip",
        env=env,
        seed=train_config.seed,
        **train_config.ppo,
    )
    train_config = dataclasses.replace(train_config, total_steps=ppo.num_timesteps + extra_steps)
    _save_json(run_dir / "train_config.json", train_config)
    log(
        run_dir,
        f"continuing level {level}, seed {train_config.seed}, from step {ppo.num_timesteps:,} "
        f"for {extra_steps:,} more steps, from {run_dir / 'final_network.zip'}",
    )
    _learn(
        ppo,
        env,
        train_config,
        rocket_spec,
        env_config,
        level,
        run_dir,
        previous_rounds=previous_rounds,
        steps=extra_steps,
        reset_num_timesteps=False,
    )
    return ppo


def _learn(
    ppo: PPO,
    env: VecEnv,
    train_config: TrainConfig,
    rocket_spec: RocketSpec,
    env_config: EnvConfig,
    level: int,
    run_dir: Path,
    previous_rounds: list[TrainingRound],
    steps: int,
    reset_num_timesteps: bool,
) -> None:
    """Train ``ppo`` for ``steps`` steps, logging and saving into ``run_dir``.

    ``previous_rounds`` are the rounds of the run so far: none for a new run.
    ``reset_num_timesteps`` is SB3's: True starts the step count (and SB3's list of recent
    training flights) again at 0; False goes on from the count saved in the network.
    """
    # Everything SB3 and our callbacks log goes to TensorBoard and to metrics.csv.
    training_metrics = TrainingMetrics(
        run_dir / "metrics.csv",
        run_dir / "training_curves.png",
        f"Training curves: level {level}, seed {train_config.seed}",
        previous_rounds,
    )
    ppo.set_logger(
        Logger(
            str(run_dir),
            [TensorBoardOutputFormat(str(run_dir / "tensorboard")), training_metrics],
        )
    )
    # The best validation so far: the highest landing rate, then the highest average reward.
    validated = [r for r in previous_rounds if r.validation_landing_rate is not None]
    best_round = max(
        validated,
        key=lambda r: (r.validation_landing_rate, r.validation_reward),
        default=None,
    )
    best_landing_rate = -1.0 if best_round is None else best_round.validation_landing_rate
    best_reward = -math.inf if best_round is None else best_round.validation_reward
    training_flights = TrainingFlightsCallback()
    validation_env = RocketLandingEnv(rocket_spec, env_config, level=level, render_mode=None)
    callbacks = [
        training_flights,
        ValidationCallback(
            train_config,
            run_dir,
            validation_env,
            training_flights,
            best_landing_rate,
            best_reward,
        ),
        CheckpointCallback(
            save_freq=max(train_config.validate_every * 4 // train_config.parallel_rockets, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="network",
        ),
    ]
    try:
        ppo.learn(steps, callback=callbacks, reset_num_timesteps=reset_num_timesteps)
    finally:
        ppo.save(run_dir / "final_network")
        env.close()
        ppo.logger.close()  # draws the final training curves
    log(run_dir, f"saved to {run_dir}")


def _training_env(
    train_config: TrainConfig, rocket_spec: RocketSpec, env_config: EnvConfig, level: int
) -> VecEnv:
    """The training rockets: one process each, flying in parallel, with scaled observations."""
    return make_vec_env(
        lambda: ScaledObservation(
            RocketLandingEnv(rocket_spec, env_config, level=level, render_mode=None)
        ),
        n_envs=train_config.parallel_rockets,
        seed=train_config.seed,
        vec_env_cls=SubprocVecEnv,
    )


def _save_json(path: Path, settings: Any) -> None:
    """Save a settings data class as JSON, which the build_* functions can read back."""
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
