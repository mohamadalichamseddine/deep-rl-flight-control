"""Train the network with PPO from Stable-Baselines3 (SB3)."""

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rocketlander.config_loader import load_config
from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.evaluation.evaluation import run_episode
from rocketlander.ppo.network_controller import ScaledObservation, scale_observation
from rocketlander.ppo.rocket_network import SB3PolicyAdapter

# Validation starts: used to watch training and pick the best network. Separate from both
# the training starts (small seeds) and the test set (seeds from 1,000,000).
FIRST_VALIDATION_SEED = 2_000_000


@dataclass(frozen=True)
class TrainConfig:
    level: int
    total_steps: int
    parallel_rockets: int
    seed: int
    validate_every: int
    validation_episodes: int
    ppo: dict[str, Any]  # passed to stable_baselines3.PPO


def load_train_config(path: str | Path) -> TrainConfig:
    return load_config(path, TrainConfig)


class ValidationCallback(BaseCallback):
    """Every ``config.validate_every`` steps, flies the current network on the validation
    starts (without exploration noise), logs the landing rate to TensorBoard and the console,
    and saves the best network so far.

    SB3 calls ``_on_step`` after every training step. ``self.model`` is SB3's name for the PPO
    trainer, which holds the network.
    """

    def __init__(self, config: TrainConfig, best_network_path: Path) -> None:
        super().__init__()
        self.config = config
        self.best_network_path = best_network_path
        self.best_landing_rate = -1.0
        self.env = RocketLandingEnv(level=config.level)
        self._next_validation_step = config.validate_every
        self._start = time.monotonic()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_validation_step:
            self._next_validation_step += self.config.validate_every
            self._validate()
        return True

    def _validate(self) -> None:
        def controller(observation):
            return self.model.predict(scale_observation(observation), deterministic=True)[0]

        results = [
            run_episode(self.env, controller, FIRST_VALIDATION_SEED + i)
            for i in range(self.config.validation_episodes)
        ]
        rate = sum(r.outcome == "landed" for r in results) / len(results)
        outcomes = Counter(r.outcome for r in results)
        self.logger.record("validation/landing_rate", rate)
        for outcome, count in outcomes.items():
            self.logger.record(f"validation/{outcome}", count / len(results))
        if rate >= self.best_landing_rate:
            self.best_landing_rate = rate
            self.model.save(self.best_network_path)
        minutes = (time.monotonic() - self._start) / 60
        print(
            f"{self.num_timesteps:>11,} steps  {minutes:5.1f} min  "
            f"landing rate {rate:4.0%}  (best {self.best_landing_rate:4.0%})  {dict(outcomes)}",
            flush=True,
        )


def train(config: TrainConfig, run_dir: Path, start_from: Path | None = None) -> PPO:
    """Train a network (or keep training ``start_from``) and save it in ``run_dir``.

    Saves best_network.zip (best validation landing rate), final_network.zip, checkpoints, the
    config, and TensorBoard logs.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    level = config.level
    env = make_vec_env(
        lambda: ScaledObservation(RocketLandingEnv(level=level)),
        n_envs=config.parallel_rockets,
        seed=config.seed,
        vec_env_cls=SubprocVecEnv,
    )
    if start_from is None:
        ppo = PPO(
            SB3PolicyAdapter, env, seed=config.seed, tensorboard_log=str(run_dir), **config.ppo
        )
    else:
        ppo = PPO.load(start_from, env=env, tensorboard_log=str(run_dir), **config.ppo)
    callbacks = [
        ValidationCallback(config, run_dir / "best_network"),
        CheckpointCallback(
            save_freq=max(config.validate_every * 4 // config.parallel_rockets, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="network",
        ),
    ]
    try:
        ppo.learn(
            config.total_steps,
            callback=callbacks,
            tb_log_name="tensorboard",
            reset_num_timesteps=start_from is None,
        )
    finally:
        ppo.save(run_dir / "final_network")
        env.close()
    return ppo
