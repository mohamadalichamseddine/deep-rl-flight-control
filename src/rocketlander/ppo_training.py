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
from rocketlander.evaluation import run_episode
from rocketlander.ppo_controller import ScaledObservation, scale_observation
from rocketlander.rocket_landing_env import RocketLandingEnv
from rocketlander.rocket_network import RocketPolicy

# Validation starts: used to watch training and pick the best network. Separate from both
# the training starts (small seeds) and the test set (seeds from 1,000,000).
VALIDATION_SEED_OFFSET = 2_000_000


@dataclass(frozen=True)
class TrainConfig:
    level: int
    total_timesteps: int
    n_envs: int
    seed: int
    eval_every: int
    eval_episodes: int
    ppo: dict[str, Any]  # passed to stable_baselines3.PPO


def load_train_config(path: str | Path) -> TrainConfig:
    return load_config(path, TrainConfig)


class LandingRateCallback(BaseCallback):
    """Every ``config.eval_every`` steps, flies the current network on the validation starts
    (without exploration noise), logs the landing rate to TensorBoard and the console, and
    saves the best network so far."""

    def __init__(self, config: TrainConfig, best_model_path: Path) -> None:
        super().__init__()
        self.config = config
        self.best_model_path = best_model_path
        self.best_rate = -1.0
        self.env = RocketLandingEnv(level=config.level)
        self._next_eval = config.eval_every
        self._start = time.monotonic()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.config.eval_every
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        def controller(observation):
            return self.model.predict(scale_observation(observation), deterministic=True)[0]

        results = [
            run_episode(self.env, controller, VALIDATION_SEED_OFFSET + i)
            for i in range(self.config.eval_episodes)
        ]
        rate = sum(r.outcome == "landed" for r in results) / len(results)
        outcomes = Counter(r.outcome for r in results)
        self.logger.record("validation/landing_rate", rate)
        for outcome, count in outcomes.items():
            self.logger.record(f"validation/{outcome}", count / len(results))
        if rate >= self.best_rate:
            self.best_rate = rate
            self.model.save(self.best_model_path)
        minutes = (time.monotonic() - self._start) / 60
        print(
            f"{self.num_timesteps:>11,} steps  {minutes:5.1f} min  "
            f"landing rate {rate:4.0%}  (best {self.best_rate:4.0%})  {dict(outcomes)}",
            flush=True,
        )


def train(config: TrainConfig, run_dir: Path, start_from: Path | None = None) -> PPO:
    """Train a network (or keep training ``start_from``) and save it in ``run_dir``.

    Saves best_model.zip (best validation landing rate), final_model.zip, checkpoints, the
    config, and TensorBoard logs.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    level = config.level
    env = make_vec_env(
        lambda: ScaledObservation(RocketLandingEnv(level=level)),
        n_envs=config.n_envs,
        seed=config.seed,
        vec_env_cls=SubprocVecEnv,
    )
    if start_from is None:
        model = PPO(RocketPolicy, env, seed=config.seed, tensorboard_log=str(run_dir), **config.ppo)
    else:
        model = PPO.load(start_from, env=env, tensorboard_log=str(run_dir), **config.ppo)
    callbacks = [
        LandingRateCallback(config, run_dir / "best_model"),
        CheckpointCallback(
            save_freq=max(config.eval_every * 4 // config.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
        ),
    ]
    try:
        model.learn(
            config.total_timesteps,
            callback=callbacks,
            tb_log_name="tensorboard",
            reset_num_timesteps=start_from is None,
        )
    finally:
        model.save(run_dir / "final_model")
        env.close()
    return model
