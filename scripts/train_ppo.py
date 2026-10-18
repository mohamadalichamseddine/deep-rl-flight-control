"""Train the landing network with PPO.

Results go to runs/ppo/<run name>/: best_model.zip (best validation landing rate),
final_model.zip, checkpoints/, config.json, and TensorBoard logs.

    uv run python scripts/train_ppo.py                  # settings: configs/train/ppo.yaml
    uv run python scripts/train_ppo.py --seed 1 --timesteps 2000000
    uv run python scripts/train_ppo.py --level 1 --start-from runs/ppo/<run>/best_model.zip

Watch progress live:  uv run tensorboard --logdir runs/ppo
"""

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

from rocketlander.ppo_training import load_train_config, train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=ROOT / "configs" / "train" / "ppo.yaml")
    parser.add_argument("--level", type=int, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--start-from", type=Path, help="keep training this saved network")
    parser.add_argument("--name", help="run name (default: level, seed, and time)")
    args = parser.parse_args()

    config = load_train_config(args.config)
    overrides = {"level": args.level, "seed": args.seed, "total_timesteps": args.timesteps}
    config = dataclasses.replace(config, **{k: v for k, v in overrides.items() if v is not None})
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    name = args.name or f"level{config.level}_seed{config.seed}_{stamp}"
    run_dir = ROOT / "runs" / "ppo" / name

    print(f"training {name}: level {config.level}, {config.total_timesteps:,} steps")
    train(config, run_dir, start_from=args.start_from)
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
