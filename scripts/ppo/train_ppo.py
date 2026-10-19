"""Train the landing network with PPO.

Results go to runs/ppo/<run name>/: best_network.zip (best validation landing rate),
final_network.zip, checkpoints/, config.json, and TensorBoard logs.

    uv run python scripts/ppo/train_ppo.py                  # settings: configs/train/ppo.yaml
    uv run python scripts/ppo/train_ppo.py --seed 1 --steps 2000000
    uv run python scripts/ppo/train_ppo.py --parallel-rockets 16   # one per CPU core
    uv run python scripts/ppo/train_ppo.py --level 1 --start-from runs/ppo/<run>/best_network.zip

Watch progress live:  uv run tensorboard --logdir runs/ppo
"""

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

from rocketlander.ppo.ppo_training import load_train_config, train

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=ROOT / "configs" / "train" / "ppo.yaml")
    parser.add_argument("--level", type=int, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--parallel-rockets",
        type=int,
        help="rockets flying at once, ideally one per CPU core; each iteration then collects "
        "parallel rockets x n_steps steps",
    )
    parser.add_argument("--start-from", type=Path, help="keep training this saved network")
    parser.add_argument("--name", help="run name (default: level, seed, and time)")
    args = parser.parse_args()

    config = load_train_config(args.config)
    overrides = {
        "level": args.level,
        "seed": args.seed,
        "total_steps": args.steps,
        "parallel_rockets": args.parallel_rockets,
    }
    config = dataclasses.replace(config, **{k: v for k, v in overrides.items() if v is not None})
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    name = args.name or f"level{config.level}_seed{config.seed}_{stamp}"
    run_dir = ROOT / "runs" / "ppo" / name

    print(
        f"training {name}: level {config.level}, {config.total_steps:,} steps, "
        f"{config.parallel_rockets} rockets in parallel"
    )
    train(config, run_dir, start_from=args.start_from)
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
