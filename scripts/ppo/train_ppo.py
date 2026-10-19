"""Train the landing network with PPO.

Results go to runs/training/<experiment>/seed<N>/level<N>/: best_network.zip (best validation
landing rate), final_network.zip, checkpoints/, config.json, env_config.yaml, training.log, and
TensorBoard logs. Name the experiment after what it tries, e.g. graded-touchdown-penalty.

Level 0 starts from random weights. Each higher level continues from the best network of the
level below, in the same experiment and seed, so train the levels in order:

    uv run python scripts/ppo/train_ppo.py --experiment my-experiment            # level 0, seed 0
    uv run python scripts/ppo/train_ppo.py --experiment my-experiment --level 1  # continues level 0
    uv run python scripts/ppo/train_ppo.py --experiment my-experiment --seed 1
    uv run python scripts/ppo/train_ppo.py --experiment my-experiment --parallel-rockets 16

Settings: configs/train/ppo.yaml. Watch progress live:  uv run tensorboard --logdir runs/training
"""

import argparse
import dataclasses
from pathlib import Path

from rocketlander.ppo.ppo_training import load_train_config, train

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", required=True, help="what this run tries, as a folder name")
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
    args = parser.parse_args()

    config = load_train_config(args.config)
    overrides = {
        "level": args.level,
        "seed": args.seed,
        "total_steps": args.steps,
        "parallel_rockets": args.parallel_rockets,
    }
    config = dataclasses.replace(config, **{k: v for k, v in overrides.items() if v is not None})

    seed_dir = ROOT / "runs" / "training" / args.experiment / f"seed{config.seed}"
    run_dir = seed_dir / f"level{config.level}"
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists: choose another --experiment, or delete it")
    start_from = None
    if config.level > 0:
        start_from = seed_dir / f"level{config.level - 1}" / "best_network.zip"
        if not start_from.exists():
            raise SystemExit(
                f"{start_from} not found: train level {config.level - 1} first, "
                "with the same experiment and seed"
            )

    print(f"experiment {args.experiment}")
    train(config, run_dir, start_from=start_from)


if __name__ == "__main__":
    main()
