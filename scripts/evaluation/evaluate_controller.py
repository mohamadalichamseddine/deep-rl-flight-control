"""Evaluate a controller on the fixed test sets: landing rate, touchdown quality, and plots.

Each level's test set is test episodes 0 to N-1, the same starting conditions for every
controller, never used in training. Results go to runs/evaluation/.

    uv run python scripts/evaluation/evaluate_controller.py   # PD autopilot, all levels
    uv run python scripts/evaluation/evaluate_controller.py --level 0 --episodes 100
    uv run python scripts/evaluation/evaluate_controller.py --network runs/ppo/<run>/<file>.zip
"""

import argparse
import json
from pathlib import Path

from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.evaluation.controllers import CONTROLLERS, make_controller
from rocketlander.evaluation.evaluation import (
    can_stop_before_the_ground,
    format_summary,
    run_test_set,
    seed_for_test_episode,
    summarize,
)
from rocketlander.evaluation.evaluation_plots import plot_evaluation

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controller", default="pd", choices=CONTROLLERS)
    parser.add_argument("--network", help="a trained network (.zip); implies --controller network")
    parser.add_argument("--level", type=int, choices=[0, 1, 2], help="default: all levels")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    if args.network:
        args.controller = "network"
    label = args.controller
    if args.network:  # e.g. network_level0_seed0_best_network
        label = f"network_{Path(args.network).parent.name}_{Path(args.network).stem}"

    out = ROOT / "runs" / "evaluation"
    for level in [args.level] if args.level is not None else [0, 1, 2]:
        env = RocketLandingEnv(level=level)
        infeasible = sum(
            not can_stop_before_the_ground(env, env.reset(seed=seed_for_test_episode(i))[0])
            for i in range(args.episodes)
        )
        controller = make_controller(args.controller, env, args.network)
        results = run_test_set(controller, env, args.episodes)
        summary = summarize(results)

        name = f"{label}_level{level}"
        print(f"\n=== {label}, level {level} ===")
        print(f"starts no controller could save (too fast to stop in time): {infeasible}")
        print(format_summary(summary))
        failed = [r.seed - seed_for_test_episode(0) for r in results if r.outcome != "landed"]
        if failed:
            shown = ", ".join(map(str, failed[:10])) + (" ..." if len(failed) > 10 else "")
            print(f"failed test episodes (watch with watch_controller.py): {shown}")

        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        title = f"{label}, level {level}: landed {summary['landing_rate']:.1%}"
        plot = plot_evaluation(results, env.config.landing, title, out / f"{name}.png")
        print(f"saved {out / (name + '.json')} and {plot}")


if __name__ == "__main__":
    main()
