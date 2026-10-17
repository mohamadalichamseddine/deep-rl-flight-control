"""Evaluate a controller on the fixed test sets: landing rate, touchdown quality, and plots.

Each level's test set is test episodes 0 to N-1, the same starting conditions for every
controller, never used in training. Results go to runs/evaluation/.

    uv run python scripts/evaluate_controller.py                     # PD autopilot, all levels
    uv run python scripts/evaluate_controller.py --level 0 --episodes 100
"""

import argparse
import json
from pathlib import Path

from rocketlander.controllers import CONTROLLERS, make_controller
from rocketlander.evaluation import (
    can_stop_before_the_ground,
    evaluate,
    format_summary,
    seed_for_test_episode,
    summarize,
)
from rocketlander.evaluation_plots import plot_evaluation
from rocketlander.rocket_landing_env import RocketLandingEnv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controller", default="pd", choices=sorted(CONTROLLERS))
    parser.add_argument("--level", type=int, choices=[0, 1, 2], help="default: all levels")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    out = ROOT / "runs" / "evaluation"
    for level in [args.level] if args.level is not None else [0, 1, 2]:
        env = RocketLandingEnv(level=level)
        infeasible = sum(
            not can_stop_before_the_ground(env, env.reset(seed=seed_for_test_episode(i))[0])
            for i in range(args.episodes)
        )
        results = evaluate(make_controller(args.controller, env), env, args.episodes)
        summary = summarize(results)

        name = f"{args.controller}_level{level}"
        print(f"\n=== {args.controller} controller, level {level} ===")
        print(f"starts no controller could save (too fast to stop in time): {infeasible}")
        print(format_summary(summary))
        failed = [r.seed - seed_for_test_episode(0) for r in results if r.outcome != "landed"]
        if failed:
            shown = ", ".join(map(str, failed[:10])) + (" ..." if len(failed) > 10 else "")
            print(f"failed test episodes (watch with watch_controller.py): {shown}")

        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        title = f"{args.controller} controller, level {level}: landed {summary['landing_rate']:.1%}"
        plot = plot_evaluation(results, env.config.landing, title, out / f"{name}.png")
        print(f"saved {out / (name + '.json')} and {plot}")


if __name__ == "__main__":
    main()
