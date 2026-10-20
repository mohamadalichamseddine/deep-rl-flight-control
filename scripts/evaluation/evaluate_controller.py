"""Evaluate a controller on the fixed test sets: landing rate, touchdown quality, and plots.

Each level's test set is test episodes 0 to N-1, the same starting conditions for every
controller, never used in training. Results go to runs/test/: the PD autopilot's to
runs/test/pd/tested_on_level<N>.json and .png; a network's to the same place as the network,
with runs/training replaced by runs/test, e.g.
runs/test/<experiment>/seed0/level0/best_network_tested_on_level1.json.

    uv run python scripts/evaluation/evaluate_controller.py   # PD autopilot, all levels
    uv run python scripts/evaluation/evaluate_controller.py --level 0 --episodes 100
    uv run python scripts/evaluation/evaluate_controller.py \\
        --network runs/training/<experiment>/seed0/level0/best_network.zip --level 0
"""

import argparse
import json
from pathlib import Path

from rocketlander.config_files import ENV_CONFIG_FILE, PD_GAINS_FILE, ROCKET_SPEC_FILE
from rocketlander.environment.env_config import build_env_config
from rocketlander.environment.observation import observation_to_state
from rocketlander.environment.outcome import Outcome
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
from rocketlander.pd.pd_controller import build_pd_gains
from rocketlander.rocket.rocket_spec import build_rocket_spec

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "runs" / "training"
TEST = ROOT / "runs" / "test"


def results_location(controller: str, network: str | None) -> tuple[Path, str]:
    """The folder for a controller's test results, and the start of their file names."""
    if network is None:
        return TEST / controller, ""
    path = Path(network).resolve()
    if path.is_relative_to(TRAINING):
        return TEST / path.parent.relative_to(TRAINING), f"{path.stem}_"
    return TEST / "other", f"{path.stem}_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controller", default="pd", choices=CONTROLLERS)
    parser.add_argument("--network", help="a trained network (.zip); implies --controller network")
    parser.add_argument("--level", type=int, choices=[0, 1, 2], help="default: all levels")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    if args.network:
        args.controller = "network"
    out, prefix = results_location(args.controller, args.network)
    label = (out.relative_to(TEST) / prefix.rstrip("_")).as_posix()  # e.g. pd

    rocket_spec = build_rocket_spec(ROCKET_SPEC_FILE)
    env_config = build_env_config(ENV_CONFIG_FILE)
    pd_gains = build_pd_gains(PD_GAINS_FILE)
    for level in [args.level] if args.level is not None else [0, 1, 2]:
        env = RocketLandingEnv(rocket_spec, env_config, level=level, render_mode=None)
        starts = [
            observation_to_state(env.reset(seed=seed_for_test_episode(i))[0])
            for i in range(args.episodes)
        ]
        infeasible = sum(not can_stop_before_the_ground(env.rocket_simulator, s) for s in starts)
        controller = make_controller(args.controller, env.rocket_simulator, pd_gains, args.network)
        results = run_test_set(controller, env, args.episodes)
        summary = summarize(results)

        name = f"{prefix}tested_on_level{level}"
        print(f"\n=== {label}, tested on level {level} ===")
        print(f"starts no controller could save (too fast to stop in time): {infeasible}")
        print(format_summary(summary))
        failed = [r.seed - seed_for_test_episode(0) for r in results if r.outcome != Outcome.LANDED]
        if failed:
            shown = ", ".join(map(str, failed[:10])) + (" ..." if len(failed) > 10 else "")
            print(
                f"failed test episodes (watch with watch_controller.py --test-episode N): {shown}"
            )

        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        title = f"{label}, tested on level {level}: landed {summary['landing_rate']:.1%}"
        plot = plot_evaluation(results, env.env_config.landing, title, out / f"{name}.png")
        print(f"saved {out / (name + '.json')} and {plot}")


if __name__ == "__main__":
    main()
