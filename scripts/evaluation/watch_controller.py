"""Watch a controller fly in the live window, from new random starts or from the test set.

With --test-episode, it re-flies test episodes exactly as the evaluation flew them: same seed,
same controller.

    uv run python scripts/evaluation/watch_controller.py          # PD, level 0, random starts
    uv run python scripts/evaluation/watch_controller.py --level 2 --test-episode 137
    uv run python scripts/evaluation/watch_controller.py --flights 3 --record runs/pd_demo.mp4
    uv run python scripts/evaluation/watch_controller.py --level 0 \\
        --network runs/training/<experiment>/seed0/level0/best_network.zip
"""

import argparse

from rocketlander.config_files import ENV_CONFIG_FILE, PD_GAINS_FILE, ROCKET_SPEC_FILE
from rocketlander.environment.env_config import build_env_config
from rocketlander.environment.landing_judge import describe_failures
from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.evaluation.controllers import CONTROLLERS, make_controller
from rocketlander.evaluation.evaluation import seed_for_test_episode
from rocketlander.pd.pd_controller import build_pd_gains
from rocketlander.rocket.rocket_spec import build_rocket_spec
from rocketlander.visualization.video_writer import VideoWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controller", default="pd", choices=CONTROLLERS)
    parser.add_argument("--network", help="a trained network (.zip); implies --controller network")
    parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument(
        "--test-episode",
        type=int,
        help="fly the test set from this test episode on (default: new random starts)",
    )
    parser.add_argument("--flights", type=int, help="stop after this many flights")
    parser.add_argument("--record", help="save the flights to this MP4 file")
    args = parser.parse_args()

    rocket_spec = build_rocket_spec(ROCKET_SPEC_FILE)
    env_config = build_env_config(ENV_CONFIG_FILE)
    env = RocketLandingEnv(rocket_spec, env_config, level=args.level, render_mode="human")
    pd_gains = build_pd_gains(PD_GAINS_FILE)
    controller = make_controller(
        "network" if args.network else args.controller,
        env.rocket_simulator,
        pd_gains,
        args.network,
    )
    fps = env.metadata["render_fps"]
    video = VideoWriter(args.record, fps) if args.record else None
    flights = 0
    try:
        while not env.window_closed and flights != args.flights:
            episode = None if args.test_episode is None else args.test_episode + flights
            seed = None if episode is None else seed_for_test_episode(episode)
            observation, _ = env.reset(seed=seed)
            if video:
                video.add(env.environment_renderer.to_array())
            done = False
            while not done and not env.window_closed:
                observation, _, terminated, truncated, info = env.step(controller(observation))
                if video:
                    video.add(env.environment_renderer.to_array())
                done = terminated or truncated
            if done:
                flights += 1
                label = "random start" if episode is None else f"test episode {episode}"
                reasons = "; ".join(describe_failures(info["touchdown"], info["failed_checks"]))
                print(f"{label}: {info['outcome']}" + (f" ({reasons})" if reasons else ""))
                if video:
                    video.hold(1.5)
    finally:
        env.close()
        if video:
            video.close()
            print(f"video saved to {video.path} ({video.frames / fps:.0f} s)")


if __name__ == "__main__":
    main()
