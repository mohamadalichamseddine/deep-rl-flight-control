"""Watch a controller fly in the live window: test episodes, or new random starts.

Re-flies a test episode exactly as the evaluation flew it: same seed, same controller.

    uv run python scripts/watch_controller.py              # PD, level 0, from test episode 0
    uv run python scripts/watch_controller.py --level 2 --test-episode 137
    uv run python scripts/watch_controller.py --random                 # new random starts
    uv run python scripts/watch_controller.py --flights 3 --record runs/pd_demo.mp4
"""

import argparse

from rocketlander.controllers import CONTROLLERS, make_controller
from rocketlander.evaluation import seed_for_test_episode
from rocketlander.rocket_landing_env import RocketLandingEnv
from rocketlander.video_writer import VideoWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controller", default="pd", choices=sorted(CONTROLLERS))
    parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--test-episode", type=int, default=0, help="first test episode to fly")
    parser.add_argument("--random", action="store_true", help="random starts, not the test set")
    parser.add_argument("--flights", type=int, help="stop after this many flights")
    parser.add_argument("--record", help="save the flights to this MP4 file")
    args = parser.parse_args()

    env = RocketLandingEnv(level=args.level, render_mode="human")
    controller = make_controller(args.controller, env)
    fps = env.metadata["render_fps"]
    video = VideoWriter(args.record, fps) if args.record else None
    flights = 0
    try:
        while not env.window_closed and flights != args.flights:
            episode = args.test_episode + flights
            seed = None if args.random else seed_for_test_episode(episode)
            observation, _ = env.reset(seed=seed)
            if video:
                video.add(env.renderer.to_array())
            done = False
            while not done and not env.window_closed:
                observation, _, terminated, truncated, info = env.step(controller(observation))
                if video:
                    video.add(env.renderer.to_array())
                done = terminated or truncated
            if done:
                flights += 1
                label = "random start" if args.random else f"test episode {episode}"
                checks = "; ".join(info.get("failed_checks", []))
                print(f"{label}: {info['outcome']}" + (f" ({checks})" if checks else ""))
                if video:
                    video.hold(1.5)
    finally:
        env.close()
        if video:
            video.close()
            print(f"video saved to {video.path} ({video.frames / fps:.0f} s)")


if __name__ == "__main__":
    main()
