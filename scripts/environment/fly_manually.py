"""Fly the rocket yourself with the keyboard, in the live window.

Up arrow      main engine (full thrust while held)
Right arrow   push right: fires the left nozzle, which also tilts the nose right
Left arrow    push left: fires the right nozzle, which also tilts the nose left
Space pause · -/+ slower/faster · Esc quit

uv run python scripts/environment/fly_manually.py                       # level 0, until you quit
uv run python scripts/environment/fly_manually.py --level 1
uv run python scripts/environment/fly_manually.py --flights 1 --record runs/my_flight.mp4
"""

import argparse

import numpy as np
import pygame

from rocketlander.config_files import ENV_CONFIG_FILE, ROCKET_SPEC_FILE
from rocketlander.environment.env_config import build_env_config
from rocketlander.environment.landing_judge import describe_failures
from rocketlander.environment.outcome import Outcome
from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.rocket.rocket_spec import build_rocket_spec
from rocketlander.visualization.video_writer import VideoWriter

CAPTION = "Rocket landing - Up: main engine, Left/Right: steer"
CONTROLS = (
    "You fly: hold Up for the main engine · Left/Right to steer · nothing pressed = engines off"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, default=0, help="difficulty level (0-2)")
    parser.add_argument("--flights", type=int, help="stop after this many flights")
    parser.add_argument("--record", help="save the flights to this MP4 file")
    args = parser.parse_args()

    rocket_spec = build_rocket_spec(ROCKET_SPEC_FILE)
    env_config = build_env_config(ENV_CONFIG_FILE)
    env = RocketLandingEnv(rocket_spec, env_config, level=args.level, render_mode="human")
    fps = env.metadata["render_fps"]
    video = VideoWriter(args.record, fps) if args.record else None
    flights = landed = 0
    try:
        while not env.window_closed and flights != args.flights:
            env.reset()
            pygame.display.set_caption(CAPTION)
            env.environment_renderer.controls_hint = CONTROLS
            if flights == 0:
                # Start paused at t = 0: pressing Space also gives the window the keyboard focus.
                env.environment_renderer.paused = True
                env.render()
            if video:
                video.add(env.environment_renderer.to_array())
            done = False
            while not done and not env.window_closed:
                keys = pygame.key.get_pressed()
                action = np.array(
                    [_on(keys[pygame.K_UP]), _on(keys[pygame.K_RIGHT]), _on(keys[pygame.K_LEFT])],
                    dtype=np.float32,
                )
                _, _, terminated, truncated, info = env.step(action)
                if video:
                    video.add(env.environment_renderer.to_array())
                done = terminated or truncated
            if done:
                flights += 1
                landed += info["outcome"] == Outcome.LANDED
                reasons = "; ".join(describe_failures(info["touchdown"], info["failed_checks"]))
                print(f"flight {flights}: {info['outcome']}" + (f" ({reasons})" if reasons else ""))
                if video:
                    video.hold(1.5)
    finally:
        env.close()
        if video:
            video.close()
            print(f"video saved to {video.path} ({video.frames / fps:.0f} s)")
    print(f"{landed} of {flights} flights landed")


def _on(pressed: bool) -> float:
    """Key held: nozzle at full throttle (action 1); released: off (action -1)."""
    return 1.0 if pressed else -1.0


if __name__ == "__main__":
    main()
