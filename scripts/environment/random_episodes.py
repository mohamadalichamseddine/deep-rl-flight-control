"""Fly episodes with random actions: checks the environment runs and measures its speed.

uv run python scripts/environment/random_episodes.py                   # 1000 episodes, level 0
uv run python scripts/environment/random_episodes.py -n 200 --level 2
"""

import argparse
import time
from collections import Counter

from rocketlander.config_files import ENV_CONFIG_FILE, ROCKET_SPEC_FILE
from rocketlander.environment.env_config import build_env_config
from rocketlander.environment.rocket_landing_env import RocketLandingEnv
from rocketlander.rocket.rocket_spec import build_rocket_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--episodes", type=int, default=1000)
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()

    rocket_spec = build_rocket_spec(ROCKET_SPEC_FILE)
    env_config = build_env_config(ENV_CONFIG_FILE)
    env = RocketLandingEnv(rocket_spec, env_config, level=args.level, render_mode=None)
    env.action_space.seed(0)
    outcomes, steps, rewards = Counter(), 0, []
    start = time.perf_counter()
    for episode in range(args.episodes):
        env.reset(seed=episode)
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            steps += 1
            done = terminated or truncated
        outcomes[info["outcome"]] += 1
        rewards.append(total)
    elapsed = time.perf_counter() - start

    print(f"{args.episodes} episodes at level {args.level}, random actions\n")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome:<18} {count:>5}  ({count / args.episodes:.0%})")
    print(f"\n  mean episode length  {steps / args.episodes * env.env_config.dt:.1f} s")
    print(f"  mean total reward    {sum(rewards) / len(rewards):.1f}")
    print(f"  speed                {steps / elapsed:,.0f} steps/s on one core")


if __name__ == "__main__":
    main()
