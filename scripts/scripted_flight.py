"""Fly the rocket through a fixed sequence of nozzle commands and plot what happens.

A visual check of the physics: free fall, braking, hovering, and moving sideways by tilting.
The sequence stays well above the ground.

    uv run python scripts/scripted_flight.py          # saves runs/scripted_flight.png
    uv run python scripts/scripted_flight.py --show   # also opens a window
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from rocketlander.rocket_simulator import STANDARD_GRAVITY, RocketSimulator
from rocketlander.rocket_spec import load_rocket
from rocketlander.rocket_state import RocketState

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show", action="store_true", help="open the plot in a window")
    args = parser.parse_args()

    rocket = load_rocket(ROOT / "configs" / "rocket" / "r100.yaml")
    sim = RocketSimulator(rocket)
    sim.reset(RocketState(x=0.0, y=50.0, vx=0.0, vy=0.0))
    hover = sim.mass * STANDARD_GRAVITY / rocket.nozzle("main").max_thrust

    # (duration [s], throttles [main, left, right], what happens)
    segments = [
        (1.0, (0.0, 0.0, 0.0), "free fall"),
        (1.0, (1.0, 0.0, 0.0), "full thrust: brake"),
        (2.0, (hover, 0.0, 0.0), "hover"),
        (0.5, (hover, 1.0, 0.0), "left nozzle: start tilting right"),
        (0.5, (hover, 0.0, 1.0), "right nozzle: stop the rotation"),
        (3.0, (hover, 0.0, 0.0), "tilted: drift right"),
        (0.5, (hover, 0.0, 1.0), "right nozzle: start tilting back"),
        (0.5, (hover, 1.0, 0.0), "left nozzle: stop upright"),
        (2.0, (hover, 0.0, 0.0), "hover, coasting right"),
    ]

    times, states, commands = [0.0], [sim.state()], []
    print(f"hover throttle = {hover:.4f}\n")
    print(f"{'t [s]':>6} {'x [m]':>7} {'y [m]':>7} {'vx':>6} {'vy':>6} {'tilt [deg]':>10}  segment")
    for duration, throttles, label in segments:
        for _ in range(round(duration / sim.dt)):
            commands.append(throttles)
            states.append(sim.step(throttles))
            times.append(times[-1] + sim.dt)
        s = states[-1]
        print(
            f"{times[-1]:6.1f} {s.x:7.2f} {s.y:7.2f} {s.vx:6.2f} {s.vy:6.2f} "
            f"{math.degrees(s.theta):10.2f}  {label}"
        )

    path = ROOT / "runs" / "scripted_flight.png"
    path.parent.mkdir(exist_ok=True)
    _plot(times, states, commands, sim.dt, path)
    print(f"\nplot saved to {path}")
    if args.show:
        plt.show()


def _plot(times, states, commands, dt: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Scripted flight (Box2D)")

    # Trajectory, with the rocket's axis drawn every 0.5 s to show its tilt.
    ax = axes[0, 0]
    ax.plot([s.x for s in states], [s.y for s in states], color="tab:blue")
    for s in states[:: round(0.5 / dt)]:
        dx, dy = -math.sin(s.theta) * 1.5, math.cos(s.theta) * 1.5
        ax.plot([s.x - dx, s.x + dx], [s.y - dy, s.y + dy], color="black", linewidth=2)
    ax.set(title="Trajectory (rocket axis every 0.5 s)", xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[0, 1]
    ax.plot(times, [s.y for s in states], label="altitude y [m]")
    ax.plot(times, [s.vy for s in states], label="vertical speed vy [m/s]")
    ax.plot(times, [s.vx for s in states], label="horizontal speed vx [m/s]")
    ax.set(title="Altitude and speed", xlabel="time [s]")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(times, [math.degrees(s.theta) for s in states])
    ax.set(title="Tilt (negative = nose toward +x)", xlabel="time [s]", ylabel="tilt [deg]")

    ax = axes[1, 1]
    for i, name in enumerate(("main", "left", "right")):
        ax.step(times[1:], [c[i] for c in commands], where="pre", label=name)
    ax.set(title="Throttle commands", xlabel="time [s]", ylabel="throttle [0-1]")
    ax.legend()

    for ax in axes.flat[1:]:
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
