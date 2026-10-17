"""Draw the rocket as Box2D sees it: its shapes, center of mass, nozzles, and legs.

The shapes and center of mass are read back from Box2D, so the picture shows exactly what is
simulated.

    uv run python scripts/draw_rocket.py          # saves runs/rocket.png
    uv run python scripts/draw_rocket.py --show   # also opens a window
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from rocketlander.rocket_simulator import RocketSimulator
from rocketlander.rocket_spec import load_rocket
from rocketlander.rocket_state import RocketState

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"body": "#c9d3df", "engine block": "#6b7a8f", "leg": "#333333"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show", action="store_true", help="open the drawing in a window")
    args = parser.parse_args()

    rocket = load_rocket(ROOT / "configs" / "rocket" / "r100.yaml")
    sim = RocketSimulator(rocket)
    # Stand the rocket upright on its legs: the ground (y = 0) is under the leg tips, and
    # everything in rocket coordinates is lifted by the height of the base.
    base_height = -min(leg.tip[1] for leg in rocket.legs)
    sim.reset(RocketState(x=0.0, y=base_height + sim.com_height, vx=0.0, vy=0.0))

    fig, ax = plt.subplots(figsize=(7, 9), constrained_layout=True)
    ax.set_title(
        f"r100 as Box2D sees it: {sim.mass:.0f} kg, inertia {sim.moment_of_inertia:.1f} kg·m²"
    )

    labeled = set()
    for name, corners in sim.shapes():
        label = None if name in labeled else name
        labeled.add(name)
        ax.add_patch(Polygon(corners, facecolor=COLORS[name], edgecolor="black", label=label))

    ax.annotate("", xy=(0.3, 0.0), xytext=(0.3, base_height), arrowprops={"arrowstyle": "<->"})
    ax.text(0.35, base_height / 2, f"{base_height:g} m engine clearance", va="center")

    com_y = base_height + sim.com_height
    ax.plot(0.0, com_y, "o", color="crimson", markersize=9, zorder=5)
    ax.annotate(
        f"center of mass\n{sim.com_height:.2f} m above the base (Box2D)",
        (0.0, com_y),
        xytext=(0.45, com_y),
        va="center",
        arrowprops={"arrowstyle": "-", "color": "crimson"},
        color="crimson",
    )

    for nozzle in rocket.nozzles:
        (x, y), (dx, dy) = nozzle.position, nozzle.direction
        y += base_height
        # Flame: the exhaust leaves opposite to the force on the rocket.
        flame = 0.6 if nozzle.name == "main" else 0.35
        ax.annotate(
            "",
            xy=(x - dx * flame, y - dy * flame),
            xytext=(x, y),
            arrowprops={"arrowstyle": "-|>", "color": "darkorange", "linewidth": 2.5},
        )
        ax.plot(x, y, "s", color="darkorange", markersize=7, zorder=5)
        label_x = x - dx * flame + (0.08 if dx <= 0 else -0.08)
        label_y = y - dy * flame - (0.12 if nozzle.name == "main" else 0.0)
        ax.text(
            label_x,
            label_y,
            f"{nozzle.name}\n{nozzle.max_thrust:g} N",
            ha="left" if dx < 0 else "right" if dx > 0 else "center",
            va="center" if dy == 0 else "top",
            color="darkorange",
        )
    ax.plot([], [], color="darkorange", linewidth=2.5, label="nozzle exhaust")

    ax.axhline(0.0, color="saddlebrown", linewidth=1)
    ax.text(-1.15, 0.03, "ground (y = 0)", color="saddlebrown", ha="left", va="bottom")

    ax.set(xlabel="x [m]", ylabel="y [m]", xlim=(-1.2, 1.4), ylim=(-0.8, 3.5))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.legend(loc="outside lower center", ncols=4)

    path = ROOT / "runs" / "rocket.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f"drawing saved to {path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
