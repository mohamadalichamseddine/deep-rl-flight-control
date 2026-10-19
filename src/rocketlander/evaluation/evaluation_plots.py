"""Plots of an evaluation: flight paths and how close touchdowns came to the safe-landing limits."""

import math
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from rocketlander.environment.env_config import LandingCriteria
from rocketlander.environment.outcome import Outcome
from rocketlander.evaluation.evaluation import EpisodeResult

SURFACE = "#fcfcfb"
TEXT, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID = "#e4e3df"
SAFE_ZONE = "#ecebe7"
LANDED, FAILED = "#0ca30c", "#d03b3b"  # status colors: good, critical


def plot_evaluation(
    results: list[EpisodeResult], criteria: LandingCriteria, title: str, path: Path
) -> Path:
    """Save a three-panel figure: flight paths, then touchdowns against the safe limits."""
    # A standalone Figure (not pyplot), so saving needs no window and changes no global state.
    fig = Figure(figsize=(16, 5.2), layout="constrained", facecolor=SURFACE)
    paths, position, attitude = fig.subplots(1, 3)
    fig.suptitle(title, color=TEXT, fontsize=13)
    landed = [r for r in results if r.outcome == Outcome.LANDED]
    failed = [r for r in results if r.outcome != Outcome.LANDED]

    # Flight paths of the center of mass, failed ones drawn on top.
    for group, color, label in ((landed, LANDED, "landed"), (failed, FAILED, "failed")):
        for i, result in enumerate(group):
            xs, ys = zip(*result.trajectory, strict=True)
            paths.plot(
                xs, ys, color=color, linewidth=1, alpha=0.25, label=label if i == 0 else None
            )
    paths.set(title="Flight paths", xlabel="x [m]", ylabel="altitude of center of mass [m]")

    # Touchdowns: one dot per episode that touched the ground, against the limits.
    touchdowns = [(r, r.touchdown) for r in results if r.touchdown is not None]
    _scatter_with_limits(
        position,
        [(r, t.state.x, -t.state.vy) for r, t in touchdowns],
        x_limit=criteria.max_off_center,
        y_range=(0.0, criteria.max_vy),
        title="Touchdown: position and vertical speed",
        xlabel="distance from pad center [m]",
        ylabel="downward speed [m/s]",
    )
    _scatter_with_limits(
        attitude,
        [(r, t.state.vx, math.degrees(t.state.tilt)) for r, t in touchdowns],
        x_limit=criteria.max_vx,
        y_range=(-criteria.max_tilt_deg, criteria.max_tilt_deg),
        title="Touchdown: horizontal speed and tilt",
        xlabel="horizontal speed [m/s]",
        ylabel="tilt [deg]",
    )

    for ax in (paths, position, attitude):
        _style(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="best")
            for handle in legend.legend_handles:
                handle.set_alpha(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    return path


def _scatter_with_limits(ax, points, x_limit, y_range, title, xlabel, ylabel) -> None:
    low, high = y_range
    ax.add_patch(
        Rectangle(
            (-x_limit, low),
            2 * x_limit,
            high - low,
            facecolor=SAFE_ZONE,
            edgecolor=TEXT_SECONDARY,
            linestyle="--",
            linewidth=1,
            label="safe-landing limits",
        )
    )
    for landed, color, marker, label in (
        (True, LANDED, "o", "landed"),
        (False, FAILED, "X", "failed"),
    ):
        chosen = [(x, y) for r, x, y in points if (r.outcome == Outcome.LANDED) == landed]
        if chosen:
            xs, ys = zip(*chosen, strict=True)
            ax.scatter(
                xs,
                ys,
                s=28,
                color=color,
                marker=marker,
                edgecolor=SURFACE,
                linewidth=0.6,
                label=f"{label} ({len(chosen)})",
                zorder=3,
            )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    # Always show the whole safe zone with some room around it, even if all dots are inside.
    ax.update_datalim([(-1.5 * x_limit, low - 0.25 * (high - low)), (1.5 * x_limit, 1.5 * high)])
    ax.autoscale_view()


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(TEXT)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
