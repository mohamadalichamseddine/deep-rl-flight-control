"""The training curves: how the reward, the landing rate and the losses changed over training."""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter, PercentFormatter

from rocketlander.ppo.training_round import TrainingRound
from rocketlander.visualization.plot_style import SURFACE, TEXT, TEXT_SECONDARY, style_axes

TRAINING, VALIDATION = "#2a78d6", "#eb6834"  # blue, orange


def plot_training_curves(training_rounds: list[TrainingRound], title: str, path: Path) -> Path:
    """Save five panels over the training steps: the average reward and the landing rate of the
    training and validation flights, then the actor, critic and total losses."""
    # A standalone Figure (not pyplot), so saving needs no window and changes no global state.
    fig = Figure(figsize=(9, 12), layout="constrained", facecolor=SURFACE)
    reward, landing_rate, actor_loss, critic_loss, total_loss = fig.subplots(5, 1, sharex=True)
    fig.suptitle(title, color=TEXT, fontsize=13)
    steps = [r.step for r in training_rounds]

    for ax, training_values, validation_values in (
        (
            reward,
            [r.training_reward for r in training_rounds],
            [r.validation_reward for r in training_rounds],
        ),
        (
            landing_rate,
            [r.training_landing_rate for r in training_rounds],
            [r.validation_landing_rate for r in training_rounds],
        ),
    ):
        _plot(ax, steps, training_values, TRAINING, "training flights (with noise)", None)
        _plot(ax, steps, validation_values, VALIDATION, "validation flights (no noise)", "o")
    reward.set(title="Average reward per flight", ylabel="reward")
    landing_rate.set(title="Landing rate", ylabel="flights landed", ylim=(-0.05, 1.05))
    landing_rate.yaxis.set_major_formatter(PercentFormatter(1.0))

    # The losses are computed on the training flights.
    for ax, values, loss_title in (
        (actor_loss, [r.actor_loss for r in training_rounds], "Actor loss (PPO's clipped loss)"),
        (
            critic_loss,
            [r.critic_loss for r in training_rounds],
            "Critic loss (error of its reward predictions)",
        ),
        (total_loss, [r.total_loss for r in training_rounds], "Total loss"),
    ):
        _plot(ax, steps, values, TRAINING, None, None)
        ax.set(title=loss_title, ylabel="loss")

    total_loss.set_xlabel("training steps")
    total_loss.xaxis.set_major_formatter(EngFormatter(sep=""))
    for ax in (reward, landing_rate, actor_loss, critic_loss, total_loss):
        style_axes(ax)
    for ax in (reward, landing_rate):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    return path


def _plot(
    ax: Axes,
    steps: list[int],
    values: list[float | None],
    color: str,
    label: str | None,
    marker: str | None,
) -> None:
    """One curve, through the rounds where the value was measured."""
    measured = [
        (step, value) for step, value in zip(steps, values, strict=True) if value is not None
    ]
    if measured:
        xs, ys = zip(*measured, strict=True)
        ax.plot(xs, ys, color=color, linewidth=1.5, marker=marker, markersize=5, label=label)
