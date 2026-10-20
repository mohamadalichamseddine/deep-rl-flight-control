"""The look shared by our matplotlib figures: evaluation plots and training curves."""

from matplotlib.axes import Axes

SURFACE = "#fcfcfb"
TEXT, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID = "#e4e3df"


def style_axes(ax: Axes) -> None:
    """Light surface, soft grid, gray labels, no top and right frame."""
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
