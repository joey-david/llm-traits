"""Figure theme: one palette, two surfaces, and the mark helpers.

Every figure in the package renders twice, once per surface, so the report can
serve the right one to a reader's theme rather than flipping a light PNG to
dark. The categorical order is fixed and never cycled; the subsets actually
used here (four slots on adjacent pairs for line charts, three on all pairs for
scatter and grouped bars) were run through the palette validator in both modes
before being written down.

Aqua and yellow fall below 3:1 on the light surface, which is allowed only with
relief: every figure that uses them carries a legend and direct labels, so
identity never rests on color alone.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


@dataclass(frozen=True)
class Theme:
    mode: str
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    sequential: tuple[str, ...]  # light -> dark, ordered low value to high value
    diverging: tuple[str, str, str]  # negative pole, neutral midpoint, positive pole


LIGHT = Theme(
    mode="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#77766f",
    grid="#e7e6e2",
    axis="#b7b6b1",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    sequential=("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"),
    diverging=("#2a78d6", "#f0efec", "#e34948"),
)

DARK = Theme(
    mode="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#8d8c84",
    grid="#2e2e2c",
    axis="#4a4a46",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    # Reversed on dark so that the low end recedes toward the surface rather
    # than glowing against it.
    sequential=("#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"),
    diverging=("#3987e5", "#383835", "#e66767"),
)

THEMES = {"light": LIGHT, "dark": DARK}


def sequential_cmap(theme: Theme):
    return LinearSegmentedColormap.from_list(f"seq_{theme.mode}", list(theme.sequential))


def diverging_cmap(theme: Theme):
    lo, mid, hi = theme.diverging
    return LinearSegmentedColormap.from_list(f"div_{theme.mode}", [lo, mid, hi])


@contextmanager
def figure(theme: Theme, figsize=(7.2, 4.4), nrows=1, ncols=1, **kwargs):
    """A themed figure. Grid and spines are recessive; only the data is not."""
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 9.5,
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.text_primary,
            "axes.labelcolor": theme.text_secondary,
            "axes.edgecolor": theme.axis,
            "axes.linewidth": 0.8,
            "xtick.color": theme.text_secondary,
            "ytick.color": theme.text_secondary,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "figure.dpi": 160,
        }
    ):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        yield fig, axes
        plt.close(fig)


def tidy(ax, theme: Theme, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(theme.axis)
    ax.spines["bottom"].set_color(theme.axis)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, linestyle="-", alpha=1.0, zorder=0)
        ax.set_axisbelow(True)


def title(ax, theme: Theme, headline: str, subtitle: str | None = None) -> None:
    ax.set_title(headline, loc="left", fontsize=11.5, color=theme.text_primary, pad=14 if subtitle else 8)
    if subtitle:
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color=theme.text_muted,
        )


def hbar(ax, y: float, value: float, height: float, color: str, radius_frac: float = 0.42, baseline: float = 0.0):
    """A horizontal bar rounded only at the data end, square at the baseline.

    Rounding both ends would detach the bar from its own zero line, which is
    the thing the reader measures against.
    """
    span = value - baseline
    if abs(span) < 1e-12:
        return
    sign = 1.0 if span > 0 else -1.0
    r = min(abs(span), height * radius_frac) * sign
    y0, y1 = y - height / 2, y + height / 2
    x0, x1 = baseline, value
    verts = [
        (x0, y0), (x1 - r, y0),
        (x1, y0), (x1, y0 + abs(r)),
        (x1, y1 - abs(r)), (x1, y1),
        (x1 - r, y1), (x0, y1), (x0, y0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO,
        MplPath.CURVE3, MplPath.LINETO,
        MplPath.LINETO, MplPath.CURVE3,
        MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def render(draw, path: str | Path, **kwargs) -> dict[str, Path]:
    """Draw once per surface and write ``name.png`` plus ``name.dark.png``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    figure_kwargs = kwargs.pop("figure_kwargs", {})
    for mode, theme in THEMES.items():
        target = path if mode == "light" else path.with_suffix(f".dark{path.suffix}")
        with figure(theme, **figure_kwargs) as (fig, ax):
            draw(fig, ax, theme, **kwargs)
            fig.savefig(target, bbox_inches="tight", pad_inches=0.28)
        written[mode] = target
    return written


def nice_limits(values: np.ndarray, pad: float = 0.06) -> tuple[float, float]:
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-9:
        return lo - 0.5, hi + 0.5
    margin = (hi - lo) * pad
    return lo - margin, hi + margin
