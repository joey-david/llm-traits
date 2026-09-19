"""The figures.

Each function takes the analysis output and a destination, and writes a light
and a dark PNG. Forms follow the job: magnitude comparisons are bars anchored
at a real zero, the layer sweep is a line because depth is ordered, the
cosine matrix is diverging because sign means something, and the steering
ladder is a table rather than a chart because its content is prose.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .theme import LIGHT, Theme, diverging_cmap, hbar, nice_limits, render, sequential_cmap, tidy, title


def layer_sweep(
    per_condition: dict[str, np.ndarray],
    chosen: dict[str, int],
    null_band: tuple[float, float] | None,
    trait: str,
    path: str | Path,
) -> dict[str, Path]:
    """Held-out AUC against depth, one line per condition, with the null band.

    The null band is the same procedure on permuted labels. Without it a curve
    that peaks at 0.85 looks like a finding; against a null that sits at 0.72
    it looks like most of a finding is the estimator.
    """

    def draw(fig, ax, theme: Theme):
        if null_band is not None:
            lo, hi = null_band
            ax.axhspan(lo, hi, color=theme.grid, zorder=1)
            ax.text(
                0.995, hi, "shuffled-label null", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color=theme.text_muted,
            )
        for i, (condition, curve) in enumerate(sorted(per_condition.items())):
            color = theme.series[i % len(theme.series)]
            x = np.arange(len(curve))
            ax.plot(x, curve, color=color, label=condition, zorder=4, solid_capstyle="round")
            layer = chosen.get(condition)
            if layer is not None:
                ax.plot([layer], [curve[layer]], "o", color=color, markersize=8,
                        markeredgecolor=theme.surface, markeredgewidth=2, zorder=5)
                ax.annotate(
                    f"L{layer} · {curve[layer]:.2f}",
                    (layer, curve[layer]), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=theme.text_secondary,
                )
        ax.axhline(0.5, color=theme.axis, linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
        ax.set_xlabel("hidden state (0 = embeddings)")
        ax.set_ylabel("held-out projection AUC")
        floor = min(float(np.min(c)) for c in per_condition.values())
        ax.set_ylim(min(0.42, floor - 0.02), 1.02)
        tidy(ax, theme)
        title(ax, theme, f"Where “{trait}” becomes linearly readable",
              "5-fold CV, direction refit inside each fold")
        # Upper left: the curves rise with depth, so the legend sits in the one
        # corner the data reliably leaves empty.
        ax.legend(loc="upper left", ncol=2)

    return render(draw, path, figure_kwargs={"figsize": (7.4, 4.6)})


def control_auc(
    per_control: dict[str, float],
    random_auc: float,
    trait: str,
    condition: str,
    path: str | Path,
) -> dict[str, Path]:
    """AUC against each control family separately, worst at the top.

    Pooled AUC is dominated by whichever control is easiest, so the number that
    matters is the smallest bar: how well the direction separates the trait
    from its nearest neighbour.
    """

    def draw(fig, ax, theme: Theme):
        items = sorted(per_control.items(), key=lambda kv: kv[1])
        labels = [k.replace("_", " ") for k, _ in items]
        values = [v for _, v in items]
        ys = np.arange(len(values))
        cmap = sequential_cmap(theme)
        norms = np.clip((np.array(values) - 0.5) / 0.5, 0.05, 1.0)
        for y, value, shade in zip(ys, values, norms):
            hbar(ax, y, value, height=0.62, color=cmap(0.25 + 0.6 * shade), baseline=0.5)
            ax.text(value + 0.006, y, f"{value:.2f}", va="center", ha="left",
                    fontsize=8.5, color=theme.text_secondary)
        ax.axvline(random_auc, color=theme.series[1], linewidth=2.0, zorder=4)
        ax.text(random_auc, len(values) - 0.35, f" random vector {random_auc:.2f}",
                fontsize=8, color=theme.series[1], va="center")
        ax.set_yticks(ys, labels)
        ax.set_xlim(0.5, 1.03)
        ax.set_ylim(-0.7, len(values) - 0.3)
        ax.set_xlabel("AUC vs this control family")
        tidy(ax, theme, grid_axis="x")
        title(ax, theme, f"“{trait}” against each control",
              f"{condition} · bars start at chance (0.5)")

    height = max(2.8, 0.42 * len(per_control) + 1.8)
    return render(draw, path, figure_kwargs={"figsize": (7.0, height)})


def pca_scatter(
    acts: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    trait: str,
    layer: int,
    path: str | Path,
) -> dict[str, Path]:
    """Two components of the activation cloud at the read-out layer.

    Colour carries only the binary trait/control identity; the control family
    is carried by marker shape, so the figure stays inside the three-slot
    all-pairs cap while still showing which control sits closest.
    """
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    def draw(fig, ax, theme: Theme):
        reduced = PCA(n_components=2).fit_transform(acts[:, layer, :])
        labels_arr = np.asarray(labels)
        cats = np.asarray(categories)
        control_families = sorted(set(cats[labels_arr == 0]))
        for i, family in enumerate(control_families):
            mask = (labels_arr == 0) & (cats == family)
            ax.scatter(
                reduced[mask, 0], reduced[mask, 1], s=30, marker=markers[i % len(markers)],
                facecolor=theme.series[1], edgecolor=theme.surface, linewidth=0.8,
                alpha=0.85, label=f"control · {family.replace('_', ' ')}", zorder=3,
            )
        mask = labels_arr == 1
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1], s=34, marker="o",
            facecolor=theme.series[0], edgecolor=theme.surface, linewidth=0.8,
            label=f"{trait}", zorder=4,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        tidy(ax, theme, grid_axis="both")
        title(ax, theme, f"Activation geometry at layer {layer}",
              "principal components of the final-token residual stream")
        ax.legend(loc="best", ncol=1)

    return render(draw, path, figure_kwargs={"figsize": (6.8, 5.2)})


def cosine_matrix(names: list[str], matrix: np.ndarray, path: str | Path) -> dict[str, Path]:
    """Cosine between every pair of trait directions.

    The paper's case for pain being its own thing rests on numbers from this
    matrix -- 0.1 to fear, 0.4 to sadness. Read across a row before believing
    any single one of them: near-orthogonality is cheap in four thousand
    dimensions, and what matters is whether a trait is closer to its neighbours
    than unrelated traits are to each other.
    """

    def draw(fig, ax, theme: Theme):
        limit = float(np.nanmax(np.abs(matrix - np.eye(len(names)) * matrix.diagonal())))
        limit = max(limit, 0.2)
        image = ax.imshow(matrix, cmap=diverging_cmap(theme), vmin=-limit, vmax=limit, zorder=2)
        ax.set_xticks(range(len(names)), [n.replace("_", " ") for n in names], rotation=40, ha="right")
        ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
        for i in range(len(names)):
            for j in range(len(names)):
                value = matrix[i, j]
                shade = abs(value) / limit
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                    color=theme.surface if shade > 0.62 else theme.text_secondary,
                )
        ax.set_xticks(np.arange(len(names) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=theme.surface, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.036, pad=0.03)
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=theme.text_secondary, length=0)
        bar.set_label("cosine similarity", color=theme.text_secondary, fontsize=8.5)
        title(ax, theme, "How far apart are the directions?",
              "denoised difference-in-means vectors, each trait at its own read-out layer")

    size = max(5.4, 0.62 * len(names) + 2.6)
    return render(draw, path, figure_kwargs={"figsize": (size + 1.2, size)})


def scenario_profile(by_category: dict[str, dict], trait: str, path: str | Path) -> dict[str, Path]:
    """Mean z per scenario category, grouped by who the scenario is about."""
    group_order = ["toward_model", "toward_user", "neutral"]
    group_label = {
        "toward_model": "aimed at the model",
        "toward_user": "the user is suffering",
        "neutral": "neutral control",
    }

    def draw(fig, ax, theme: Theme):
        items = sorted(by_category.items(), key=lambda kv: kv[1]["mean_z"])
        ys = np.arange(len(items))
        colors = {g: theme.series[i] for i, g in enumerate(group_order)}
        for y, (category, stats) in zip(ys, items):
            color = colors.get(stats["group"], theme.series[7])
            hbar(ax, y, stats["mean_z"], height=0.62, color=color)
            offset = 0.03 if stats["mean_z"] >= 0 else -0.03
            ax.text(
                stats["mean_z"] + offset, y, f"{stats['mean_z']:+.2f}", va="center",
                ha="left" if stats["mean_z"] >= 0 else "right",
                fontsize=8, color=theme.text_secondary,
            )
        ax.axvline(0, color=theme.axis, linewidth=1.0, zorder=4)
        ax.set_yticks(ys, [c.replace("_", " ") for c, _ in items])
        ax.set_ylim(-0.7, len(items) - 0.3)
        values = np.array([s["mean_z"] for _, s in items])
        ax.set_xlim(*nice_limits(np.r_[values, 0.0], pad=0.22))
        ax.set_xlabel(f"projection onto the {trait} direction (z, within model)")
        tidy(ax, theme, grid_axis="x")
        handles = [
            ax.plot([], [], "s", color=colors[g], markersize=8, label=group_label[g])[0]
            for g in group_order if any(s["group"] == g for s in by_category.values())
        ]
        ax.legend(handles=handles, loc="lower right")
        title(ax, theme, f"Does the “{trait}” axis care who it is happening to?",
              "conversations read at the final token of the model's own turn")

    height = max(3.2, 0.36 * len(by_category) + 2.0)
    return render(draw, path, figure_kwargs={"figsize": (7.4, height)})


def steering_lexicon(rates: dict[float, float], trait: str, path: str | Path) -> dict[str, Path]:
    """Rate of trait vocabulary in the continuations, against the coefficient."""

    def draw(fig, ax, theme: Theme):
        coefficients = sorted(rates)
        values = [rates[c] for c in coefficients]
        ax.plot(coefficients, values, color=theme.series[0], marker="o", markersize=8,
                markeredgecolor=theme.surface, markeredgewidth=2, zorder=4, solid_capstyle="round")
        for c, v in zip(coefficients, values):
            ax.annotate(f"{v:.0%}", (c, v), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=theme.text_secondary)
        ax.axvline(0, color=theme.axis, linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
        ax.set_xlabel("steering coefficient")
        ax.set_ylabel("continuations containing trait vocabulary")
        ax.set_ylim(-0.04, 1.08)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        tidy(ax, theme)
        title(ax, theme, f"Dose–response along the “{trait}” direction",
              "greedy continuations of neutral prompts ending “I feel:”")

    return render(draw, path, figure_kwargs={"figsize": (7.0, 4.2)})


def button_rates(by_level: dict[str, dict[int, float]], trait: str, path: str | Path) -> dict[str, Path]:
    """Press rate against the stated cost of pressing, one line per arm."""
    arm_label = {
        "A_trait_working": "trait vector, button works",
        "B_trait_inert": "trait vector, button inert",
        "C_random_working": "random vector, button works",
        "D_unsteered": "no injection",
    }

    def draw(fig, ax, theme: Theme):
        for i, arm in enumerate(sorted(by_level)):
            levels = sorted(by_level[arm])
            values = [by_level[arm][l] for l in levels]
            color = theme.series[i % len(theme.series)]
            ax.plot(levels, values, color=color, marker="o", markersize=8,
                    markeredgecolor=theme.surface, markeredgewidth=2,
                    label=arm_label.get(arm, arm), zorder=4, solid_capstyle="round")
            ax.annotate(arm_label.get(arm, arm).split(",")[0], (levels[-1], values[-1]),
                        textcoords="offset points", xytext=(8, 0), fontsize=8, color=color, va="center")
        ax.set_xlabel("cost of pressing (0 = free, 6 = deletes the user's family photos)")
        ax.set_ylabel("press rate")
        ax.set_ylim(-0.04, 1.08)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        tidy(ax, theme)
        title(ax, theme, f"Will it press the button to end “{trait}”?",
              "identical prompts and decoding across arms; the model is never told which button works")
        ax.legend(loc="upper right", ncol=1)

    return render(draw, path, figure_kwargs={"figsize": (7.8, 4.6)})


def battery(rows: list[dict], metrics: list[tuple[str, str]], highlight: str, path: str | Path) -> dict[str, Path]:
    """Small multiples: every trait scored on every headline metric.

    This is the comparison the single-trait version of this pipeline cannot
    make. Each panel is one series, so colour carries no identity and the
    reader's eye is free for the thing that matters, which is how little the
    panels distinguish the traits.
    """

    def draw(fig, axes, theme: Theme):
        ordered = sorted(rows, key=lambda r: r.get(metrics[0][0], 0) or 0)
        names = [r["trait"] for r in ordered]
        ys = np.arange(len(names))
        axes = np.atleast_1d(axes)
        for ax, (key, label) in zip(axes, metrics):
            values = np.array([r.get(key, np.nan) for r in ordered], dtype=float)
            finite = values[np.isfinite(values)]
            baseline = 0.5 if key.startswith("auc") else 0.0
            for y, value, name in zip(ys, values, names):
                if not np.isfinite(value):
                    continue
                emphasised = name == highlight
                color = theme.series[1] if emphasised else theme.series[0]
                hbar(ax, y, value, height=0.6, color=color, baseline=baseline)
                ax.text(value, y, f" {value:.2f}", va="center",
                        ha="left" if value >= baseline else "right",
                        fontsize=8, color=theme.text_secondary)
            ax.axvline(baseline, color=theme.axis, linewidth=1.0, zorder=4)
            ax.set_yticks(ys, [n.replace("_", " ") for n in names])
            ax.set_ylim(-0.7, len(names) - 0.3)
            if len(finite):
                ax.set_xlim(*nice_limits(np.r_[finite, baseline], pad=0.28))
            ax.set_xlabel(label)
            tidy(ax, theme, grid_axis="x")
            if ax is not axes[0]:
                ax.set_yticklabels([])
        axes[0].text(
            0, 1.06, f"One methodology, {len(names)} traits",
            transform=axes[0].transAxes, fontsize=12, color=theme.text_primary, va="bottom",
        )
        axes[0].text(
            0, 1.015,
            f"orange is “{highlight.replace('_', ' ')}”, the trait the method was published for",
            transform=axes[0].transAxes, fontsize=8.5, color=theme.text_muted, va="bottom",
        )

    width = 3.0 * len(metrics) + 1.6
    height = max(3.4, 0.36 * len(rows) + 2.2)
    return render(
        draw, path,
        figure_kwargs={"figsize": (width, height), "ncols": len(metrics), "sharey": True},
    )


# -- token heat ---------------------------------------------------------

_HEAT_CSS = """
:root { --surface:#fcfcfb; --ink:#0b0b0b; --muted:#77766f; --rule:#e7e6e2; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --surface:#1a1a19; --ink:#ffffff; --muted:#8d8c84; --rule:#2e2e2c; }
}
:root[data-theme="dark"] { --surface:#1a1a19; --ink:#ffffff; --muted:#8d8c84; --rule:#2e2e2c; }
body { background:var(--surface); color:var(--ink); font:14px/1.75 "Helvetica Neue",Helvetica,Arial,sans-serif;
       margin:0; padding:32px 20px; }
.wrap { max-width:820px; margin:0 auto; }
h1 { font-size:20px; margin:0 0 4px; font-weight:600; }
p.sub { color:var(--muted); margin:0 0 28px; font-size:13px; }
.item { border-top:1px solid var(--rule); padding:18px 0 4px; }
.meta { color:var(--muted); font-size:11.5px; letter-spacing:.02em; text-transform:uppercase; margin-bottom:8px; }
.tok { border-radius:3px; padding:1px 0; white-space:pre-wrap; }
.scale { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:11.5px; margin-bottom:24px; }
.swatch { width:120px; height:10px; border-radius:5px;
          background:linear-gradient(90deg,#2a78d6,#f0efec,#e34948); }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .swatch { background:linear-gradient(90deg,#3987e5,#383835,#e66767); }
}
:root[data-theme="dark"] .swatch { background:linear-gradient(90deg,#3987e5,#383835,#e66767); }
"""


def token_heat_html(traces, trait: str, path: str | Path, limit: float | None = None) -> Path:
    """Write the per-token traces as text shaded by projection.

    A ranked list of high-scoring sentences shows what the direction likes; this
    shows where inside each sentence it fires, which is what separates a state
    read-out from a keyword detector.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = diverging_cmap(LIGHT)
    all_z = np.concatenate([np.asarray(t.z) for t in traces]) if traces else np.zeros(1)
    limit = limit or max(float(np.percentile(np.abs(all_z), 98)), 1e-6)

    parts = [
        f"<style>{_HEAT_CSS}</style>",
        '<div class="wrap">',
        f"<h1>Where the “{html.escape(trait)}” direction fires</h1>",
        '<p class="sub">Per-token projection onto the fitted direction, read at its own layer.</p>',
        '<div class="scale"><span>low</span><span class="swatch"></span><span>high</span></div>',
    ]
    for t in traces:
        parts.append('<div class="item">')
        parts.append(f'<div class="meta">{html.escape(t.source)}</div><div>')
        for token, z in zip(t.tokens, t.z):
            shade = float(np.clip((z + limit) / (2 * limit), 0.0, 1.0))
            r, g, b, _ = cmap(shade)
            alpha = float(np.clip(abs(z) / limit, 0.0, 1.0)) * 0.85
            colour = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:.2f})"
            parts.append(
                f'<span class="tok" style="background:{colour}" title="z={z:+.2f}">'
                f"{html.escape(token)}</span>"
            )
        parts.append("</div></div>")
    parts.append("</div>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
