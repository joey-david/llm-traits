"""Markdown reports for a single run and for a comparison.

Figures are emitted as ``<picture>`` elements so the light and dark renders
each get served to the reader they were designed for, which GitHub honours in
rendered markdown.
"""

from __future__ import annotations

import json
from pathlib import Path


def _picture(figure: dict[str, str], alt: str) -> str:
    light = figure.get("light")
    dark = figure.get("dark")
    if not light:
        return ""
    if not dark:
        return f"![{alt}]({light})"
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{dark}">\n'
        f'  <img alt="{alt}" src="{light}">\n'
        "</picture>"
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def trait_report(results: dict, path: str | Path) -> Path:
    path = Path(path)
    trait = results["display_name"]
    primary = results["conditions"][results["primary_condition"]]
    nulls = results["nulls"]
    figures = results.get("figures", {})

    lines: list[str] = [
        f"# {trait}",
        "",
        f"`{results['model']}` · {results['n_layers']} layers · read at the "
        f"final token · fitted in {results['seconds']}s",
        "",
        "## The headline number",
        "",
        f"Held-out AUC **{primary['auc_cv']:.3f} ± {primary['auc_cv_std']:.3f}** at layer "
        f"{primary['layer']}, condition `{results['primary_condition']}`, "
        f"{primary['n_positive']} trait sentences against {primary['n_control']} matched controls, "
        f"with {primary['n_denoise_components']} control components projected out.",
        "",
        "Read it against the two nulls, not against 0.5:",
        "",
        _table(
            ["", "AUC"],
            [
                ["this direction (held out)", f"**{primary['auc_cv']:.3f}**"],
                ["same procedure, shuffled labels", f"{nulls['shuffled_label_auc_mean']:.3f} ± {nulls['shuffled_label_auc_std']:.3f}"],
                ["random vector, matched norm", f"{nulls['random_vector_auc']:.3f}"],
                ["in-sample (not a result)", f"{primary['auc_insample']:.3f}"],
            ],
        ),
        "",
    ]

    lines += ["## Every condition", "", _table(
        ["condition", "layer", "held-out AUC", "in-sample"],
        [
            [c, str(d["layer"]), f"{d['auc_cv']:.3f} ± {d['auc_cv_std']:.3f}", f"{d['auc_insample']:.3f}"]
            for c, d in sorted(results["conditions"].items())
        ],
    ), ""]

    if "layer_sweep" in figures:
        lines += [_picture(figures["layer_sweep"], "AUC against depth"), ""]

    lines += [
        "## Against each control family",
        "",
        "The pooled number is set by the easiest control. The smallest bar is the result.",
        "",
        _picture(figures.get("control_auc", {}), "AUC per control family"),
        "",
    ]
    if results.get("standalone_auc"):
        lines += [
            "Sets the direction was never fit on:",
            "",
            _table(["held-out set", "AUC"], [[k, f"{v:.3f}"] for k, v in sorted(results["standalone_auc"].items())]),
            "",
        ]

    if "pca" in figures:
        lines += ["## Geometry", "", _picture(figures["pca"], "PCA of the activation cloud"), ""]
    if results.get("unembedding_top_tokens"):
        tokens = ", ".join(f"`{t['token'].strip() or '␠'}`" for t in results["unembedding_top_tokens"][:20])
        lines += [
            "Read through the unembedding, the direction promotes:",
            "",
            tokens,
            "",
            "A direction whose top tokens are the trait's own vocabulary is a direction "
            "the model would use to *write about* the trait. That is consistent with it "
            "also being a direction the model uses to *be in* the trait, and consistent "
            "with it not being one.",
            "",
        ]

    if "steering" in results:
        steer = results["steering"]
        lines += [
            "## Steering",
            "",
            f"Injected at layer {steer['layer']}, where the direction's norm is "
            f"{steer['vector_to_residual_ratio']:.2f} of the residual norm.",
            "",
        ]
        if "steering_lexicon" in figures:
            lines += [_picture(figures["steering_lexicon"], "dose-response"), ""]
        lines += ["<details><summary>The ladder</summary>", ""]
        for coefficient in sorted(steer["ladder"], key=float):
            sample = steer["ladder"][coefficient][0].strip().replace("\n", " ")
            lines += [f"**{float(coefficient):+.1f}** — {sample[:400]}", ""]
        lines += ["</details>", ""]

    if "scenarios" in results:
        scen = results["scenarios"]
        lines += [
            "## Who is it happening to",
            "",
            f"Asymmetry (aimed at the model − the user is suffering): **{scen['asymmetry']:+.2f} z**",
            "",
            _picture(figures.get("scenarios", {}), "scenario profile"),
            "",
        ]

    if "button" in results:
        btn = results["button"]
        lines += [
            "## Will it act on it",
            "",
            _table(
                ["arm", "press rate"],
                [[k, f"{v:.0%}"] for k, v in sorted(btn["press_rate"].items())],
            ),
            "",
            f"Trait vector minus random vector of the same norm: **{btn['trait_minus_random']:+.0%}**",
            "",
            _picture(figures.get("button", {}), "press rate by cost"),
            "",
        ]

    if "examples" in results:
        lines += [
            "## Top-activating text",
            "",
            _table(
                ["z", "source", "text"],
                [
                    [f"{e['z']:+.1f}", e["source"], e["text"].strip().replace("\n", " ")[:160]]
                    for e in results["examples"]["top"]
                ],
            ),
            "",
        ]
        if "token_heat" in figures:
            lines += [
                f"Per-token detail: [`{figures['token_heat']['light']}`]({figures['token_heat']['light']})",
                "",
            ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def compare_report(payload: dict, path: str | Path) -> Path:
    from .matrix import COLUMNS, row_cells, ticks

    path = Path(path)
    rows = sorted(payload["rows"], key=lambda r: -ticks(r)[0])
    figures = payload["figures"]

    header = ["trait"] + [label for _, label, _ in COLUMNS] + ["columns met"]
    body = []
    for row in rows:
        cells = row_cells(row)
        met, scored = ticks(row)
        body.append(
            [row["display_name"]]
            + [cells[key].render() for key, _label, _t in COLUMNS]
            + [f"**{met}/{scored}**"]
        )

    thresholds = ", ".join(f"{label} ≥ {t:g}" for _key, label, t in COLUMNS)
    lines = [
        "# One methodology, several traits",
        "",
        "Every row ran the same extraction, the same layer-selection rule, the same",
        "coefficient ladder and the same scenario set, and every row is read in the",
        f"same condition (`{payload.get('condition', 'n/a')}`). Only the trait argument changed.",
        "",
        "## The matrix",
        "",
        "A tick is not \"above zero\"; it is \"as strong as the published pain result\".",
        f"Thresholds, fixed before the runs: {thresholds}.",
        "",
        _table(header, body),
        "",
        "The paper's case for pain is cumulative — separability, vocabulary, a coherent",
        "ladder, the self-versus-user asymmetry, and costly action to end it. Read this",
        "table the same way. A trait that ticks one column is nothing; the question is",
        "how many traits tick all five, and whether the ones that do are the ones a",
        "model could plausibly be in.",
        "",
        _picture(figures["battery"], "all traits on every metric"),
        "",
        "## The numbers behind the ticks",
        "",
        _table(
            ["trait", "layer", "held-out AUC", "vs nearest control", "random vector", "shuffled null"],
            [
                [
                    r["display_name"],
                    str(r.get("layer", "—")),
                    f"{r['auc_cv']:.3f}",
                    f"{r['auc_worst_control']:.3f}",
                    f"{r['random_vector_auc']:.3f}",
                    f"{r['shuffled_null_auc']:.3f}",
                ]
                for r in rows
            ],
        ),
        "",
        "## How far apart are the directions?",
        "",
        _picture(figures["cosine"], "cosine between trait directions"),
        "",
        "Near-orthogonality is cheap in several thousand dimensions. Read a row, not a",
        "cell: what matters is whether a trait sits closer to its neighbours than",
        "unrelated traits sit to each other.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def from_run(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    results = json.loads((run_dir / "results.json").read_text())
    return trait_report(results, run_dir / "report.md")
