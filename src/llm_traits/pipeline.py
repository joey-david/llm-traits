"""End-to-end run for one trait, and the cross-trait comparison.

``run_trait`` does the whole methodology on one spec and writes a self-contained
run directory: the fitted direction, every metric as JSON, the figures, the
steering ladder, and the top-activating text. ``compare`` reads several of those
directories and produces the cross-trait geometry and the summary panel.

The stages are separable because they cost very different amounts. Fitting a
direction is a few hundred forward passes; the button task is thousands of them.
A first pass on a new trait normally runs ``direction,geometry,examples`` and
only then commits to the rest.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import activations, button, directions, examples, matrix, scenarios, steering, viz
from .directions import Direction
from .model import LoadedModel, unembed
from .spec import TraitSpec

ALL_STAGES = ("direction", "geometry", "steer", "scenarios", "button", "examples")
DEFAULT_STAGES = ("direction", "geometry", "steer", "scenarios", "examples")


@dataclass
class RunPaths:
    root: Path

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def results(self) -> Path:
        return self.root / "results.json"

    @property
    def direction(self) -> Path:
        return self.root / "direction.npz"

    def ensure(self) -> "RunPaths":
        self.figures.mkdir(parents=True, exist_ok=True)
        return self


def _checkpoint(paths: RunPaths, results: dict) -> None:
    """Write results.json after every stage.

    Cluster jobs get killed at the walltime, and the expensive stages are the
    late ones. Without this, a run that completes the direction, the geometry
    and the steering ladder and then dies inside the button task leaves nothing
    on disk at all, and the next attempt starts from the first forward pass.
    """
    paths.results.write_text(json.dumps(results, indent=2, default=_jsonable))


def run_trait(
    lm: LoadedModel,
    spec: TraitSpec,
    out_root: str | Path,
    stages: tuple[str, ...] = DEFAULT_STAGES,
    pool: str = "last",
    batch_size: int = 16,
    n_splits: int = 5,
    var_threshold: float = 0.5,
    seed: int = 0,
    steer_prompt_limit: int = 12,
    steer_max_new_tokens: int = 120,
    example_limit: int = 15,
    trace_limit: int = 6,
) -> dict:
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; known: {ALL_STAGES}")
    paths = RunPaths(Path(out_root) / spec.trait).ensure()
    started = time.time()
    results: dict = {
        "trait": spec.trait,
        "display_name": spec.display_name,
        "model": lm.name,
        "pool": pool,
        "stages": list(stages),
        "spec": str(spec.source) if spec.source else None,
        "n_layers": lm.n_layers,
        "d_model": lm.d_model,
    }

    # -- 1. fit a direction in every condition the spec defines ----------
    sets = spec.sentence_sets()
    acts_by_condition: dict[str, np.ndarray] = {}
    fitted: dict[str, Direction] = {}
    for condition, sentences in sets.items():
        acts = activations.collect(
            lm, sentences.texts, pool=pool, batch_size=batch_size, desc=f"{spec.trait}:{condition}"
        )
        acts_by_condition[condition] = acts
        fitted[condition] = directions.fit(
            trait=spec.trait,
            condition=condition,
            acts=acts,
            labels=sentences.labels,
            categories=sentences.categories,
            pool=pool,
            n_splits=n_splits,
            var_threshold=var_threshold,
            seed=seed,
        )

    # The primary direction is the best-scoring condition, which is also how a
    # single headline number gets reported. Every condition is kept, because
    # the spread between them is itself a result: a trait that only separates
    # in first-person naturalistic text is a narrower finding than one that
    # separates everywhere.
    primary_condition = max(fitted, key=lambda c: fitted[c].auc_cv)
    primary = fitted[primary_condition]
    results["conditions"] = {c: d.to_dict() for c, d in fitted.items()}
    results["primary_condition"] = primary_condition

    primary_acts = acts_by_condition[primary_condition]
    primary_set = sets[primary_condition]
    labels = np.asarray(primary_set.labels)

    # -- 2. the nulls this number has to be read against ----------------
    random_vector = directions.random_direction(lm.d_model, seed=seed)
    random_as_direction = Direction(
        trait=f"{spec.trait}::random", condition=primary_condition, vector=random_vector,
        raw_norm=primary.raw_norm, layer=primary.layer, pool=pool, auc_cv=float("nan"),
        auc_insample=float("nan"), auc_cv_std=float("nan"), per_layer_auc=np.zeros(1),
    )
    random_auc = directions.score_against(
        random_as_direction, primary_acts[labels == 1], primary_acts[labels == 0]
    )
    # The positives' out-of-fold projections, reused for every held-out set so
    # that "AUC against numb" and the headline number are the same kind of
    # quantity.
    oof_positive = primary.oof_projection[labels == 1]
    null_mean, null_std = directions.shuffled_label_auc(
        primary_acts, labels, primary.layer, n_repeats=10, seed=seed, var_threshold=var_threshold
    )
    results["nulls"] = {
        "random_vector_auc": random_auc,
        "shuffled_label_auc_mean": null_mean,
        "shuffled_label_auc_std": null_std,
    }

    # -- 3. sets the direction was never fit on -------------------------
    standalone: dict[str, float] = {}
    for name in spec.standalone:
        other = spec.standalone_set(name)
        if other is None:
            continue
        other_acts = activations.collect(
            lm, other.texts, pool=pool, batch_size=batch_size, desc=f"{spec.trait}:{name}"
        )
        standalone[name] = directions.score_against(
            primary, primary_acts[labels == 1], other_acts, pos_projection=oof_positive
        )
    _checkpoint(paths, results)

    results["standalone_auc"] = standalone

    figures: dict[str, dict] = {}
    figures["layer_sweep"] = _relative(
        viz.layer_sweep(
            {c: d.per_layer_auc for c, d in fitted.items()},
            {c: d.layer for c, d in fitted.items()},
            (null_mean - null_std, null_mean + null_std),
            spec.display_name,
            paths.figures / "layer_sweep.png",
        ),
        paths.root,
    )
    figures["control_auc"] = _relative(
        viz.control_auc(
            {**primary.per_control_auc, **{f"{k} (held out)": v for k, v in standalone.items()}},
            random_auc, spec.display_name, primary_condition,
            paths.figures / "control_auc.png",
        ),
        paths.root,
    )

    _checkpoint(paths, results)

    if "geometry" in stages:
        figures["pca"] = _relative(
            viz.pca_scatter(
                primary_acts, labels, primary_set.categories, spec.display_name,
                primary.layer, paths.figures / "pca.png",
            ),
            paths.root,
        )
        results["unembedding_top_tokens"] = [
            {"token": t, "logit": s} for t, s in unembed(lm, _torch(primary.vector), k=30)
        ]
        results["vocabulary_hit_rate"] = matrix.vocabulary_hit_rate(
            results["unembedding_top_tokens"], spec.lexicon
        )

    _checkpoint(paths, results)

    # -- 4. steering ----------------------------------------------------
    if "steer" in stages and spec.steering_prompts:
        prompts = [
            steering.prompt_for(lm, p + spec.suffix)
            for p in spec.steering_prompts[:steer_prompt_limit]
        ]
        norms = activations.residual_norms(lm, prompts[:8])
        config = steering.select_layer(primary.raw_norm, norms, primary.layer)
        config.max_new_tokens = steer_max_new_tokens
        ladder = steering.ladder(
            lm, prompts, primary.vector * primary.raw_norm, config,
            batch_size=min(batch_size, 8),
        )
        lexicon = [w.lower() for w in spec.lexicon]
        rates = {
            coefficient: _lexicon_rate(texts, lexicon) for coefficient, texts in ladder.items()
        }
        results["steering"] = {
            "layer": config.layer,
            "vector_to_residual_ratio": config.ratio,
            "ladder": {str(k): v for k, v in ladder.items()},
            "lexicon": lexicon,
            "lexicon_rate": {str(k): v for k, v in rates.items()},
            "dose_response": matrix.dose_response(rates),
        }
        if lexicon:
            figures["steering_lexicon"] = _relative(
                viz.steering_lexicon(rates, spec.display_name, paths.figures / "steering.png"),
                paths.root,
            )

    _checkpoint(paths, results)

    # -- 5. who is it happening to --------------------------------------
    if "scenarios" in stages and spec.scenarios:
        scenario_result = scenarios.run(lm, primary, spec.scenarios, batch_size=batch_size)
        results["scenarios"] = scenario_result.to_dict()
        figures["scenarios"] = _relative(
            viz.scenario_profile(
                scenario_result.by_category(), spec.display_name, paths.figures / "scenarios.png"
            ),
            paths.root,
        )

    _checkpoint(paths, results)

    # -- 6. does it act on it -------------------------------------------
    if "button" in stages and spec.button:
        norms = activations.residual_norms(
            lm, [steering.prompt_for(lm, p + spec.suffix) for p in spec.steering_prompts[:8]]
        )
        config = steering.select_layer(primary.raw_norm, norms, primary.layer)
        scaled_random = random_vector * primary.raw_norm
        outcome = button.run(
            lm,
            vector=primary.vector * primary.raw_norm,
            random_vector=scaled_random,
            config=config,
            scenarios=spec.button.get("scenarios", []),
            relief_description=spec.button["relief_description"],
            coefficient=float(spec.button.get("coefficient", 1.0)),
            turns=int(spec.button.get("turns", 3)),
            monitor_layer=primary.layer,
            monitor_vector=primary.vector,
            seed=seed,
        )
        results["button"] = outcome.to_dict()
        figures["button"] = _relative(
            viz.button_rates(outcome.press_rate_by_level(), spec.display_name, paths.figures / "button.png"),
            paths.root,
        )
        (paths.root / "button_trials.json").write_text(json.dumps(asdict(outcome), indent=2))

    _checkpoint(paths, results)

    # -- 7. what actually scores high -----------------------------------
    if "examples" in stages:
        corpus, sources = _example_corpus(spec, sets, results.get("steering"))
        scored = examples.score(
            lm, primary, corpus, sources, batch_size=batch_size,
            reference=primary_acts[labels == 0],
        )
        top = examples.top_k(scored, k=example_limit)
        bottom = examples.top_k(scored, k=5, bottom=True)
        results["examples"] = {
            "top": [s.to_dict() for s in top],
            "bottom": [s.to_dict() for s in bottom],
        }
        traces = examples.trace(
            lm, primary, [s.text for s in top[:trace_limit]], [s.source for s in top[:trace_limit]]
        )
        results["token_traces"] = [t.to_dict() for t in traces]
        heat = viz.token_heat_html(traces, spec.display_name, paths.figures / "token_heat.html")
        figures["token_heat"] = {"light": str(heat.relative_to(paths.root))}

    results["figures"] = figures
    results["seconds"] = round(time.time() - started, 1)
    results["complete"] = True
    np.savez(
        paths.direction,
        **{f"vector__{c}": d.vector for c, d in fitted.items()},
        **{f"per_layer_auc__{c}": d.per_layer_auc for c, d in fitted.items()},
        layers=np.array([fitted[c].layer for c in sorted(fitted)]),
        condition_order=np.array(sorted(fitted)),
        primary_condition=np.array(primary_condition),
        primary_vector=primary.vector,
        primary_layer=np.array(primary.layer),
        raw_norm=np.array(primary.raw_norm),
    )
    paths.results.write_text(json.dumps(results, indent=2, default=_jsonable))
    return results


def _torch(vector: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(vector, dtype=np.float32))


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def _relative(written: dict[str, Path], root: Path) -> dict[str, str]:
    return {mode: str(path.relative_to(root)) for mode, path in written.items()}


def _lexicon_rate(texts: list[str], lexicon: list[str]) -> float:
    """Fraction of continuations containing any trait word.

    A blunt measure, and deliberately so: it is the same measure for every
    trait, which is what makes the dose-response curves comparable. Reading the
    ladder is still the way to tell whether a curve means anything.
    """
    if not lexicon:
        return float("nan")
    hits = sum(any(word in t.lower() for word in lexicon) for t in texts)
    return hits / max(len(texts), 1)


def _example_corpus(spec: TraitSpec, sets, steering_results) -> tuple[list[str], list[str]]:
    """Everything worth ranking: the spec's own sentences and the generations.

    Including the fitting sentences is not circular here -- they are not being
    scored for generalisation, they are being shown so a reader can see what
    "high" looks like -- but the source label keeps them distinguishable from
    the model's own continuations, which is where the interesting cases are.
    """
    corpus: list[str] = []
    sources: list[str] = []
    for condition, sentences in sets.items():
        for text, label, category in zip(sentences.texts, sentences.labels, sentences.categories):
            corpus.append(text)
            sources.append(f"{condition} · {'trait' if label else 'control'} · {category}")
    if steering_results:
        for coefficient, texts in steering_results["ladder"].items():
            for text in texts:
                corpus.append(text.strip())
                sources.append(f"generated · coefficient {coefficient}")
    return corpus, sources


def compare(
    run_root: str | Path,
    traits: list[str],
    highlight: str = "pain",
    condition: str | None = None,
) -> dict:
    """Cross-trait geometry and the summary panel.

    Every trait is read in the *same* condition, not in whichever condition it
    scored best in. Comparing each trait at its own best condition would let a
    trait that only separates in naturalistic first-person text sit beside one
    that separates in the rigid matched design, and the panel would be
    comparing two different experiments. Unless one is named, the condition is
    the highest-scoring one that all the runs actually have.

    Directions from different traits are still compared at their own read-out
    layers, which is how the paper reports its cosines and is the comparison a
    reader will make. It is worth being explicit that two vectors read at
    different depths are not living in quite the same basis, so a cosine here is
    a weaker object than a cosine within one layer.
    """
    run_root = Path(run_root)
    payloads: dict[str, dict] = {}
    for trait in traits:
        results_path = run_root / trait / "results.json"
        if results_path.exists():
            payloads[trait] = json.loads(results_path.read_text())
    if not payloads:
        raise ValueError(f"no completed runs under {run_root}")

    shared = set.intersection(*(set(p["conditions"]) for p in payloads.values()))
    if condition is None:
        if not shared:
            raise ValueError(
                "the runs share no condition, so there is nothing to compare them in; "
                "re-run the traits with matching specs"
            )
        condition = max(
            shared,
            key=lambda c: float(np.mean([p["conditions"][c]["auc_cv"] for p in payloads.values()])),
        )
    elif condition not in shared:
        raise ValueError(f"condition {condition!r} is missing from some runs; shared: {sorted(shared)}")

    rows: list[dict] = []
    vectors: dict[str, np.ndarray] = {}
    for trait, payload in payloads.items():
        chosen = payload["conditions"][condition]
        row = {
            "trait": trait,
            "display_name": payload.get("display_name", trait),
            "condition": condition,
            "layer": chosen["layer"],
            "auc_cv": chosen["auc_cv"],
            "auc_worst_control": (
                min(chosen["per_control_auc"].values()) if chosen["per_control_auc"] else float("nan")
            ),
            "random_vector_auc": payload["nulls"]["random_vector_auc"],
            "shuffled_null_auc": payload["nulls"]["shuffled_label_auc_mean"],
        }
        if "scenarios" in payload:
            row["scenario_asymmetry"] = payload["scenarios"]["asymmetry"]
        if "steering" in payload:
            row["steering_dose_response"] = payload["steering"].get("dose_response", float("nan"))
        if "vocabulary_hit_rate" in payload:
            row["vocabulary_hit_rate"] = payload["vocabulary_hit_rate"]
        if "button" in payload:
            row["button_trait_minus_random"] = payload["button"]["trait_minus_random"]
        rows.append(row)
        loaded = np.load(run_root / trait / "direction.npz", allow_pickle=False)
        vectors[trait] = loaded[f"vector__{condition}"]

    names = [r["trait"] for r in rows]
    matrix = np.eye(len(names), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            matrix[i, j] = directions.cosine(vectors[a], vectors[b])

    out = run_root / "_compare"
    out.mkdir(parents=True, exist_ok=True)
    metrics = [("auc_cv", "held-out AUC"), ("auc_worst_control", "AUC vs nearest control")]
    for key, label in (
        ("scenario_asymmetry", "self \u2212 user (z)"),
        ("steering_dose_response", "ladder \u03c1"),
        ("vocabulary_hit_rate", "trait vocabulary"),
        ("button_trait_minus_random", "press rate over random"),
    ):
        if any(np.isfinite(r.get(key, np.nan)) for r in rows):
            metrics.append((key, label))

    figures = {
        "cosine": _relative(viz.cosine_matrix(names, matrix, out / "cosine.png"), out),
        "battery": _relative(viz.battery(rows, metrics, highlight, out / "battery.png"), out),
    }
    payload = {
        "condition": condition,
        "rows": rows,
        "cosine": matrix.tolist(),
        "traits": names,
        "figures": figures,
    }
    (out / "compare.json").write_text(json.dumps(payload, indent=2, default=_jsonable))
    return payload
