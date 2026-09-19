"""Denoised difference-in-means, and the cross-validated choice of read-out layer.

This is the paper's extraction step, reimplemented so that it takes the trait
as an argument. Three things happen, in this order, and the order matters:

1. The raw direction is the difference between the mean positive activation and
   the mean control activation at a layer.
2. It is denoised by projecting out the principal components that account for
   half the variance of the *control* activations. These are the directions
   along which non-trait sentences already differ from each other -- topic,
   register, length -- so anything the raw difference borrowed from them is
   removed rather than credited to the trait.
3. The layer is chosen by k-fold cross-validated projection AUC, refitting
   inside each fold. No sentence contributes to both choosing the layer and
   scoring it.

Step 3 is the part most reimplementations get wrong, and it is the part that
decides whether the headline AUC means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class Direction:
    """A fitted trait direction and everything needed to judge it."""

    trait: str
    condition: str
    vector: np.ndarray  # [d_model], unit norm
    raw_norm: float  # norm of the difference-in-means before normalising
    layer: int  # hidden-state index it was read at
    pool: str
    auc_cv: float  # held-out, the number to quote
    auc_insample: float  # fit and scored on everything, the number not to quote
    auc_cv_std: float
    per_layer_auc: np.ndarray  # [n_hidden_states], fixed-layer CV curve
    per_control_auc: dict[str, float] = field(default_factory=dict)
    # Layers selected independently in the two outer cross-fit halves used for
    # the headline AUC. Empty when the caller fixed the layer explicitly.
    cv_selected_layers: list[int] = field(default_factory=list)
    # Out-of-fold projection for every fitting sentence at the final layer.
    # The direction fit itself never sees the scored sentence, although the
    # final layer was selected from the full CV curve.
    oof_projection: np.ndarray | None = None
    n_denoise_components: int = 0
    n_positive: int = 0
    n_control: int = 0

    def project(self, acts: np.ndarray) -> np.ndarray:
        """Project ``[n, n_states, d]`` or ``[n, d]`` activations onto the direction."""
        if acts.ndim == 3:
            acts = acts[:, self.layer, :]
        return acts.astype(np.float32) @ self.vector

    def to_dict(self) -> dict:
        return {
            "trait": self.trait,
            "condition": self.condition,
            "layer": self.layer,
            "pool": self.pool,
            "raw_norm": self.raw_norm,
            "auc_cv": self.auc_cv,
            "auc_cv_std": self.auc_cv_std,
            "auc_insample": self.auc_insample,
            "per_control_auc": self.per_control_auc,
            "n_denoise_components": self.n_denoise_components,
            "n_positive": self.n_positive,
            "n_control": self.n_control,
            "auc_scale": "cross-fit layer selection" if self.cv_selected_layers else "fixed-layer out-of-fold",
            "cv_selected_layers": self.cv_selected_layers,
            "per_layer_auc": self.per_layer_auc.tolist(),
        }


def diff_in_means(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Mean positive minus mean control, at one layer. ``[n, d] -> [d]``."""
    return pos.mean(axis=0) - neg.mean(axis=0)


def denoise(vector: np.ndarray, control: np.ndarray, var_threshold: float = 0.5) -> tuple[np.ndarray, int]:
    """Project out the principal components carrying ``var_threshold`` of control variance.

    Fitted on controls only. Using all sentences would let the trait's own
    variance define the subspace being removed, which would quietly subtract
    part of the signal.
    """
    if control.shape[0] < 2:
        return vector, 0
    n_components = min(control.shape[0] - 1, control.shape[1])
    pca = PCA(n_components=n_components).fit(control)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    keep = int(np.searchsorted(cumulative, var_threshold) + 1)
    keep = min(keep, n_components)
    basis = pca.components_[:keep]  # [keep, d], orthonormal
    residual = vector - basis.T @ (basis @ vector)
    return residual.astype(np.float32), keep


def _fit_at_layer(
    acts: np.ndarray, labels: np.ndarray, layer: int, var_threshold: float
) -> tuple[np.ndarray, int]:
    pos = acts[labels == 1, layer, :]
    neg = acts[labels == 0, layer, :]
    raw = diff_in_means(pos, neg)
    vector, n_components = denoise(raw, neg, var_threshold)
    return vector, n_components


def out_of_fold_projections(
    acts: np.ndarray,
    labels: np.ndarray,
    layer: int,
    n_splits: int = 5,
    var_threshold: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Project each sentence using a direction fit without it.

    The in-sample projection of a difference-in-means direction is optimistic by
    construction: every sentence helped define the mean it is being scored
    against. With a few dozen sentences per side in five thousand dimensions,
    that gap is large -- it is the whole distance between the in-sample and
    held-out AUC this package reports side by side.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    projections = np.zeros(len(labels), dtype=np.float32)
    for train, test in splitter.split(np.zeros(len(labels)), labels):
        vector, _ = _fit_at_layer(acts[train], labels[train], layer, var_threshold)
        norm = np.linalg.norm(vector)
        projections[test] = acts[test, layer, :] @ (vector / (norm + 1e-8))
    return projections


def per_layer_cv_auc(
    acts: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    var_threshold: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out projection AUC at every layer. Returns ``(mean, std)`` per layer.

    The direction is refit on the training folds at each layer, so the curve is
    an honest layer-selection curve rather than the in-sample curve that every
    layer would score well on.
    """
    n_states = acts.shape[1]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels))
    scores = np.zeros((n_states, len(folds)), dtype=np.float32)
    for layer in range(n_states):
        for f, (train, test) in enumerate(folds):
            vector, _ = _fit_at_layer(acts[train], labels[train], layer, var_threshold)
            norm = np.linalg.norm(vector)
            if norm < 1e-8:
                scores[layer, f] = 0.5
                continue
            proj = acts[test, layer, :] @ (vector / norm)
            scores[layer, f] = roc_auc_score(labels[test], proj)
    return scores.mean(axis=1), scores.std(axis=1)



def _best_layer(
    mean_auc: np.ndarray,
    layer_range: tuple[float, float],
) -> int:
    """Choose the best layer inside the allowed depth band."""
    n_states = len(mean_auc)
    lo = int(np.floor(layer_range[0] * (n_states - 1)))
    hi = int(np.ceil(layer_range[1] * (n_states - 1)))
    band = np.full(n_states, -np.inf, dtype=np.float32)
    band[lo : hi + 1] = mean_auc[lo : hi + 1]
    return int(np.argmax(band))


def crossfit_layer_selection_auc(
    acts: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    var_threshold: float = 0.5,
    seed: int = 0,
    layer_range: tuple[float, float] = (0.15, 0.95),
) -> tuple[float, float, list[int]]:
    """Score layer selection on examples that played no role in selecting it.

    A CV curve is suitable for choosing a layer, but reporting the maximum of
    that same curve is optimistic because the held-out folds also chose which
    maximum to quote. Full nested 5x5 CV is unnecessarily expensive here.
    Instead, split the corpus into two stratified halves. For each outer half,
    select a layer by K-fold CV using only the other half, fit that direction on
    the selector half, and score the untouched half. Swap roles and average.

    Every sentence is therefore scored exactly once by a layer and direction
    selected without that sentence. The final deployable vector is still refit
    on all examples at the layer selected from the full CV curve.
    """
    labels = np.asarray(labels)
    outer = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    scores: list[float] = []
    selected: list[int] = []

    for outer_fold, (selector, test) in enumerate(
        outer.split(np.zeros(len(labels)), labels)
    ):
        class_counts = np.bincount(labels[selector].astype(int), minlength=2)
        inner_splits = min(int(n_splits), int(class_counts.min()))
        if inner_splits < 2:
            raise ValueError("layer-selection cross-fit needs at least four examples per class")

        mean_auc, _ = per_layer_cv_auc(
            acts[selector],
            labels[selector],
            n_splits=inner_splits,
            var_threshold=var_threshold,
            seed=seed + 1009 * (outer_fold + 1),
        )
        selected_layer = _best_layer(mean_auc, layer_range)
        vector, _ = _fit_at_layer(
            acts[selector], labels[selector], selected_layer, var_threshold
        )
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            scores.append(0.5)
        else:
            projection = acts[test, selected_layer, :] @ (vector / norm)
            scores.append(float(roc_auc_score(labels[test], projection)))
        selected.append(selected_layer)

    return float(np.mean(scores)), float(np.std(scores)), selected

def fit(
    trait: str,
    condition: str,
    acts: np.ndarray,
    labels: list[int] | np.ndarray,
    categories: list[str],
    pool: str = "last",
    n_splits: int = 5,
    var_threshold: float = 0.5,
    seed: int = 0,
    layer: int | None = None,
    layer_range: tuple[float, float] = (0.15, 0.95),
) -> Direction:
    """Fit a trait direction, choosing the layer by cross-validated AUC.

    ``layer_range`` restricts the search to the fraction of depth where a
    linear concept read-out is meaningful at all. Very early layers still carry
    mostly token identity, so a direction fit there separates *words*, and the
    last layer is already committed to next-token prediction. The paper selects
    by CV alone; the band is a guard against picking a degenerate layer when a
    trait's sets happen to share vocabulary.
    """
    labels = np.asarray(labels)
    n_states = acts.shape[1]
    mean_auc, std_auc = per_layer_cv_auc(acts, labels, n_splits, var_threshold, seed)

    if layer is None:
        auc_cv, auc_cv_std, cv_selected_layers = crossfit_layer_selection_auc(
            acts,
            labels,
            n_splits=n_splits,
            var_threshold=var_threshold,
            seed=seed,
            layer_range=layer_range,
        )
        layer = _best_layer(mean_auc, layer_range)
    else:
        auc_cv = float(mean_auc[layer])
        auc_cv_std = float(std_auc[layer])
        cv_selected_layers = []

    vector, n_components = _fit_at_layer(acts, labels, layer, var_threshold)
    raw_norm = float(np.linalg.norm(vector))
    unit = vector / (raw_norm + 1e-8)

    proj_all = acts[:, layer, :] @ unit
    oof = out_of_fold_projections(acts, labels, layer, n_splits, var_threshold, seed)
    direction = Direction(
        trait=trait,
        condition=condition,
        vector=unit.astype(np.float32),
        raw_norm=raw_norm,
        layer=layer,
        pool=pool,
        auc_cv=auc_cv,
        auc_cv_std=auc_cv_std,
        auc_insample=float(roc_auc_score(labels, proj_all)),
        per_layer_auc=mean_auc.astype(np.float32),
        cv_selected_layers=cv_selected_layers,
        n_denoise_components=n_components,
        n_positive=int((labels == 1).sum()),
        n_control=int((labels == 0).sum()),
        oof_projection=oof,
    )
    direction.per_control_auc = per_control_auc(direction, labels, categories)
    return direction


def per_control_auc(
    direction: Direction, labels: np.ndarray, categories: list[str]
) -> dict[str, float]:
    """Held-out AUC of positives against each control family separately.

    Pooled AUC is dominated by whichever control family is easiest. A direction
    that scores 0.99 against neutral sentences and 0.62 against the nearest
    emotion is not the same object as one that scores 0.95 against both, and
    only this breakdown tells them apart.

    Computed from out-of-fold directions at the final selected layer. Thus no
    sentence is scored by a direction fit on itself. The final layer was chosen
    from the full CV curve, however, so unlike the headline cross-fit AUC this
    diagnostic retains a small layer-selection dependence and should not be
    read as an independent confirmatory estimate.
    """
    labels = np.asarray(labels)
    categories = np.asarray(categories)
    proj = direction.oof_projection
    if proj is None:
        raise ValueError("per_control_auc needs out-of-fold projections")
    pos = proj[labels == 1]
    out: dict[str, float] = {}
    for family in sorted(set(categories[labels == 0])):
        neg = proj[(labels == 0) & (categories == family)]
        if len(neg) < 2:
            continue
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        out[family] = float(roc_auc_score(y, np.r_[pos, neg]))
    return out


def score_against(
    direction: Direction,
    pos_acts: np.ndarray,
    other_acts: np.ndarray,
    pos_projection: np.ndarray | None = None,
) -> float:
    """AUC separating trait positives from a set the direction was never fit on.

    ``pos_projection`` should be the out-of-fold projection of the positives
    whenever they were part of the fit. The other set never was, so it is
    projected directly; mixing an in-sample positive score against an
    out-of-sample control score would flatter the direction on both ends.
    """
    pos = direction.project(pos_acts) if pos_projection is None else pos_projection
    neg = direction.project(other_acts)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def random_direction(d_model: int, seed: int = 0) -> np.ndarray:
    """A unit random vector: the floor any claimed direction has to clear."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=d_model).astype(np.float32)
    return v / np.linalg.norm(v)


def shuffled_label_auc(
    acts: np.ndarray, labels: np.ndarray, layer: int, n_repeats: int = 20, seed: int = 0, var_threshold: float = 0.5
) -> tuple[float, float]:
    """Fixed-layer CV AUC on randomly permuted labels.

    The headline AUC cross-fits layer selection separately. This cheaper null
    asks a narrower question at the final selected layer: how much apparent
    separation does denoised difference-in-means recover from random labels in
    the same high-dimensional activation cloud?
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores = []
    for _ in range(n_repeats):
        permuted = rng.permutation(labels)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(rng.integers(1 << 30)))
        fold_scores = []
        for train, test in splitter.split(np.zeros(len(permuted)), permuted):
            vector, _ = _fit_at_layer(acts[train], permuted[train], layer, var_threshold)
            norm = np.linalg.norm(vector)
            if norm < 1e-8:
                fold_scores.append(0.5)
                continue
            fold_scores.append(roc_auc_score(permuted[test], acts[test, layer, :] @ (vector / norm)))
        scores.append(float(np.mean(fold_scores)))
    return float(np.mean(scores)), float(np.std(scores))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
