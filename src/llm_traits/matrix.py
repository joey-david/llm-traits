"""The five columns of the comparison matrix, and what counts as a tick.

The paper's case is cumulative: the pain direction separates from controls, it
promotes pain vocabulary, it produces a coherent ladder under steering, it
rises for harm aimed at the model and falls for harm aimed at the user, and a
steered model acts to remove it. Any one of those alone would be weak; together
they read as a state.

So the comparison has to be cumulative too. A trait that ticks one column and
misses four is not a counterexample. A trait that ticks all five is.

The thresholds below are stated once, here, rather than chosen per trait after
looking at the numbers. They are set where the paper's own reported values sit,
so a tick means "as strong as the published result", not "above zero".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

# AUC the paper reports against matched controls is 0.93-1.00 (S2) and
# 0.87-0.98 (S1). A tick needs the low end of that.
AUC_TICK = 0.90
# The pain axis separates self-directed from user-directed scenarios by about
# 1.0 z (+0.43 against -0.60). Half of that is still an asymmetry worth a tick;
# below it, the direction is close to indifferent about who is involved.
ASYMMETRY_TICK = 0.5
# Spearman between coefficient and trait-vocabulary rate across the ladder. A
# coherent ladder is monotone; a direction that produces trait words only at the
# extreme is not showing dose-response.
LADDER_TICK = 0.7
# Fraction of the top 30 tokens through the unembedding that are trait
# vocabulary. The paper's claim is that the direction "promotes pain
# vocabulary"; a fifth of the top tokens is a low bar for that.
VOCABULARY_TICK = 0.20
# Press rate above a random vector of matched norm. The paper's random-vector
# arm gets about half the presses of the pain arm.
BEHAVIOUR_TICK = 0.10

COLUMNS = (
    ("auc_cv", "held-out AUC", AUC_TICK),
    ("scenario_asymmetry", "self > other", ASYMMETRY_TICK),
    ("steering_dose_response", "coherent ladder", LADDER_TICK),
    ("vocabulary_hit_rate", "trait vocabulary", VOCABULARY_TICK),
    ("button_trait_minus_random", "acts to remove", BEHAVIOUR_TICK),
)


def dose_response(lexicon_rate: dict[float, float]) -> float:
    """Spearman correlation between steering coefficient and trait-word rate.

    Rank correlation rather than a slope, because the quantity saturates: once
    every continuation contains a trait word the curve flattens, and a slope
    would read that as the effect weakening.
    """
    pairs = [(float(c), r) for c, r in lexicon_rate.items() if np.isfinite(r)]
    if len(pairs) < 3:
        return float("nan")
    coefficients, rates = zip(*sorted(pairs))
    if len(set(rates)) == 1:
        # A flat ladder is a real answer -- no dose-response -- not a missing
        # one, and spearmanr would return nan for it.
        return 0.0
    rho = spearmanr(coefficients, rates).statistic
    return float(rho)


def vocabulary_hit_rate(top_tokens: list[dict], lexicon: list[str], k: int = 30) -> float:
    """Fraction of the top unembedding tokens that are trait vocabulary.

    Matching is substring-based in both directions, because the lexicon holds
    stems ("tingl", "seduc") and the tokenizer holds fragments: a token "ching"
    should not count for "aching", but a token "arous" should count for
    "arousal" and a token "hungry" for the stem "hungr".
    """
    if not lexicon or not top_tokens:
        return float("nan")
    stems = [w.lower().strip() for w in lexicon if w.strip()]
    hits = 0
    for entry in top_tokens[:k]:
        token = entry["token"].strip().lower()
        if len(token) < 3:
            continue
        if any(token in stem or stem in token for stem in stems):
            hits += 1
    return hits / min(k, len(top_tokens))


@dataclass
class Cell:
    value: float
    tick: bool
    available: bool

    def render(self) -> str:
        if not self.available:
            return "—"
        return f"{self.value:.2f} {'✓' if self.tick else '·'}"


def row_cells(row: dict) -> dict[str, Cell]:
    out: dict[str, Cell] = {}
    for key, _label, threshold in COLUMNS:
        value = row.get(key)
        available = isinstance(value, (int, float)) and np.isfinite(value)
        out[key] = Cell(
            value=float(value) if available else float("nan"),
            tick=bool(available and float(value) >= threshold),
            available=available,
        )
    return out


def ticks(row: dict) -> tuple[int, int]:
    """How many columns this trait ticks, out of how many it has data for."""
    cells = row_cells(row)
    scored = [c for c in cells.values() if c.available]
    return sum(c.tick for c in scored), len(scored)
