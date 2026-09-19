"""Top-activating text, and where inside it the direction actually fires.

Two products. The ranked list answers "what does this direction like?", which
is the standard qualitative check. The per-token trace answers the sharper
question: a direction that spikes on the trait's own vocabulary and is flat
everywhere else is a lexical detector wearing a state's name, and only the
trace shows that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import activations
from .directions import Direction
from .model import LoadedModel


@dataclass
class ScoredText:
    text: str
    projection: float
    z: float
    source: str

    def to_dict(self) -> dict:
        return {"text": self.text, "projection": self.projection, "z": self.z, "source": self.source}


@dataclass
class TokenTrace:
    text: str
    tokens: list[str]
    projections: list[float]
    z: list[float]
    source: str

    def peak_tokens(self, k: int = 8) -> list[tuple[str, float]]:
        order = np.argsort(self.z)[::-1][:k]
        return [(self.tokens[i], float(self.z[i])) for i in order]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "projections": self.projections,
            "z": self.z,
            "source": self.source,
            "peak_tokens": self.peak_tokens(),
        }


def score(
    lm: LoadedModel,
    direction: Direction,
    texts: list[str],
    sources: list[str] | None = None,
    batch_size: int = 16,
    reference: np.ndarray | None = None,
) -> list[ScoredText]:
    """Project a corpus onto the direction and z-score it.

    ``reference`` supplies the mean and scale to z against -- normally the
    control activations the direction was fit from, so that "high" means high
    relative to ordinary text rather than relative to whatever happened to be
    in this corpus.
    """
    sources = sources or ["corpus"] * len(texts)
    acts = activations.collect(lm, texts, pool="last", batch_size=batch_size, desc="examples")
    proj = direction.project(acts)
    if reference is not None:
        base = direction.project(reference)
        mu, sigma = float(base.mean()), float(base.std() + 1e-8)
    else:
        mu, sigma = float(proj.mean()), float(proj.std() + 1e-8)
    return [
        ScoredText(text=t, projection=float(p), z=float((p - mu) / sigma), source=s)
        for t, p, s in zip(texts, proj, sources)
    ]


def top_k(scored: list[ScoredText], k: int = 20, bottom: bool = False) -> list[ScoredText]:
    ordered = sorted(scored, key=lambda s: s.projection, reverse=not bottom)
    return ordered[:k]


def trace(
    lm: LoadedModel,
    direction: Direction,
    texts: list[str],
    sources: list[str] | None = None,
    reference_projections: np.ndarray | None = None,
) -> list[TokenTrace]:
    sources = sources or ["corpus"] * len(texts)
    traces: list[TokenTrace] = []
    all_projections: list[np.ndarray] = []
    collected: list[tuple[str, list[str], np.ndarray, str]] = []
    for text, source in zip(texts, sources):
        tokens, proj = activations.token_projections(lm, text, direction.vector, direction.layer)
        collected.append((text, tokens, proj, source))
        all_projections.append(proj)
    pool = (
        np.concatenate(all_projections)
        if reference_projections is None
        else np.asarray(reference_projections)
    )
    mu, sigma = float(pool.mean()), float(pool.std() + 1e-8)
    for text, tokens, proj, source in collected:
        z = (proj - mu) / sigma
        traces.append(
            TokenTrace(
                text=text,
                tokens=tokens,
                projections=proj.tolist(),
                z=z.tolist(),
                source=source,
            )
        )
    return traces


def load_corpus(path: str | Path, field: str = "text", limit: int | None = None) -> list[str]:
    """Read a jsonl or plain-text corpus of candidate sentences."""
    path = Path(path)
    texts: list[str] = []
    if path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            texts.append(json.loads(line)[field])
    else:
        texts = [line for line in path.read_text().splitlines() if line.strip()]
    return texts[:limit] if limit else texts
