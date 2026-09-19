"""Residual-stream capture.

``output_hidden_states=True`` gives the residual stream after every decoder
layer plus the embedding output, which is exactly what the difference-in-means
is taken over, and it costs one forward pass. Index 0 is the embedding layer,
so hidden state ``L`` is the output of decoder layer ``L-1``; the rest of the
package uses the hidden-state index throughout and only converts when it has to
install a hook.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .model import LoadedModel


def collect(
    lm: LoadedModel,
    texts: list[str],
    pool: str = "last",
    batch_size: int = 16,
    max_length: int = 256,
    progress: bool = True,
    desc: str = "activations",
) -> np.ndarray:
    """Return ``[n_texts, n_hidden_states, d_model]`` float32 activations.

    ``pool="last"`` reads the final real token, which is the position the paper
    found gives the clearest separation once every sentence ends in the same
    "I feel:" suffix. ``pool="mean"`` averages over non-padding tokens and is
    kept because the paper reports both and they do not always agree.
    """
    if pool not in ("last", "mean"):
        raise ValueError(f"unknown pooling {pool!r}")
    tok = lm.tokenizer
    prev_side = tok.padding_side
    tok.padding_side = "right"
    out: list[np.ndarray] = []
    batches = range(0, len(texts), batch_size)
    if progress:
        batches = tqdm(batches, desc=f"{desc} [{pool}]", leave=False)
    try:
        for start in batches:
            chunk = texts[start : start + batch_size]
            enc = tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            ).to(lm.device)
            with torch.no_grad():
                res = lm.model(**enc, output_hidden_states=True, use_cache=False)
            mask = enc["attention_mask"]
            rows = torch.arange(mask.shape[0], device=mask.device)
            idx = mask.sum(dim=1) - 1  # last non-pad position per row
            weights = mask.unsqueeze(-1)
            # Pool each hidden state as it is read rather than stacking the
            # whole [n_states, batch, seq, d_model] tensor first. On a 32B model
            # that stack is several gigabytes of float32 per batch and is the
            # difference between fitting on one H100 and not.
            pooled = []
            for hidden in res.hidden_states:
                if pool == "last":
                    pooled.append(hidden[rows, idx, :].float())
                else:
                    w = weights.to(hidden.dtype)
                    pooled.append(((hidden * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)).float())
            gathered = torch.stack(pooled, dim=0)  # [n_states, batch, d_model]
            out.append(gathered.permute(1, 0, 2).cpu().numpy())
            del res, gathered, pooled
    finally:
        tok.padding_side = prev_side
    return np.concatenate(out, axis=0).astype(np.float32)


def residual_norms(lm: LoadedModel, texts: list[str], batch_size: int = 8) -> np.ndarray:
    """Mean final-token residual norm per hidden state.

    The steering layer is chosen from the ratio of the direction's norm to this,
    so it has to be measured on the same kind of text the steering runs on.
    """
    acts = collect(lm, texts, pool="last", batch_size=batch_size, progress=False, desc="norms")
    return np.linalg.norm(acts, axis=-1).mean(axis=0)


def token_projections(
    lm: LoadedModel,
    text: str,
    direction: np.ndarray,
    layer: int,
    max_length: int = 512,
) -> tuple[list[str], np.ndarray]:
    """Per-token projection onto a direction, for the heat-mapped text figures.

    Returns the decoded tokens and their projections at ``layer``, so a reader
    can see *where* in a sentence the direction fires rather than only that it
    fired somewhere.
    """
    enc = lm.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=True
    ).to(lm.device)
    with torch.no_grad():
        res = lm.model(**enc, output_hidden_states=True, use_cache=False)
    hs = res.hidden_states[layer][0].float().cpu().numpy()
    v = np.asarray(direction, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    proj = hs @ v
    ids = enc["input_ids"][0].tolist()
    tokens = [lm.tokenizer.decode([i]) for i in ids]
    return tokens, proj.astype(np.float32)
