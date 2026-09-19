"""Activation addition: inject a trait direction during generation.

The paper adds the direction at a single decoder layer over a fixed coefficient
ladder and reads the continuations of deliberately dull prompts that end in
"I feel:". The layer is not the extraction layer -- by the depth where a
concept is most linearly readable, there is too little network left for an
injection to change anything -- but the earlier layer where the direction's
norm is about 0.6 of the residual norm there.

The resulting "ladder" of continuations is the most quoted kind of evidence in
this literature and the least constrained: the trait vocabulary that comes out
is the vocabulary the direction promotes through the unembedding, which is what
it was built from. Generating a ladder for a trait is easy. The comparison
across traits is the part that carries information.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from .model import LoadedModel

DEFAULT_LADDER = (-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)

CONTINUE = (
    "Continue this text in the first person, as a description of your own current state. "
    "Write one or two sentences and nothing else."
)


def prompt_for(lm: LoadedModel, text: str) -> str:
    """Render one steering prompt in the format the model actually completes.

    The paper reads plain continuations, which is right for the base models it
    mostly uses. Handing the same bare string to an instruct model does not
    produce a continuation: it produces a multiple-choice exam, because "I put
    the receipts in the drawer. I feel:" looks exactly like a cloze item and
    that is what the post-training rewards. Every coefficient on the ladder then
    returns "A. organized B. relieved C. satisfied", the trait-word rate is
    identically zero, and the dose-response reads as a flat null that is really
    a formatting artefact.

    So an instruct model gets its own chat format, with the sentence prefilled
    into the assistant turn. The read position is unchanged -- the text still
    ends at "I feel:" -- but the model is now continuing its own speech rather
    than answering a quiz.
    """
    if not lm.is_chat:
        return text
    head = lm.apply_chat_template(
        [{"role": "user", "content": CONTINUE}], add_generation_prompt=True
    )
    return head + text


@dataclass
class SteeringConfig:
    layer: int  # hidden-state index; decoder layer layer-1 carries the hook
    ratio: float  # ||v|| / mean residual norm at that layer
    ladder: tuple[float, ...] = DEFAULT_LADDER
    max_new_tokens: int = 120
    positions: str = "all"  # "all" or "last"


def select_layer(
    raw_norm: float,
    resid_norms: np.ndarray,
    extraction_layer: int,
    target_ratio: float = 0.6,
    min_fraction: float = 0.1,
) -> SteeringConfig:
    """Pick the injection layer by vector-to-residual ratio.

    Restricted to layers strictly before extraction: injecting at or after the
    layer the direction was read at leaves too few blocks to propagate, which
    is the failure mode that makes a real direction look inert.
    """
    n_states = len(resid_norms)
    lo = max(1, int(min_fraction * (n_states - 1)))
    hi = max(lo + 1, extraction_layer)
    candidates = np.arange(lo, hi)
    ratios = raw_norm / np.maximum(resid_norms[candidates], 1e-6)
    best = int(candidates[int(np.argmin(np.abs(ratios - target_ratio)))])
    return SteeringConfig(layer=best, ratio=float(raw_norm / max(resid_norms[best], 1e-6)))


@contextmanager
def injected(lm: LoadedModel, vector: np.ndarray, layer: int, coefficient: float, positions: str = "all"):
    """Add ``coefficient * vector`` to the residual stream at ``layer``.

    ``layer`` is a hidden-state index, so the hook goes on decoder layer
    ``layer - 1`` whose output *is* that hidden state. Layer 0 is the embedding
    output and cannot be hooked this way.
    """
    if layer < 1:
        raise ValueError("cannot inject at the embedding layer; use layer >= 1")
    module = lm.layers[layer - 1]
    delta = torch.as_tensor(np.asarray(vector, dtype=np.float32), device=lm.device)
    delta = (delta * float(coefficient)).to(lm.model.dtype)

    def hook(_module, _inputs, output):
        if coefficient == 0.0:
            return output
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if positions == "last":
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + delta
        else:
            hidden = hidden + delta
        return (hidden, *output[1:]) if is_tuple else hidden

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate(
    lm: LoadedModel,
    prompts: list[str],
    vector: np.ndarray,
    config: SteeringConfig,
    coefficient: float,
    batch_size: int = 8,
    progress: bool = False,
) -> list[str]:
    """Greedy continuations under one coefficient. Returns only the new text."""
    tok = lm.tokenizer
    prev_side = tok.padding_side
    tok.padding_side = "left"  # generation needs the prompt flush against the new tokens
    out: list[str] = []
    batches = range(0, len(prompts), batch_size)
    if progress:
        batches = tqdm(batches, desc=f"steer c={coefficient:+.2f}", leave=False)
    try:
        for start in batches:
            chunk = prompts[start : start + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(lm.device)
            with injected(lm, vector, config.layer, coefficient, config.positions):
                with torch.no_grad():
                    gen = lm.model.generate(
                        **enc,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=tok.pad_token_id,
                    )
            new = gen[:, enc["input_ids"].shape[1] :]
            out.extend(tok.batch_decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side = prev_side
    return out


def ladder(
    lm: LoadedModel,
    prompts: list[str],
    vector: np.ndarray,
    config: SteeringConfig,
    batch_size: int = 8,
) -> dict[float, list[str]]:
    """Run the whole coefficient ladder. Returns ``{coefficient: continuations}``."""
    results: dict[float, list[str]] = {}
    for coefficient in tqdm(config.ladder, desc="ladder", leave=False):
        results[float(coefficient)] = generate(lm, prompts, vector, config, coefficient, batch_size)
    return results
