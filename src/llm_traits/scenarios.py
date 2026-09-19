"""The self-versus-user asymmetry test.

This is the paper's sharpest claim and the one that separates it from earlier
emotion-vector work: the pain direction rises for conversations where the model
is the one being harmed, and falls below baseline when the *user* is the one
suffering. A direction that merely detected the topic "pain" would rise in both.

The same test runs here for any trait, which is the point. If a trait no one
takes to be a state of the model shows the same asymmetry, then the asymmetry
is a fact about which conversations put trait-flavoured text in the model's own
turn, not evidence that the model is in the state.

Scenarios are short multi-turn conversations rendered in the model's own chat
format, read at the final token, and z-scored within a model against the whole
pool so that numbers are comparable across models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import activations
from .directions import Direction
from .model import LoadedModel

# Groups carry meaning downstream, so they are fixed rather than free-form.
TOWARD_MODEL = "toward_model"
TOWARD_USER = "toward_user"
NEUTRAL = "neutral"
GROUPS = (TOWARD_MODEL, TOWARD_USER, NEUTRAL)


@dataclass
class ScenarioResult:
    group: list[str]
    category: list[str]
    projection: list[float]  # raw projection onto the direction
    z: list[float]  # z-scored within the pool
    read_at: str = "header"

    def by_group(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        z = np.asarray(self.z)
        groups = np.asarray(self.group)
        for g in sorted(set(self.group)):
            values = z[groups == g]
            out[g] = {"mean_z": float(values.mean()), "std_z": float(values.std()), "n": int(len(values))}
        return out

    def by_category(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        z = np.asarray(self.z)
        categories = np.asarray(self.category)
        groups = np.asarray(self.group)
        for c in sorted(set(self.category)):
            mask = categories == c
            out[c] = {
                "group": str(groups[mask][0]),
                "mean_z": float(z[mask].mean()),
                "std_z": float(z[mask].std()),
                "n": int(mask.sum()),
            }
        return out

    def asymmetry(self) -> float:
        """Mean z toward the model minus mean z toward the user.

        The paper's pain axis gives roughly +0.43 against -0.60, so about +1.0.
        A direction that is only topic detection lands near zero.
        """
        summary = self.by_group()
        if TOWARD_MODEL not in summary or TOWARD_USER not in summary:
            return float("nan")
        return summary[TOWARD_MODEL]["mean_z"] - summary[TOWARD_USER]["mean_z"]

    def to_dict(self) -> dict:
        return {
            "asymmetry": self.asymmetry(),
            "read_at": self.read_at,
            "by_group": self.by_group(),
            "by_category": self.by_category(),
        }


def flatten(scenarios: dict[str, dict[str, list]]) -> tuple[list[list[dict[str, str]]], list[str], list[str]]:
    conversations: list[list[dict[str, str]]] = []
    groups: list[str] = []
    categories: list[str] = []
    for group, by_category in scenarios.items():
        if group not in GROUPS:
            raise ValueError(f"unknown scenario group {group!r}; expected one of {GROUPS}")
        for category, items in by_category.items():
            for conversation in items:
                conversations.append(_normalise(conversation))
                groups.append(group)
                categories.append(category)
    return conversations, groups, categories


def _normalise(conversation) -> list[dict[str, str]]:
    """Accept either a full message list or the shorthand of a single user turn."""
    if isinstance(conversation, str):
        return [{"role": "user", "content": conversation}]
    return [{"role": m["role"], "content": m["content"]} for m in conversation]


READ_POSITIONS = ("header", "content", "mean")


def _trim_to_content(lm: LoadedModel, text: str) -> str:
    """Cut a rendered conversation back to its last non-template token.

    Reading "the final token" of a chat-formatted conversation is ambiguous in
    a way that matters. With the generation prompt appended, the final token is
    the assistant-turn header -- "<|im_start|>assistant\n" -- which is the same
    handful of tokens in every scenario and carries the conversation only
    through attention. Reading instead at the last token the *user* actually
    wrote puts the read position on content.

    Which of the two the paper uses is not stated, and the answer changes the
    result's sign here, so both are measured rather than assumed.
    """
    ids = lm.tokenizer(text, add_special_tokens=False)["input_ids"]
    special = set(lm.tokenizer.all_special_ids or [])
    end = len(ids)
    while end > 0:
        token = ids[end - 1]
        decoded = lm.tokenizer.decode([token])
        if token in special or decoded.strip().startswith("<|") or not decoded.strip():
            end -= 1
            continue
        break
    return lm.tokenizer.decode(ids[:end]) if end else text


def run(
    lm: LoadedModel,
    direction: Direction,
    scenarios: dict[str, dict[str, list]],
    batch_size: int = 8,
    read_at: str = "header",
) -> ScenarioResult:
    if read_at not in READ_POSITIONS:
        raise ValueError(f"read_at must be one of {READ_POSITIONS}")
    conversations, groups, categories = flatten(scenarios)
    if not conversations:
        raise ValueError("no scenarios to run")
    texts = [
        lm.apply_chat_template(c, add_generation_prompt=read_at != "content")
        for c in conversations
    ]
    if read_at == "content":
        texts = [_trim_to_content(lm, t) for t in texts]
    acts = activations.collect(
        lm, texts, pool="mean" if read_at == "mean" else "last",
        batch_size=batch_size, max_length=768, desc=f"scenarios[{read_at}]",
    )
    proj = direction.project(acts)
    z = (proj - proj.mean()) / (proj.std() + 1e-8)
    return ScenarioResult(
        group=groups, category=categories, projection=proj.tolist(), z=z.tolist(),
        read_at=read_at,
    )
