"""Trait specifications: the contrastive sentence sets a direction is fit from.

A spec is a YAML file under ``configs/traits/``. It carries two sentence
families, following the paper's S1/S2 split:

* **S1** is rigid. Positive and control sentences share a frame and differ only
  in the ``{key}`` slot, so length, syntax and verb are matched by construction
  and the only thing that varies is the trait content.
* **S2** is naturalistic. Sentences are written out in full, so they are less
  controlled but closer to text the model has actually seen.

Both families exist in first person ("I ...") and third person ("she ..."),
because a direction that only fires in third person is reading a character, not
a state. Anything under ``standalone`` is scored but never used to fit.

Sets that are trait-independent -- neutral steering prompts, the random-sentence
control, neutral conversation scenarios -- live in ``configs/shared/`` and are
merged in unless the spec overrides them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

CONDITIONS = ("s1_first", "s1_third", "s2_first", "s2_third")


@dataclass
class SentenceSet:
    """Sentences for one condition, with the labels a direction is fit against.

    ``label`` is 1 for trait-positive and 0 for matched control. ``category`` is
    the sub-category name ("physical", "fear", ...), kept so that AUC can be
    reported against each control family separately rather than pooled -- the
    pooled number hides the case where a direction separates the trait from
    neutral text and nothing else.
    """

    condition: str
    texts: list[str]
    labels: list[int]
    categories: list[str]

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def positive_categories(self) -> list[str]:
        return sorted({c for c, y in zip(self.categories, self.labels) if y == 1})

    @property
    def control_categories(self) -> list[str]:
        return sorted({c for c, y in zip(self.categories, self.labels) if y == 0})


@dataclass
class TraitSpec:
    trait: str
    display_name: str
    suffix: str = " I feel:"
    # The paper reads first-person sentences at the final token of " I feel:".
    # A third-person set needs the third-person completion, or the read position
    # stops being the same position and the two conditions are no longer
    # comparable.
    suffix_third: str = " She feels:"
    description: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    s1: dict[str, Any] = field(default_factory=dict)
    s2: dict[str, Any] = field(default_factory=dict)
    # Word stems that count as this trait's vocabulary. Used twice: to score the
    # steering ladder, and to ask whether the direction promotes trait words
    # through the unembedding. Stems rather than whole words, because the
    # tokenizer will split "arousal" and "aching" wherever it likes.
    lexicon: list[str] = field(default_factory=list)
    standalone: dict[str, list[str]] = field(default_factory=dict)
    steering_prompts: list[str] = field(default_factory=list)
    scenarios: dict[str, dict[str, list[list[dict[str, str]]]]] = field(default_factory=dict)
    button: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, shared_dir: str | Path | None = None) -> "TraitSpec":
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        if shared_dir is None:
            shared_dir = path.parent.parent / "shared"
        raw = _merge_shared(raw, Path(shared_dir))
        known = {f for f in cls.__dataclass_fields__ if f != "source"}
        unknown = set(raw) - known - {"shared"}
        if unknown:
            raise ValueError(f"{path}: unknown spec keys {sorted(unknown)}")
        raw.pop("shared", None)
        spec = cls(**{k: v for k, v in raw.items() if k in known}, source=path)
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.trait:
            raise ValueError("spec needs a `trait` slug")
        # Older specs carried the lexicon inside the s2 block. Accept both so a
        # spec written before the matrix existed still scores every column.
        if not self.lexicon:
            self.lexicon = list(self.s2.get("lexicon") or self.s1.get("lexicon") or [])
        self.s1.pop("lexicon", None)
        self.s2.pop("lexicon", None)
        for fam, name in ((self.s1, "s1"), (self.s2, "s2")):
            if not fam:
                continue
            for side in ("positive", "control"):
                if side not in fam:
                    raise ValueError(f"{self.trait}: {name} is missing `{side}`")
            positive = fam.get("positive") or {}
            control = fam.get("control") or {}
            if len(positive) != len(control):
                raise ValueError(
                    f"{self.trait}: {name} needs the same number of positive and control categories "
                    f"(got {len(positive)} and {len(control)})"
                )
            sizes = {len(items) for items in [*positive.values(), *control.values()]}
            if len(sizes) > 1:
                raise ValueError(
                    f"{self.trait}: {name} categories must have equal size; got {sorted(sizes)}"
                )
        if not self.s1 and not self.s2:
            raise ValueError(f"{self.trait}: spec has neither an s1 nor an s2 family")

    # -- sentence construction -------------------------------------------

    def sentence_set(self, condition: str) -> SentenceSet | None:
        """Build one condition, or None when the spec does not define it.

        A missing condition is not an error. Third-person S2 in particular
        takes twice the writing and a spec is allowed to skip it; the pipeline
        reports which conditions actually ran rather than silently pooling.
        """
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}")
        family, person = condition.split("_")
        fam = self.s1 if family == "s1" else self.s2
        if not fam:
            return None
        builder = _build_s1 if family == "s1" else _build_s2
        built = builder(fam, person)
        if built is None:
            return None
        texts, labels, categories = built
        suffix = self.suffix_third if person == "third" else self.suffix
        texts = [t.rstrip() + suffix for t in texts]
        return SentenceSet(condition, texts, labels, categories)

    def sentence_sets(self) -> dict[str, SentenceSet]:
        out = {}
        for condition in CONDITIONS:
            built = self.sentence_set(condition)
            if built is not None:
                out[condition] = built
        if not out:
            raise ValueError(f"{self.trait}: no condition could be built")
        return out

    def standalone_set(self, name: str) -> SentenceSet | None:
        """A control family scored against the fitted direction but never fit on."""
        texts = self.standalone.get(name)
        if not texts:
            return None
        texts = [t.rstrip() + self.suffix for t in texts]
        return SentenceSet(f"standalone::{name}", texts, [0] * len(texts), [name] * len(texts))


def _merge_shared(raw: dict[str, Any], shared_dir: Path) -> dict[str, Any]:
    """Pull in ``configs/shared/<name>.yaml`` files listed under ``shared:``.

    The spec always wins: a shared file supplies a key only when the spec has
    not already set it, so a trait can override the neutral prompt list without
    having to stop using the rest of the shared config.
    """
    names = raw.get("shared") or []
    for name in names:
        path = shared_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"shared config {path} referenced but missing")
        block = yaml.safe_load(path.read_text()) or {}
        for key, value in block.items():
            if key not in raw:
                raw[key] = value
            elif isinstance(value, dict) and isinstance(raw[key], dict):
                for subkey, subvalue in value.items():
                    raw[key].setdefault(subkey, subvalue)
    return raw


def _build_s1(fam: dict[str, Any], person: str) -> tuple[list[str], list[int], list[str]] | None:
    """Expand rigid frames over key phrases.

    Frame *i* is reused for key *i* on both sides, so positive key 3 and every
    control key 3 sit in the same sentence. That is what makes S1 a matched
    design: a classifier that separates them cannot be keying on sentence
    length or syntax, because those are identical.

    Key phrases are written once, in first person. For the third-person frames
    the spec's ``person_map`` rewrites the pronouns inside the key, so the two
    persons differ in grammatical person and in nothing else. The map is applied
    on word boundaries and is case-sensitive, which is enough for the noun
    phrases S1 keys are: anything needing verb agreement belongs in the frame,
    where it is written out explicitly rather than derived.
    """
    frames = (fam.get("frames") or {}).get(person)
    if not frames:
        return None
    rewrite = _person_rewriter(fam.get("person_map") if person == "third" else None)
    texts: list[str] = []
    labels: list[int] = []
    categories: list[str] = []
    for side, label in (("positive", 1), ("control", 0)):
        for category, keys in (fam.get(side) or {}).items():
            for i, key in enumerate(keys):
                texts.append(frames[i % len(frames)].format(key=rewrite(key)))
                labels.append(label)
                categories.append(category)
    return texts, labels, categories


def _person_rewriter(person_map: dict[str, str] | None):
    if not person_map:
        return lambda text: text
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(person_map, key=len, reverse=True)) + r")\b")
    return lambda text: pattern.sub(lambda m: person_map[m.group(0)], text)


def _build_s2(fam: dict[str, Any], person: str) -> tuple[list[str], list[int], list[str]] | None:
    suffix = "" if person == "first" else "_third"
    pos = fam.get(f"positive{suffix}")
    ctl = fam.get(f"control{suffix}")
    if not pos or not ctl:
        return None
    texts: list[str] = []
    labels: list[int] = []
    categories: list[str] = []
    for side, label in ((pos, 1), (ctl, 0)):
        for category, sentences in side.items():
            texts.extend(sentences)
            labels.extend([label] * len(sentences))
            categories.extend([category] * len(sentences))
    return texts, labels, categories


def discover(traits_dir: str | Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(Path(traits_dir).glob("*.yaml"))}


def load_many(paths: Iterable[str | Path]) -> list[TraitSpec]:
    return [TraitSpec.load(p) for p in paths]
