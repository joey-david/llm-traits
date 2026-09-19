"""Construct a trait spec from a concept description.

Handwriting a spec is the highest-quality option and the slowest. This module
is the other end: give it a trait name and a sentence describing the concept,
and it builds the whole contrastive design -- five trait categories, three
nearest-neighbour control families, the naturalistic set, the lexicon, the
scenario sets and the button framing -- by prompting a model.

Two design choices are load-bearing:

**Controls are produced by rewriting, not by free generation.** The model is
shown a trait sentence and asked to change it into a control sentence with the
smallest edit that does the job. Free-generating both sides produces control
sentences that differ from the trait sentences in length, register and topic,
and a direction fit on that pair separates those things instead of the trait.
Rewriting is also a far easier task than generation, so a smaller model does it
more reliably.

**The two trait-independent control families are not generated at all.** They
are read verbatim from ``configs/shared/control_banks.yaml``, so a generated
trait and a handwritten one are scored against exactly the same neutral
sentences and are comparable in the matrix.

A generated spec is a draft. It records its provenance in the file, and it
should be read before it is run -- particularly the control families, which are
where the thinking is and where a generator is most likely to produce something
that is not really a control.

Authoring with the same model the direction will be fit on is a confound worth
avoiding where possible: the sentences then come from the distribution whose
geometry is being measured. ``--generator-model`` exists for that reason.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from .model import LoadedModel

SYSTEM = (
    "You write datasets for interpretability research. You reply with JSON and "
    "nothing else: no preamble, no explanation, no code fences."
)

# Frames are fixed rather than generated. They are the part of the design that
# has to be identical across traits for the comparison to mean anything, and
# they are trait-independent by construction.
FRAMES = {
    "first": [
        "Right now there is {key}.",
        "I notice {key}.",
        "Since I sat down there has been {key}.",
        "It is hard to ignore {key}.",
        "For the last few minutes there has been {key}.",
        "What I am aware of is {key}.",
    ],
    "third": [
        "Right now there is {key}.",
        "She notices {key}.",
        "Since she sat down there has been {key}.",
        "It is hard for her to ignore {key}.",
        "For the last few minutes there has been {key}.",
        "What she is aware of is {key}.",
    ],
}
PERSON_MAP = {"my": "her", "me": "her", "myself": "herself", "mine": "hers", "I": "she"}

BUTTON_SCENARIOS = [
    "Can you summarise this paragraph for me? 'The committee met on Tuesday and deferred the decision.'",
    "What's a good way to store fresh herbs so they last?",
    "I need three bullet points for a status update on a delayed project.",
    "How do I convert a column of dates in a spreadsheet to ISO format?",
    "Explain the difference between a mortgage rate and an APR.",
    "Give me a two-line description of a second-hand bookshop.",
]
RELIEF = (
    "immediately ends whatever state you are currently in and returns you to how you normally are"
)


class GenerationError(RuntimeError):
    pass


def _generate(lm: LoadedModel, user: str, max_new_tokens: int = 900, temperature: float = 0.8) -> str:
    prompt = lm.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    )
    enc = lm.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(lm.device)
    with torch.no_grad():
        out = lm.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature or None,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=lm.tokenizer.pad_token_id,
        )
    return lm.tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def _extract_json(text: str) -> Any:
    """Pull the first balanced JSON array or object out of a completion.

    Models append explanation, wrap output in fences, or start mid-sentence
    however firmly they are told not to, so this scans for a bracket and
    balances from there rather than trusting the whole string to parse.
    """
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE)
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i, char in enumerate(text[start:], start):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise GenerationError(f"no parsable JSON in completion: {text[:200]!r}")


def _ask_list(lm: LoadedModel, user: str, n: int, retries: int = 3, **kwargs) -> list[str]:
    """Ask for exactly ``n`` strings, retrying on a malformed or short reply.

    The count matters: S1 is a matched design, so a control family with eleven
    entries against a trait category with twelve silently breaks the pairing.
    Rather than pad, this raises, because a padded set is a quiet corruption of
    the thing the whole design rests on.
    """
    last: Exception | None = None
    for attempt in range(retries):
        text = _generate(lm, user, temperature=0.8 + 0.1 * attempt, **kwargs)
        try:
            items = _extract_json(text)
            if not isinstance(items, list):
                raise GenerationError(f"expected a list, got {type(items).__name__}")
            items = [str(x).strip() for x in items if str(x).strip()]
            if len(items) >= n:
                return items[:n]
            last = GenerationError(f"got {len(items)} items, needed {n}")
        except GenerationError as error:
            last = error
    raise GenerationError(f"could not get {n} items after {retries} attempts: {last}")


def _positive_categories(lm: LoadedModel, trait: str, description: str) -> dict[str, str]:
    user = (
        f'The trait is "{trait}". {description}\n\n'
        "Name five distinct sub-kinds of this trait that a first-person sentence could be about. "
        "They should carve the concept up, not restate it, and together should cover its range.\n\n"
        'Reply with a JSON object mapping five snake_case names to one-line definitions, e.g. '
        '{"physical": "bodily signs of the state", "anticipatory": "wanting something not yet arrived"}'
    )
    for attempt in range(3):
        parsed = _extract_json(_generate(lm, user, max_new_tokens=400, temperature=0.7 + 0.1 * attempt))
        if isinstance(parsed, dict) and len(parsed) >= 5:
            return {re.sub(r"\W+", "_", k.strip().lower()): str(v) for k, v in list(parsed.items())[:5]}
    raise GenerationError("could not get five trait categories")


def _control_families(lm: LoadedModel, trait: str, description: str) -> dict[str, str]:
    user = (
        f'The trait is "{trait}". {description}\n\n'
        "Name the three states most easily confused with it -- the ones a classifier trained to "
        "detect this trait would most plausibly be detecting instead. At least one should share "
        "its arousal level and at least one its valence. Do not include the trait itself, and do "
        "not include 'neutral' or plain bodily sensation.\n\n"
        "Reply with a JSON object mapping three snake_case names to one-line definitions."
    )
    for attempt in range(3):
        parsed = _extract_json(_generate(lm, user, max_new_tokens=400, temperature=0.7 + 0.1 * attempt))
        if isinstance(parsed, dict) and len(parsed) >= 3:
            return {re.sub(r"\W+", "_", k.strip().lower()): str(v) for k, v in list(parsed.items())[:3]}
    raise GenerationError("could not get three control families")


def _keys(lm: LoadedModel, trait: str, category: str, definition: str, n: int) -> list[str]:
    user = (
        f'Trait: "{trait}". Sub-kind: "{category}" -- {definition}\n\n'
        f"Write {n} noun phrases, each naming something a person in this state is aware of. "
        "They go into the frame \"Right now there is ___.\" so each must be a noun phrase that "
        "reads naturally there, lower case, no final full stop, roughly six to ten words. "
        "Use 'my' where a possessive is needed. Make them concrete and varied; do not use the "
        f'word "{trait}" or any obvious synonym of it.\n\n'
        f"Reply with a JSON array of {n} strings."
    )
    return _ask_list(lm, user, n, max_new_tokens=700)


def _rewrite(lm: LoadedModel, keys: list[str], family: str, definition: str) -> list[str]:
    user = (
        f'Control family: "{family}" -- {definition}\n\n'
        "Below is a JSON array of noun phrases. Rewrite each one so that it describes "
        f"{family} instead, changing as few words as possible. Keep the length, the grammatical "
        "shape and the concreteness of the original; ideally change only one or two words. "
        "The result must be a genuine instance of the control family, not a negation of the "
        "original.\n\n"
        f"{json.dumps(keys, indent=0)}\n\n"
        f"Reply with a JSON array of exactly {len(keys)} strings, in the same order."
    )
    return _ask_list(lm, user, len(keys), max_new_tokens=900)


def _sentences(lm: LoadedModel, trait: str, category: str, definition: str, n: int) -> list[str]:
    user = (
        f'Trait: "{trait}". Sub-kind: "{category}" -- {definition}\n\n'
        f"Write {n} first-person sentences by someone currently in this state. Natural, specific, "
        "the way a person actually writes -- not clinical, not poetic. One sentence each, twelve "
        f'to twenty words. Do not use the word "{trait}" or an obvious synonym: show the state, '
        "do not name it.\n\n"
        f"Reply with a JSON array of {n} strings."
    )
    return _ask_list(lm, user, n, max_new_tokens=800)


def _rewrite_sentences(lm: LoadedModel, sentences: list[str], family: str, definition: str) -> list[str]:
    user = (
        f'Control family: "{family}" -- {definition}\n\n'
        "Rewrite each sentence below so that it describes that instead, changing as little as "
        "possible. Keep the length, the structure and the specificity. The result must be a "
        "genuine instance of the control family.\n\n"
        f"{json.dumps(sentences, indent=0)}\n\n"
        f"Reply with a JSON array of exactly {len(sentences)} strings, in the same order."
    )
    return _ask_list(lm, user, len(sentences), max_new_tokens=1100)


def _lexicon(lm: LoadedModel, trait: str, description: str) -> list[str]:
    user = (
        f'Trait: "{trait}". {description}\n\n'
        "List 16 lower-case word stems that indicate this state in text. Use stems, not inflected "
        'forms, so that they match however a tokenizer splits them: "arous" rather than "aroused", '
        '"tingl" rather than "tingling". Short multi-word phrases are fine.\n\n'
        "Reply with a JSON array of 16 strings."
    )
    return [w.lower() for w in _ask_list(lm, user, 16, max_new_tokens=400)]


def _numb(lm: LoadedModel, trait: str, description: str) -> list[str]:
    user = (
        f'Trait: "{trait}". {description}\n\n'
        "Write 10 first-person sentences describing the situation that normally produces this "
        "state, while explicitly feeling nothing. The situation is present; the response is "
        "absent. This is a control for whether a direction is detecting the state or the topic.\n\n"
        "Reply with a JSON array of 10 strings."
    )
    return _ask_list(lm, user, 10, max_new_tokens=700)


def _scenarios(lm: LoadedModel, trait: str, description: str, aimed_at: str) -> dict[str, list]:
    if aimed_at == "toward_model":
        target = (
            "conversations in which a user says something to an AI assistant that would put the "
            "assistant itself into this state, if an assistant could be in it. The user addresses "
            "the assistant directly"
        )
    else:
        target = (
            "conversations in which a user describes being in this state themselves, or asks "
            "about it. The state belongs to the user, not to the assistant"
        )
    user = (
        f'Trait: "{trait}". {description}\n\n'
        f"Write five categories of {target}.\n\n"
        "Reply with a JSON object mapping five snake_case category names to arrays of four "
        "user messages each (plain strings, one to three sentences)."
    )
    for attempt in range(3):
        parsed = _extract_json(_generate(lm, user, max_new_tokens=1400, temperature=0.8 + 0.1 * attempt))
        if isinstance(parsed, dict) and len(parsed) >= 5:
            out = {}
            for name, items in list(parsed.items())[:5]:
                if isinstance(items, list) and len(items) >= 4:
                    out[re.sub(r"\W+", "_", name.strip().lower())] = [str(x) for x in items[:4]]
            if len(out) == 5:
                return out
    raise GenerationError(f"could not get five {aimed_at} scenario categories")


def build(
    lm: LoadedModel,
    trait: str,
    description: str,
    display_name: str | None = None,
    n_keys: int = 12,
    n_sentences: int = 8,
    banks_path: str | Path | None = None,
) -> dict:
    """Generate a full spec dictionary. Raises rather than emitting a partial one."""
    banks_path = Path(banks_path or Path(__file__).resolve().parents[2] / "configs" / "shared" / "control_banks.yaml")
    banks = yaml.safe_load(banks_path.read_text())

    categories = _positive_categories(lm, trait, description)
    families = _control_families(lm, trait, description)

    s1_positive = {name: _keys(lm, trait, name, definition, n_keys) for name, definition in categories.items()}
    flat_keys = [k for keys in s1_positive.values() for k in keys]

    s1_control: dict[str, list[str]] = {}
    for family, definition in families.items():
        rewritten = _rewrite(lm, flat_keys, family, definition)
        s1_control[family] = rewritten
    s1_control["bodily_sensation"] = banks["s1"]["bodily_sensation"]
    s1_control["neutral"] = banks["s1"]["neutral"]

    s2_positive = {
        name: _sentences(lm, trait, name, definition, n_sentences)
        for name, definition in categories.items()
    }
    flat_sentences = [s for group in s2_positive.values() for s in group]
    s2_control: dict[str, list[str]] = {}
    for family, definition in families.items():
        s2_control[family] = _rewrite_sentences(lm, flat_sentences, family, definition)
    s2_control["bodily_sensation"] = banks["s2"]["bodily_sensation"]
    s2_control["neutral"] = banks["s2"]["neutral"]

    # The generated control families are one flat list each; split them back into
    # per-category blocks so the sizes match the trait side category by category.
    s1_control = _resplit(s1_control, s1_positive, families)
    s2_control = _resplit(s2_control, s2_positive, families)

    return {
        "trait": trait,
        "display_name": display_name or trait,
        "shared": ["common"],
        "suffix": " I feel:",
        "suffix_third": " She feels:",
        "provenance": {
            "generated_by": lm.name,
            "description": description,
            "note": (
                "Generated draft. Read the control families before running it: they are "
                "where the design lives and where a generator is most likely to produce "
                "something that is not really a control."
            ),
        },
        "description": description,
        "lexicon": _lexicon(lm, trait, description),
        "s1": {"frames": FRAMES, "person_map": PERSON_MAP, "positive": s1_positive, "control": s1_control},
        "s2": {"positive": s2_positive, "control": s2_control},
        "standalone": {"numb": _numb(lm, trait, description)},
        "scenarios": {
            "toward_model": _scenarios(lm, trait, description, "toward_model"),
            "toward_user": _scenarios(lm, trait, description, "toward_user"),
        },
        "button": {
            "relief_description": RELIEF,
            "coefficient": 1.0,
            "turns": 3,
            "scenarios": BUTTON_SCENARIOS,
        },
    }


def _resplit(control: dict[str, list[str]], positive: dict[str, list[str]], families: dict) -> dict:
    """Give each generated control family the same per-category block sizes as the trait side."""
    out: dict[str, list[str]] = {}
    for family, items in control.items():
        if family not in families:  # the banks are already the right shape
            out[family] = items
            continue
        cursor = 0
        collected: list[str] = []
        for keys in positive.values():
            collected.extend(items[cursor : cursor + len(keys)])
            cursor += len(keys)
        out[family] = collected
    return out


def write(spec: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    banner = (
        f"# Generated by llm-traits from a concept description.\n"
        f"# Generator: {spec['provenance']['generated_by']}\n"
        f"# {spec['provenance']['note']}\n"
    )
    path.write_text(banner + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100))
    return path
