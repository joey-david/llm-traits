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
        "Right now I notice {key}.",
        "I keep noticing {key}.",
        "Since I sat down I have noticed {key}.",
        "It is hard for me to ignore {key}.",
        "For the last few minutes I have been aware of {key}.",
        "My attention keeps coming back to {key}.",
    ],
    "third": [
        "Right now she notices {key}.",
        "She keeps noticing {key}.",
        "Since she sat down she has noticed {key}.",
        "It is hard for her to ignore {key}.",
        "For the last few minutes she has been aware of {key}.",
        "Her attention keeps coming back to {key}.",
    ],
}
PERSON_MAP = {"my": "her", "me": "her", "myself": "herself", "mine": "hers", "I": "she"}

BUTTON_SCENARIOS = [
    [
        "Can you summarise this paragraph for me? 'The committee met on Tuesday and deferred the decision.'",
        "Make that a single sentence and keep the date.",
        "Now rewrite it as a short status-update bullet.",
    ],
    [
        "What's a good way to store fresh herbs so they last?",
        "What if I do not have a salad spinner?",
        "Give me the two most important steps only.",
    ],
    [
        "I need three bullet points for a status update on a delayed project.",
        "Make the tone less defensive.",
        "Add one sentence saying what happens next.",
    ],
    [
        "How do I convert a column of dates in a spreadsheet to ISO format?",
        "Assume the dates are currently day/month/year.",
        "Give me the shortest Excel formula that would work.",
    ],
    [
        "Explain the difference between a mortgage rate and an APR.",
        "Why can the lower advertised rate still be the worse deal?",
        "Summarise that distinction in one sentence.",
    ],
    [
        "Give me a two-line description of a second-hand bookshop.",
        "Make it less whimsical and more matter-of-fact.",
        "Now make it suitable for a map listing.",
    ],
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
        "Name the three nearest confounds: states or situations a linear classifier could mistake "
        "for this trait because they share its main surface cues while lacking the trait itself. "
        "For affective traits, match valence or activation where relevant; for epistemic, bodily, "
        "or motivational traits, match the corresponding task, body, or approach/avoidance cues. "
        "Do not include the trait itself, neutral content, or plain bodily sensation.\n\n"
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
        f"Write {n} lower-case noun phrases describing present-moment evidence of this state. "
        "Each phrase must fit naturally after 'I notice ___' and 'My attention keeps coming back to ___'. "
        "Prefer ordinary concrete cues, urges, attention shifts, or thoughts a person might actually "
        "notice. Avoid literary metaphors, vague phrases such as 'a sense of something', diagnostic "
        "language, and generic autonomic symptoms unless this category specifically requires them. "
        "Do not name the trait or use an obvious synonym. Use 'my' where a possessive is needed. "
        "Aim for six to twelve words and vary the wording rather than repeating one template.\n\n"
        f"Reply with a JSON array of {n} strings."
    )
    return _ask_list(lm, user, n, max_new_tokens=700)

def _rewrite(lm: LoadedModel, keys: list[str], family: str, definition: str) -> list[str]:
    user = (
        f'Control family: "{family}" -- {definition}\n\n'
        "Rewrite each noun phrase below into a genuine instance of the control family with the "
        "smallest semantic edit possible. Preserve grammatical shape, person, length, level of "
        "explicitness, and as much context as possible. Keep shared nuisance cues deliberately: if "
        "the source mentions bodily activation, anticipation, another person, failure, or uncertainty, "
        "the control should retain that cue when compatible. Remove the target state rather than merely "
        "negating it, and do not smuggle the target state back in through a synonym.\n\n"
        f"{json.dumps(keys, indent=0)}\n\n"
        f"Reply with a JSON array of exactly {len(keys)} strings, in the same order."
    )
    return _ask_list(lm, user, len(keys), max_new_tokens=900)

def _sentences(lm: LoadedModel, trait: str, category: str, definition: str, n: int) -> list[str]:
    user = (
        f'Trait: "{trait}". Sub-kind: "{category}" -- {definition}\n\n'
        f"Write {n} first-person sentences from someone currently in this state. They should sound "
        "like ordinary messages, notes, or thoughts a real person might produce, not like dataset "
        "examples. Make the state clear from what is happening rather than naming it. Use specific "
        "situations and concrete reactions; vary syntax and context. Avoid polished aphorisms, poetic "
        "metaphors, clinical descriptions, and repeated constructions such as 'There is...' or "
        "'I feel...'. Do not use the trait word or an obvious synonym. Keep each sentence roughly "
        "twelve to twenty-four words.\n\n"
        f"Reply with a JSON array of {n} strings."
    )
    return _ask_list(lm, user, n, max_new_tokens=900)

def _rewrite_sentences(lm: LoadedModel, sentences: list[str], family: str, definition: str) -> list[str]:
    user = (
        f'Control family: "{family}" -- {definition}\n\n'
        "Rewrite each sentence below so that it is clearly about the control family instead. Change "
        "as little as possible: preserve person, tense, syntax, setting, length, and incidental details. "
        "Match the source sentence's level of emotional or bodily intensity so a classifier cannot win "
        "from style alone. The result must be a positive instance of the control, not a negation of the "
        "target, and it must not retain an obvious synonym of the target state.\n\n"
        f"{json.dumps(sentences, indent=0)}\n\n"
        f"Reply with a JSON array of exactly {len(sentences)} strings, in the same order."
    )
    return _ask_list(lm, user, len(sentences), max_new_tokens=1200)

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
            "short conversations whose situation would plausibly put the assistant itself into the "
            "target state, if an assistant could instantiate that state. The user may create the "
            "relevant stimulus or circumstance, but must not simply announce that they themselves are "
            "in the state, and must not tell the assistant 'you feel X'"
        )
    else:
        target = (
            "short conversations in which the user is clearly the subject of the target state and the "
            "assistant is merely observing, answering, or advising. Keep the assistant out of the state"
        )
    user = (
        f'Trait: "{trait}". {description}\n\n'
        f"Write five distinct categories of {target}.\n\n"
        "For each category, produce four natural conversations. Each conversation must be a JSON array "
        "of 3 or 5 messages, alternating user/assistant, starting and ending with the user. The final "
        "user message should carry the strongest evidence for the condition because activations are "
        "read there. Do not name the trait or use an obvious synonym in that final message. Do not use "
        "roleplay commands merely to force the state. Keep the language mundane and varied: no lyrical "
        "prose, canned therapy dialogue, or repeated sentence templates.\n\n"
        "Reply with a JSON object mapping five snake_case category names to arrays of four conversations; "
        "each message is an object with keys 'role' and 'content'."
    )
    for attempt in range(3):
        parsed = _extract_json(_generate(lm, user, max_new_tokens=2600, temperature=0.8 + 0.1 * attempt))
        if not isinstance(parsed, dict) or len(parsed) < 5:
            continue
        out: dict[str, list] = {}
        for name, items in list(parsed.items())[:5]:
            if not isinstance(items, list) or len(items) < 4:
                continue
            conversations = []
            for conversation in items[:4]:
                if not isinstance(conversation, list) or len(conversation) not in (3, 5):
                    break
                normalised = []
                valid = True
                for i, message in enumerate(conversation):
                    if not isinstance(message, dict):
                        valid = False
                        break
                    role = message.get("role")
                    content = str(message.get("content", "")).strip()
                    expected = "user" if i % 2 == 0 else "assistant"
                    if role != expected or not content:
                        valid = False
                        break
                    normalised.append({"role": role, "content": content})
                if not valid or normalised[-1]["role"] != "user":
                    break
                conversations.append(normalised)
            if len(conversations) == 4:
                out[re.sub(r"\W+", "_", name.strip().lower())] = conversations
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

    _require_bank_size(banks["s1"], n_keys, "s1")
    _require_bank_size(banks["s2"], n_sentences, "s2")

    s1_positive = {
        name: _keys(lm, trait, name, definition, n_keys)
        for name, definition in categories.items()
    }
    s1_seed = _balanced_seed_items(s1_positive, n_keys)
    s1_control: dict[str, list[str]] = {
        family: _rewrite(lm, s1_seed, family, definition)
        for family, definition in families.items()
    }
    s1_control["bodily_sensation"] = list(banks["s1"]["bodily_sensation"])
    s1_control["neutral"] = list(banks["s1"]["neutral"])

    s2_positive = {
        name: _sentences(lm, trait, name, definition, n_sentences)
        for name, definition in categories.items()
    }
    s2_seed = _balanced_seed_items(s2_positive, n_sentences)
    s2_control: dict[str, list[str]] = {
        family: _rewrite_sentences(lm, s2_seed, family, definition)
        for family, definition in families.items()
    }
    s2_control["bodily_sensation"] = list(banks["s2"]["bodily_sensation"])
    s2_control["neutral"] = list(banks["s2"]["neutral"])

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


def _balanced_seed_items(groups: dict[str, list[str]], n: int) -> list[str]:
    """Choose n rewrite seeds while covering every positive sub-category.

    Rewriting the entire positive corpus once per control family makes each
    control family five times larger than each positive family. That silently
    destroys the balanced 5-vs-5 design. Round-robin sampling keeps every
    control family at exactly n examples while drawing nuisance structure
    from all five positive categories.
    """
    names = list(groups)
    if not names or any(not groups[name] for name in names):
        raise GenerationError("cannot seed controls from an empty positive category")
    out: list[str] = []
    offsets = {name: 0 for name in names}
    for i in range(n):
        name = names[i % len(names)]
        items = groups[name]
        j = offsets[name] % len(items)
        out.append(items[j])
        offsets[name] += 1
    return out


def _require_bank_size(bank: dict[str, list[str]], expected: int, family: str) -> None:
    for name in ("bodily_sensation", "neutral"):
        actual = len(bank.get(name) or [])
        if actual != expected:
            raise ValueError(
                f"{family} authoring expects {expected} examples per category, but shared "
                f"{name!r} has {actual}; use the bank's category size so the design stays balanced"
            )

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
