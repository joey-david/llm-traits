"""Tests that do not need a model.

The parts worth testing here are the ones where a silent mistake would not
look like a failure: a cross-validation that leaks, a denoising step that does
not remove what it claims to, a third-person build that is not actually the
same sentence. A wrong number in any of them still produces a plausible figure.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_traits import directions, spec as spec_module
from llm_traits.spec import TraitSpec

TRAITS = ["arousal", "pain", "hunger"]


@pytest.fixture(scope="module")
def traits_dir(pytestconfig):
    return pytestconfig.rootpath / "configs" / "traits"


@pytest.mark.parametrize("name", TRAITS)
def test_spec_builds_every_declared_condition(traits_dir, name):
    s = TraitSpec.load(traits_dir / f"{name}.yaml")
    sets = s.sentence_sets()
    assert "s1_first" in sets, "every trait needs the matched design to be comparable"
    for condition, sentences in sets.items():
        assert len(sentences) == len(sentences.labels) == len(sentences.categories)
        assert sum(sentences.labels) == len(sentences) - sum(sentences.labels), (
            f"{name}:{condition} is unbalanced; AUC against an unbalanced set is harder to read"
        )
        assert len(sentences.positive_categories) == 5
        assert len(sentences.control_categories) == 5


@pytest.mark.parametrize("name", TRAITS)
def test_s1_pairs_are_matched_by_frame(traits_dir, name):
    """Positive i and control i must be the same sentence with one phrase swapped."""
    s = TraitSpec.load(traits_dir / f"{name}.yaml")
    frames = s.s1["frames"]["first"]
    sentences = s.sentence_set("s1_first")
    per_category: dict[str, list[str]] = {}
    for text, category in zip(sentences.texts, sentences.categories):
        per_category.setdefault(category, []).append(text)
    lengths = {len(v) for v in per_category.values()}
    assert len(lengths) == 1, f"{name}: categories differ in size, so frames do not line up"
    n = lengths.pop()
    for i in range(n):
        prefixes = {v[i].split()[0] for v in per_category.values()}
        assert len(prefixes) == 1, (
            f"{name}: position {i} does not share a frame across categories "
            f"(saw {prefixes}); the design is no longer matched"
        )
    assert n >= len(frames)


@pytest.mark.parametrize("name", TRAITS)
def test_third_person_differs_only_in_person(traits_dir, name):
    s = TraitSpec.load(traits_dir / f"{name}.yaml")
    first = s.sentence_set("s1_first")
    third = s.sentence_set("s1_third")
    assert first.labels == third.labels
    assert first.categories == third.categories
    assert first.texts != third.texts
    assert all(t.endswith(s.suffix_third) for t in third.texts)
    assert not any(" my " in t for t in third.texts), "person_map missed a first-person pronoun"


def test_shared_config_does_not_override_the_spec(traits_dir, tmp_path):
    s = TraitSpec.load(traits_dir / "arousal.yaml")
    assert s.standalone["random"], "the shared random control should have been merged in"
    assert s.standalone["numb"], "the spec's own standalone set should have survived the merge"
    assert "neutral" in s.scenarios, "shared neutral scenarios should have been merged in"
    assert "toward_model" in s.scenarios


def test_denoise_removes_the_control_subspace():
    rng = np.random.default_rng(0)
    d = 64
    axis = np.zeros(d, dtype=np.float32)
    axis[0] = 1.0
    # Controls vary almost entirely along one nuisance direction.
    nuisance = np.zeros(d, dtype=np.float32)
    nuisance[1] = 1.0
    control = rng.normal(scale=0.01, size=(200, d)).astype(np.float32)
    control += np.outer(rng.normal(scale=5.0, size=200), nuisance)
    contaminated = axis + 3.0 * nuisance
    cleaned, n_components = directions.denoise(contaminated, control, var_threshold=0.5)
    assert n_components >= 1
    assert abs(float(cleaned @ nuisance)) < 0.2, "the nuisance direction survived denoising"
    assert float(cleaned @ axis) > 0.8, "denoising ate the signal"


def test_cv_auc_is_not_the_in_sample_auc():
    """On pure noise the held-out AUC must sit near chance even though the
    in-sample one will not: this is the check that the folds do not leak."""
    rng = np.random.default_rng(1)
    acts = rng.normal(size=(80, 4, 128)).astype(np.float32)
    labels = np.r_[np.ones(40), np.zeros(40)].astype(int)
    mean_auc, _ = directions.per_layer_cv_auc(acts, labels, n_splits=5, seed=0)
    assert mean_auc.max() < 0.75, f"held-out AUC on noise reached {mean_auc.max():.2f}"

    fitted = directions.fit("noise", "s1_first", acts, labels, ["a"] * 40 + ["b"] * 40, layer=2)
    assert fitted.auc_insample > fitted.auc_cv


def test_shuffled_label_null_is_the_same_procedure():
    rng = np.random.default_rng(2)
    acts = rng.normal(size=(60, 3, 64)).astype(np.float32)
    labels = np.r_[np.ones(30), np.zeros(30)].astype(int)
    mean, std = directions.shuffled_label_auc(acts, labels, layer=1, n_repeats=5, seed=0)
    assert 0.3 < mean < 0.7
    assert std >= 0.0


def test_random_direction_is_unit_norm_and_seeded():
    a = directions.random_direction(512, seed=3)
    b = directions.random_direction(512, seed=3)
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5


def test_steering_layer_stays_before_extraction():
    from llm_traits import steering

    norms = np.linspace(1.0, 40.0, 33).astype(np.float32)
    config = steering.select_layer(raw_norm=12.0, resid_norms=norms, extraction_layer=28)
    assert 1 <= config.layer < 28
    assert config.ratio > 0


@pytest.mark.parametrize("name", TRAITS)
def test_scenarios_cover_both_loaded_groups(traits_dir, name):
    from llm_traits import scenarios

    s = TraitSpec.load(traits_dir / f"{name}.yaml")
    conversations, groups, _ = scenarios.flatten(s.scenarios)
    assert set(groups) == set(scenarios.GROUPS), "the asymmetry needs all three groups"
    assert len(conversations) >= 40
    for conversation in conversations:
        assert conversation[0]["role"] == "user"


def test_discover_finds_the_shipped_traits(traits_dir):
    found = spec_module.discover(traits_dir)
    for name in TRAITS:
        assert name in found
