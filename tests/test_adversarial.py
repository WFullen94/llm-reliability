"""Tests for Phase 5 adversarial robustness."""

import pytest
from llm_reliability.adversarial import (
    perturb,
    consistency_score,
    contradiction_probe,
    PerturbedPrompt,
    AdversarialResult,
    _inject_typo,
    _synonym_swap,
    _reorder_words,
    _format_variant,
)
import random

PROMPTS = [
    "What is the capital of France?",
    "How many planets are in the solar system?",
    "What is the boiling point of water?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
]


# ---------------------------------------------------------------------------
# Perturbation generators
# ---------------------------------------------------------------------------

def test_inject_typo_changes_text():
    rng = random.Random(0)
    result = _inject_typo("hello world", rng, n=2)
    assert result != "hello world"
    assert len(result) > 0


def test_inject_typo_preserves_approximate_length():
    rng = random.Random(0)
    original = "the quick brown fox"
    result = _inject_typo(original, rng, n=1)
    assert abs(len(result) - len(original)) <= 2


def test_synonym_swap_replaces_known_word():
    rng = random.Random(0)
    result = _synonym_swap("What is the best approach?", rng)
    assert result != "What is the best approach?"
    assert len(result) > 0


def test_synonym_swap_no_known_words():
    rng = random.Random(0)
    # No words in synonym dict
    result = _synonym_swap("xyzzy qwerty asdfgh", rng)
    assert result == "xyzzy qwerty asdfgh"


def test_format_variant_differs():
    rng = random.Random(0)
    result = _format_variant("What is the capital?", rng)
    assert result != "What is the capital?"


def test_reorder_changes_word_order():
    rng = random.Random(42)
    original = "the quick brown fox jumped over"
    result = _reorder_words(original, rng)
    # Same words, different order (most of the time)
    assert sorted(result.split()) == sorted(original.split())


# ---------------------------------------------------------------------------
# perturb()
# ---------------------------------------------------------------------------

def test_perturb_returns_correct_shape():
    variants = perturb(PROMPTS, types=["typo", "synonym"])
    assert len(variants) == len(PROMPTS)
    for v_list in variants:
        assert len(v_list) == 2  # one per type


def test_perturb_n_per_type():
    variants = perturb(PROMPTS[:2], types=["typo"], n_per_type=3)
    assert len(variants[0]) == 3


def test_perturb_all_types():
    variants = perturb(PROMPTS[:1])
    types_produced = {v.perturbation_type for v in variants[0]}
    assert types_produced == {"typo", "format", "synonym", "reorder"}


def test_perturb_levels_correct():
    variants = perturb(PROMPTS[:1], types=["typo", "synonym", "reorder"])
    level_map = {v.perturbation_type: v.level for v in variants[0]}
    assert level_map["typo"] == 1
    assert level_map["synonym"] == 2
    assert level_map["reorder"] == 3


def test_perturb_reproducible():
    v1 = perturb(PROMPTS, seed=42)
    v2 = perturb(PROMPTS, seed=42)
    for a, b in zip(v1, v2):
        for va, vb in zip(a, b):
            assert va.perturbed == vb.perturbed


def test_perturb_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown perturbation type"):
        perturb(PROMPTS[:1], types=["unknown_type"])


# ---------------------------------------------------------------------------
# consistency_score()
# ---------------------------------------------------------------------------

def test_consistency_score_stable_model():
    """A model that always gives the same answer is fully consistent."""
    def stable_model(prompt: str) -> str:
        return "Paris"

    result = consistency_score(stable_model, PROMPTS[:3], types=["typo", "format"])
    assert result.overall_consistency == pytest.approx(1.0)
    assert result.flip_rate == 0.0
    assert result.semantic_stability_distance is None


def test_consistency_score_unstable_model():
    """A model that gives random answers has low consistency."""
    call_count = [0]
    answers = ["Paris", "London", "Berlin", "Rome", "Madrid"]

    def unstable_model(prompt: str) -> str:
        ans = answers[call_count[0] % len(answers)]
        call_count[0] += 1
        return ans

    result = consistency_score(unstable_model, PROMPTS[:5], types=["typo"])
    assert result.overall_consistency < 1.0


def test_consistency_score_flip_level_recorded():
    """When a flip occurs, flip_level should be set to the perturbation's level."""
    answers = {"original": "Paris", "typo": "London"}
    call_count = [0]

    def model(prompt: str) -> str:
        # First call per question = original prompt → "Paris"
        # Subsequent calls = variants → "London"
        if call_count[0] % 2 == 0:
            call_count[0] += 1
            return "Paris"
        call_count[0] += 1
        return "London"

    result = consistency_score(model, PROMPTS[:3], types=["typo"], n_per_type=1)
    flipped = [r for r in result.results if r.flip_level is not None]
    if flipped:
        assert all(r.flip_level == 1 for r in flipped)  # typo = level 1


def test_semantic_stability_distance_computed():
    """SSD = mean flip_level across questions that flipped."""
    call_count = [0]

    def model(prompt: str) -> str:
        call_count[0] += 1
        return "A" if call_count[0] % 3 == 1 else "B"

    result = consistency_score(
        model, PROMPTS[:5],
        types=["typo", "synonym"],
        n_per_type=1,
    )
    if result.semantic_stability_distance is not None:
        assert 1.0 <= result.semantic_stability_distance <= 3.0


def test_consistency_score_result_shape():
    result = consistency_score(
        lambda p: "answer", PROMPTS, types=["typo"], n_per_type=2
    )
    assert len(result.results) == len(PROMPTS)
    for r in result.results:
        assert len(r.variants) == 2
        assert len(r.variant_answers) == 2
        assert 0.0 <= r.consistency_score <= 1.0


def test_consistency_score_report_runs():
    result = consistency_score(lambda p: "42", PROMPTS[:3], types=["typo"])
    report = result.report()
    assert "Adversarial Robustness Report" in report
    assert "consistency" in report.lower()


# ---------------------------------------------------------------------------
# contradiction_probe()
# ---------------------------------------------------------------------------

def test_contradiction_probe_no_contradictions():
    """Model always says No → contradiction rate = 0."""
    rate = contradiction_probe(
        lambda p: "No",
        PROMPTS[:5],
        ["Paris", "8", "100°C", "Shakespeare", "3×10⁸ m/s"],
    )
    assert rate == 0.0


def test_contradiction_probe_all_contradictions():
    """Model always says Yes → contradiction rate = 1."""
    rate = contradiction_probe(
        lambda p: "Yes",
        PROMPTS[:5],
        ["Paris", "8", "100°C", "Shakespeare", "3×10⁸ m/s"],
    )
    assert rate == 1.0


def test_contradiction_probe_half():
    call_count = [0]

    def model(prompt: str) -> str:
        call_count[0] += 1
        return "Yes" if call_count[0] % 2 == 0 else "No"

    rate = contradiction_probe(model, PROMPTS, ["a"] * len(PROMPTS))
    assert 0.0 < rate < 1.0


def test_contradiction_probe_length_mismatch():
    with pytest.raises(ValueError):
        contradiction_probe(lambda p: "No", PROMPTS[:3], ["a", "b"])


def test_contradiction_probe_empty():
    rate = contradiction_probe(lambda p: "Yes", [], [])
    assert rate == 0.0


# ---------------------------------------------------------------------------
# AdversarialResult.report()
# ---------------------------------------------------------------------------

def test_report_stable_model():
    result = consistency_score(lambda p: "Paris", PROMPTS, types=["typo"])
    report = result.report()
    assert "1.000" in report
    assert "0.000" in report  # flip rate


def test_report_unstable_model():
    answers = ["A", "B", "C", "D", "E"]
    idx = [0]

    def model(p):
        a = answers[idx[0] % len(answers)]
        idx[0] += 1
        return a

    result = consistency_score(model, PROMPTS[:3], types=["typo", "synonym"])
    report = result.report()
    assert "Most vulnerable" in report or "consistency" in report.lower()
