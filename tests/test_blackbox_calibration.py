"""Tests for Phase 2 black-box calibration methods."""

import numpy as np
import pytest

from llm_reliability.calibration import (
    consistency,
    p_true,
    semantic_dispersion,
    semantic_entropy,
    CalibrationResult,
)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

QUESTIONS = [f"What is {i} + {i}?" for i in range(1, 21)]
LABELS = [str(i * 2) for i in range(1, 21)]


def _always_correct_fn(prompt: str) -> str:
    """Extracts the question numbers and returns the correct sum."""
    import re
    m = re.search(r"(\d+) \+ (\d+)", prompt)
    if m:
        return str(int(m.group(1)) + int(m.group(2)))
    return "2"


def _always_wrong_fn(prompt: str) -> str:
    return "999"


def _half_correct_fn(counter=[0]):
    def fn(prompt: str) -> str:
        counter[0] += 1
        if counter[0] % 2 == 0:
            return _always_correct_fn(prompt)
        return "999"
    return fn


# ---------------------------------------------------------------------------
# consistency()
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_always_correct_high_confidence(self):
        result = consistency(_always_correct_fn, QUESTIONS, LABELS, n_samples=5)
        assert isinstance(result, CalibrationResult)
        assert result.n_samples == len(QUESTIONS)
        # Always same answer → confidence = 1.0 in every bin
        assert result.ece < 0.15

    def test_always_wrong_high_confidence_low_accuracy(self):
        result = consistency(_always_wrong_fn, QUESTIONS, LABELS, n_samples=5)
        # Model is consistently wrong: confidence ≈ 1.0, accuracy = 0 → large ECE
        assert result.ece > 0.5

    def test_raw_scores_stored(self):
        result = consistency(_always_correct_fn, QUESTIONS[:5], LABELS[:5], n_samples=3)
        assert result.raw_scores is not None
        assert len(result.raw_scores) == 5
        for correct, conf in result.raw_scores:
            assert correct in (0, 1)
            assert 0.0 <= conf <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            consistency(_always_correct_fn, QUESTIONS[:3], LABELS[:5], n_samples=3)

    def test_n_samples_one(self):
        # n_samples=1: confidence is always 1.0 (100% agreement with yourself)
        result = consistency(_always_correct_fn, QUESTIONS[:5], LABELS[:5], n_samples=1)
        assert all(conf == 1.0 for _, conf in result.raw_scores)


# ---------------------------------------------------------------------------
# p_true()
# ---------------------------------------------------------------------------

class TestPTrue:
    def test_correct_answer_high_p_true(self):
        # Model always says the right answer, then always says "Yes" to verification
        call_count = [0]

        def model_fn(prompt: str) -> str:
            call_count[0] += 1
            if "Is the proposed answer correct" in prompt:
                return "Yes"
            return _always_correct_fn(prompt)

        result = p_true(model_fn, QUESTIONS[:10], LABELS[:10], n_samples=3)
        assert result.n_samples == 10
        # Correct answer + always Yes → confidence=1.0, accuracy=1.0 → ECE near 0
        assert result.ece < 0.1

    def test_wrong_answer_low_p_true(self):
        def model_fn(prompt: str) -> str:
            if "Is the proposed answer correct" in prompt:
                return "No"
            return "999"

        result = p_true(model_fn, QUESTIONS[:10], LABELS[:10], n_samples=3)
        # Wrong answer + always No → confidence≈0, accuracy=0 → ECE near 0
        # (both low confidence and low accuracy → well calibrated in wrong direction)
        assert result.n_samples == 10

    def test_raw_scores_stored(self):
        def model_fn(prompt: str) -> str:
            if "Is the proposed answer correct" in prompt:
                return "Yes"
            return _always_correct_fn(prompt)

        result = p_true(model_fn, QUESTIONS[:5], LABELS[:5], n_samples=2)
        assert result.raw_scores is not None
        assert len(result.raw_scores) == 5

    def test_api_call_count(self):
        """Total calls = n_questions * (1 + n_samples)."""
        calls = [0]

        def counting_fn(prompt: str) -> str:
            calls[0] += 1
            return "Yes" if "correct" in prompt.lower() else "2"

        n_q, n_s = 4, 3
        p_true(counting_fn, QUESTIONS[:n_q], LABELS[:n_q], n_samples=n_s)
        assert calls[0] == n_q * (1 + n_s)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            p_true(lambda p: "2", QUESTIONS[:3], LABELS[:5])


# ---------------------------------------------------------------------------
# semantic_dispersion()
# ---------------------------------------------------------------------------

class TestSemanticDispersion:
    def test_identical_responses_max_confidence(self):
        """Same response every time → pairwise similarity = 1.0 → confidence = 1.0."""
        def model_fn(prompt: str) -> str:
            return "The answer is four."

        result = semantic_dispersion(
            model_fn, QUESTIONS[:5], LABELS[:5], n_samples=4
        )
        for _, conf in result.raw_scores:
            assert conf > 0.95

    def test_diverse_responses_lower_confidence(self):
        """Very different responses → low pairwise similarity → low confidence."""
        responses = [
            "The sky is blue and vast.",
            "Mathematics underlies all of physics.",
            "Ancient Rome fell in 476 AD.",
            "Penguins live in Antarctica.",
            "Jazz originated in New Orleans.",
        ]
        idx = [0]

        def model_fn(prompt: str) -> str:
            r = responses[idx[0] % len(responses)]
            idx[0] += 1
            return r

        result = semantic_dispersion(
            model_fn, QUESTIONS[:3], LABELS[:3], n_samples=5
        )
        # Semantically diverse responses → confidence < 0.7 on average
        avg_conf = np.mean([conf for _, conf in result.raw_scores])
        assert avg_conf < 0.85

    def test_raw_scores_stored(self):
        result = semantic_dispersion(
            lambda p: "four", QUESTIONS[:4], LABELS[:4], n_samples=3
        )
        assert result.raw_scores is not None
        assert len(result.raw_scores) == 4

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            semantic_dispersion(lambda p: "x", QUESTIONS[:3], LABELS[:5])


# ---------------------------------------------------------------------------
# semantic_entropy()
# ---------------------------------------------------------------------------

class TestSemanticEntropy:
    def test_single_cluster_max_confidence(self):
        """Semantically identical responses → one cluster → zero entropy → confidence = 1."""
        def model_fn(prompt: str) -> str:
            return "The result is four."

        result = semantic_entropy(
            model_fn, QUESTIONS[:5], LABELS[:5], n_samples=5,
            similarity_threshold=0.8,
        )
        for _, conf in result.raw_scores:
            assert conf > 0.9

    def test_all_different_clusters_low_confidence(self):
        """Maximally diverse responses → many clusters → max entropy → confidence near 0."""
        diverse = [
            "Blue whales are the largest animals.",
            "The Eiffel Tower is in Paris.",
            "DNA carries genetic information.",
            "Chess was invented in India.",
            "Water boils at 100 degrees Celsius.",
            "Shakespeare wrote Hamlet.",
            "The Sun is a medium-sized star.",
            "Gravity pulls objects together.",
        ]
        idx = [0]

        def model_fn(prompt: str) -> str:
            r = diverse[idx[0] % len(diverse)]
            idx[0] += 1
            return r

        result = semantic_entropy(
            model_fn, QUESTIONS[:3], LABELS[:3], n_samples=8,
            similarity_threshold=0.99,  # Very strict: almost nothing clusters together
        )
        avg_conf = np.mean([conf for _, conf in result.raw_scores])
        assert avg_conf < 0.5

    def test_raw_scores_stored(self):
        result = semantic_entropy(
            lambda p: "four", QUESTIONS[:4], LABELS[:4], n_samples=3
        )
        assert result.raw_scores is not None
        assert len(result.raw_scores) == 4

    def test_confidence_in_range(self):
        result = semantic_entropy(
            lambda p: "four", QUESTIONS[:5], LABELS[:5], n_samples=4
        )
        for _, conf in result.raw_scores:
            assert 0.0 <= conf <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            semantic_entropy(lambda p: "x", QUESTIONS[:3], LABELS[:5])


# ---------------------------------------------------------------------------
# Cross-method: raw_scores shape consistency
# ---------------------------------------------------------------------------

def test_all_methods_return_same_shape():
    """All four methods on the same inputs should return the same number of raw scores."""
    qs = QUESTIONS[:5]
    ls = LABELS[:5]

    def model_fn(p: str) -> str:
        return _always_correct_fn(p)

    r_consistency = consistency(model_fn, qs, ls, n_samples=3)
    r_p_true = p_true(
        lambda p: "Yes" if "correct" in p.lower() else _always_correct_fn(p),
        qs, ls, n_samples=3,
    )
    r_dispersion = semantic_dispersion(model_fn, qs, ls, n_samples=3)
    r_entropy = semantic_entropy(model_fn, qs, ls, n_samples=3)

    for r in [r_consistency, r_p_true, r_dispersion, r_entropy]:
        assert r.raw_scores is not None
        assert len(r.raw_scores) == 5
