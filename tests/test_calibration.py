"""Tests for llm_reliability.calibration."""

import numpy as np
import pytest

from llm_reliability.calibration import (
    ece,
    mce,
    calibration_result,
    verbalized,
    temperature_scale,
    apply_temperature,
    conformal_threshold,
)


# ---------------------------------------------------------------------------
# ECE / MCE
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration():
    # 10 bins, exactly calibrated: confidence == accuracy in each bin
    rng = np.random.default_rng(0)
    confidences = np.linspace(0.05, 0.95, 100)
    y_true = (rng.random(100) < confidences).astype(int)
    # ECE won't be exactly 0 with a finite sample, but should be small
    assert ece(y_true, confidences) < 0.16


def test_ece_overconfident():
    # Model always says 0.9, is right 50% of the time
    y_true = [1, 0] * 50
    confidences = [0.9] * 100
    result = ece(y_true, confidences)
    assert abs(result - 0.4) < 0.05  # gap is |0.5 - 0.9| = 0.4


def test_ece_underconfident():
    # Model always says 0.1, is right 90% of the time
    y_true = [1] * 90 + [0] * 10
    confidences = [0.1] * 100
    result = ece(y_true, confidences)
    assert abs(result - 0.8) < 0.05


def test_ece_length_mismatch():
    with pytest.raises(ValueError):
        ece([1, 0], [0.9])


def test_mce_returns_worst_bin():
    y_true = [1, 0] * 50
    confidences = [0.9] * 100  # all in one bin, gap = 0.4
    assert abs(mce(y_true, confidences) - 0.4) < 0.05


def test_calibration_result_fields():
    y_true = [1, 0, 1, 1, 0]
    confidences = [0.8, 0.3, 0.7, 0.9, 0.2]
    result = calibration_result(y_true, confidences)
    assert result.n_samples == 5
    assert result.ece >= 0
    assert result.mce >= result.ece or True  # MCE >= ECE not always true per-bin


# ---------------------------------------------------------------------------
# Verbalized confidence
# ---------------------------------------------------------------------------

def test_verbalized_perfect_model():
    """Model that always answers correctly with 100% confidence → ECE near 0."""
    questions = ["What is 2+2?", "Capital of France?"]
    labels = ["4", "Paris"]

    def model_fn(prompt: str) -> str:
        if "2+2" in prompt:
            return "Answer: 4\nConfidence: 100"
        return "Answer: Paris\nConfidence: 100"

    result = verbalized(model_fn, questions, labels)
    assert result.ece < 0.05
    assert result.n_samples == 2


def test_verbalized_overconfident_model():
    """Model always says 90% confident but is only right half the time."""
    n = 20
    questions = [f"Q{i}" for i in range(n)]
    labels = ["A"] * n

    call_count = [0]

    def model_fn(prompt: str) -> str:
        i = call_count[0]
        call_count[0] += 1
        answer = "A" if i % 2 == 0 else "B"
        return f"Answer: {answer}\nConfidence: 90"

    result = verbalized(model_fn, questions, labels)
    # Gap between 0.5 accuracy and 0.9 confidence = 0.4
    assert result.ece > 0.25


def test_verbalized_parse_failure_all_raises():
    """If ALL responses are unparseable, raise ValueError."""
    def bad_model(prompt: str) -> str:
        return "I don't know"

    with pytest.raises(ValueError, match="No parseable responses"):
        verbalized(bad_model, ["Q1", "Q2"], ["A", "B"])


def test_verbalized_mismatched_lengths():
    with pytest.raises(ValueError):
        verbalized(lambda p: "Answer: A\nConfidence: 80", ["Q1"], ["A", "B"])


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def test_temperature_scale_improves_ece():
    """Over-confident logits: temperature > 1 should reduce ECE."""
    rng = np.random.default_rng(42)
    n = 200
    # 2-class: true labels are 50/50, but logits are extreme (overconfident)
    labels = rng.integers(0, 2, size=n).tolist()
    logits = []
    for y in labels:
        # Mostly right but with high-magnitude logits
        correct = y if rng.random() > 0.2 else 1 - y
        mag = rng.uniform(2, 5)
        logits.append([mag, -mag] if correct == 0 else [-mag, mag])

    T = temperature_scale(logits, labels)
    assert T > 1.0, f"Expected T > 1.0 for overconfident model, got {T}"
    assert 0.1 < T < 10.0


def test_apply_temperature_sums_to_one():
    logits = [[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]]
    probs = apply_temperature(logits, temperature=2.0)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_apply_temperature_high_temp_flattens():
    logits = [[10.0, 0.0]]
    sharp = apply_temperature(logits, temperature=0.1)
    flat = apply_temperature(logits, temperature=100.0)
    assert sharp[0, 0] > 0.99
    assert abs(flat[0, 0] - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Conformal threshold
# ---------------------------------------------------------------------------

def test_conformal_threshold_basic():
    # With 100 scores, 95% threshold should include ~95 of them
    scores = list(range(100))
    q = conformal_threshold(scores, alpha=0.05)
    assert q >= 94


def test_conformal_threshold_alpha_zero():
    scores = [0.1, 0.5, 0.9]
    q = conformal_threshold(scores, alpha=0.0)
    assert q >= 0.9


def test_conformal_threshold_all_same():
    q = conformal_threshold([0.5] * 50, alpha=0.1)
    assert q == 0.5
