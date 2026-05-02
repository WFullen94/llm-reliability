"""Tests for Phase 4 drift detection."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from llm_reliability.drift import (
    DriftSnapshot,
    DriftResult,
    capture,
    compare,
    _rbf_mmd,
    _js_divergence,
    _compute_text_stats,
)

PROMPTS = [f"Question {i}: What is {i} + {i}?" for i in range(1, 21)]


# ---------------------------------------------------------------------------
# Text statistics helpers
# ---------------------------------------------------------------------------

def test_compute_text_stats_lengths():
    responses = ["hello world", "a b c d e"]
    lengths, ttr, avg_sl = _compute_text_stats(responses)
    assert lengths == [11, 9]


def test_compute_text_stats_ttr():
    # All unique words → TTR = 1.0
    responses = ["alpha beta gamma delta"]
    _, ttr, _ = _compute_text_stats(responses)
    assert ttr == pytest.approx(1.0)


def test_compute_text_stats_repeated_words():
    # "the the the" → 1 unique / 3 total = 0.333
    responses = ["the the the"]
    _, ttr, _ = _compute_text_stats(responses)
    assert ttr == pytest.approx(1 / 3, abs=0.01)


def test_js_divergence_identical():
    a = np.array([1, 2, 3, 4, 5] * 10, dtype=float)
    assert _js_divergence(a, a) < 0.01


def test_js_divergence_different():
    a = np.ones(50, dtype=float)
    b = np.ones(50, dtype=float) * 100
    assert _js_divergence(a, b) > 0.5


def test_mmd_identical_distributions():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 8))
    assert _rbf_mmd(X, X) == pytest.approx(0.0, abs=1e-6)


def test_mmd_different_distributions():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 8))
    Y = rng.standard_normal((30, 8)) + 5.0  # shifted mean
    assert _rbf_mmd(X, Y) > 0.1


# ---------------------------------------------------------------------------
# DriftSnapshot
# ---------------------------------------------------------------------------

def test_capture_basic():
    def model_fn(p: str) -> str:
        return f"The answer is {p[-1]}."

    snap = capture(model_fn, PROMPTS[:5], label="test", embed=False)
    assert len(snap.responses) == 5
    assert len(snap.lengths) == 5
    assert snap.label == "test"
    assert snap.embeddings is None
    assert snap.timestamp != ""


def test_capture_with_embeddings():
    snap = capture(lambda p: "The answer is four.", PROMPTS[:4], embed=True)
    assert snap.embeddings is not None
    assert len(snap.embeddings) == 4
    assert len(snap.embeddings[0]) > 0


def test_snapshot_save_load(tmp_path):
    snap = capture(lambda p: f"Response to: {p}", PROMPTS[:3], embed=False)
    path = tmp_path / "snap.json"
    snap.save(path)

    loaded = DriftSnapshot.load(path)
    assert loaded.prompts == snap.prompts
    assert loaded.responses == snap.responses
    assert loaded.ttr == pytest.approx(snap.ttr, abs=1e-6)
    assert loaded.lengths == snap.lengths


def test_snapshot_roundtrip_with_embeddings(tmp_path):
    snap = capture(lambda p: "consistent answer", PROMPTS[:3], embed=True)
    path = tmp_path / "snap.json"
    snap.save(path)
    loaded = DriftSnapshot.load(path)
    assert loaded.embeddings is not None
    np.testing.assert_allclose(
        np.array(loaded.embeddings), np.array(snap.embeddings), atol=1e-5
    )


# ---------------------------------------------------------------------------
# compare() — no drift case
# ---------------------------------------------------------------------------

def test_compare_identical_snapshots_no_drift():
    def model_fn(p: str) -> str:
        return "The answer is always the same consistent response."

    snap = capture(model_fn, PROMPTS[:15], embed=True)
    result = compare(snap, snap)

    assert not result.any_significant
    assert result.n_baseline == 15
    assert result.n_current == 15


def test_compare_same_model_stable():
    """Same deterministic model twice → no drift detected."""
    def model_fn(p: str) -> str:
        return f"Answer: {len(p)}"

    baseline = capture(model_fn, PROMPTS[:10], embed=False)
    current = capture(model_fn, PROMPTS[:10], embed=False)
    result = compare(baseline, current)

    ks_test = next(t for t in result.tests if "KS" in t.name)
    assert not ks_test.significant


# ---------------------------------------------------------------------------
# compare() — drift case
# ---------------------------------------------------------------------------

def test_compare_length_drift_detected():
    """Short responses baseline vs long responses current → KS test fires."""
    short_fn = lambda p: "Yes."
    long_fn = lambda p: "This is a much longer response that provides extensive detail about the topic at hand and continues for many words to ensure significant length difference from the baseline."

    baseline = capture(short_fn, PROMPTS, embed=False)
    current = capture(long_fn, PROMPTS, embed=False)
    result = compare(baseline, current)

    ks_test = next(t for t in result.tests if "KS" in t.name)
    assert ks_test.significant
    assert result.any_significant


def test_compare_semantic_drift_detected():
    """Semantically different responses → MMD and centroid distance fire."""
    science_fn = lambda p: "Quantum mechanics describes physics at atomic scales using wave functions."
    cooking_fn = lambda p: "Sauté onions in olive oil until golden, then add garlic and tomatoes."

    baseline = capture(science_fn, PROMPTS[:10], embed=True)
    current = capture(cooking_fn, PROMPTS[:10], embed=True)
    result = compare(baseline, current)

    assert result.mmd is not None
    assert result.centroid_distance is not None
    assert result.centroid_distance > 0.1


def test_compare_changed_examples_populated():
    baseline = capture(lambda p: "Short.", PROMPTS[:5], embed=True)
    current = capture(
        lambda p: "Quantum entanglement is a phenomenon where particles remain connected.",
        PROMPTS[:5], embed=True
    )
    result = compare(baseline, current, n_changed_examples=3)
    assert len(result.changed_examples) <= 3
    for ex in result.changed_examples:
        assert ex.cosine_distance is not None
        assert ex.cosine_distance >= 0.0


# ---------------------------------------------------------------------------
# DriftResult.report()
# ---------------------------------------------------------------------------

def test_report_contains_verdict():
    baseline = capture(lambda p: "Yes.", PROMPTS[:5], embed=False)
    current = capture(
        lambda p: "This is a very different and much longer answer than before.",
        PROMPTS[:5], embed=False
    )
    result = compare(baseline, current)
    report = result.report()
    assert "Drift Report" in report
    assert "n=5" in report
    assert "Length" in report


def test_report_no_drift_message():
    fn = lambda p: "Consistent answer every time."
    snap = capture(fn, PROMPTS[:8], embed=False)
    result = compare(snap, snap)
    report = result.report()
    assert "No significant drift" in report


# ---------------------------------------------------------------------------
# Research Idea 2: calibration curve stored in snapshot
# ---------------------------------------------------------------------------

def test_calibration_bins_stored_when_labels_provided():
    labels = [str(i * 2) for i in range(1, len(PROMPTS) + 1)]
    snap = capture(
        lambda p: "2",
        PROMPTS,
        embed=False,
        labels=labels,
    )
    assert snap.calibration_bins is not None
    assert len(snap.calibration_bins) > 0
    for conf, acc in snap.calibration_bins:
        assert 0.0 <= conf <= 1.0
        assert 0.0 <= acc <= 1.0


def test_calibration_curve_distance_in_compare():
    labels = ["2"] * len(PROMPTS)
    baseline = capture(lambda p: "2", PROMPTS, embed=False, labels=labels)
    # Current model gives wrong answers → different accuracy profile
    current = capture(lambda p: "999", PROMPTS, embed=False, labels=labels)
    result = compare(baseline, current)
    assert result.calibration_curve_distance is not None
