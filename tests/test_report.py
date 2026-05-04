"""Tests for Phase 6 unified audit report."""

import pytest
from llm_reliability.report import audit, AuditResult

PROMPTS = [
    "What is the capital of France?",
    "How many planets are in the solar system?",
    "What is the boiling point of water?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
]
LABELS = ["Paris", "8", "100°C", "Shakespeare", "3×10⁸ m/s"]


# ---------------------------------------------------------------------------
# AuditResult.report() format
# ---------------------------------------------------------------------------

def test_report_contains_header():
    result = audit(
        lambda p: "Paris",
        PROMPTS,
        labels=LABELS,
        run_drift=False,
        n_calibration_samples=2,
    )
    report = result.report()
    assert "LLM Reliability Audit" in report
    assert "Verdict" in report


def test_report_shows_calibration_section():
    result = audit(
        lambda p: "Paris",
        PROMPTS,
        labels=LABELS,
        run_drift=False,
        n_calibration_samples=2,
    )
    report = result.report()
    assert "Calibration" in report
    assert "ECE" in report


def test_report_shows_adversarial_section():
    result = audit(
        lambda p: "Paris",
        PROMPTS,
        run_drift=False,
        n_calibration_samples=2,
    )
    report = result.report()
    assert "Adversarial" in report
    assert "Consistency" in report or "consistency" in report.lower()


def test_report_no_drift_section_when_skipped():
    result = audit(
        lambda p: "answer",
        PROMPTS,
        run_drift=False,
        run_calibration=False,
    )
    report = result.report()
    assert "Drift" not in report


def test_report_pass_verdict_stable_model():
    # Model that always gives same answer → perfect consistency → PASS
    # Skip calibration since a constant-answer model won't have low ECE on varied labels
    result = audit(
        lambda p: "Paris",
        PROMPTS,
        run_calibration=False,
        run_drift=False,
    )
    assert result.passed is True
    assert "PASS" in result.report()


# ---------------------------------------------------------------------------
# audit() return shape
# ---------------------------------------------------------------------------

def test_audit_returns_audit_result():
    result = audit(lambda p: "A", PROMPTS, run_drift=False, run_calibration=False)
    assert isinstance(result, AuditResult)


def test_audit_no_calibration_when_no_labels():
    result = audit(
        lambda p: "Paris",
        PROMPTS,
        labels=None,
        run_calibration=True,
        run_drift=False,
    )
    assert result.calibration is None
    assert any("labels not provided" in w for w in result.warnings)


def test_audit_calibration_runs_with_labels():
    result = audit(
        lambda p: "Paris",
        PROMPTS,
        labels=LABELS,
        run_drift=False,
        n_calibration_samples=2,
    )
    assert result.calibration is not None
    assert 0.0 <= result.calibration.ece <= 1.0


def test_audit_adversarial_present_by_default():
    result = audit(lambda p: "answer", PROMPTS, run_drift=False, run_calibration=False)
    assert result.adversarial is not None
    assert 0.0 <= result.adversarial.overall_consistency <= 1.0


def test_audit_skip_adversarial():
    result = audit(
        lambda p: "answer",
        PROMPTS,
        run_adversarial=False,
        run_drift=False,
        run_calibration=False,
    )
    assert result.adversarial is None


def test_audit_n_prompts_correct():
    result = audit(lambda p: "x", PROMPTS[:3], run_drift=False, run_calibration=False)
    assert result.n_prompts == 3


def test_audit_warnings_no_baseline():
    result = audit(
        lambda p: "x",
        PROMPTS,
        run_drift=True,
        run_calibration=False,
        run_adversarial=False,
        baseline=None,
    )
    assert any("baseline" in w.lower() for w in result.warnings)
    assert result.drift is None


def test_audit_drift_compare_with_baseline():
    from llm_reliability.drift import capture

    baseline = capture(lambda p: "Paris", PROMPTS, embed=True)
    result = audit(
        lambda p: "Paris",
        PROMPTS,
        run_calibration=False,
        run_adversarial=False,
        run_drift=True,
        baseline=baseline,
    )
    assert result.drift is not None
    assert not result.drift.any_significant


def test_audit_fail_on_high_flip_rate():
    """A model that changes answers on every variant should fail."""
    answers = ["A", "B", "C", "D"]
    idx = [0]

    def flip_model(p):
        a = answers[idx[0] % len(answers)]
        idx[0] += 1
        return a

    result = audit(
        flip_model,
        PROMPTS,
        run_calibration=False,
        run_drift=False,
        flip_rate_pass_threshold=0.1,
    )
    assert result.passed is False
    assert "FAIL" in result.report()


# ---------------------------------------------------------------------------
# Module-level import
# ---------------------------------------------------------------------------

def test_audit_importable_from_top_level():
    from llm_reliability import audit as top_level_audit, AuditResult as AR
    assert callable(top_level_audit)
    assert AR is AuditResult
