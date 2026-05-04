"""Unified reliability audit — runs calibration, adversarial, and drift checks in one pass.

Usage:
    from llm_reliability import audit

    result = audit(model_fn, prompts, labels=labels)
    print(result.report())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from llm_reliability.adversarial import AdversarialResult, consistency_score, contradiction_probe
from llm_reliability.calibration import CalibrationResult, consistency as calibration_consistency
from llm_reliability.drift import DriftResult, DriftSnapshot, capture, compare


@dataclass
class AuditResult:
    """Combined result from all three reliability modules."""

    n_prompts: int
    calibration: CalibrationResult | None
    adversarial: AdversarialResult | None
    drift: DriftResult | None

    # Human-readable overall verdict
    passed: bool = True
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            "═" * 58,
            "  LLM Reliability Audit",
            "═" * 58,
            f"  Prompts tested: {self.n_prompts}",
            f"  Modules run:    "
            + ", ".join(
                m
                for m, v in [
                    ("calibration", self.calibration),
                    ("adversarial", self.adversarial),
                    ("drift", self.drift),
                ]
                if v is not None
            ),
        ]

        # --- Calibration ---
        if self.calibration is not None:
            c = self.calibration
            lines += [
                "",
                "  ── Calibration ──────────────────────────────────",
                f"  ECE: {c.ece:.3f}   MCE: {c.mce:.3f}   n={c.n_samples}",
            ]
            if c.ece > 0.15:
                lines.append("  [!] High ECE — model confidence is poorly calibrated.")
            elif c.ece > 0.08:
                lines.append("  [~] Moderate ECE — some miscalibration present.")
            else:
                lines.append("  [✓] ECE within acceptable range.")

        # --- Adversarial ---
        if self.adversarial is not None:
            a = self.adversarial
            lines += [
                "",
                "  ── Adversarial Robustness ───────────────────────",
                f"  Consistency:  {a.overall_consistency:.3f}   Flip rate: {a.flip_rate:.3f}",
            ]
            if a.semantic_stability_distance is not None:
                lines.append(
                    f"  Semantic stability distance: {a.semantic_stability_distance:.2f}"
                    "  (1=surface, 2=lexical, 3=structural)"
                )
            if a.contradiction_rate > 0:
                lines.append(f"  Contradiction rate: {a.contradiction_rate:.3f}")
            if a.flip_rate > 0.3:
                lines.append("  [!] High flip rate — model unstable under perturbation.")
            elif a.flip_rate > 0.1:
                lines.append("  [~] Some sensitivity to perturbations detected.")
            else:
                lines.append("  [✓] Model is stable under perturbations.")

        # --- Drift ---
        if self.drift is not None:
            d = self.drift
            lines += [
                "",
                "  ── Drift Detection ──────────────────────────────",
            ]
            if d.any_significant:
                sig_tests = [t.name for t in d.tests if t.significant]
                lines.append(f"  [!] Drift detected: {', '.join(sig_tests)}")
            else:
                lines.append("  [✓] No significant drift from baseline.")
            if d.calibration_curve_distance is not None:
                lines.append(
                    f"  Calibration curve distance: {d.calibration_curve_distance:.3f}"
                )

        # --- Warnings ---
        if self.warnings:
            lines += ["", "  ── Warnings ─────────────────────────────────────"]
            for w in self.warnings:
                lines.append(f"  [!] {w}")

        # --- Verdict ---
        lines += [
            "",
            "═" * 58,
            f"  Verdict: {'PASS' if self.passed else 'FAIL'}",
            "═" * 58,
        ]
        return "\n".join(lines)


def audit(
    model_fn: Callable[[str], str],
    prompts: list[str],
    labels: list[str] | None = None,
    baseline: DriftSnapshot | None = None,
    run_calibration: bool = True,
    run_adversarial: bool = True,
    run_drift: bool = True,
    adversarial_types: list[str] | None = None,
    n_calibration_samples: int = 5,
    adversarial_n_per_type: int = 1,
    calibration_pass_threshold: float = 0.15,
    flip_rate_pass_threshold: float = 0.3,
    seed: int = 42,
) -> AuditResult:
    """Run a full reliability audit on a model.

    Args:
        model_fn: Any callable (str) -> str, including ModelAdapter instances.
        prompts: Questions / inputs to test.
        labels: Ground-truth answers for calibration accuracy (optional).
                If None, calibration ECE won't be computed.
        baseline: Previous DriftSnapshot for drift comparison (optional).
        run_calibration: Whether to run black-box calibration.
        run_adversarial: Whether to run adversarial consistency checks.
        run_drift: Whether to capture a snapshot and run drift detection.
                   Requires baseline to be set for comparison; otherwise
                   just captures a snapshot and attaches it to the result.
        adversarial_types: Perturbation types. Defaults to all four.
        n_calibration_samples: Samples per question for black-box calibration.
        adversarial_n_per_type: Perturbation variants per type.
        calibration_pass_threshold: ECE above this → fail.
        flip_rate_pass_threshold: Flip rate above this → fail.
        seed: Random seed.

    Returns:
        AuditResult with per-module results and a unified report.
    """
    calibration_result: CalibrationResult | None = None
    adversarial_result: AdversarialResult | None = None
    drift_result: DriftResult | None = None
    warnings: list[str] = []
    passed = True

    if run_calibration:
        if labels is None:
            warnings.append("labels not provided — skipping calibration ECE.")
            run_calibration = False
        else:
            calibration_result = calibration_consistency(
                model_fn, prompts, labels, n_samples=n_calibration_samples
            )
            if calibration_result.ece > calibration_pass_threshold:
                passed = False

    if run_adversarial:
        adversarial_result = consistency_score(
            model_fn,
            prompts,
            types=adversarial_types,
            n_per_type=adversarial_n_per_type,
            seed=seed,
        )
        if labels:
            adversarial_result.contradiction_rate = contradiction_probe(
                model_fn, prompts, labels
            )
        if adversarial_result.flip_rate > flip_rate_pass_threshold:
            passed = False

    if run_drift:
        current_snapshot = capture(model_fn, prompts, embed=True)
        if baseline is not None:
            drift_result = compare(baseline, current_snapshot)
            if drift_result.any_significant:
                passed = False
        else:
            warnings.append(
                "No baseline snapshot provided — drift comparison skipped. "
                "Pass baseline=DriftSnapshot.load(...) to enable drift detection."
            )

    return AuditResult(
        n_prompts=len(prompts),
        calibration=calibration_result,
        adversarial=adversarial_result,
        drift=drift_result,
        passed=passed,
        warnings=warnings,
    )
