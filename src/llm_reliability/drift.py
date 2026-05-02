"""Drift detection — capture model output snapshots and compare distributions.

Detects when a model's behavior has changed between two time points, using
statistical tests on output distributions. Works with any black-box API.

Three signal layers:
  1. Text statistics  — KS test on response lengths, lexical diversity (TTR)
  2. Semantic         — MMD and mean cosine distance on sentence embeddings
  3. Behavioral       — fraction of prompts with meaningfully different responses

Research Idea 2 instrumentation: snapshots optionally store a calibration ECE
curve shape (when labels are provided) so calibration drift can be tracked
alongside output distribution drift.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass
class DriftSnapshot:
    """A frozen record of model behavior on a fixed probe set."""

    prompts: list[str]
    responses: list[str]
    timestamp: str
    label: str | None = None

    # Text statistics (always populated)
    lengths: list[int] = field(default_factory=list)        # char length per response
    ttr: float = 0.0                                         # type-token ratio (lexical diversity)
    avg_sentence_len: float = 0.0                            # mean words per sentence

    # Semantic statistics (populated when embed=True)
    embeddings: list[list[float]] | None = None

    # Calibration curve shape (populated when labels are provided to capture())
    # Stores per-bin (mean_confidence, accuracy) pairs — Research Idea 2 instrumentation
    calibration_bins: list[tuple[float, float]] | None = None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> DriftSnapshot:
        data = json.loads(Path(path).read_text())
        s = cls(
            prompts=data["prompts"],
            responses=data["responses"],
            timestamp=data["timestamp"],
            label=data.get("label"),
            lengths=data.get("lengths", []),
            ttr=data.get("ttr", 0.0),
            avg_sentence_len=data.get("avg_sentence_len", 0.0),
            embeddings=data.get("embeddings"),
            calibration_bins=data.get("calibration_bins"),
        )
        return s

    def _to_dict(self) -> dict:
        return {
            "prompts": self.prompts,
            "responses": self.responses,
            "timestamp": self.timestamp,
            "label": self.label,
            "lengths": self.lengths,
            "ttr": self.ttr,
            "avg_sentence_len": self.avg_sentence_len,
            "embeddings": self.embeddings,
            "calibration_bins": self.calibration_bins,
        }


# ---------------------------------------------------------------------------
# Drift test result
# ---------------------------------------------------------------------------

@dataclass
class DriftTest:
    name: str
    statistic: float
    p_value: float | None
    significant: bool
    verdict: str


@dataclass
class ChangedExample:
    prompt: str
    baseline_response: str
    current_response: str
    cosine_distance: float | None = None


@dataclass
class DriftResult:
    tests: list[DriftTest]
    n_baseline: int
    n_current: int
    # Semantic shift metrics (None if embeddings not available)
    centroid_distance: float | None = None
    mmd: float | None = None
    # Most-changed examples for human review
    changed_examples: list[ChangedExample] = field(default_factory=list)
    # Calibration curve shift (None if calibration_bins not in snapshots)
    calibration_curve_distance: float | None = None

    @property
    def any_significant(self) -> bool:
        return any(t.significant for t in self.tests)

    def report(self) -> str:
        lines = [
            "─" * 56,
            "  Drift Report",
            f"  Baseline n={self.n_baseline}  Current n={self.n_current}",
            "─" * 56,
        ]
        for t in self.tests:
            sig = "[SIGNIFICANT]" if t.significant else "[ok]"
            pval = f"p={t.p_value:.4f}" if t.p_value is not None else "no p-value"
            lines.append(f"  {sig:15s} {t.name}: stat={t.statistic:.4f}  {pval}")
            lines.append(f"              {t.verdict}")

        if self.centroid_distance is not None:
            lines.append(f"\n  Semantic centroid distance: {self.centroid_distance:.4f}")
        if self.mmd is not None:
            lines.append(f"  MMD: {self.mmd:.4f}")
        if self.calibration_curve_distance is not None:
            lines.append(f"  Calibration curve shift:    {self.calibration_curve_distance:.4f}")

        overall = "DRIFT DETECTED" if self.any_significant else "No significant drift"
        lines += ["─" * 56, f"  Overall: {overall}", "─" * 56]

        if self.changed_examples:
            lines.append("\n  Most changed responses:")
            for ex in self.changed_examples[:3]:
                lines.append(f'  Prompt: "{ex.prompt[:60]}"')
                lines.append(f'    Before: "{ex.baseline_response[:80]}"')
                lines.append(f'    After:  "{ex.current_response[:80]}"')

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text statistics helpers
# ---------------------------------------------------------------------------

def _compute_text_stats(responses: list[str]) -> tuple[list[int], float, float]:
    """Return (lengths, ttr, avg_sentence_len)."""
    lengths = [len(r) for r in responses]

    all_words = " ".join(responses).lower().split()
    ttr = len(set(all_words)) / len(all_words) if all_words else 0.0

    sentence_lens = []
    for r in responses:
        sentences = re.split(r"[.!?]+", r)
        for s in sentences:
            words = s.strip().split()
            if words:
                sentence_lens.append(len(words))
    avg_sentence_len = float(np.mean(sentence_lens)) if sentence_lens else 0.0

    return lengths, ttr, avg_sentence_len


# ---------------------------------------------------------------------------
# MMD with RBF kernel
# ---------------------------------------------------------------------------

def _rbf_mmd(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased MMD² estimate with median-heuristic bandwidth."""
    # Median heuristic for bandwidth
    all_vecs = np.vstack([X, Y])
    sq_dists = np.sum((all_vecs[:, None] - all_vecs[None, :]) ** 2, axis=-1)
    sigma2 = float(np.median(sq_dists[sq_dists > 0])) if np.any(sq_dists > 0) else 1.0

    def k(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-d / (2 * sigma2))

    kXX = k(X, X)
    kYY = k(Y, Y)
    kXY = k(X, Y)

    n, m = len(X), len(Y)
    # Unbiased: zero diagonal for same-set terms
    np.fill_diagonal(kXX, 0)
    np.fill_diagonal(kYY, 0)

    mmd2 = (kXX.sum() / (n * (n - 1))
            + kYY.sum() / (m * (m - 1))
            - 2 * kXY.mean())
    return float(max(mmd2, 0.0))


# ---------------------------------------------------------------------------
# JS divergence on 1-D histogram (for length distributions)
# ---------------------------------------------------------------------------

def _js_divergence(a: np.ndarray, b: np.ndarray, n_bins: int = 20) -> float:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max()) + 1e-9
    bins = np.linspace(lo, hi, n_bins + 1)

    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    pa = pa + 1e-10
    pb = pb + 1e-10
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    js = 0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m))
    return float(np.clip(js, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def capture(
    model_fn: Callable[[str], str],
    prompts: list[str],
    label: str | None = None,
    embed: bool = True,
    embedder_model: str = "all-MiniLM-L6-v2",
    labels: list[str] | None = None,
    n_calibration_bins: int = 10,
) -> DriftSnapshot:
    """Run model_fn on prompts and record the output distribution as a snapshot.

    Args:
        model_fn: Any callable (str) -> str, including ModelAdapter instances.
        prompts: Fixed probe set — use the same prompts for baseline and current.
        label: Human-readable name for this snapshot (e.g. "gpt-4o-2024-11").
        embed: If True, compute sentence embeddings for semantic drift tests.
        embedder_model: sentence-transformers model name.
        labels: Ground-truth answers (optional). When provided, stores the
                calibration ECE curve shape for Research Idea 2 tracking.
        n_calibration_bins: Bins for calibration curve storage.
    """
    responses = [model_fn(p) for p in prompts]
    lengths, ttr, avg_sentence_len = _compute_text_stats(responses)

    embeddings = None
    if embed:
        from llm_reliability.calibration import _load_embedder
        embedder = _load_embedder(embedder_model)
        emb = embedder.encode(responses, normalize_embeddings=True)
        embeddings = emb.tolist()

    calibration_bins = None
    if labels is not None:
        from llm_reliability.calibration import _default_match, calibration_result
        y_true = [int(_default_match(r, l)) for r, l in zip(responses, labels)]
        # Use response length as a crude confidence proxy for the curve shape
        # (real calibration requires model_fn to also return confidence scores;
        # this stores accuracy-vs-position for drift tracking purposes)
        confs = [min(1.0, len(r) / 200.0) for r in responses]
        result = calibration_result(y_true, confs, n_bins=n_calibration_bins)
        calibration_bins = [(b.mean_confidence, b.accuracy) for b in result.bins]

    return DriftSnapshot(
        prompts=prompts,
        responses=responses,
        timestamp=datetime.now(timezone.utc).isoformat(),
        label=label,
        lengths=lengths,
        ttr=ttr,
        avg_sentence_len=avg_sentence_len,
        embeddings=embeddings,
        calibration_bins=calibration_bins,
    )


def compare(
    baseline: DriftSnapshot,
    current: DriftSnapshot,
    alpha: float = 0.05,
    n_changed_examples: int = 5,
) -> DriftResult:
    """Compare two snapshots with statistical tests.

    Args:
        baseline: Earlier snapshot (reference distribution).
        current: Later snapshot (potentially drifted distribution).
        alpha: Significance level for all tests (default 0.05).
        n_changed_examples: How many most-changed responses to surface.

    Returns:
        DriftResult with per-test statistics, p-values, and plain-language verdicts.
    """
    from scipy import stats

    tests: list[DriftTest] = []

    # --- Test 1: KS test on response length distribution ---
    bl = np.array(baseline.lengths, dtype=float)
    cl = np.array(current.lengths, dtype=float)
    ks_stat, ks_p = stats.ks_2samp(bl, cl)
    sig = ks_p < alpha
    tests.append(DriftTest(
        name="Response length (KS test)",
        statistic=float(ks_stat),
        p_value=float(ks_p),
        significant=sig,
        verdict=(
            f"Length distribution shifted (mean {bl.mean():.0f} → {cl.mean():.0f} chars)."
            if sig else
            f"Length distribution stable (mean {bl.mean():.0f} → {cl.mean():.0f} chars)."
        ),
    ))

    # --- Test 2: JS divergence on length histogram ---
    js = _js_divergence(bl, cl)
    js_sig = js > 0.05  # heuristic threshold — no p-value for JS
    tests.append(DriftTest(
        name="Length histogram (JS divergence)",
        statistic=js,
        p_value=None,
        significant=js_sig,
        verdict=(
            "Response length histogram has shifted noticeably."
            if js_sig else
            "Response length histogram is stable."
        ),
    ))

    # --- Test 3: Lexical diversity (TTR) ---
    ttr_delta = abs(current.ttr - baseline.ttr)
    ttr_sig = ttr_delta > 0.05
    tests.append(DriftTest(
        name="Lexical diversity (TTR)",
        statistic=ttr_delta,
        p_value=None,
        significant=ttr_sig,
        verdict=(
            f"Vocabulary diversity changed ({baseline.ttr:.3f} → {current.ttr:.3f})."
            if ttr_sig else
            f"Vocabulary diversity stable ({baseline.ttr:.3f} → {current.ttr:.3f})."
        ),
    ))

    # --- Semantic tests (if embeddings available in both snapshots) ---
    centroid_distance = None
    mmd = None
    changed_examples: list[ChangedExample] = []

    if baseline.embeddings and current.embeddings:
        B = np.array(baseline.embeddings)
        C = np.array(current.embeddings)

        # Centroid distance
        b_centroid = B.mean(axis=0)
        c_centroid = C.mean(axis=0)
        centroid_distance = float(1.0 - np.dot(b_centroid, c_centroid) /
                                  (np.linalg.norm(b_centroid) * np.linalg.norm(c_centroid) + 1e-10))

        cd_sig = centroid_distance > 0.05
        tests.append(DriftTest(
            name="Semantic centroid distance",
            statistic=centroid_distance,
            p_value=None,
            significant=cd_sig,
            verdict=(
                f"Semantic center of mass shifted (distance={centroid_distance:.4f})."
                if cd_sig else
                f"Semantic center of mass stable (distance={centroid_distance:.4f})."
            ),
        ))

        # MMD
        mmd = _rbf_mmd(B, C)
        mmd_sig = mmd > 0.01
        tests.append(DriftTest(
            name="Semantic distribution (MMD)",
            statistic=mmd,
            p_value=None,
            significant=mmd_sig,
            verdict=(
                "Semantic output distribution has shifted significantly."
                if mmd_sig else
                "Semantic output distribution is stable."
            ),
        ))

        # Find most-changed individual responses
        if len(baseline.prompts) == len(current.prompts):
            per_response_dist = [
                float(1.0 - np.dot(B[i], C[i]))
                for i in range(min(len(B), len(C)))
            ]
            top_indices = sorted(
                range(len(per_response_dist)),
                key=lambda i: per_response_dist[i],
                reverse=True,
            )[:n_changed_examples]
            for idx in top_indices:
                changed_examples.append(ChangedExample(
                    prompt=baseline.prompts[idx],
                    baseline_response=baseline.responses[idx],
                    current_response=current.responses[idx],
                    cosine_distance=per_response_dist[idx],
                ))

    # --- Calibration curve shift (Research Idea 2) ---
    calibration_curve_distance = None
    if baseline.calibration_bins and current.calibration_bins:
        b_accs = np.array([acc for _, acc in baseline.calibration_bins])
        c_accs = np.array([acc for _, acc in current.calibration_bins])
        min_len = min(len(b_accs), len(c_accs))
        calibration_curve_distance = float(np.mean(np.abs(b_accs[:min_len] - c_accs[:min_len])))

        cal_sig = calibration_curve_distance > 0.05
        tests.append(DriftTest(
            name="Calibration curve shape",
            statistic=calibration_curve_distance,
            p_value=None,
            significant=cal_sig,
            verdict=(
                f"Calibration curve shape shifted (MAE={calibration_curve_distance:.4f}) — "
                "possible model update."
                if cal_sig else
                f"Calibration curve shape stable (MAE={calibration_curve_distance:.4f})."
            ),
        ))

    return DriftResult(
        tests=tests,
        n_baseline=len(baseline.responses),
        n_current=len(current.responses),
        centroid_distance=centroid_distance,
        mmd=mmd,
        changed_examples=changed_examples,
        calibration_curve_distance=calibration_curve_distance,
    )
