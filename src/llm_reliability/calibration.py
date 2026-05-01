"""Calibration metrics and post-hoc correction for LLM confidence scores."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BinStats:
    lower: float
    upper: float
    mean_confidence: float
    accuracy: float
    count: int


@dataclass
class CalibrationResult:
    ece: float
    mce: float
    n_samples: int
    n_bins: int
    bins: list[BinStats] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"ECE:     {self.ece:.4f}",
            f"MCE:     {self.mce:.4f}",
            f"Samples: {self.n_samples}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _bin_stats(
    y_true: np.ndarray,
    confidences: np.ndarray,
    n_bins: int,
) -> list[BinStats]:
    bins: list[BinStats] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if hi == 1.0:
            mask |= confidences == 1.0
        if mask.sum() == 0:
            continue
        bins.append(BinStats(
            lower=lo,
            upper=hi,
            mean_confidence=float(confidences[mask].mean()),
            accuracy=float(y_true[mask].mean()),
            count=int(mask.sum()),
        ))
    return bins


def ece(
    y_true: Sequence[int | bool],
    confidences: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error — mean absolute gap between confidence and accuracy."""
    y = np.asarray(y_true, dtype=float)
    c = np.asarray(confidences, dtype=float)
    if len(y) != len(c):
        raise ValueError("y_true and confidences must have the same length")
    bins = _bin_stats(y, c, n_bins)
    n = len(y)
    return float(sum(b.count / n * abs(b.accuracy - b.mean_confidence) for b in bins))


def mce(
    y_true: Sequence[int | bool],
    confidences: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error — worst-bin gap."""
    y = np.asarray(y_true, dtype=float)
    c = np.asarray(confidences, dtype=float)
    bins = _bin_stats(y, c, n_bins)
    if not bins:
        return 0.0
    return float(max(abs(b.accuracy - b.mean_confidence) for b in bins))


def calibration_result(
    y_true: Sequence[int | bool],
    confidences: Sequence[float],
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute ECE, MCE, and per-bin stats in a single pass."""
    y = np.asarray(y_true, dtype=float)
    c = np.asarray(confidences, dtype=float)
    bins = _bin_stats(y, c, n_bins)
    n = len(y)
    ece_val = float(sum(b.count / n * abs(b.accuracy - b.mean_confidence) for b in bins))
    mce_val = float(max((abs(b.accuracy - b.mean_confidence) for b in bins), default=0.0))
    return CalibrationResult(ece=ece_val, mce=mce_val, n_samples=n, n_bins=n_bins, bins=bins)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def reliability_diagram(
    y_true: Sequence[int | bool],
    confidences: Sequence[float],
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    ax=None,
):
    """Plot a reliability diagram. Returns the matplotlib Figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("pip install matplotlib to use reliability_diagram()")

    y = np.asarray(y_true, dtype=float)
    c = np.asarray(confidences, dtype=float)
    result = calibration_result(y, c, n_bins)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    centers = [(b.lower + b.upper) / 2 for b in result.bins]
    accs = [b.accuracy for b in result.bins]
    widths = [(b.upper - b.lower) * 0.9 for b in result.bins]
    counts = [b.count for b in result.bins]

    ax.bar(centers, accs, width=widths, alpha=0.7, color="#38bdf8", label="Model accuracy")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    for cx, acc, cnt in zip(centers, accs, counts):
        ax.annotate(str(cnt), (cx, acc), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=7, color="gray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE={result.ece:.4f}  MCE={result.mce:.4f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    return fig


# ---------------------------------------------------------------------------
# Verbalized confidence
# ---------------------------------------------------------------------------

_VERBALIZED_TEMPLATE = """\
Answer the following question, then state your confidence that your answer is correct.

Question: {question}

Respond in EXACTLY this format (nothing else):
Answer: <your answer>
Confidence: <integer from 0 to 100>"""


def _parse_verbalized_response(text: str) -> tuple[str | None, float | None]:
    """Extract (answer, confidence) from a verbalized response."""
    answer = None
    confidence = None

    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
        elif line.lower().startswith("confidence:"):
            raw = line.split(":", 1)[1].strip()
            m = re.search(r"(\d+(?:\.\d+)?)", raw)
            if m:
                val = float(m.group(1))
                # Normalize: if > 1, assume it's a percentage
                confidence = val / 100.0 if val > 1.0 else val

    return answer, confidence


def verbalized(
    model_fn: Callable[[str], str],
    questions: list[str],
    labels: list[str],
    template: str = _VERBALIZED_TEMPLATE,
    n_bins: int = 10,
    match_fn: Callable[[str, str], bool] | None = None,
) -> CalibrationResult:
    """Measure calibration of a model's verbalized confidence.

    Args:
        model_fn: Callable that takes a prompt string and returns the model's response.
        questions: List of question strings.
        labels: Ground-truth answer strings, one per question.
        template: Prompt template with `{question}` placeholder.
        n_bins: Number of ECE bins.
        match_fn: How to compare model answer to label. Defaults to case-insensitive substring.

    Returns:
        CalibrationResult with ECE, MCE, and bin stats.
    """
    if len(questions) != len(labels):
        raise ValueError("questions and labels must have the same length")

    if match_fn is None:
        def match_fn(pred: str, label: str) -> bool:
            return label.lower() in pred.lower() or pred.lower() in label.lower()

    y_true: list[int] = []
    confs: list[float] = []
    parse_failures = 0

    for question, label in zip(questions, labels):
        prompt = template.format(question=question)
        response = model_fn(prompt)
        answer, confidence = _parse_verbalized_response(response)

        if answer is None or confidence is None:
            parse_failures += 1
            continue

        correct = int(match_fn(answer, label))
        y_true.append(correct)
        confs.append(np.clip(confidence, 0.0, 1.0))

    if not y_true:
        raise ValueError(
            f"No parseable responses. All {parse_failures} responses failed to parse."
        )

    result = calibration_result(y_true, confs, n_bins)
    return result


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def temperature_scale(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    init_temp: float = 1.5,
) -> float:
    """Find optimal temperature T minimizing NLL of softmax(logits / T).

    Args:
        logits: Shape (n_samples, n_classes) pre-softmax logit arrays.
        labels: Integer class indices, shape (n_samples,).
        init_temp: Starting temperature for optimization.

    Returns:
        Optimal temperature scalar.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import log_softmax

    L = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=int)

    def nll(T: float) -> float:
        if T <= 0:
            return 1e9
        scaled = L / T
        log_probs = log_softmax(scaled, axis=1)
        return -float(log_probs[np.arange(len(y)), y].mean())

    result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
    return float(result.x)


def apply_temperature(logits: Sequence[Sequence[float]], temperature: float) -> np.ndarray:
    """Apply temperature scaling; return calibrated probabilities."""
    from scipy.special import softmax
    L = np.asarray(logits, dtype=float)
    return softmax(L / temperature, axis=1)


# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------

def conformal_threshold(
    nonconformity_scores: Sequence[float],
    alpha: float = 0.05,
) -> float:
    """Compute the conformal prediction threshold from calibration scores.

    Args:
        nonconformity_scores: Scores on held-out calibration set. Higher = more nonconforming.
        alpha: Desired miscoverage rate (0.05 → 95% coverage guarantee).

    Returns:
        Threshold q such that a prediction set with score ≤ q covers 1-alpha of true labels.
    """
    scores = np.asarray(nonconformity_scores, dtype=float)
    n = len(scores)
    # RAPS / standard conformal: ceil((n+1)*(1-alpha)) / n quantile
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level))
