"""llm-reliability CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """LLM reliability toolkit — calibration, drift, and adversarial robustness."""


# ---------------------------------------------------------------------------
# calibrate command
# ---------------------------------------------------------------------------

@main.command()
@click.argument("scores_file", type=click.Path(exists=True))
@click.option("--n-bins", default=10, show_default=True, help="Number of calibration bins.")
@click.option("--plot", type=click.Path(), default=None, help="Save reliability diagram to file.")
@click.option("--output", type=click.Path(), default=None, help="Save results to JSON.")
def calibrate(scores_file: str, n_bins: int, plot: str | None, output: str | None) -> None:
    """Compute ECE and reliability diagram from a scores JSON file.

    SCORES_FILE must be a JSON array of objects with fields:
      - "label": int (1=correct, 0=incorrect) or bool
      - "confidence": float in [0, 1]

    Example:
      [{"label": 1, "confidence": 0.9}, {"label": 0, "confidence": 0.7}, ...]
    """
    from llm_reliability.calibration import calibration_result, reliability_diagram

    data = json.loads(Path(scores_file).read_text())
    if not isinstance(data, list):
        click.echo("Error: scores file must be a JSON array.", err=True)
        sys.exit(1)

    try:
        y_true = [int(d["label"]) for d in data]
        confidences = [float(d["confidence"]) for d in data]
    except (KeyError, TypeError) as e:
        click.echo(f"Error parsing scores file: {e}", err=True)
        sys.exit(1)

    result = calibration_result(y_true, confidences, n_bins=n_bins)

    table = Table(title="Calibration Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("ECE", f"{result.ece:.4f}")
    table.add_row("MCE", f"{result.mce:.4f}")
    table.add_row("Samples", str(result.n_samples))
    table.add_row("Bins", str(result.n_bins))
    console.print(table)

    _print_bin_table(result)

    if plot:
        fig = reliability_diagram(y_true, confidences, n_bins=n_bins)
        fig.savefig(plot, dpi=150, bbox_inches="tight")
        console.print(f"[green]Reliability diagram saved to {plot}[/green]")

    if output:
        out = {
            "ece": result.ece,
            "mce": result.mce,
            "n_samples": result.n_samples,
            "n_bins": result.n_bins,
            "bins": [
                {
                    "lower": b.lower,
                    "upper": b.upper,
                    "mean_confidence": b.mean_confidence,
                    "accuracy": b.accuracy,
                    "count": b.count,
                }
                for b in result.bins
            ],
        }
        Path(output).write_text(json.dumps(out, indent=2))
        console.print(f"[green]Results saved to {output}[/green]")


def _print_bin_table(result) -> None:
    table = Table(title="Per-Bin Stats", show_lines=False)
    table.add_column("Bin", style="dim")
    table.add_column("Mean Conf", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("n", justify="right")
    table.add_column("Gap", justify="right", style="yellow")

    for b in result.bins:
        gap = abs(b.accuracy - b.mean_confidence)
        gap_str = f"{gap:.3f}"
        if gap > 0.15:
            gap_str = f"[red]{gap_str}[/red]"
        elif gap > 0.08:
            gap_str = f"[yellow]{gap_str}[/yellow]"
        else:
            gap_str = f"[green]{gap_str}[/green]"
        table.add_row(
            f"[{b.lower:.1f}, {b.upper:.1f})",
            f"{b.mean_confidence:.3f}",
            f"{b.accuracy:.3f}",
            str(b.count),
            gap_str,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# temperature-scale command
# ---------------------------------------------------------------------------

@main.command("temperature-scale")
@click.argument("logits_file", type=click.Path(exists=True))
def temperature_scale_cmd(logits_file: str) -> None:
    """Find optimal temperature T for logit calibration.

    LOGITS_FILE must be a JSON object with:
      - "logits": list of lists (n_samples x n_classes)
      - "labels": list of int class indices

    Example:
      {"logits": [[2.1, 0.3], [0.5, 1.8]], "labels": [0, 1]}
    """
    from llm_reliability.calibration import temperature_scale, apply_temperature, calibration_result

    data = json.loads(Path(logits_file).read_text())
    logits = data["logits"]
    labels = data["labels"]

    import numpy as np
    from scipy.special import softmax
    raw_probs = softmax(np.array(logits), axis=1)
    raw_confs = raw_probs.max(axis=1)
    raw_correct = (raw_probs.argmax(axis=1) == np.array(labels)).astype(int)
    before = calibration_result(raw_correct, raw_confs)

    T = temperature_scale(logits, labels)
    scaled_probs = apply_temperature(logits, T)
    scaled_confs = scaled_probs.max(axis=1)
    scaled_correct = (scaled_probs.argmax(axis=1) == np.array(labels)).astype(int)
    after = calibration_result(scaled_correct, scaled_confs)

    table = Table(title="Temperature Scaling")
    table.add_column("", style="cyan")
    table.add_column("ECE Before", justify="right")
    table.add_column("ECE After", justify="right")
    table.add_column("Temperature", justify="right")
    table.add_row("Result", f"{before.ece:.4f}", f"{after.ece:.4f}", f"{T:.4f}")
    console.print(table)
