"""Research Idea 4: Method Ranking Consistency.

Do all four black-box calibration methods agree on which model is better calibrated?

Run all four methods (consistency, p_true, semantic_dispersion, semantic_entropy)
on GPT-4o-mini against MMLU questions. Compare ECE rankings and raw confidence
distributions to see if method choice affects conclusions.

Usage:
    OPENAI_API_KEY=sk-... python experiments/run_calibration_comparison.py
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... python experiments/run_calibration_comparison.py --both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running from project root or experiments/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_mmlu import load as load_mmlu
from llm_reliability import (
    AnthropicAdapter,
    OpenAIAdapter,
    consistency,
    p_true,
    semantic_dispersion,
    semantic_entropy,
)

RESULTS_DIR = Path(__file__).parent / "results"
METHODS = {
    "consistency": consistency,
    "p_true": p_true,
    "semantic_dispersion": semantic_dispersion,
    "semantic_entropy": semantic_entropy,
}


def run_methods(
    model_name: str,
    model_fn,
    questions: list[str],
    labels: list[str],
    n_samples: int = 5,
) -> dict:
    """Run all four black-box calibration methods. Returns dict with per-method results."""
    results = {}

    for method_name, method_fn in METHODS.items():
        print(f"  [{method_name}] running...", end="", flush=True)
        t0 = time.time()
        try:
            if method_name in ("consistency", "p_true", "semantic_dispersion"):
                cal = method_fn(model_fn, questions, labels, n_samples=n_samples)
            else:
                # semantic_entropy uses n_responses parameter
                cal = semantic_entropy(model_fn, questions, labels, n_responses=n_samples)

            elapsed = time.time() - t0
            results[method_name] = {
                "ece": cal.ece,
                "mce": cal.mce,
                "n_samples": cal.n_samples,
                "raw_scores": cal.raw_scores,
                "elapsed_s": round(elapsed, 1),
            }
            print(f" ECE={cal.ece:.3f}  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            results[method_name] = {"error": str(e), "elapsed_s": round(elapsed, 1)}
            print(f" ERROR: {e}")

    return results


def print_summary(model_name: str, results: dict) -> None:
    """Print a formatted comparison table for one model."""
    print(f"\n{'─'*52}")
    print(f"  {model_name}")
    print(f"{'─'*52}")
    print(f"  {'Method':<22} {'ECE':>6}  {'MCE':>6}  {'Time':>6}")
    print(f"  {'─'*22} {'─'*6}  {'─'*6}  {'─'*6}")

    for method, data in results.items():
        if "error" in data:
            print(f"  {method:<22} {'ERROR':>6}")
        else:
            print(
                f"  {method:<22} {data['ece']:>6.3f}  {data['mce']:>6.3f}  {data['elapsed_s']:>5.1f}s"
            )
    print()


def ranking_agreement(all_results: dict[str, dict]) -> None:
    """Compute rank correlation across methods to check Research Idea 4."""
    import itertools

    models = list(all_results.keys())
    if len(models) < 2:
        print("  (need ≥2 models to compare rankings)")
        return

    print(f"\n{'─'*52}")
    print("  Method Ranking Agreement (Research Idea 4)")
    print(f"{'─'*52}")

    # For each pair of methods, check whether they agree on model ranking
    method_names = list(METHODS.keys())
    agreements = []
    for m1, m2 in itertools.combinations(method_names, 2):
        eces_m1 = []
        eces_m2 = []
        for model in models:
            d1 = all_results[model].get(m1, {})
            d2 = all_results[model].get(m2, {})
            if "ece" in d1 and "ece" in d2:
                eces_m1.append(d1["ece"])
                eces_m2.append(d2["ece"])

        if len(eces_m1) < 2:
            continue

        # Rank correlation (Spearman)
        import numpy as np
        ranks_m1 = np.argsort(np.argsort(eces_m1))
        ranks_m2 = np.argsort(np.argsort(eces_m2))
        rho = np.corrcoef(ranks_m1, ranks_m2)[0, 1]
        agree = "✓ agree" if rho > 0.7 else "✗ disagree"
        print(f"  {m1:<22} vs {m2:<22}: ρ={rho:+.2f}  {agree}")
        agreements.append(rho)

    if agreements:
        import numpy as np
        print(f"\n  Mean rank correlation: {np.mean(agreements):.3f}")
        if np.mean(agreements) > 0.7:
            print("  → Methods broadly agree: any single method is reliable.")
        else:
            print("  → Methods disagree: method choice affects conclusions!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration method comparison")
    parser.add_argument("--n", type=int, default=50, help="Number of MMLU questions (default: 50)")
    parser.add_argument("--n-samples", type=int, default=5, help="Samples per question (default: 5)")
    parser.add_argument("--both", action="store_true", help="Also run Anthropic Claude")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nResearch Idea 4: Method Ranking Consistency")
    print(f"Questions: {args.n}  |  Samples/question: {args.n_samples}  |  Seed: {args.seed}")
    print(f"Loading MMLU...", end=" ", flush=True)
    questions, labels = load_mmlu(n=args.n, seed=args.seed)
    print(f"{len(questions)} questions loaded.\n")

    # Check API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set. Export it and re-run.")
        sys.exit(1)

    models_to_run: list[tuple[str, object]] = []

    print("Building adapters...")
    openai_adapter = OpenAIAdapter(model="gpt-4o-mini", api_key=openai_key)
    models_to_run.append(("gpt-4o-mini", openai_adapter))

    if args.both:
        if not anthropic_key:
            print("WARNING: --both passed but ANTHROPIC_API_KEY not set. Skipping Claude.")
        else:
            claude_adapter = AnthropicAdapter(model="claude-haiku-4-5-20251001", api_key=anthropic_key)
            models_to_run.append(("claude-haiku-4-5", claude_adapter))

    all_results: dict[str, dict] = {}

    for model_name, adapter in models_to_run:
        print(f"\nRunning {model_name}...")
        results = run_methods(
            model_name,
            adapter,
            questions,
            labels,
            n_samples=args.n_samples,
        )
        all_results[model_name] = results
        print_summary(model_name, results)

    # Research Idea 4: ranking agreement
    ranking_agreement(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"calibration_comparison_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "n_questions": len(questions),
        "n_samples": args.n_samples,
        "seed": args.seed,
        "results": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
