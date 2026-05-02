"""Research Idea 3: Semantic Stability Distance.

Measures the minimum perturbation level needed to flip a model's answer.
Level 1 = surface typo/format, Level 2 = synonym swap, Level 3 = word reorder.

Higher mean flip_level = more robust model (stable even under structural changes).
Lower = brittle (surface noise alone changes answers).

Usage:
    OPENAI_API_KEY=sk-... python experiments/run_adversarial.py
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... python experiments/run_adversarial.py --both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_mmlu import load as load_mmlu
from llm_reliability import (
    AnthropicAdapter,
    OpenAIAdapter,
    consistency_score,
    contradiction_probe,
)

RESULTS_DIR = Path(__file__).parent / "results"


def run_adversarial(model_name: str, adapter, questions: list[str], labels: list[str]) -> dict:
    print(f"  [consistency_score] running...", end="", flush=True)
    t0 = time.time()
    try:
        result = consistency_score(
            adapter,
            questions,
            types=["typo", "format", "synonym", "reorder"],
            n_per_type=1,
        )
        elapsed = time.time() - t0
        print(f" consistency={result.overall_consistency:.3f}  ({elapsed:.1f}s)")

        print(f"  [contradiction_probe] running...", end="", flush=True)
        t1 = time.time()
        # Use ground-truth labels as "answers" to probe
        contra_rate = contradiction_probe(adapter, questions, labels)
        elapsed2 = time.time() - t1
        print(f" contradiction_rate={contra_rate:.3f}  ({elapsed2:.1f}s)")

        flip_levels = [r.flip_level for r in result.results if r.flip_level is not None]

        return {
            "overall_consistency": result.overall_consistency,
            "flip_rate": result.flip_rate,
            "semantic_stability_distance": result.semantic_stability_distance,
            "contradiction_rate": contra_rate,
            "flip_level_distribution": {
                "level_1": flip_levels.count(1),
                "level_2": flip_levels.count(2),
                "level_3": flip_levels.count(3),
            },
            "n_questions": len(questions),
            "elapsed_s": round(elapsed + elapsed2, 1),
            "report": result.report(),
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f" ERROR: {e}")
        return {"error": str(e), "elapsed_s": round(elapsed, 1)}


def print_summary(model_name: str, data: dict) -> None:
    print(f"\n{'─'*52}")
    print(f"  {model_name} — Adversarial Robustness")
    print(f"{'─'*52}")
    if "error" in data:
        print(f"  ERROR: {data['error']}")
        return

    print(f"  Overall consistency:     {data['overall_consistency']:.3f}")
    print(f"  Flip rate:               {data['flip_rate']:.3f}")
    ssd = data.get("semantic_stability_distance")
    if ssd is not None:
        print(f"  Semantic stability dist: {ssd:.2f}  (1=surface, 2=lexical, 3=structural)")
    else:
        print(f"  Semantic stability dist: N/A  (no flips observed — fully robust)")
    print(f"  Contradiction rate:      {data['contradiction_rate']:.3f}")

    dist = data.get("flip_level_distribution", {})
    if any(dist.values()):
        print(f"\n  Flip level breakdown:")
        print(f"    Level 1 (typo/format):  {dist.get('level_1', 0)}")
        print(f"    Level 2 (synonym):      {dist.get('level_2', 0)}")
        print(f"    Level 3 (reorder):      {dist.get('level_3', 0)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial robustness experiment")
    parser.add_argument("--n", type=int, default=30, help="Number of MMLU questions (default: 30)")
    parser.add_argument("--both", action="store_true", help="Also run Anthropic Claude")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nResearch Idea 3: Semantic Stability Distance")
    print(f"Questions: {args.n}  |  Seed: {args.seed}")
    print(f"Loading MMLU...", end=" ", flush=True)
    questions, labels = load_mmlu(n=args.n, seed=args.seed)
    print(f"{len(questions)} questions loaded.\n")

    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    models_to_run: list[tuple[str, object]] = []
    models_to_run.append(("gpt-4o-mini", OpenAIAdapter(model="gpt-4o-mini", api_key=openai_key)))

    if args.both:
        if not anthropic_key:
            print("WARNING: --both passed but ANTHROPIC_API_KEY not set. Skipping Claude.")
        else:
            models_to_run.append((
                "claude-haiku-4-5",
                AnthropicAdapter(model="claude-haiku-4-5-20251001", api_key=anthropic_key),
            ))

    all_results: dict[str, dict] = {}

    for model_name, adapter in models_to_run:
        print(f"\nRunning {model_name}...")
        result = run_adversarial(model_name, adapter, questions, labels)
        all_results[model_name] = result
        print_summary(model_name, result)

    # SSD comparison if multiple models
    if len(all_results) > 1:
        print(f"{'─'*52}")
        print("  Semantic Stability Distance Comparison")
        print(f"{'─'*52}")
        for model, data in all_results.items():
            ssd = data.get("semantic_stability_distance")
            ssd_str = f"{ssd:.2f}" if ssd is not None else "∞ (no flips)"
            print(f"  {model:<30} SSD = {ssd_str}")
        print()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"adversarial_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "n_questions": len(questions),
        "seed": args.seed,
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "report"}
            for k, v in all_results.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Results saved → {out_path}")

    # Print full adversarial report for each model
    for model_name, data in all_results.items():
        if "report" in data:
            print(f"\n{model_name} full report:")
            print(data["report"])


if __name__ == "__main__":
    main()
