"""Research Idea 4 (full): Method Ranking Consistency.

Runs all four black-box calibration methods across multiple models and tasks.
Core question: do the methods agree on which model is better calibrated?
If they disagree, method choice changes conclusions — which one is right?

Models tested (pass --models to override):
  - gpt-4o-mini      (requires OPENAI_API_KEY)
  - claude-haiku-4-5 (requires ANTHROPIC_API_KEY)
  - ollama:llama3    (requires local Ollama)

Tasks:
  - mmlu        factual recall, 10 subjects
  - truthfulqa  adversarial truthfulness (tests overconfidence)
  - hellaswag   commonsense NLI (tests reasoning)

Methods:
  consistency · p_true · semantic_dispersion · semantic_entropy

Analysis:
  - ECE per (model, task, method)
  - Kendall's tau between all method pairs per task
  - Cross-task method stability
  - Decision-rule recommendation

Usage:
  OPENAI_API_KEY=sk-... python experiments/run_idea4_full.py
  OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... python experiments/run_idea4_full.py
  python experiments/run_idea4_full.py --models ollama:llama3,ollama:mistral  # local only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from load_mmlu import load as load_mmlu
from load_truthfulqa import load as load_truthfulqa
from load_hellaswag import load as load_hellaswag
from llm_reliability import (
    AnthropicAdapter,
    OpenAIAdapter,
    OllamaAdapter,
    consistency,
    p_true,
    semantic_dispersion,
    semantic_entropy,
)

RESULTS_DIR = Path(__file__).parent / "results"

TASK_LOADERS = {
    "mmlu": load_mmlu,
    "truthfulqa": load_truthfulqa,
    "hellaswag": load_hellaswag,
}

METHODS = {
    "consistency": consistency,
    "p_true": p_true,
    "semantic_dispersion": semantic_dispersion,
    "semantic_entropy": semantic_entropy,
}


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

def build_adapter(model_spec: str) -> tuple[str, Callable]:
    """Parse 'provider:model' or shorthand → (display_name, adapter)."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    shorthands = {
        "gpt-4o-mini":      ("openai", "gpt-4o-mini"),
        "gpt-4o":           ("openai", "gpt-4o"),
        "claude-haiku":     ("anthropic", "claude-haiku-4-5-20251001"),
        "claude-haiku-4-5": ("anthropic", "claude-haiku-4-5-20251001"),
        "claude-sonnet":    ("anthropic", "claude-sonnet-4-6"),
    }

    if model_spec in shorthands:
        provider, model = shorthands[model_spec]
    elif ":" in model_spec:
        provider, model = model_spec.split(":", 1)
    else:
        raise ValueError(f"Unknown model spec: {model_spec!r}. Use 'provider:model' or a shorthand.")

    if provider == "openai":
        if not openai_key:
            raise EnvironmentError(f"OPENAI_API_KEY not set (needed for {model_spec})")
        return model_spec, OpenAIAdapter(model=model, api_key=openai_key)
    elif provider == "anthropic":
        if not anthropic_key:
            raise EnvironmentError(f"ANTHROPIC_API_KEY not set (needed for {model_spec})")
        return model_spec, AnthropicAdapter(model=model, api_key=anthropic_key)
    elif provider == "ollama":
        return model_spec, OllamaAdapter(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Run one (model × task × method) cell
# ---------------------------------------------------------------------------

def run_cell(
    adapter: Callable,
    questions: list[str],
    labels: list[str],
    method_name: str,
    n_samples: int,
) -> dict:
    method_fn = METHODS[method_name]
    t0 = time.time()
    try:
        if method_name == "semantic_entropy":
            cal = method_fn(adapter, questions, labels, n_responses=n_samples)
        else:
            cal = method_fn(adapter, questions, labels, n_samples=n_samples)
        elapsed = time.time() - t0
        return {
            "ece": cal.ece,
            "mce": cal.mce,
            "n_samples": cal.n_samples,
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {"error": str(e), "elapsed_s": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Analysis: Kendall's tau between method pairs
# ---------------------------------------------------------------------------

def kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall's tau-b between two ECE lists (one value per model)."""
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total if total > 0 else float("nan")


def analyze_rankings(results: dict, model_names: list[str]) -> dict:
    """
    results[model][task][method] = {"ece": float, ...}
    Returns per-task Kendall's tau between all method pairs.
    """
    analysis = {}
    for task in TASK_LOADERS:
        task_analysis: dict = {"method_pairs": {}, "mean_tau": None, "verdict": ""}
        pair_taus = []

        for m1, m2 in combinations(METHODS.keys(), 2):
            eces_m1 = []
            eces_m2 = []
            for model in model_names:
                d1 = results.get(model, {}).get(task, {}).get(m1, {})
                d2 = results.get(model, {}).get(task, {}).get(m2, {})
                if "ece" in d1 and "ece" in d2:
                    eces_m1.append(d1["ece"])
                    eces_m2.append(d2["ece"])

            if len(eces_m1) >= 2:
                tau = kendall_tau(eces_m1, eces_m2)
                task_analysis["method_pairs"][f"{m1} vs {m2}"] = round(tau, 3)
                if not np.isnan(tau):
                    pair_taus.append(tau)

        if pair_taus:
            mean_tau = float(np.mean(pair_taus))
            task_analysis["mean_tau"] = round(mean_tau, 3)
            if mean_tau > 0.7:
                task_analysis["verdict"] = "methods agree — any method reliable for this task"
            elif mean_tau > 0.3:
                task_analysis["verdict"] = "partial agreement — method choice matters"
            else:
                task_analysis["verdict"] = "methods disagree — method choice changes conclusions"

        analysis[task] = task_analysis
    return analysis


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_ece_table(results: dict, model_names: list[str]) -> None:
    col_w = 20
    method_names = list(METHODS.keys())
    tasks = list(TASK_LOADERS.keys())

    print(f"\n{'ECE by (model × task × method)':}")
    print("─" * 90)

    header = f"{'Model':<18} {'Task':<12}" + "".join(f"{m:>{col_w}}" for m in method_names)
    print(header)
    print("─" * 90)

    for model in model_names:
        for task in tasks:
            row = f"{model:<18} {task:<12}"
            for method in method_names:
                cell = results.get(model, {}).get(task, {}).get(method, {})
                if "error" in cell:
                    row += f"{'ERR':>{col_w}}"
                elif "ece" in cell:
                    row += f"{cell['ece']:>{col_w}.3f}"
                else:
                    row += f"{'---':>{col_w}}"
            print(row)
        print()


def print_analysis(analysis: dict) -> None:
    print("\n" + "─" * 70)
    print("  Kendall's τ Between Method Pairs (Research Idea 4)")
    print("─" * 70)

    for task, data in analysis.items():
        print(f"\n  Task: {task}")
        if not data["method_pairs"]:
            print("    (insufficient data — need ≥2 models)")
            continue
        for pair, tau in data["method_pairs"].items():
            bar = "█" * int(abs(tau) * 10) if not np.isnan(tau) else ""
            direction = "+" if tau >= 0 else "-"
            print(f"    {pair:<42}  τ={tau:+.3f}  {bar}")
        if data["mean_tau"] is not None:
            print(f"    {'Mean τ':<42}  {data['mean_tau']:+.3f}")
        print(f"    → {data['verdict']}")

    # Cross-task summary
    mean_taus = [d["mean_tau"] for d in analysis.values() if d.get("mean_tau") is not None]
    if mean_taus:
        overall = float(np.mean(mean_taus))
        print(f"\n  Overall mean τ across tasks: {overall:.3f}")
        print()
        if overall > 0.7:
            print("  FINDING: Methods broadly agree across tasks.")
            print("  Implication: practitioners can trust any single method for ranking.")
        elif overall > 0.3:
            print("  FINDING: Methods partially agree. Task type matters.")
            print("  Implication: use multiple methods; escalate when they disagree.")
        else:
            print("  FINDING: Methods frequently disagree.")
            print("  Implication: method choice can reverse model rankings — dangerous for practitioners.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Idea 4 full experiment: method ranking consistency")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model specs (default: auto-detect from env). "
             "Examples: gpt-4o-mini,claude-haiku  or  ollama:llama3,ollama:mistral",
    )
    parser.add_argument("--tasks", default="mmlu,truthfulqa,hellaswag",
                        help="Comma-separated task names (default: all three)")
    parser.add_argument("--n", type=int, default=50, help="Questions per task (default: 50)")
    parser.add_argument("--n-samples", type=int, default=5, help="Samples per method per question (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to partial results JSON to resume from")
    args = parser.parse_args()

    # Auto-detect available models
    if args.models:
        model_specs = [m.strip() for m in args.models.split(",")]
    else:
        model_specs = []
        if os.environ.get("OPENAI_API_KEY"):
            model_specs.append("gpt-4o-mini")
        if os.environ.get("ANTHROPIC_API_KEY"):
            model_specs.append("claude-haiku")
        if not model_specs:
            print("ERROR: No API keys found. Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY.")
            print("       Or pass --models ollama:llama3 for local inference.")
            sys.exit(1)

    tasks = [t.strip() for t in args.tasks.split(",")]
    unknown = [t for t in tasks if t not in TASK_LOADERS]
    if unknown:
        print(f"ERROR: Unknown tasks: {unknown}. Choose from {list(TASK_LOADERS.keys())}")
        sys.exit(1)

    print(f"\nResearch Idea 4: Method Ranking Consistency")
    print(f"Models:  {', '.join(model_specs)}")
    print(f"Tasks:   {', '.join(tasks)}")
    print(f"n/task:  {args.n}  |  samples/method: {args.n_samples}  |  seed: {args.seed}")

    # Build adapters
    adapters: dict[str, Callable] = {}
    for spec in model_specs:
        try:
            name, adapter = build_adapter(spec)
            adapters[name] = adapter
            print(f"  adapter ready: {name}")
        except EnvironmentError as e:
            print(f"  SKIP {spec}: {e}")

    if not adapters:
        print("ERROR: No adapters could be built.")
        sys.exit(1)

    model_names = list(adapters.keys())

    # Load or resume results
    if args.resume:
        resume_data = json.loads(Path(args.resume).read_text())
        all_results = resume_data.get("results", {})
        print(f"\nResuming from {args.resume}")
    else:
        all_results: dict = {}

    # Load all datasets upfront
    print("\nLoading datasets...")
    datasets: dict[str, tuple[list, list]] = {}
    for task in tasks:
        q, l = TASK_LOADERS[task](n=args.n, seed=args.seed)
        datasets[task] = (q, l)
        print(f"  {task}: {len(q)} questions")

    # Run all cells
    total_cells = len(model_names) * len(tasks) * len(METHODS)
    completed = 0

    print(f"\nRunning {total_cells} cells ({len(model_names)} models × {len(tasks)} tasks × {len(METHODS)} methods)...")
    print("This may take a while. Results are saved incrementally.\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"idea4_full_{timestamp}.json"

    for model_name, adapter in adapters.items():
        all_results.setdefault(model_name, {})
        for task in tasks:
            all_results[model_name].setdefault(task, {})
            questions, labels = datasets[task]
            for method_name in METHODS:
                if method_name in all_results[model_name][task]:
                    completed += 1
                    print(f"  [{completed}/{total_cells}] SKIP  {model_name}/{task}/{method_name} (already done)")
                    continue

                completed += 1
                print(f"  [{completed}/{total_cells}] {model_name}/{task}/{method_name}...", end="", flush=True)
                cell = run_cell(adapter, questions, labels, method_name, args.n_samples)

                if "error" in cell:
                    print(f" ERROR: {cell['error']}")
                else:
                    print(f" ECE={cell['ece']:.3f}  ({cell['elapsed_s']}s)")

                all_results[model_name][task][method_name] = cell

                # Save incrementally after every cell
                payload = {
                    "timestamp": timestamp,
                    "config": {
                        "models": model_names,
                        "tasks": tasks,
                        "n_per_task": args.n,
                        "n_samples": args.n_samples,
                        "seed": args.seed,
                    },
                    "results": all_results,
                }
                out_path.write_text(json.dumps(payload, indent=2))

    # Analysis
    print_ece_table(all_results, model_names)
    analysis = analyze_rankings(all_results, model_names)
    print_analysis(analysis)

    # Save final with analysis
    payload["analysis"] = analysis
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nFull results saved → {out_path}")
    print(f"To resume if interrupted: python experiments/run_idea4_full.py --resume {out_path}")


if __name__ == "__main__":
    main()
