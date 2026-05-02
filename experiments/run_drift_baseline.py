"""Research Idea 2: Calibration Drift as Behavioral Fingerprint.

Step 1 — Capture a baseline snapshot of GPT-4o-mini on MMLU questions.
Step 2 — Capture a "current" snapshot (re-run same questions).
Step 3 — Compare snapshots: report length drift, semantic drift, calibration curve drift.

The calibration_curve_distance metric is the novel one: if two snapshots have identical
response lengths but different calibration curves, something changed in the model's
confidence behavior even if surface statistics look stable.

Run once to establish baseline, then re-run later to detect any silent model updates.

Usage:
    OPENAI_API_KEY=sk-... python experiments/run_drift_baseline.py --mode capture
    OPENAI_API_KEY=sk-... python experiments/run_drift_baseline.py --mode compare --baseline results/baseline_*.json
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
from llm_reliability import OpenAIAdapter
from llm_reliability.drift import DriftSnapshot, capture, compare

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture or compare drift snapshots")
    parser.add_argument(
        "--mode",
        choices=["capture", "compare"],
        default="capture",
        help="capture: save new snapshot; compare: compare two snapshots",
    )
    parser.add_argument("--n", type=int, default=50, help="MMLU questions (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", type=str, help="Path to baseline snapshot JSON (compare mode)")
    parser.add_argument("--current", type=str, help="Path to current snapshot JSON (compare mode, optional)")
    parser.add_argument("--label", type=str, default=None, help="Snapshot label")
    parser.add_argument("--no-embed", action="store_true", help="Skip sentence embeddings (faster)")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key and args.mode == "capture":
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    if args.mode == "capture":
        print(f"\nResearch Idea 2: Capturing Drift Baseline")
        print(f"Questions: {args.n}  |  Embeddings: {not args.no_embed}")
        print(f"Loading MMLU...", end=" ", flush=True)
        questions, labels = load_mmlu(n=args.n, seed=args.seed)
        print(f"{len(questions)} questions loaded.\n")

        adapter = OpenAIAdapter(model="gpt-4o-mini", api_key=openai_key)
        label = args.label or f"gpt-4o-mini-{datetime.now().strftime('%Y%m%d')}"

        print(f"Capturing snapshot (label={label!r})...")
        t0 = time.time()
        snap = capture(
            adapter,
            questions,
            label=label,
            embed=not args.no_embed,
            labels=labels,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Responses: {len(snap.responses)}")
        print(f"  Avg length: {sum(snap.lengths)/len(snap.lengths):.1f} chars")
        print(f"  TTR: {snap.ttr:.3f}")
        print(f"  Embeddings: {'yes' if snap.embeddings is not None else 'no'}")
        if snap.calibration_bins:
            print(f"  Calibration bins: {len(snap.calibration_bins)}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"snapshot_{label}_{timestamp}.json"
        snap.save(out_path)
        print(f"\nSnapshot saved → {out_path}")
        print(f"\nTo compare later, run:")
        print(f"  python experiments/run_drift_baseline.py --mode compare --baseline {out_path}")

    elif args.mode == "compare":
        if not args.baseline:
            print("ERROR: --baseline required in compare mode.")
            sys.exit(1)

        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            print(f"ERROR: baseline file not found: {baseline_path}")
            sys.exit(1)

        print(f"\nResearch Idea 2: Comparing Snapshots")
        baseline = DriftSnapshot.load(baseline_path)
        print(f"  Baseline: {baseline.label!r}  ({baseline.timestamp})")

        if args.current:
            current_path = Path(args.current)
            current = DriftSnapshot.load(current_path)
            print(f"  Current:  {current.label!r}  ({current.timestamp})")
        else:
            # Re-run the model on the same prompts to get a fresh snapshot
            print(f"  No --current provided. Re-running model on same prompts...")
            if not openai_key:
                print("ERROR: OPENAI_API_KEY not set.")
                sys.exit(1)
            adapter = OpenAIAdapter(model="gpt-4o-mini", api_key=openai_key)
            questions, labels = load_mmlu(n=len(baseline.prompts), seed=args.seed)
            current = capture(
                adapter,
                baseline.prompts,  # use same exact prompts
                label=f"gpt-4o-mini-{datetime.now().strftime('%Y%m%d')}-rerun",
                embed=baseline.embeddings is not None,
                labels=labels,
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_path = RESULTS_DIR / f"snapshot_current_{timestamp}.json"
            current.save(current_path)
            print(f"  New snapshot saved → {current_path}")

        print("\nComparing...")
        result = compare(baseline, current)

        print(result.report())

        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"drift_compare_{timestamp}.json"
        comparison_data = {
            "timestamp": timestamp,
            "baseline_label": baseline.label,
            "baseline_timestamp": baseline.timestamp,
            "current_label": current.label,
            "current_timestamp": current.timestamp,
            "any_significant": result.any_significant,
            "calibration_curve_distance": result.calibration_curve_distance,
            "centroid_distance": result.centroid_distance,
            "mmd": result.mmd,
            "tests": [
                {
                    "name": t.name,
                    "statistic": t.statistic,
                    "p_value": t.p_value,
                    "significant": t.significant,
                }
                for t in result.tests
            ],
        }
        out_path.write_text(json.dumps(comparison_data, indent=2))
        print(f"\nComparison saved → {out_path}")


if __name__ == "__main__":
    main()
