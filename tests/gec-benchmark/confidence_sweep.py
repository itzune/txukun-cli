#!/usr/bin/env python3
"""
Confidence threshold sweep for txukun-cli evaluation.

Loads eval_results.json (from run_eval.py), then simulates applying
ONLY corrections with confidence >= threshold, for various thresholds.
Reports the net effect on accuracy at each threshold.

This answers: "Can we filter out wrong corrections by confidence?"

Usage:
    uv run python tests/gec-benchmark/confidence_sweep.py
    uv run python tests/gec-benchmark/confidence_sweep.py --results /tmp/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def apply_subset(text: str, errors: list[dict]) -> str:
    """Apply a subset of corrections (right-to-left to preserve offsets)."""
    result = text
    for e in sorted(errors, key=lambda e: e["from"], reverse=True):
        result = result[:e["from"]] + e["suggestion"] + result[e["to"]:]
    return result


def classify(input_text: str, expected: str, output: str, is_clean: bool) -> str:
    if output == expected:
        return "exact_match"
    if is_clean:
        return "false_pos"
    if output == input_text:
        return "false_neg"
    return "over_correct"


def main():
    parser = argparse.ArgumentParser(description="Confidence threshold sweep")
    parser.add_argument("--results", type=str, default="/tmp/eval_results.json")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Threshold step size")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: results not found at {results_path}", file=sys.stderr)
        print("Run run_eval.py --output /tmp/eval_results.json first.", file=sys.stderr)
        sys.exit(1)

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    cases = data["cases"]

    # Check we have offsets
    sample = cases[0]["errors_detected"][0] if cases[0]["errors_detected"] else None
    if sample and "from" not in sample:
        print("ERROR: results JSON lacks from/to offsets.", file=sys.stderr)
        print("Re-run run_eval.py (updated version includes offsets).", file=sys.stderr)
        sys.exit(1)

    # Thresholds to sweep: 0.0 (no filter) to 1.0
    thresholds = [round(t, 2) for t in 
                  [i * args.step for i in range(int(1.0 / args.step) + 1)]]

    print("Confidence threshold sweep")
    print(f"  Cases: {len(cases)}")
    print(f"  Total errors: {sum(len(c['errors_detected']) for c in cases)}")
    has_conf = sum(1 for c in cases for e in c["errors_detected"]
                   if e.get("confidence") is not None)
    no_conf = sum(1 for c in cases for e in c["errors_detected"]
                  if e.get("confidence") is None)
    print(f"  Errors with confidence: {has_conf}")
    print(f"  Errors without confidence (kept always): {no_conf}")
    print()

    # For each threshold, simulate
    print(f"{'Threshold':>9s} {'Applied':>7s} {'Suppressed':>10s} "
          f"{'✅Match':>7s} {'⬜Miss':>7s} {'❌Over':>7s} {'⚠️FP':>6s} {'Acc%':>6s} "
          f"{'Δ vs 0.0':>8s}")
    print("-" * 80)

    baseline_acc = None

    for threshold in thresholds:
        stats = defaultdict(int)
        total_applied = 0
        total_suppressed = 0

        for case in cases:
            inp = case["input"]
            exp = case["expected"]
            is_clean = case["category"] == "clean"

            # Filter errors by confidence
            applied = []
            for e in case["errors_detected"]:
                conf = e.get("confidence")
                # Keep if no confidence (can't filter) or above threshold
                if conf is None or conf >= threshold:
                    applied.append(e)
                    total_applied += 1
                else:
                    total_suppressed += 1

            output = apply_subset(inp, applied)
            result = classify(inp, exp, output, is_clean)
            stats[result] += 1
            stats["total"] += 1

        acc = stats["exact_match"] / stats["total"] * 100
        if baseline_acc is None:
            baseline_acc = acc
        delta = acc - baseline_acc

        print(f"  {threshold:7.2f} {total_applied:7d} {total_suppressed:10d} "
              f"{stats['exact_match']:7d} {stats['false_neg']:7d} "
              f"{stats['over_correct']:7d} {stats['false_pos']:6d} "
              f"{acc:5.1f}% {delta:+7.1f}%")

    # Per-category best threshold
    print("\n" + "=" * 80)
    print("PER-CATEGORY: Best threshold (accuracy)")
    print("=" * 80)
    print(f"{'Category':20s} {'Base%':>6s} {'Best%':>6s} {'Best Thresh':>11s} {'Δ':>6s}")
    print("-" * 55)

    for cat in sorted(set(c["category"] for c in cases)):
        cat_cases = [c for c in cases if c["category"] == cat]
        best_acc = -1
        best_thresh = 0.0
        base_acc = -1

        for threshold in thresholds:
            exact = 0
            for case in cat_cases:
                inp = case["input"]
                exp = case["expected"]
                applied = [e for e in case["errors_detected"]
                           if e.get("confidence") is None or
                           e.get("confidence", 0) >= threshold]
                output = apply_subset(inp, applied)
                if output == exp:
                    exact += 1
            acc = exact / len(cat_cases) * 100
            if threshold == 0.0:
                base_acc = acc
            if acc > best_acc:
                best_acc = acc
                best_thresh = threshold

        delta = best_acc - base_acc
        marker = " ◄" if delta > 0 else ""
        print(f"  {cat:20s} {base_acc:5.1f}% {best_acc:5.1f}% "
              f"{best_thresh:9.2f} {delta:+5.1f}%{marker}")


if __name__ == "__main__":
    main()
