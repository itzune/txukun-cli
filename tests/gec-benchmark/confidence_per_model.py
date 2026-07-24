#!/usr/bin/env python3
"""
Per-model confidence threshold optimizer for txukun-cli.

Error categories map 1:1 to models:
  - grammar  → GECToR (confidence = P(INCORRECT) from detection head)
  - spelling → BERTeus (confidence = cosine sim, normalized 0–1)
  - cappunct → MarianMT (confidence = LCS match rate)

This script:
  1. Sweeps each model's threshold independently
  2. Combines the best per-model thresholds and tests jointly
  3. Does a full grid search to confirm the global optimum

Usage:
    uv run python tests/gec-benchmark/confidence_per_model.py
    uv run python tests/gec-benchmark/confidence_per_model.py --results /tmp/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

# Category → model name (for display)
MODEL_NAMES = {
    "grammar": "GECToR",
    "spelling": "BERTeus",
    "cappunct": "MarianMT",
}


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


def filter_errors(errors: list[dict], thresholds: dict[str, float]) -> list[dict]:
    """Filter errors by per-category confidence threshold.

    Errors with no confidence are always kept.
    """
    out = []
    for e in errors:
        cat = e["category"]
        conf = e.get("confidence")
        thresh = thresholds.get(cat, 0.0)
        if conf is None or conf >= thresh:
            out.append(e)
    return out


def evaluate(cases: list[dict], thresholds: dict[str, float]) -> dict:
    """Evaluate accuracy with given per-category thresholds."""
    stats = defaultdict(int)
    for case in cases:
        inp = case["input"]
        exp = case["expected"]
        is_clean = case["category"] == "clean"

        applied = filter_errors(case["errors_detected"], thresholds)
        output = apply_subset(inp, applied)
        result = classify(inp, exp, output, is_clean)
        stats[result] += 1
        stats["total"] += 1
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Per-model confidence threshold optimizer")
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
        sys.exit(1)

    # Confidence summary per category
    print("=" * 80)
    print("CONFIDENCE DISTRIBUTION PER MODEL")
    print("=" * 80)
    print(f"{'Model (category)':25s} {'Errors':>6s} {'NoConf':>6s} "
          f"{'Mean':>7s} {'Median':>7s} {'Min':>6s} {'Max':>6s}")
    print("-" * 70)

    for cat, model in MODEL_NAMES.items():
        confs = []
        no_conf = 0
        for case in cases:
            for e in case["errors_detected"]:
                if e["category"] == cat:
                    c = e.get("confidence")
                    if c is None:
                        no_conf += 1
                    else:
                        confs.append(c)
        if confs:
            confs_sorted = sorted(confs)
            mean = sum(confs) / len(confs)
            median = confs_sorted[len(confs_sorted) // 2]
            print(f"  {model:12s} ({cat:8s}) {len(confs):6d} {no_conf:6d} "
                  f"{mean:7.3f} {median:7.3f} {min(confs):6.3f} {max(confs):6.3f}")
        else:
            print(f"  {model:12s} ({cat:8s}) {0:6d} {no_conf:6d}  (no confidence data)")

    # Thresholds to test
    thresholds = [round(t, 2) for t in
                  [i * args.step for i in range(int(1.0 / args.step) + 1)]]

    # ── Phase 1: Independent per-model sweep ──
    print("\n" + "=" * 80)
    print("PHASE 1: Independent per-model threshold sweep")
    print("=" * 80)

    best_per_model: dict[str, tuple[float, float, float]] = {}
    # category -> (best_threshold, best_acc, base_acc)

    for cat, model in MODEL_NAMES.items():
        print(f"\n  {model} ({cat}):")
        print(f"  {'Threshold':>9s} {'✅Match':>7s} {'⬜Miss':>7s} {'❌Over':>7s} "
              f"{'⚠️FP':>6s} {'Acc%':>6s} {'Δ':>7s}")
        print("  " + "-" * 55)

        base_acc = None
        best_acc = -1
        best_thresh = 0.0

        for t in thresholds:
            stats = evaluate(cases, {cat: t})
            acc = stats["exact_match"] / stats["total"] * 100
            if base_acc is None:
                base_acc = acc
            delta = acc - base_acc
            marker = ""
            if acc > best_acc:
                best_acc = acc
                best_thresh = t
                marker = " ◄"
            print(f"  {t:9.2f} {stats['exact_match']:7d} {stats['false_neg']:7d} "
                  f"{stats['over_correct']:7d} {stats['false_pos']:6d} "
                  f"{acc:5.1f}% {delta:+6.1f}%{marker}")

        best_per_model[cat] = (best_thresh, best_acc, base_acc)
        print(f"\n  → Best: threshold={best_thresh:.2f}, "
              f"acc={best_acc:.1f}% (was {base_acc:.1f}%, "
              f"Δ={best_acc - base_acc:+.1f}%)")

    # ── Phase 2: Combined per-model thresholds ──
    print("\n" + "=" * 80)
    print("PHASE 2: Combined per-model thresholds (independent bests)")
    print("=" * 80)

    combined_thresholds = {cat: best_per_model[cat][0] for cat in MODEL_NAMES}
    print(f"\n  Thresholds: " + ", ".join(
        f"{MODEL_NAMES[c]}={combined_thresholds[c]:.2f}" for c in MODEL_NAMES))

    base_stats = evaluate(cases, {c: 0.0 for c in MODEL_NAMES})
    combined_stats = evaluate(cases, combined_thresholds)

    base_acc = base_stats["exact_match"] / base_stats["total"] * 100
    combined_acc = combined_stats["exact_match"] / combined_stats["total"] * 100

    print(f"\n  {'':20s} {'✅Match':>7s} {'⬜Miss':>7s} {'❌Over':>7s} {'⚠️FP':>6s} {'Acc%':>6s}")
    print("  " + "-" * 55)
    print(f"  {'Baseline (no filter)':20s} {base_stats['exact_match']:7d} "
          f"{base_stats['false_neg']:7d} {base_stats['over_correct']:7d} "
          f"{base_stats['false_pos']:6d} {base_acc:5.1f}%")
    print(f"  {'Per-model combined':20s} {combined_stats['exact_match']:7d} "
          f"{combined_stats['false_neg']:7d} {combined_stats['over_correct']:7d} "
          f"{combined_stats['false_pos']:6d} {combined_acc:5.1f}%")
    print(f"\n  Δ = {combined_acc - base_acc:+.1f}%")

    # Per-category breakdown with combined thresholds
    print(f"\n  Per-category with combined thresholds:")
    print(f"  {'Category':20s} {'Base%':>6s} {'Combined%':>9s} {'Δ':>6s}")
    print("  " + "-" * 45)
    for cat in sorted(set(c["category"] for c in cases)):
        cat_cases = [c for c in cases if c["category"] == cat]
        base_exact = 0
        comb_exact = 0
        for case in cat_cases:
            inp = case["input"]
            exp = case["expected"]
            is_clean = case["category"] == "clean"

            base_applied = filter_errors(case["errors_detected"],
                                         {c: 0.0 for c in MODEL_NAMES})
            comb_applied = filter_errors(case["errors_detected"],
                                         combined_thresholds)

            if apply_subset(inp, base_applied) == exp:
                base_exact += 1
            if apply_subset(inp, comb_applied) == exp:
                comb_exact += 1

        base_pct = base_exact / len(cat_cases) * 100
        comb_pct = comb_exact / len(cat_cases) * 100
        print(f"  {cat:20s} {base_pct:5.1f}% {comb_pct:8.1f}% {comb_pct - base_pct:+5.1f}%")

    # ── Phase 3: Full grid search ──
    print("\n" + "=" * 80)
    print("PHASE 3: Full grid search (joint optimization)")
    print("=" * 80)
    print(f"\n  Grid: {len(thresholds)}³ = {len(thresholds)**3} combinations...")

    grid_best_acc = -1
    grid_best_thresholds = {}
    n_combinations = 0

    for tg, ts, tc in product(thresholds, repeat=3):
        n_combinations += 1
        t = {"grammar": tg, "spelling": ts, "cappunct": tc}
        stats = evaluate(cases, t)
        acc = stats["exact_match"]
        if acc > grid_best_acc:
            grid_best_acc = acc
            grid_best_thresholds = t

    grid_stats = evaluate(cases, grid_best_thresholds)
    grid_acc = grid_stats["exact_match"] / grid_stats["total"] * 100

    print(f"\n  Grid search complete ({n_combinations} combinations evaluated)")
    print(f"\n  Optimal thresholds:")
    for cat, model in MODEL_NAMES.items():
        print(f"    {model:12s} ({cat:8s}): {grid_best_thresholds[cat]:.2f}")

    print(f"\n  {'':20s} {'✅Match':>7s} {'⬜Miss':>7s} {'❌Over':>7s} {'⚠️FP':>6s} {'Acc%':>6s}")
    print("  " + "-" * 55)
    print(f"  {'Baseline':20s} {base_stats['exact_match']:7d} "
          f"{base_stats['false_neg']:7d} {base_stats['over_correct']:7d} "
          f"{base_stats['false_pos']:6d} {base_acc:5.1f}%")
    print(f"  {'Independent bests':20s} {combined_stats['exact_match']:7d} "
          f"{combined_stats['false_neg']:7d} {combined_stats['over_correct']:7d} "
          f"{combined_stats['false_pos']:6d} {combined_acc:5.1f}%")
    print(f"  {'Grid optimum':20s} {grid_stats['exact_match']:7d} "
          f"{grid_stats['false_neg']:7d} {grid_stats['over_correct']:7d} "
          f"{grid_stats['false_pos']:6d} {grid_acc:5.1f}%")
    print(f"\n  Δ (grid vs baseline) = {grid_acc - base_acc:+.1f}%")
    print(f"  Δ (grid vs independent) = {grid_acc - combined_acc:+.1f}%")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Baseline accuracy:           {base_acc:.1f}% ({base_stats['exact_match']}/{base_stats['total']})")
    print(f"  Per-model thresholds:        {combined_acc:.1f}% ({combined_stats['exact_match']}/{combined_stats['total']})")
    print(f"  Grid-optimal thresholds:     {grid_acc:.1f}% ({grid_stats['exact_match']}/{grid_stats['total']})")
    print(f"\n  Recommended thresholds:")
    for cat, model in MODEL_NAMES.items():
        print(f"    {model:12s}: {grid_best_thresholds[cat]:.2f}")

    # Output as JSON for easy import
    print(f"\n  JSON config:")
    config = {cat: grid_best_thresholds[cat] for cat in MODEL_NAMES}
    print(f"    {json.dumps(config)}")


if __name__ == "__main__":
    main()
