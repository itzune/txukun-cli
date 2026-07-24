#!/usr/bin/env python3
"""
Evaluation runner for txukun-cli.

Loads the curated eval_dataset.json, runs each case through the full
3-model pipeline (cap-punct + spelling + grammar), and computes
per-category metrics in correct mode (exact string match).

Metrics per category:
  - exact_match:   output == expected (the goal)
  - fixed:         input had errors AND output == expected
  - false_neg:     input had errors AND output == input (didn't fix)
  - over_correct:  input had errors AND output != input AND output != expected
  - false_pos:     (clean only) output != input (introduced errors)

Usage:
    uv run python tests/gec-benchmark/run_eval.py
    uv run python tests/gec-benchmark/run_eval.py --category spelling
    uv run python tests/gec-benchmark/run_eval.py --output results.json
    uv run python tests/gec-benchmark/run_eval.py --verbose
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to path so we can import txukun_lib
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from txukun_lib.analyze import analyze_text
from txukun_lib.errors import apply_corrections


def load_dataset(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Case timed out")


def run_case(input_text: str, models: tuple, timeout: int = 30) -> tuple[str, list, float]:
    """Run one case through the pipeline. Returns (output, errors, elapsed).
    Raises TimeoutError if the case takes longer than `timeout` seconds.
    """
    cappunct, spell, grammar = models
    t0 = time.time()

    # Set per-case timeout (Unix only)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        errors = analyze_text(input_text, cappunct, spell, grammar)
        output = apply_corrections(input_text, errors)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    elapsed = time.time() - t0
    return output, errors, elapsed


def classify_case(case: dict, output: str) -> str:
    """Classify the result of a single case.

    Returns one of: 'exact_match', 'false_neg', 'over_correct', 'false_pos'
    """
    inp = case["input"]
    exp = case["expected"]
    is_clean = case["category"] == "clean"

    if output == exp:
        return "exact_match"
    if is_clean:
        # Clean case that wasn't left alone = false positive
        return "false_pos"
    if output == inp:
        # Had errors but didn't change anything = false negative
        return "false_neg"
    # Changed something but didn't match expected = over-correction
    return "over_correct"


def main():
    parser = argparse.ArgumentParser(description="Run txukun-cli evaluation")
    parser.add_argument(
        "--dataset", type=str,
        default=str(Path(__file__).parent / "eval_dataset.json"),
    )
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Write JSON results to file")
    parser.add_argument("--category", "-c", type=str, default=None,
                        help="Run only one category")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-case details")
    parser.add_argument("--no-models", action="store_true",
                        help="Skip model loading (dry run for debugging)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset(dataset_path)
    cases = dataset["cases"]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
        print(f"Filtering to category '{args.category}': {len(cases)} cases")

    print(f"Loading dataset: {len(cases)} cases\n")

    # ── Load models ──
    models = (None, None, None)
    if not args.no_models:
        print("Loading models (cap-punct + spelling + grammar)...")
        sys.path.insert(0, str(PROJECT_ROOT))
        # Import here so --no-models doesn't require deps
        from txukun import get_models
        t0 = time.time()
        models = get_models({"cap-punct", "spell", "grammar"}, quiet=True)
        # Force-load all models (lazy by default)
        cappunct, spell, grammar = models
        print(f"  Loading cap-punct...", end=" ", flush=True)
        if cappunct and not cappunct.ready:
            cappunct._load()
        print("✓" if cappunct and cappunct.ready else "✗ (skipped)")
        print(f"  Loading grammar...", end=" ", flush=True)
        if grammar and not grammar.ready:
            grammar._load()
        print("✓" if grammar and grammar.ready else "✗ (skipped)")
        print(f"  Loading spelling (BERTeus)...", end=" ", flush=True)
        if spell:
            # SpellChecker loads hunspell eagerly, BERTeus lazily
            if spell._bert and not spell._bert.ready:
                spell._bert._load()
        print("✓" if spell and spell.ready else "✗ (skipped)")
        load_time = time.time() - t0
        print(f"  Models loaded in {load_time:.1f}s\n")
    else:
        print("(dry run — no models)\n")

    # ── Run evaluation ──
    results_by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0, "exact_match": 0, "false_neg": 0,
        "over_correct": 0, "false_pos": 0,
    })
    detailed_results = []
    total_time = 0.0

    print(f"{'ID':25s} {'Category':15s} {'Result':15s} {'Time':>6s}")
    print("-" * 70)

    for i, case in enumerate(cases):
        cat = case["category"]
        cid = case["id"]
        inp = case["input"]
        exp = case["expected"]

        try:
            output, errors, elapsed = run_case(inp, models, timeout=30)
        except TimeoutError:
            print(f"  ⏰ {cid:25s} {cat:15s} TIMEOUT (>30s)", file=sys.stderr)
            output = inp  # treat as unchanged
            errors = []
            elapsed = 30.0
        except Exception as e:
            print(f"  {cid:25s} {cat:15s} ERROR: {e}", file=sys.stderr)
            output = inp  # treat as unchanged
            errors = []
            elapsed = 0.0

        total_time += elapsed
        result = classify_case(case, output)
        results_by_cat[cat]["total"] += 1
        results_by_cat[cat][result] += 1

        detailed_results.append({
            "id": cid,
            "category": cat,
            "input": inp,
            "expected": exp,
            "output": output,
            "result": result,
            "n_errors_expected": len(case.get("errors", [])),
            "n_errors_detected": len(errors),
            "errors_detected": [
                {"from": e.frm, "to": e.to,
                 "original": e.original, "suggestion": e.suggestion,
                 "category": e.category, "confidence": e.confidence}
                for e in errors
            ],
            "elapsed": round(elapsed, 3),
        })

        if args.verbose:
            status_icon = {
                "exact_match": "✅", "false_neg": "⬜",
                "over_correct": "❌", "false_pos": "⚠️"
            }[result]
            print(f"  {status_icon} {cid:23s} {cat:15s} {result:15s} {elapsed:5.2f}s")
            print(f"     input:    {inp[:100]}")
            print(f"     expected: {exp[:100]}")
            print(f"     output:   {output[:100]}")
            if errors:
                for e in errors:
                    print(f"     → {e.original} → {e.suggestion} ({e.category})")
            print()
        else:
            status_icon = {
                "exact_match": "✅", "false_neg": "⬜",
                "over_correct": "❌", "false_pos": "⚠️"
            }[result]
            print(f"  {status_icon} {cid:23s} {cat:15s} {result:15s} {elapsed:5.2f}s")

    # ── Summary report ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Category':20s} {'Total':>5s} {'✅Match':>7s} {'⬜Miss':>7s} "
          f"{'❌Over':>7s} {'⚠️FP':>6s} {'Acc%':>6s}")
    print("-" * 65)

    grand = {"total": 0, "exact_match": 0, "false_neg": 0,
             "over_correct": 0, "false_pos": 0}
    for cat in sorted(results_by_cat.keys()):
        r = results_by_cat[cat]
        for k in grand:
            grand[k] += r[k]
        acc = r["exact_match"] / r["total"] * 100 if r["total"] else 0
        print(f"  {cat:20s} {r['total']:5d} {r['exact_match']:7d} "
              f"{r['false_neg']:7d} {r['over_correct']:7d} {r['false_pos']:6d} "
              f"{acc:5.1f}%")

    print("-" * 65)
    acc = grand["exact_match"] / grand["total"] * 100 if grand["total"] else 0
    print(f"  {'TOTAL':20s} {grand['total']:5d} {grand['exact_match']:7d} "
          f"{grand['false_neg']:7d} {grand['over_correct']:7d} "
          f"{grand['false_pos']:6d} {acc:5.1f}%")

    print(f"\n  Total time: {total_time:.1f}s "
          f"({total_time / len(cases):.2f}s/case avg)")

    # ── Write JSON output ──
    if args.output:
        output_path = Path(args.output)
        report = {
            "summary": {cat: dict(r) for cat, r in results_by_cat.items()},
            "grand_total": grand,
            "total_time": round(total_time, 2),
            "n_cases": len(cases),
            "cases": detailed_results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Detailed results written to {output_path}")


if __name__ == "__main__":
    main()
