#!/usr/bin/env python3
"""
Quick analysis of detection strategies from cached scores.

Tries multiple detection conditions beyond the simple margin threshold:
  1. Top-k exclusion: don't flag if actual word is in top-k
  2. Relative margin: flag only if margin exceeds sentence-level baseline
  3. Combined approaches
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
CACHE_PATH = HERE / "detection_scores_cache.json"
ELHUYAR_DIR = HERE / "elhuyar"

import sys
sys.path.insert(0, str(HERE))
from tier1 import load_elhuyar_tsv, find_differences


def load_cache():
    with open(CACHE_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return raw["dem_single"], raw["dem_none"]


def eval_strategy(dem_single_raw, dem_none_raw, dem_single_truth, should_flag_fn, label):
    """Evaluate a flagging strategy."""
    tp = fp = fn = 0
    total_errors = 0
    correct_in_top5 = 0
    correct_in_top1 = 0

    for sent_scores, truth in zip(dem_single_raw, dem_single_truth):
        error_positions = set(t["position"] for t in truth)
        correct_words = {t["position"]: t["correct_word"].lower() for t in truth}
        total_errors += len(error_positions)

        # Compute sentence-level stats for relative margin
        margins = [ws["margin"] for ws in sent_scores]
        sent_median = statistics.median(margins) if margins else 0
        sent_mean = statistics.mean(margins) if margins else 0
        sent_stdev = statistics.stdev(margins) if len(margins) > 1 else 0

        flagged_positions = set()
        for ws in sent_scores:
            if not should_flag_fn(ws, sent_median, sent_mean, sent_stdev):
                continue
            flagged_positions.add(ws["position"])

            if ws["position"] in error_positions:
                tp += 1
                correct_word = correct_words.get(ws["position"], "")
                top5_lower = [w.lower() for w in ws["top5_words"]]
                if correct_word in top5_lower:
                    correct_in_top5 += 1
                if top5_lower and top5_lower[0] == correct_word:
                    correct_in_top1 += 1
            else:
                fp += 1

        for pos in error_positions:
            if pos not in flagged_positions:
                fn += 1

    # FP on clean
    fp_none = 0
    total_checked = 0
    for sent_scores in dem_none_raw:
        margins = [ws["margin"] for ws in sent_scores]
        sent_median = statistics.median(margins) if margins else 0
        sent_mean = statistics.mean(margins) if margins else 0
        sent_stdev = statistics.stdev(margins) if len(margins) > 1 else 0
        for ws in sent_scores:
            total_checked += 1
            if should_flag_fn(ws, sent_median, sent_mean, sent_stdev):
                fp_none += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = fp_none / total_checked if total_checked > 0 else 0

    print(f"  {label:50s}  R={recall:5.1%}  P={precision:5.1%}  F1={f1:5.1%}  "
          f"FP%={fp_rate:5.1%}  TP={tp:3}  FP={fp:4}  top5={correct_in_top5:3}  top1={correct_in_top1:3}")
    return {"label": label, "recall": recall, "precision": precision, "f1": f1,
            "fp_rate": fp_rate, "tp": tp, "fp": fp, "fn": fn,
            "correct_in_top5": correct_in_top5, "correct_in_top1": correct_in_top1}


def main():
    dem_single_raw, dem_none_raw = load_cache()
    dem_single_data = load_elhuyar_tsv(str(ELHUYAR_DIR / "Dem_single.tsv"))
    dem_single_truth = [find_differences(gc["correct"], gc["erroneous"])
                        for gc in dem_single_data]
    total_errors = sum(len(t) for t in dem_single_truth)

    print(f"Loaded: {len(dem_single_raw)} Dem_single, {len(dem_none_raw)} Dem_none, "
          f"{total_errors} ground-truth errors\n")

    print("=" * 120)
    print("STRATEGY COMPARISON")
    print("=" * 120)

    # Strategy 1: Simple margin threshold (baseline)
    for m in [0.05, 0.10, 0.15, 0.20]:
        eval_strategy(
            dem_single_raw, dem_none_raw, dem_single_truth,
            lambda ws, md, mn, sd, m=m: (
                ws["margin"] > m
                and ws["top_word"].lower() != ws["clean_word"].lower()
            ),
            f"1. margin>{m:.2f} + top≠actual"
        )

    print()

    # Strategy 2: Top-k exclusion (don't flag if actual is in top-k)
    for k in [3, 5]:
        for m in [0.05, 0.10, 0.15]:
            eval_strategy(
                dem_single_raw, dem_none_raw, dem_single_truth,
                lambda ws, md, mn, sd, k=k, m=m: (
                    ws["margin"] > m
                    and ws["clean_word"].lower() not in [w.lower() for w in ws["top5_words"][:k]]
                ),
                f"2. margin>{m:.2f} + actual∉top{k}"
            )

    print()

    # Strategy 3: Relative margin (word margin > k * sentence median)
    for mult in [1.5, 2.0, 3.0]:
        eval_strategy(
            dem_single_raw, dem_none_raw, dem_single_truth,
            lambda ws, md, mn, sd, mult=mult: (
                ws["margin"] > mult * md
                and ws["top_word"].lower() != ws["clean_word"].lower()
                and ws["margin"] > 0.05
            ),
            f"3. margin>{mult}×median + >0.05 + top≠actual"
        )

    print()

    # Strategy 4: Z-score (margin > mean + k*std)
    for k in [1.0, 1.5, 2.0]:
        eval_strategy(
            dem_single_raw, dem_none_raw, dem_single_truth,
            lambda ws, md, mn, sd, k=k: (
                sd > 0
                and ws["margin"] > mn + k * sd
                and ws["top_word"].lower() != ws["clean_word"].lower()
                and ws["margin"] > 0.05
            ),
            f"4. margin>mean+{k}σ + >0.05 + top≠actual"
        )

    print()

    # Strategy 5: Combined — top-k exclusion + relative margin
    for k in [5]:
        for mult in [2.0, 3.0]:
            eval_strategy(
                dem_single_raw, dem_none_raw, dem_single_truth,
                lambda ws, md, mn, sd, k=k, mult=mult: (
                    ws["margin"] > mult * md
                    and ws["margin"] > 0.05
                    and ws["clean_word"].lower() not in [w.lower() for w in ws["top5_words"][:k]]
                ),
                f"5. margin>{mult}×median + >0.05 + actual∉top{k}"
            )

    print()

    # Strategy 6: Top-1 in sentence (flag only the highest-margin word per sentence)
    # + absolute threshold
    for m in [0.05, 0.10, 0.15]:
        def flag_top1(sent_scores_raw, truth, threshold):
            """Flag only the highest-margin word in each sentence."""
            if not sent_scores_raw:
                return set()
            best = max(sent_scores_raw, key=lambda ws: ws["margin"])
            if best["margin"] > threshold and best["top_word"].lower() != best["clean_word"].lower():
                return {best["position"]}
            return set()

        # Custom eval for top-1-per-sentence
        tp = fp = fn = 0
        correct_in_top5 = 0
        correct_in_top1 = 0
        for sent_scores, truth in zip(dem_single_raw, dem_single_truth):
            error_positions = set(t["position"] for t in truth)
            correct_words = {t["position"]: t["correct_word"].lower() for t in truth}
            flagged = flag_top1(sent_scores, truth, m)
            for pos in flagged:
                if pos in error_positions:
                    tp += 1
                    cw = correct_words.get(pos, "")
                    ws = next(w for w in sent_scores if w["position"] == pos)
                    top5_lower = [w.lower() for w in ws["top5_words"]]
                    if cw in top5_lower:
                        correct_in_top5 += 1
                    if top5_lower and top5_lower[0] == cw:
                        correct_in_top1 += 1
                else:
                    fp += 1
            for pos in error_positions:
                if pos not in flagged:
                    fn += 1
        fp_none = 0
        total_checked = 0
        for sent_scores in dem_none_raw:
            total_checked += len(sent_scores)
            flagged = flag_top1(sent_scores, [], m)
            fp_none += len(flagged)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fp_rate = fp_none / total_checked if total_checked > 0 else 0
        print(f"  {'6. top-1/sentence + margin>'+str(m)+' + top≠actual':50s}  "
              f"R={recall:5.1%}  P={precision:5.1%}  F1={f1:5.1%}  "
              f"FP%={fp_rate:5.1%}  TP={tp:3}  FP={fp:4}  top5={correct_in_top5:3}  top1={correct_in_top1:3}")


if __name__ == "__main__":
    main()
