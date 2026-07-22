#!/usr/bin/env python3
"""
GEC Benchmark — Tier 1 + Tier 2 evaluation harness.

Measures spell-correction accuracy on synthetic typos (generated from
Elhuyar GEC correct sentences) and real-word grammar errors (Elhuyar
Dem_single/Dem_multi). Compares Tier 1 (frequency re-ranking) vs
Tier 2 (LM surprisal re-ranking via futo GGUF model).

Usage:
    uv run --extra bench python tests/gec-benchmark/eval.py
    uv run --extra bench python tests/gec-benchmark/eval.py --no-tier2  # Tier 1 only
    uv run --extra bench python tests/gec-benchmark/eval.py --max-typo-cases 200  # quick run

Prerequisites:
    - data/eu-words-freq.txt (160k wordlist, already in repo)
    - models/eu_futo_v2_nobos.gguf (symlinked to futo-transformer-basque)
    - tests/gec-benchmark/elhuyar/*.tsv (Elhuyar GEC dataset)
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, field

# Add parent dirs to path for imports
HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from tier1 import (
    edits1, levenshtein, match_case, get_ranked_candidates,
    load_freq_map, tokenize, should_check_word,
    load_elhuyar_tsv, load_correct_sentences, find_differences,
    Candidate, SCORE_BETA, SCORE_DELTA,
)
from typo_gen import generate_typo_sentences, TypoCase, TypoEdit

# ── Paths ────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data"
ELHUYAR_DIR = HERE / "elhuyar"
MODEL_PATH = REPO_ROOT / "models" / "eu_futo_v2_nobos.gguf"
FREQ_PATH = DATA_DIR / "eu-words-freq.txt"


# ── Helpers ─────────────────────────────────────────

def pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{n / total * 100:.1f}%"


# ── Tier 1 evaluation ───────────────────────────────

@dataclass
class Tier1Result:
    total: int = 0
    detected: int = 0        # typo word NOT in wordlist
    has_candidates: int = 0
    correct_in_pool: int = 0
    top1: int = 0
    top5: int = 0
    failures: list = field(default_factory=list)


def eval_tier1_spelling(fmap: dict[str, int], typo_cases: list[TypoCase]) -> Tier1Result:
    """Evaluate Tier 1 spelling correction on synthetic typos."""
    r = Tier1Result()

    for tc in typo_cases:
        for edit in tc.edits:
            r.total += 1
            typo_word = edit.typo.lower()
            correct_word = edit.word.lower()

            # Detection
            in_dict = typo_word in fmap
            if not in_dict:
                r.detected += 1

            # Candidate generation
            ranked = get_ranked_candidates(edit.typo, [], fmap)
            if ranked:
                r.has_candidates += 1

            # Correct word in pool?
            correct_idx = -1
            for i, c in enumerate(ranked):
                if c.word.lower() == correct_word:
                    correct_idx = i
                    break
            if correct_idx >= 0:
                r.correct_in_pool += 1
            if correct_idx == 0:
                r.top1 += 1
            if 0 <= correct_idx < 5:
                r.top5 += 1

            if correct_idx != 0:
                r.failures.append({
                    "typo": edit.typo,
                    "correct": edit.word,
                    "type": edit.type,
                    "detected": not in_dict,
                    "correct_in_pool": correct_idx >= 0,
                    "correct_rank": correct_idx,
                    "top3": [(c.word, round(c.score, 2)) for c in ranked[:3]],
                })

    return r


def eval_tier1_false_positives(fmap: dict[str, int], correct_sentences: list[str], label: str):
    """Evaluate false positive rate on correct sentences."""
    total_words = 0
    checked_words = 0
    false_detections = 0
    false_corrections = 0
    samples = []

    for sentence in correct_sentences:
        tokens = tokenize(sentence)
        for i, (word, start, end) in enumerate(tokens):
            prev_word = tokens[i - 1][0] if i > 0 else None
            if not should_check_word(word, prev_word):
                continue
            total_words += 1
            checked_words += 1
            word_lower = word.lower()

            if word_lower not in fmap:
                false_detections += 1
                ranked = get_ranked_candidates(word, [], fmap)
                if ranked:
                    false_corrections += 1
                    if len(samples) < 15:
                        samples.append({
                            "word": word,
                            "suggestion": ranked[0].word,
                            "score": round(ranked[0].score, 2),
                            "sentence": sentence[:60] + "...",
                        })

    print(f"\n{'='*60}")
    print(f"  FALSE POSITIVES ({label})")
    print(f"{'='*60}\n")
    print(f"  Sentences:            {len(correct_sentences)}")
    print(f"  Words checked:        {checked_words}")
    print(f"  False detections:     {false_detections}  ({pct(false_detections, checked_words)})")
    print(f"  False corrections:    {false_corrections}  ({pct(false_corrections, checked_words)})")

    if samples:
        print("\n  --- Sample false corrections ---")
        for s in samples:
            print(f"    {s['word']} → {s['suggestion']} ({s['score']})  | {s['sentence']}")

    return false_detections, false_corrections, checked_words


def eval_tier1_grammar(fmap: dict[str, int], grammar_cases: list[dict], label: str):
    """Evaluate grammar correction baseline (expected ~0% — real-word errors)."""
    total_errors = 0
    detected = 0
    correct_in_pool = 0
    top1 = 0

    for gc in grammar_cases:
        diffs = find_differences(gc["correct"], gc["erroneous"])
        for diff in diffs:
            total_errors += 1
            err_word = diff["erroneous_word"].lower()
            correct_word = diff["correct_word"].lower()

            if err_word not in fmap:
                detected += 1

            ranked = get_ranked_candidates(diff["erroneous_word"], [], fmap)
            correct_idx = -1
            for i, c in enumerate(ranked):
                if c.word.lower() == correct_word:
                    correct_idx = i
                    break
            if correct_idx >= 0:
                correct_in_pool += 1
            if correct_idx == 0:
                top1 += 1

    print(f"\n{'='*60}")
    print(f"  GRAMMAR BASELINE ({label})")
    print(f"{'='*60}\n")
    print(f"  Cases:                {len(grammar_cases)}")
    print(f"  Total error words:    {total_errors}")
    print(f"  Detected (∉ dict):    {detected}  ({pct(detected, total_errors)})")
    print(f"  Correct in pool:      {correct_in_pool}  ({pct(correct_in_pool, total_errors)})")
    print(f"  Top-1 (would fix):    {top1}  ({pct(top1, total_errors)})")
    print(f"  (Expected ~0% — grammar errors are real-word errors,")
    print(f"   not spelling. This is the baseline for Tier 2.5/3.)")

    return total_errors, top1


# ── Tier 2 evaluation ───────────────────────────────

@dataclass
class Tier2Case:
    """A single Tier 2 evaluation case with pre-computed surprisals."""
    typo: str
    correct: str
    context: str
    candidates: list[Candidate]           # Tier 1 ranked candidates
    surprisals: list[float] = field(default_factory=list)  # aligned with candidates
    tier1_correct: bool = False
    tier1_rank: int = -1                  # -1 = correct not in pool


def prepare_tier2_cases(typo_cases: list[TypoCase], fmap: dict[str, int]) -> list[Tier2Case]:
    """Build Tier 2 cases from typo cases. Only includes cases with ≥2 candidates."""
    results = []
    for tc in typo_cases:
        for edit in tc.edits:
            ranked = get_ranked_candidates(edit.typo, [], fmap)
            if len(ranked) < 2:
                continue

            # Context: text before the typo word in the erroneous sentence
            err_words = tc.erroneous.split()
            context_words = err_words[:edit.position]
            context = " ".join(context_words)

            correct_lower = edit.word.lower()
            tier1_rank = -1
            for i, c in enumerate(ranked):
                if c.word.lower() == correct_lower:
                    tier1_rank = i
                    break

            results.append(Tier2Case(
                typo=edit.typo,
                correct=edit.word,
                context=context,
                candidates=ranked[:5],  # MAX_LM_CANDIDATES = 5
                tier1_correct=(tier1_rank == 0),
                tier1_rank=tier1_rank,
            ))
    return results


def compute_surprisals(cases: list[Tier2Case], reranker, quiet: bool = False) -> None:
    """Pre-compute surprisal scores for all candidates in all cases."""
    total_candidates = sum(len(c.candidates) for c in cases)
    done = 0
    t0 = time.time()

    for case in cases:
        candidate_words = [c.word.lower() for c in case.candidates]
        results = reranker.score_candidates(case.context, candidate_words)
        case.surprisals = [r.surprisal for r in results]
        done += len(case.candidates)
        if not quiet and done % 50 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_candidates - done) / rate if rate > 0 else 0
            print(f"  Surprisal: {done}/{total_candidates} ({rate:.1f}/s, ETA {eta:.0f}s)", end="\r")

    if not quiet:
        elapsed = time.time() - t0
        print(f"  Surprisal: {done}/{total_candidates} done in {elapsed:.1f}s" + " " * 30)


def eval_tier2(cases: list[Tier2Case], lm_weight: float) -> dict:
    """Evaluate Tier 2 at a given LM_WEIGHT. Uses pre-computed surprisals."""
    tier1_correct = 0
    tier2_correct = 0
    tier2_changed = 0
    tier2_improved = 0
    tier2_worsened = 0
    lm_fallback = 0

    for case in cases:
        correct_lower = case.correct.lower()

        # Tier 1 top-1
        tier1_top = case.candidates[0].word.lower() if case.candidates else ""
        tier1_ok = tier1_top == correct_lower
        if tier1_ok:
            tier1_correct += 1

        # Tier 2: combined score
        all_zero = all(s == 0 for s in case.surprisals)
        if all_zero:
            lm_fallback += 1

        best_combined = -float("inf")
        tier2_idx = 0
        for j, (cand, surp) in enumerate(zip(case.candidates, case.surprisals)):
            combined = cand.score + lm_weight * surp
            if combined > best_combined:
                best_combined = combined
                tier2_idx = j

        tier2_top = case.candidates[tier2_idx].word.lower() if case.candidates else ""
        tier2_ok = tier2_top == correct_lower
        if tier2_ok:
            tier2_correct += 1

        if tier2_idx != 0:
            tier2_changed += 1
        if not tier1_ok and tier2_ok:
            tier2_improved += 1
        if tier1_ok and not tier2_ok:
            tier2_worsened += 1

    return {
        "total": len(cases),
        "tier1_correct": tier1_correct,
        "tier2_correct": tier2_correct,
        "tier2_changed": tier2_changed,
        "tier2_improved": tier2_improved,
        "tier2_worsened": tier2_worsened,
        "lm_fallback": lm_fallback,
        "net": tier2_improved - tier2_worsened,
    }


def grid_search(cases: list[Tier2Case], weights: list[float]) -> list[dict]:
    """Grid search LM_WEIGHT. Returns list of result dicts."""
    results = []
    for w in weights:
        r = eval_tier2(cases, w)
        r["lm_weight"] = w
        results.append(r)
    return results


# ── Main ────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GEC Benchmark — Tier 1 + Tier 2 evaluation")
    parser.add_argument("--no-tier2", action="store_true", help="Skip Tier 2 (LM re-ranking)")
    parser.add_argument("--max-typo-cases", type=int, default=0, help="Limit typo cases (0 = all)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="GGUF model path")
    parser.add_argument("--weights", type=str, default=None, help="Comma-separated LM_WEIGHT values for grid search")
    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  TXUKUN GEC BENCHMARK — Tier 1 + Tier 2                   ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # ── Load data ──
    print("Loading frequency map (eu-words-freq.txt)...")
    fmap = load_freq_map(str(FREQ_PATH))
    print(f"  {len(fmap)} words loaded\n")

    print("Loading Elhuyar datasets...")
    dem_none = load_correct_sentences(str(ELHUYAR_DIR / "Dem_none.tsv"))
    dem_single = load_elhuyar_tsv(str(ELHUYAR_DIR / "Dem_single.tsv"))
    dem_multi = load_elhuyar_tsv(str(ELHUYAR_DIR / "Dem_multi.tsv"))
    dea_none = load_correct_sentences(str(ELHUYAR_DIR / "Dea_none.tsv"))
    print(f"  Dem_none: {len(dem_none)} correct sentences")
    print(f"  Dem_single: {len(dem_single)} grammar-error sentences")
    print(f"  Dem_multi: {len(dem_multi)} grammar-error sentences")
    print(f"  Dea_none: {len(dea_none)} correct sentences")

    # ── Generate synthetic typos ──
    print("\nGenerating synthetic typos from correct sentences...")
    correct_sentences = list(dem_none) + list(dea_none)
    typo_cases = generate_typo_sentences(correct_sentences, seed=42, typos_per_sentence=1)
    if args.max_typo_cases > 0:
        typo_cases = typo_cases[: args.max_typo_cases]
    print(f"  Generated {len(typo_cases)} typo cases")

    # ── Tier 1: Spelling correction ──
    print(f"\n{'='*60}")
    print("  SPELLING CORRECTION (synthetic typos) — TIER 1")
    print(f"{'='*60}\n")

    t1 = eval_tier1_spelling(fmap, typo_cases)
    print(f"  Total typos:          {t1.total}")
    print(f"  Detected (∉ dict):    {t1.detected}  ({pct(t1.detected, t1.total)})")
    print(f"  Has candidates:       {t1.has_candidates}  ({pct(t1.has_candidates, t1.total)})")
    print(f"  Correct in pool:      {t1.correct_in_pool}  ({pct(t1.correct_in_pool, t1.total)})")
    print(f"  Top-1 accuracy:       {t1.top1}  ({pct(t1.top1, t1.total)})")
    print(f"  Top-5 accuracy:       {t1.top5}  ({pct(t1.top5, t1.total)})")

    print("\n  --- Sample failures (top-1 misses) ---")
    for f in t1.failures[:15]:
        rank = f"#{f['correct_rank'] + 1}" if f["correct_rank"] >= 0 else "NOT IN POOL"
        det = "✓" if f["detected"] else "✗"
        top3_str = ", ".join(f"{w}({s})" for w, s in f["top3"])
        print(f"    [{det}] {f['typo']} → {f['correct']} ({rank}) [{f['type']}]  top3: {top3_str}")
    if len(t1.failures) > 15:
        print(f"    ... and {len(t1.failures) - 15} more failures")

    # ── Tier 1: False positives ──
    eval_tier1_false_positives(fmap, dem_none, "Dem_none — manually reviewed")

    # ── Tier 1: Grammar baseline ──
    eval_tier1_grammar(fmap, dem_single, "Dem_single")
    eval_tier1_grammar(fmap, dem_multi, "Dem_multi")

    # ── Tier 2: LM re-ranking ──
    if args.no_tier2:
        print(f"\n{'='*60}")
        print("  Tier 2 skipped (--no-tier2)")
        print(f"{'='*60}\n")
        return

    if not Path(args.model).exists():
        print(f"\n⚠️  Model not found: {args.model}")
        print("  Skipping Tier 2. Symlink or copy the GGUF to models/eu_futo_v2_nobos.gguf")
        return

    print(f"\n{'='*60}")
    print("  TIER 2 — LM SURPRISAL RE-RANKING")
    print(f"{'='*60}\n")

    # Prepare cases
    tier2_cases = prepare_tier2_cases(typo_cases, fmap)
    tier1_successes = sum(1 for c in tier2_cases if c.tier1_correct)
    fixable = sum(1 for c in tier2_cases if not c.tier1_correct and c.tier1_rank >= 0)
    not_in_pool = sum(1 for c in tier2_cases if c.tier1_rank < 0)
    print(f"  Cases with ≥2 candidates: {len(tier2_cases)}")
    print(f"    Tier 1 successes:       {tier1_successes}")
    print(f"    Fixable (T1✗, in pool): {fixable}")
    print(f"    Not in pool:            {not_in_pool}")

    # Load model and compute surprisals
    print(f"\n  Loading model: {args.model}")
    from lm_rerank import LMReranker
    reranker = LMReranker(args.model)
    reranker.load()
    print("  Model loaded.\n")

    print("  Computing surprisal scores...")
    compute_surprisals(tier2_cases, reranker)

    # Default grid search weights
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
    else:
        weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    # Grid search
    print(f"\n  Grid search LM_WEIGHT:")
    print(f"  {'Weight':>8}  {'T1':>6}  {'T2':>6}  {'Δ':>6}  {'Improved':>8}  {'Worsened':>8}  {'Net':>5}  {'Fallback':>8}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*8}")

    results = grid_search(tier2_cases, weights)
    best = max(results, key=lambda r: r["tier2_correct"])

    for r in results:
        delta = r["tier2_correct"] - r["tier1_correct"]
        delta_str = f"{delta:+d}" if delta != 0 else " 0"
        marker = " ◄ best" if r is best else ""
        print(f"  {r['lm_weight']:>8.1f}  {r['tier1_correct']:>6}  {r['tier2_correct']:>6}  {delta_str:>6}  "
              f"{r['tier2_improved']:>8}  {r['tier2_worsened']:>8}  {r['net']:>+5}  {r['lm_fallback']:>8}{marker}")

    # Detailed results at best weight
    print(f"\n  Best: LM_WEIGHT={best['lm_weight']}")
    print(f"  Tier 1: {best['tier1_correct']}/{best['total']} ({pct(best['tier1_correct'], best['total'])})")
    print(f"  Tier 2: {best['tier2_correct']}/{best['total']} ({pct(best['tier2_correct'], best['total'])})")
    print(f"  Improved: {best['tier2_improved']}  Worsened: {best['tier2_worsened']}  Net: {best['net']:+d}")

    # Also show LM_WEIGHT=1.0 (the browser default) for comparison
    r10 = next((r for r in results if r["lm_weight"] == 1.0), None)
    if r10 and r10 is not best:
        print(f"\n  At LM_WEIGHT=1.0 (browser default):")
        print(f"  Tier 2: {r10['tier2_correct']}/{r10['total']} ({pct(r10['tier2_correct'], r10['total'])})")
        print(f"  Improved: {r10['tier2_improved']}  Worsened: {r10['tier2_worsened']}  Net: {r10['net']:+d}")

    # Show sample Tier 2 failures at best weight
    print(f"\n  --- Sample Tier 2 failures at LM_WEIGHT={best['lm_weight']} ---")
    failures_shown = 0
    for case in tier2_cases:
        if failures_shown >= 10:
            break
        correct_lower = case.correct.lower()
        best_combined = -float("inf")
        tier2_idx = 0
        for j, (cand, surp) in enumerate(zip(case.candidates, case.surprisals)):
            combined = cand.score + best["lm_weight"] * surp
            if combined > best_combined:
                best_combined = combined
                tier2_idx = j
        tier2_top = case.candidates[tier2_idx].word.lower()
        if tier2_top != correct_lower:
            tier1_top = case.candidates[0].word.lower()
            print(f"    {case.typo} → want {case.correct} | T1: {case.candidates[0].word} T2: {case.candidates[tier2_idx].word} | ctx: {case.context[:40]}")
            for j, (cand, surp) in enumerate(zip(case.candidates[:4], case.surprisals[:4])):
                combined = cand.score + best["lm_weight"] * surp
                mark = " ◄" if j == tier2_idx else ""
                print(f"      {cand.word:<15} tier1={cand.score:.2f} surprisal={surp:+.2f} combined={combined:.2f}{mark}")
            failures_shown += 1

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
