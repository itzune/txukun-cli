#!/usr/bin/env python3
"""Export a subset of benchmark cases as JSON for browser validation."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from eval import (
    FREQ_PATH, ELHUYAR_DIR, load_correct_sentences, generate_typo_sentences,
    prepare_tier2_cases, load_freq_map,
)

N_CASES = 30

def main():
    fmap = load_freq_map(str(FREQ_PATH))
    correct = list(load_correct_sentences(str(ELHUYAR_DIR / "Dem_none.tsv"))) + \
              list(load_correct_sentences(str(ELHUYAR_DIR / "Dea_none.tsv")))
    typo_cases = generate_typo_sentences(correct, seed=42, typos_per_sentence=1)
    tier2_cases = prepare_tier2_cases(typo_cases, fmap)

    # Load cached BERTeus scores
    with open("tests/gec-benchmark/bert_scores_cache.json") as f:
        cached = json.load(f)

    # Pick a mix: some where BERTeus improved, some worsened, some unchanged
    export = []
    for i in range(min(N_CASES, len(tier2_cases))):
        case = tier2_cases[i]
        entry = {
            "case_idx": i,
            "typo": case.typo,
            "correct": case.correct,
            "sentence_words": case.sentence_words,
            "target_idx": case.target_idx,
            "context": case.context,
            "tier1_correct": case.tier1_correct,
            "tier1_rank": case.tier1_rank,
            "candidates": [
                {"word": c.word, "tier1_score": c.score, "freq": c.freq, "ed": c.ed}
                for c in case.candidates
            ],
            "bert_scores_pll_mean": cached[i]["pll_mean"],
            "bert_scores_pll_sum": cached[i]["pll_sum"],
        }
        export.append(entry)

    out_path = "tests/gec-benchmark/browser_test_cases.json"
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(export)} cases to {out_path}")
    print(f"File size: {os.path.getsize(out_path)/1024:.1f} KB")

if __name__ == "__main__":
    main()
