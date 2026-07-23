#!/usr/bin/env python3
"""Test if BERTeus can detect/correct h-dropping real-word errors.

"hura" (that one) is valid, but in "egarri naiz eta hura nahiko nuke"
the intended word is "ura" (water). Can BERTeus re-ranking catch this?
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from bert_rerank import BerteusReranker

scorer = BerteusReranker()
scorer.load()

# Test cases: (sentence_words, target_idx, candidates, expected, desc)
cases = [
    {
        "desc": "h-dropping: hura→ura (thirsty context → water)",
        "words": ["egarri", "naiz", "eta", "hura", "nahiko", "nuke"],
        "target_idx": 3,
        "candidates": ["hura", "ura"],
        "expected": "ura",
    },
    {
        "desc": "Control: ura in correct context (water, thirsty)",
        "words": ["egarri", "naiz", "ura", "edan", "nahi", "dut"],
        "target_idx": 2,
        "candidates": ["hura", "ura"],
        "expected": "ura",
    },
    {
        "desc": "Control: hura in correct context (demonstrative — followed him)",
        "words": ["gizona", "etorri", "da", "eta", "hura", "jarraitu", "dut"],
        "target_idx": 4,
        "candidates": ["hura", "ura"],
        "expected": "hura",
    },
    {
        "desc": "h-dropping: hau→au? (context: this book)",
        "words": ["hau", "liburua", "irakurri", "dut"],
        "target_idx": 0,
        "candidates": ["Hau", "au", "hori"],
        "expected": "??",
    },
    {
        "desc": "h-dropping: hori→ori? (context: saw that in the street)",
        "words": ["hori", "ikusi", "nuen", "kalean"],
        "target_idx": 0,
        "candidates": ["hori", "ori"],
        "expected": "??",
    },
    {
        "desc": "h-dropping: hura→ura (beach context → water)",
        "words": ["hondartzan", "hura", "hotza", "da"],
        "target_idx": 1,
        "candidates": ["hura", "ura"],
        "expected": "ura",
    },
]

print()
for case in cases:
    print(f"{'='*70}")
    print(f"TEST: {case['desc']}")
    print(f"SENTENCE: {' '.join(case['words'])}")
    print(f"TARGET: '{case['words'][case['target_idx']]}' (idx={case['target_idx']})")
    print(f"CANDIDATES: {case['candidates']}")
    print(f"EXPECTED: {case['expected']}")
    print(f"{'='*70}")

    scores = scorer.score_candidates(case['words'], case['target_idx'], case['candidates'])
    for bs in scores:
        print(f"  {bs.word:10s}  pll_sum={bs.pll_sum:.6f}  pll_mean={bs.pll_mean:.6f}")

    # pll_mean is the primary metric (used in production)
    best = max(scores, key=lambda s: s.pll_mean)
    margin = best.pll_mean - sorted([s.pll_mean for s in scores])[-2] if len(scores) > 1 else 0
    print(f"  → BERTeus picks: '{best.word}' (pll_mean={best.pll_mean:.6f}, margin={margin:.6f})")
    print()
