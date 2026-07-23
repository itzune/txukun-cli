#!/usr/bin/env python3
"""Test cap-punct model: full paragraph vs sentence-split."""
import sys
sys.path.insert(0, '.')
from txukun_lib.cappunct import CapPunctModel as TxukunModel

model = TxukunModel(quiet=True)
model._load()

# Test: multi-sentence paragraph (what the user would type)
paragraph = "kaixo egun on guztioi gaur ez dut asko urduri faktoria e i te beko irratian entzuten da informazio gehiago web horrian"

print("=== INPUT (one paragraph, 4 sentences, no punctuation) ===")
print(paragraph)
print()

# Test 1: pass the whole paragraph as-is (current behavior)
result_whole = model.correct(paragraph)
print("=== OUTPUT (whole paragraph, current behavior) ===")
print(result_whole)
print()

# Test 2: split into sentences first, correct each, rejoin
import re
# Simple sentence splitter: split on common Basque sentence boundaries
# For this test we know the sentence boundaries
sentences = [
    "kaixo egun on guztioi",
    "gaur ez dut asko urduri",
    "faktoria e i te beko irratian entzuten da",
    "informazio gehiago web horrian",
]
results_split = [model.correct(s) for s in sentences]
result_split = " ".join(results_split)
print("=== OUTPUT (sentence-split) ===")
for s, r in zip(sentences, results_split):
    print(f"  {s!r}")
    print(f"  → {r!r}")
    print()
print("Joined:", result_split)
print()

# Compare
print("=== COMPARISON ===")
print(f"Whole:     {result_whole}")
print(f"Split:     {result_split}")
print()
print(f"Whole has {result_whole.count('.')} periods, {result_whole.count(',')} commas")
print(f"Split has {result_split.count('.')} periods, {result_split.count(',')} commas")
