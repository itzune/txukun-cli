#!/usr/bin/env python3
"""Test a specific sentence through cap-punct — show raw model output + constrained output."""
import sys
sys.path.insert(0, '.')
from txukun_lib.cappunct import CapPunctModel as TxukunModel

model = TxukunModel(quiet=True)
model._load()

text = "nire izena xabi da. ea ondo zuzentzen diren akatsak testu honetan"

print(f"INPUT:  {text!r}")
print()

# Test 1: pass as-is
result = model.correct(text)
print(f"OUTPUT: {result!r}")
print()

# Show what changed
import re
def tokenize(s):
    return re.findall(r'(\S+|\s+)', s)

in_tok = tokenize(text)
out_tok = tokenize(result)
print("Token-by-token comparison:")
for i, (a, b) in enumerate(zip(in_tok, out_tok)):
    if a != b:
        print(f"  [{i}] {a!r} → {b!r}")
if len(in_tok) != len(out_tok):
    print(f"  Length mismatch: in={len(in_tok)}, out={len(out_tok)}")
