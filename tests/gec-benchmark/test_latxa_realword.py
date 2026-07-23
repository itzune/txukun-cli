#!/usr/bin/env python3
"""Test Latxa-7B (Llama-based) for h-dropping real-word error detection.

Latxa-7B is a 7B parameter causal LM trained on Basque. If scale solves
the frequency-bias problem that BERTeus (125M) and GPT2-eus (124M) have,
then a larger model is the path forward.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading Latxa-7B-v1.1...")
tok = AutoTokenizer.from_pretrained("HiTZ/latxa-7b-v1.1")
model = AutoModelForCausalLM.from_pretrained(
    "HiTZ/latxa-7b-v1.1",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model.eval()
print(f"  Loaded ({model.num_parameters()/1e9:.1f}B params)")

def logp(prefix, word):
    """log P(word | prefix)"""
    pids = tok(prefix, return_tensors="pt")["input_ids"].to("cuda:0")
    full = tok(prefix + " " + word, return_tensors="pt")["input_ids"].to("cuda:0")
    wstart = pids.size(1)
    with torch.no_grad():
        logits = model(full).logits
    lp = torch.log_softmax(logits[0], dim=-1)
    total = 0.0
    for pos in range(wstart - 1, full.size(1) - 1):
        total += lp[pos, full[0, pos + 1].item()].item()
    return total

cases = [
    # (prefix, word1, word2, expected, desc)
    ("egarri naiz eta",        "hura", "ura", "ura",  "h-dropping: thirsty → water"),
    ("egarri naiz,",            "hura", "ura", "ura",  "h-dropping: thirsty, → water"),
    ("gizona etorri da eta",   "hura", "ura", "hura", "control: demonstrative (followed him)"),
    ("hondartzan",              "hura", "ura", "ura",  "h-dropping: beach → water"),
    ("begiratu",                "hura", "ura", "hura", "control: look at → that"),
    ("ibaian",                  "hura", "ura", "ura",  "h-dropping: river → water"),
    ("edateko",                 "hura", "ura", "ura",  "h-dropping: to drink → water"),
    ("hotza",                   "hura", "ura", "ura",  "h-dropping: cold → water"),
    ("gizona etorri da",       "hura", "ura", "hura", "control: the man came → that"),
    ("liburua irakurri dut",   "hau",  "au",  "??",   "h-dropping: this book"),
    ("etxea",                   "hau",  "au",  "??",   "h-dropping: this house"),
]

print()
correct = 0
total = 0
for prefix, w1, w2, expected, desc in cases:
    l1 = logp(prefix, w1)
    l2 = logp(prefix, w2)
    pick = w1 if l1 > l2 else w2
    margin = abs(l1 - l2)
    if expected != "??":
        total += 1
        ok = pick == expected
        if ok:
            correct += 1
        symbol = "✅" if ok else "❌"
    else:
        symbol = "🔍"
    print(f"{symbol} \"{prefix}\" → {w1}={l1:.2f} vs {w2}={l2:.2f} (Δ={margin:.2f}) → picks '{pick}' (expected {expected})")
    print(f"   {desc}")

print(f"\n{'='*40}")
print(f"Score: {correct}/{total} correct ({100*correct/total:.0f}%)" if total else "")
