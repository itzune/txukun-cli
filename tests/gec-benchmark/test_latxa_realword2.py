#!/usr/bin/env python3
"""Comprehensive real-word error test with Latxa-7B.

Two scoring methods:
1. Prefix logprob: P(word | prefix) — only uses left context
2. Full-sentence logprob: P(entire sentence) — uses both left AND right context
   (causal LM scores each token given all previous tokens, so right context
    is naturally included when scoring the words AFTER the target)
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

def logp_prefix(prefix, word):
    """log P(word | prefix) — left context only."""
    pids = tok(prefix, return_tensors="pt")["input_ids"].to("cuda:0")
    full = tok(prefix + " " + word, return_tensors="pt")["input_ids"].to("cuda:0")
    wstart = pids.size(1)
    with torch.no_grad():
        logits = model(full).logits
    lp = torch.log_softmax(logits[0], dim=-1)
    total = 0.0
    n = 0
    for pos in range(wstart - 1, full.size(1) - 1):
        total += lp[pos, full[0, pos + 1].item()].item()
        n += 1
    return total, n

def logp_full(sentence):
    """log P(entire sentence) — full sequence score."""
    ids = tok(sentence, return_tensors="pt")["input_ids"].to("cuda:0")
    with torch.no_grad():
        logits = model(ids).logits
    lp = torch.log_softmax(logits[0], dim=-1)
    total = 0.0
    for pos in range(ids.size(1) - 1):
        total += lp[pos, ids[0, pos + 1].item()].item()
    return total, ids.size(1) - 1

def logp_full_excluding_target(sentence_words, target_idx):
    """log P(sentence) but only summing tokens AFTER the target.
    
    This uses both left context (naturally) and right context (the target word
    affects how predictable the following words are).
    
    If "ura" (water) is correct, the following words "nahiko nuke" should be
    MORE predictable than if "hura" (that) is there.
    """
    ids = tok(" ".join(sentence_words), return_tensors="pt")["input_ids"].to("cuda:0")
    with torch.no_grad():
        logits = model(ids).logits
    lp = torch.log_softmax(logits[0], dim=-1)
    
    # Find token positions for words after target
    # Approximate: count tokens for words 0..target_idx
    prefix_words = sentence_words[:target_idx + 1]
    prefix_ids = tok(" ".join(prefix_words), return_tensors="pt")["input_ids"]
    target_end = prefix_ids.size(1) - 1  # -1 for BOS
    
    total = 0.0
    n = 0
    for pos in range(target_end, ids.size(1) - 1):
        total += lp[pos, ids[0, pos + 1].item()].item()
        n += 1
    return total, n

# (wrong_sentence_words, target_idx, wrong_word, right_word, desc)
cases = [
    # h-dropping: hura → ura
    (["egarri","naiz","eta","hura","nahiko","nuke"], 3, "hura", "ura",
     "thirsty → want water (user's example)"),
    (["egarri","naiz","hura","edan","nahi","dut"], 2, "hura", "ura",
     "thirsty → want to drink water"),
    (["hondartzan","hura","hotza","da"], 1, "hura", "ura",
     "beach → water is cold"),
    (["ibaian","hura","garbia","da"], 1, "hura", "ura",
     "river → water is clean"),
    # Controls: hura is correct
    (["gizona","etorri","da","eta","hura","jarraitu","dut"], 4, "hura", "ura",
     "control: followed HIM"),
    (["begiratu","hura","begira"], 1, "hura", "ura",
     "control: look at THAT"),
    (["hura","etorri","da"], 0, "hura", "ura",
     "control: HE came"),
    # h-dropping: hau → au
    (["hau","liburua","da"], 0, "hau", "au",
     "this book"),
    # h-dropping: hori → ori  
    (["hori","ikusi","nuen"], 0, "hori", "ori",
     "saw that"),
    # Other real-word errors: non-h
    (["ez","naiz","hori","esan"], 2, "hori", "horrela",
     "didn't say THAT → didn't say SO"),
]

print(f"\n{'='*90}")
print(f"{'TEST':50s} {'PREFIX':>10s} {'FULL-AFTER':>10s} {'EXPECTED':>10s}")
print(f"{'='*90}")

prefix_correct = 0
after_correct = 0
total = 0

for words, tidx, wrong, right, desc in cases:
    prefix = " ".join(words[:tidx])
    
    # Method 1: prefix logprob
    lp_wrong_p, _ = logp_prefix(prefix, wrong)
    lp_right_p, _ = logp_prefix(prefix, right)
    pick_p = right if lp_right_p > lp_wrong_p else wrong
    
    # Method 2: full-sentence logprob (tokens after target)
    # Build both variants
    wrong_words = list(words)
    wrong_words[tidx] = wrong
    right_words = list(words)
    right_words[tidx] = right
    
    lp_wrong_a, _ = logp_full_excluding_target(wrong_words, tidx)
    lp_right_a, _ = logp_full_excluding_target(right_words, tidx)
    pick_a = right if lp_right_a > lp_wrong_a else wrong
    
    total += 1
    if pick_p == right:
        prefix_correct += 1
    if pick_a == right:
        after_correct += 1
    
    ok_p = "✅" if pick_p == right else "❌"
    ok_a = "✅" if pick_a == right else "❌"
    
    print(f"{desc:50s} {ok_p} {pick_p:5s}    {ok_a} {pick_a:5s}    {right:5s}")
    print(f"  {'  prefix:':50s} {wrong}={lp_wrong_p:.2f} vs {right}={lp_right_p:.2f} (Δ={abs(lp_right_p-lp_wrong_p):.2f})")
    print(f"  {'  after:':50s} {wrong}={lp_wrong_a:.2f} vs {right}={lp_right_a:.2f} (Δ={abs(lp_right_a-lp_wrong_a):.2f})")
    print()

print(f"{'='*90}")
print(f"Prefix method:  {prefix_correct}/{total} ({100*prefix_correct/total:.0f}%)")
print(f"After method:   {after_correct}/{total} ({100*after_correct/total:.0f}%)")
