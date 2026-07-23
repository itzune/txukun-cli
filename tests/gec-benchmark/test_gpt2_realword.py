#!/usr/bin/env python3
"""Test GPT2-eus (causal LM) for h-dropping real-word error detection.

GPT2-eus has a proper LM head, so we can compute:
  P("ura" | "egarri naiz eta")  vs  P("hura" | "egarri naiz eta")

This is a more direct approach than BERTeus embedding similarity.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

print("Loading GPT2-eus...")
tokenizer = GPT2TokenizerFast.from_pretrained("HiTZ/gpt2-eus-euscrawl")
model = GPT2LMHeadModel.from_pretrained("HiTZ/gpt2-eus-euscrawl").to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
device = next(model.parameters()).device
print(f"  Loaded on {device} ({model.num_parameters()/1e6:.0f}M params)")

def score_word_continuation(prefix: str, word: str):
    """Compute log P(word | prefix) using the causal LM.

    Tokenizes prefix+word, then sums log-probs of the word's tokens
    given the prefix context.
    """
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"].to(device)
    full_text = prefix + " " + word
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)

    # Word token IDs = full_ids minus prefix_ids
    # (tokenizer may produce different segmentation at boundary, so re-tokenize)
    word_start = prefix_ids.size(1)

    with torch.no_grad():
        logits = model(full_ids).logits  # (1, seq_len, vocab)

    # Log-probs for positions predicting word tokens
    log_probs = torch.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)

    total_logprob = 0.0
    n_tokens = 0
    for pos in range(word_start - 1, full_ids.size(1) - 1):
        target_id = full_ids[0, pos + 1].item()
        total_logprob += log_probs[pos, target_id].item()
        n_tokens += 1

    return total_logprob, n_tokens


cases = [
    {
        "desc": "h-dropping: hura→ura (thirsty context)",
        "prefix": "egarri naiz eta",
        "candidates": ["hura", "ura"],
        "expected": "ura",
    },
    {
        "desc": "Control: ura correct (thirsty, want to drink water)",
        "prefix": "egarri naiz,",
        "candidates": ["ura", "hura"],
        "expected": "ura",
    },
    {
        "desc": "Control: hura correct (demonstrative — followed him)",
        "prefix": "gizona etorri da eta",
        "candidates": ["hura", "ura"],
        "expected": "hura",
    },
    {
        "desc": "h-dropping: hura→ura (beach, cold water)",
        "prefix": "hondartzan",
        "candidates": ["hura", "ura"],
        "expected": "ura",
    },
    {
        "desc": "h-dropping: hura→ura (river context)",
        "prefix": "ibaian",
        "candidates": ["hura", "ura"],
        "expected": "ura",
    },
    {
        "desc": "Control: hura correct (look at that)",
        "prefix": "begiratu",
        "candidates": ["hura", "ura"],
        "expected": "hura",
    },
]

print()
for case in cases:
    print(f"{'='*70}")
    print(f"TEST: {case['desc']}")
    print(f"PREFIX: \"{case['prefix']}\"")
    print(f"EXPECTED: {case['expected']}")
    print(f"{'='*70}")

    results = []
    for cand in case["candidates"]:
        logprob, n_tok = score_word_continuation(case["prefix"], cand)
        # Normalize by token count (per-token log-prob)
        per_token = logprob / n_tok if n_tok > 0 else 0
        results.append((cand, logprob, per_token, n_tok))
        print(f"  {cand:10s}  logP={logprob:8.4f}  per_tok={per_token:8.4f}  ({n_tok} tok)")

    best = max(results, key=lambda r: r[1])
    best_pt = max(results, key=lambda r: r[2])
    print(f"  → GPT2 picks (total): '{best[0]}' (logP={best[1]:.4f})")
    print(f"  → GPT2 picks (per-tok): '{best_pt[0]}' (per_tok={best_pt[2]:.4f})")
    print()
