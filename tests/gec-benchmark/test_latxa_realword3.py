#!/usr/bin/env python3
"""Latxa-7B real-word error test — corrected labels.

Each case has: (sentence_words, target_idx, word_A, word_B, correct_word, desc)
The model picks whichever word has higher log P(word | prefix).
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

def logp_prefix(prefix, word):
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

# (words, target_idx, wordA, wordB, correct, desc)
cases = [
    # h-dropping: "ura" (water) is correct, "hura" (that) is wrong
    (["egarri","naiz","eta","hura","nahiko","nuke"], 3, "hura", "ura", "ura",
     "thirsty → want water (USER'S EXAMPLE)"),
    (["egarri","naiz","hura","edan","nahi","dut"], 2, "hura", "ura", "ura",
     "thirsty → want to drink water"),
    (["hondartzan","hura","hotza","da"], 1, "hura", "ura", "ura",
     "beach → water is cold"),
    (["ibaian","hura","garbia","da"], 1, "hura", "ura", "ura",
     "river → water is clean"),
    (["edateko","hura","nahi","dut"], 1, "hura", "ura", "ura",
     "to drink → want water"),
    # Controls: "hura" (that one, demonstrative) is correct
    (["gizona","etorri","da","eta","hura","jarraitu","dut"], 4, "hura", "ura", "hura",
     "control: followed HIM (demonstrative)"),
    (["begiratu","hura","orain"], 1, "hura", "ura", "hura",
     "control: look at THAT"),
    (["hura","etorri","da","berriro"], 0, "hura", "ura", "hura",
     "control: HE came again"),
    (["liburu","hau","irakurri","dut"], 1, "hura", "ura", "hura",
     "control: read that book (hura=demonstrative, not water)"),
    # h-dropping: hau → au (standard vs dialectal)
    (["hau","liburua","da"], 0, "hau", "au", "hau",
     "control: THIS is a book (hau=standard)"),
    # Other real-word confusion
    (["ez","naiz","hori","esan"], 2, "hori", "horrela", "horrela",
     "didn't say THAT → didn't say SO"),
    (["baita","hori","ere"], 1, "hori", "horrela", "horrela",
     "not just THAT → not just SO"),
]

print(f"\n{'='*85}")
correct = 0
for words, tidx, wA, wB, expected, desc in cases:
    prefix = " ".join(words[:tidx])
    lpA = logp_prefix(prefix, wA)
    lpB = logp_prefix(prefix, wB)
    pick = wA if lpA > lpB else wB
    ok = pick == expected
    if ok:
        correct += 1
    sym = "✅" if ok else "❌"
    print(f"{sym} {desc}")
    print(f"   \"{prefix}\" → {wA}={lpA:.2f} vs {wB}={lpB:.2f} (Δ={abs(lpA-lpB):.2f}) → picks '{pick}' (correct: '{expected}')")
    print()

print(f"{'='*85}")
print(f"Score: {correct}/{len(cases)} ({100*correct/len(cases):.0f}%)")
