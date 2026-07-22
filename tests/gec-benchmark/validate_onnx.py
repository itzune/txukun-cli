#!/usr/bin/env python3
"""Validate ONNX BERTeus against PyTorch scores.

Tests two scoring approaches:
  A) Raw embedding similarity (matches PyTorch BerteusReranker exactly)
  B) Isolated contextual embedding (run model on candidate alone — no embedding file needed)

If A matches PyTorch: ONNX export is valid.
If B also works: we can skip the embedding matrix file in the browser.
"""
import os
import sys
import json
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from eval import (
    FREQ_PATH, ELHUYAR_DIR, load_correct_sentences, generate_typo_sentences,
    prepare_tier2_cases, load_freq_map,
)

ONNX_DIR = "/root/berteus-onnx"
N_VALIDATE = 50  # number of cases to validate


def load_onnx_model():
    """Load ONNX model and return session + input/output names."""
    model_path = os.path.join(ONNX_DIR, "model.onnx")  # use fp32 for validation
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    print(f"  ONNX inputs: {input_names}")
    print(f"  ONNX outputs: {output_names}")
    return sess, input_names, output_names


def load_embedding_matrix():
    """Load word embedding matrix from binary file."""
    path = os.path.join(ONNX_DIR, "word_embeddings_f32.bin")
    emb = np.fromfile(path, dtype=np.float32)
    # Reshape — we know it's (50099, 768) or (50101, 768)
    # Check config for vocab_size
    with open(os.path.join(ONNX_DIR, "embedding_meta.json")) as f:
        meta = json.load(f)
    emb_dim = meta["embedding_dim"]
    # Derive vocab_size from actual file size (meta vocab_size may include
    # special tokens not present in the embedding matrix)
    vocab_size = len(emb) // emb_dim
    emb = emb.reshape(vocab_size, emb_dim)
    print(f"  Embedding matrix: {emb.shape}")
    return emb, meta


def run_bert(sess, input_names, output_names, input_ids, attention_mask, token_type_ids):
    """Run BERT ONNX model and return last_hidden_state."""
    feeds = {}
    if "input_ids" in input_names:
        feeds["input_ids"] = input_ids
    if "attention_mask" in input_names:
        feeds["attention_mask"] = attention_mask
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = token_type_ids
    outputs = sess.run(output_names, feeds)
    return outputs[0]  # last_hidden_state


def score_approach_a(sess, input_names, output_names, tokenizer, emb_matrix, meta,
                     sentence_words, target_idx, candidates):
    """Approach A: Raw embedding similarity (matches PyTorch)."""
    # Build masked sentence
    words = list(sentence_words)
    mask_token = meta["mask_token"]
    mask_id = meta["mask_token_id"]
    if target_idx < len(words):
        words[target_idx] = mask_token
    text = " ".join(words)

    enc = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    token_type_ids = enc.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64)

    last_hidden = run_bert(sess, input_names, output_names, input_ids, attention_mask, token_type_ids)
    # Find mask position
    mask_positions = np.where(input_ids[0] == mask_id)[0]
    if len(mask_positions) == 0:
        return [0.0] * len(candidates)
    mask_pos = mask_positions[0]
    mask_hidden = last_hidden[0, mask_pos]  # (768,)
    mask_norm = mask_hidden / (np.linalg.norm(mask_hidden) + 1e-8)

    scores = []
    for cand in candidates:
        cand_ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
        if len(cand_ids) == 0:
            scores.append(0.0)
            continue
        # pll_sum: cosine with first subword
        first_emb = emb_matrix[cand_ids[0]]
        first_norm = first_emb / (np.linalg.norm(first_emb) + 1e-8)
        cos_first = np.dot(mask_norm, first_norm)

        # pll_mean: cosine with mean of all subwords
        cand_emb = emb_matrix[cand_ids].mean(axis=0)
        cand_norm = cand_emb / (np.linalg.norm(cand_emb) + 1e-8)
        cos_mean = np.dot(mask_norm, cand_norm)

        scores.append((cos_first, cos_mean))
    return scores


def score_approach_b(sess, input_names, output_names, tokenizer, meta,
                     sentence_words, target_idx, candidates):
    """Approach B: Isolated contextual embedding (no embedding file needed).

    Run BERT on each candidate alone, take mean of token outputs.
    """
    words = list(sentence_words)
    mask_token = meta["mask_token"]
    mask_id = meta["mask_token_id"]
    if target_idx < len(words):
        words[target_idx] = mask_token
    text = " ".join(words)

    enc = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    token_type_ids = enc.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64)

    last_hidden = run_bert(sess, input_names, output_names, input_ids, attention_mask, token_type_ids)
    mask_positions = np.where(input_ids[0] == mask_id)[0]
    if len(mask_positions) == 0:
        return [0.0] * len(candidates)
    mask_pos = mask_positions[0]
    mask_hidden = last_hidden[0, mask_pos]
    mask_norm = mask_hidden / (np.linalg.norm(mask_hidden) + 1e-8)

    scores = []
    for cand in candidates:
        # Run BERT on candidate alone
        enc_cand = tokenizer(cand, return_tensors="np", truncation=True, max_length=512)
        cand_ids = enc_cand["input_ids"].astype(np.int64)
        cand_mask = enc_cand["attention_mask"].astype(np.int64)
        cand_type = enc_cand.get("token_type_ids", np.zeros_like(cand_ids)).astype(np.int64)

        cand_hidden = run_bert(sess, input_names, output_names, cand_ids, cand_mask, cand_type)
        # Mean of non-special token positions (exclude CLS=1, SEP=2)
        # Use attention mask but exclude first and last
        token_mask = cand_mask[0].astype(bool).copy()
        token_mask[0] = False  # CLS
        # Find last True (SEP) and exclude
        true_indices = np.where(token_mask)[0]
        if len(true_indices) > 0:
            token_mask[true_indices[-1]] = False  # SEP
        if token_mask.sum() == 0:
            scores.append((0.0, 0.0))
            continue
        cand_mean = cand_hidden[0][token_mask].mean(axis=0)
        cand_norm = cand_mean / (np.linalg.norm(cand_mean) + 1e-8)
        cos_sim = np.dot(mask_norm, cand_norm)

        # For approach B, both "sum" and "mean" are the same (we use mean of tokens)
        scores.append((cos_sim, cos_sim))
    return scores


def main():
    print("=== Loading ONNX model + embeddings ===")
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(ONNX_DIR)
    sess, input_names, output_names = load_onnx_model()
    emb_matrix, meta = load_embedding_matrix()

    print("\n=== Regenerating Tier 2 cases ===")
    fmap = load_freq_map(str(FREQ_PATH))
    dem_none = load_correct_sentences(str(ELHUYAR_DIR / "Dem_none.tsv"))
    dea_none = load_correct_sentences(str(ELHUYAR_DIR / "Dea_none.tsv"))
    correct = list(dem_none) + list(dea_none)
    typo_cases = generate_typo_sentences(correct, seed=42, typos_per_sentence=1)
    tier2_cases = prepare_tier2_cases(typo_cases, fmap)
    print(f"  Total cases: {len(tier2_cases)}")

    # Load cached PyTorch scores
    with open("tests/gec-benchmark/bert_scores_cache.json") as f:
        cached = json.load(f)

    # Validate on first N cases
    print(f"\n=== Validating {N_VALIDATE} cases ===")
    max_diff_a_mean = 0
    max_diff_a_sum = 0
    max_diff_b = 0
    corr_a_mean = []
    corr_b_mean = []

    for i in range(min(N_VALIDATE, len(tier2_cases))):
        case = tier2_cases[i]
        cands = [c.word for c in case.candidates]

        # Approach A (raw embeddings)
        scores_a = score_approach_a(
            sess, input_names, output_names, tokenizer, emb_matrix, meta,
            case.sentence_words, case.target_idx, cands
        )

        # Approach B (isolated contextual)
        scores_b = score_approach_b(
            sess, input_names, output_names, tokenizer, meta,
            case.sentence_words, case.target_idx, cands
        )

        # Compare to cached PyTorch
        cached_mean = cached[i]["pll_mean"]
        cached_sum = cached[i]["pll_sum"]

        for j, (sa, sb) in enumerate(zip(scores_a, scores_b)):
            diff_a_mean = abs(sa[1] - cached_mean[j])
            diff_a_sum = abs(sa[0] - cached_sum[j])
            diff_b = abs(sb[1] - cached_mean[j])
            max_diff_a_mean = max(max_diff_a_mean, diff_a_mean)
            max_diff_a_sum = max(max_diff_a_sum, diff_a_sum)
            max_diff_b = max(max_diff_b, diff_b)
            corr_a_mean.append((cached_mean[j], sa[1]))
            corr_b_mean.append((cached_mean[j], sb[1]))

    print(f"\n=== Results ===")
    print(f"Approach A (raw embeddings) vs PyTorch:")
    print(f"  Max diff (pll_mean): {max_diff_a_mean:.6f}")
    print(f"  Max diff (pll_sum):  {max_diff_a_sum:.6f}")
    if max_diff_a_mean < 0.001:
        print(f"  ✅ ONNX matches PyTorch (Approach A valid)")
    else:
        print(f"  ⚠️  ONNX differs from PyTorch")

    print(f"\nApproach B (isolated contextual) vs PyTorch:")
    print(f"  Max diff: {max_diff_b:.6f}")
    # Check correlation even if absolute values differ
    cached_vals = [x[0] for x in corr_b_mean]
    onnx_b_vals = [x[1] for x in corr_b_mean]
    if len(cached_vals) > 2:
        corr = np.corrcoef(cached_vals, onnx_b_vals)[0, 1]
        print(f"  Correlation with PyTorch: {corr:.4f}")
        if corr > 0.9:
            print(f"  ✅ High correlation — Approach B viable (no embedding file needed!)")
        elif corr > 0.7:
            print(f"  ⚠️  Moderate correlation — may need tuning")
        else:
            print(f"  ❌ Low correlation — Approach B not viable")

    # Show a few examples
    print(f"\n=== Sample comparisons (first 3 cases) ===")
    for i in range(3):
        case = tier2_cases[i]
        cands = [c.word for c in case.candidates]
        scores_a = score_approach_a(
            sess, input_names, output_names, tokenizer, emb_matrix, meta,
            case.sentence_words, case.target_idx, cands
        )
        scores_b = score_approach_b(
            sess, input_names, output_names, tokenizer, meta,
            case.sentence_words, case.target_idx, cands
        )
        print(f"\n  Case {i}: {' '.join(case.sentence_words[:10])}...")
        print(f"  Target idx: {case.target_idx}")
        for j, cand in enumerate(cands[:4]):
            print(f"    {cand:15s}  PyTorch={cached[i]['pll_mean'][j]:.4f}  "
                  f"ONNX-A={scores_a[j][1]:.4f}  ONNX-B={scores_b[j][1]:.4f}")


if __name__ == "__main__":
    main()
