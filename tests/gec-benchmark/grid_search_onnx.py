#!/usr/bin/env python3
"""Run full grid search with ONNX int8 BERTeus scores.

Computes ONNX int8 scores for all 933 cases, then grid-searches weights.
Compares to PyTorch results to verify int8 quantization preserves the signal.
"""
import os, sys, json, numpy as np
import onnxruntime as ort
sys.path.insert(0, os.path.dirname(__file__))
from eval import (
    FREQ_PATH, ELHUYAR_DIR, load_correct_sentences, generate_typo_sentences,
    prepare_tier2_cases, load_freq_map, eval_tier2,
)
from transformers import BertTokenizerFast

ONNX_DIR = "/root/berteus-onnx"
N_CASES = 933


def main():
    print("=== Loading ONNX int8 model + embeddings ===")
    tokenizer = BertTokenizerFast.from_pretrained(ONNX_DIR)
    sess = ort.InferenceSession(
        os.path.join(ONNX_DIR, "model_quantized.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]

    # Load embedding matrix
    from validate_onnx import load_embedding_matrix
    emb_matrix, meta = load_embedding_matrix()

    print("\n=== Regenerating cases ===")
    fmap = load_freq_map(str(FREQ_PATH))
    correct = list(load_correct_sentences(str(ELHUYAR_DIR / "Dem_none.tsv"))) + \
              list(load_correct_sentences(str(ELHUYAR_DIR / "Dea_none.tsv")))
    typo_cases = generate_typo_sentences(correct, seed=42, typos_per_sentence=1)
    tier2_cases = prepare_tier2_cases(typo_cases, fmap)
    print(f"  Total cases: {len(tier2_cases)}")

    print("\n=== Computing ONNX int8 scores ===")
    mask_id = meta["mask_token_id"]
    mask_token = meta["mask_token"]

    for i, case in enumerate(tier2_cases):
        cands = [c.word for c in case.candidates]

        # Build masked sentence
        words = list(case.sentence_words)
        if case.target_idx < len(words):
            words[case.target_idx] = mask_token
        text = " ".join(words)

        enc = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)
        token_type_ids = enc.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = token_type_ids

        last_hidden = sess.run(output_names, feeds)[0]

        mask_positions = np.where(input_ids[0] == mask_id)[0]
        if len(mask_positions) == 0:
            case.bert_scores = [0.0] * len(cands)
            case.bert_scores_sum = [0.0] * len(cands)
            continue

        mask_pos = mask_positions[0]
        mask_hidden = last_hidden[0, mask_pos]
        mask_norm = mask_hidden / (np.linalg.norm(mask_hidden) + 1e-8)

        scores_mean = []
        scores_sum = []
        for cand in cands:
            cand_ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
            if not cand_ids:
                scores_mean.append(0.0)
                scores_sum.append(0.0)
                continue

            # pll_sum: cosine with first subword
            first_emb = emb_matrix[cand_ids[0]]
            first_norm = first_emb / (np.linalg.norm(first_emb) + 1e-8)
            cos_first = float(np.dot(mask_norm, first_norm))

            # pll_mean: cosine with mean of all subwords
            cand_emb = emb_matrix[cand_ids].mean(axis=0)
            cand_norm = cand_emb / (np.linalg.norm(cand_emb) + 1e-8)
            cos_mean = float(np.dot(mask_norm, cand_norm))

            scores_sum.append(cos_first)
            scores_mean.append(cos_mean)

        case.bert_scores = scores_mean
        case.bert_scores_sum = scores_sum

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(tier2_cases)}")

    print("\n=== Grid search (ONNX int8, pll_mean) ===")
    print(f"  {'Weight':>8}  {'T2':>6}  {'Imp':>5}  {'Wor':>5}  {'Net':>5}")
    best_net = -999
    best_w = 0
    for w in [0.0, 0.5, 1.0, 2.0, 5.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 20.0, 50.0]:
        r = eval_tier2(tier2_cases, w, score_attr="bert_scores")
        net = r["net"]
        marker = ""
        if net > best_net:
            best_net = net
            best_w = w
            marker = " ◄"
        print(f"  {w:>8.1f}  {r['tier2_correct']:>6}  {r['tier2_improved']:>5}  {r['tier2_worsened']:>5}  {net:>+5}{marker}")

    print(f"\n=== Summary ===")
    print(f"  PyTorch best:    +99 net (w=12, 786/933, 84.2%)")
    print(f"  ONNX int8 best:  +{best_net} net (w={best_w})")

    # Also check pll_sum
    print("\n=== Grid search (ONNX int8, pll_sum) ===")
    print(f"  {'Weight':>8}  {'T2':>6}  {'Imp':>5}  {'Wor':>5}  {'Net':>5}")
    best_net_sum = -999
    best_w_sum = 0
    for w in [0.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        r = eval_tier2(tier2_cases, w, score_attr="bert_scores_sum")
        net = r["net"]
        marker = ""
        if net > best_net_sum:
            best_net_sum = net
            best_w_sum = w
            marker = " ◄"
        print(f"  {w:>8.1f}  {r['tier2_correct']:>6}  {r['tier2_improved']:>5}  {r['tier2_worsened']:>5}  {net:>+5}{marker}")


if __name__ == "__main__":
    main()
