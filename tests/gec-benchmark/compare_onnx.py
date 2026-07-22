#!/usr/bin/env python3
"""Compare fp32 vs int8 ONNX models against PyTorch cached scores."""
import os, sys, json, numpy as np
import onnxruntime as ort
sys.path.insert(0, os.path.dirname(__file__))
from eval import (
    FREQ_PATH, ELHUYAR_DIR, load_correct_sentences, generate_typo_sentences,
    prepare_tier2_cases, load_freq_map,
)
from validate_onnx import load_embedding_matrix, run_bert, score_approach_a
from transformers import BertTokenizerFast

ONNX_DIR = "/root/berteus-onnx"
tokenizer = BertTokenizerFast.from_pretrained(ONNX_DIR)
emb_matrix, meta = load_embedding_matrix()

fmap = load_freq_map(str(FREQ_PATH))
correct = list(load_correct_sentences(str(ELHUYAR_DIR / "Dem_none.tsv"))) + \
          list(load_correct_sentences(str(ELHUYAR_DIR / "Dea_none.tsv")))
typo_cases = generate_typo_sentences(correct, seed=42, typos_per_sentence=1)
tier2_cases = prepare_tier2_cases(typo_cases, fmap)

with open("tests/gec-benchmark/bert_scores_cache.json") as f:
    cached = json.load(f)

for name, model_file in [("fp32", "model.onnx"), ("int8", "model_quantized.onnx")]:
    path = os.path.join(ONNX_DIR, model_file)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]

    diffs = []
    exact_match = 0
    total = 0
    rank_changes = 0
    for i in range(50):
        case = tier2_cases[i]
        cands = [c.word for c in case.candidates]
        scores_a = score_approach_a(
            sess, input_names, output_names, tokenizer, emb_matrix, meta,
            case.sentence_words, case.target_idx, cands
        )
        onnx_scores = [s[1] for s in scores_a]  # pll_mean
        pytorch_scores = cached[i]["pll_mean"]

        for j in range(len(cands)):
            diff = abs(onnx_scores[j] - pytorch_scores[j])
            diffs.append(diff)
            if diff < 0.001:
                exact_match += 1
            total += 1

        # Check if ranking changed
        onnx_rank = np.argsort(onnx_scores)[::-1]
        pt_rank = np.argsort(pytorch_scores)[::-1]
        if onnx_rank[0] != pt_rank[0]:
            rank_changes += 1

    diffs = np.array(diffs)
    print(f"{name}: exact(<0.001)={exact_match}/{total}, "
          f"mean_diff={diffs.mean():.6f}, max={diffs.max():.6f}, "
          f"p95={np.percentile(diffs, 95):.6f}, "
          f"rank_changes={rank_changes}/50")
