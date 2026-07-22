#!/usr/bin/env python3
"""Export BERTeus to ONNX for browser deployment via Transformers.js.

Produces:
  - model.onnx / model_quantized.onnx (BERT encoder, returns last_hidden_state)
  - tokenizer.json + config files (for Transformers.js AutoModel)
  - word_embeddings_f32.bin / f16.bin (embedding matrix for cosine similarity)

Usage: python export_berteus_onnx.py
"""
import os
import sys
import numpy as np
import torch

MODEL_NAME = "ixa-ehu/berteus-base-cased"
OUTPUT_DIR = "/root/berteus-onnx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Load tokenizer and model ──────────────────────────────────
print("=== Loading model and tokenizer ===")
from transformers import BertModel, BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()

tokenizer.save_pretrained(OUTPUT_DIR)
model.config.save_pretrained(OUTPUT_DIR)
print(f"Tokenizer + config saved to {OUTPUT_DIR}")

# ── Step 2: Export to ONNX ────────────────────────────────────────────
print("\n=== Exporting to ONNX ===")
onnx_exported = False

# Try Optimum first
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    onnx_model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
    onnx_model.save_pretrained(OUTPUT_DIR)
    print(f"Exported with Optimum ORTModelForFeatureExtraction")
    onnx_exported = True
except Exception as e:
    print(f"Optimum export failed: {e}")

# Fallback: torch.onnx.export
if not onnx_exported:
    print("Falling back to torch.onnx.export...")
    dummy = tokenizer("Test sentence for export", return_tensors="pt",
                      padding="max_length", max_length=16)
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        os.path.join(OUTPUT_DIR, "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids":          {0: "batch", 1: "sequence"},
            "attention_mask":     {0: "batch", 1: "sequence"},
            "token_type_ids":     {0: "batch", 1: "sequence"},
            "last_hidden_state":  {0: "batch", 1: "sequence"},
        },
        opset_version=14,
    )
    print(f"Exported with torch.onnx.export")
    onnx_exported = True

# ── Step 3: Quantize to int8 ──────────────────────────────────────────
print("\n=== Quantizing to int8 ===")
# Find the model.onnx (might be in onnx/ subdirectory from Optimum)
possible_paths = [
    os.path.join(OUTPUT_DIR, "model.onnx"),
    os.path.join(OUTPUT_DIR, "onnx", "model.onnx"),
]
model_path = None
for p in possible_paths:
    if os.path.exists(p):
        model_path = p
        break

if model_path is None:
    # Search recursively
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f == "model.onnx":
                model_path = os.path.join(root, f)
                break

if model_path:
    print(f"Found model at: {model_path}")
    print(f"  Size: {os.path.getsize(model_path)/1e6:.1f} MB")

    quant_path = model_path.replace("model.onnx", "model_quantized.onnx")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(model_path, quant_path, weight_type=QuantType.QUInt8)
        print(f"Quantized: {quant_path}")
        print(f"  Size: {os.path.getsize(quant_path)/1e6:.1f} MB")
    except Exception as e:
        print(f"Quantization failed: {e}")
else:
    print("ERROR: model.onnx not found!")

# ── Step 4: Extract word embedding matrix ─────────────────────────────
print("\n=== Extracting word embedding matrix ===")
emb = model.embeddings.word_embeddings.weight.detach().numpy()
print(f"Embedding matrix: shape={emb.shape}, dtype={emb.dtype}")

# Save float32 (for Python validation)
emb_f32_path = os.path.join(OUTPUT_DIR, "word_embeddings_f32.bin")
emb.tofile(emb_f32_path)
print(f"  f32: {os.path.getsize(emb_f32_path)/1e6:.1f} MB")

# Save float16 (for browser — smaller download)
emb_f16 = emb.astype(np.float16)
emb_f16_path = os.path.join(OUTPUT_DIR, "word_embeddings_f16.bin")
emb_f16.tofile(emb_f16_path)
print(f"  f16: {os.path.getsize(emb_f16_path)/1e6:.1f} MB")

# Save vocab metadata
import json
vocab = tokenizer.get_vocab()
meta = {
    "vocab_size": len(vocab),
    "embedding_dim": emb.shape[1],
    "mask_token": tokenizer.mask_token,
    "mask_token_id": tokenizer.mask_token_id,
    "cls_token": tokenizer.cls_token,
    "cls_token_id": tokenizer.cls_token_id,
    "sep_token": tokenizer.sep_token,
    "sep_token_id": tokenizer.sep_token_id,
}
with open(os.path.join(OUTPUT_DIR, "embedding_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"  meta: {meta}")

# ── Step 5: List all output files ─────────────────────────────────────
print("\n=== Output files ===")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in sorted(files):
        path = os.path.join(root, f)
        rel = os.path.relpath(path, OUTPUT_DIR)
        print(f"  {rel}: {os.path.getsize(path)/1e6:.1f} MB")

print("\n=== Done ===")
