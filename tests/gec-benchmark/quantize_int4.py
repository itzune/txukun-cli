#!/usr/bin/env python3
"""Quantize BERTeus ONNX fp32 → int4 (MatMul) + int8 (embeddings).

Produces model_q4.onnx compatible with Transformers.js dtype='q4'.

Two-step quantization:
  1. MatMulNBitsQuantizer: encoder MatMul ops → int4 (weight-only, block_size=128)
  2. quantize_dynamic: embedding Gather ops → int8 (QDQ: DequantizeLinear)

Step 2 is critical: the embedding matrix (50099×768×fp32 = 147 MB) dominates
the model size. Without quantizing it, int4 is actually LARGER than int8
(201 MB vs 119 MB). int8 QDQ for embeddings works on WASM (proven by the
existing model_quantized.onnx). We do NOT use GatherBlockQuantized (int4
embeddings), which is NOT implemented on the WASM backend.

Result: ~86 MB (vs int8's 119 MB, fp32's 496 MB).
"""
import argparse
import os
import sys
import tempfile
from collections import Counter

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.quantization.quantize import quantize_dynamic, QuantType, QuantFormat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/root/berteus-onnx/model.onnx",
                    help="fp32 ONNX model path")
    ap.add_argument("--output", default="model_q4.onnx",
                    help="Output int4+int8 ONNX path")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--symmetric", action="store_true",
                    help="Use symmetric quantization (default: asymmetric)")
    args = ap.parse_args()

    in_size = os.path.getsize(args.input) / 1e6
    print(f"Input: {args.input} ({in_size:.1f} MB)")

    # Load to inspect original ops
    orig = onnx.load(args.input)
    orig_ops = Counter(n.op_type for n in orig.graph.node)
    print(f"Original ops: {dict(orig_ops)}")
    del orig  # free memory

    # ---- Step 1: int4 quantization on MatMul ops ----
    print(f"\n--- Step 1: int{args.bits} quantization (MatMul only) ---")
    tmp_int4 = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False, dir=".")
    tmp_int4.close()

    quantizer = MatMulNBitsQuantizer(
        args.input,
        bits=args.bits,
        block_size=args.block_size,
        is_symmetric=args.symmetric,
        op_types_to_quantize=("MatMul",),
    )
    quantizer.process()
    quantizer.model.save_model_to_file(tmp_int4.name, use_external_data_format=False)

    int4_size = os.path.getsize(tmp_int4.name) / 1e6
    print(f"After step 1: {int4_size:.1f} MB")

    # ---- Step 2: int8 dynamic quantization on Gather ops (embeddings) ----
    print(f"\n--- Step 2: int8 quantization (Gather/embeddings only) ---")
    quantize_dynamic(
        tmp_int4.name,
        args.output,
        op_types_to_quantize=["Gather"],
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
    os.unlink(tmp_int4.name)

    out_size = os.path.getsize(args.output) / 1e6
    print(f"\nFinal output: {args.output} ({out_size:.1f} MB)")

    # Verify output ops
    m = onnx.load(args.output)
    new_ops = Counter(n.op_type for n in m.graph.node)
    print(f"Output ops: {dict(new_ops)}")

    # Check for problematic ops
    if "GatherBlockQuantized" in new_ops:
        print("\n⚠️  WARNING: GatherBlockQuantized found! Will NOT work on WASM.")
        sys.exit(1)
    else:
        print("\n✅ No GatherBlockQuantized — WASM-safe")

    n_matmulnbits = new_ops.get("MatMulNBits", 0)
    n_matmul = new_ops.get("MatMul", 0)
    n_dequant = new_ops.get("DequantizeLinear", 0)
    print(f"MatMulNBits: {n_matmulnbits}, remaining MatMul: {n_matmul}, "
          f"DequantizeLinear: {n_dequant}")

    # Check largest initializers
    inits = sorted(m.graph.initializer,
                   key=lambda x: len(x.raw_data) if x.raw_data else x.ByteSize(),
                   reverse=True)[:5]
    print("\nTop 5 initializers:")
    for init in inits:
        dims = list(init.dims)
        dtype = init.data_type
        size = (len(init.raw_data) if init.raw_data else 0) / 1e6
        print(f"  {init.name}: dims={dims}, dtype={dtype}, size={size:.1f} MB")

    print(f"\nSize reduction: {in_size:.1f} MB → {out_size:.1f} MB "
          f"({(1 - out_size/in_size)*100:.0f}% smaller)")
    print(f"vs int8 model (119 MB): {(out_size/119 - 1)*100:+.0f}%")


if __name__ == "__main__":
    main()
