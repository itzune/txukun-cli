# PROJECT.md — Txukun CLI

## Overview

**txukun-cli** is a command-line tool for Basque text correction: capitalization, punctuation restoration, and spell checking. It uses the **same ONNX int8 quantized model** as the Txukun web app (`itzune/txukun-cap-punct-eu`, ~77 MB) via `optimum[onnxruntime]`.

## Architecture

```
txukun-cli/
├── txukun.py              # Single-file CLI (Click-based)
├── data/
│   ├── eu-words.txt       # 160k Basque word list (1.6 MB)
│   └── eu-words-freq.txt  # Frequency data for suggestion ranking (2.0 MB)
├── pyproject.toml         # uv project config + dependencies
├── README.md              # User-facing docs (Basque-first)
└── PROJECT.md             # This file (agent-facing docs)
```

## Design decisions

- **Single file**: `txukun.py` contains everything — CLI, model wrapper (ONNX), spell checker. No `src/` nesting. Easy for agents to read and modify.
- **ONNX int8 via optimum**: Uses `ORTModelForSeq2SeqLM` from `optimum.onnxruntime` to load quantized ONNX models. Same model files used by the Txukun web app (Transformers.js).
- **Tokenizer from HiTZ**: Tokenizer loads from `HiTZ/cap-punct-eu` (needs `source.spm` + `vocab.json`). The `itzune/txukun-cap-punct-eu` repo also has these files but Marian tokenizer path resolution doesn't work with HF Hub dirs in transformers >= 4.57.
- **Lazy model loading**: `TxukunModel` only imports `optimum`/`transformers` and loads the model on first use. Fast startup for spell-only mode (`--no-punct`).
- **Spell checker is disabled by default**: `--spell` flag enables it.
- **CPU-only**: `CPUExecutionProvider` by default. No GPU needed.

## Dependencies

| Package | Purpose |
|---|---|
| `optimum[onnxruntime]` | ONNX Runtime inference for seq2seq models (pulls transformers, torch, onnxruntime) |
| `click` | CLI framework |

Total install size: ~2.5 GB (mostly torch + onnxruntime libs). ONNX model download: ~77 MB.

## Model details

- **ONNX model repo**: `itzune/txukun-cap-punct-eu` on HF Hub
- **Encoder**: `encoder_model_quantized.onnx` (34 MB, int8)
- **Decoder (merged, with-past)**: `decoder_model_merged_quantized.onnx` (41 MB, int8)
- **Tokenizer**: SentencePiece (Marian), loaded from `HiTZ/cap-punct-eu`
- **Original model**: `HiTZ/cap-punct-eu` (PyTorch, 154 MB safetensors)

The ONNX q8 model produces *different* output than the original PyTorch model. On well-formed Basque sentences, q8 often produces better results. Both versions hallucinate on short/unusual inputs (known limitation).

## Script entry point

The project uses `uv` for dependency management. Run with:
```
uv run python txukun.py "text to correct"
```

There is no `[project.scripts]` entry point — this avoids the complexity of package builds for a simple CLI tool.

## Adding features

To add a new CLI option:
1. Add a `@click.option()` decorator to the `main()` function
2. Handle the flag in the function body
3. Update README.md usage table

To improve spell checking:
1. Update `data/eu-words.txt` with a new word list
2. Or implement Hunspell-style affix rules in `SpellChecker`

## Testing

```bash
# Quick smoke test
uv run python txukun.py "kaixo mundua"

# With pipe
echo "ser gertatu da hemen" | uv run python txukun.py --stdin

# Spell check
uv run python txukun.py --spell "ser gertatu da hemen"

# No model (spell only)
uv run python txukun.py --no-punct --spell "akats bat dauka"

# Verify word list loading
uv run python -c "from pathlib import Path; w=Path('data/eu-words.txt'); print(f'{len(w.read_text().splitlines())} words')"
```

## Related projects

- **Txukun web app**: `../txukun/` — same ONNX model, browser-based with Transformers.js
- **txukun-cap-punct-eu model**: `itzune/txukun-cap-punct-eu` on HF Hub (int8 ONNX + tokenizer configs)
- **Original model**: `HiTZ/cap-punct-eu` on HF Hub (PyTorch)
- **Dictionary source**: `../txukun/public/dicts/` — Xuxen .aff/.dic + word lists

## License

Code: Apache 2.0 — same as the Txukun web app.
