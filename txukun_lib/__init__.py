"""
Txukun library — Basque text correction pipeline (3 models).

Port of the txukun web app's detection and correction algorithm:
  - Cap-punct (MarianMT ONNX) — capitalization & punctuation
  - Spelling (Hunspell + Tier1 freq + BERTeus ONNX re-ranking)
  - Grammar (GECToR ONNX) — grammatical error correction

All models run on plain text (markdown stripped). Error offsets are
mapped back to the original text positions.
"""
