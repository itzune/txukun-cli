"""
GECToR grammar correction (Tier 3) — ONNX version.

Port of txukun's gector.js. Uses GECToR (edit-based GEC) fine-tuned on
RoBERTa-eus-base, trained on 1M Elhuyar GEC pairs.

Architecture:
  - Encoder: RoBERTa-eus-base (110M, 12L/768H)
  - Two heads: label classifier ($KEEP/$DELETE/$REPLACE_x/$APPEND_x)
                + error detector ($CORRECT/$INCORRECT)
  - Inference: iterative (up to 5 passes), non-autoregressive

Model: itzune/gector-eus-onnx (int4 ONNX, ~85MB)
"""
from __future__ import annotations

import re
import json
import numpy as np

HF_REPO = "itzune/gector-eus-onnx"

# Inference parameters (match gector.js)
KEEP_CONFIDENCE = 0.0
MIN_ERROR_PROB = 0.5
MAX_ITERATIONS = 5

PUNCT_RE = re.compile(r"([.,;:!?()«»\"'\-\u2013\u2014])")


class GectorModel:
    """GECToR ONNX grammar corrector + detector."""

    def __init__(self, quiet: bool = False):
        self._session = None
        self._tokenizer = None
        self._vocab: dict | None = None
        self._quiet = quiet
        self._failed = False
        self._loading = False

    def _load(self):
        if self._session is not None or self._failed or self._loading:
            return
        self._loading = True

        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            from huggingface_hub import hf_hub_download

            if not self._quiet:
                import click
                click.echo("⏳ GECToR-eus ONNX kargatzen (~85MB)...", err=True)

            model_path = hf_hub_download(HF_REPO, "onnx/model_q4.onnx")
            vocab_path = hf_hub_download(HF_REPO, "gector_vocab.json")

            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = AutoTokenizer.from_pretrained(HF_REPO)

            with open(vocab_path) as f:
                self._vocab = json.load(f)

            if not self._quiet:
                import click
                click.echo(f"✅ GECToR kargatuta (labels={self._vocab['num_labels']}).", err=True)
        except Exception as e:
            if not self._quiet:
                import click
                click.echo(f"⚠️  GECToR kargatzeak huts egin du: {e}", err=True)
            self._failed = True
        finally:
            self._loading = False

    @property
    def ready(self) -> bool:
        return self._session is not None and self._vocab is not None

    @property
    def failed(self) -> bool:
        return self._failed

    # ── Tokenization ────────────────────────────────

    def _tokenize_with_word_ids(self, words: list[str], max_len: int) -> tuple[list[int], list[int]]:
        """Replicate HuggingFace's is_split_into_words=True + word_ids()."""
        bos_id = self._tokenizer.bos_token_id or 0
        eos_id = self._tokenizer.eos_token_id or 2

        input_ids = [bos_id]
        word_ids: list[int | None] = [None]

        for w_idx, word in enumerate(words):
            enc = self._tokenizer(" " + word, add_special_tokens=False)
            ids = enc["input_ids"]

            for tid in ids:
                if len(input_ids) >= max_len - 1:
                    break
                input_ids.append(tid)
                word_ids.append(w_idx)
            if len(input_ids) >= max_len - 1:
                break

        input_ids.append(eos_id)
        word_ids.append(None)
        return input_ids, word_ids

    def _build_word_masks(self, word_ids: list[int | None]) -> list[int]:
        masks = []
        prev = None
        for wid in word_ids:
            if wid is None:
                masks.append(0)
            elif wid != prev:
                masks.append(1)
            else:
                masks.append(0)
            prev = wid
        return masks

    # ── Core prediction ─────────────────────────────

    def _softmax(self, logits_row: np.ndarray) -> np.ndarray:
        """Compute softmax over a 1D array of logits for a single token."""
        slc = logits_row.astype(np.float64)
        slc = slc - np.max(slc)
        exps = np.exp(slc)
        return exps / np.sum(exps)

    def _predict_labels(self, input_ids: list[int], attention_mask: list[int], word_ids: list[int | None]) -> list[int]:
        """Run a forward pass and return predicted label IDs per token."""
        seq_len = len(input_ids)
        keep_idx = self._vocab["label2id"][self._vocab["keep_label"]]
        incor_idx = self._vocab["d_label2id"][self._vocab["incorrect_label"]]

        input_ids_arr = np.array([input_ids], dtype=np.int64)
        attention_mask_arr = np.array([attention_mask], dtype=np.int64)

        outputs = self._session.run(
            None,
            {"input_ids": input_ids_arr, "attention_mask": attention_mask_arr},
        )
        logits_labels = outputs[0][0]  # (seq_len, num_labels)
        logits_d = outputs[1][0]       # (seq_len, d_num_labels)

        # Detection: compute max error probability
        word_masks = self._build_word_masks(word_ids)
        max_error_prob = 0.0
        for t in range(seq_len):
            if word_masks[t] != 1:
                continue
            probs_d = self._softmax(logits_d[t])
            p_incor = probs_d[incor_idx]
            if p_incor > max_error_prob:
                max_error_prob = float(p_incor)

        sentence_keep_all = max_error_prob < MIN_ERROR_PROB

        pred_labels = [0] * seq_len
        for t in range(seq_len):
            if sentence_keep_all:
                pred_labels[t] = keep_idx
                continue

            probs = self._softmax(logits_labels[t])
            probs = probs.copy()
            probs[keep_idx] += KEEP_CONFIDENCE

            max_prob = float(np.max(probs))
            if max_prob < MIN_ERROR_PROB:
                pred_labels[t] = keep_idx
                continue

            pred_labels[t] = int(np.argmax(probs))

        return pred_labels

    def _align_to_words(self, pred_labels: list[int], word_ids: list[int | None]) -> tuple[list[str], bool]:
        """Align token labels to words."""
        word_labels: list[str] = []
        no_correction = {
            self._vocab["label2id"]["$KEEP"],
            self._vocab["label2id"]["<OOV>"],
            self._vocab["label2id"]["<PAD>"],
        }
        has_corrections = False
        prev_wid = None

        for t, wid in enumerate(word_ids):
            if wid is None or wid == prev_wid:
                continue
            label_id = pred_labels[t]
            label = self._vocab["id2label"][str(label_id)]
            word_labels.append(label)
            if label_id not in no_correction:
                has_corrections = True
            prev_wid = wid

        return word_labels, has_corrections

    def _apply_edits(self, words: list[str], labels: list[str]) -> list[str]:
        edited = []
        for i, word in enumerate(words):
            label = labels[i] if i < len(labels) else "$KEEP"

            if word == "$START":
                edited.append("$START")
            elif label in ("<PAD>", "<OOV>", "$KEEP"):
                edited.append(word)
            elif label.startswith("$REPLACE_"):
                edited.append(label[9:])
            elif label.startswith("$APPEND_"):
                edited.append(word)
                edited.append(label[8:])
            elif label == "$DELETE":
                edited.append("$DELETE")
            else:
                edited.append(word)

        result = " ".join(edited)
        result = re.sub(r" \$DELETE\b", "", result)
        result = re.sub(r"\$DELETE ", "", result)
        result = re.sub(r"\$DELETE", "", result)
        result = re.sub(r"\$START ", "", result)
        result = re.sub(r"\$START", "", result)
        return [w for w in result.split() if w]

    # ── Public API ──────────────────────────────────

    def correct(self, text: str) -> tuple[str, bool]:
        """Correct grammar errors. Returns (corrected, changed)."""
        if not self.ready:
            self._load()
        if not self.ready:
            return text, False

        max_len = self._vocab.get("max_length", 128)
        current = _tokenize_punctuation(text)

        for _ in range(MAX_ITERATIONS):
            words = ["$START"] + current.split()
            input_ids, word_ids = self._tokenize_with_word_ids(words, max_len)
            attention_mask = [1] * len(input_ids)

            pred_labels = self._predict_labels(input_ids, attention_mask, word_ids)
            word_labels, has_corrections = self._align_to_words(pred_labels, word_ids)

            if not has_corrections:
                break

            new_words = self._apply_edits(words, word_labels)
            current = " ".join(new_words)

        corrected = _detokenize_punctuation(current)
        return corrected, corrected != text

    def detect(self, text: str) -> list[dict]:
        """Detection-only pass: per-word P(INCORRECT) with character positions."""
        if not self.ready:
            self._load()
        if not self.ready:
            return []

        max_len = self._vocab.get("max_length", 128)
        incor_idx = self._vocab["d_label2id"][self._vocab["incorrect_label"]]

        # Tokenize input into words with character positions
        word_tokens = []
        for m in re.finditer(r"\S+", text):
            word_tokens.append((m.group(0), m.start(), m.end()))

        if not word_tokens:
            return []

        words = ["$START"] + [w[0] for w in word_tokens]
        input_ids, word_ids = self._tokenize_with_word_ids(words, max_len)
        attention_mask = [1] * len(input_ids)

        input_ids_arr = np.array([input_ids], dtype=np.int64)
        attention_mask_arr = np.array([attention_mask], dtype=np.int64)

        outputs = self._session.run(
            None,
            {"input_ids": input_ids_arr, "attention_mask": attention_mask_arr},
        )
        logits_d = outputs[1][0]  # (seq_len, d_num_labels)

        word_masks = self._build_word_masks(word_ids)
        detections = []

        for t in range(len(input_ids)):
            if word_masks[t] != 1:
                continue
            probs_d = self._softmax(logits_d[t])
            p_incor = float(probs_d[incor_idx])

            wid = word_ids[t]
            if wid is not None and wid > 0 and wid - 1 < len(word_tokens):
                word, start, end = word_tokens[wid - 1]
                detections.append({
                    "word": word,
                    "pIncorrect": p_incor,
                    "start": start,
                    "end": end,
                })

        return detections


def _tokenize_punctuation(text: str) -> str:
    return PUNCT_RE.sub(r" \1 ", text).replace("  ", " ").strip()


def _detokenize_punctuation(text: str) -> str:
    return re.sub(r"\s+([.,;:!?()«»\"'\-\u2013\u2014])", r"\1", text)
