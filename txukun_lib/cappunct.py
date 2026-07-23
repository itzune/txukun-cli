"""
Cap-punct model (MarianMT ONNX) — capitalization & punctuation restoration.

Port of txukun's models.js: splitIntoSegments() + constrainCapPunct() (LCS)
+ correctCapPunct().

The MarianMT model was trained on 9.7M individual sentences. Multi-sentence
paragraphs must be split into sentence-length segments first. The model's
output is constrained to case/punctuation-only changes (word substitutions
/hallucinations are rejected via LCS alignment).
"""
from __future__ import annotations

import io
import re
import warnings
from contextlib import redirect_stdout, nullcontext


MODEL_ID = "itzune/txukun-cap-punct-eu"
CLEANUP_RE = re.compile(r"</?s>|</?pad>|<unk>")


class CapPunctModel:
    """Lazy-loading MarianMT cap-punct model via ONNX Runtime (int8)."""

    def __init__(self, quiet: bool = False):
        self._pipeline = None
        self._quiet = quiet
        self._failed = False

    def _load(self):
        if self._pipeline is not None or self._failed:
            return

        if not self._quiet:
            import click
            click.echo("⏳ cap-punct-eu ONNX int8 kargatzen...", err=True)

        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer, pipeline
        from transformers.utils import logging as hf_logging

        if self._quiet:
            hf_logging.set_verbosity_error()

        with redirect_stdout(io.StringIO()) if self._quiet else nullcontext():
            model = ORTModelForSeq2SeqLM.from_pretrained(
                MODEL_ID,
                encoder_file_name="encoder_model_quantized.onnx",
                decoder_file_name="decoder_model_merged_quantized.onnx",
                decoder_with_past_file_name="decoder_model_merged_quantized.onnx",
                provider="CPUExecutionProvider",
                use_cache=True,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tokenizer = AutoTokenizer.from_pretrained("HiTZ/cap-punct-eu")

            self._pipeline = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
            )

        if not self._quiet:
            import click
            click.echo("✅ cap-punct kargatuta (~77 MB).", err=True)

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    @property
    def failed(self) -> bool:
        return self._failed

    def correct(self, text: str) -> str:
        """Run cap-punct correction with sentence splitting + LCS constraint."""
        if self._failed:
            return text
        self._load()
        if not self._pipeline:
            return text

        segments = _split_into_segments(text)
        results = []
        for seg in segments:
            if not seg["text"]:
                results.append("")
                continue
            out = self._pipeline(seg["text"])
            corrected = out[0]["translation_text"] if out else seg["text"]
            corrected = clean_output(corrected)
            results.append(_constrain_lcs(seg["text"], corrected) or seg["text"])

        # Rejoin with original separators
        return "".join(r + s for r, s in zip(results, [seg["sep"] for seg in segments])).rstrip()

    def correct_simple(self, text: str) -> str:
        """Run cap-punct without sentence splitting (legacy single-call mode)."""
        if self._failed:
            return text
        self._load()
        if not self._pipeline:
            return text
        out = self._pipeline(text)
        corrected = out[0]["translation_text"] if out else text
        return clean_output(corrected) or text


def clean_output(text: str) -> str:
    """Clean model output: remove special tokens, collapse whitespace."""
    text = CLEANUP_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _split_into_segments(text: str) -> list[dict]:
    """Split text into sentence-length segments for the cap-punct model.

    Strategy:
      1. Split on newlines (hard breaks).
      2. Within each line, split on existing sentence-ending punctuation.
      3. For long unpunctuated segments (>25 words), split by word count.
    """
    segments: list[dict] = []
    lines = text.split('\n')
    for li, line in enumerate(lines):
        if not line.strip():
            segments.append({"text": "", "sep": "\n"})
            continue

        # Split on sentence-ending punctuation followed by whitespace
        sentence_ends = re.split(r'(?<=[.?!])\s+', line)
        for sent in sentence_ends:
            trimmed = sent.strip()
            if not trimmed:
                continue

            word_count = len(trimmed.split())
            if word_count > 25:
                # Long unpunctuated segment — split by word count
                words = trimmed.split()
                for i in range(0, len(words), 20):
                    chunk = " ".join(words[i:i + 20])
                    sep = "\n" if (i + 20 >= len(words) and li < len(lines) - 1) else " "
                    segments.append({"text": chunk, "sep": sep})
            else:
                sep = "\n" if li < len(lines) - 1 else ""
                segments.append({"text": trimmed, "sep": sep})

    return segments


def _constrain_lcs(input_line: str, output_line: str) -> str:
    """Constrain MarianMT output to case/punctuation-only changes.

    Uses LCS alignment on lowercased alphanumeric content to accept valid
    cap/punct changes on matched tokens while rejecting hallucinated
    substitutions. Unlike positional matching (which bails out on token
    count mismatch), LCS alignment preserves valid corrections even when
    the model hallucinates in the same segment.
    """
    input_tokens = input_line.split()
    output_tokens = output_line.split()

    def norm(t: str) -> str:
        return re.sub(r'[^a-zà-ÿñü]', '', t.lower())

    a_norm = [norm(t) for t in input_tokens]
    b_norm = [norm(t) for t in output_tokens]
    n = len(a_norm)
    m = len(b_norm)

    if n == 0:
        return input_line

    # LCS DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a_norm[i] == b_norm[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    # Walk alignment: use output token where matched, keep input where unmatched,
    # skip hallucinated output tokens.
    result = []
    i = j = 0
    while i < n and j < m:
        if a_norm[i] == b_norm[j]:
            result.append(output_tokens[j])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            result.append(input_tokens[i])
            i += 1
        else:
            j += 1  # skip hallucinated token

    while i < n:
        result.append(input_tokens[i])
        i += 1

    return " ".join(result)
