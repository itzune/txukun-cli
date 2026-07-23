"""
BERTeus neural re-ranking (Tier 2) — ONNX version.

Port of txukun's bert-rerank.js. Uses BERTeus (ixa-ehu/berteus-base-cased,
int4 ONNX, 85MB) via onnxruntime to score spell-correction candidates by
masked embedding similarity.

  score = cosine_sim(mask_hidden_state, mean(candidate_word_embeddings))

The misspelled word is replaced with [MASK] in its sentence context
(bidirectional — BERT sees both left and right context). The [MASK]
hidden state is compared against each candidate's static word embedding
(mean of subword piece embeddings from the embedding matrix).

Model: itzune/berteus-onnx (int4 ONNX + float16 embeddings, 85MB + 74MB)
"""
from __future__ import annotations

import numpy as np

HF_REPO = "itzune/berteus-onnx"
EMB_DIM = 768
MASK_TOKEN_ID = 4
BERT_WEIGHT = 18.0
MAX_CANDIDATES = 5
CONTEXT_CHARS = 200


class BerteusReranker:
    """BERTeus ONNX re-ranker for spell candidate scoring."""

    def __init__(self, quiet: bool = False):
        self._session = None
        self._tokenizer = None
        self._embeddings: np.ndarray | None = None
        self._vocab_size = 0
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
                click.echo("⏳ BERTeus ONNX kargatzen (85MB + 74MB embeddings)...", err=True)

            model_path = hf_hub_download(HF_REPO, "onnx/model_q4.onnx")
            emb_path = hf_hub_download(HF_REPO, "word_embeddings_f16.bin")

            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = AutoTokenizer.from_pretrained(HF_REPO)

            # Load float16 embeddings → float32
            emb_raw = np.fromfile(emb_path, dtype=np.float16)
            self._embeddings = emb_raw.astype(np.float32).reshape(-1, EMB_DIM)
            self._vocab_size = self._embeddings.shape[0]

            if not self._quiet:
                import click
                click.echo(f"✅ BERTeus kargatuta (vocab={self._vocab_size}).", err=True)
        except Exception as e:
            if not self._quiet:
                import click
                click.echo(f"⚠️  BERTeus kargatzeak huts egin du: {e}", err=True)
            self._failed = True
        finally:
            self._loading = False

    @property
    def ready(self) -> bool:
        return self._session is not None

    @property
    def failed(self) -> bool:
        return self._failed

    def score_candidates(
        self,
        text: str,
        error_start: int,
        error_end: int,
        candidates: list[str],
    ) -> list[float]:
        """Score candidates by BERTeus masked embedding similarity.

        Returns cosine sim scores [-1, 1], aligned with candidates.
        """
        if not self.ready:
            self._load()
        if not self.ready:
            return [0.0] * len(candidates)

        limited = candidates[:MAX_CANDIDATES]

        # Build masked text with context window
        window_start = max(0, error_start - CONTEXT_CHARS)
        window_end = min(len(text), error_end + CONTEXT_CHARS)
        left = text[window_start:error_start]
        right = text[error_end:window_end]
        masked_text = left + self._tokenizer.mask_token + right

        # Tokenize
        inputs = self._tokenizer(
            masked_text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors="np",
        )

        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        token_type_ids = np.zeros_like(input_ids)

        # Find [MASK] position
        ids_flat = input_ids.flatten()
        mask_positions = np.where(ids_flat == MASK_TOKEN_ID)[0]
        if len(mask_positions) == 0:
            return [0.0] * len(candidates)
        mask_pos = int(mask_positions[0])

        # Run BERT encoder
        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        last_hidden = outputs[0]  # (1, seq_len, 768)

        # Extract [MASK] hidden state
        mask_hidden = last_hidden[0, mask_pos]  # (768,)
        mask_norm = mask_hidden / (np.linalg.norm(mask_hidden) + 1e-8)

        # Score each candidate
        scores: list[float] = []
        for cand in limited:
            cand_ids = self._tokenizer(cand, add_special_tokens=False)["input_ids"]
            if not cand_ids:
                scores.append(0.0)
                continue

            # Mean of subword embeddings from the static embedding matrix
            valid_ids = [i for i in cand_ids if i < self._vocab_size]
            if not valid_ids:
                scores.append(0.0)
                continue

            cand_emb = self._embeddings[valid_ids].mean(axis=0)  # (768,)
            cand_norm = cand_emb / (np.linalg.norm(cand_emb) + 1e-8)
            scores.append(float(np.dot(mask_norm, cand_norm)))

        # Pad with 0 for truncated candidates
        while len(scores) < len(candidates):
            scores.append(0.0)
        return scores
