"""
BERTeus MLM re-ranking — bidirectional context scoring.

NOTE: The ixa-ehu/berteus-base-cased checkpoint was saved as a BertModel
(encoder only) WITHOUT the MLM prediction head. Loading as BertForMaskedLM
silently randomizes the decoder, making PLL scores meaningless.

Instead, we use MASKED EMBEDDING SIMILARITY:
  1. Replace target word with [MASK]
  2. Run BERT encoder → get hidden state at [MASK] position
  3. For each candidate, compute cosine similarity between the [MASK]
     hidden state and the candidate's word embedding(s)
  4. Higher similarity = BERT "expects" a word like this candidate

This is the standard approach for lexical substitution with BERT when
the MLM head is unavailable (cf. Paetzold & Specia 2017, Zhou et al. 2019).

Model: ixa-ehu/berteus-base-cased (BERT-base, ~110M params, cased, 50k vocab)
"""
from __future__ import annotations

import torch
from transformers import BertTokenizerFast, BertModel
from dataclasses import dataclass


@dataclass
class BertScore:
    word: str
    pll_sum: float    # cosine sim with first subword token (primary)
    pll_mean: float   # cosine sim with mean of all subword tokens
    n_tokens: int


class BerteusReranker:
    def __init__(self, model_name: str = "ixa-ehu/berteus-base-cased", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and device == "cuda":
            print("  ⚠️  CUDA not available, using CPU")
        self.tokenizer = None
        self.model = None
        self.mask_token_id = None
        self.word_embeddings = None

    def load(self):
        print(f"  Loading BERTeus: {self.model_name}")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        # Load as BertModel (encoder only) — checkpoint has no MLM head
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.mask_token_id = self.tokenizer.mask_token_id
        self.word_embeddings = self.model.embeddings.word_embeddings.weight  # (vocab, 768)
        print(f"  BERTeus loaded on {self.device} "
              f"({self.model.num_parameters() / 1e6:.0f}M params, "
              f"vocab={self.word_embeddings.size(0)})")

    def score_candidates(
        self,
        sentence_words: list[str],
        target_idx: int,
        candidate_words: list[str],
    ) -> list[BertScore]:
        """Score candidate words using masked embedding similarity.

        1. Replace target word with [MASK], run BERT encoder
        2. Get hidden state at [MASK] position
        3. For each candidate:
           - Tokenize → get subword token IDs
           - Compute cosine sim between [MASK] hidden state and first token emb
           - Compute cosine sim between [MASK] hidden state and mean token emb
        """
        # Build sentence with [MASK] at target position
        words = list(sentence_words)
        if target_idx < len(words):
            words[target_idx] = self.tokenizer.mask_token
        else:
            words.append(self.tokenizer.mask_token)
        text = " ".join(words)

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Find [MASK] position(s) — use the first one
        mask_positions = (input_ids[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            return [BertScore(c, 0.0, 0.0, 0) for c in candidate_words]
        mask_pos = mask_positions[0].item()

        # Run BERT encoder
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            mask_hidden = outputs.last_hidden_state[0, mask_pos]  # (768,)

        # Normalize for cosine similarity
        mask_hidden_norm = torch.nn.functional.normalize(mask_hidden, dim=0)

        results = []
        for cand in candidate_words:
            # Tokenize candidate (no special tokens)
            cand_ids = self.tokenizer(cand, add_special_tokens=False)["input_ids"]
            if not cand_ids:
                results.append(BertScore(cand, 0.0, 0.0, 0))
                continue

            # Get candidate's embedding(s)
            cand_emb = self.word_embeddings[cand_ids]  # (n_tokens, 768)

            # Cosine sim with first subword token
            first_emb = cand_emb[0]
            first_norm = torch.nn.functional.normalize(first_emb, dim=0)
            cos_first = torch.dot(mask_hidden_norm, first_norm).item()

            # Cosine sim with mean of all subword tokens
            mean_emb = cand_emb.mean(dim=0)
            mean_norm = torch.nn.functional.normalize(mean_emb, dim=0)
            cos_mean = torch.dot(mask_hidden_norm, mean_norm).item()

            results.append(BertScore(
                word=cand,
                pll_sum=cos_first,   # first-token similarity (primary)
                pll_mean=cos_mean,   # mean-token similarity
                n_tokens=len(cand_ids),
            ))

        return results
