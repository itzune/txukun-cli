"""
LM surprisal scorer — uses llama-cpp-python to score spell-correction
candidates by contextual surprisal.

Surprisal = log P(candidate | context) − log P(candidate)

This isolates the contextual signal by cancelling the candidate's
inherent token-probability bias. Validated against the token-by-token
reference (9/10 on surprisal_sum, no BOS — see futo-transformer-basque
scripts/eval/echo_test.py).

Uses the echo fast path: a single create_completion call per candidate
with echo=True returns logprobs for all prompt tokens. This is ~10x
faster than token-by-token scoring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SurprisalResult:
    """Surprisal score for a single candidate."""
    word: str
    ctx_sum: float       # sum of in-context logprobs
    base_sum: float      # sum of baseline logprobs
    surprisal: float     # ctx_sum - base_sum (higher = more contextually likely)


class LMReranker:
    """Lazy-loads the GGUF model and scores candidates by surprisal."""

    def __init__(self, model_path: str, n_ctx: int = 2048):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._llm = None

    def load(self):
        if self._llm is not None:
            return
        import llama_cpp
        self._llm = llama_cpp.Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=0,
            verbose=False,
            logits_all=True,
        )

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def _echo_logprobs(self, text: str) -> tuple[list, list]:
        """Single echo call. Returns (tokens, logprobs) for the prompt portion."""
        r = self._llm.create_completion(
            prompt=text,
            max_tokens=1,
            logprobs=0,
            echo=True,
            temperature=0,
        )
        lp = r["choices"][0]["logprobs"]
        return lp["tokens"], lp["token_logprobs"]

    @staticmethod
    def _common_prefix_len(a: list, b: list) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def score_word(self, context: str, word: str) -> SurprisalResult:
        """Compute surprisal for a single candidate given context.

        Uses the echo fast path:
        - Baseline: " word" (leading space matches in-context tokenization)
        - In-context: "context word" (find candidate tokens via common prefix)

        Returns SurprisalResult with surprisal = ctx_sum - base_sum.
        """
        llm = self._llm

        # Baseline: " word"
        _, base_lps = self._echo_logprobs(f" {word}")
        base_vals = [x for x in base_lps[1:] if x is not None and math.isfinite(x)]

        # In-context: "context word"
        ctx_tokens = llm.tokenize(context.encode("utf-8"), add_bos=False)
        full_tokens = llm.tokenize(f"{context} {word}".encode("utf-8"), add_bos=False)
        cpl = self._common_prefix_len(ctx_tokens, full_tokens)
        cand_start = max(cpl, 1)
        _, full_lps = self._echo_logprobs(f"{context} {word}")
        ctx_vals = [x for x in full_lps[cand_start:] if x is not None and math.isfinite(x)]

        ctx_sum = sum(ctx_vals)
        base_sum = sum(base_vals)

        return SurprisalResult(
            word=word,
            ctx_sum=ctx_sum,
            base_sum=base_sum,
            surprisal=ctx_sum - base_sum,
        )

    def score_candidates(self, context: str, candidates: list[str]) -> list[SurprisalResult]:
        """Score multiple candidates. Returns results aligned with input order."""
        return [self.score_word(context, c) for c in candidates]
