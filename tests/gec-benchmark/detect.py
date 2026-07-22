#!/usr/bin/env python3
"""
BERTeus real-word error detection (Tier 2.5).

Detects real-word errors — valid dictionary words used in the wrong
context (grammar, morphology, confusable pairs) — which the dictionary-
based detector (Tier 1) cannot catch because the word IS in the dictionary.

ALGORITHM (masked embedding similarity — same approach as Tier 2 re-ranking,
but applied to EVERY in-dictionary content word, not just flagged ones):

  For each content word w at position i in the sentence:
    1. Replace w with [MASK], run BERT encoder (one forward pass)
    2. Extract [MASK] hidden state h (768-dim) — the context's "prediction"
    3. sim_actual = cosine(h, embedding(w))     — how well does the actual word fit?
    4. sim_best   = max over all dict words of cosine(h, embedding(word))
    5. margin     = sim_best - sim_actual
    6. If margin > THRESHOLD and top_word != w → flag as real-word error

Detection and candidate generation are the same operation: the top-k
nearest words to h ARE the correction candidates.

EVALUATION:
  - Dem_single (250 manually-revised grammar errors): recall, precision, F1
    All errors are real-word (R1-R4: tense, agreement, case, suffix)
  - Dem_none (201 clean sentences): false-positive rate
  - Candidate quality: is the correct word in top-5 / top-1?

Usage:
    uv run python tests/gec-benchmark/detect.py
    uv run python tests/gec-benchmark/detect.py --no-cache  # force recompute
    uv run python tests/gec-benchmark/detect.py --skip-short 3  # min word length

Prerequisites:
    - ixa-ehu/berteus-base-cased (downloaded from HuggingFace, cached)
    - data/eu-words-freq.txt (160k wordlist)
    - tests/gec-benchmark/elhuyar/*.tsv (Elhuyar GEC dataset)
"""
from __future__ import annotations

import sys
import re
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertModel

# Add parent dirs to path for imports
HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from tier1 import (
    load_freq_map, load_elhuyar_tsv, find_differences,
    edits1, levenshtein,
)

DATA_DIR = REPO_ROOT / "data"
ELHUYAR_DIR = HERE / "elhuyar"
FREQ_PATH = DATA_DIR / "eu-words-freq.txt"
CACHE_PATH = HERE / "detection_scores_cache.json"


# ── Data structures ──────────────────────────────────

@dataclass
class WordScore:
    """Detection score for a single word in a sentence."""
    position: int
    word: str           # original word (with punctuation)
    clean_word: str     # stripped of punctuation
    in_dict: bool       # is this word in our dictionary?
    sim_actual: float   # cosine sim of actual word's embedding with mask hidden
    sim_best: float     # best cosine sim over all dictionary words
    margin: float       # sim_best - sim_actual
    top_word: str       # best-matching dictionary word
    top5_words: list    # top-5 best-matching words
    top5_sims: list     # their similarity scores
    # Confusable-set approach: compare only against edit-distance 1-2 neighbors
    conf_candidates: list   # edit-distance 1-2 neighbors in dictionary
    conf_sims: list         # their cosine sims with mask hidden
    conf_margin: float      # best confusable sim - actual sim (0 if no candidates)
    conf_top_word: str      # best confusable candidate ("" if none)


# ── Detector ─────────────────────────────────────────

class BerteusDetector:
    """BERTeus-based real-word error detector.

    Uses the same masked embedding similarity approach as the Tier 2
    re-ranker, but applies it to every in-dictionary content word for
    detection (not just dictionary-flagged words).
    """

    def __init__(self, model_name: str = "ixa-ehu/berteus-base-cased", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and device == "cuda":
            print("  ⚠️  CUDA not available, using CPU")
        self.tokenizer: Optional[BertTokenizerFast] = None
        self.model: Optional[BertModel] = None
        self.mask_token_id: Optional[int] = None
        self.word_embeddings = None  # (vocab, 768) — raw BERT token embeddings

        # Word-level embedding index (built from frequency map)
        self.word_emb_matrix_norm: Optional[torch.Tensor] = None  # (n_words, 768)
        self.idx_to_word: list[str] = []
        self.word_to_idx: dict[str, int] = {}

    def load(self):
        """Load BERTeus model + tokenizer."""
        print(f"  Loading BERTeus: {self.model_name}")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.mask_token_id = self.tokenizer.mask_token_id
        self.word_embeddings = self.model.embeddings.word_embeddings.weight  # (vocab, 768)
        print(f"  BERTeus loaded on {self.device} "
              f"({self.model.num_parameters() / 1e6:.0f}M params, "
              f"vocab={self.word_embeddings.size(0)})")

    def build_word_index(self, word_list: list[str]):
        """Build a normalized word-level embedding matrix from a word list.

        For each word, tokenize it (subword) and take the mean of its
        token embeddings. Normalize each row for cosine similarity.
        """
        print(f"  Building word index for {len(word_list)} words...")
        t0 = time.time()
        batch_size = 1024
        embeddings_list = []
        valid_words = []

        for i in range(0, len(word_list), batch_size):
            batch = word_list[i:i + batch_size]
            encodings = self.tokenizer(
                batch,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=20,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device).float()

            # (batch, max_len, 768)
            embs = self.word_embeddings[input_ids]
            # Mask out padding tokens
            embs = embs * attention_mask.unsqueeze(-1)
            # Mean over actual tokens
            sums = embs.sum(dim=1)                     # (batch, 768)
            counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            means = sums / counts                       # (batch, 768)

            has_tokens = attention_mask.sum(dim=1) > 0
            for j, (word, valid) in enumerate(zip(batch, has_tokens)):
                if valid:
                    embeddings_list.append(means[j])
                    valid_words.append(word)

        self.word_emb_matrix_norm = F.normalize(torch.stack(embeddings_list), dim=1)
        self.idx_to_word = valid_words
        self.word_to_idx = {w: i for i, w in enumerate(valid_words)}
        print(f"  Word index: {len(valid_words)} words "
              f"({self.word_emb_matrix_norm.size()}), "
              f"built in {time.time() - t0:.1f}s")

    def _get_word_embedding(self, word: str) -> Optional[torch.Tensor]:
        """Compute mean token embedding for a word on the fly."""
        ids = self.tokenizer(word, add_special_tokens=False)["input_ids"]
        if not ids:
            return None
        ids_tensor = torch.tensor(ids, device=self.device)
        embs = self.word_embeddings[ids_tensor]  # (n_tokens, 768)
        return embs.mean(dim=0)                   # (768,)

    def score_sentence(self, sentence: str, skip_short: int = 2) -> list[WordScore]:
        """Score every content word in a sentence for real-word error detection.

        Two scoring approaches are computed for each word:

        1. ALL-WORDS margin: sim_best (over all 160k dict words) - sim_actual.
           Broad but noisy — the actual word is rarely the top-1 out of 160k,
           so margin > 0 even for correct words.

        2. CONFUSABLE margin: sim_best (over edit-distance 1-2 neighbors
           in dictionary) - sim_actual. Targeted — only flags when a
           spelling-similar word fits the context significantly better.
           This is the standard real-word error detection approach.

        Args:
            sentence: The input sentence (erroneous for detection, correct for FP test)
            skip_short: Skip words shorter than this many characters (after stripping
                        punctuation). Default 2 = skip 1-char tokens (punctuation artifacts).

        Returns:
            List of WordScore for each checked word.
        """
        words = sentence.split()
        if len(words) < 2:
            return []

        scores = []
        for i, word in enumerate(words):
            clean = re.sub(r"[^A-Za-zÀ-ÿ'\-]", "", word)
            if len(clean) < skip_short:
                continue
            # Skip proper nouns / acronyms (ALLCAPS, len > 1)
            if word.isupper() and len(word) > 1:
                continue
            # Skip pure numbers
            if not clean[0].isalpha():
                continue

            word_lower = clean.lower()

            # Get actual word's embedding (lowercase to match word index)
            actual_emb = self._get_word_embedding(word_lower)
            if actual_emb is None:
                continue

            in_dict = word_lower in self.word_to_idx

            # Mask the word and run BERT
            masked_words = list(words)
            masked_words[i] = self.tokenizer.mask_token
            text = " ".join(masked_words)

            encoding = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            mask_positions = (input_ids[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                continue
            mask_pos = mask_positions[0].item()

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                mask_hidden = outputs.last_hidden_state[0, mask_pos]  # (768,)

            mask_hidden_norm = F.normalize(mask_hidden, dim=0)
            actual_norm = F.normalize(actual_emb, dim=0)
            sim_actual = torch.dot(mask_hidden_norm, actual_norm).item()

            # ── All-words margin ──
            sims = self.word_emb_matrix_norm @ mask_hidden_norm  # (n_words,)
            top_k = min(5, len(sims))
            top_sims, top_indices = torch.topk(sims, top_k)
            top_words = [self.idx_to_word[idx.item()] for idx in top_indices]
            top_sims_list = [round(s.item(), 6) for s in top_sims]
            sim_best = top_sims_list[0]
            top_word = top_words[0]
            margin = round(sim_best - sim_actual, 6)

            # ── Confusable-set margin ──
            # Generate edit-distance 1 variants, filter by dictionary
            e1 = edits1(word_lower)
            conf_candidates = [w for w in e1 if w in self.word_to_idx]
            # Also add edit-distance 2 (edits1 of edits1, filtered)
            for w in list(e1)[:200]:  # limit to avoid explosion
                for w2 in edits1(w):
                    if w2 in self.word_to_idx and w2 not in conf_candidates and w2 != word_lower:
                        conf_candidates.append(w2)
            # Deduplicate, preserve order
            seen = set()
            conf_candidates = [w for w in conf_candidates if not (w in seen or seen.add(w))]

            conf_sims = []
            conf_top_word = ""
            conf_margin = 0.0
            if conf_candidates:
                # Look up embeddings for all confusable candidates
                cand_indices = [self.word_to_idx[w] for w in conf_candidates]
                cand_embs = self.word_emb_matrix_norm[cand_indices]  # (n_cand, 768)
                conf_sims_tensor = cand_embs @ mask_hidden_norm  # (n_cand,)
                conf_sims = [round(s.item(), 6) for s in conf_sims_tensor]
                best_idx = conf_sims_tensor.argmax().item()
                conf_best_sim = conf_sims[best_idx]
                conf_top_word = conf_candidates[best_idx]
                conf_margin = round(conf_best_sim - sim_actual, 6)

            scores.append(WordScore(
                position=i,
                word=word,
                clean_word=clean,
                in_dict=in_dict,
                sim_actual=round(sim_actual, 6),
                sim_best=sim_best,
                margin=margin,
                top_word=top_word,
                top5_words=top_words,
                top5_sims=top_sims_list,
                conf_candidates=conf_candidates[:20],  # cap for cache size
                conf_sims=conf_sims[:20],
                conf_margin=conf_margin,
                conf_top_word=conf_top_word,
            ))

        return scores


# ── Evaluation ───────────────────────────────────────

def eval_at_margin(
    dem_single_scores: list[list[WordScore]],
    dem_single_truth: list[list[dict]],
    dem_none_scores: list[list[WordScore]],
    margin: float,
    use_confusable: bool = False,
) -> dict:
    """Evaluate detection at a given margin threshold.

    A word is flagged if: margin > threshold AND top_word != actual_word.

    Args:
        use_confusable: If True, use confusable-set margin (edit-distance 1-2
                       neighbors) instead of all-words margin.
    """
    margin_attr = "conf_margin" if use_confusable else "margin"
    top_attr = "conf_top_word" if use_confusable else "top_word"
    top5_attr = "conf_candidates" if use_confusable else "top5_words"
    sims_attr = "conf_sims" if use_confusable else "top5_sims"

    tp = 0       # correctly flagged an actual error
    fp = 0       # flagged a non-error word (in Dem_single)
    fn = 0       # missed an actual error
    total_errors = 0
    correct_in_top5 = 0
    correct_in_top1 = 0

    for scores, truth in zip(dem_single_scores, dem_single_truth):
        error_positions = set(t["position"] for t in truth)
        correct_words_by_pos = {t["position"]: t["correct_word"].lower() for t in truth}
        total_errors += len(error_positions)

        flagged_positions = set()
        for ws in scores:
            ws_margin = getattr(ws, margin_attr)
            ws_top = getattr(ws, top_attr)
            ws_top5 = getattr(ws, top5_attr)
            ws_top5_sims = getattr(ws, sims_attr)

            is_flagged = (
                ws_margin > margin
                and ws_top
                and ws_top.lower() != ws.clean_word.lower()
            )
            if not is_flagged:
                continue

            flagged_positions.add(ws.position)

            if ws.position in error_positions:
                tp += 1
                correct_word = correct_words_by_pos.get(ws.position, "")
                top5_lower = [w.lower() for w in ws_top5]
                if correct_word in top5_lower:
                    correct_in_top5 += 1
                if top5_lower and top5_lower[0] == correct_word:
                    correct_in_top1 += 1
            else:
                fp += 1

        for pos in error_positions:
            if pos not in flagged_positions:
                fn += 1

    # Dem_none: false positives on clean text
    fp_none = 0
    total_checked_none = 0
    for scores in dem_none_scores:
        for ws in scores:
            total_checked_none += 1
            ws_margin = getattr(ws, margin_attr)
            ws_top = getattr(ws, top_attr)
            if (ws_margin > margin
                    and ws_top
                    and ws_top.lower() != ws.clean_word.lower()):
                fp_none += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    fp_rate = fp_none / total_checked_none if total_checked_none > 0 else 0.0

    return {
        "margin": margin,
        "use_confusable": use_confusable,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_errors": total_errors,
        "fp_none": fp_none,
        "total_checked_none": total_checked_none,
        "fp_rate_none": fp_rate,
        "correct_in_top5": correct_in_top5,
        "correct_in_top1": correct_in_top1,
        "detected_errors": tp,
    }


# ── Cache I/O ────────────────────────────────────────

def save_cache(path: Path, data: dict):
    """Save detection scores to JSON cache."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Cached scores → {path.name} ({path.stat().st_size / 1024:.0f} KB)")


def load_cache(path: Path) -> dict:
    """Load detection scores from JSON cache."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # Reconstruct WordScore objects
    data = {
        "dem_single": [[WordScore(**ws) for ws in sent] for sent in raw["dem_single"]],
        "dem_none": [[WordScore(**ws) for ws in sent] for sent in raw["dem_none"]],
    }
    return data


# ── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BERTeus real-word error detection (Tier 2.5)"
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="Force recompute detection scores (ignore cache)")
    parser.add_argument("--skip-short", type=int, default=2,
                        help="Skip words shorter than this (default: 2)")
    parser.add_argument("--margins", type=str, default=None,
                        help="Comma-separated margin thresholds for grid search")
    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  TIER 2.5 — BERTEUS REAL-WORD ERROR DETECTION             ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # ── Load data ──
    print("Loading frequency map (eu-words-freq.txt)...")
    fmap = load_freq_map(str(FREQ_PATH))
    print(f"  {len(fmap)} words loaded\n")

    print("Loading Elhuyar datasets...")
    dem_single = load_elhuyar_tsv(str(ELHUYAR_DIR / "Dem_single.tsv"))
    dem_none = load_elhuyar_tsv(str(ELHUYAR_DIR / "Dem_none.tsv"))
    print(f"  Dem_single: {len(dem_single)} grammar-error sentences")
    print(f"  Dem_none:   {len(dem_none)} clean sentences")

    # Ground truth: which words differ in each Dem_single pair
    dem_single_truth = [find_differences(gc["correct"], gc["erroneous"])
                        for gc in dem_single]
    total_gt_errors = sum(len(t) for t in dem_single_truth)
    print(f"  Ground truth errors: {total_gt_errors} words across {len(dem_single)} sentences")

    # ── Compute or load cached detection scores ──
    if CACHE_PATH.exists() and not args.no_cache:
        print(f"\nLoading cached detection scores from {CACHE_PATH.name}...")
        cached = load_cache(CACHE_PATH)
        dem_single_scores = cached["dem_single"]
        dem_none_scores = cached["dem_none"]
        print(f"  Loaded {len(dem_single_scores)} + {len(dem_none_scores)} sentence scores")
    else:
        print("\nComputing detection scores (one-time, ~30s on GPU)...")
        detector = BerteusDetector()
        detector.load()
        detector.build_word_index(list(fmap.keys()))

        t0 = time.time()
        print(f"\n  Scoring Dem_single ({len(dem_single)} sentences)...")
        dem_single_scores = []
        for i, gc in enumerate(dem_single):
            dem_single_scores.append(
                detector.score_sentence(gc["erroneous"], skip_short=args.skip_short)
            )
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(dem_single)}...")

        print(f"  Scoring Dem_none ({len(dem_none)} sentences)...")
        dem_none_scores = []
        for i, gc in enumerate(dem_none):
            dem_none_scores.append(
                detector.score_sentence(gc["correct"], skip_short=args.skip_short)
            )
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(dem_none)}...")

        elapsed = time.time() - t0
        n_sentences = len(dem_single) + len(dem_none)
        print(f"\n  Done: {n_sentences} sentences in {elapsed:.1f}s "
              f"({n_sentences / elapsed:.1f} sent/s)")

        # Cache results
        cache_data = {
            "dem_single": [[asdict(ws) for ws in sent] for sent in dem_single_scores],
            "dem_none": [[asdict(ws) for ws in sent] for sent in dem_none_scores],
        }
        save_cache(CACHE_PATH, cache_data)

    # ── Grid search margin thresholds (both approaches) ──
    if args.margins:
        margins = [float(m) for m in args.margins.split(",")]
    else:
        margins = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]

    best_overall = None
    for approach, use_conf in [("ALL-WORDS", False), ("CONFUSABLE", True)]:
        print(f"\n{'='*70}")
        print(f"  GRID SEARCH: {approach} margin threshold")
        print(f"{'='*70}")
        print(f"  {'margin':>7}  {'recall':>7}  {'prec':>7}  {'F1':>7}  "
              f"{'TP':>4}  {'FP':>4}  {'FN':>4}  "
              f"{'FP%':>6}  {'top5':>5}  {'top1':>5}")
        print(f"  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  "
              f"{'─'*4}  {'─'*4}  {'─'*4}  "
              f"{'─'*6}  {'─'*5}  {'─'*5}")

        best_f1 = 0
        best_result = None
        for margin in margins:
            r = eval_at_margin(dem_single_scores, dem_single_truth,
                               dem_none_scores, margin, use_confusable=use_conf)
            print(f"  {margin:>7.2f}  {r['recall']:>7.1%}  {r['precision']:>7.1%}  "
                  f"{r['f1']:>7.1%}  "
                  f"{r['tp']:>4}  {r['fp']:>4}  {r['fn']:>4}  "
                  f"{r['fp_rate_none']:>5.1%}  "
                  f"{r['correct_in_top5']:>5}  {r['correct_in_top1']:>5}")
            if r["f1"] > best_f1:
                best_f1 = r["f1"]
                best_result = r

        print(f"\n  ★ Best F1 = {best_f1:.1%} at margin={best_result['margin']:.2f}")
        print(f"    recall={best_result['recall']:.1%}  "
              f"precision={best_result['precision']:.1%}  "
              f"FP rate on clean={best_result['fp_rate_none']:.1%}")
        print(f"    Of {best_result['detected_errors']} detected errors: "
              f"{best_result['correct_in_top5']} correct in top-5, "
              f"{best_result['correct_in_top1']} correct as top-1")

        if best_overall is None or best_f1 > best_overall[1]:
            best_overall = (approach, best_f1, best_result)

    # ── Error type breakdown at best approach/margin ──
    best_approach, _, best_result = best_overall
    use_conf = best_approach == "CONFUSABLE"
    margin_attr = "conf_margin" if use_conf else "margin"
    top_attr = "conf_top_word" if use_conf else "top_word"

    print(f"\n{'='*70}")
    print(f"  ERROR TYPE BREAKDOWN ({best_approach}, margin={best_result['margin']:.2f})")
    print(f"{'='*70}")
    type_stats = {}
    for gc, scores, truth in zip(dem_single, dem_single_scores, dem_single_truth):
        ets = gc.get("error_types", "").split(",")
        ets = [e.strip() for e in ets if e.strip()]
        primary_type = ets[0] if ets else "?"
        if primary_type not in type_stats:
            type_stats[primary_type] = {"total": 0, "detected": 0}
        for t in truth:
            type_stats[primary_type]["total"] += 1
            pos = t["position"]
            for ws in scores:
                if (ws.position == pos
                        and getattr(ws, margin_attr) > best_result["margin"]
                        and getattr(ws, top_attr)
                        and getattr(ws, top_attr).lower() != ws.clean_word.lower()):
                    type_stats[primary_type]["detected"] += 1
                    break

    type_names = {"R1": "R1 (tense)", "R2": "R2 (agreement)",
                  "R3": "R3 (case)", "R4": "R4 (suffix)"}
    for et in sorted(type_stats):
        s = type_stats[et]
        name = type_names.get(et, et)
        det_pct = f"{s['detected']}/{s['total']}" if s['total'] > 0 else "0/0"
        print(f"  {name:20s}  recall={s['detected']/s['total']:.1%}  ({det_pct})")


if __name__ == "__main__":
    main()
