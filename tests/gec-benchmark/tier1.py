"""
Tier 1 spell-correction candidate generation — Python port of txukun's
src/spell.js pure functions (edits1, levenshtein, matchCase, getRankedCandidates).

Candidate pool = (edit-distance-1 variants ∩ wordlist) ∪ hunspell_suggestions
Score = β·log(freq+1) + δ·(1/(1+ed))

This is a standalone module — no Hunspell dependency. The wordlist is the
160k Xuxen-derived eu-words-freq.txt (same as the browser version).
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass

# ── Constants (match src/spell.js) ───────────────────

SCORE_BETA = 0.3
SCORE_DELTA = 0.5
EU_ALPHABET = "abcdefghijklmnopqrstuvwxyzáéíóúüñçàèìòùâêîôû"

# Basque word tokenizer (matches txukun.py WORD_RE)
WORD_RE = re.compile(
    r"https?://\S+"
    r"|[\w.-]+@[\w.-]+"
    r"|[a-zA-ZáéíóúüñÁÉÍÓÚÜÑàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛçÇ'\-]+"
    r"|\d+(?:[.,]\d+)*"
)


@dataclass
class Candidate:
    word: str
    score: float
    freq: int
    ed: int


# ── Pure functions ───────────────────────────────────

def edits1(word: str) -> set[str]:
    """Generate all edit-distance-1 variants (deletions, transpositions,
    substitutions, insertions) using the Basque alphabet."""
    w = (word or "").lower()
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    results: set[str] = set()
    for a, b in splits:
        if b:
            results.add(a + b[1:])                         # deletion
        if len(b) > 1:
            results.add(a + b[1] + b[0] + b[2:])           # transposition
        for c in EU_ALPHABET:
            if b:
                results.add(a + c + b[1:])                 # substitution
            results.add(a + c + b)                         # insertion
    results.discard(w)
    return results


def levenshtein(a: str, b: str) -> int:
    a = (a or "").lower()
    b = (b or "").lower()
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def match_case(source: str, target: str) -> str:
    """Preserve the casing pattern of source in target."""
    if not source or not target:
        return target
    letters = [ch for ch in source if ch.isalpha()]
    if not letters:
        return target
    upper_count = sum(1 for ch in letters if ch.isupper())
    if upper_count == len(letters) and len(letters) > 1:
        return target.upper()
    if letters[0].isupper() and all(not ch.isupper() for ch in letters[1:]):
        t = target.lower()
        return t[0].upper() + t[1:]
    return target.lower()


def get_ranked_candidates(
    typed: str,
    extra_suggestions: list[str] | None,
    fmap: dict[str, int],
) -> list[Candidate]:
    """Generate and rank spell-correction candidates.

    Pool = (edits1(typed) ∩ wordlist) ∪ extra_suggestions
    Score = β·log(freq+1) + δ·(1/(1+ed))
    """
    if not typed:
        return []
    typed_lower = typed.lower()
    ed1_variants = edits1(typed_lower)
    pool: set[str] = set()

    for v in ed1_variants:
        if v in fmap:
            pool.add(v)

    if extra_suggestions:
        for s in extra_suggestions:
            if not s:
                continue
            s_low = s.lower()
            if s_low != typed_lower:
                pool.add(s_low)

    ranked: list[Candidate] = []
    for cand in pool:
        freq = fmap.get(cand, 0)
        ed = 1 if cand in ed1_variants else levenshtein(typed_lower, cand)
        if freq <= 0 and ed != 1:
            continue
        score = SCORE_BETA * math.log(freq + 1) + SCORE_DELTA * (1 / (1 + ed))
        ranked.append(Candidate(word=cand, score=score, freq=freq, ed=ed))

    ranked.sort(key=lambda c: c.score, reverse=True)
    # Apply casing after sorting
    for c in ranked:
        c.word = match_case(typed, c.word)
    return ranked


# ── Data loading ─────────────────────────────────────

def load_freq_map(path: str) -> dict[str, int]:
    """Load eu-words-freq.txt → {word: freq} dict (lowercase keys)."""
    fmap: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            tab = line.find("\t")
            if tab <= 0:
                continue
            word = line[:tab].strip().lower()
            try:
                count = int(line[tab + 1:])
            except ValueError:
                count = 0
            if word:
                fmap[word] = count
    return fmap


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Tokenize text. Returns list of (word, start, end)."""
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def should_check_word(word: str, prev_word: str | None) -> bool:
    """Check if a word should be spell-checked (matches txukun.py filter)."""
    if re.match(r"^\d+([.,]\d+)*$", word):
        return False
    if word.startswith("http"):
        return False
    if "@" in word:
        return False
    if len(word) < 2:
        return False
    if word == word.upper() and len(word) > 1:
        return False
    if len(word) <= 5 and prev_word and re.match(r"^\d+([.,]\d+)*$", prev_word):
        return False
    return True


# ── Elhuyar data loading ─────────────────────────────

def load_elhuyar_tsv(path: str) -> list[dict[str, str]]:
    """Parse an Elhuyar TSV file.
    Format: ORIGINAL_SENTENCE\\tSENTENCE_WITH_ERRORS\\tERROR_TYPES
    Header row is skipped.
    """
    results = []
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")[1:]  # skip header
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        correct = parts[0].strip() if len(parts) > 0 else ""
        erroneous = parts[1].strip() if len(parts) > 1 else ""
        error_types = parts[2].strip() if len(parts) > 2 else ""
        if correct and erroneous:
            results.append({
                "correct": correct,
                "erroneous": erroneous,
                "error_types": error_types,
            })
    return results


def load_correct_sentences(path: str) -> list[str]:
    """Load only the correct (first column) sentences from an Elhuyar TSV."""
    return [p["correct"] for p in load_elhuyar_tsv(path)]


def find_differences(correct: str, erroneous: str) -> list[dict[str, str]]:
    """Find which words differ between correct and erroneous sentences."""
    c_words = correct.split()
    e_words = erroneous.split()
    diffs = []
    for i in range(max(len(c_words), len(e_words))):
        cw = c_words[i] if i < len(c_words) else ""
        ew = e_words[i] if i < len(e_words) else ""
        if cw != ew:
            cw_clean = re.sub(r"[^A-Za-zÀ-ÿ''\-]", "", cw)
            ew_clean = re.sub(r"[^A-Za-zÀ-ÿ''\-]", "", ew)
            if cw_clean and ew_clean and cw_clean.lower() != ew_clean.lower():
                diffs.append({
                    "correct_word": cw_clean,
                    "erroneous_word": ew_clean,
                    "position": i,
                })
    return diffs
