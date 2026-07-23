"""
Spell checking — Hunspell + Tier 1 freq re-ranking + Tier 2 BERTeus.

Port of txukun's spell.js. Two-tier re-ranking:
  Tier 1 (fast): frequency + edit distance via get_ranked_candidates()
  Tier 2 (slow): BERTeus masked embedding similarity (lazy-loaded)

Candidate pool = (edit-distance-1 variants ∩ wordlist) ∪ hunspell_suggestions
Score = β·log(freq+1) + δ·(1/(1+ed)) + BERT_WEIGHT × cosine_sim
"""
from __future__ import annotations

import re
import math
import subprocess
from pathlib import Path
from dataclasses import dataclass

from .bert import BerteusReranker, BERT_WEIGHT

# ── Constants (match txukun src/spell.js) ───────────

SCORE_BETA = 0.3
SCORE_DELTA = 0.5
EU_ALPHABET = "abcdefghijklmnopqrstuvwxyzáéíóúüñçàèìòùâêîôû"

WORD_RE = re.compile(
    r"https?://\S+"
    r"|[\w.-]+@[\w.-]+"
    r"|[a-zA-ZáéíóúüñÁÉÍÓÚÜÑàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛçÇ'\-]+"
    r"|\d+(?:[.,]\d+)*"
)

DICT_PATH = Path(__file__).parent.parent / "data" / "eu"


@dataclass
class Candidate:
    word: str
    score: float
    freq: int = 0
    ed: int = 0


@dataclass
class SpellError:
    word: str
    start: int
    end: int
    suggestions: list[str]


# ── Tier 1: edit distance + frequency ───────────────

def edits1(word: str) -> set[str]:
    """Generate all edit-distance-1 variants (Norvig-style)."""
    w = (word or "").lower()
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    results: set[str] = set()
    for a, b in splits:
        if b:
            results.add(a + b[1:])                        # deletion
        if len(b) > 1:
            results.add(a + b[1] + b[0] + b[2:])          # transposition
        for c in EU_ALPHABET:
            if b:
                results.add(a + c + b[1:])                # substitution
            results.add(a + c + b)                        # insertion
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
    """Restore the case pattern of source onto target."""
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
    hunspell_suggestions: list[str] | None,
    fmap: dict[str, int],
) -> list[Candidate]:
    """Generate and rank spell-correction candidates (Tier 1)."""
    if not typed:
        return []
    typed_lower = typed.lower()
    ed1_variants = edits1(typed_lower)
    pool: set[str] = set()

    for v in ed1_variants:
        if v in fmap:
            pool.add(v)

    if hunspell_suggestions:
        for s in hunspell_suggestions:
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
    for c in ranked:
        c.word = match_case(typed, c.word)
    return ranked


def load_freq_map(path: str | Path) -> dict[str, int]:
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


# ── Hunspell subprocess ──────────────────────────────

class HunspellChecker:
    """Hunspell via persistent subprocess pipe (ispell -a format)."""

    def __init__(self, dict_path: Path | None = None):
        self._dict = str(dict_path or DICT_PATH)
        self._proc: subprocess.Popen | None = None
        self._broken = False

    def _ensure(self):
        if self._proc is not None and self._proc.poll() is not None:
            self._close()
        if self._proc is None and not self._broken:
            try:
                self._proc = subprocess.Popen(
                    ["hunspell", "-a", "-d", self._dict],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                self._proc.stdout.readline()  # consume header
            except (FileNotFoundError, OSError):
                self._broken = True
                self._proc = None

    def _close(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write("#\n")
                self._proc.stdin.flush()
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    @property
    def loaded(self) -> bool:
        if self._broken:
            return False
        self._ensure()
        return self._proc is not None and self._proc.poll() is None

    def correct(self, word: str) -> bool:
        if not self.loaded:
            return True
        if word.isdigit() or "@" in word or word.startswith("http"):
            return True
        self._ensure()
        try:
            self._proc.stdin.write(word + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline().strip()
            if not line:
                line = self._proc.stdout.readline().strip()
                if not line:
                    return True
            return line.startswith("*") or line.startswith("+")
        except (BrokenPipeError, OSError):
            self._close()
            return True

    def suggest(self, word: str) -> list[str]:
        if not self.loaded:
            return []
        if word.isdigit() or "@" in word or word.startswith("http"):
            return []
        self._ensure()
        try:
            self._proc.stdin.write(word + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline().strip()
            if not line:
                line = self._proc.stdout.readline().strip()
                if not line:
                    return []
            if (line.startswith("&") or line.startswith("#")) and ":" in line:
                part = line.split(":", 1)[1].strip()
                return [s.strip() for s in part.split(", ") if s.strip()]
            return []
        except (BrokenPipeError, OSError):
            self._close()
            return []

    def __del__(self):
        self._close()


# ── Spelling detector + corrector ───────────────────

class SpellChecker:
    """Full spell checker: Hunspell detection + Tier 1 + Tier 2 re-ranking."""

    def __init__(self, freq_path: Path | None = None, bert: BerteusReranker | None = None, quiet: bool = False):
        self._hunspell = HunspellChecker()
        self._freq_map: dict[str, int] | None = None
        self._freq_path = freq_path or (Path(__file__).parent.parent / "data" / "eu-words-freq.txt")
        self._bert = bert
        self._quiet = quiet

    def _ensure_freq(self):
        if self._freq_map is None:
            self._freq_map = load_freq_map(self._freq_path)

    @property
    def ready(self) -> bool:
        return self._hunspell.loaded

    def check_spelling(self, text: str) -> list[SpellError]:
        """Find all misspelled words in text."""
        if not self._hunspell.loaded:
            return []

        tokens = [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]
        if not tokens:
            return []

        # Filter non-words
        candidates = []
        for i, (word, start, end) in enumerate(tokens):
            if re.match(r"^\d+([.,]\d+)*$", word):
                continue
            if word.startswith("http") or "@" in word:
                continue
            if len(word) < 2:
                continue
            if word == word.upper() and len(word) > 1:
                continue
            if len(word) <= 5 and i > 0 and re.match(r"^\d+([.,]\d+)*$", tokens[i - 1][0]):
                continue
            candidates.append((word, start, end))

        if not candidates:
            return []

        # Batch check
        results = [self._hunspell.correct(w) for w, _, _ in candidates]
        errors = []
        misspelled = [(w, s, e) for (w, s, e), ok in zip(candidates, results) if not ok]

        for word, start, end in misspelled:
            suggestions = self._hunspell.suggest(word)
            suggestions = [s for s in suggestions if not s.isupper()]
            errors.append(SpellError(word=word, start=start, end=end, suggestions=suggestions))

        return errors

    def get_best_correction(self, full_text: str, err: SpellError) -> Candidate | None:
        """Two-tier re-ranking: Tier 1 freq + Tier 2 BERTeus."""
        self._ensure_freq()
        ranked = get_ranked_candidates(err.word, err.suggestions, self._freq_map)
        if not ranked:
            return None

        best = ranked[0]

        # Tier 2: BERTeus when ≥2 candidates
        if len(ranked) >= 2 and self._bert is not None:
            if not self._bert.ready and not self._bert.failed:
                self._bert._load()
            if self._bert.ready:
                candidates_lower = [c.word.lower() for c in ranked[:5]]
                bert_scores = self._bert.score_candidates(
                    full_text, err.start, err.end, candidates_lower
                )
                best_combined = -math.inf
                for i in range(min(len(ranked), len(bert_scores))):
                    combined = ranked[i].score + BERT_WEIGHT * bert_scores[i]
                    if combined > best_combined:
                        best_combined = combined
                        best = ranked[i]

        return best

    def auto_correct(self, text: str) -> tuple[str, int]:
        """Replace each misspelled word with the best candidate."""
        errors = self.check_spelling(text)
        if not errors:
            return text, 0

        result = text
        changes = 0
        for err in sorted(errors, key=lambda e: e.start, reverse=True):
            best = self.get_best_correction(result, err)
            if best and best.word != err.word:
                result = result[:err.start] + best.word + result[err.end:]
                changes += 1

        return result, changes
