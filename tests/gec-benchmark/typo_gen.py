"""
Synthetic typo generator for GEC benchmark — port of futo-transformer-basque's
typo_synthesis.py, stripped of the FUTO shortcut dependency.

Strategies (weighted, mixed per-word):
  1. Keyboard-adjacency (QWERTY, letters only)
  2. Missing-diacritic / ñ loss (NFD decompose, drop combining marks)
  3. Transposed adjacent letters
  4. Single-char insertion
  5. Single-char deletion
  6. Doubled char

Used to generate synthetic spelling-error sentences from the Elhuyar
correct-sentence set. The correct version is known by construction —
no Basque expertise needed.
"""
from __future__ import annotations

import re
import unicodedata
import random
from dataclasses import dataclass

# QWERTY adjacency map (letters only). Each key → its letter neighbours.
ADJ: dict[str, str] = {
    "q": "wa", "w": "qesa", "e": "wrds", "r": "etfd", "t": "rygf",
    "y": "tuhg", "u": "yijh", "i": "uokj", "o": "iplk", "p": "ol",
    "a": "qsz", "s": "awxd", "d": "sexf", "f": "drcvg", "g": "ftvbh",
    "h": "gybnj", "j": "hubmnk", "k": "jimol", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xvdf", "v": "cbfg", "b": "vghn",
    "n": "bhjm", "m": "njkl",
}

EU_ALPHABET = "abcdefghijklmnopqrstuvwxyzáéíóúüñçàèìòùâêîôû"

RULE_NAMES = ["drop_accent", "adj_typo", "transpose", "delete", "insert", "double"]


@dataclass
class TypoEdit:
    word: str        # original correct word (stripped of punctuation)
    typo: str        # the misspelled version
    type: str        # which rule produced this typo
    position: int    # word index in the sentence


@dataclass
class TypoCase:
    correct: str     # original correct sentence
    erroneous: str   # sentence with typo(s) injected
    edits: list[TypoEdit]


# ── Helpers ─────────────────────────────────────────

def _strip_accents(s: str) -> str:
    """á → a, ñ → n, ü → u. NFD-decompose then drop combining marks."""
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch) != "Mn":
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _has_accent(s: str) -> bool:
    return unicodedata.normalize("NFD", s) != s


# ── Typo rules ──────────────────────────────────────

def _drop_accent(w: str, rng: random.Random) -> str:
    if not _has_accent(w):
        return w
    return _strip_accents(w)


def _adj_typo(w: str, rng: random.Random) -> str:
    if not w:
        return w
    chars = list(w)
    candidates = [i for i, c in enumerate(chars) if c.lower() in ADJ]
    if not candidates:
        return w
    i = rng.choice(candidates)
    c = chars[i].lower()
    neighbours = ADJ.get(c, "")
    if not neighbours:
        return w
    new_c = rng.choice(neighbours)
    if chars[i].isupper():
        new_c = new_c.upper()
    chars[i] = new_c
    return "".join(chars)


def _transpose(w: str, rng: random.Random) -> str:
    if len(w) < 3:
        return w
    i = rng.randrange(len(w) - 1)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def _insert(w: str, rng: random.Random) -> str:
    if not w:
        return w
    i = rng.randrange(len(w) + 1)
    extra = rng.choice(EU_ALPHABET)
    return w[:i] + extra + w[i:]


def _delete(w: str, rng: random.Random) -> str:
    if len(w) <= 2:
        return w
    i = rng.randrange(len(w))
    return w[:i] + w[i + 1:]


def _double(w: str, rng: random.Random) -> str:
    if not w:
        return w
    i = rng.randrange(len(w))
    return w[:i] + w[i] + w[i:]


# Per-rule weight (matches futo repo: adjacency dominates, accent-drop lighter
# because standard Basque uses few diacritics).
RULES = [
    (_drop_accent, 20),
    (_adj_typo,    30),
    (_transpose,   15),
    (_delete,      12),
    (_insert,      11),
    (_double,      12),
]


def synth_typo(word: str, rng: random.Random) -> tuple[str, str] | None:
    """Generate one plausible typo for `word`.

    Returns (typo, type_name) or None if word is too short / non-alphabetic.
    Uses weighted rule selection (matching the original Python, not the buggy JS port).
    """
    if len(word) < 3 or not re.match(r"^[A-Za-zÀ-ÿ''\-]+$", word):
        return None

    rules, weights = zip(*RULES)
    for _ in range(3):
        rule = rng.choices(rules, weights=weights, k=1)[0]
        idx = rules.index(rule)
        out = rule(word, rng)
        if out != word:
            return out, RULE_NAMES[idx]
    return None


def generate_typo_sentences(
    sentences: list[str],
    seed: int = 42,
    typos_per_sentence: int = 1,
) -> list[TypoCase]:
    """Generate synthetic typo sentences from a list of correct sentences.

    For each sentence, picks 1 word (length >= 4, alphabetic) and injects a
    single typo. Returns the original (correct) and erroneous versions.
    """
    rng = random.Random(seed)
    results: list[TypoCase] = []

    for sentence in sentences:
        words = sentence.split()
        eligible = []
        for i, raw in enumerate(words):
            core = re.sub(r"[^A-Za-zÀ-ÿ''\-]", "", raw)
            if len(core) >= 4 and re.match(r"^[A-Za-zÀ-ÿ''\-]+$", core):
                eligible.append((i, core, raw))

        if not eligible:
            continue

        rng.shuffle(eligible)
        chosen = eligible[: min(typos_per_sentence, len(eligible))]

        edits: list[TypoEdit] = []
        new_words = list(words)
        success = False

        for idx, core, raw in chosen:
            result = synth_typo(core, rng)
            if result is None:
                continue
            typo, typo_type = result

            # Preserve surrounding punctuation
            prefix_match = re.match(r"^[^A-Za-zÀ-ÿ''\-]*", raw)
            suffix_match = re.search(r"[^A-Za-zÀ-ÿ''\-]*$", raw)
            prefix = prefix_match.group(0) if prefix_match else ""
            suffix = suffix_match.group(0) if suffix_match else ""

            # Match case of original word
            if core[0].isupper() and core[0].lower() != core[0]:
                typo = typo[0].upper() + typo[1:]

            new_words[idx] = prefix + typo + suffix
            edits.append(TypoEdit(word=core, typo=typo, type=typo_type, position=idx))
            success = True

        if success:
            results.append(TypoCase(
                correct=sentence,
                erroneous=" ".join(new_words),
                edits=edits,
            ))

    return results
