#!/usr/bin/env python3
"""
Build a curated evaluation dataset for txukun / txukun-cli.

All sentences are REAL Basque text from reputable sources:
  - Wikipedia (eu.wikipedia.org/wiki/Euskara) — encyclopedic prose
  - Elhuyar GEC dataset (Beloki et al., SEPLN 2020) — news sentences
    with real grammatical errors from native speakers

No sentences are invented. Errors are introduced in a controlled way:
  - grammar:     real error→correct pairs from Elhuyar (tests GECToR)
  - cappunct:    lowercase + strip punctuation (tests MarianMT cap-punct model)
  - spelling:    inject a single typo via typo_gen (tests Hunspell + BERTeus)
  - mixed:       combine multiple error types in one sentence
  - clean:       no errors (tests false-positive rate)
  - realword:    swap a valid word for its confusable partner (semantic error)
  - multi_error: 2-3 typos in one sentence (tests batch correction)
  - paragraph:   multi-sentence paragraphs with errors (tests sentence splitting)
  - markdown:    markdown-formatted text with errors (tests stripMarkdown)
  - missing_extra: duplicated or deleted word (tests structural error detection)

Output: tests/gec-benchmark/eval_dataset.json

Usage:
  python build_eval_dataset.py [--seed 42] [--n-grammar 40] [--n-cappunct 30]
         [--n-spelling 30] [--n-mixed 20] [--n-clean 20] [--n-realword 20]
         [--n-multi-error 15] [--n-paragraph 15] [--n-markdown 15]
         [--n-missing-extra 15]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

# Import typo generator from same directory
sys.path.insert(0, str(Path(__file__).parent))
from typo_gen import synth_typo, TypoEdit

# ──────────────────────────────────────────────────────────────
# Real Basque sentences from Wikipedia (eu.wikipedia.org/wiki/Euskara)
# These are verbatim extracts from the article, chosen for cap-punct
# richness: proper nouns, dates, acronyms, complex punctuation.
# ──────────────────────────────────────────────────────────────

WIKIPEDIA_SENTENCES: list[str] = [
    "Euskara Euskal Herriko hizkuntza da.",
    "Hizkuntza bakartua da, ez baitzaio munduko hizkuntzen artean ahaidetasunik aurkitu.",
    "Euskaraz mintzo direnei euskaldun deritze.",
    "Gaur egun, Euskal Herrian bertan ere hizkuntza gutxitua da, lurralde horretan gaztelania eta frantsesa nagusitu baitira.",
    "Morfologiari dagokionez, hizkuntza ergatibokoa eta eranskaria da, indoeuropar hizkuntzak ez bezala.",
    "Aitzineuskararen sustrairik zaharrenak, Erromatar Inperioaren aurretikoak, Akitania osoan eta Bizkaiko golkotik Andorraraino bitarteko Pirinioen bi aldeetan agertu dira.",
    "Erdi Aroan, gutxienez Errioxan eta Burgosko ipar-ekialdean ere hitz egin zela dokumentatuta dago.",
    "Frankismoaren garaian (1936–1977) euskarak jazarpen arrotza pairatu zuen eta ia desagertu zen, baina berriro ere indartu egin da 1960ko hamarkadaz geroztik.",
    "Gernikako Estatutuak euskara Euskal Autonomia Erkidegoko berezko hizkuntza izendatu zuen eta gaztelaniarekin batera ofizial egin zuen Araba, Bizkai eta Gipuzkoan.",
    "Nafarroa Garaian, Euskararen Legearen eraginez, herrialdeko ipar-mendebaldean soilik da koofiziala.",
    "Ipar Euskal Herrian, euskarak ez du aginpidea duten erakunde publikoen onarpenik.",
    "2016ko VI. Inkesta Soziolinguistikoaren arabera, euskararen eremu osoan bizi ziren 16 urtetik gorako biztanleen % 28,4 euskalduna zen.",
    "1545ean argitaratu zen lehen liburua euskaraz, «Linguae Vasconum Primitiae» izenekoa.",
    "Olerki liburua da eta haren egilea, Bernart Etxepare, Nafarroa Behereko herri txiki bateko apaiza izan zen.",
    "1729an, Manuel Larramendi jesuitak El Imposible Vencido izeneko euskararen gramatika bat argitaratu zuen Salamancan.",
    "1968an Euskaltzaindiak Arantzazuko santutegian (Oñati, Gipuzkoa) egin zuen batzarrean abiatu zen batasun prozesutik sortu zen euskara batua.",
    "Euskarak inguruko hizkuntzekin milaka urteko harremana izan du.",
    "Latindar alfabetoaren bidez idazten da.",
    "Euskararen dialektoei euskalkiak deritze.",
    "Euskara Araba, Bizkaia, Gipuzkoa, Lapurdi, Nafarroa eta Zuberoa herrialdeetan hitz egiten da.",
]

# ──────────────────────────────────────────────────────────────
# Elhuyar error type descriptions
# ──────────────────────────────────────────────────────────────

ELHUYAR_ERROR_DESC = {
    "R1": "R1: postposed finite verb → non-finite (tense/mood)",
    "R2": "R2: auxiliary verb disagreement (person/number)",
    "R3": "R3: case suffix error (ergative/absolutive/dative)",
    "R4": "R4: subordination/connector morphology error",
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def load_elhuyar_tsv(path: Path) -> list[dict]:
    """Load an Elhuyar evaluation TSV file.
    Returns list of {original, erroneous, error_types}.
    """
    cases = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            orig = (row.get("ORIGINAL SENTENCE") or "").strip()
            err = (row.get("SENTENCE WITH ERRORS") or "").strip()
            etypes = (row.get("ERROR TYPES") or "").strip()
            if orig and err:
                cases.append({
                    "original": orig,
                    "erroneous": err,
                    "error_types": etypes,
                })
    return cases


def strip_caps_punct(sentence: str) -> str:
    """Degrade a sentence for cap-punct testing:
    - Lowercase everything
    - Remove all punctuation (.,;:?!()«»""'')
    - Collapse multiple spaces
    """
    # Remove punctuation
    s = re.sub(r"[.,;:?!()\u00ab\u00bb\u201c\u201d\u2018\u2019\-\u2013\u2014]", " ", sentence)
    # Remove % signs attached to numbers but keep the number
    s = s.replace("%", "")
    # Lowercase
    s = s.lower()
    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_trailing_punct(sentence: str) -> str:
    """Only remove trailing punctuation + lowercase sentence start.
    A milder cap-punct degradation.
    """
    s = sentence.rstrip()
    # Remove trailing punctuation
    s = re.sub(r"[.?!,:]+$", "", s)
    # Lowercase first letter
    if s:
        s = s[0].lower() + s[1:]
    return s


def inject_typo(sentence: str, rng: random.Random) -> tuple[str, TypoEdit | None]:
    """Inject a single typo into a sentence. Returns (modified, edit)."""
    words = sentence.split()
    eligible = []
    for i, raw in enumerate(words):
        core = re.sub(r"[^A-Za-zÀ-ÿ''\-]", "", raw)
        if len(core) >= 4 and re.match(r"^[A-Za-zÀ-ÿ''\-]+$", core):
            eligible.append((i, core, raw))

    if not eligible:
        return sentence, None

    rng.shuffle(eligible)
    for idx, core, raw in eligible:
        result = synth_typo(core, rng)
        if result is None:
            continue
        typo, typo_type = result

        # Preserve surrounding punctuation
        prefix_match = re.match(r"^[^A-Za-zÀ-ÿ''\-]*", raw)
        suffix_match = re.search(r"[^A-Za-zÀ-ÿ''\-]*$", raw)
        prefix = prefix_match.group(0) if prefix_match else ""
        suffix = suffix_match.group(0) if suffix_match else ""

        # Match case
        if core[0].isupper():
            typo = typo[0].upper() + typo[1:]

        words[idx] = prefix + typo + suffix
        edit = TypoEdit(word=core, typo=typo, type=typo_type, position=idx)
        return " ".join(words), edit

    return sentence, None


def find_word_diff(s1: str, s2: str) -> tuple[str, str] | None:
    """Find the first word that differs between two space-tokenized strings.
    Returns (word_in_s1, word_in_s2) or None.
    """
    w1 = s1.split()
    w2 = s2.split()
    for a, b in zip(w1, w2):
        if a != b:
            return a, b
    if len(w1) != len(w2):
        if len(w1) > len(w2):
            return w1[len(w2)], "<missing>"
        return "<missing>", w2[len(w1)]
    return None


# ──────────────────────────────────────────────────────────────
# Dataset builders
# ──────────────────────────────────────────────────────────────

def build_grammar_cases(
    elhuyar_dir: Path,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample real grammar error→correct pairs from Elhuyar.
    Prefers manually-verified (Dem) over auto-generated (Dea).
    Covers R1-R4 error types.
    """
    # Load manually verified first
    dem_single = load_elhuyar_tsv(elhuyar_dir / "Dem_single.tsv")
    dem_multi = load_elhuyar_tsv(elhuyar_dir / "Dem_multi.tsv")
    dea_single = load_elhuyar_tsv(elhuyar_dir / "Dea_single.tsv")
    dea_multi = load_elhuyar_tsv(elhuyar_dir / "Dea_multi.tsv")

    # Filter: only cases where original != erroneous (real errors)
    def real_errors(cases):
        return [c for c in cases if c["original"] != c["erroneous"]]

    dem_all = real_errors(dem_single + dem_multi)
    dea_all = real_errors(dea_single + dea_multi)

    # Try to balance R1-R4 from Dem first, then fill from Dea
    by_type_dem: dict[str, list[dict]] = {"R1": [], "R2": [], "R3": [], "R4": []}
    for c in dem_all:
        et = c["error_types"].strip()
        if et in by_type_dem:
            by_type_dem[et].append(c)

    by_type_dea: dict[str, list[dict]] = {"R1": [], "R2": [], "R3": [], "R4": []}
    for c in dea_all:
        et = c["error_types"].strip()
        if et in by_type_dea:
            by_type_dea[et].append(c)

    # Shuffle each bucket
    for d in (by_type_dem, by_type_dea):
        for k in d:
            rng.shuffle(d[k])

    # Target: distribute n across R1-R4 (R2 and R4 are most common)
    targets = {"R1": max(2, n // 6), "R2": n // 3, "R3": max(2, n // 6), "R4": n // 3}
    # Adjust to sum to n
    total = sum(targets.values())
    targets["R2"] += n - total

    selected = []
    for rtype in ["R1", "R2", "R3", "R4"]:
        quota = targets[rtype]
        # Take from Dem first
        pool = by_type_dem[rtype][:quota]
        selected.append((pool, rtype, "elhuyar-dem"))
        remaining = quota - len(pool)
        if remaining > 0:
            pool2 = by_type_dea[rtype][:remaining]
            selected.append((pool2, rtype, "elhuyar-dea"))

    # Flatten and limit
    cases_out = []
    for pool, rtype, source in selected:
        for c in pool:
            if len(cases_out) >= n:
                break
            diff = find_word_diff(c["original"], c["erroneous"])
            if diff is None:
                continue  # skip if we can't identify the changed word
            orig_word, err_word = diff
            cases_out.append({
                "id": f"grammar_{len(cases_out)+1:03d}",
                "category": "grammar",
                "source": source,
                "input": c["erroneous"],
                "expected": c["original"],
                "errors": [{
                    "type": "grammar",
                    "original": err_word,
                    "correction": orig_word,
                    "description": ELHUYAR_ERROR_DESC.get(rtype, rtype),
                }],
                "metadata": {
                    "error_types": rtype,
                    "n_errors": 1 if rtype in ("R1", "R2", "R3", "R4") else 0,
                    "elhuyar_error_types": c["error_types"],
                },
            })
        if len(cases_out) >= n:
            break

    return cases_out


def build_cappunct_cases(
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Take real correct sentences, strip caps + punctuation.
    Sources: Wikipedia (prioritized — richer proper nouns) + Elhuyar clean.
    """
    # Wikipedia sentences first (they have proper nouns, dates, acronyms)
    wiki_sentences = list(WIKIPEDIA_SENTENCES)
    rng.shuffle(wiki_sentences)

    # Then Elhuyar clean sentences for variety
    elhuyar_dir = Path(__file__).parent / "elhuyar"
    elh_sentences = []
    if (elhuyar_dir / "Dem_none.tsv").exists():
        dem_none = load_elhuyar_tsv(elhuyar_dir / "Dem_none.tsv")
        elh_sentences.extend([c["original"] for c in dem_none])
    if (elhuyar_dir / "Dea_none.tsv").exists():
        dea_none = load_elhuyar_tsv(elhuyar_dir / "Dea_none.tsv")
        elh_sentences.extend([c["original"] for c in dea_none[:200]])
    rng.shuffle(elh_sentences)

    # Interleave: Wikipedia first, then Elhuyar to fill remaining slots
    all_sentences = wiki_sentences + elh_sentences

    cases = []
    for sent in all_sentences:
        if len(cases) >= n:
            break
        # Skip sentences that are too short or have no caps/punct to remove
        if len(sent) < 20:
            continue
        degraded = strip_caps_punct(sent)
        if degraded == sent:
            continue  # nothing was stripped

        # Identify what was changed (for the errors array)
        errors = []
        # Capitalization: find words that were capitalized in original
        orig_words = sent.split()
        deg_words = degraded.split()
        cap_changes = []
        for ow, dw in zip(orig_words, deg_words):
            # Check if the word was capitalized (first letter upper) in original
            # but not in degraded
            if ow and ow[0].isupper() and dw and dw[0].islower():
                cap_changes.append((dw, ow))

        # Punctuation: check if trailing punct was removed
        has_trailing_punct = bool(re.search(r"[.?!]$", sent))

        desc_parts = []
        if cap_changes:
            desc_parts.append(f"Capitalize {len(cap_changes)} word(s)")
        if has_trailing_punct:
            desc_parts.append("Add trailing punctuation")

        errors.append({
            "type": "cappunct",
            "original": degraded,
            "correction": sent,
            "description": "; ".join(desc_parts),
        })

        is_wiki = sent in WIKIPEDIA_SENTENCES
        cases.append({
            "id": f"cappunct_{len(cases)+1:03d}",
            "category": "cappunct",
            "source": "wikipedia" if is_wiki else "elhuyar",
            "input": degraded,
            "expected": sent,
            "errors": errors,
            "metadata": {
                "n_capitalization": len(cap_changes),
                "has_trailing_punct": has_trailing_punct,
                "n_errors": len(errors),
            },
        })

    return cases


def build_spelling_cases(
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Take real correct sentences, inject a single controlled typo.
    Sources: Wikipedia (prioritized) + Elhuyar clean sentences.
    """
    wiki_sentences = list(WIKIPEDIA_SENTENCES)
    rng.shuffle(wiki_sentences)

    elhuyar_dir = Path(__file__).parent / "elhuyar"
    elh_sentences = []
    if (elhuyar_dir / "Dem_none.tsv").exists():
        dem_none = load_elhuyar_tsv(elhuyar_dir / "Dem_none.tsv")
        elh_sentences.extend([c["original"] for c in dem_none])
    if (elhuyar_dir / "Dea_none.tsv").exists():
        dea_none = load_elhuyar_tsv(elhuyar_dir / "Dea_none.tsv")
        elh_sentences.extend([c["original"] for c in dea_none[:200]])
    rng.shuffle(elh_sentences)

    all_sentences = wiki_sentences + elh_sentences

    cases = []
    for sent in all_sentences:
        if len(cases) >= n:
            break
        if len(sent) < 20:
            continue

        modified, edit = inject_typo(sent, rng)
        if edit is None or modified == sent:
            continue

        cases.append({
            "id": f"spelling_{len(cases)+1:03d}",
            "category": "spelling",
            "source": "wikipedia" if sent in WIKIPEDIA_SENTENCES else "elhuyar",
            "input": modified,
            "expected": sent,
            "errors": [{
                "type": "spelling",
                "original": edit.typo,
                "correction": edit.word,
                "description": f"Typo type: {edit.type}",
            }],
            "metadata": {
                "typo_type": edit.type,
                "word_position": edit.position,
                "n_errors": 1,
            },
        })

    return cases


def build_mixed_cases(
    elhuyar_dir: Path,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Combine multiple error types:
    - mixed_cappunct_spelling: strip caps/punct + inject typo
    - mixed_cappunct_grammar: Elhuyar error sentence + strip caps/punct
    """
    cases = []
    half = n // 2

    # ── Type 1: cappunct + spelling ──
    wiki_sentences = list(WIKIPEDIA_SENTENCES)
    rng.shuffle(wiki_sentences)
    elh_sentences = []
    if (elhuyar_dir / "Dem_none.tsv").exists():
        dem_none = load_elhuyar_tsv(elhuyar_dir / "Dem_none.tsv")
        elh_sentences.extend([c["original"] for c in dem_none])
    rng.shuffle(elh_sentences)

    all_sentences = wiki_sentences + elh_sentences
    for sent in all_sentences:
        if len(cases) >= half:
            break
        if len(sent) < 20:
            continue

        # First strip caps/punct, then inject typo
        degraded = strip_caps_punct(sent)
        if degraded == sent:
            continue

        modified, edit = inject_typo(degraded, rng)
        if edit is None or modified == degraded:
            continue

        cases.append({
            "id": f"mixed_{len(cases)+1:03d}",
            "category": "mixed",
            "source": "wikipedia+typo" if sent in WIKIPEDIA_SENTENCES else "elhuyar+typo",
            "input": modified,
            "expected": sent,
            "errors": [
                {
                    "type": "spelling",
                    "original": edit.typo,
                    "correction": edit.word,
                    "description": f"Typo type: {edit.type}",
                },
                {
                    "type": "cappunct",
                    "original": degraded,
                    "correction": sent,
                    "description": "Capitalize + add punctuation",
                },
            ],
            "metadata": {
                "sub_type": "cappunct+spelling",
                "typo_type": edit.type,
                "n_errors": 2,
            },
        })

    # ── Type 2: cappunct + grammar (Elhuyar error + strip caps/punct) ──
    dem_single = load_elhuyar_tsv(elhuyar_dir / "Dem_single.tsv")
    dea_single = load_elhuyar_tsv(elhuyar_dir / "Dea_single.tsv")
    grammar_pool = [c for c in dem_single + dea_single if c["original"] != c["erroneous"]]
    rng.shuffle(grammar_pool)

    for c in grammar_pool:
        if len(cases) >= n:
            break
        orig = c["original"]
        err = c["erroneous"]
        if len(orig) < 20:
            continue

        # Strip caps/punct from the ERROR version
        degraded_err = strip_caps_punct(err)
        if degraded_err == err:
            continue

        diff = find_word_diff(orig, err)
        if diff is None:
            continue
        orig_word, err_word = diff
        rtype = c["error_types"].strip()

        cases.append({
            "id": f"mixed_{len(cases)+1:03d}",
            "category": "mixed",
            "source": "elhuyar+cappunct",
            "input": degraded_err,
            "expected": orig,
            "errors": [
                {
                    "type": "grammar",
                    "original": err_word,
                    "correction": orig_word,
                    "description": ELHUYAR_ERROR_DESC.get(rtype, rtype),
                },
                {
                    "type": "cappunct",
                    "original": degraded_err,
                    "correction": err,
                    "description": "Capitalize + add punctuation",
                },
            ],
            "metadata": {
                "sub_type": "cappunct+grammar",
                "error_types": rtype,
                "n_errors": 2,
            },
        })

    return cases[:n]


def build_clean_cases(
    elhuyar_dir: Path,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sentences with NO errors — test false-positive rate.
    Sources: Elhuyar 'none' files + Wikipedia.
    """
    wiki_sentences = list(WIKIPEDIA_SENTENCES)
    rng.shuffle(wiki_sentences)

    elh_sentences = []
    if (elhuyar_dir / "Dem_none.tsv").exists():
        dem_none = load_elhuyar_tsv(elhuyar_dir / "Dem_none.tsv")
        elh_sentences.extend([c["original"] for c in dem_none])
    if (elhuyar_dir / "Dea_none.tsv").exists():
        dea_none = load_elhuyar_tsv(elhuyar_dir / "Dea_none.tsv")
        elh_sentences.extend([c["original"] for c in dea_none[:300]])
    rng.shuffle(elh_sentences)

    all_sentences = wiki_sentences + elh_sentences

    cases = []
    for sent in all_sentences:
        if len(cases) >= n:
            break
        if len(sent) < 15:
            continue

        cases.append({
            "id": f"clean_{len(cases)+1:03d}",
            "category": "clean",
            "source": "wikipedia" if sent in WIKIPEDIA_SENTENCES else "elhuyar",
            "input": sent,
            "expected": sent,
            "errors": [],
            "metadata": {
                "n_errors": 0,
            },
        })

    return cases


# ──────────────────────────────────────────────────────────────
# Real-word errors: valid word used in wrong context
# ──────────────────────────────────────────────────────────────

# Confusable word pairs where BOTH words are valid Basque dictionary
# entries (verified against the 159k-word frequency dictionary).
# The error is semantic/contextual, not orthographic.
CONFUSABLE_PAIRS: list[tuple[str, str, str]] = [
    # (correct, erroneous, description)
    ("hura",   "ura",   "demonstrative 'that one' → 'ura' (the water)"),
    ("hari",   "ari",   "dative 'to that one' → 'ari' (busy/engaged)"),
    ("hala",   "ala",   "'thus/so' → 'ala' (or/either in questions)"),
    ("bere",   "beren", "singular possessor 'his/her' → plural 'their'"),
    ("da",     "du",    "'is' (3sg intransitive) → 'du' (3sg transitive 'has')"),
    ("honi",   "oni",   "dative 'to this' → 'oni'"),
    ("honek",  "onek",  "ergative 'this' → 'onek'"),
    ("horrek", "orrek", "ergative 'that' → 'orrek'"),
    ("hark",   "ark",   "ergative 'that one' → 'ark'"),
    ("haiek",  "aiek",  "plural demonstrative 'those' → 'aiek'"),
    ("hitz",   "itz",   "'word' → 'itz'"),
    ("hots",   "ots",   "'sound/voice' → 'ots' (echo)"),
    ("hondar", "ondar", "'sand/remainder' → 'ondar'"),
    ("zoro",   "soro",  "'crazy' → 'soro'"),
    ("hur",    "ur",    "'hazel/nut' → 'ur' (water)"),
    ("haran",  "aran",  "'valley' → 'aran' (plum)"),
]


def build_realword_cases(
    elhuyar_dir: Path,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Real-word errors: swap a valid word for its confusable partner.

    Searches the real corpus (Wikipedia + Elhuyar) for sentences containing
    a target word, then replaces it with its confusable partner. Both words
    are valid dictionary entries — the error is semantic/contextual.
    """
    # Load all sentences from corpus
    all_sentences = list(WIKIPEDIA_SENTENCES)
    for fname in ["Dem_none.tsv", "Dea_none.tsv", "Dem_single.tsv",
                  "Dea_single.tsv", "Dem_multi.tsv", "Dea_multi.tsv"]:
        p = elhuyar_dir / fname
        if p.exists():
            for c in load_elhuyar_tsv(p):
                if c["original"]:
                    all_sentences.append(c["original"])

    cases = []
    # Track which pairs we've used to ensure variety
    pair_usage: dict[tuple[str, str], int] = {}

    # Build a list of (sentence, correct_word, erroneous_word, description, source)
    candidates = []
    for sent in all_sentences:
        if len(sent) < 20:
            continue
        is_wiki = sent in WIKIPEDIA_SENTENCES
        for correct, erroneous, desc in CONFUSABLE_PAIRS:
            # Word-boundary search for the correct word
            pattern = r"\b" + re.escape(correct) + r"\b"
            if re.search(pattern, sent, re.IGNORECASE):
                candidates.append((sent, correct, erroneous, desc, is_wiki))

    rng.shuffle(candidates)

    for sent, correct, erroneous, desc, is_wiki in candidates:
        if len(cases) >= n:
            break
        # Limit usage per pair to ensure variety (max n/len(pairs) + 2)
        pair_key = (correct, erroneous)
        max_per_pair = max(2, n // len(CONFUSABLE_PAIRS) + 1)
        if pair_usage.get(pair_key, 0) >= max_per_pair:
            continue

        # Replace the first occurrence (case-sensitive: match the case)
        pattern = r"\b" + re.escape(correct) + r"\b"
        match = re.search(pattern, sent)
        if not match:
            continue

        matched_text = match.group(0)
        # Preserve case
        if matched_text[0].isupper():
            replacement = erroneous[0].upper() + erroneous[1:]
        else:
            replacement = erroneous

        modified = sent[:match.start()] + replacement + sent[match.end():]

        pair_usage[pair_key] = pair_usage.get(pair_key, 0) + 1

        cases.append({
            "id": f"realword_{len(cases)+1:03d}",
            "category": "realword",
            "source": "wikipedia" if is_wiki else "elhuyar",
            "input": modified,
            "expected": sent,
            "errors": [{
                "type": "realword",
                "original": replacement,
                "correction": matched_text,
                "description": f"Real-word confusion: {desc}",
            }],
            "metadata": {
                "confusable_pair": f"{correct}↔{erroneous}",
                "n_errors": 1,
            },
        })

    return cases


# ──────────────────────────────────────────────────────────────
# Multiple errors of the same type in one sentence
# ──────────────────────────────────────────────────────────────

def build_multi_error_cases(
    elhuyar_dir: Path,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Inject 2-3 typos into a single real sentence.
    Tests the corrector's ability to handle multiple spelling errors at once.
    """
    wiki_sentences = list(WIKIPEDIA_SENTENCES)
    rng.shuffle(wiki_sentences)

    elh_sentences = []
    if (elhuyar_dir / "Dem_none.tsv").exists():
        elh_sentences.extend([c["original"] for c in load_elhuyar_tsv(elhuyar_dir / "Dem_none.tsv")])
    if (elhuyar_dir / "Dea_none.tsv").exists():
        elh_sentences.extend([c["original"] for c in load_elhuyar_tsv(elhuyar_dir / "Dea_none.tsv")[:200]])
    rng.shuffle(elh_sentences)

    all_sentences = wiki_sentences + elh_sentences

    cases = []
    for sent in all_sentences:
        if len(cases) >= n:
            break
        if len(sent) < 30:  # need longer sentences for multiple typos
            continue

        # Inject 2-3 typos sequentially
        n_typos = rng.choice([2, 2, 3])  # bias toward 2
        modified = sent
        edits: list[TypoEdit] = []

        for _ in range(n_typos):
            modified, edit = inject_typo(modified, rng)
            if edit is not None:
                edits.append(edit)

        if len(edits) < 2:
            continue  # need at least 2 successful typos

        is_wiki = sent in WIKIPEDIA_SENTENCES
        cases.append({
            "id": f"multi_error_{len(cases)+1:03d}",
            "category": "multi_error",
            "source": "wikipedia" if is_wiki else "elhuyar",
            "input": modified,
            "expected": sent,
            "errors": [{
                "type": "spelling",
                "original": e.typo,
                "correction": e.word,
                "description": f"Typo type: {e.type}",
            } for e in edits],
            "metadata": {
                "n_errors": len(edits),
                "typo_types": [e.type for e in edits],
            },
        })

    return cases


# ──────────────────────────────────────────────────────────────
# Paragraph-level / multi-sentence
# ──────────────────────────────────────────────────────────────

def build_paragraph_cases(
    elhuyar_dir: Path,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Multi-sentence paragraphs with 1-2 errors.
    Tests sentence splitting and cross-sentence processing.
    Uses consecutive Wikipedia sentences + Elhuyar sentences.
    """
    cases = []

    # Type 1: Consecutive Wikipedia sentences (known order)
    wiki = list(WIKIPEDIA_SENTENCES)
    half = n // 2

    for i in range(0, len(wiki) - 2, 2):
        if len(cases) >= half:
            break
        # Take 2-3 consecutive sentences
        n_sents = rng.choice([2, 3])
        if i + n_sents > len(wiki):
            continue
        paragraph_sents = wiki[i:i + n_sents]
        paragraph = " ".join(paragraph_sents)

        if len(paragraph) < 40:
            continue

        # Inject 1-2 errors (typo or cap-punct strip)
        if rng.random() < 0.5:
            # Inject a typo
            modified, edit = inject_typo(paragraph, rng)
            if edit is None or modified == paragraph:
                continue
            errors = [{
                "type": "spelling",
                "original": edit.typo,
                "correction": edit.word,
                "description": f"Typo type: {edit.type}",
            }]
        else:
            # Strip caps/punct from the whole paragraph
            modified = strip_caps_punct(paragraph)
            if modified == paragraph:
                continue
            errors = [{
                "type": "cappunct",
                "original": modified,
                "correction": paragraph,
                "description": "Capitalize + add punctuation (paragraph-level)",
            }]

        cases.append({
            "id": f"paragraph_{len(cases)+1:03d}",
            "category": "paragraph",
            "source": "wikipedia",
            "input": modified,
            "expected": paragraph,
            "errors": errors,
            "metadata": {
                "n_sentences": n_sents,
                "n_errors": len(errors),
            },
        })

    # Type 2: Elhuyar multi-error sentences (already paragraph-like, 20+ words)
    dem_multi = load_elhuyar_tsv(elhuyar_dir / "Dem_multi.tsv")
    dea_multi = load_elhuyar_tsv(elhuyar_dir / "Dea_multi.tsv")
    multi_pool = [c for c in dem_multi + dea_multi if c["original"] != c["erroneous"]]
    rng.shuffle(multi_pool)

    for c in multi_pool:
        if len(cases) >= n:
            break
        orig = c["original"]
        err = c["erroneous"]
        if len(orig) < 40:
            continue

        diff = find_word_diff(orig, err)
        if diff is None:
            continue
        orig_word, err_word = diff
        rtype = c["error_types"].strip()

        cases.append({
            "id": f"paragraph_{len(cases)+1:03d}",
            "category": "paragraph",
            "source": "elhuyar-multi",
            "input": err,
            "expected": orig,
            "errors": [{
                "type": "grammar",
                "original": err_word,
                "correction": orig_word,
                "description": ELHUYAR_ERROR_DESC.get(rtype, rtype),
            }],
            "metadata": {
                "n_sentences": 1,  # technically 1 long sentence
                "error_types": rtype,
                "n_errors": 1,
            },
        })

    return cases[:n]


# ──────────────────────────────────────────────────────────────
# Markdown-formatted text
# ──────────────────────────────────────────────────────────────

# Markdown templates: (format_function, description)
MARKDOWN_TEMPLATES = [
    # Heading + paragraph
    (lambda s, h: f"# {h}\n\n{s}", "heading"),
    # Bold inline
    (lambda s, h: f"**{s[:s.find(' ') + 1]}**{s[s.find(' ') + 1:]}", "bold-first-word"),
    # Bullet list
    (lambda s, h: f"- {s}", "bullet"),
    # Numbered list
    (lambda s, h: f"1. {s}", "numbered"),
    # Blockquote
    (lambda s, h: f"> {s}", "blockquote"),
    # Link wrapping (simulated)
    (lambda s, h: f"[{s}]", "link-wrapped"),
]


def build_markdown_cases(
    elhuyar_dir: Path,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Markdown-formatted text with errors.
    Tests stripMarkdown() + offset mapping + error detection through formatting.
    """
    wiki_sentences = list(WIKIPEDIA_SENTENCES)
    rng.shuffle(wiki_sentences)

    elh_sentences = []
    if (elhuyar_dir / "Dem_none.tsv").exists():
        elh_sentences.extend([c["original"] for c in load_elhuyar_tsv(elhuyar_dir / "Dem_none.tsv")])
    rng.shuffle(elh_sentences)

    all_sentences = wiki_sentences + elh_sentences

    cases = []
    for sent in all_sentences:
        if len(cases) >= n:
            break
        if len(sent) < 20:
            continue

        is_wiki = sent in WIKIPEDIA_SENTENCES

        # Pick a random markdown template
        template_fn, md_type = rng.choice(MARKDOWN_TEMPLATES)

        # Generate a heading from the first 3-4 words
        words = sent.split()
        heading = " ".join(words[:rng.choice([3, 4])])

        # First inject a typo into the plain text
        modified, edit = inject_typo(sent, rng)
        if edit is None or modified == sent:
            continue

        # Then wrap in markdown
        md_input = template_fn(modified, heading)
        md_expected = template_fn(sent, heading)

        cases.append({
            "id": f"markdown_{len(cases)+1:03d}",
            "category": "markdown",
            "source": "wikipedia" if is_wiki else "elhuyar",
            "input": md_input,
            "expected": md_expected,
            "errors": [{
                "type": "spelling",
                "original": edit.typo,
                "correction": edit.word,
                "description": f"Typo type: {edit.type} (in markdown {md_type})",
            }],
            "metadata": {
                "markdown_type": md_type,
                "typo_type": edit.type,
                "n_errors": 1,
            },
        })

    return cases


# ──────────────────────────────────────────────────────────────
# Missing / extra word errors
# ──────────────────────────────────────────────────────────────

def build_missing_extra_cases(
    elhuyar_dir: Path,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Missing or extra word errors.
    - Extra word: duplicate a word (common typing mistake)
    - Missing word: delete a function word (eta, da, ez, du, etc.)
    """
    wiki_sentences = list(WIKIPEDIA_SENTENCES)
    rng.shuffle(wiki_sentences)

    elh_sentences = []
    if (elhuyar_dir / "Dem_none.tsv").exists():
        elh_sentences.extend([c["original"] for c in load_elhuyar_tsv(elhuyar_dir / "Dem_none.tsv")])
    if (elhuyar_dir / "Dea_none.tsv").exists():
        elh_sentences.extend([c["original"] for c in load_elhuyar_tsv(elhuyar_dir / "Dea_none.tsv")[:200]])
    rng.shuffle(elh_sentences)

    all_sentences = wiki_sentences + elh_sentences

    # Function words whose deletion creates a grammatical error
    FUNCTION_WORDS = {"eta", "da", "ez", "du", "dira", "dute", "bat",
                      "baina", "edo", "ni", "zu", "gu", "hau", "hori",
                      "hura", "eta", "ere", "bakarrik", "orain", "gero"}

    cases = []
    for sent in all_sentences:
        if len(cases) >= n:
            break
        if len(sent) < 25:
            continue

        is_wiki = sent in WIKIPEDIA_SENTENCES
        words = sent.split()
        if len(words) < 6:
            continue

        if rng.random() < 0.5:
            # ── Extra word: duplicate a random word ──
            # Pick a word that's not punctuation-only
            eligible = [(i, w) for i, w in enumerate(words)
                       if len(re.sub(r"[^A-Za-zÀ-ÿ]", "", w)) >= 2]
            if not eligible:
                continue
            idx, word = rng.choice(eligible)
            # Insert duplicate right after the original
            new_words = words[:idx + 1] + [word] + words[idx + 1:]
            modified = " ".join(new_words)

            cases.append({
                "id": f"missing_extra_{len(cases)+1:03d}",
                "category": "missing_extra",
                "source": "wikipedia" if is_wiki else "elhuyar",
                "input": modified,
                "expected": sent,
                "errors": [{
                    "type": "extra_word",
                    "original": f"{word} {word}",
                    "correction": word,
                    "description": f"Duplicated word: '{word}'",
                }],
                "metadata": {
                    "sub_type": "extra_word",
                    "duplicated_word": word,
                    "position": idx,
                    "n_errors": 1,
                },
            })
        else:
            # ── Missing word: delete a function word ──
            # Find function words in the sentence
            func_indices = [(i, w) for i, w in enumerate(words)
                           if re.sub(r"[^A-Za-zÀ-ÿ]", "", w).lower() in FUNCTION_WORDS]
            if not func_indices:
                continue
            idx, word = rng.choice(func_indices)
            # Delete the word
            new_words = words[:idx] + words[idx + 1:]
            modified = " ".join(new_words)

            cases.append({
                "id": f"missing_extra_{len(cases)+1:03d}",
                "category": "missing_extra",
                "source": "wikipedia" if is_wiki else "elhuyar",
                "input": modified,
                "expected": sent,
                "errors": [{
                    "type": "missing_word",
                    "original": "(missing)",
                    "correction": word,
                    "description": f"Missing word: '{word}' was deleted",
                }],
                "metadata": {
                    "sub_type": "missing_word",
                    "deleted_word": word,
                    "position": idx,
                    "n_errors": 1,
                },
            })

    return cases


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build txukun evaluation dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-grammar", type=int, default=40)
    parser.add_argument("--n-cappunct", type=int, default=30)
    parser.add_argument("--n-spelling", type=int, default=30)
    parser.add_argument("--n-mixed", type=int, default=20)
    parser.add_argument("--n-clean", type=int, default=20)
    parser.add_argument("--n-realword", type=int, default=20)
    parser.add_argument("--n-multi-error", type=int, default=15)
    parser.add_argument("--n-paragraph", type=int, default=15)
    parser.add_argument("--n-markdown", type=int, default=15)
    parser.add_argument("--n-missing-extra", type=int, default=15)
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "eval_dataset.json"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    elhuyar_dir = Path(__file__).parent / "elhuyar"

    if not elhuyar_dir.exists():
        print(f"ERROR: Elhuyar dataset not found at {elhuyar_dir}", file=sys.stderr)
        sys.exit(1)

    print("Building evaluation dataset...")
    print(f"  Grammar:       {args.n_grammar} cases (Elhuyar real errors, R1-R4)")
    print(f"  Cap-punct:     {args.n_cappunct} cases (Wikipedia + Elhuyar, stripped)")
    print(f"  Spelling:      {args.n_spelling} cases (Wikipedia + Elhuyar, typo-injected)")
    print(f"  Mixed:         {args.n_mixed} cases (cap-punct + spelling/grammar)")
    print(f"  Clean:         {args.n_clean} cases (no errors, false-positive test)")
    print(f"  Real-word:     {args.n_realword} cases (confusable valid words swapped)")
    print(f"  Multi-error:   {args.n_multi_error} cases (2-3 typos in one sentence)")
    print(f"  Paragraph:     {args.n_paragraph} cases (multi-sentence, tests splitting)")
    print(f"  Markdown:      {args.n_markdown} cases (markdown-formatted text)")
    print(f"  Missing/Extra: {args.n_missing_extra} cases (duplicated/deleted words)")
    print()

    all_cases = []

    grammar_cases = build_grammar_cases(elhuyar_dir, args.n_grammar, rng)
    all_cases.extend(grammar_cases)
    print(f"  ✓ Grammar:   {len(grammar_cases)} cases")

    cappunct_cases = build_cappunct_cases(args.n_cappunct, rng)
    all_cases.extend(cappunct_cases)
    print(f"  ✓ Cap-punct: {len(cappunct_cases)} cases")

    spelling_cases = build_spelling_cases(args.n_spelling, rng)
    all_cases.extend(spelling_cases)
    print(f"  ✓ Spelling:  {len(spelling_cases)} cases")

    mixed_cases = build_mixed_cases(elhuyar_dir, args.n_mixed, rng)
    all_cases.extend(mixed_cases)
    print(f"  ✓ Mixed:     {len(mixed_cases)} cases")

    clean_cases = build_clean_cases(elhuyar_dir, args.n_clean, rng)
    all_cases.extend(clean_cases)
    print(f"  ✓ Clean:         {len(clean_cases)} cases")

    realword_cases = build_realword_cases(elhuyar_dir, args.n_realword, rng)
    all_cases.extend(realword_cases)
    print(f"  ✓ Real-word:     {len(realword_cases)} cases")

    multi_error_cases = build_multi_error_cases(elhuyar_dir, args.n_multi_error, rng)
    all_cases.extend(multi_error_cases)
    print(f"  ✓ Multi-error:   {len(multi_error_cases)} cases")

    paragraph_cases = build_paragraph_cases(elhuyar_dir, args.n_paragraph, rng)
    all_cases.extend(paragraph_cases)
    print(f"  ✓ Paragraph:     {len(paragraph_cases)} cases")

    markdown_cases = build_markdown_cases(elhuyar_dir, args.n_markdown, rng)
    all_cases.extend(markdown_cases)
    print(f"  ✓ Markdown:      {len(markdown_cases)} cases")

    missing_extra_cases = build_missing_extra_cases(elhuyar_dir, args.n_missing_extra, rng)
    all_cases.extend(missing_extra_cases)
    print(f"  ✓ Missing/Extra: {len(missing_extra_cases)} cases")

    print(f"\nTotal: {len(all_cases)} cases")

    # Source attribution
    sources = {}
    for c in all_cases:
        s = c["source"]
        sources[s] = sources.get(s, 0) + 1
    print("\nSources:")
    for s, count in sorted(sources.items()):
        print(f"  {s}: {count}")

    # Category breakdown
    categories = {}
    for c in all_cases:
        cat = c["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("\nCategories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Write output
    output_path = Path(args.output)
    dataset = {
        "description": "Curated evaluation dataset for txukun / txukun-cli",
        "version": "1.0",
        "license": "CC-BY-NC-SA 4.0 (Elhuyar GEC dataset license)",
        "sources": [
            "Wikipedia (eu.wikipedia.org/wiki/Euskara) — CC-BY-SA",
            "Elhuyar GEC dataset (Beloki et al., SEPLN 2020) — CC-BY-NC-SA 4.0",
        ],
        "categories": {
            "grammar": "Real grammatical errors from Elhuyar GEC dataset (R1-R4)",
            "cappunct": "Real sentences with caps/punctuation stripped",
            "spelling": "Real sentences with a single controlled typo injected",
            "mixed": "Multiple error types combined (cap-punct + spelling/grammar)",
            "clean": "No errors — tests false-positive rate",
            "realword": "Valid word swapped for its confusable partner (semantic error)",
            "multi_error": "2-3 typos in one sentence (tests batch correction)",
            "paragraph": "Multi-sentence paragraphs with errors (tests sentence splitting)",
            "markdown": "Markdown-formatted text with errors (tests stripMarkdown)",
            "missing_extra": "Duplicated or deleted word (tests structural error detection)",
        },
        "schema": {
            "id": "Unique identifier (category_NNN)",
            "category": "grammar | cappunct | spelling | mixed | clean | realword | multi_error | paragraph | markdown | missing_extra",
            "source": "Source attribution",
            "input": "Text with errors (what the corrector receives)",
            "expected": "Correct text (ground truth)",
            "errors": [
                {
                    "type": "grammar | spelling | cappunct",
                    "original": "The erroneous word/phrase",
                    "correction": "The correct word/phrase",
                    "description": "Human-readable error description",
                }
            ],
            "metadata": "Additional info (error types, typo type, etc.)",
        },
        "cases": all_cases,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Written to {output_path}")


if __name__ == "__main__":
    main()
