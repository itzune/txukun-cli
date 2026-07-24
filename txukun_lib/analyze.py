"""
Analysis bridge — orchestrates all 3 detectors and produces a unified
list of Error objects.

Port of txukun's analyze.js. Runs the three detectors on plain text
(markdown stripped), merges overlapping errors, and maps offsets back
to original text positions.

  1. detect_grammar_errors  — GECToR correct → diff_words → replace changes
  2. detect_spelling_errors  — Hunspell → get_best_correction (Tier1+BERTeus)
  3. detect_cappunct_errors  — MarianMT correct → diff_words → case/punct-only
  4. merge_cap_punct         — merge cap-punct case changes into spelling/grammar
  5. dedupe_overlaps         — sort by position, remove overlaps
"""
from __future__ import annotations

from .errors import Error, next_id, reset_counter
from .markdown import strip_markdown, map_offset, build_context
from .diff import diff_words, is_case_punct_only, apply_case_pattern, is_in_heading

# ──────────────────────────────────────────────────────────────
# Confidence thresholds
#
# Per-model minimum confidence for a correction to be shown.
# Errors below the threshold are silently suppressed.
#
# These values were tuned on the 220-case evaluation dataset
# (tests/gec-benchmark/eval_dataset.json) via grid search
# (tests/gec-benchmark/confidence_per_model.py).
#
# Result: 22.7% → 38.6% accuracy (+15.9% absolute), cutting
# over-corrections from 139 → 66 and false positives from 12 → 1.
#
# Confidence source per model:
#   grammar  → GECToR P(INCORRECT) from detection head (0.0–1.0)
#   spelling → BERTeus cosine similarity, normalized to 0–1
#   cappunct → MarianMT LCS match rate (fraction of input
#              tokens that survived alignment; 1.0 = no word
#              substitution, only case/punctuation changes)
#
# ⚠️  REVIEW THESE VALUES if models are updated or retrained.
#     Re-run: tests/gec-benchmark/run_eval.py + confidence_per_model.py
# ──────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "grammar": 0.05,
    "spelling": 0.50,
    "cappunct": 1.00,
}


def _filter_by_confidence(errors: list[Error]) -> list[Error]:
    """Suppress errors whose confidence is below the per-category threshold.

    Errors with confidence=None are always kept (can't evaluate).
    """
    out = []
    for e in errors:
        threshold = CONFIDENCE_THRESHOLDS.get(e.category, 0.0)
        if e.confidence is None or e.confidence >= threshold:
            out.append(e)
    return out


def analyze_text(
    md_text: str,
    cappunct_model=None,
    spell_checker=None,
    grammar_model=None,
) -> list[Error]:
    """Analyze text and return a list of Error objects.

    Each model is optional — if None or not ready, that detector is skipped.
    """
    if not md_text or not md_text.strip():
        return []

    reset_counter()

    plain_text, pos_map, heading_ranges = strip_markdown(md_text)
    if not plain_text.strip():
        return []

    # Run detectors sequentially
    grammar_errors = _detect_grammar(plain_text, grammar_model)
    spelling_errors = _detect_spelling(plain_text, spell_checker)
    cappunct_errors = _detect_cappunct(plain_text, cappunct_model, heading_ranges)

    # Merge cap-punct case changes into spelling/grammar corrections
    cappunct_errors = _merge_cap_punct(
        spelling_errors + grammar_errors, cappunct_errors
    )

    # Build context (in plain text, paragraph-bounded)
    all_plain = grammar_errors + spelling_errors + cappunct_errors
    for e in all_plain:
        e.context = build_context(plain_text, e.frm)

    # Map offsets from plain text → original
    all_errors = [
        Error(
            id=e.id,
            frm=map_offset(e.frm, pos_map, False),
            to=map_offset(e.to, pos_map, True),
            original=e.original,
            suggestion=e.suggestion,
            category=e.category,
            title=e.title,
            context=e.context,
            confidence=e.confidence,
        )
        for e in all_plain
    ]

    # Sort by position; longer spans first when tied
    all_errors.sort(key=lambda e: (e.frm, -(e.to - e.frm)))
    # Remove overlaps (keep earliest, then longest)
    all_errors = _dedupe_overlaps(all_errors)
    # Suppress low-confidence corrections (per-model thresholds)
    all_errors = _filter_by_confidence(all_errors)
    return all_errors


# ── Grammar (GECToR) ────────────────────────────────

def _detect_grammar(text: str, model) -> list[Error]:
    errors: list[Error] = []
    if model is None or not model.ready:
        if model is not None and not model.failed:
            model._load()
        if model is None or not model.ready:
            return errors

    try:
        corrected, changed = model.correct(text)
        if not changed:
            return errors

        # Get per-word P(INCORRECT) from detection head for confidence
        try:
            detections = model.detect(text)
        except Exception:
            detections = []

        for ch in diff_words(text, corrected):
            if ch.type != "replace":
                continue
            # Find detection confidence for this word
            conf = None
            for d in detections:
                if d["start"] == ch.from_offset and d["end"] == ch.to_offset:
                    conf = d["pIncorrect"]
                    break
            errors.append(Error(
                id=next_id(),
                frm=ch.from_offset,
                to=ch.to_offset,
                original=ch.from_text,
                suggestion=ch.to_text,
                category="grammar",
                title=_grammar_title(ch.from_text, ch.to_text),
                confidence=conf,
            ))
    except Exception as e:
        import sys
        print(f"[analyze] grammar detection failed: {e}", file=sys.stderr)
    return errors


def _grammar_title(original: str, suggestion: str) -> str:
    if len(suggestion.split()) > len(original.split()):
        return "Hitza gehitu"
    if len(suggestion.split()) < len(original.split()):
        return "Hitza kendu"
    if suggestion.lower() == original.lower():
        return "Maiuskula"
    return "Gramatika"


# ── Spelling (Hunspell + BERTeus) ───────────────────

def _detect_spelling(text: str, checker) -> list[Error]:
    errors: list[Error] = []
    if checker is None or not checker.ready:
        return errors

    try:
        spell_errors = checker.check_spelling(text)
        for err in spell_errors:
            if not err.suggestions:
                continue
            best = checker.get_best_correction(text, err)
            if not best:
                continue
            if best.word == err.word:
                continue
            # BERTeus cosine similarity as confidence (normalize to 0–1)
            conf = None
            if best.bert_score is not None:
                conf = (best.bert_score + 1.0) / 2.0
            errors.append(Error(
                id=next_id(),
                frm=err.start,
                to=err.end,
                original=err.word,
                suggestion=best.word,
                category="spelling",
                title="Ortografia",
                confidence=conf,
            ))
    except Exception as e:
        import sys
        print(f"[analyze] spelling detection failed: {e}", file=sys.stderr)
    return errors


# ── Cap-punct (MarianMT) ────────────────────────────

def _detect_cappunct(text: str, model, heading_ranges: list[tuple[int, int]]) -> list[Error]:
    errors: list[Error] = []
    if model is None or not model.ready:
        if model is not None and not model.failed:
            model._load()
        if model is None or not model.ready:
            return errors

    try:
        corrected, match_rate = model.correct(text)
        if not corrected or corrected == text:
            return errors

        for ch in diff_words(text, corrected):
            if ch.type != "replace":
                continue
            if not is_case_punct_only(ch.from_text, ch.to_text):
                continue
            # Suppress punctuation hints inside headings
            if (is_in_heading(ch.from_offset, heading_ranges) and
                    ch.from_text.lower() != ch.to_text.lower()):
                continue
            errors.append(Error(
                id=next_id(),
                frm=ch.from_offset,
                to=ch.to_offset,
                original=ch.from_text,
                suggestion=ch.to_text,
                category="cappunct",
                title=_cappunct_title(ch.from_text, ch.to_text),
                confidence=match_rate,
            ))
    except Exception as e:
        import sys
        print(f"[analyze] cap-punct detection failed: {e}", file=sys.stderr)
    return errors


def _cappunct_title(from_text: str, to_text: str) -> str:
    if from_text.lower() == to_text.lower():
        return "Maiuskula"
    return "Puntuazioa"


# ── Merge cap-punct into spelling/grammar ───────────

def _merge_cap_punct(target_errors: list[Error], cappunct_errors: list[Error]) -> list[Error]:
    """Merge cap-punct case changes into spelling/grammar corrections.

    When a correction and a cap-punct case change overlap, apply the
    capitalization to the correction suggestion and drop the redundant
    cap-punct hint.
      spelling: laister→laster, cap-punct: laister→Laister → laister→Laster
      grammar:  ama→amak,      cap-punct: ama→Ama         → ama→Amak
    """
    removed: set[int] = set()
    for tg in target_errors:
        for i, cp in enumerate(cappunct_errors):
            if i in removed:
                continue
            if tg.frm >= cp.to or cp.frm >= tg.to:
                continue
            # Only merge pure case changes
            if cp.original.lower() != cp.suggestion.lower():
                continue
            merged = apply_case_pattern(cp.original, cp.suggestion, tg.suggestion)
            if merged and merged != tg.suggestion:
                tg.suggestion = merged
                removed.add(i)
    return [cp for i, cp in enumerate(cappunct_errors) if i not in removed]


# ── Overlap resolution ──────────────────────────────

def _dedupe_overlaps(errors: list[Error]) -> list[Error]:
    out: list[Error] = []
    last_end = -1
    for e in errors:
        if e.frm < last_end:
            continue
        out.append(e)
        last_end = e.to
    return out
