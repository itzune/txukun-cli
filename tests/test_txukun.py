"""Tests for txukun_lib — clean_output, WORD_RE, and HunspellChecker."""

import re
from pathlib import Path

import pytest

# Import from the new txukun_lib package
from txukun_lib.cappunct import clean_output
from txukun_lib.spelling import WORD_RE, HunspellChecker, SpellChecker, DICT_PATH


# ── clean_output ──────────────────────────────────────────

def test_clean_output_removes_special_tokens():
    assert clean_output("<s>Kaixo</s>") == "Kaixo"
    assert clean_output("<pad>test</pad>") == "test"
    assert clean_output("hello <unk> world") == "hello world"
    assert clean_output("no special tokens") == "no special tokens"


def test_clean_output_collapses_whitespace():
    assert clean_output("hello    world") == "hello world"
    assert clean_output("  leading spaces") == "leading spaces"
    assert clean_output("trailing   ") == "trailing"


def test_clean_output_handles_mixed():
    assert clean_output("<s>  Kaixo   mundua  </s>") == "Kaixo mundua"


# ── WORD_RE tokenizer ─────────────────────────────────────

def test_word_re_basque_words():
    matches = WORD_RE.findall("kaixo mundua")
    assert matches == ["kaixo", "mundua"]


def test_word_re_basque_special_chars():
    matches = WORD_RE.findall("ñabardura ütz egizu")
    assert "ñabardura" in matches
    assert "egizu" in matches


def test_word_re_apostrophes():
    matches = WORD_RE.findall("d'Artagnan l'herbe")
    assert "d'Artagnan" in matches
    assert "l'herbe" in matches


def test_word_re_hyphenated():
    matches = WORD_RE.findall("hitz-armak etxe-aurrean")
    assert "hitz-armak" in matches
    assert "etxe-aurrean" in matches


def test_word_re_numbers():
    matches = WORD_RE.findall("42 katu 3.14 zenbakia")
    assert "42" in matches
    assert "3.14" in matches


def test_word_re_urls():
    matches = WORD_RE.findall("ikus https://itzune.eus orria")
    assert "https://itzune.eus" in matches


def test_word_re_emails():
    matches = WORD_RE.findall("bidali hi@itzune.eus helbidera")
    assert "hi@itzune.eus" in matches


def test_word_re_skips_punctuation():
    matches = WORD_RE.findall("Kaixo, mundua!")
    assert "Kaixo" in matches
    assert "mundua" in matches
    assert "," not in matches
    assert "!" not in matches


# ── HunspellChecker ───────────────────────────────────────

@pytest.fixture
def hunspell():
    """Fixture: a loaded HunspellChecker shared across tests."""
    h = HunspellChecker()
    assert h.loaded, "Hunspell not available — install with: sudo apt install hunspell"
    yield h
    h._close()


def test_hunspell_loaded(hunspell):
    assert hunspell.loaded


def test_hunspell_correct_valid_common_words(hunspell):
    assert hunspell.correct("etxea") is True
    assert hunspell.correct("kaixo") is True
    assert hunspell.correct("mundua") is True
    assert hunspell.correct("egun") is True
    assert hunspell.correct("eskerrik") is True
    assert hunspell.correct("asko") is True


def test_hunspell_correct_valid_with_affixes(hunspell):
    """Affix rules should recognize declined/conjugated forms."""
    assert hunspell.correct("etxearekin") is True   # soziatiboa
    assert hunspell.correct("etxeetara") is True     # adlatiboa (plural)
    assert hunspell.correct("etxetik") is True       # ablatiboa


def test_hunspell_correct_compounds(hunspell):
    """Hunspell handles Basque compounds."""
    assert hunspell.correct("hitz-armak") is True


def test_hunspell_correct_misspelled(hunspell):
    assert hunspell.correct("ser") is False
    assert hunspell.correct("asdfg") is False
    assert hunspell.correct("egingozu") is False


def test_hunspell_correct_numbers_are_skipped(hunspell):
    """Numbers are skipped by correct_text, but correct() passes them through."""
    assert hunspell.correct("42") is True


def test_hunspell_correct_acronyms(hunspell):
    assert hunspell.correct("EITB") is True
    assert hunspell.correct("UPV") is True


def test_hunspell_suggest_returns_list(hunspell):
    suggestions = hunspell.suggest("ser")
    assert isinstance(suggestions, list)
    assert "zer" in suggestions


def test_hunspell_suggest_no_results_returns_empty(hunspell):
    suggestions = hunspell.suggest("asdfgzxcvb")
    assert suggestions == []


# ── Not loaded (no hunspell) ─────────────────────────────

def test_hunspell_checker_not_loaded_returns_true():
    """Without hunspell, correct() should return True (assume correct)."""
    nonexistent = Path("/tmp/nonexistent_dict_for_test")
    h = HunspellChecker(nonexistent)
    assert h.loaded is False
    assert h.correct("ser") is True
    assert h.suggest("ser") == []
