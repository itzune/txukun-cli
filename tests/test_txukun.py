"""Tests for txukun.py — unit tests for clean_output, WORD_RE, and SpellChecker."""

import re
from pathlib import Path

import pytest

# Import the module under test
from txukun import clean_output, WORD_RE, SpellChecker, DICT_PATH

DICT = Path(__file__).parent.parent / "data" / "eu"


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


# ── SpellChecker ──────────────────────────────────────────

@pytest.fixture
def spell():
    """Fixture: a loaded SpellChecker shared across tests."""
    s = SpellChecker(DICT)
    assert s.loaded, "Hunspell not available — install with: sudo apt install hunspell"
    yield s
    s._close()


def test_spell_loaded(spell):
    assert spell.loaded


def test_spell_correct_valid_common_words(spell):
    assert spell.correct("etxea") is True
    assert spell.correct("kaixo") is True
    assert spell.correct("mundua") is True
    assert spell.correct("egun") is True
    assert spell.correct("eskerrik") is True
    assert spell.correct("asko") is True


def test_spell_correct_valid_with_affixes(spell):
    """Affix rules should recognize declined/conjugated forms."""
    assert spell.correct("etxearekin") is True   # soziatiboa
    assert spell.correct("etxeetara") is True     # adlatiboa (plural)
    assert spell.correct("etxetik") is True       # ablatiboa


def test_spell_correct_compounds(spell):
    """Hunspell handles Basque compounds."""
    assert spell.correct("hitz-armak") is True


def test_spell_correct_misspelled(spell):
    assert spell.correct("ser") is False
    # asdfg is clearly not a word
    assert spell.correct("asdfg") is False
    # Common misspelling that should be caught
    assert spell.correct("funtzionamendua") is True  # "mendua" suffix pattern, valid form
    # Test an actual misspelling
    assert spell.correct("egingozu") is False  # should be "egingo duzu" or "egiozu"


def test_spell_correct_numbers_are_skipped(spell):
    """Numbers are skipped by correct_text, but correct() passes them through."""
    # Hunspell itself would mark numbers as correct
    assert spell.correct("42") is True


def test_spell_correct_acronyms(spell):
    assert spell.correct("EITB") is True
    assert spell.correct("UPV") is True


def test_spell_suggest_returns_list(spell):
    suggestions = spell.suggest("ser")
    assert isinstance(suggestions, list)
    assert "zer" in suggestions


def test_spell_suggest_no_results_returns_empty(spell):
    suggestions = spell.suggest("asdfgzxcvb")
    assert suggestions == []


def test_spell_correct_text_no_changes(spell):
    text = "kaixo mundua"
    result, changes = spell.correct_text(text)
    assert result == text
    assert changes == 0


def test_spell_correct_text_fixes_misspelling(spell):
    text = "etsea handia da"
    result, changes = spell.correct_text(text)
    # "etsea" is misspelled — should be replaced (etzea or etxea)
    assert "etsea" not in result
    assert changes > 0
    # The replacement should not be ALL-CAPS
    assert "ETSEA" not in result


def test_spell_correct_text_preserves_casing(spell):
    """Title-case misspellings should get title-case replacements."""
    text = "Etsea handia da"
    result, changes = spell.correct_text(text)
    # "Etsea" is misspelled — should be replaced with title-case (Etzea or Etxea)
    assert changes > 0
    words = result.split()
    replacement = words[0]
    assert replacement[0].isupper(), f"Expected title-case replacement, got {replacement!r}"
    assert replacement != "ETSEA", "ALL-CAPS should be filtered"


def test_spell_correct_text_skips_all_caps(spell):
    text = "EITB da onena"
    result, changes = spell.correct_text(text)
    # EITB should be left alone
    assert "EITB" in result
    assert changes == 0


def test_spell_correct_text_skips_numbers(spell):
    text = "42 katu daude"
    result, changes = spell.correct_text(text)
    assert "42" in result
    assert changes == 0


def test_spell_correct_text_skips_number_suffixes(spell):
    """Words ≤5 chars after a number should be skipped (Basque suffixes)."""
    text = "2024ko ekitaldia"
    result, changes = spell.correct_text(text)
    assert "2024ko" in result
    # ekitaldia is a valid word
    assert "ekitaldia" in result


def test_spell_correct_text_skips_urls(spell):
    text = "ikus https://itzune.eus orria"
    result, changes = spell.correct_text(text)
    assert "https://itzune.eus" in result
    assert changes == 0


def test_spell_correct_text_empty_string(spell):
    result, changes = spell.correct_text("")
    assert result == ""
    assert changes == 0


def test_spell_correct_text_only_punctuation(spell):
    result, changes = spell.correct_text("..., ---!!!")
    assert result == "..., ---!!!"
    assert changes == 0


def test_spell_correct_text_realistic_basque(spell):
    """A realistic example with multiple misspellings."""
    text = "etsea handia da baina txikixe"
    result, changes = spell.correct_text(text)
    # "etsea" should be corrected to something (etzea/etxea)
    assert "etsea" not in result
    assert changes >= 1


def test_spell_suggest_filters_allcaps_for_lowercase_input(spell):
    """The ALL-CAPS filtering happens in correct_text, not suggest().
    suggest() returns raw Hunspell output.
    But correct_text() should never pick ALL-CAPS as a replacement for lowercase input."""
    text = "ser ez da etsea"
    result, changes = spell.correct_text(text)
    # "ser" should be replaced with "zer", never "SER"
    assert "SER" not in result
    assert "zer" in result
    assert changes > 0


# ── Not loaded (no hunspell) ─────────────────────────────

def test_spell_checker_not_loaded_returns_true():
    """Without hunspell, correct() should return True (assume correct)."""
    nonexistent = Path("/tmp/nonexistent_dict_for_test")
    s = SpellChecker(nonexistent)
    assert s.loaded is False
    # Should return True (= assume valid, don't block) when hunspell not available
    assert s.correct("ser") is True
    assert s.suggest("ser") == []
    text, changes = s.correct_text("ser gertatu da")
    assert text == "ser gertatu da"
    assert changes == 0
