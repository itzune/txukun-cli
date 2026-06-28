#!/usr/bin/env python3
"""
Txukun CLI — Basque text capitalization, punctuation restoration, and spell checking.

Uses HiTZ/cap-punct-eu (MarianMT) via ONNX Runtime (int8 quantized) for cap+punct,
and a pre-built Basque word list (160k words) for optional spell checking
with automatic first-suggestion replacement.

Usage:
    uv run python txukun.py "ser gertatu da hemen"
    uv run python txukun.py --spell "ser gertatu da hemen"
    cat input.txt | uv run python txukun.py --stdin
    uv run python txukun.py --file euskaraz.txt --output zuzendua.txt
"""

import re
import sys
import time
from pathlib import Path

try:
    import click
except ImportError:
    print("Error: click not installed. Run: uv sync", file=sys.stderr)
    sys.exit(1)


# ── Constants ──────────────────────────────────────────────

MODEL_ID = "itzune/txukun-cap-punct-eu"

WORDLIST_PATH = Path(__file__).parent / "data" / "eu-words.txt"
FREQ_PATH = Path(__file__).parent / "data" / "eu-words-freq.txt"

# MarianMT special tokens to clean
CLEANUP_RE = re.compile(r"</?s>|</?pad>|<unk>")

# Basque word tokenizer (matches the browser version)
WORD_RE = re.compile(
    r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛçÇ'\-]+"
    r"|\d+(?:[.,]\d+)*"
    r"|https?://\S+"
    r"|[\w.-]+@[\w.-]+"
)


# ── Spell Checker ──────────────────────────────────────────

class SpellChecker:
    """Simple spell checker backed by a 160k-word Basque word list."""

    def __init__(self, wordlist_path: Path, freq_path: Path | None = None):
        self.words: set[str] = set()
        self.freq: dict[str, int] = {}

        if wordlist_path.exists():
            with open(wordlist_path) as f:
                self.words = {line.strip() for line in f if line.strip()}
        else:
            bundled = Path(__file__).parent / "eu-words.txt"
            if bundled.exists():
                with open(bundled) as f:
                    self.words = {line.strip() for line in f if line.strip()}

        for path in [freq_path, FREQ_PATH]:
            if path and path.exists():
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if "\t" in line:
                            word, count = line.split("\t", 1)
                            self.freq[word] = int(count)
                break

    @property
    def loaded(self) -> bool:
        return len(self.words) > 0

    def correct(self, word: str) -> bool:
        """Check if a word is correct (case-insensitive, including uppercase acronyms)."""
        if word in self.words:
            return True
        lower = word.lower()
        if lower in self.words:
            return True
        # Also try uppercase (acronyms like EITB are stored in uppercase form)
        upper = word.upper()
        if upper in self.words and upper != lower:
            return True
        return False

    def _is_valid_word(self, word: str) -> bool:
        """Check if a word is valid, including hyphen-split compound parts."""
        if not word or len(word) < 2:
            return True  # short parts are OK in compounds
        if self.correct(word):
            return True
        # Hyphenated? Check each part
        if "-" in word:
            parts = word.split("-")
            return all(self._is_valid_word(p) for p in parts)
        return False

    def suggest(self, word: str) -> list[str]:
        """Levenshtein distance ≤ 2, sorted by distance then frequency."""
        lower = word.lower()
        candidates = []

        for w in self.words:
            wl = w.lower()
            if wl == lower:
                continue
            if abs(len(wl) - len(lower)) > 2:
                continue
            dist = self._levenshtein(lower, wl)
            if dist <= 2 and dist > 0:
                candidates.append((dist, w))

        candidates.sort(key=lambda x: (x[0], -self.freq.get(x[1], 0)))
        return [w for _, w in candidates[:5]]

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = min(
                    dp[j] + 1,
                    dp[j - 1] + 1,
                    prev + (0 if a[i - 1] == b[j - 1] else 1),
                )
                prev = temp
        return dp[n]

    def correct_text(self, text: str) -> tuple[str, int]:
        """Replace misspelled words with first suggestion. Returns (text, changes)."""
        if not self.loaded:
            return text, 0

        changes = 0
        tokens = list(WORD_RE.finditer(text))
        result_chars = list(text)

        for i, match in enumerate(tokens):
            word = match.group(0)
            start, end = match.start(), match.end()

            # Skip numbers, short words, all-caps
            if word.isdigit():
                continue
            if len(word) < 2:
                continue
            if word == word.upper() and len(word) > 1:
                continue

            # Skip short suffixes attached to numbers (42koa, 15ekoa, 42ko)
            if len(word) <= 5 and i > 0 and tokens[i - 1].group(0).isdigit():
                continue

            # Check validity (including hyphen-split)
            if self._is_valid_word(word):
                continue

            suggestions = self.suggest(word.lower())
            if suggestions:
                changes += 1
                replacement = suggestions[0]
                # Preserve original casing pattern if word was title-case
                if word[0].isupper() and word[1:].islower():
                    replacement = replacement[0].upper() + replacement[1:]
                result_chars[start:end] = list(replacement)

        return "".join(result_chars), changes


# ── Model (ONNX Runtime via optimum) ────────────────────────

class TxukunModel:
    """Lazy-loads the cap-punct-eu MarianMT model via ONNX Runtime (int8 quantized)."""

    def __init__(self, quiet: bool = False):
        self._pipeline = None
        self._quiet = quiet

    def _load(self):
        if self._pipeline is not None:
            return

        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer

        click.echo("⏳ Deskargatzen cap-punct-eu ONNX int8 modeloa...", err=True)

        # Load ONNX model via optimum
        # Our decoder_model_merged_quantized.onnx IS a with-past decoder
        model = ORTModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            encoder_file_name="encoder_model_quantized.onnx",
            decoder_file_name="decoder_model_merged_quantized.onnx",
            decoder_with_past_file_name="decoder_model_merged_quantized.onnx",
            provider="CPUExecutionProvider",
            use_cache=True,
        )

        # Tokenizer: load from HiTZ/cap-punct-eu (has source.spm + vocab.json)
        click.echo("  ▸ Tokenizadorea kargatzen...", err=True)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenizer = AutoTokenizer.from_pretrained("HiTZ/cap-punct-eu")

        from transformers import pipeline
        self._pipeline = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
        )

        click.echo("✅ Eredua kargatuta (ONNX int8, ~77 MB).", err=True)

    def correct(self, text: str) -> str:
        self._load()
        result = self._pipeline(text)
        output = result[0]["translation_text"]
        output = clean_output(output)
        if not output.strip():
            output = text
        return output


def clean_output(text: str) -> str:
    """Clean model output: remove special tokens, collapse whitespace."""
    text = CLEANUP_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ── CLI ─────────────────────────────────────────────────────

MODEL_INSTANCE = None
SPELL_INSTANCE = None


def get_model():
    global MODEL_INSTANCE
    if MODEL_INSTANCE is None:
        MODEL_INSTANCE = TxukunModel()
    return MODEL_INSTANCE


def get_spell():
    global SPELL_INSTANCE
    if SPELL_INSTANCE is None:
        SPELL_INSTANCE = SpellChecker(WORDLIST_PATH, FREQ_PATH)
    return SPELL_INSTANCE


@click.command()
@click.argument("text", required=False, default=None)
@click.option(
    "--file", "-f",
    type=click.Path(exists=True),
    help="Read input from a file instead of argument",
)
@click.option(
    "--stdin",
    is_flag=True,
    help="Read input from stdin",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Write output to a file (otherwise prints to stdout)",
)
@click.option(
    "--spell",
    is_flag=True,
    default=False,
    help="Enable spell checking (default: disabled)",
)
@click.option(
    "--no-punct/--punct",
    default=False,
    help="Disable cap+punct correction (default: --punct, enabled)",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress status messages (stderr)",
)
def main(text, file, stdin, output, spell, no_punct, quiet):
    """
    Txukun CLI — Basque text capitalization, punctuation, and spell checker.

    Cap+punct uses the int8-quantized ONNX model (itzune/txukun-cap-punct-eu,
    ~77 MB). Spell checking uses a 160k-word dictionary (--spell to enable).

    \b
    Examples:
      uv run python txukun.py "ser gertatu da hemen"
      uv run python txukun.py --spell "ser gertatu da hemen"
      uv run python txukun.py --file input.txt --output output.txt
      cat raw.txt | uv run python txukun.py --stdin
    """
    if stdin:
        text = sys.stdin.read().strip()
    elif file:
        with open(file) as f:
            text = f.read().strip()
    elif text is None:
        click.echo("Error: no input provided. Use TEXT, --file, or --stdin.", err=True)
        sys.exit(1)

    if not text:
        click.echo("Error: empty input.", err=True)
        sys.exit(1)

    result = text

    # Step 1: Cap+punct correction
    if not no_punct:
        if not quiet:
            click.echo("🔤 Maiuskulak eta puntuazioa zuzentzen...", err=True)
        try:
            t0 = time.time()
            model = get_model()
            result = model.correct(result)
            if not quiet:
                click.echo(f"  ✓ {time.time() - t0:.1f}s", err=True)
        except Exception as e:
            if not quiet:
                click.echo(f"⚠️  Eredu errorea: {e}", err=True)

    # Step 2: Spell check (disabled by default, --spell to enable)
    if spell:
        if not quiet:
            click.echo("🔍 Ortografia zuzentzen...", err=True)
        spell_checker = get_spell()
        if spell_checker.loaded:
            t0 = time.time()
            result, spell_changes = spell_checker.correct_text(result)
            if not quiet:
                if spell_changes > 0:
                    click.echo(f"  ✓ {spell_changes} hitz zuzenduta ({time.time() - t0:.1f}s)", err=True)
                else:
                    click.echo(f"  ✓ Zuzena ({time.time() - t0:.1f}s)", err=True)
        elif not quiet:
            click.echo("  ⚠ Hiztegia ez dago eskuragarri (data/eu-words.txt).", err=True)

    if output:
        with open(output, "w") as f:
            f.write(result)
        if not quiet:
            click.echo(f"💾 Emaitza gordeta: {output}", err=True)
    else:
        sys.stdout.write(result + "\n")


if __name__ == "__main__":
    main()
