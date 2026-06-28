#!/usr/bin/env python3
"""
Txukun CLI — Basque text capitalization, punctuation restoration, and spell checking.

Uses HiTZ/cap-punct-eu (MarianMT) via ONNX Runtime (int8 quantized) for cap+punct,
and Hunspell with Xuxen Basque dictionary for optional spell checking.

Usage:
    uv run python txukun.py "ser gertatu da hemen"
    uv run python txukun.py --spell "ser gertatu da hemen"
    cat input.txt | uv run python txukun.py --stdin
    uv run python txukun.py --file euskaraz.txt --output zuzendua.txt
"""

import re
import sys
import time
import subprocess
from pathlib import Path

try:
    import click
except ImportError:
    print("Error: click not installed. Run: uv sync", file=sys.stderr)
    sys.exit(1)


# ── Constants ──────────────────────────────────────────────

MODEL_ID = "itzune/txukun-cap-punct-eu"

DICT_PATH = Path(__file__).parent / "data" / "eu"  # eu.aff + eu.dic

# MarianMT special tokens to clean
CLEANUP_RE = re.compile(r"</?s>|</?pad>|<unk>")

# Basque word tokenizer (matches the browser version)
WORD_RE = re.compile(
    r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛçÇ'\-]+"
    r"|\d+(?:[.,]\d+)*"
    r"|https?://\S+"
    r"|[\w.-]+@[\w.-]+"
)


# ── Spell Checker (Hunspell via subprocess) ─────────────────

class SpellChecker:
    """Spell checker backed by Hunspell with Xuxen Basque dictionary.

    Uses 'hunspell -a -d dict_path' in persistent pipe mode (ispell format).
    The subprocess stays alive between calls for low-latency checking.
    """

    def __init__(self, dict_path: Path):
        self._dict = str(dict_path)
        self._proc: subprocess.Popen | None = None
        self._broken = False

    def _ensure_proc(self):
        """Lazily start the hunspell subprocess."""
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
                # Consume the initial header line
                self._proc.stdout.readline()
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
        self._ensure_proc()
        return self._proc is not None and self._proc.poll() is None

    def correct(self, word: str) -> bool:
        """Check if a single word is spelled correctly."""
        if not self.loaded:
            return True  # if hunspell is unavailable, assume correct
        self._ensure_proc()
        try:
            self._proc.stdin.write(word + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline().strip()
            # Skip blank line that ispell uses as response terminator
            if not line:
                line = self._proc.stdout.readline().strip()
            # ispell format:
            #   "*"       → correct
            #   "+"       → correct (root form shown)
            #   "& word ..." → misspelled
            #   "# word"  → no suggestions
            if line.startswith("*") or line.startswith("+"):
                return True
            return False
        except (BrokenPipeError, OSError):
            self._close()
            return True

    def suggest(self, word: str) -> list[str]:
        """Get spelling suggestions for a word."""
        if not self.loaded:
            return []
        self._ensure_proc()
        try:
            self._proc.stdin.write(word + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline().strip()
            # Skip blank line that ispell uses as response terminator
            if not line:
                line = self._proc.stdout.readline().strip()

            if line.startswith("&") or line.startswith("#"):
                # Format: "& word count offset: sug1, sug2, ..."
                #          "# word offset" (no suggestions)
                if ":" in line:
                    suggestions_part = line.split(":", 1)[1].strip()
                    return [s.strip() for s in suggestions_part.split(", ") if s.strip()]
            return []
        except (BrokenPipeError, OSError):
            self._close()
            return []

    def correct_text(self, text: str) -> tuple[str, int]:
        """Check all words in text, replace misspelled ones with first suggestion."""
        if not self.loaded:
            return text, 0

        changes = 0
        tokens = list(WORD_RE.finditer(text))
        result_chars = list(text)

        for i, match in enumerate(tokens):
            word = match.group(0)
            start, end = match.start(), match.end()

            # Skip numbers, URLs, emails, short words, all-caps
            if word.isdigit():
                continue
            if "@" in word or word.startswith("http"):
                continue
            if len(word) < 2:
                continue
            if word == word.upper() and len(word) > 1:
                continue

            # Skip short suffixes attached to numbers (42koa, 15ekoa, 42ko)
            if len(word) <= 5 and i > 0 and tokens[i - 1].group(0).isdigit():
                continue

            # Check via hunspell
            if self.correct(word):
                continue

            suggestions = self.suggest(word)
            # Filter: prefer suggestions matching input case pattern
            # Skip ALL-CAPS suggestions when input is lowercase (e.g., "SER" for "ser")
            if word.islower():
                suggestions = [s for s in suggestions if not s.isupper()]
            elif word[0].isupper() and word[1:].islower():
                # Title case: prefer title-case suggestions, but accept lowercase too
                suggestions = [s for s in suggestions if not s.isupper()]
            if suggestions:
                changes += 1
                replacement = suggestions[0]
                # Preserve original casing pattern if word was title-case
                if word[0].isupper() and word[1:].islower():
                    replacement = replacement[0].upper() + replacement[1:]
                result_chars[start:end] = list(replacement)

        return "".join(result_chars), changes

    def __del__(self):
        self._close()


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
        SPELL_INSTANCE = SpellChecker(DICT_PATH)
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
    ~77 MB). Spell checking uses Hunspell with the Xuxen Basque dictionary
    (data/eu.aff + data/eu.dic).

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
            click.echo("  ⚠ Hunspell ez dago eskuragarri. Instalatu: sudo apt install hunspell.", err=True)

    if output:
        with open(output, "w") as f:
            f.write(result)
        if not quiet:
            click.echo(f"💾 Emaitza gordeta: {output}", err=True)
    else:
        sys.stdout.write(result + "\n")


if __name__ == "__main__":
    main()
