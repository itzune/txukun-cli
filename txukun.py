#!/usr/bin/env python3
"""
Txukun CLI — Basque text correction (3 models).

Uses the same detection and correction pipeline as the txukun web app:
  1. Cap-punct (MarianMT ONNX) — capitalization & punctuation
  2. Spelling (Hunspell + Tier1 freq + BERTeus ONNX re-ranking)
  3. Grammar (GECToR ONNX) — grammatical error correction

Usage:
    # Correct mode (default): outputs corrected text
    uv run python txukun.py "zer moduz zaude"
    uv run python txukun.py -c "zer moduz zaude"

    # Detect mode: outputs JSON with all errors
    uv run python txukun.py -d "zer moduz zaude"

    # Enable/disable specific models
    uv run python txukun.py --disable grammar "text"
    uv run python txukun.py --enable spell --enable cappunct "text"

    # Other I/O options
    uv run python txukun.py --file input.txt --output output.txt
    cat raw.txt | uv run python txukun.py --stdin
"""
import sys
import time

try:
    import click
except ImportError:
    print("Error: click not installed. Run: uv sync", file=sys.stderr)
    sys.exit(1)

from txukun_lib.cappunct import CapPunctModel
from txukun_lib.spelling import SpellChecker
from txukun_lib.bert import BerteusReranker
from txukun_lib.grammar import GectorModel
from txukun_lib.analyze import analyze_text
from txukun_lib.errors import errors_to_json, apply_corrections
from txukun_lib.markdown import strip_markdown

# ── Singleton models ────────────────────────────────

_cappunct: CapPunctModel | None = None
_spell: SpellChecker | None = None
_bert: BerteusReranker | None = None
_grammar: GectorModel | None = None


def get_models(enabled: set[str], quiet: bool = False):
    """Get or create model instances for the enabled models."""
    global _cappunct, _spell, _bert, _grammar

    if "cap-punct" in enabled and _cappunct is None:
        _cappunct = CapPunctModel(quiet=quiet)
    if "grammar" in enabled and _grammar is None:
        _grammar = GectorModel(quiet=quiet)
    if "spell" in enabled:
        if _bert is None:
            _bert = BerteusReranker(quiet=quiet)
        if _spell is None:
            _spell = SpellChecker(bert=_bert, quiet=quiet)

    return _cappunct, _spell, _grammar


# ── CLI ─────────────────────────────────────────────

ALL_MODELS = ["cap-punct", "spell", "grammar"]


@click.command()
@click.argument("text", required=False, default=None)
@click.option("-d", "--detect", is_flag=True, help="Output JSON with detected errors")
@click.option("-c", "--correct", is_flag=True, help="Output corrected text (default)")
@click.option(
    "--enable", multiple=True, type=click.Choice(ALL_MODELS),
    help="Enable specific models (if given, ONLY these are enabled)",
)
@click.option(
    "--disable", multiple=True, type=click.Choice(ALL_MODELS),
    help="Disable specific models",
)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read input from a file")
@click.option("--stdin", "use_stdin", is_flag=True, help="Read input from stdin")
@click.option("--output", "-o", type=click.Path(), help="Write output to a file")
@click.option("--quiet", "-q", is_flag=True, help="Suppress status messages")
def main(text, detect, correct, enable, disable, file, use_stdin, output, quiet):
    """Txukun CLI — Basque text correction (cap-punct + spelling + grammar).

    \b
    Examples:
      uv run python txukun.py "zer moduz zaude"
      uv run python txukun.py -d "zer moduz zaude"
      uv run python txukun.py --disable grammar "text"
      uv run python txukun.py -f input.txt -o output.txt
      cat raw.txt | uv run python txukun.py --stdin
    """
    # ── Read input ──
    if use_stdin:
        text = sys.stdin.read().strip()
    elif file:
        with open(file) as f:
            text = f.read().strip()
    elif text is None:
        click.echo("Error: no input. Use TEXT, --file, or --stdin.", err=True)
        sys.exit(1)

    if not text:
        click.echo("Error: empty input.", err=True)
        sys.exit(1)

    # ── Determine enabled models ──
    if enable:
        enabled = set(enable) - set(disable)
    else:
        enabled = set(ALL_MODELS) - set(disable)

    if not enabled:
        click.echo("Error: no models enabled.", err=True)
        sys.exit(1)

    # ── Run analysis ──
    if not quiet:
        model_names = " + ".join(sorted(enabled))
        click.echo(f"🔍 Analizatzen ({model_names})...", err=True)

    t0 = time.time()
    cappunct, spell, grammar = get_models(enabled, quiet=quiet)
    errors = analyze_text(text, cappunct, spell, grammar)
    elapsed = time.time() - t0

    if not quiet:
        click.echo(f"  ✓ {len(errors)} akats aurkituta ({elapsed:.1f}s)", err=True)

    # ── Output ──
    if detect:
        # JSON output with all errors
        result = errors_to_json(errors)
    else:
        # Corrected text
        result = apply_corrections(text, errors)

    if output:
        with open(output, "w") as f:
            f.write(result)
        if not quiet:
            click.echo(f"💾 Emaitza gordeta: {output}", err=True)
    else:
        sys.stdout.write(result + "\n")


if __name__ == "__main__":
    main()
