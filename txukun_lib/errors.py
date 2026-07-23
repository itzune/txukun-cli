"""
Error dataclass + JSON serialization + correction application.

An Error represents a single suggested correction:
  {id, from, to, original, suggestion, category, title, context}

category: 'grammar' | 'spelling' | 'cappunct'
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class Error:
    """A single detected error with a suggested correction."""
    id: str
    frm: int          # start offset in original text
    to: int           # end offset (exclusive)
    original: str
    suggestion: str
    category: str     # 'grammar' | 'spelling' | 'cappunct'
    title: str
    context: str = ""
    confidence: float | None = None   # 0.0–1.0, model-specific


_err_counter = 0


def next_id() -> str:
    global _err_counter
    _err_counter += 1
    return f"e{_err_counter}"


def reset_counter() -> None:
    global _err_counter
    _err_counter = 0


def errors_to_json(errors: list[Error]) -> str:
    """Serialize errors to a JSON array."""
    return json.dumps([asdict(e) for e in errors], ensure_ascii=False, indent=2)


def apply_corrections(text: str, errors: list[Error]) -> str:
    """Apply all corrections to text (right-to-left to preserve offsets)."""
    result = text
    for e in sorted(errors, key=lambda e: e.frm, reverse=True):
        result = result[:e.frm] + e.suggestion + result[e.to:]
    return result
