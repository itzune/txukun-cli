"""
Word-level LCS diff + case/punctuation utilities.

Port of txukun's analyze.js diffWords(). Tokenizes both texts into words
(with whitespace preserved as separate tokens), runs an LCS alignment,
and emits changes with original character offsets.

Case-only changes (e.g. "nire" → "Nire") are detected as replace changes
— the LCS uses toLowerCase() for alignment but checks actual text equality.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class DiffChange:
    type: str          # 'replace' | 'insert' | 'delete'
    from_text: str
    to_text: str
    from_offset: int
    to_offset: int


_TOKEN_RE = re.compile(r'(\s+|\S+)')


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Tokenize into whitespace/non-whitespace tokens with offsets."""
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def diff_words(original_text: str, corrected_text: str) -> list[DiffChange]:
    """Word-level LCS diff. Returns list of changes with offsets in original_text."""
    a = tokenize_with_offsets(original_text)
    b = tokenize_with_offsets(corrected_text)

    # Only compare non-whitespace tokens for alignment
    a_words = [(t, i) for i, t in enumerate(a) if t[0].strip()]
    b_words = [(t, i) for i, t in enumerate(b) if t[0].strip()]

    n = len(a_words)
    m = len(b_words)

    # LCS DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a_words[i][0][0].lower() == b_words[j][0][0].lower():
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    # Backtrack to build the edit script
    changes: list[DiffChange] = []
    i = j = 0
    while i < n and j < m:
        if a_words[i][0][0].lower() == b_words[j][0][0].lower():
            # Words match case-insensitively. If actual text differs,
            # emit a replace so case-only changes are not silently dropped.
            if a_words[i][0][0] != b_words[j][0][0]:
                changes.append(DiffChange(
                    'replace',
                    a_words[i][0][0],
                    b_words[j][0][0],
                    a_words[i][0][1],
                    a_words[i][0][2],
                ))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            # a_words[i] deleted or replaced
            if j < m and dp[i + 1][j + 1] < dp[i + 1][j]:
                # pure delete
                changes.append(DiffChange(
                    'delete',
                    a_words[i][0][0],
                    '',
                    a_words[i][0][1],
                    a_words[i][0][2],
                ))
            else:
                # replace a_words[i] with b_words[j]
                changes.append(DiffChange(
                    'replace',
                    a_words[i][0][0],
                    b_words[j][0][0],
                    a_words[i][0][1],
                    a_words[i][0][2],
                ))
                j += 1
            i += 1
        else:
            # b_words[j] inserted
            insert_offset = a_words[i][0][1] if i < n else len(original_text)
            changes.append(DiffChange(
                'insert',
                '',
                b_words[j][0][0],
                insert_offset,
                insert_offset,
            ))
            j += 1

    # Remaining insertions
    while j < m:
        insert_offset = len(original_text)
        changes.append(DiffChange('insert', '', b_words[j][0][0], insert_offset, insert_offset))
        j += 1

    # Remaining deletions
    while i < n:
        changes.append(DiffChange('delete', a_words[i][0][0], '', a_words[i][0][1], a_words[i][0][2]))
        i += 1

    return changes


def is_case_punct_only(a: str, b: str) -> bool:
    """Check if a→b is a case/punctuation-only change (same letters)."""
    if a == b:
        return False
    strip_a = re.sub(r'[^\w]', '', a, flags=re.UNICODE).lower()
    strip_b = re.sub(r'[^\w]', '', b, flags=re.UNICODE).lower()
    return strip_a == strip_b and len(strip_a) > 0


def apply_case_pattern(original: str, corrected: str, target: str) -> str | None:
    """Apply the case pattern from (original→corrected) to target.

    Handles first-letter capitalization (sentence-initial, proper nouns).
    Returns None if not applicable.
    """
    if original.lower() != corrected.lower():
        return None
    if (corrected and original and
            corrected[0].isupper() and original[0].islower() and
            corrected[0].lower() == original[0].lower()):
        return target[0].upper() + target[1:]
    return None


def is_in_heading(offset: int, heading_ranges: list[tuple[int, int]]) -> bool:
    """Check whether a plain-text offset falls within a heading line."""
    for start, end in heading_ranges:
        if start <= offset < end:
            return True
    return False
