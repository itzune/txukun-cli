"""
Markdown stripping with offset mapping.

Port of txukun's analyze.js stripMarkdown(). Strips markdown syntax
markers before passing text to models (trained on plain text), then
maps error offsets back to original positions.

Strips: headings, bold, italic, strikethrough, inline code, links,
images, blockquotes, list markers, horizontal rules, code fences.
Also strips ASCII double quotes (GECToR's tokenizer splits them into
standalone tokens, breaking the LCS diff).

Returns (plain_text, position_map, heading_ranges) where:
  - position_map[plain_idx] = original_idx
  - heading_ranges = list of [start, end] plain-text offsets for heading content
"""
from __future__ import annotations


def strip_markdown(md: str) -> tuple[str, list[int], list[tuple[int, int]]]:
    """Strip markdown syntax, return plain text + position map + heading ranges."""
    plain = []
    pos_map: list[int] = []
    heading_ranges: list[tuple[int, int]] = []
    i = 0
    n = len(md)
    in_code_block = False
    in_heading = False
    heading_start = 0

    while i < n:
        # Code fence: toggle state, skip entire line
        if md[i:i+3] == '```':
            in_code_block = not in_code_block
            eol = md.find('\n', i)
            i = n if eol == -1 else eol + 1
            continue

        # Inside code block: skip entire line
        if in_code_block:
            eol = md.find('\n', i)
            i = n if eol == -1 else eol + 1
            continue

        # Start of line: strip block-level markers
        if i == 0 or md[i - 1] == '\n':
            j = i

            # Blockquote markers (> or >> ...)
            bq = True
            while bq:
                bq = False
                rest = md[j:]
                if rest.startswith('>'):
                    k = j
                    while k < n and md[k] == '>':
                        k += 1
                    if k < n and md[k] == ' ':
                        k += 1
                    j = k
                    bq = True

            # Heading markers (# to ######)
            rest = md[j:]
            h_match = _match_heading(rest)
            if h_match:
                j += h_match
                in_heading = True
                heading_start = len(plain)

            # List markers (- * + or 1.)
            rest = md[j:]
            l_match = _match_list(rest)
            if l_match:
                j += l_match

            # Horizontal rule (entire line is --- / *** / ___)
            rest = md[j:]
            if _is_hr(rest):
                eol = md.find('\n', j)
                i = n if eol == -1 else eol + 1
                continue

            i = j

        # Image ![alt](url) — skip entirely
        if md[i] == '!' and i + 1 < n and md[i + 1] == '[':
            close_bracket = md.find('](', i + 2)
            if close_bracket != -1:
                close_paren = md.find(')', close_bracket + 2)
                if close_paren != -1:
                    i = close_paren + 1
                    continue

        # Link [text](url) — keep text, drop URL
        if md[i] == '[':
            close_bracket = md.find('](', i + 1)
            if close_bracket != -1:
                close_paren = md.find(')', close_bracket + 2)
                if close_paren != -1:
                    for k in range(i + 1, close_bracket):
                        plain.append(md[k])
                        pos_map.append(k)
                    i = close_paren + 1
                    continue

        # Inline code `text` — keep content
        if md[i] == '`':
            end = md.find('`', i + 1)
            if end != -1:
                for k in range(i + 1, end):
                    plain.append(md[k])
                    pos_map.append(k)
                i = end + 1
                continue

        # Bold **text** or __text__ — keep content
        if (md[i:i+2] == '**') or (md[i:i+2] == '__'):
            marker = md[i:i+2]
            end = md.find(marker, i + 2)
            if end != -1:
                for k in range(i + 2, end):
                    plain.append(md[k])
                    pos_map.append(k)
                i = end + 2
                continue

        # Strikethrough ~~text~~ — keep content
        if md[i:i+2] == '~~':
            end = md.find('~~', i + 2)
            if end != -1:
                for k in range(i + 2, end):
                    plain.append(md[k])
                    pos_map.append(k)
                i = end + 2
                continue

        # Italic *text* or _text_ — keep content
        # (must come after bold/strikethrough; require non-space after opener)
        if i + 1 < n and md[i] in ('*', '_') and md[i + 1] != md[i] and md[i + 1] not in (' ', '\n', ''):
            ch = md[i]
            end = i + 1
            while end < n:
                if md[end] == ch and end + 1 < n and md[end + 1] != ch and (end == 0 or md[end - 1] != ch):
                    break
                if md[end] == ch and end + 1 >= n:
                    break
                end += 1
            if end < n:
                for k in range(i + 1, end):
                    plain.append(md[k])
                    pos_map.append(k)
                i = end + 1
                continue

        # ASCII double quotes — strip (breaks GECToR tokenization)
        if md[i] == '"':
            i += 1
            continue

        # Record heading range at end of heading line (before the newline)
        if md[i] == '\n' and in_heading:
            heading_ranges.append((heading_start, len(plain)))
            in_heading = False

        plain.append(md[i])
        pos_map.append(i)
        i += 1

    # Heading at EOF (no trailing newline)
    if in_heading:
        heading_ranges.append((heading_start, len(plain)))

    return ''.join(plain), pos_map, heading_ranges


def _match_heading(s: str) -> int:
    """Match # to ###### followed by space. Return length of match, 0 if no match."""
    count = 0
    while count < len(s) and count < 6 and s[count] == '#':
        count += 1
    if count > 0 and count < len(s) and s[count] == ' ':
        return count + 1
    return 0


def _match_list(s: str) -> int:
    """Match list markers (- * + or 1.). Return length of match, 0 if no match."""
    if len(s) >= 2 and s[0] in '-*+' and s[1] == ' ':
        return 2
    # numbered list: 1. 2. etc.
    j = 0
    while j < len(s) and s[j].isdigit():
        j += 1
    if j > 0 and j < len(s) and s[j] == '.' and j + 1 < len(s) and s[j + 1] == ' ':
        return j + 2
    return 0


def _is_hr(s: str) -> bool:
    """Check if line is a horizontal rule (--- / *** / ___)."""
    stripped = s.split('\n', 1)[0].strip()
    if len(stripped) < 3:
        return False
    ch = stripped[0]
    if ch not in '-*_':
        return False
    return all(c == ch for c in stripped)


def map_offset(plain_offset: int, pos_map: list[int], is_end: bool = False) -> int:
    """Map a plain-text offset to an original-text offset."""
    if not pos_map:
        return plain_offset
    if is_end:
        if plain_offset >= len(pos_map):
            return pos_map[-1] + 1
        if plain_offset <= 0:
            return pos_map[0]
        return pos_map[plain_offset - 1] + 1
    if plain_offset >= len(pos_map):
        return pos_map[-1] + 1
    return pos_map[max(0, plain_offset)]


def build_context(plain_text: str, from_offset: int) -> str:
    """Build a leading-context snippet, bounded by the current paragraph."""
    para_start = plain_text.rfind('\n', 0, from_offset) + 1
    ctx_start = max(para_start, from_offset - 28)
    ctx = plain_text[ctx_start:from_offset]
    if ctx_start > para_start:
        ctx = '\u2026' + ctx
    return ctx.rstrip()
