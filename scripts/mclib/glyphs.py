"""Which characters each bundled font can draw.

A note in a script the fonts do not have (Cyrillic in the Marker notes, a star emoji) is not drawn
as written: the font has no glyph for those characters, and the note shows a gap where they should
be. The card has to say so instead of drawing the gap in silence, and this is where it knows.

glyphs.json holds, for every bundled font, the code points it has a glyph for, as [first, last]
ranges with both ends included. The table is made from the fonts themselves (scripts/make_glyphs.py
reads their cmap tables) and committed, and tests/test_glyphs.py fails when a font changes and the
table does not, so the skill needs no font library to read it.
"""

from __future__ import annotations

import bisect
import functools
import json
from pathlib import Path

from .config import FONT_FILES

TABLE = Path(__file__).with_name("glyphs.json")


@functools.cache
def _table() -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """The table of every font as (first code points, last code points) of its ranges, by file name."""
    raw = json.loads(TABLE.read_text(encoding="utf-8"))
    return {
        name: (tuple(first for first, _ in found), tuple(last for _, last in found))
        for name, found in raw.items()
    }


def covers(role: str, char: str) -> bool:
    """Whether the bundled font of a FONT_FILES role ("marker", "comic", ...) has a glyph for one
    character."""
    starts, ends = _table()[FONT_FILES[role][0]]
    code = ord(char)
    index = bisect.bisect_right(starts, code) - 1
    return index >= 0 and code <= ends[index]


def missing(role: str, text: str, case: str = "asis") -> list[str]:
    """The characters of a text that the font of a role can't draw, each once, in the order they
    first appear. Spaces are no glyphs, and are left out. case is how the style sets the text,
    "upper" or "asis" (layout.apply_case): the font is asked about the letters as they are drawn, so
    a lowercase letter whose capital the font lacks is missing, and it is the letter as written that
    is named."""
    if case not in ("upper", "asis"):
        raise ValueError(f'case must be "upper" or "asis", got {case!r}')
    found: list[str] = []
    for char in text:
        if char.isspace() or char in found:
            continue
        drawn = char.upper() if case == "upper" else char
        if not all(covers(role, letter) for letter in drawn if not letter.isspace()):
            found.append(char)
    return found
