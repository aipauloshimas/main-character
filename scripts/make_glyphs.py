"""Build the table of the characters each bundled font draws. For maintainers only.

The skill never runs this. Its table, scripts/mclib/glyphs.json, is committed, and it is read from there
at run time (mclib/glyphs.py), so the skill needs no font library to know which characters a note
can be set in. Run it again whenever a font in fonts/ changes:

    python scripts/make_glyphs.py

Each font's characters are read from its cmap table, the font's own list of the code points it has a
glyph for, with a small reader of the formats a TrueType font uses (0, 4, 6 and 12). The table
keeps them as ranges, [first, last] with both ends included, so a font with a few hundred characters
is a line or two of numbers. tests/test_glyphs.py reads the fonts again and fails when the table no
longer matches them.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from mclib.config import FONT_FILES, FONTS_DIR
from mclib.glyphs import TABLE


def font_characters(path: Path) -> set[int]:
    """The code points a font file has a glyph for, from its Unicode cmap subtables (platform 0, and
    platform 3 with encoding 1 or 10: the ones a text is drawn through). ValueError for a file with no
    cmap, or whose Unicode subtable is in a format this reader does not know."""
    data = Path(path).read_bytes()
    (tables,) = struct.unpack(">H", data[4:6])
    cmap = None
    for number in range(tables):
        tag, _, offset, _ = struct.unpack(">4sIII", data[12 + 16 * number : 28 + 16 * number])
        if tag == b"cmap":
            cmap = offset
    if cmap is None:
        raise ValueError(f"{path} has no cmap table")
    _, subtables = struct.unpack(">HH", data[cmap : cmap + 4])
    found: set[int] = set()
    for number in range(subtables):
        platform, encoding, offset = struct.unpack(">HHI", data[cmap + 4 + 8 * number : cmap + 12 + 8 * number])
        if platform == 0 or (platform == 3 and encoding in (1, 10)):
            found |= _subtable(data, cmap + offset)
    return found


def _subtable(data: bytes, at: int) -> set[int]:
    """The code points one cmap subtable maps to a glyph (glyph 0 is .notdef: not a character)."""
    (kind,) = struct.unpack(">H", data[at : at + 2])
    chars: set[int] = set()
    if kind == 0:  # one byte for each of the first 256 code points
        chars.update(code for code, glyph in enumerate(data[at + 6 : at + 262]) if glyph)
    elif kind == 4:  # segments of the BMP, with a delta or a table of glyphs for each
        _, _, double = struct.unpack(">HHH", data[at + 2 : at + 8])
        segments = double // 2
        ends_at = at + 14
        starts_at, deltas_at = ends_at + double + 2, ends_at + 2 * double + 2
        offsets_at = ends_at + 3 * double + 2
        ends = struct.unpack(f">{segments}H", data[ends_at : ends_at + double])
        starts = struct.unpack(f">{segments}H", data[starts_at : starts_at + double])
        deltas = struct.unpack(f">{segments}h", data[deltas_at : deltas_at + double])
        offsets = struct.unpack(f">{segments}H", data[offsets_at : offsets_at + double])
        for segment in range(segments):
            for code in range(starts[segment], ends[segment] + 1):
                if code == 0xFFFF:  # (the last segment is a sentinel)
                    continue
                if offsets[segment] == 0:
                    glyph = (code + deltas[segment]) & 0xFFFF
                else:
                    where = offsets_at + 2 * segment + offsets[segment] + 2 * (code - starts[segment])
                    (glyph,) = struct.unpack(">H", data[where : where + 2])
                    glyph = (glyph + deltas[segment]) & 0xFFFF if glyph else 0
                if glyph:
                    chars.add(code)
    elif kind == 6:  # a run of code points from the first one
        first, count = struct.unpack(">HH", data[at + 6 : at + 10])
        glyphs = struct.unpack(f">{count}H", data[at + 10 : at + 10 + 2 * count])
        chars.update(first + index for index, glyph in enumerate(glyphs) if glyph)
    elif kind == 12:  # groups of consecutive code points with consecutive glyphs: the whole of Unicode
        (groups,) = struct.unpack(">I", data[at + 12 : at + 16])
        for group in range(groups):
            first, last, glyph = struct.unpack(">III", data[at + 16 + 12 * group : at + 28 + 12 * group])
            if glyph:
                chars.update(range(first, last + 1))
    else:
        raise ValueError(f"a cmap subtable of format {kind}, which this reader does not know")
    return chars


def ranges(characters: set[int]) -> list[list[int]]:
    """Code points as sorted [first, last] ranges, both ends included."""
    result: list[list[int]] = []
    for code in sorted(characters):
        if result and code == result[-1][1] + 1:
            result[-1][1] = code
        else:
            result.append([code, code])
    return result


def build(fonts: Path = FONTS_DIR) -> dict[str, list[list[int]]]:
    """The table: the ranges of every bundled font (the files of config.FONT_FILES), by file name."""
    files = sorted({name for name, _ in FONT_FILES.values()})
    return {name: ranges(font_characters(fonts / name)) for name in files}


def as_text(table: dict[str, list[list[int]]]) -> str:
    """The table as the JSON file holds it: one font to a line, so a change to one font is a change
    to one line."""
    lines = [f"  {json.dumps(name)}: {json.dumps(found, separators=(',', ':'))}" for name, found in table.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def main() -> int:
    try:
        table = build()
    except (OSError, ValueError, struct.error) as error:
        print(f"Could not read the fonts: {error}", file=sys.stderr)
        return 1
    TABLE.write_text(as_text(table), encoding="utf-8", newline="\n")
    for name, found in table.items():
        print(f"ok  {name}  {sum(last - first + 1 for first, last in found)} characters in {len(found)} ranges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
