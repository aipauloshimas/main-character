"""Where the four notes of the card go: which corner, how big their text is, and where their arrows
run.

The card has a note in each corner of the stage, around the person: top left (tl), top right (tr),
bottom left (bl) and bottom right (br), each with an arrow that points at the person. fit_note
sizes a note's text and gives its box. face_box finds the part of the picture no note may cover,
and prepare_matte can be given other boxes to keep notes out of (the title room and the safe area of a
9:16 card).
prepare_matte does the slow work that depends on the matte alone, once. plan_notes gives all the
notes of a card one font size, the biggest at which every note can be set whole and put on a spot
clear of the face and of the other notes, which keep their distance from each other (see
_apart), and runs an arrow from each note to the person when there is room for one. This module
only measures and places; the styles draw. Everything is in stage pixels, with (0, 0) at the
top-left corner of the stage, and every number it returns is a plain int.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from PIL import ImageFont

from .analyze import person_boxes
from .glyphs import missing as missing_glyphs
from .raster import dilate, font, quad_points

# The ellipsis character is built with chr(), not typed, so no tool can change it on the way in.
_ELLIPSIS = chr(0x2026)
_MAX_LINES = 3  # a note is at most this many lines

# face_box
_FACE_PAD = 0.04  # of the stage's width: the air on either side of the face
_FACE_AIR = 0.03  # of the stage's height: the air below the face
_FACE_AIR_EDGE = 0.05  # and when the head touches a side of the stage, which hides part of it
_WIDEST_IN = 0.45  # the head's widest row is in the top 45% of the person
_NECK_BY = 0.8  # and the neck is above 80% of the way down
_NECK_NARROWER = 0.85  # a neck is narrower than this share of the head's widest row
_SHOULDERS_WIDER = 1.15  # and some row below it is at least this much wider than the neck
_SHARP_WIDENING = 1.4  # a row this many times the head's width is shoulders or raised arms
_HEAD_MIN_ROWS = 0.15  # of the person's height: before this a widening or a narrowing is no end
_HEAD_AT_LEAST = 1.3  # a head is at least this many widths tall: a narrow row above is no neck
_HEAD_USUALLY = 1.45  # and about this many: where the face ends when there is no clear neck
_HEAD_ABOVE = 0.63  # a head's widest row is this many widths below its top (for a cropped top)
_DEFAULT_FACE = (0.36, 0.08, 0.64, 0.55)  # of the stage: the face when there is nobody in it
_PLATEAU_ROWS = 0.10  # of the person's height: the rows under the cap whose median width is the head's
_HOOD_GROWTH = 1.15  # a row this many times that width is a hood or shoulders widening the silhouette
_LEDGE = 0.25  # of the head's width: an edge this far past the rows above it is a hood or an arm
_LEDGE_ROWS = 8  # rows: how far down an edge is followed, since a ledge takes a few rows to stand out
_LEDGE_FROM = 0.1  # of a ledge's reach: the first row this far out is where the ledge starts

# prepare_matte and plan_notes
_MARGIN = 0.04  # of the stage's width: the gap to the stage's edge, and on either side of the face
_MAX_W = 0.42  # of the stage's width: the widest a note's box gets
_STEP = 8  # px between the spots tried
_CLASH_COST = 10**9  # per pixel of a note on the face or on another note: beats all the crowding
_EDGE_COST = 3  # per pixel of a note on the sticker edge
_NEAR_COST = 1  # per pixel of a note near the person (arrows on only)
_DISTANCE_COST = 0.02  # per pixel between a note's outer corner and its quadrant's
_ROOM = 56  # px of room an arrow is given between the sticker edge and the notes
_NEAR_SHRINK = 4  # the near mask is only a soft cost, so it is found at a quarter of the size
_CACHE_BYTES = 32 * 2**20  # the most prices of boxes a MatteContext keeps for the next call
_NOTE_GAP = 8  # px: the least space between two notes, at their sides
_STACK_GAP = 0.4  # of the font size: the space between two notes that stand one above the other

# arrows
_TARGET_GAP = 8  # px past the sticker edge where an arrow ends
_OUT = 10  # px, along each axis, an arrow starts out from the corner of its note facing the person
_SIDE_OUT = 8  # px an arrow starts out from a side of its note that faces the person
_SIDE_STEP = 16  # px between the starts tried along such a side
_MARCH = 2  # px per step of the walk from an arrow's start to the person
_CHORD = (28, 180)  # px at the usual stage height: shorter is a stub, and longer is a rope
_CHORD_BEST = 90  # px at the usual stage height: the chord the arrow chosen is nearest to
_CHORD_STAGE = 608  # the usual stage height, and the taller the stage the longer the arrows
_CHORD_SCALE_MAX = 2  # but not more than twice as long
_BEND = 0.18  # of an arrow's length: how far its curve bows out of the straight line
_SAMPLES = 16  # points of a curve that are checked against the person and the notes
_NEAR_START = 6  # px from its start within which a curve may touch a note
_FACING = {"tl": (1, 1), "tr": (-1, 1), "bl": (1, -1), "br": (-1, -1)}  # which way the person is


# --- styles and boxes -------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteStyle:
    """How a style sets its notes.

    font is a role of raster.font ("marker", "comic", "hand", "pixel", "terminal" or "label"), and
    the text is tried from max_size down to min_size. case is "upper" or "asis". line_h is the
    height of a line as a multiple of the size, pad the space between the text and the edge of its
    box, and band the width in px of the sticker edge the notes keep off (0 for none). arrows is
    whether the notes get arrows.
    """

    font: str
    max_size: int
    min_size: int
    case: str
    line_h: float = 1.0
    pad: int = 0
    band: int = 0
    arrows: bool = True


@dataclass
class NoteBox:
    """A note as it was laid out.

    lines are the lines of text as they are drawn (the case applied, an ellipsis on the end of a
    text that was cut), size the font size they were fitted at, and (x, y, w, h) the box, padding
    included, with (x, y) its top-left corner. arrow is (start, control, end), the three points of
    a quadratic curve from the note to the person, or None when the style has no arrows or the
    note has no room for one. clash is True when the note is not a clean one: its text was cut,
    or it overlaps the face box or a keep-out box, or it overlaps another note or is closer to
    it than notes are kept (see _apart), or it sits outside its own quadrant (a note had to
    leave its corner to find room). cut is True when its text was cut (it ends in an ellipsis
    the planner added) or has a character its font can't draw (glyphs.json: another alphabet, a
    star), so that a hero rule that moves the note and checks it again (clash_of) keeps the flag
    that planning gave it.
    """

    key: str
    lines: list[str]
    size: int
    x: int
    y: int
    w: int
    h: int
    arrow: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None
    clash: bool = False
    cut: bool = False


# --- fitting a note's text --------------------------------------------------------------------


def apply_case(text: str, case: str) -> str:
    """The text in a style's case: "upper" makes capitals, "asis" leaves it alone."""
    if case == "upper":
        return text.upper()
    if case == "asis":
        return text
    raise ValueError(f'case must be "upper" or "asis", got {case!r}')


def _break(words: list[str], fits: Callable[[str], bool]) -> list[list[str]]:
    """Words broken into lines, greedily: a line takes the next word as long as the result still
    fits. A word that does not fit even alone gets a line to itself, since words are never split."""
    lines: list[list[str]] = []
    for word in words:
        if lines and fits(" ".join([*lines[-1], word])):
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines


def _ellipsize(words: list[str], fits: Callable[[str], bool]) -> str:
    """The longest start of the words that still fits with an ellipsis after it. Whole words are
    dropped first, and only when even the first one is too long are its characters."""
    for count in range(len(words), 0, -1):
        line = " ".join(words[:count]) + _ELLIPSIS
        if fits(line):
            return line
    first = words[0]
    for count in range(len(first) - 1, 0, -1):
        line = first[:count] + _ELLIPSIS
        if fits(line):
            return line
    return _ELLIPSIS


def _clip(lines: list[list[str]], fits: Callable[[str], bool], max_lines: int) -> list[str]:
    """Lines of words as strings, cut to max_lines. The last line then takes all the words that
    are left and is trimmed to end in an ellipsis."""
    if len(lines) <= max_lines:
        return [" ".join(line) for line in lines]
    kept = [" ".join(line) for line in lines[: max_lines - 1]]
    left = [word for line in lines[max_lines - 1 :] for word in line]
    return [*kept, _ellipsize(left, fits)]


def wrap(text: str, max_chars: int, max_lines: int = _MAX_LINES) -> list[str]:
    """The text broken into lines of at most max_chars characters, greedily and without splitting
    a word (a word longer than max_chars keeps a line to itself). A text that needs more than
    max_lines lines is cut there: the last line ends in an ellipsis and still fits max_chars."""
    if max_lines < 1:
        raise ValueError(f"max_lines must be at least 1, got {max_lines}")

    def fits(line: str) -> bool:
        return len(line) <= max_chars

    return _clip(_break(text.split(), fits), fits, max_lines)


@lru_cache(maxsize=8192)
def _length(fnt: ImageFont.FreeTypeFont, line: str) -> float:
    """The width in px of a line set in a font. It is cached, because setting a note at every
    width and size measures the same lines over and over."""
    return fnt.getlength(line)


def _fits_in(fnt: ImageFont.FreeTypeFont, room: int) -> Callable[[str], bool]:
    """A test for _break and _ellipsize: does a line of text fit in `room` px set in this font?"""
    return lambda line: _length(fnt, line) <= room


def _sizes(style: NoteStyle) -> list[int]:
    """The font sizes to try, biggest first: steps of 2 down to min_size, except for two fonts. The
    pixel role steps by 8, because Press Start 2P is only crisp on its 8 px grid (24, then 16), and
    the terminal role (VT323) steps by 4 (52, 48 ... 36)."""
    if style.font == "pixel":
        sizes = list(range(style.max_size // 8 * 8, style.min_size - 1, -8))
    else:
        step = 4 if style.font == "terminal" else 2
        sizes = [*range(style.max_size, style.min_size, -step), style.min_size]
    return sizes or [style.min_size]


class _Fit(NamedTuple):
    """A text set in a box: its lines, the font size, the box (w, h), and whether words were cut."""

    lines: list[str]
    size: int
    w: int
    h: int
    cut: bool


def _fit(text: str, style: NoteStyle, max_w: int) -> _Fit:
    """fit_note, and whether the text had to be cut (more than three lines at the smallest size)."""
    words = apply_case(text, style.case).split()
    room = max_w - 2 * style.pad
    for size in _sizes(style):
        fnt = font(style.font, size)
        fits = _fits_in(fnt, room)
        broken = _break(words, fits)
        if len(broken) <= _MAX_LINES and all(fits(" ".join(line)) for line in broken):
            break
    # Whichever size it ended on, only the smallest can have too many lines left to cut
    cut = len(broken) > _MAX_LINES
    lines = _clip(broken, fits, _MAX_LINES)

    widest = max((_length(fnt, line) for line in lines), default=0)
    return _Fit(
        lines,
        size,
        math.ceil(widest) + 2 * style.pad,
        len(lines) * round(size * style.line_h) + 2 * style.pad,
        cut,
    )


def fit_note(text: str, style: NoteStyle, max_w: int) -> tuple[list[str], int, int, int]:
    """Fit a note's text in a box at most max_w px wide. Returns (lines, size, w, h).

    The text gets the style's case, and lines are measured in pixels with the style's font, not
    counted in characters. From max_size down in steps of 2, the text is broken greedily to
    max_w - 2 * pad, and the first size that takes three lines or fewer, with every line fitting,
    is used. (A word wider than the box is the one way a line can not fit: it is never split, so
    only a smaller size can make it fit.) When even min_size needs more than three lines, it keeps
    three and the last one ends in an ellipsis (whole words are dropped until it fits, then
    characters). A word still too wide at min_size stays whole, and the box is as wide as it. The
    box is the widest line plus the padding on both sides, and the lines, each round(size * line_h)
    tall, plus the padding above and below.
    """
    return tuple(_fit(text, style, max_w)[:4])


def fit_note_cut(text: str, style: NoteStyle, max_w: int) -> tuple[list[str], int, int, int, bool]:
    """fit_note, and whether the text had to be cut as a fifth value: the planner's own answer (more
    than three lines at the smallest size), which does not depend on how the words are spaced."""
    return tuple(_fit(text, style, max_w))


@lru_cache(maxsize=1024)
def _wraps(text: str, style: NoteStyle, max_w: int, size: int) -> tuple[_Fit, ...]:
    """Every way to set a text at exactly this size without cutting a word, widest first (so
    fewest lines first). A tight spot needs a narrower wrap, which is one more line. fit_note
    gives the lines for a width, so the width goes down from max_w, and after each wrap it jumps
    to one px under that wrap's own width: every width in between would give the same lines. It
    stops when the text would have to be cut (a narrower width only cuts more) or when even the
    narrowest wrap overflows the width. Nothing at all when the text is cut even at max_w."""
    fixed = replace(style, max_size=size, min_size=size)
    wraps, width = [], max(max_w, 1)
    while width >= 1:
        fit = _fit(text, fixed, width)
        if fit.cut:
            break
        wraps.append(fit)
        if fit.w > width:
            break
        width = fit.w - 1
    return tuple(wraps)


# --- the face and the quadrants ---------------------------------------------------------------


def _default_face(width: int, height: int) -> tuple[int, int, int, int]:
    """The face box to use when there is nobody in the matte: a box around the middle top."""
    x0, y0, x1, y1 = _DEFAULT_FACE
    return round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)


def _ledge(
    widths: np.ndarray, left: np.ndarray, right: np.ndarray, first: int, last: int
) -> int | None:
    """The row, from `first` up to `last`, where a hood, a shoulder or an arm starts to stand out on
    one side of the head, as an index into the arrays, or None. widths, left and right are the width
    of each row of the person and its first and last column, from the top of the person down.

    It is found when, within the next _LEDGE_ROWS rows, the first or the last column gets more than
    _LEDGE of the head's width (the widest row so far) past the columns of all the rows above, and
    the person does not narrow to a neck (a row under _NECK_NARROWER of that width) within a head's
    height (_HEAD_USUALLY widths) below. The ledge starts at the first row that is _LEDGE_FROM of
    the way out to that reach: the rows before it are still the head, however far ahead the reach
    was seen."""
    run = np.maximum.accumulate(widths)
    out_left, out_right = np.minimum.accumulate(left), np.maximum.accumulate(right)
    for row in range(first, last):
        ahead, head = slice(row, row + _LEDGE_ROWS + 1), run[row - 1]
        past = (
            np.maximum(out_left[row - 1] - left[ahead], 0),
            np.maximum(right[ahead] - out_right[row - 1], 0),
        )
        side = max(past, key=np.max)
        below = widths[row : row + int(_HEAD_USUALLY * head)]
        if side.max() > _LEDGE * head and below.min() >= _NECK_NARROWER * head:
            return row + int(np.argmax(side >= _LEDGE_FROM * side.max()))
    return None


def face_box(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    """The part of the picture no note may cover, as (x0, y0, x1, y1) with x1 and y1 one past the
    last pixel; None when nobody is in the matte (plan_notes then uses a default box).

    It reaches from the top of the person down past the chin, where the neck is, and spans the
    head's columns with 4% of the width as air on either side. Everything is read from the width
    w(y) of the person in each row (alpha over 0.5), and from the first and last column of the
    person in it, inside person_boxes' bbox:

    - The head is the first hill of the width, read from the top. It ends at the first row (after
      15% of the person's height) that is more than 1.4 times as wide as the widest so far, which
      is raised arms or shoulders, or narrower than 85% of it, which is the jaw, or wider than
      1.15 times the median width of the rows just under the cap (the 10% of the person's height
      from the 15% mark), which is a hood or shoulders that widen the silhouette a little at a
      time, so that no row is much wider than the one before it. It ends too where a hood, a
      shoulder or an arm starts to stand out on one side: when, within the next 8 rows, the
      leftmost or the rightmost pixel of the person gets more than a quarter of the head's width
      (the widest row so far) past all the rows above, and the person does not narrow to a neck (a
      row under 85% of that width) within a head's height (1.45 widths) below, the head ends at the
      first row that is a tenth of the way out to that reach. That is a person turned three
      quarters: the hood can start as an island that adds little to a width, and it can come while
      the rows just under the cap are not over, so that their median is the hood's and the width no
      longer tells it from the head. With a neck showing no edge is taken for a hood, and a hand at
      the cheek or a brim is left to the other rules. Its widest row is the widest of the rows above
      that (and in the top 45% of the bbox), so hood, arms and shoulders cannot pass for the head
      however high the person sits in the frame.
    - The sharp widening that follows, a row over 1.4 times the head's width (shoulders or arms),
      ends the search for a neck. The neck is the narrowest row between the head's widest row and
      there (and 80% of the bbox). It counts only when it is narrower than 85% of the widest row,
      some row below it is at least 15% wider than it (the shoulders), and it is at least 1.3
      head widths below the top: a head is taller than wide, so a narrow row higher up is a cheek
      or a collar, where a hood starts to widen the silhouette on one side, and not a neck.
    - The face ends at the neck. Without a neck it ends 1.45 head widths below the top (a head's
      usual proportions), and never below the person. When the person touches the top of the
      stage the top of the head is out of the picture, and is taken to be 0.63 head widths above
      its widest row. The box adds 3% of the stage's height of air below that, or 5% when the
      head touches a side of the stage, which hides part of its width.
    - Its columns are the head's: the leftmost and rightmost of the person in the rows from the top
      down to the head's widest row, where only the head is, and the middle of them is the head's
      center. Below that row, down to the narrowest, a row counts only as far as it stays within the
      head's width and its air (4% of the width) around that center, so that a hood or a shoulder on
      one side of a head, or the shoulders below it, do not widen the box. They are clamped to the
      stage.
    """
    bbox = person_boxes(alpha)["bbox"]
    if bbox is None:
        return None
    height, width = alpha.shape
    mask = np.asarray(alpha) > 0.5
    x0, y0, x1, y1 = bbox
    person_h = y1 - y0
    row_w = mask.sum(axis=1)

    after = y0 + _HEAD_MIN_ROWS * person_h
    # How far out the person reaches in each row, as its first and last column: a hood or an arm
    # starts as an island that adds little to the width of a row, but its edge stands out at once.
    last = min(y0 + int(_WIDEST_IN * person_h) + 1, y1)
    seen = mask[y0:y1]
    present = seen.any(axis=1)
    reach_left = np.where(present, seen.argmax(axis=1), width)
    reach_right = np.where(present, width - 1 - seen[:, ::-1].argmax(axis=1), -1)
    ledge = _ledge(row_w[y0:y1], reach_left, reach_right, math.ceil(after) - y0, last - y0)
    ledge_at = y1 if ledge is None else y0 + ledge

    # What the head's width is, read from the rows just under the cap. A hood that widens the
    # silhouette a little at a time never has a row much wider than the last, so the width so far
    # alone would follow it down over the shoulders.
    first = math.ceil(after)
    under_cap = row_w[first : first + max(int(_PLATEAU_ROWS * person_h), 1)]
    plateau = float(np.median(under_cap)) if len(under_cap) else math.inf
    widest = y0
    for y in range(y0 + 1, last):
        run = row_w[widest]
        if y >= after and not _NECK_NARROWER * run <= row_w[y] <= _SHARP_WIDENING * run:
            break
        if y >= after and row_w[y] > _HOOD_GROWTH * plateau:
            break
        if y >= ledge_at:
            break
        if row_w[y] > run:
            widest = y
    head_w = int(row_w[widest])

    start = max(widest + 1, math.ceil(after))
    wider = np.flatnonzero(row_w[start:y1] > _SHARP_WIDENING * head_w)
    wide = start + int(wider[0]) if len(wider) else y1
    rows = np.arange(widest, max(min(wide, int(y0 + _NECK_BY * person_h), y1 - 1), widest) + 1)
    narrowest = int(rows[np.argmin(row_w[rows])])

    top = y0 if y0 > 0 else widest - _HEAD_ABOVE * head_w
    neck = (
        row_w[narrowest] < _NECK_NARROWER * head_w
        and bool((row_w[narrowest + 1 : y1] >= _SHOULDERS_WIDER * row_w[narrowest]).any())
        and narrowest >= top + _HEAD_AT_LEAST * head_w
    )
    bottom = narrowest if neck else min(top + _HEAD_USUALLY * head_w, y1)

    # The head's columns are those of the rows down to its widest. The rows below that, down to the
    # narrowest, count only inside the head's width and its air around the head's center.
    pad = round(_FACE_PAD * width)
    head = np.flatnonzero(mask[y0 : widest + 1].any(axis=0))
    low, high = max(int(head[0]) - pad, 0), int(head[-1]) + pad + 1
    below = low + np.flatnonzero(mask[widest + 1 : narrowest + 1, low:high].any(axis=0))
    columns = np.union1d(head, below)
    air = _FACE_AIR_EDGE if columns[0] == 0 or columns[-1] == width - 1 else _FACE_AIR
    return (
        max(int(columns[0]) - pad, 0),
        y0,
        min(int(columns[-1]) + 1 + pad, width),
        min(height, round(bottom + air * height)),
    )


def quadrants(
    stage_w: int,
    stage_h: int,
    head: tuple[int, int, int, int] | None,
    margin: int,
    rows: tuple[int, int] | None = None,
) -> dict[str, tuple[int, int, int, int]]:
    """The room each note has, as (x0, y0, x1, y1): the stage cut in four at the head's center
    (the middle of the stage when there is no head) and at half the height, with `margin` px kept
    clear of the stage's edge and of the cut itself. plan_notes passes the face box as the head.

    rows = (top, bottom) keeps the quadrants between those two rows of the stage instead of
    between the margins, and cuts them in the middle of those rows: plan_notes uses it so that on
    a tall stage the notes stay within reach of a person who is not as tall as the stage."""
    cx = (head[0] + head[2]) // 2 if head else stage_w // 2
    top, bottom = rows or (margin, stage_h - margin)
    mid = (top + bottom) // 2 if rows else stage_h // 2
    return {
        "tl": (margin, top, cx - margin, mid),
        "tr": (cx + margin, top, stage_w - margin, mid),
        "bl": (margin, mid, cx - margin, bottom),
        "br": (cx + margin, mid, stage_w - margin, bottom),
    }


def _middle(box: tuple[int, int, int, int]) -> tuple[int, int]:
    """The middle of an (x0, y0, x1, y1) box, in whole pixels."""
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def _stage_scale(height: int) -> float:
    """How much longer than usual the arrows may be on a stage this tall: in proportion to its
    height over the usual 608 px, between 1 and 2."""
    return min(max(height / _CHORD_STAGE, 1), _CHORD_SCALE_MAX)


def _free_rows(
    keep: list[tuple[int, int, int, int]], width: int, height: int, top: int, bottom: int
) -> tuple[int, int]:
    """The rows of the stage the notes may use, as (top, bottom), given the rows they could use
    without keep-out boxes: a box as wide as the stage that starts at its top edge (the title room
    of a 9:16 card) pushes the top down to its bottom, and one that ends at its bottom edge
    (the bottom band of the Reels safe area) pulls the bottom up to its top. A box that spans the
    whole stage is no band, and when the bands would leave no rows at all nothing moves."""
    new_top, new_bottom = top, bottom
    for x0, y0, x1, y1 in keep:
        if x0 > 0 or x1 < width:
            continue
        if y0 <= 0 and y1 < height:
            new_top = max(new_top, y1)
        elif y1 >= height and y0 > 0:
            new_bottom = min(new_bottom, y0)
    return (new_top, new_bottom) if new_top < new_bottom else (top, bottom)


def _free_sides(
    quads: dict[str, tuple[int, int, int, int]], keep: list[tuple[int, int, int, int]], width: int
) -> dict[str, tuple[int, int, int, int]]:
    """The quadrants with the keep-out boxes that stand at a side of the stage cut away from them: a
    box that reaches the right edge (the rail of buttons of the Reels safe area) pulls the right
    side of every quadrant whose rows it covers back to its own left side, and one that reaches the
    left edge pushes the left side out to its right side. A box as wide as the stage is a band, which
    _free_rows has dealt with, and a box that is not at a side is left to the price of the spots."""
    cut = {}
    for key, (x0, y0, x1, y1) in quads.items():
        for kx0, ky0, kx1, ky1 in keep:
            if ky0 >= y1 or ky1 <= y0 or (kx0 <= 0 and kx1 >= width):
                continue
            if kx1 >= width and x0 < kx0 < x1:
                x1 = kx0
            elif kx0 <= 0 and x0 < kx1 < x1:
                x0 = kx1
        cut[key] = (x0, y0, x1, y1)
    return cut


# --- the matte, prepared ----------------------------------------------------------------------


class _Prices(dict):
    """What a MatteContext remembers of the prices of boxes, so that the next layout of the same
    matte does not work them out again. It forgets all of them once they hold more than
    _CACHE_BYTES, so a long session on a big matte cannot make it grow without end."""

    held = 0  # bytes held

    def keep(self, token: tuple, found: tuple) -> None:
        """Remember `found` under `token`, first forgetting everything if it would not fit."""
        size = sum(part.nbytes for part in found)
        if self.held + size > _CACHE_BYTES:
            self.clear()
            self.held = 0
        self[token] = found
        self.held += size


@dataclass(frozen=True, eq=False)
class MatteContext:
    """What plan_notes works out from the matte alone, so that it is done once for a picture and
    not again for every layout of it (every drag of a note in the studio). Make one with
    prepare_matte and give it to plan_notes as ctx.

    shape is (height, width), band and arrows are what the style it was made for had. face is the
    face box (the default one when nobody is there), bbox the person's box (None when nobody is
    there), quads the quadrants, and margin and max_w the gap to the stage's edge and the widest a
    note may be. person is the matte over 0.5, edge and near are summed-area tables (see
    _summed_area) of the sticker edge and of the near mask. The face is a rectangle, so it needs
    no table. grown is the person grown by band + 8 px, border its border as an n x 2 array of
    (x, y), and aims the bbox center and the face center, in that order (just the face center
    with nobody). near, grown and border are None without arrows or without a person. keepout is
    a tuple of (x0, y0, x1, y1) boxes in whole stage pixels that no note may cover, priced like the
    face (see prepare_matte); it is empty unless prepare_matte was given some. cache keeps the
    prices of the boxes already tried, for the next call.
    """

    shape: tuple[int, int]
    band: int
    arrows: bool
    bbox: tuple[int, int, int, int] | None
    face: tuple[int, int, int, int]
    margin: int
    max_w: int
    quads: dict[str, tuple[int, int, int, int]]
    person: np.ndarray
    edge: np.ndarray
    near: np.ndarray | None
    grown: np.ndarray | None
    border: np.ndarray | None
    aims: tuple[tuple[int, int], ...]
    keepout: tuple[tuple[int, int, int, int], ...] = ()
    cache: _Prices = field(default_factory=_Prices, repr=False)


def _summed_area(mask: np.ndarray) -> np.ndarray:
    """A mask as a summed-area table. Entry [y, x] of the table counts the set pixels above row y
    and left of column x, so the count under any box takes four lookups."""
    table = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.int64)
    table[1:, 1:] = mask.cumsum(axis=0, dtype=np.int64).cumsum(axis=1)
    return table


def _near_mask(alpha: np.ndarray, radius: float) -> np.ndarray:
    """dilate(alpha, radius) > 0.5, found at a quarter of the size and enlarged again, to within a
    few px. It is only the soft cost that pushes notes off the body, and the dilation is the
    slowest step, which costs a sixteenth of the time at that size."""
    height, width = alpha.shape
    rows, columns = -(-height // _NEAR_SHRINK), -(-width // _NEAR_SHRINK)
    padded = np.pad(
        alpha, ((0, rows * _NEAR_SHRINK - height), (0, columns * _NEAR_SHRINK - width)), mode="edge"
    )
    small = padded.reshape(rows, _NEAR_SHRINK, columns, _NEAR_SHRINK).mean(axis=(1, 3))
    grown = dilate(small, radius / _NEAR_SHRINK) > 0.5
    return grown.repeat(_NEAR_SHRINK, axis=0).repeat(_NEAR_SHRINK, axis=1)[:height, :width]


def prepare_matte(
    alpha: np.ndarray,
    style: NoteStyle,
    keepout: list[tuple[int, int, int, int]] | None = None,
) -> MatteContext:
    """Everything plan_notes needs from the matte alone (height x width, 0-1) for a style: the
    face box, the person's box, the quadrants, the sticker edge (dilated by style.band) and, with
    arrows and somebody in the picture, the person grown by band + 8 px with its border, and the
    near mask (grown by band + 8 + 56, at a quarter of the size). Those are the dilations, and the
    only ones a layout needs, which is why a context saves most of its time.

    keepout is a list of (x0, y0, x1, y1) boxes in stage pixels, x1 and y1 one past the last pixel,
    that no note may cover: the title room and the safe area of a 9:16 card (card._keepout), and the
    room a hero element keeps. plan_notes prices them as it prices the face (a note
    that can keep off them does, whatever else that costs, and one that cannot is flagged as a
    clash), and an arrow's curve may not enter them. The boxes are rounded to whole pixels. None,
    or an empty list, is no keep-out boxes, and then the layout is exactly what it was without the
    argument. A box of other than four numbers raises ValueError.

    The quadrants are planned inside the room the boxes leave, not only priced off them: a box as
    wide as the stage at its top or bottom edge (the title room, the bottom band of a 9:16 card's
    safe area) moves the top or bottom rows of the quadrants to its edge, and a box at the
    left or right edge (the rail of buttons) cuts the quadrants that its rows cross back to its side.
    A box that is anywhere else, and a box over the whole stage, only costs the spots it covers."""
    keep = []
    for box in keepout or ():
        if len(box) != 4:
            raise ValueError(f"keepout boxes are (x0, y0, x1, y1), got {box!r}")
        keep.append(tuple(round(value) for value in box))
    height, width = alpha.shape
    bbox = person_boxes(alpha)["bbox"]
    face = face_box(alpha) or _default_face(width, height)
    margin = round(_MARGIN * width)
    person = alpha > 0.5
    edge = _summed_area((dilate(alpha, style.band) > 0.5) & ~person)

    # The quadrants stay within an arrow's longest reach of the person, up and down: on a stage
    # taller than the person that keeps the notes near enough for their arrows. (On the usual
    # stage, where the person fills the height, this changes nothing, and a style with no arrows
    # has no reach to keep within: its quadrants are the whole stage's.)
    rows = None
    if bbox is not None and style.arrows:
        reach = _CHORD[1] * _stage_scale(height)
        rows = (max(margin, round(bbox[1] - reach)), min(height - margin, round(bbox[3] + reach)))
    # The rows also stay inside the bands the keep-out boxes leave, and the quadrants out of the boxes
    # at a side: planned inside the room there is, a note is not pushed out of a box it was put in.
    usual = rows or (margin, height - margin)
    free = _free_rows(keep, width, height, *usual)
    if free != usual:
        rows = free
    quads = _free_sides(quadrants(width, height, face, margin, rows), keep, width)

    grown = near = border = None
    aims = (_middle(face),)
    if style.arrows and bbox is not None:
        grown = dilate(alpha, style.band + _TARGET_GAP) > 0.5
        near = _summed_area(_near_mask(alpha, style.band + _TARGET_GAP + _ROOM))
        border = _border(grown)
        aims = (_middle(bbox), _middle(face))
    return MatteContext(
        (height, width), style.band, style.arrows, bbox, face, margin, round(_MAX_W * width),
        quads, person, edge, near, grown, border, aims, tuple(keep),
    )


# --- placing the notes ------------------------------------------------------------------------


def _clash(
    xs: np.ndarray, ys: np.ndarray, size: tuple[int, int], box: tuple[int, int, int, int]
) -> np.ndarray:
    """The pixels a (w, h) box shares with `box` at every spot (rows of ys by columns of xs). A
    rectangle's overlap is two overlaps of intervals multiplied."""
    w, h = size
    over_x = np.clip(np.minimum(xs + w, box[2]) - np.maximum(xs, box[0]), 0, None)
    over_y = np.clip(np.minimum(ys + h, box[3]) - np.maximum(ys, box[1]), 0, None)
    return over_y[:, None] * over_x[None, :]


def _shared(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """The pixels two (x0, y0, x1, y1) boxes share."""
    return max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)


def _apart(box: tuple[int, int, int, int], size: int) -> tuple[int, int, int, int]:
    """A note's box with the space the other notes keep from it: _NOTE_GAP px at the sides, and above
    and below _STACK_GAP of the font size, rounded up. Notes that share a column, one above the other
    in a side margin or a quadrant, have to be that far apart, so that they do not read as one
    paragraph; and a note that shares a pixel with this box is too close to it."""
    x0, y0, x1, y1 = box
    gap = math.ceil(_STACK_GAP * size)
    return x0 - _NOTE_GAP, y0 - gap, x1 + _NOTE_GAP, y1 + gap


def _pinned(
    position: list[float], size: tuple[int, int], stage: tuple[int, int]
) -> tuple[int, int]:
    """A top-left corner [x, y] from the project file, rounded to whole pixels and moved onto the
    stage if the box would stick out of it."""
    x, y = position
    (w, h), (width, height) = size, stage
    return min(max(round(x), 0), max(width - w, 0)), min(max(round(y), 0), max(height - h, 0))


class _Planner:
    """The search for one card: the size its notes share, and where each one goes.

    A note is "clean" when its whole text is set (nothing cut), and its box shares no pixel with
    the face box, a keep-out box or the space around another note (_apart). All the notes get one
    size, the biggest of the style's sizes at which they can all be set clean: sizes are bisected
    (each size is tried with the notes in the default order, tl tr bl br, and then in the other
    orders, since which note is placed first decides what room is left), first with every note
    kept in its own quadrant, and only if no size does with notes allowed to leave their quadrant
    for any room inside the margins. Where nothing is clean at any size, the arrangement with the
    least overlap wins, and there a note may also sit flush against the notes before it.
    """

    def __init__(self, ctx: MatteContext, style: NoteStyle, texts: dict, positions: dict | None):
        self.ctx, self.style = ctx, style
        self.sizes = _sizes(style)
        self.keys = [key for key in ctx.quads if (texts.get(key) or "").strip()]
        self.texts = {key: texts[key] for key in self.keys}
        positions = positions or {}
        self.pins = {key: positions[key] for key in self.keys if positions.get(key) is not None}
        self.free = [key for key in self.keys if key not in self.pins]
        height, width = ctx.shape
        self.inner = (ctx.margin, ctx.margin, width - ctx.margin, height - ctx.margin)
        self.whole = (0, 0, width, height)
        # A text too long for three lines even at the smallest size is cut there, whatever the
        # card's size: it is the one case where words may be lost
        self.cut = {}
        smallest = replace(style, max_size=self.sizes[-1], min_size=self.sizes[-1])
        for key in self.keys:
            if not _wraps(self.texts[key], style, ctx.max_w, self.sizes[-1]):
                self.cut[key] = _fit(self.texts[key], smallest, ctx.max_w)

    def wraps(self, key: str, size: int) -> tuple[_Fit, ...]:
        """The ways to set a note at a size (see _wraps), or its one cut way."""
        if key in self.cut:
            return (self.cut[key],)
        return _wraps(self.texts[key], self.style, self.ctx.max_w, size)

    def spots(self, key: str, fit: _Fit, field_: tuple[int, int, int, int]) -> tuple:
        """What a box costs at every spot of a field, apart from the other notes: (xs, ys, hard,
        soft), with hard the pixels on the face box and on the keep-out boxes, and soft the sticker
        edge, the near mask and the distance from the quadrant's corner. These do not depend on
        the text or on the other notes, so the context keeps them for the next call."""
        ctx = self.ctx
        use_near = self.style.arrows and ctx.near is not None
        token = (key, fit.w, fit.h, field_, use_near)
        found = ctx.cache.get(token)
        if found is not None:
            return found
        fx0, fy0, fx1, fy1 = field_
        xs = np.arange(fx0, max(fx1 - fit.w, fx0) + 1, _STEP)
        ys = np.arange(fy0, max(fy1 - fit.h, fy0) + 1, _STEP)
        found = self.prices(key, fit, xs, ys)
        ctx.cache.keep(token, found)
        return found

    def flush_spots(
        self, key: str, fit: _Fit, field_: tuple[int, int, int, int], notes: list, size: int
    ) -> tuple:
        """spots, with the spots flush against each of the `notes` (touching it, or at the distance
        notes keep at the font `size`) and against the far edges of the field added to the grid. On
        the 8 px grid alone a row of notes that fits only flush does not fit at all. Not kept: these
        depend on the other notes."""
        xs, ys, _, _ = self.spots(key, fit, field_)
        fx0, fy0, fx1, fy1 = field_
        w, h = fit.w, fit.h
        gap = math.ceil(_STACK_GAP * size) if size else 0
        more_x, more_y = [fx1 - w], [fy1 - h]
        for x0, y0, x1, y1 in notes:
            more_x += [x1, x1 + _NOTE_GAP, x0 - w, x0 - w - _NOTE_GAP]
            more_y += [y1, y1 + gap, y0 - h, y0 - h - gap]
        more_x = [x for x in more_x if fx0 <= x <= max(fx1 - w, fx0)]
        more_y = [y for y in more_y if fy0 <= y <= max(fy1 - h, fy0)]
        xs = np.union1d(xs, np.array(more_x, dtype=xs.dtype))
        ys = np.union1d(ys, np.array(more_y, dtype=ys.dtype))
        return self.prices(key, fit, xs, ys)

    def prices(self, key: str, fit: _Fit, xs: np.ndarray, ys: np.ndarray) -> tuple:
        """What a box costs at the spots of columns xs by rows ys: (xs, ys, hard, soft), as spots
        says."""
        ctx = self.ctx
        use_near = self.style.arrows and ctx.near is not None
        height, width = ctx.shape
        w, h = fit.w, fit.h
        qx0, qy0, qx1, qy1 = ctx.quads[key]
        left, top = key[1] == "l", key[0] == "t"
        x1, y1 = np.minimum(xs + w, width), np.minimum(ys + h, height)

        def under(table: np.ndarray) -> np.ndarray:
            sums = table[np.ix_(y1, x1)] - table[np.ix_(ys, x1)]
            return sums - table[np.ix_(y1, xs)] + table[np.ix_(ys, xs)]

        away_x = (xs if left else xs + w) - (qx0 if left else qx1)
        away_y = (ys if top else ys + h) - (qy0 if top else qy1)
        soft = _EDGE_COST * under(ctx.edge)
        soft = soft + _DISTANCE_COST * np.hypot(away_x[None, :], away_y[:, None])
        if use_near:
            soft = soft + _NEAR_COST * under(ctx.near)
        hard = _clash(xs, ys, (w, h), ctx.face)
        for box in ctx.keepout:
            hard = hard + _clash(xs, ys, (w, h), box)
        return xs, ys, hard, soft

    def search(
        self,
        key: str,
        fit: _Fit,
        field_: tuple[int, int, int, int],
        notes: list[tuple[int, int, int, int]],
        size: int = 0,
        flush: bool = False,
    ) -> tuple[int, int, int]:
        """The best spot for a box in a field, as (x, y, clash), among the spots of `spots`, or of
        `flush_spots` when `flush`. The clash of a spot counts, first,
        the pixels the box would share with the face, the keep-out boxes and the `notes` (_CLASH_COST
        each), and then the pixels it would share with the space notes keep around the `notes`
        (_apart) at the card's font `size`, which is none when it is 0, as for the box of a hero
        rule. A spot that overlaps is worse than one that only crowds, and a spot with neither
        beats any spot with either. It is what counts first, and then come the sticker edge, the
        near mask and the distance, the soft costs. Spots are 8 px apart, and the first of equal
        prices (the topmost, then the leftmost) wins."""
        if flush:
            xs, ys, clash, soft = self.flush_spots(key, fit, field_, notes, size)
        else:
            xs, ys, clash, soft = self.spots(key, fit, field_)
        crowd = np.zeros_like(clash)
        for box in notes:
            over = _clash(xs, ys, (fit.w, fit.h), box)
            clash = clash + over
            if size:
                crowd = crowd + _clash(xs, ys, (fit.w, fit.h), _apart(box, size)) - over
        least = clash == clash.min()
        least &= crowd == crowd[least].min()
        row, column = np.unravel_index(np.argmin(np.where(least, soft, np.inf)), clash.shape)
        clashes = int(clash[row, column]) * _CLASH_COST + int(crowd[row, column])
        return int(xs[column]), int(ys[row]), clashes

    def fields(self, key: str, strict: bool) -> tuple[tuple[int, int, int, int], ...]:
        """Where a note may go: its own quadrant, and unless strict, anywhere inside the margins."""
        quad = self.ctx.quads[key]
        return (quad,) if strict else (quad, self.inner)

    def pins_at(self, wraps: dict) -> tuple[dict, list]:
        """The pinned notes at a size, set as wide as they can be: ({key: (fit, (x, y))}, boxes)."""
        placed, boxes = {}, []
        for key, position in self.pins.items():
            fit = wraps[key][0]
            x, y = _pinned(position, (fit.w, fit.h), self.whole[2:])
            placed[key] = (fit, (x, y))
            boxes.append((x, y, x + fit.w, y + fit.h))
        return placed, boxes

    def first_clean(
        self, key: str, fits: tuple, fields: tuple, notes: list, size: int
    ) -> tuple | None:
        """The first way of setting a note, widest first, that has a clean spot in the first field
        that has one, with the notes keeping their distance at the font `size`: (fit, (x, y)), or
        None."""
        for field_ in fields:
            for fit in fits:
                if fit.w > field_[2] - field_[0] or fit.h > field_[3] - field_[1]:
                    continue
                x, y, clash = self.search(key, fit, field_, notes, size)
                if clash == 0:
                    return fit, (x, y)
        return None

    def try_size(self, size: int, strict: bool) -> dict | None:
        """A clean arrangement of all the notes at one size, or None. Free notes are placed in
        the default order and then in every other, until one works. When strict, the quadrants
        are apart, so the order cannot matter and only the default one is tried."""
        wraps = {key: self.wraps(key, size) for key in self.keys}
        if not all(wraps.values()):
            return None
        fixed, notes = self.pins_at(wraps)
        for order in itertools.permutations(self.free):
            placed, taken = dict(fixed), list(notes)
            for key in order:
                found = self.first_clean(key, wraps[key], self.fields(key, strict), taken, size)
                if found is None:
                    break
                fit, (x, y) = found
                placed[key] = found
                taken.append((x, y, x + fit.w, y + fit.h))
            else:
                return placed
            if strict:
                return None
        return None

    def largest_clean(self, strict: bool) -> dict | None:
        """The clean arrangement at the biggest size that has one, found by bisection (a smaller
        size is never harder than a bigger one), or None when not even the smallest does."""
        last = len(self.sizes) - 1
        best = self.try_size(self.sizes[last], strict)
        if best is None:
            return None
        low, high = 0, last  # sizes[high] is clean
        while low < high:
            mid = (low + high) // 2
            found = self.try_size(self.sizes[mid], strict)
            if found is not None:
                best, high = found, mid
            else:
                low = mid + 1
        return best

    def place_all(
        self, order: tuple, wraps: dict, fixed: dict, notes: list, bound: int | float, size: int
    ) -> tuple[dict | None, int, int]:
        """Every free note in an order, each on its least clashing spot (its quadrant first, then
        anywhere inside the margins, then anywhere on the stage, widest way first), as (placed,
        total clash, notes outside their quadrant), the clash being what search counts, overlap
        first and then crowding. Gives up, with None, once the total passes `bound`."""
        placed, taken, total, outside = dict(fixed), list(notes), 0, 0
        for key in order:
            pick = None
            for field_ in (*self.fields(key, False), self.whole):
                for fit in wraps[key]:
                    if field_ is not self.whole and (
                        fit.w > field_[2] - field_[0] or fit.h > field_[3] - field_[1]
                    ):
                        continue
                    x, y, clash = self.search(key, fit, field_, taken, size, flush=True)
                    if pick is None or clash < pick[0]:
                        pick = (clash, fit, x, y)
                    if clash == 0:
                        break
                if pick is not None and pick[0] == 0:
                    break
            clash, fit, x, y = pick
            total += clash
            if total > bound:
                return None, total, outside
            qx0, qy0, qx1, qy1 = self.ctx.quads[key]
            outside += not (qx0 <= x and qy0 <= y and x + fit.w <= qx1 and y + fit.h <= qy1)
            placed[key] = (fit, (x, y))
            taken.append((x, y, x + fit.w, y + fit.h))
        return placed, total, outside

    def sweep(self, crowd: bool, ceiling: int | float) -> tuple[dict | None, tuple]:
        """The arrangement with the least clash of every size and order (place_all), as (placed,
        (total clash, notes outside their quadrant)), with the crowding counted when `crowd`, and
        only among those whose total is not over `ceiling`. None when there are none."""
        best, bound = None, (ceiling, math.inf)
        for size in self.sizes:
            wraps = {key: self.wraps(key, size) for key in self.keys}
            if not all(wraps.values()):
                continue
            fixed, notes = self.pins_at(wraps)
            for order in itertools.permutations(self.free):
                placed, total, outside = self.place_all(
                    order, wraps, fixed, notes, bound[0], size if crowd else 0
                )
                if placed is not None and (total, outside) < bound:
                    best, bound = placed, (total, outside)
        return best, bound

    def least_clash(self) -> dict:
        """When no size is clean: of all the sizes and orders, the arrangement with the least
        overlap between notes and the face, the keep-out boxes and each other (px), then the least
        crowding (the pixels the notes share with the space they keep between them: a note that
        overlaps another is worse than one that is too close to it), then the fewest notes outside
        their quadrant, then the biggest size. Here the notes are still set whole, and only a text
        that is cut anyway is not.

        Each note takes the spot that overlaps least, and then crowds least, given the notes before
        it, and a note that keeps its distance can take the room a later one needed. So the search
        is made first without the crowding, and then with it among the arrangements that overlap no
        more than the first found: one of those is the answer, when there is one."""
        plain, (overlap, _) = self.sweep(False, math.inf)
        kept, _ = self.sweep(True, overlap + _CLASH_COST - 1)
        return plain if kept is None else kept

    def plan(self) -> dict:
        """Where every note goes and how it is set: {key: (fit, (x, y))}, in the order of keys."""
        if not self.keys:
            return {}
        placed = self.largest_clean(True) or self.largest_clean(False) or self.least_clash()
        return {key: placed[key] for key in self.keys}


def plan_notes(
    alpha: np.ndarray,
    texts: dict[str, str],
    style: NoteStyle,
    positions: dict | None = None,
    ctx: MatteContext | None = None,
) -> dict[str, NoteBox]:
    """Lay out the notes of a card around the person in `alpha` (height x width, 0-1).

    texts holds the note of each corner under "tl", "tr", "bl" and "br"; a corner with no text
    gets no box. All the notes share one font size: the biggest at which every note is set whole
    and put on a spot clear of the face (face_box, or a box in the middle top when nobody is
    there) and of the other notes, each in its own quadrant if the notes can all be kept in their
    corners, and else anywhere inside the margins. The quadrants are cut at the face's center,
    with a margin of 4% of the stage's width, and a note is at most 42% of the stage's width
    wide. A text is cut (ending in an ellipsis) only when it is too long for three lines at the
    style's smallest size. Notes keep their distance: 8 px at their sides, and when one stands
    above another (they share columns, in a side margin or a quadrant) ceil(0.4 x the font size)
    px between them, so that they do not read as one paragraph; a size at which they cannot is not
    clean. See _Planner for how the size and the places are found, and for what happens when no
    size is clean: the notes then overlap, or crowd each other, as little as they can.

    Among the clean spots a note takes the cheapest: off the sticker edge, away from the body
    (when the style has arrows, to leave room for them) and near its own corner.

    positions maps a key to [x, y], the top-left corner of its box, and pins that note there:
    rounded, moved onto the stage if it would stick out, and set at the card's size. The other
    notes keep off it. A None value, and a key that is not a note ("burst"), are ignored, as in
    the project file.

    With style.arrows and somebody in the matte, each note gets an arrow (start, control, end)
    once all the notes are placed, or None when no arrow is valid for it (see _arrow). Each
    NoteBox has clash set when it is cut, overlaps the face or a keep-out box, overlaps a note or
    is closer to it than that, or sits outside its quadrant. A text with a character that its font
    can't draw (glyphs.py: a letter of another alphabet, a star) is cut too, however much room it
    has, so that it is flagged: the font draws a gap there, and the person has to be told.

    The boxes prepare_matte was given to keep notes out of (ctx.keepout) are avoided like the
    face, and a note on one is flagged. An arrow does not enter one either.

    The slow work that depends on the matte alone (three dilations, the sticker edge, the face
    box) is done by prepare_matte. Pass its result as ctx to lay out the same matte again, as when
    a note is dragged, without any of it: it must have been made for this matte's size and this
    style's band, and for arrows if the style has them. Without a ctx, one is made for the call.
    """
    if ctx is None:
        ctx = prepare_matte(alpha, style)
    elif ctx.shape != alpha.shape:
        raise ValueError(f"ctx is for a matte of {ctx.shape}, not {alpha.shape}")
    elif ctx.band != style.band or (style.arrows and not ctx.arrows):
        raise ValueError("ctx was prepared for another band, or without the arrows this style has")

    placed = _Planner(ctx, style, texts, positions).plan()
    rects = {key: (x, y, x + fit.w, y + fit.h) for key, (fit, (x, y)) in placed.items()}

    size = max((fit.size for fit, _ in placed.values()), default=0)
    boxes: dict[str, NoteBox] = {}
    for key, (fit, (x, y)) in placed.items():
        box = rects[key]
        arrow = None
        if style.arrows and ctx.grown is not None:
            notes = list(rects.values())
            arrow = _arrow(key, box, notes, ctx.person, ctx.grown, ctx.border, ctx.aims, ctx.keepout)
        qx0, qy0, qx1, qy1 = ctx.quads[key]
        # a text with a character the font can't draw is not set whole, whatever the room: it is cut
        cut = fit.cut or bool(missing_glyphs(style.font, texts[key], style.case))
        clash = (
            cut
            or _shared(box, ctx.face) > 0
            or any(_shared(box, kept) > 0 for kept in ctx.keepout)
            or any(_shared(box, _apart(other, size)) > 0 for k, other in rects.items() if k != key)
            or not (qx0 <= box[0] and qy0 <= box[1] and box[2] <= qx1 and box[3] <= qy1)
        )
        boxes[key] = NoteBox(key, fit.lines, fit.size, x, y, fit.w, fit.h, arrow, clash, cut)
    return boxes


# --- what a hero rule asks of the layout ------------------------------------------------------


def _rect_of(box) -> tuple[int, int, int, int]:
    """A NoteBox, or an (x0, y0, x1, y1) tuple, as (x0, y0, x1, y1)."""
    if isinstance(box, NoteBox):
        return box.x, box.y, box.x + box.w, box.y + box.h
    return tuple(box)


def clash_of(box, ctx: MatteContext, others, stage_size: tuple[int, int]) -> bool:
    """Whether a box is in the way: True when it shares a pixel with the face box, a keep-out box
    or any box in `others`, or when it leaves the stage. This is what a hero rule asks after it
    moves a box (it keeps the planner's cut flag on its own). box and the boxes of `others` are
    NoteBoxes or (x0, y0, x1, y1) tuples, `others` does not hold the box itself (the very same
    object is skipped if it does) and stage_size is the stage's (width, height).
    """
    x0, y0, x1, y1 = rect = _rect_of(box)
    width, height = stage_size
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return True
    return (
        _shared(rect, ctx.face) > 0
        or any(_shared(rect, kept) > 0 for kept in ctx.keepout)
        or any(_shared(rect, _rect_of(other)) > 0 for other in others if other is not box)
    )


def arrow_for(key: str, box, others, ctx: MatteContext):
    """The arrow of the note `key` at `box`, found as plan_notes finds one, among the boxes of
    `others` (the rest of the layout, hero boxes included): (start, control, end), or None when the
    style has no arrows, nobody is in the picture or no arrow is valid. A hero rule that moves or
    adds boxes finds the arrows of the notes again with it."""
    if not ctx.arrows or ctx.grown is None:
        return None
    rect = _rect_of(box)
    rects = [rect, *(_rect_of(other) for other in others)]
    return _arrow(key, rect, rects, ctx.person, ctx.grown, ctx.border, ctx.aims, ctx.keepout)


def place_box(
    key: str, size: tuple[int, int], ctx: MatteContext, style: NoteStyle, others
) -> tuple[int, int, bool]:
    """Where a box of `size` (w, h) goes for the corner `key`, as (x, y, clean): the spot a note
    of that size would get from plan_notes, with the boxes of `others` as the notes already
    placed. It is the cheapest clean spot (off the face and the keep-out boxes, off the others,
    off the sticker edge, near the corner) of the corner's own quadrant, and else of anywhere
    inside the margins, and else of anywhere on the stage. When there is no clean spot at all
    it is the one that overlaps least, and clean is False. A hero rule uses it for a box that
    plan_notes did not size: the panel that takes a note's place, or the note a burst pushed.
    """
    width, height = size
    planner = _Planner(ctx, style, {}, None)
    fit = _Fit([], 0, width, height, False)
    taken = [_rect_of(other) for other in others]
    best = None  # (overlap, order of the field, x, y): the least overlap, then the first field
    for order, field_ in enumerate((*planner.fields(key, False), planner.whole)):
        if width > field_[2] - field_[0] or height > field_[3] - field_[1]:
            continue
        x, y, clash = planner.search(key, fit, field_, taken)
        if clash == 0:
            return x, y, True
        if best is None or (clash, order) < best[:2]:
            best = (clash, order, x, y)
    if best is None:  # bigger than the stage: held to the corner it is named for
        x = max(ctx.shape[1] - width, 0) if key[1] == "r" else 0
        y = max(ctx.shape[0] - height, 0) if key[0] == "b" else 0
        return x, y, False
    return best[2], best[3], False


def room_below(ctx: MatteContext) -> int:
    """The bottom of the stage as far as a box may go, from the keep-out boxes of the context: the
    top of the bottom band of a 9:16 card's safe area, or the stage's height when there is
    no such box (any other shape). A hero rule that puts a box at the bottom of the stage uses it."""
    height, width = ctx.shape
    return _free_rows(list(ctx.keepout), width, height, 0, height)[1]


def room_right(ctx: MatteContext, y0: int, y1: int) -> int:
    """The right edge of the stage as far as a box may go in the rows y0 to y1: the left side of the
    rail of buttons when a keep-out box at the right edge covers any of those rows, or the stage's
    width. A hero rule that puts a box at the right of the stage uses it."""
    height, width = ctx.shape
    return _free_sides({"row": (0, y0, width, y1)}, list(ctx.keepout), width)["row"][2]


# --- arrows -----------------------------------------------------------------------------------


def _gap(point: tuple[float, float], box: tuple[int, int, int, int]) -> float:
    """How far a point is from an (x0, y0, x1, y1) box: 0 on or inside it."""
    return math.hypot(
        max(box[0] - point[0], 0, point[0] - box[2]), max(box[1] - point[1], 0, point[1] - box[3])
    )


def _march(
    start: tuple[int, int], aim: tuple[float, float], grown: np.ndarray
) -> tuple[int, int] | None:
    """Where an arrow from `start` ends when it heads for `aim`: it walks toward it in steps of
    2 px, and the end is the last point before the grown person (dilate(alpha, band + 8) > 0.5),
    so it is just outside the sticker edge, on bare background. None when the walk reaches aim
    without meeting the person."""
    dx, dy = aim[0] - start[0], aim[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    steps = np.arange(_MARCH, length + 1e-9, _MARCH)
    if len(steps) == 0:
        return None
    xs = np.rint(start[0] + dx / length * steps).astype(int)
    ys = np.rint(start[1] + dy / length * steps).astype(int)
    met = grown[np.clip(ys, 0, grown.shape[0] - 1), np.clip(xs, 0, grown.shape[1] - 1)]
    if not met.any():
        return None
    first = int(np.argmax(met))
    return start if first == 0 else (int(xs[first - 1]), int(ys[first - 1]))


def _bow(
    start: tuple[int, int],
    end: tuple[int, int],
    box: tuple[int, int, int, int],
    stage_middle: tuple[float, float],
) -> tuple[int, int]:
    """The control point of an arrow's curve: the middle of the chord moved sideways by 18% of the
    chord's length. Both sides are tried, with the curve sampled at 16 points, and the side that
    keeps the curve farther from its own note wins (the larger smallest distance); on a tie, the
    side that points away from the center of the stage."""
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    best = None
    for side in (1, -1):
        nx, ny = -dy / length * side, dx / length * side
        control = (mx + nx * _BEND * length, my + ny * _BEND * length)
        clearance = min(_gap(point, box) for point in quad_points(start, control, end, _SAMPLES))
        away = nx * (mx - stage_middle[0]) + ny * (my - stage_middle[1])
        candidate = (round(clearance, 6), away, control)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return round(best[2][0]), round(best[2][1])


def _along(low: int, high: int) -> list[int]:
    """The positions every 16 px from low to high, with high itself as the last."""
    return sorted({*range(low, high + 1, _SIDE_STEP), high})


def _starts(key: str, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Where an arrow of a note may start. First the corner of the box that faces the person
    (bottom right for tl, bottom left for tr, top right for bl, top left for br), 10 px out along
    each axis. Then points every 16 px (the ends too) along the two sides that face the person
    (tl: right and bottom; tr: left and bottom; bl: right and top; br: left and top), 8 px out."""
    x0, y0, x1, y1 = box
    sx, sy = _FACING[key]
    side_x, side_y = (x1 if sx > 0 else x0), (y1 if sy > 0 else y0)
    return [
        (side_x + _OUT * sx, side_y + _OUT * sy),
        *((side_x + _SIDE_OUT * sx, y) for y in _along(y0, y1)),
        *((x, side_y + _SIDE_OUT * sy) for x in _along(x0, x1)),
    ]


def _clear(
    start: tuple[int, int],
    control: tuple[int, int],
    end: tuple[int, int],
    person: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    keepout: tuple[tuple[int, int, int, int], ...] = (),
) -> bool:
    """Whether an arrow's curve, sampled at 16 points, never enters the person (alpha over 0.5),
    never touches a note box (its own included) except within 6 px of its start, and never enters
    a keep-out box (not even at its start)."""
    height, width = person.shape
    for x, y in quad_points(start, control, end, _SAMPLES):
        if person[min(max(round(y), 0), height - 1), min(max(round(x), 0), width - 1)]:
            return False
        if math.dist((x, y), start) > _NEAR_START and any(_gap((x, y), box) == 0 for box in boxes):
            return False
        if any(_gap((x, y), kept) == 0 for kept in keepout):
            return False
    return True


def _arrow(
    key: str,
    box: tuple[int, int, int, int],
    boxes: list[tuple[int, int, int, int]],
    person: np.ndarray,
    grown: np.ndarray,
    border: np.ndarray | None,
    aims: tuple[tuple[int, int], ...],
    keepout: tuple[tuple[int, int, int, int], ...] = (),
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    """The arrow from a note to the person, as (start, control, end), or None.

    Every start of _starts that is on the stage and outside the grown person (dilate(alpha,
    band + 8) > 0.5) is tried. From each, _march heads for each of three targets: the aims (the
    person's bbox center, then the face box's center) and the nearest point of the grown person's
    border, and ends at the last point outside the grown person. The control point is _bow's.

    An arrow is valid when its chord is 28 to 180 px, its curve never enters the person or a
    keep-out box, and the curve keeps out of every note in `boxes`, its own included, except within
    6 px of its start.
    Of the valid arrows the one whose chord is nearest 90 px wins, and on a tie the one that ends
    nearer the face box's center (the last of the aims). With none valid, the note has no arrow.
    The chord limits (not the 28 px floor) and the 90 px grow with a stage taller than 608 px, in
    proportion to its height up to twice as much, so that on a 9:16 stage the notes, which
    sit farther from the person, still reach it.
    """
    height, width = grown.shape
    stage_middle = (width / 2, height / 2)
    scale = _stage_scale(height)
    longest, best_chord = _CHORD[1] * scale, _CHORD_BEST * scale
    best = None
    for start in _starts(key, box):
        if not (0 <= start[0] < width and 0 <= start[1] < height) or grown[start[1], start[0]]:
            continue
        targets = aims if border is None else [*aims, _nearest(border, start)]
        for aim in targets:
            end = _march(start, aim, grown)
            if end is None:
                continue
            chord = math.dist(start, end)
            if not _CHORD[0] <= chord <= longest:
                continue
            control = _bow(start, end, box, stage_middle)
            if not _clear(start, control, end, person, boxes, keepout):
                continue
            score = (round(abs(chord - best_chord), 6), math.dist(end, aims[-1]))
            if best is None or score < best[0]:
                best = (score, (start, control, end))
    return None if best is None else best[1]


def _border(mask: np.ndarray) -> np.ndarray | None:
    """The (x, y) of the pixels on the border of a mask, as an n x 2 array: set pixels with a
    neighbor (up, down, left or right) that is not set. The edge of the picture is not an outside,
    as the person may run off it. None when there are no such pixels."""
    padded = np.pad(mask, 1, mode="edge")
    inner = padded[1:-1, 1:-1]
    surrounded = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    ys, xs = np.nonzero(inner & ~surrounded)
    return np.stack([xs, ys], axis=1) if len(xs) else None


def _reach(alpha: np.ndarray, band: int) -> np.ndarray | None:
    """The border of the person grown by band + 8 px, as the (x, y) of its pixels (an n x 2 array).
    None when there is nothing to aim at: no person, or one that fills the picture.

    The grown person is dilate(alpha, band + 8) > 0.5. It is found at half size, where there are a
    quarter of the pixels (and the blur is half as wide), and each pixel of the border is put back
    at the middle of its 2 x 2 block, so a point of it is within a couple of pixels of where the
    full-size mask would have it.
    """
    height, width = alpha.shape
    rows, columns = height // 2, width // 2
    if rows == 0 or columns == 0:
        return None
    half = alpha[: rows * 2, : columns * 2].reshape(rows, 2, columns, 2).mean(axis=(1, 3))
    border = _border(dilate(half, (band + _TARGET_GAP) / 2) > 0.5)
    return None if border is None else 2 * border + 1


def _nearest(points: np.ndarray, start: tuple[float, float]) -> tuple[int, int]:
    """The closest of some (x, y) points to start."""
    distance = (points[:, 0] - start[0]) ** 2 + (points[:, 1] - start[1]) ** 2
    x, y = points[int(np.argmin(distance))]
    return int(x), int(y)


def silhouette_target(alpha: np.ndarray, start: tuple[float, float], band: int) -> tuple[int, int]:
    """The point nearest to `start` of the person grown by band + 8 px (dilate(alpha, band + 8) >
    0.5): just outside the sticker edge, on bare background. This is the third target of an arrow
    in plan_notes, which finds it on the full-size grown mask it already has, so that it does not
    dilate again for every start.

    A start inside the grown person has no nearest point of it, so it is moved out to the nearest
    point of its border instead. The point is found at half size and is within about 2 px of the
    full-size answer. With nobody in the matte it is the middle of the default face box.
    """
    reach = _reach(alpha, band)
    if reach is None:
        return _middle(_default_face(alpha.shape[1], alpha.shape[0]))
    return _nearest(reach, start)
