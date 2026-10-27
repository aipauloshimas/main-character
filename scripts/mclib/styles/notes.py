"""Everything a card draws on top of its stage: the notes, their arrows, the hero element and the
little extras.

Notes and arrows are drawn from the layout (layout.NoteBox), which is in stage pixels, onto an RGBA
layer the size of the stage. A note is drawn as the layout gives it: the lines as they are (the
case is already applied), at the size and the place it was fitted at. The visible set says which of
the groups "bottom", "top" and "arrows" are on at this moment of the card.

A note that has a tilt is turned about the middle of its box, as the layout's box is not turned: a
few degrees move the ink only a few pixels out of the box the studio drags.

Every size below is in px of the plan, 1080 wide, where the layout is made. The drawing functions
take the layout as it was made, and a raster.Scale for a picture bigger than the plan (a render at the
clip's own size): the boxes are scaled (scaled_layout), and every size in px is multiplied by the
scale's k. The extras decide where a star or a piece of tape goes at the plan and draw them scaled, so
that a bigger picture has the same stars and tape. At the plan (IDENTITY) nothing changes, to the bit.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..config import CONTRAST_MIN, NOTE_MIN_PX
from ..layout import NoteBox, NoteStyle
from ..palette import contrast
from ..project import NOTE_KEYS
from ..raster import (
    IDENTITY,
    Scale,
    dilate,
    draw_arrowhead,
    draw_circle,
    draw_path,
    draw_polygon_aa,
    draw_text,
    font,
    hex_rgb,
    mix,
    quad_points,
)
from . import edges
from .recipes import LABELS, unavailable

# How each notes style sets its notes (see layout.NoteStyle), by the recipe's notes id. The text is
# never drawn below config.NOTE_MIN_PX (36 px, about 13 pt on a phone): that is every style's
# smallest size, and a note that cannot be set whole at it is cut, never drawn smaller. The band
# (how far the sticker edge reaches) is the edge's, whichever notes it is drawn with, and
# note_style puts it in over the one written here.
NOTE_STYLES: dict[str, NoteStyle] = {
    "marker": NoteStyle("marker", 56, NOTE_MIN_PX, "upper", line_h=1.0, pad=0, band=22, arrows=True),
    "comic": NoteStyle("comic", 46, NOTE_MIN_PX, "upper", pad=16),
    "hand": NoteStyle("hand", 52, NOTE_MIN_PX, "asis", pad=16),
    "pixel": NoteStyle("terminal", 52, NOTE_MIN_PX, "upper", line_h=1.0),
    "label": NoteStyle("label", 46, NOTE_MIN_PX, "upper", line_h=1.3, pad=14),
}
# The color role and the width in px of the arrows that go with each notes style.
ARROW_STYLE: dict[str, tuple[str, int]] = {
    "marker": ("ink", 5), "comic": ("dark", 8), "hand": ("ink", 5), "pixel": ("ink", 4), "label": ("edge", 6),
}

# A marker note sits on a dark halo, as the mockup's text-shadow stacks one: a tight glow that
# outlines the strokes (so the lettering reads over a cream sticker edge) under a wide one that
# shades the room around it. The blurs are in px, drawn in this order, the first at the bottom.
_MARKER_GLOWS = (3, 10)
_MARKER_SHADOW = 3  # px: a hard copy of the text, this far down, over the glows

# A comic note is a box of the light color toward white with a dark border and black lettering.
_COMIC_TILTS = (-3, 2, -2, 3)  # degrees clockwise, for tl, tr, bl and br
_COMIC_FILL = 0.45  # how far the light color is mixed toward white
_COMIC_BORDER = 5  # px, inside the box
_COMIC_TEXT = "#111111"

# A handwritten note is a strip of paper torn at the top and the bottom, turned a few degrees, with
# a soft shadow (a CSS shadow of 0 3 8 at 25%, so a blur of sigma 4) and the lettering in ink. Its
# outline is the corners of the mockup's clip-path, as a percentage of the box's width and height.
_HAND_TILTS = (-4, 3, -2, 4)  # degrees clockwise, for tl, tr, bl and br
_HAND_PAPER = "#fffdf5"
_HAND_STRIP = (
    (0, 6), (4, 0), (9, 7), (15, 1), (22, 6), (30, 0), (38, 5), (47, 1), (55, 6), (63, 0), (71, 5),
    (80, 1), (88, 6), (95, 0), (100, 5), (99, 95), (94, 100), (86, 94), (78, 100), (69, 95),
    (60, 100), (51, 94), (42, 100), (33, 95), (24, 100), (15, 94), (7, 100), (1, 95),
)
_HAND_SHADOW = (3, 4, 0.25)  # px down, sigma of the blur, strength
# The word a note marks (extras.emphasis) gets a wavy line in the texture color under it.
_EMPHASIS_WAVE = (2.5, 14, 4, 8)  # px high each way, px long, px thick, px under the baseline

# A pixel note is VT323 (the terminal role: narrow, so that a whole sentence stays big enough to
# read on a phone, and still pixel type) in the ink color with a hard black shadow 3 px down and
# right; the layout sets it from 52 px down to 36 in steps of 4. Press Start 2P is too wide for
# that, and is kept for the HUD's two short labels, whose shadow is the same.
_PIXEL_SHADOW = ("#000000", 3, 3)

# A bold label is a pill for each line, in the edge color, with the text in the dark color.
_LABEL_TILTS = (-2, 1, -1, 2)  # degrees clockwise, for tl, tr, bl and br
_LABEL_RADIUS, _LABEL_GAP = 10, 6  # px: the pill's corners, and the space between two pills

_HEAD_MAX, _HEAD_MIN, _HEAD_SHARE = 24, 10, 0.35  # an arrowhead is 35% of the arrow, from 10 to 24 px
_STRAIGHT = (3, 9)  # a straight arrow: a line 3 px wide, ending in an open circle of radius 9
_DOODLE = (24, 4)  # a doodle arrow is 24 points of the curve, each moved up to 4 px sideways

# The tape: 120 x 36 px at 80%, two stripes of 12 px at 45 degrees, from the texture color toward
# cream. It lies across a top corner of a note (the left one on tl, the right one on tr and br),
# turned as listed.
_TAPE = (120, 36, 0.8)
_TAPE_STRIPE, _TAPE_CREAM, _TAPE_MIXES = 12, "#fff6c8", (0.6, 0.75)
_TAPE_TILTS = {"tl": -28, "tr": 32, "br": -14}  # degrees clockwise
_TAPE_IN, _TAPE_UP = 10, 4  # px inside the note's corner along its top edge, and above it
# The stars: outlines of 5 points, 4 px thick in the ink color, beside the face box: one at its top
# left corner and two on its right side, as (radius, along, down), where along is px from the
# box's edge or the share of the way from it to the stage's edge, and down is px or a share of
# the box's height from its top.
_STARS = ((22, 20, -6), (18, 3, 0.62), (14, 0.77, 0.62))
_STAR_WIDTH, _STAR_VALLEY = 4, 0.45
# A star that would sit on a note, tape, an arrow or another star is moved up or down, 16 px at a
# time, to the nearest place that is free, and left out if there is none within 128 px.
_STAR_NUDGES = (0, *(step * sign for step in range(16, 129, 16) for sign in (-1, 1)))

# The burst: a star of 15 points, 310 x 220 px with its name in the middle. The points reach 150 x
# 105 px from its center and the valleys 33% of the box's width and height. That is its full size: the
# layout may shrink it (burst_box, burst_size), the box and its lettering and lines together, and
# the star is drawn at the scale its box says.
BURST_BOX = (310, 220)
_BURST_POINTS = 15
_BURST_OUTER = (150, 105)
_BURST_INNER = 0.33
_BURST_LINE = 6  # px, the star's outline
_BURST_SIZE, _BURST_MIN_SIZE = 66, 24  # the lettering, and the least it shrinks to
_BURST_ROOM = 184  # px: the widest the name is, so that it stays inside the star's valleys
_BURST_DROP = 24  # px the baseline of the name sits under the center of the star
_BURST_STROKE = 3  # px, the outline of the lettering, centered on the edge of the letters

# The stats panel takes a note's place: a box of the dark color at 82% with a 3 px border of the
# texture color, and in it a row for each stat: its label (VT323 at 34 px, white) and 10 cells of
# 13 x 18 px, 3 px apart, that are on up to the stat's value and off (white at 15%) after it. The
# cells that are on are the color that shows best on the dark one (stats_bar_color), and the low
# stats' are pink. The border is the texture color when that shows on the dark one, and the bars'
# color when it does not (stats_border_color).
STATS_SIZE = 34  # the label's size in px
_STATS_BORDER = 3
_STATS_PAD = (12, 14, 12)  # px above the rows, at their sides and under them
_STATS_ROW, _STATS_GAP, _STATS_LABEL_GAP = 34, 8, 14  # a row's height; the space between rows; the label's
_STATS_CELLS, _STATS_CELL, _STATS_SPACE = 10, (13, 18), 3
_STATS_FILL, _STATS_LOW, _STATS_PINK = 0.82, 2, "#ff3ea5"  # the panel's; the stat that is low; its color
_STATS_DEFAULT = (("tl", 9), ("bl", 1), ("br", 8))  # without stats: these notes, as these stats
_STATS_CUT = 14  # the characters of a note that make a label

# The game HUD: two labels of Press Start 2P (text, size).
HUD_TOP, HUD_BOTTOM = ("PLAYER 1", 24), ("PRESS START", 16)


def note_style(recipe: dict) -> NoteStyle:
    """The layout style a recipe's notes are planned with: the notes style's own, with the band of
    the recipe's edge (the notes keep off the sticker edge, and the arrows end just outside it),
    except that with the arrows layer set to "none" it plans for no arrows, so no room is kept for
    them. An id that is no notes style is a ValueError."""
    try:
        style = NOTE_STYLES[recipe["notes"]]
    except KeyError:
        raise unavailable("notes", recipe["notes"]) from None
    style = replace(style, band=edges.reach(recipe["edge"]))
    return replace(style, arrows=False) if recipe["arrows"] == "none" else style


def drawn_texts(recipe: dict, copy: dict) -> list[tuple[str, str, str, str, str]]:
    """The texts of a copy that a card in this recipe draws, each with the font it is set in, as
    (where, text, font role, case, what the studio calls the layer that chose the font).

    where is the text's address in the copy ("copy.notes.tl"), font role one of raster.font's and
    case "upper" or "asis" as layout.apply_case takes it. The four notes are set in the notes style's
    font and case, except that the stats panel takes the tr note's place, so with that hero the tr
    note is not drawn. The burst says extras.burst, or the name when there is none, in capitals in
    Bangers, and the stats panel has its labels in capitals in VT323. A hero that has none of the
    person's words (none, and the Game HUD, whose two labels are fixed) adds nothing. A stats panel
    with no stats of its own shows the notes tl, bl and br cut to 14 characters: those are the notes
    listed above, and the layout flags the panel itself if its font lacks one of their characters."""
    style = NOTE_STYLES[recipe["notes"]]
    layer = LABELS["notes"][recipe["notes"]]
    hero = recipe["hero"]
    found = []
    for key in NOTE_KEYS:
        if key == "tr" and hero == "stats":
            continue
        found.append((f"copy.notes.{key}", copy["notes"].get(key) or "", style.font, style.case, layer))
    extras = copy.get("extras") or {}
    if hero == "burst":
        burst = extras.get("burst") or ""
        where, text = ("copy.extras.burst", burst) if burst.strip() else ("copy.name", copy.get("name") or "")
        found.append((where, text, "comic", "upper", LABELS["hero"]["burst"]))
    elif hero == "stats":
        for index, (label, _) in enumerate(extras.get("stats") or []):
            found.append((f"copy.extras.stats[{index}]", str(label), "terminal", "upper", LABELS["hero"]["stats"]))
    return found


# The visibility group of every key a layout holds: the top notes come in at frame 2 of the card
# and the bottom ones at frame 0, and each hero box comes with the group it sits in.
_GROUPS = {
    "tl": "top", "tr": "top", "burst": "top", "hud_top": "top",
    "bl": "bottom", "br": "bottom", "hud_bottom": "bottom",
}


def group(key: str) -> str:
    """The visibility group of a layout key, "top" or "bottom". A key the layout does not use is
    a ValueError."""
    try:
        return _GROUPS[key]
    except KeyError:
        raise ValueError(f"{key!r} is not a key of a layout; they are {', '.join(_GROUPS)}") from None


_HUD_KEYS = ("hud_top", "hud_bottom")  # the boxes set in Press Start 2P, which is only crisp on its 8 px grid


def scaled_layout(layout: dict[str, NoteBox], scale: Scale = IDENTITY) -> dict[str, NoteBox]:
    """The boxes of a layout, made at the plan, as they are drawn on a picture `scale` bigger: the
    same boxes with their corners and sizes in that picture's px, rounded to whole pixels, and their
    arrows, whose points are not rounded. The lines, the cut flag and the clash flag stay. At the plan
    it is the layout itself.

    A text is not as wide at k times the size as k times its width, since every advance of a glyph is
    rounded to a pixel, so it can be a few px wider than its box at k times the size: _fitted brings
    the size of the notes down by that much, and draw_notes does it. The notes that are drawn on a
    fill (comic, hand and label) are centered in their box and the others are set from its top left
    corner, as the layout sets them.

    The HUD's labels are Press Start 2P, which is only crisp on its 8 px grid, so their size goes to
    the nearest multiple of 8 and is not k times the plan's. Their box is then the label as it is
    drawn: as wide as its text is at that size, and as tall as the size. "PLAYER 1" keeps its left
    corner, and "PRESS START" its right edge, which is where the layout set it against the room's."""
    if scale.identity:
        return layout
    scaled = {}
    for key, box in layout.items():
        arrow = None
        if box.arrow is not None:
            arrow = tuple((scale.x(x), scale.y(y)) for x, y in box.arrow)
        x, y = round(scale.x(box.x)), round(scale.y(box.y))
        w, h = round(scale.x(box.w)), round(scale.y(box.h))
        size = scale.font(box.size, pixel=key in _HUD_KEYS)
        if key in _HUD_KEYS and box.lines:
            w, h = max(hud_width(line, size) for line in box.lines), size * len(box.lines)
            if key == "hud_bottom":
                x = round(scale.x(box.x + box.w)) - w
        scaled[key] = replace(box, x=x, y=y, w=w, h=h, size=size, arrow=arrow)
    return scaled


def _at(style: NoteStyle, scale: Scale) -> NoteStyle:
    """A notes style with its padding as it is drawn on a picture `scale` bigger than the plan."""
    return style if scale.identity else replace(style, pad=scale.n(style.pad))


_FIT_LEAST = 0.95  # a note's size is brought down to fit its box by at most this much of itself


def _fitted(layout: dict[str, NoteBox], style: NoteStyle, recipe: dict, scale: Scale) -> dict[str, NoteBox]:
    """The scaled layout with the notes' size brought down, where it has to be, so that no line of a
    note is wider than its box: a text is not as wide at k times the size as k times its width (every
    advance is rounded to a pixel), and a note that is set from its box's corner, with nothing drawn
    behind it (marker and pixel), would come out a few px past the room the layout kept for it. It is
    one pixel at a time and by at most 5% of the size (the most it has been measured to need is under
    3.5%), and the notes that shared a size at the plan share the smallest any of them fits at. At the
    plan it is the layout itself."""
    if scale.identity:
        return layout
    fits = {}
    for key in NOTE_KEYS:
        box = layout.get(key)
        if box is None or not box.lines or (key == "tr" and recipe["hero"] == "stats"):
            continue
        size, floor = box.size, max(1, round(box.size * _FIT_LEAST))
        while size > floor and max(font(style.font, size).getlength(line) for line in box.lines) > box.w:
            size -= 1
        fits[key] = size
    shared: dict[int, int] = {}  # a size the layout has -> the smallest it fits at
    for key, size in fits.items():
        shared[layout[key].size] = min(shared.get(layout[key].size, size), size)
    return {
        key: replace(box, size=shared[box.size]) if key in fits else box for key, box in layout.items()
    }


# --- drawing helpers --------------------------------------------------------------------------


def _lay(img: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    """Composite an RGBA layer onto img in place, its top-left corner at (x, y), clipped at the
    edges of img."""
    left, top = max(x, 0), max(y, 0)
    right, bottom = min(x + layer.width, img.width), min(y + layer.height, img.height)
    if right <= left or bottom <= top:
        return
    if (left, top, right, bottom) != (x, y, x + layer.width, y + layer.height):
        layer = layer.crop((left - x, top - y, right - x, bottom - y))
    img.alpha_composite(layer, (left, top))


def _turn(points, center, tilt: float) -> list[tuple[float, float]]:
    """Points turned about a center by `tilt` degrees clockwise on the screen, as CSS rotate()
    turns them (draw_text turns the other way, so it is given -tilt)."""
    sin, cos = math.sin(math.radians(tilt)), math.cos(math.radians(tilt))
    cx, cy = center
    return [
        (cx + (x - cx) * cos - (y - cy) * sin, cy + (x - cx) * sin + (y - cy) * cos) for x, y in points
    ]


def _corners(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _block(box: NoteBox, style: NoteStyle):
    """What a note drawn turned about its middle needs: the font, the line height in px, the
    top-left corner that centers the lines on the middle of the box, and that middle."""
    fnt = font(style.font, box.size)
    pitch = round(box.size * style.line_h)
    width = max((fnt.getlength(line) for line in box.lines), default=0.0)
    center = (box.x + box.w / 2, box.y + box.h / 2)
    return fnt, pitch, (center[0] - width / 2, center[1] - pitch * len(box.lines) / 2), center


# --- notes ------------------------------------------------------------------------------------


def _marker(
    img: Image.Image, box: NoteBox, style: NoteStyle, pal: dict, copy: dict, scale: Scale = IDENTITY
) -> None:
    """A marker note: ink-colored lettering with a tight and a wide dark glow behind it and a dark
    line under it, so it reads on any background, even over the sticker edge. The box's top-left
    corner is the text's (marker has no padding)."""
    draw_text(
        img,
        (box.x + style.pad, box.y + style.pad),
        box.lines,
        font(style.font, box.size),
        pal["ink"],
        line_h=round(box.size * style.line_h),
        glow=[(pal["dark"], scale.px(radius)) for radius in _MARKER_GLOWS],
        shadow=(pal["dark"], 0, scale.px(_MARKER_SHADOW)),
    )


def _comic(
    img: Image.Image, box: NoteBox, style: NoteStyle, pal: dict, copy: dict, scale: Scale = IDENTITY
) -> None:
    """A comic note: a box of the light color toward white, its border of the dark color 5 px
    thick inside the box's edge, and the lettering in #111111, all turned a few degrees."""
    fnt, pitch, xy, center = _block(box, style)
    tilt = _COMIC_TILTS[NOTE_KEYS.index(box.key)]
    x0, y0, x1, y1 = box.x, box.y, box.x + box.w, box.y + box.h
    border = scale.px(_COMIC_BORDER)
    draw_polygon_aa(img, _turn(_corners(x0, y0, x1, y1), center, tilt), pal["dark"])
    inside = _corners(x0 + border, y0 + border, x1 - border, y1 - border)
    draw_polygon_aa(img, _turn(inside, center, tilt), mix(pal["light"], "#ffffff", _COMIC_FILL))
    draw_text(img, xy, box.lines, fnt, _COMIC_TEXT, line_h=pitch, rotate=-tilt)


def _shadow(img: Image.Image, points, dy: int, sigma: float, strength: float) -> None:
    """A soft black shadow of a shape, `dy` px lower than it: the shape's mask blurred by sigma
    and laid on at `strength`."""
    pad = math.ceil(3 * sigma + abs(dy)) + 2
    left, top = math.floor(min(x for x, _ in points)) - pad, math.floor(min(y for _, y in points)) - pad
    right, bottom = math.ceil(max(x for x, _ in points)) + pad, math.ceil(max(y for _, y in points)) + pad
    mask = Image.new("L", (right - left, bottom - top), 0)
    ImageDraw.Draw(mask).polygon([(x - left, y - top + dy) for x, y in points], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(sigma)).point(lambda level: round(level * strength))
    layer = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    layer.putalpha(mask)
    _lay(img, layer, left, top)


def _emphasis_spans(lines: list[str], word: str | None) -> list[tuple[int, int, int]]:
    """Where the first whole-word match of `word` (any case) is in a note's lines, as the
    (line, first character, end) of each part of it: the lines are read as the one text they were
    broken from (joined by a space), so a phrase can run over the end of a line. No word, or no
    match, is no spans."""
    word = (word or "").strip()
    if not word:
        return []
    found = re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", " ".join(lines), re.IGNORECASE)
    if found is None:
        return []
    spans, offset = [], 0
    for number, line in enumerate(lines):
        first, end = max(found.start() - offset, 0), min(found.end() - offset, len(line))
        if first < end:
            spans.append((number, first, end))
        offset += len(line) + 1
    return spans


def _wave(
    left: float, right: float, y: float, scale: Scale = IDENTITY
) -> list[tuple[float, float]]:
    """The points of a wavy line from `left` to `right`, its middle at height y."""
    height, length = scale.px(_EMPHASIS_WAVE[0]), scale.px(_EMPHASIS_WAVE[1])
    count = max(math.ceil((right - left) / 2), 1)
    span = right - left
    return [
        (left + span * i / count, y + height * math.sin(2 * math.pi * span * i / count / length))
        for i in range(count + 1)
    ]


def _hand(
    img: Image.Image, box: NoteBox, style: NoteStyle, pal: dict, copy: dict, scale: Scale = IDENTITY
) -> None:
    """A handwritten note: a torn strip of paper, a soft shadow under it, the lettering in the ink
    color (as the copy was written, in its own case), all turned a few degrees. On the bl note, a
    wavy line in the texture color goes under the word that extras.emphasis names, if the note has it."""
    fnt, pitch, xy, center = _block(box, style)
    tilt = _HAND_TILTS[NOTE_KEYS.index(box.key)]
    strip = [(box.x + box.w * across / 100, box.y + box.h * down / 100) for across, down in _HAND_STRIP]
    strip = _turn(strip, center, tilt)
    drop_by, sigma, strength = _HAND_SHADOW
    _shadow(img, strip, scale.px(drop_by), scale.px(sigma), strength)
    draw_polygon_aa(img, strip, _HAND_PAPER)
    draw_text(img, xy, box.lines, fnt, pal["ink"], line_h=pitch, rotate=-tilt)

    if box.key == "bl":
        thick, drop = scale.px(_EMPHASIS_WAVE[2]), scale.px(_EMPHASIS_WAVE[3])
        ascent, descent = fnt.getmetrics()
        baseline = xy[1] + math.floor((pitch - (ascent + descent)) / 2) + ascent
        for number, first, end in _emphasis_spans(box.lines, (copy.get("extras") or {}).get("emphasis")):
            line = box.lines[number]
            left, right = xy[0] + fnt.getlength(line[:first]), xy[0] + fnt.getlength(line[:end])
            wave = _wave(left, right, baseline + number * pitch + drop, scale)
            draw_path(img, _turn(wave, center, tilt), pal["tex"], thick)


def _pixel_shadow(scale: Scale) -> tuple[str, float, float]:
    """The hard shadow of the pixel type, as it is drawn on a picture `scale` bigger than the plan."""
    color, right, down = _PIXEL_SHADOW
    return color, scale.px(right), scale.px(down)


def _pixel(
    img: Image.Image, box: NoteBox, style: NoteStyle, pal: dict, copy: dict, scale: Scale = IDENTITY
) -> None:
    """A pixel note: VT323 capitals in the ink color with a hard black shadow."""
    draw_text(
        img,
        (box.x + style.pad, box.y + style.pad),
        box.lines,
        font(style.font, box.size),
        pal["ink"],
        line_h=round(box.size * style.line_h),
        shadow=_pixel_shadow(scale),
    )


def _rounded(x0: float, y0: float, x1: float, y1: float, radius: float, steps: int = 8):
    """The points of a rounded rectangle, clockwise from the top right corner's arc."""
    points = []
    for cx, cy, start in (
        (x1 - radius, y0 + radius, -90), (x1 - radius, y1 - radius, 0),
        (x0 + radius, y1 - radius, 90), (x0 + radius, y0 + radius, 180),
    ):
        for i in range(steps + 1):
            angle = math.radians(start + 90 * i / steps)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _label(
    img: Image.Image, box: NoteBox, style: NoteStyle, pal: dict, copy: dict, scale: Scale = IDENTITY
) -> None:
    """A bold label: each line is a pill of the edge color, as wide as its text and the padding,
    the lines stacked (left aligned) in the middle of the box, with the text in the dark color, and
    the whole turned a few degrees about the middle of the box."""
    fnt, pitch, _, center = _block(box, style)
    tilt = _LABEL_TILTS[NOTE_KEYS.index(box.key)]
    gap, radius = scale.px(_LABEL_GAP), scale.px(_LABEL_RADIUS)
    height = pitch - gap
    top = box.y + (box.h - (len(box.lines) * pitch - gap)) / 2
    for number, line in enumerate(box.lines):
        width = fnt.getlength(line)
        y = top + number * pitch
        pill = _rounded(box.x, y, box.x + width + 2 * style.pad, y + height, radius)
        draw_polygon_aa(img, _turn(pill, center, tilt), pal["edge"])
        # the pill is turned about the middle of the box, so the text goes where the turn puts it
        (x, y), = _turn([(box.x + style.pad + width / 2, y + height / 2)], center, tilt)
        draw_text(img, (x - width / 2, y - height / 2), [line], fnt, pal["dark"], line_h=height, rotate=-tilt)


# One drawing function for each notes style that is built: (img, box, style, palette, copy).
_NOTES = {"marker": _marker, "comic": _comic, "hand": _hand, "pixel": _pixel, "label": _label}


def draw_notes(
    img: Image.Image, layout: dict[str, NoteBox], copy: dict, recipe: dict, visible,
    scale: Scale = IDENTITY,
) -> None:
    """Draw the four notes of the layout whose group (bottom or top) is visible, in place on an
    RGBA stage-sized image. `copy` is the card's copy: a style that marks a word or sets a name
    reads it, and the others only need the lines the layout holds. The layout is the one made at the
    plan, and the image is `scale` bigger than that."""
    style = _at(note_style(recipe), scale)
    layout = _fitted(scaled_layout(layout, scale), style, recipe, scale)
    draw = _NOTES[recipe["notes"]]
    for key in NOTE_KEYS:
        if key == "tr" and recipe["hero"] == "stats":  # that box is the panel of stats, a hero
            continue
        box = layout.get(key)
        if box is not None and group(key) in visible:
            draw(img, box, style, recipe["palette"], copy, scale)


# --- arrows -----------------------------------------------------------------------------------


def _head_size(start, end, scale: Scale = IDENTITY) -> float:
    """The size of an arrow's head: 35% of the arrow's length, from 10 to 24 px, so a short arrow
    does not end in a big head."""
    return min(
        scale.px(_HEAD_MAX), max(scale.px(_HEAD_MIN), round(_HEAD_SHARE * math.dist(start, end)))
    )


def _curly(img: Image.Image, arrow, color: str, width: float, scale: Scale = IDENTITY) -> None:
    """A curve from the note to the person, with an open head at its end."""
    start, control, end = arrow
    draw_path(img, quad_points(start, control, end), color, width)
    draw_arrowhead(img, end, control, color, width, head=_head_size(start, end, scale))


def _doodle(img: Image.Image, arrow, color: str, width: float, scale: Scale = IDENTITY) -> None:
    """The curve drawn by a shaky hand: 24 points of it, each moved sideways by 4 px times the
    sine of five half turns, less and less toward the end, so the line wobbles and still lands on
    its end. A pen stroke 1 px thinner than the others, and the head at the end."""
    start, control, end = arrow
    count, wobble = _DOODLE
    wobble = scale.px(wobble)
    points = []
    for i in range(count + 1):
        t = i / count
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t * t * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t * t * end[1]
        dx = 2 * (1 - t) * (control[0] - start[0]) + 2 * t * (end[0] - control[0])
        dy = 2 * (1 - t) * (control[1] - start[1]) + 2 * t * (end[1] - control[1])
        length = math.hypot(dx, dy) or 1.0
        push = math.sin(t * math.pi * 5) * wobble * (1 - t * 0.5)
        points.append((x - dy / length * push, y + dx / length * push))
    thin = width - scale.px(1)
    draw_path(img, points, color, thin)
    draw_arrowhead(img, end, control, color, thin, head=_head_size(start, end, scale))


def _straight(img: Image.Image, arrow, color: str, width: float, scale: Scale = IDENTITY) -> None:
    """A straight line from the note to the person, 3 px wide whatever the notes style's arrows are,
    ending in an open circle of radius 9 (and no head)."""
    start, _, end = arrow
    line, radius = scale.px(_STRAIGHT[0]), scale.px(_STRAIGHT[1])
    draw_path(img, [start, end], color, line)
    draw_circle(img, end, radius, color, line)


# One drawing function for each arrows option that draws something: (img, arrow, color, width).
_ARROWS = {"curly": _curly, "doodle": _doodle, "straight": _straight}


def draw_arrows(
    img: Image.Image, layout: dict[str, NoteBox], recipe: dict, visible, scale: Scale = IDENTITY
) -> None:
    """Draw the arrows of the layout when "arrows" is visible, each from its start (arrow[0]) as
    the layout placed it, in place on an RGBA stage-sized image. A note with no arrow gets none,
    and the arrows layer "none" draws nothing. The layout is the one made at the plan, and the image
    is `scale` bigger than that."""
    kind = recipe["arrows"]
    if "arrows" not in visible or kind == "none":
        return
    try:
        draw = _ARROWS[kind]
    except KeyError:
        raise unavailable("arrows", kind) from None
    try:
        role, width = ARROW_STYLE[recipe["notes"]]
    except KeyError:
        raise unavailable("notes", recipe["notes"]) from None
    for box in scaled_layout(layout, scale).values():
        if box.arrow is not None:
            draw(img, box.arrow, recipe["palette"][role], scale.px(width), scale)


# --- the hero and the extras ------------------------------------------------------------------


def burst_text(copy: dict) -> str:
    """What the burst says: the copy's own burst text when it has one, else the name and "!", in
    capitals. Blank text is no text."""
    text = ((copy.get("extras") or {}).get("burst") or "").strip()
    if not text:
        name = (copy.get("name") or "").strip()
        text = f"{name}!" if name else ""
    return text.upper()


def burst_box(scale: float = 1.0) -> tuple[int, int]:
    """The (width, height) of the burst at a scale of its full size, BURST_BOX: 1 is 310 x 220."""
    return round(BURST_BOX[0] * scale), round(BURST_BOX[1] * scale)


def burst_size(text: str, scale: float = 1.0) -> int:
    """The size of the burst's lettering: 66 px, or smaller (down to 24) for a text that would
    not fit inside the star, and at a smaller scale of the burst that much smaller (rounded)."""
    size = _BURST_SIZE
    while size > _BURST_MIN_SIZE and font("comic", size).getlength(text) > _BURST_ROOM:
        size -= 2
    return round(size * scale)


def _stroked_text(
    img: Image.Image, xy, lines, fnt, fill: str, outline: str, stroke: float, line_h: int
) -> None:
    """Lines of text with an outline `stroke` px wide, centered on the edge of the letters as an
    SVG stroke is: half of it grows the letters and half of it eats into them. The letters are
    drawn on a tile of their own, their mask grown and shrunk by half the stroke, and the outline
    color goes under the shrunk letters in the grown ones."""
    pad = math.ceil(stroke) + 10
    width = math.ceil(max((fnt.getlength(line) for line in lines), default=0)) + 2 * pad
    height = line_h * len(lines) + 2 * pad
    left, top = math.floor(xy[0]), math.floor(xy[1])
    tile = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_text(tile, (xy[0] - left + pad, xy[1] - top + pad), lines, fnt, "#ffffff", line_h=line_h)
    mask = np.asarray(tile, np.float32)[..., 3] / 255
    grown = dilate(mask, stroke / 2)
    shrunk = 1 - dilate(1 - mask, stroke / 2)
    for color, coverage in ((outline, grown), (fill, shrunk)):
        red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
        layer = np.empty((height, width, 4), np.uint8)
        layer[..., :3] = (red, green, blue)
        layer[..., 3] = np.rint(np.clip(coverage, 0.0, 1.0) * 255)
        _lay(img, Image.fromarray(layer, "RGBA"), left - pad, top - pad)


def _no_hero(
    img: Image.Image, layout: dict[str, NoteBox], copy: dict, recipe: dict, visible,
    scale: Scale = IDENTITY,
) -> None:
    """The recipe has no hero element."""


def _burst(
    img: Image.Image, layout: dict[str, NoteBox], copy: dict, recipe: dict, visible,
    scale: Scale = IDENTITY,
) -> None:
    """The name burst: a white star with a black outline and the name in the texture color, with
    a black outline of its own, in the middle. It is drawn at the scale its box has against the
    full-size BURST_BOX (a burst that shrank to fit): its points, its outline, the drop of its
    lettering and the lettering's own outline go with it (the lettering's size is the box's)."""
    box = layout.get("burst")
    if box is None or group("burst") not in visible:
        return
    across, down = box.w / BURST_BOX[0], box.h / BURST_BOX[1]
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    points = []
    for i in range(2 * _BURST_POINTS):
        angle = i * math.pi / _BURST_POINTS
        if i % 2 == 0:
            rx, ry = _BURST_OUTER[0] * across, _BURST_OUTER[1] * down
        else:
            rx, ry = _BURST_INNER * box.w, _BURST_INNER * box.h
        points.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    draw_polygon_aa(img, points, "#ffffff", "#111111", _BURST_LINE * across)

    fnt = font("comic", box.size)
    ascent, descent = fnt.getmetrics()
    pitch = ascent + descent
    width = max((fnt.getlength(line) for line in box.lines), default=0.0)
    top = cy + _BURST_DROP * down - ascent  # the baseline lands 24 px under the middle of the star
    _stroked_text(
        img, (cx - width / 2, top), box.lines, fnt, recipe["palette"]["tex"], "#111111",
        _BURST_STROKE * across, pitch,
    )


def stats_rows(copy: dict) -> list[tuple[str, float]]:
    """The rows of the stats panel, as (label, value): the copy's stats, with the labels in
    capitals; or, when the copy has none, the notes tl, bl and br at 9, 1 and 8, each cut to its
    first 14 characters."""
    stats = (copy.get("extras") or {}).get("stats")
    if stats:
        return [(str(label).upper(), value) for label, value in stats]
    notes = copy.get("notes") or {}
    return [((notes.get(key) or "").upper()[:_STATS_CUT].rstrip(), value) for key, value in _STATS_DEFAULT]


def stats_size(rows) -> tuple[int, int]:
    """The size of the stats panel for its rows: as wide as its widest label, the gap, ten cells
    and the padding and border, and as tall as the rows, the space between them, the padding and
    the border."""
    fnt = font("terminal", STATS_SIZE)
    widest = max((fnt.getlength(label) for label, _ in rows), default=0)
    cells = _STATS_CELLS * _STATS_CELL[0] + (_STATS_CELLS - 1) * _STATS_SPACE
    above, side, below = _STATS_PAD
    width = math.ceil(widest) + _STATS_LABEL_GAP + cells + 2 * side + 2 * _STATS_BORDER
    height = len(rows) * _STATS_ROW + max(len(rows) - 1, 0) * _STATS_GAP + above + below + 2 * _STATS_BORDER
    return width, height


def stats_bar_color(pal: dict) -> str:
    """The color of the cells of the stats panel that are on: of the palette's texture, edge and ink
    colors, the one with the highest contrast against the dark color (the panel's fill), and the
    first of them (in that order) when two are as good. The texture color alone was not enough: in
    Riso and Mono it is the dark color itself, and bars in it would not show."""
    return max((pal["tex"], pal["edge"], pal["ink"]), key=lambda color: contrast(color, pal["dark"]))


def stats_border_color(pal: dict) -> str:
    """The color of the stats panel's border: the texture color when it shows on the dark one (a
    contrast of config.CONTRAST_MIN or more), else the color of the bars. In Riso and Mono the
    texture color is the dark color itself, and Magenta, Lagoon and Tangerine have one darker still,
    so a border in it would be lost against the panel."""
    if contrast(pal["tex"], pal["dark"]) >= CONTRAST_MIN:
        return pal["tex"]
    return stats_bar_color(pal)


def _fill(img: Image.Image, rects, color) -> None:
    """Fill rectangles (x0, y0, x1, y1, on whole pixels, the end one past the last) with a color,
    "#rrggbb" or (r, g, b, a) up to 255, laid over what is there."""
    if isinstance(color, str):
        color = (*(int(color[i : i + 2], 16) for i in (1, 3, 5)), 255)
    left, top = min(r[0] for r in rects), min(r[1] for r in rects)
    right, bottom = max(r[2] for r in rects), max(r[3] for r in rects)
    mask = Image.new("L", (right - left, bottom - top), 0)
    draw = ImageDraw.Draw(mask)
    for x0, y0, x1, y1 in rects:
        draw.rectangle((x0 - left, y0 - top, x1 - left - 1, y1 - top - 1), fill=color[3])
    layer = Image.new("RGBA", mask.size, (*color[:3], 0))
    layer.putalpha(mask)
    _lay(img, layer, left, top)


def _stats(
    img: Image.Image, layout: dict[str, NoteBox], copy: dict, recipe: dict, visible,
    scale: Scale = IDENTITY,
) -> None:
    """The stats panel, in the box the layout keeps under the key "tr", with the top notes. Its
    border is stats_border_color, and its cells that are on are stats_bar_color, or pink for a
    stat of 2 or less. Drawn bigger than the plan, its parts are scaled and the label and the cells
    stay at the two sides of the box."""
    panel = layout.get("tr")
    if panel is None or group("tr") not in visible:
        return
    pal = recipe["palette"]
    bar = stats_bar_color(pal)
    x0, y0, x1, y1 = panel.x, panel.y, panel.x + panel.w, panel.y + panel.h
    border = scale.n(_STATS_BORDER, minimum=1)
    above, side = scale.n(_STATS_PAD[0]), scale.n(_STATS_PAD[1])
    row, gap = scale.n(_STATS_ROW), scale.n(_STATS_GAP)
    cell_w, cell_h, space = scale.n(_STATS_CELL[0]), scale.n(_STATS_CELL[1]), scale.n(_STATS_SPACE)
    dark = tuple(int(pal["dark"][i : i + 2], 16) for i in (1, 3, 5))
    _fill(img, [(x0, y0, x1, y1)], (*dark, round(_STATS_FILL * 255)))
    _fill(
        img,
        [(x0, y0, x1, y0 + border), (x0, y1 - border, x1, y1),
         (x0, y0 + border, x0 + border, y1 - border), (x1 - border, y0 + border, x1, y1 - border)],
        stats_border_color(pal),
    )
    fnt = font("terminal", scale.font(STATS_SIZE))
    cells = _STATS_CELLS * cell_w + (_STATS_CELLS - 1) * space
    right = x1 - border - side
    for number, (label, value) in enumerate(stats_rows(copy)):
        top = y0 + border + above + number * (row + gap)
        draw_text(img, (x0 + border + side, top), [label], fnt, "#ffffff", line_h=row)
        lit = min(max(int(value + 0.5), 0), _STATS_CELLS)
        on, off = [], []
        for cell in range(_STATS_CELLS):
            left = right - cells + cell * (cell_w + space)
            top_cell = top + (row - cell_h) // 2
            (on if cell < lit else off).append((left, top_cell, left + cell_w, top_cell + cell_h))
        if on:
            _fill(img, on, _STATS_PINK if value <= _STATS_LOW else bar)
        if off:
            _fill(img, off, (255, 255, 255, round(0.15 * 255)))


def hud_width(text: str, size: int) -> int:
    """The width in px of a HUD label."""
    return math.ceil(font("pixel", size).getlength(text))


def _hud(
    img: Image.Image, layout: dict[str, NoteBox], copy: dict, recipe: dict, visible,
    scale: Scale = IDENTITY,
) -> None:
    """The two labels of the game HUD, in the ink color with a hard black shadow, each with its
    group: "PLAYER 1" with the top notes and "PRESS START" with the bottom ones."""
    for key in ("hud_top", "hud_bottom"):
        box = layout.get(key)
        if box is not None and group(key) in visible:
            draw_text(
                img, (box.x, box.y), box.lines, font("pixel", box.size), recipe["palette"]["ink"],
                line_h=box.size, shadow=_pixel_shadow(scale),
            )


# One drawing function for each hero that is built: (img, layout, copy, recipe, visible).
_HEROES = {"none": _no_hero, "burst": _burst, "stats": _stats, "hud": _hud}


def draw_hero(
    img: Image.Image, layout: dict[str, NoteBox], copy: dict, recipe: dict, visible,
    scale: Scale = IDENTITY,
) -> None:
    """Draw the recipe's hero element, in place on an RGBA stage-sized image. An id that is no hero
    is a ValueError. The layout is the one made at the plan, and the image is `scale` bigger."""
    try:
        draw = _HEROES[recipe["hero"]]
    except KeyError:
        raise unavailable("hero", recipe["hero"]) from None
    draw(img, scaled_layout(layout, scale), copy, recipe, visible, scale)


def _tape_tile(pal: dict, scale: Scale = IDENTITY) -> Image.Image:
    """One piece of tape, flat: an RGBA picture with two stripes at 45 degrees, from the texture
    color toward cream, at 80%. The stripes go on a little past the tape (where it is transparent) so
    that turning it does not bleed black into its edge."""
    width, height, opacity = _TAPE
    width, height, stripe_px = scale.n(width), scale.n(height), scale.px(_TAPE_STRIPE)
    margin = 4
    across = np.arange(width + 2 * margin)[None, :] - margin
    down = np.arange(height + 2 * margin)[:, None] - margin
    stripe = ((across + height - down) * math.sqrt(0.5) // stripe_px % 2).astype(int)
    colors = np.array([hex_rgb(mix(pal["tex"], _TAPE_CREAM, share)) for share in _TAPE_MIXES])
    tile = np.zeros((height + 2 * margin, width + 2 * margin, 4), np.uint8)
    tile[..., :3] = np.rint(colors[stripe] * 255)
    tile[margin : margin + height, margin : margin + width, 3] = round(opacity * 255)
    return Image.fromarray(tile, "RGBA")


def _tape_spots(layout: dict[str, NoteBox], keepout=()) -> dict[str, tuple[float, float, int]]:
    """Where each piece of tape goes, for the notes that are there: its middle and its turn. A piece
    that would lie across a hero element (the burst, the HUD's labels) or across one of the boxes
    to keep out of (the title room of a 9:16 card) is left out: a top note that sits just under
    that room has no tape, since the tape rises above its edge."""
    heroes = [box for key, box in layout.items() if key not in NOTE_KEYS]
    avoid = [(h.x, h.y, h.x + h.w, h.y + h.h) for h in heroes] + [tuple(kept) for kept in keepout]
    spots = {}
    for key, tilt in _TAPE_TILTS.items():
        box = layout.get(key)
        if box is not None:
            x = box.x + _TAPE_IN if key == "tl" else box.x + box.w - _TAPE_IN
            left, top, right, bottom = _tape_box(x, box.y - _TAPE_UP, tilt)
            if not any(left < c and right > a and top < d and bottom > b for a, b, c, d in avoid):
                spots[key] = (x, box.y - _TAPE_UP, tilt)
    return spots


def _tape(
    img: Image.Image, layout: dict[str, NoteBox], pal: dict, visible, keepout=(), scale: Scale = IDENTITY
) -> None:
    """The pieces of tape on the notes that are there, each with its note (so it is on when the
    note's group is). Where they go is worked out at the plan, from the layout as it was made."""
    tile = _tape_tile(pal, scale)
    for key, (x, y, tilt) in _tape_spots(layout, keepout).items():
        if group(key) in visible:
            turned = tile.rotate(-tilt, Image.Resampling.BICUBIC, expand=True)
            _lay(img, turned, round(scale.x(x) - turned.width / 2), round(scale.y(y) - turned.height / 2))


def _tape_box(x: float, y: float, tilt: int) -> tuple[float, float, float, float]:
    """The box around a piece of tape of the given middle and turn."""
    width, height, _ = _TAPE
    sin, cos = abs(math.sin(math.radians(tilt))), abs(math.cos(math.radians(tilt)))
    half_w, half_h = (width * cos + height * sin) / 2, (width * sin + height * cos) / 2
    return x - half_w, y - half_h, x + half_w, y + half_h


def _star_spots(taken: list, curves: list, face, width: int, height: int) -> list[tuple[float, float, float]]:
    """Where the stars beside the face box go, as (x, y, radius) of each one that has a place, in the
    px of the plan (of a stage `width` x `height` px): one at its top left corner and two on its right
    side. A star that would sit on one of the boxes `taken`, on a point of the arrows' `curves` or on
    another star is moved to the nearest free place up or down, and one that would go off the stage, or
    has no place, is left out. (The boxes to keep out of are among the boxes `taken`.)"""
    x0, y0, x1, y1 = face
    (r1, in1, up1), (r2, out2, down2), (r3, share3, down3) = _STARS
    spots = (
        (x0 + in1, y0 + up1, r1),
        (x1 + out2, y0 + down2 * (y1 - y0), r2),
        (x1 + share3 * (width - x1), y0 + down3 * (y1 - y0) - 4, r3),
    )
    taken = list(taken)
    found = []
    for x, y, radius in spots:
        reach = radius + _STAR_WIDTH / 2 + 6
        for nudge in _STAR_NUDGES:
            top, bottom = y + nudge - reach, y + nudge + reach
            if x - reach < 0 or top < 0 or x + reach > width or bottom > height:
                continue
            if any(x + reach > a and x - reach < c and bottom > b and top < d for a, b, c, d in taken):
                continue
            if any(math.hypot(px - x, py - (y + nudge)) < reach + 4 for px, py in curves):
                continue
            taken.append((x - reach, top, x + reach, bottom))
            found.append((x, y + nudge, radius))
            break
    return found


def _stars(
    img: Image.Image, taken: list, curves: list, pal: dict, face, scale: Scale = IDENTITY
) -> None:
    """The stars beside the face box (see _star_spots, which decides where, at the plan), drawn
    `scale` bigger than that."""
    width, height = img.size if scale.identity else scale.plan
    for x, y, radius in _star_spots(taken, curves, face, width, height):
        points = []
        for i in range(10):
            r = radius * (_STAR_VALLEY if i % 2 else 1)
            angle = -math.pi / 2 + i * math.pi / 5
            points.append((scale.x(x + r * math.cos(angle)), scale.y(y + r * math.sin(angle))))
        draw_polygon_aa(img, points, None, pal["ink"], scale.px(_STAR_WIDTH))


def draw_extras(
    img: Image.Image, layout: dict[str, NoteBox], recipe: dict, visible, face, keepout=(),
    scale: Scale = IDENTITY,
) -> None:
    """Draw the first little extras that sit on the card, in place on an RGBA stage-sized image: the tape
    on the notes, and the stars beside the face. (The film grain, vhs and the light leak are part of the
    stage itself, and engine.compose_stage adds them; the eleven that came later are styles.extras'.) face is the face box (x0, y0, x1, y1) of the layout's context, or
    None, which is no stars. keepout is the layout's context's boxes to keep out of, (x0, y0, x1,
    y1) in stage pixels (the title room of a 9:16 card): no piece of tape lies across one
    and no star sits on one. Tape comes in with its note's group, and the stars with the arrows;
    where the stars go does not depend on what is visible, since they keep off every box, every
    piece of tape, every arrow of the layout and every box to keep out of.

    The layout, the face box and the boxes to keep out of are the ones made at the plan (they are not
    scaled for the image, which is `scale` bigger): what is left out and where a star lands is decided
    there, and only drawn bigger, so that a render has the stars and the tape the studio showed."""
    extras = recipe["extras"]
    if extras["tape"]:
        _tape(img, layout, recipe["palette"], visible, keepout, scale)
    if extras["stars"] and face is not None and "arrows" in visible:
        taken, curves = obstacles(layout, recipe, keepout)
        _stars(img, taken, curves, recipe["palette"], face, scale)


def obstacles(layout: dict[str, NoteBox], recipe: dict, keepout=()) -> tuple[list, list]:
    """What the stars, and every extra that is placed as they are (styles.extras), keep off, in px of the
    plan: the boxes of the layout (the notes and the hero's boxes), the pieces of tape when the tape is on,
    and the keep-out boxes, as (x0, y0, x1, y1); and the points of the arrows' curves, as (x, y), none when
    the recipe has no arrows. The lists are new ones, for the caller to add to."""
    taken = [(box.x, box.y, box.x + box.w, box.y + box.h) for box in layout.values()]
    if recipe["extras"]["tape"]:
        taken += [_tape_box(*spot) for spot in _tape_spots(layout, keepout).values()]
    taken += [tuple(kept) for kept in keepout]
    curves = []
    if recipe["arrows"] != "none":
        curves = [point for box in layout.values() if box.arrow for point in quad_points(*box.arrow, 16)]
    return taken, curves
