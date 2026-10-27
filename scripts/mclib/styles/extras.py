"""The eleven little extras that came after tape, stars and grain: hearts, a scribble circle, a doodle
crown, sparkles, motion lines, confetti, a REC timestamp, VHS lines, a light leak, a name tag and a
speech bubble.

Two of them are part of the stage and the rest are drawn over the card:

- vhs and leak change the picture itself, as the film grain does, and engine.compose_stage calls
  stage_effects for them (the person is in the picture when they are drawn, so they cover them too).
- The other nine are drawn by draw, after the notes, onto the overlay. Each is drawn by the code, in
  the palette's colors (no emoji: the video fonts cannot draw them), and each is decided at the plan,
  1080 px wide, and only drawn bigger: where it goes is worked out in px of the plan, from the layout
  as it was made, and the drawing multiplies every size by the scale's k. So a render at the clip's own
  size has the extras the studio showed, as it has the tape and the stars.

Where an extra goes is the rule of the stars, for every one of them: it keeps off the face box, the
notes, the arrows, the hero's boxes, the boxes to keep out of (the title room, the bottom band and the
rail of a 9:16 card), the tape and the stars, and off the extras that were placed before it; it never
sits on the person, except the name tag, which is on their chest, and the scribble circle, which is
around a word of a note. An extra that finds no room is left out, never drawn over something. They
are placed in a fixed order (the circle and the REC text first, then the name tag and the bubble, the
doodles, and the confetti last), whatever is visible at the moment, so that nothing jumps when a group of
the card comes in.

The groups are the tape's and the stars': the hearts, the crown, the sparkles and the motion lines come
in with the arrows, the circle with the note whose word it rings, the name tag with the notes (the first
of the two groups), and the bubble with the arrows too. The confetti, the REC text, vhs and leak are on
from the first frame of the card.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

from ..layout import _FACE_AIR, _MARGIN
from ..palette import contrast, relative_luminance
from ..project import NOTE_KEYS
from ..raster import (
    IDENTITY,
    Scale,
    draw_circle,
    draw_path,
    draw_polygon_aa,
    dilate,
    draw_text,
    font,
    screen,
)
from . import background, notes

# --- what a recipe's extras are, and what is placed -----------------------------------------------

_EDGE = 6  # px of the stage's edge that no doodle touches
_RED = "#ff2a2a"  # the REC dot, whatever the palette is
_WHITE, _OUTLINE = "#ffffff", "#0d0d0d"  # the REC text and the outline that lets it read on any room
_PAPER = "#fffdf5"  # the paper of the name tag's body and of the speech bubble


@dataclass
class _Room:
    """What a new extra has to keep off, in px of the plan: boxes that are taken (the notes, the face,
    the keep-out boxes, the tape, the stars and every extra placed so far), the points of the arrows'
    curves, and the person's mask. fits asks whether a box is free, and claim takes it."""

    width: int
    height: int
    taken: list
    curves: np.ndarray  # n x 2, may have no rows
    person: np.ndarray | None  # height x width, True where the person is

    def fits(self, box, margin: float = 0.0, person_max: float | None = 0.0, near: float = 6.0) -> bool:
        """True when the box (left, top, right, bottom) is on the stage, clear of every taken box by
        `margin` px and of every point of an arrow's curve by `near`, and covers no more than
        `person_max` of its area with the person (None: it does not matter)."""
        left, top, right, bottom = box
        if left < _EDGE or top < _EDGE or right > self.width - _EDGE or bottom > self.height - _EDGE:
            return False
        for a, b, c, d in self.taken:
            if right + margin > a and left - margin < c and bottom + margin > b and top - margin < d:
                return False
        if len(self.curves):
            across, down = self.curves[:, 0], self.curves[:, 1]
            if ((across > left - near) & (across < right + near) & (down > top - near) & (down < bottom + near)).any():
                return False
        if person_max is not None and self.person is not None:
            share = self.person[max(int(top), 0) : int(bottom) + 1, max(int(left), 0) : int(right) + 1].mean()
            if share > person_max:
                return False
        return True

    def claim(self, box, margin: float = 0.0) -> None:
        left, top, right, bottom = box
        self.taken.append((left - margin, top - margin, right + margin, bottom + margin))


@dataclass
class _Item:
    """One thing an extra draws: the groups of the card it comes in with, and how it is painted onto
    the overlay `scale` bigger than the plan. `when` is None for one that is on the whole card."""

    when: Callable[[frozenset], bool] | None
    paint: Callable[[Image.Image, Scale], None]
    box: tuple | None = None  # the box it took in the room, in px of the plan (None: it took none)


def _arrows_on(visible) -> bool:
    return "arrows" in visible


def _notes_on(visible) -> bool:
    return "bottom" in visible or "top" in visible


def _note_group(key: str) -> Callable[[frozenset], bool]:
    group = notes.group(key)
    return lambda visible: group in visible


def _box_of(points, pad: float = 0.0) -> tuple[float, float, float, float]:
    xs, ys = zip(*points)
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def _scaled(points, scale: Scale):
    return [(scale.x(x), scale.y(y)) for x, y in points]


def _color_on(bg: str, pal: dict, roles) -> str:
    """The first of the palette's `roles` that reads on the color `bg` (a contrast of 3 or more), or
    else the one that reads best, so that a doodle is never lost in the room it is drawn on."""
    best = max(roles, key=lambda role: contrast(pal[role], bg))
    for role in roles:
        if contrast(pal[role], bg) >= 3.0:
            return pal[role]
    return pal[best]


def _wobbly(points, rng: random.Random, amount: float):
    """Points moved by a smooth hand-made wobble of up to about `amount` px: two sines of their own
    frequency and phase along the line, in each direction."""
    phases = [rng.uniform(0, 2 * math.pi) for _ in range(4)]
    count = max(len(points) - 1, 1)
    moved = []
    for index, (x, y) in enumerate(points):
        t = index / count * 2 * math.pi
        moved.append((
            x + amount * (0.6 * math.sin(2.3 * t + phases[0]) + 0.4 * math.sin(4.1 * t + phases[1])),
            y + amount * (0.6 * math.sin(2.1 * t + phases[2]) + 0.4 * math.sin(3.7 * t + phases[3])),
        ))
    return moved


# --- placing: the order of the extras, and what they keep off -----------------------------------------


def _room(layout, recipe, face, keepout, width, height, person) -> _Room:
    """The room the new extras share: everything notes.draw_extras keeps the stars off (the boxes of
    the layout, the tape when it is on, the keep-out boxes, the arrows' curves), the stars themselves
    when they are on (where they go is worked out here exactly as the stars work it out, so that the two
    agree), and the face box."""
    taken, curves = notes.obstacles(layout, recipe, keepout)
    extras = recipe["extras"]
    stars = []
    if extras["stars"] and face is not None:
        for x, y, radius in notes._star_spots(taken, curves, face, width, height):
            reach = radius + notes._STAR_WIDTH / 2 + 6
            stars.append((x - reach, y - reach, x + reach, y + reach))
    taken = taken + stars
    if face is not None:
        taken.append(tuple(face))
    return _Room(width, height, taken, np.array(curves, float).reshape(-1, 2), person)


def plan(layout, recipe, copy, face, keepout, width: int, height: int, person, variant: int = 0) -> dict[str, list[_Item]]:
    """Where every extra that is on goes, as {extra id: its items}, in the order they were placed (see
    _PLANNERS). Everything is in px of the plan, `width` x `height`."""
    extras = recipe["extras"]
    wanted = [key for key in _PLANNERS if extras.get(key)]
    if not wanted:
        return {}
    room = _room(layout, recipe, face, keepout, width, height, person)
    pal, bg = recipe["palette"], background.tone(recipe["bg"], recipe["palette"])
    return {key: _PLANNERS[key](room, layout, recipe, copy, face, keepout, pal, bg, variant) for key in wanted}


def draw(
    img: Image.Image,
    layout: dict,
    recipe: dict,
    copy: dict,
    visible,
    face,
    keepout=(),
    scale: Scale = IDENTITY,
    person: np.ndarray | None = None,
    variant: int = 0,
) -> None:
    """Draw the overlay extras that are on, in place on an RGBA image the size of the stage.

    layout, face and keepout are the ones made at the plan (notes.draw_extras takes the same), person
    is the plan's mask of the person (MatteContext.person, True where they are) or None, and variant is
    the boil picture, which only the confetti follows. The image is `scale` bigger than the plan."""
    width, height = img.size if scale.identity else scale.plan
    items = plan(layout, recipe, copy, face, keepout, width, height, person, variant)
    visible = frozenset(visible)
    for key in _PAINT_ORDER:
        for item in items.get(key, ()):
            if item.when is None or item.when(visible):
                item.paint(img, scale)


# --- hearts -------------------------------------------------------------------------------------------

# The hearts are a doodle cluster, as (half width, tilt in degrees, the offsets from the big one, across and down,
# that it tries in turn): one big
# heart (the others are about 1.6 and 2 times smaller) and two smaller ones, each with its own tilt, offset
# diagonally from it, so they never stand in a column or a row. On the left of the head the offsets are mirrored.
_HEARTS = (
    (30, -20, ((0, 0),)),
    (20, 12, ((50, -40), (46, 42), (0, -62), (0, 64))),
    (17, -5, ((28, 52), (-30, 56), (70, 8), (-24, -58))),
)
_HEART_WIDTH = 4
_NUDGES = (0, *(step * sign for step in range(12, 145, 12) for sign in (-1, 1)))


def _heart_points(cx: float, cy: float, r: float, tilt: float, rng: random.Random):
    """The outline of a hand-drawn heart of half width r around (cx, cy): the classic heart curve, which
    starts at the dip at the top and goes round, a little past where it began, so that the line does
    not meet itself (a pen that was lifted), wobbling as a hand does, and turned `tilt` degrees."""
    count = 56
    start, end = 0.12, 2 * math.pi + 0.34
    raw = []
    for index in range(count + 1):
        t = start + (end - start) * index / count
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        grow = 1 + 0.05 * index / count  # the closing line lands a little outside the start
        raw.append((x * grow, y * grow))
    wobbled = _wobbly(raw, rng, 0.9)
    left, top, right, bottom = _box_of(wobbled)
    mid_x, mid_y = (left + right) / 2, (top + bottom) / 2
    k = r / 16
    return notes._turn([(cx + (x - mid_x) * k, cy + (y - mid_y) * k) for x, y in wobbled], (cx, cy), tilt)


def _heart_cluster(room, cx: float, cy: float, mirror: int):
    """The hearts of the cluster that find a place with the big one at (cx, cy): the big heart has to fit, and each
    smaller one takes the first of its offsets (mirrored for the left side of the head) where it fits and is clear of
    the hearts before it. Returns [(x, y, half width, tilt, box)], empty when the big one does not fit."""
    found = []
    for r, tilt, offsets in _HEARTS:
        for dx, dy in offsets:
            hx, hy = cx + mirror * dx, cy + dy
            box = (hx - r - 6, hy - r - 6, hx + r + 6, hy + r + 6)
            near = any(box[0] < b[2] + 4 and b[0] < box[2] + 4 and box[1] < b[3] + 4 and b[1] < box[3] + 4 for *_, b in found)
            if not near and room.fits(box, margin=4):
                found.append((hx, hy, r, mirror * tilt, box))
                break
        else:
            if not found:
                return []
    return found


def _plan_hearts(room, layout, recipe, copy, face, keepout, pal, bg, variant) -> list[_Item]:
    """A cluster of hearts near the top of the head (see _HEARTS), beside the face box: the big heart takes the first
    free place on the right of the box, else on its left, near the box before far, from the top of the head down, and
    the smaller ones take their offsets from it where there is room. A cluster of two or more is preferred, and the
    first that has them wins; a big heart alone is only the last resort."""
    if face is None:
        return []
    rng = random.Random(101)
    x0, y0, x1, y1 = face
    color = _color_on(bg, pal, ("edge", "ink", "light"))
    best: list = []
    for side, mirror in (("R", 1), ("L", -1)):
        for gap in (_HEARTS[0][0] + 14, _HEARTS[0][0] + 46):
            for nudge in _NUDGES:
                cx = x1 + gap if side == "R" else x0 - gap
                found = _heart_cluster(room, cx, y0 + 0.2 * (y1 - y0) + nudge, mirror)
                if len(found) >= 2:
                    best = found
                    break
                best = best or found
            if len(best) >= 2:
                break
        if len(best) >= 2:
            break
    items = []
    for hx, hy, r, tilt, box in best:
        points = _heart_points(hx, hy, r, tilt, rng)
        room.claim(box, margin=4)
        items.append(_Item(
            _arrows_on,
            lambda img, scale, points=points: draw_path(img, _scaled(points, scale), color, scale.px(_HEART_WIDTH)),
            box,
        ))
    return items


# --- crown --------------------------------------------------------------------------------------------

_CROWN_WIDTHS = (150, 120, 92)
_CROWN_TILT = 7
_CROWN_WIDTH = 5


def _crown_points(cx: float, bottom: float, w: float, tilt: float, rng: random.Random):
    """The outline of a doodle crown, w wide and 0.62 w tall, its base on the line `bottom` under
    (cx, bottom - h / 2): three points on a base, each tip with a dot, drawn with a wobbling hand and
    turned. Returns (the outline, the dots' centers), and the turn's center is the crown's middle."""
    h = 0.62 * w
    left, right, top = cx - w / 2, cx + w / 2, bottom - h
    low = bottom - 0.30 * h  # the valleys
    outline = [
        (left, bottom), (left - 0.03 * w, top + 0.12 * h), (left + 0.25 * w, low), (cx, top),
        (right - 0.25 * w, low), (right + 0.03 * w, top + 0.12 * h), (right, bottom), (left, bottom),
    ]
    # each side of the outline is cut into pieces, so that the wobble bends the straight sides too
    pieces = []
    for (ax, ay), (bx, by) in zip(outline, outline[1:]):
        pieces += [(ax + (bx - ax) * t / 4, ay + (by - ay) * t / 4) for t in range(4)]
    pieces.append(outline[-1])
    pieces = _wobbly(pieces, rng, 1.6)
    dots = [outline[1], outline[3], outline[5]]
    middle = (cx, bottom - h / 2)
    return notes._turn(pieces, middle, tilt), notes._turn(dots, middle, tilt)


def _plan_crown(room, layout, recipe, copy, face, keepout, pal, bg, variant) -> list[_Item]:
    """An outline crown standing on the top of the face box (the top of the head), a little tilted. It
    is left out when the head touches the top of the stage or there is no room above it."""
    if face is None:
        return []
    rng = random.Random(202)
    x0, y0, x1, y1 = face
    color = _color_on(bg, pal, ("edge", "ink", "light"))
    for w in _CROWN_WIDTHS:
        cx, bottom = (x0 + x1) / 2, y0 - 4
        outline, dots = _crown_points(cx, bottom, w, _CROWN_TILT, rng)
        # The turn drops one corner of the base below the line, so the crown is lifted until its lowest
        # point, with its stroke and a little air, is above the face box.
        lift = max(0.0, _box_of(outline, pad=_CROWN_WIDTH)[3] + 4 - y0)
        outline, dots = [(x, y - lift) for x, y in outline], [(x, y - lift) for x, y in dots]
        box = _box_of(outline, pad=_CROWN_WIDTH)
        if room.fits(box, margin=3):
            room.claim(box, margin=3)
            return [_Item(
                _arrows_on,
                lambda img, scale, outline=outline, dots=dots: _paint_crown(img, outline, dots, color, scale),
                box,
            )]
    return []


def _paint_crown(img, outline, dots, color, scale) -> None:
    draw_path(img, _scaled(outline, scale), color, scale.px(_CROWN_WIDTH))
    for point in _scaled(dots, scale):
        draw_circle(img, point, scale.px(5.5), color, 0)


# --- sparkles -------------------------------------------------------------------------------------------

_SPARKLE_SIZES = (21, 16, 12)  # radius of the points, biggest first
_SPARKLE_REACH = (22, 84)  # px from the outline: the nearest and the farthest a sparkle's middle is
_SPARKLE_CELL = 6  # px of the plan: the grid the outline's neighborhood is found on


def _grown(mask: np.ndarray, cells: int) -> np.ndarray:
    """A coarse mask grown by `cells` in every direction (a square window, one pass along each axis)."""
    result = mask.copy()
    for axis in (0, 1):
        padded = np.pad(result, [(cells, cells) if a == axis else (0, 0) for a in (0, 1)])
        length = result.shape[axis]
        windows = [np.take(padded, range(shift, shift + length), axis=axis) for shift in range(2 * cells + 1)]
        result = np.logical_or.reduce(windows)
    return result


def _outline_cells(person: np.ndarray) -> list[tuple[float, float]]:
    """Middles, in px of the plan, of the cells of a coarse grid that lie near the person's outline but
    outside them: between the nearest and the farthest reach of the sparkles."""
    cell = _SPARKLE_CELL
    rows, columns = person.shape[0] // cell, person.shape[1] // cell
    if rows == 0 or columns == 0:
        return []
    coarse = person[: rows * cell, : columns * cell].reshape(rows, cell, columns, cell).mean(axis=(1, 3)) > 0.5
    near, far = (round(reach / cell) for reach in _SPARKLE_REACH)
    ring = _grown(coarse, far) & ~_grown(coarse, near)
    ys, xs = np.nonzero(ring)
    return [((x + 0.5) * cell, (y + 0.5) * cell) for x, y in zip(xs, ys)]


def _sparkle_points(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """A four-point sparkle: long points up, down, left and right, and short corners between them."""
    points = []
    for index in range(8):
        radius = r if index % 2 == 0 else 0.26 * r
        angle = index * math.pi / 4
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _plan_sparkles(room, layout, recipe, copy, face, keepout, pal, bg, variant) -> list[_Item]:
    """Three to five four-point sparkles near the person's outline, on the room just outside it. Their
    places are drawn from the ring of cells around the outline by a fixed seed, and each needs a free
    box and some air from the others."""
    if room.person is None:
        return []
    rng = random.Random(303)
    cells = _outline_cells(room.person)
    rng.shuffle(cells)
    wanted = rng.randint(3, 5)
    color = _color_on(bg, pal, ("edge", "ink", "light"))
    items = []
    for cx, cy in cells[:400]:
        if len(items) == wanted:
            break
        for r in _SPARKLE_SIZES:
            box = (cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3)
            if room.fits(box, margin=2):
                room.claim(box, margin=46)  # (air: the sparkles are scattered, not in a clump)
                points = _sparkle_points(cx, cy, r)
                items.append(_Item(
                    _arrows_on,
                    lambda img, scale, points=points: draw_polygon_aa(img, _scaled(points, scale), color),
                    box,
                ))
                break
    return items


# --- motion lines ---------------------------------------------------------------------------------------

_MOTION_ANGLES = (-62, -32, -2)  # degrees: the strokes fan out from the side of the head, up and out
_MOTION_LENGTHS = (46, 52, 40)
_MOTION_FROM = 24  # px from the middle of the fan to where a stroke starts (the strokes stay apart)
_MOTION_WIDTH = 5


def _plan_motion(room, layout, recipe, copy, face, keepout, pal, bg, variant) -> list[_Item]:
    """Three short strokes that radiate from beside the top of the head, the manga lines of surprise:
    on the right of the face box, else on its left, at the first height that is free."""
    if face is None:
        return []
    x0, y0, x1, y1 = face
    color = _color_on(bg, pal, ("ink", "edge", "light"))
    for side in ("R", "L"):
        for nudge in (0, 14, 28, 42, 56, 70):
            hub = (x1 + 14, y0 + 0.14 * (y1 - y0) + nudge) if side == "R" else (x0 - 14, y0 + 0.14 * (y1 - y0) + nudge)
            strokes = []
            for angle, length in zip(_MOTION_ANGLES, _MOTION_LENGTHS):
                theta = math.radians(angle if side == "R" else 180 - angle)
                start = (hub[0] + _MOTION_FROM * math.cos(theta), hub[1] + _MOTION_FROM * math.sin(theta))
                end = (hub[0] + (_MOTION_FROM + length) * math.cos(theta), hub[1] + (_MOTION_FROM + length) * math.sin(theta))
                strokes.append((start, end))
            box = _box_of([point for stroke in strokes for point in stroke], pad=_MOTION_WIDTH)
            if room.fits(box, margin=3):
                room.claim(box, margin=3)
                return [_Item(
                    _arrows_on,
                    lambda img, scale, strokes=strokes: _paint_lines(img, strokes, color, _MOTION_WIDTH, scale),
                    box,
                )]
    return []


def _paint_lines(img, lines, color, width: float, scale: Scale) -> None:
    """Each of `lines` (lists of points, in px of the plan) as a stroke `width` px thick."""
    for line in lines:
        draw_path(img, _scaled(list(line), scale), color, scale.px(width))


# --- scribble circle ------------------------------------------------------------------------------------



def _word_ellipse(box, recipe: dict, copy: dict):
    """Where the emphasis word of `copy` is in a note's box (the first place it is whole): a list with the (center, half
    width, half height, tilt, font size, stroke) of the box of each part of it (one, unless the phrase runs over the end
    of a line), in px of the plan, for _ring to make a loop of; None when the note does not have it. It follows how each notes style sets its lines: the marker and
    pixel notes from the box's corner (plus the padding), the label's pills stacked in the middle, the
    others centered as a block, and a style that tilts its notes tilts the ellipse the same."""
    word = (copy.get("extras") or {}).get("emphasis")
    spans = notes._emphasis_spans(box.lines, word)
    if not spans:
        return None
    style = notes.note_style(recipe)
    kind = recipe["notes"]
    fnt = font(style.font, box.size)
    pitch = round(box.size * style.line_h)
    widest = max(fnt.getlength(line) for line in box.lines)
    if kind in ("marker", "pixel"):
        left0, top0, row_mid = box.x + style.pad, box.y + style.pad, pitch / 2
    elif kind == "label":
        left0 = box.x + style.pad
        top0 = box.y + (box.h - (len(box.lines) * pitch - notes._LABEL_GAP)) / 2
        row_mid = (pitch - notes._LABEL_GAP) / 2
    else:  # comic and hand: the block is centered in the box
        left0, top0, row_mid = box.x + box.w / 2 - widest / 2, box.y + box.h / 2 - pitch * len(box.lines) / 2, pitch / 2
    tilts = {"comic": notes._COMIC_TILTS, "hand": notes._HAND_TILTS, "label": notes._LABEL_TILTS}.get(kind)
    tilt = tilts[NOTE_KEYS.index(box.key)] if tilts else 0
    stroke = notes.ARROW_STYLE[kind][1]  # (the loops are as thick as the arrows, so they stand off the letters by that)
    found = []
    for number, first, end in spans:  # (a phrase that runs over the end of a line is a loop for each of its parts)
        line = box.lines[number]
        left, right = left0 + fnt.getlength(line[:first]), left0 + fnt.getlength(line[:end])
        middle = top0 + number * pitch + row_mid
        center = ((left + right) / 2, middle)
        if tilt:
            (center,) = notes._turn([center], (box.x + box.w / 2, box.y + box.h / 2), tilt)
        # (a font's widths run a few % over what getlength says)
        found.append((center, 1.04 * (right - left) / 2, 0.36 * box.size + stroke / 2 + 2, tilt, box.size, stroke))
    return found


_LOOP_SHAPE = 2.6  # the loop is a rounded rectangle between an ellipse (2) and a box: it takes in a long word's corners
# without reaching far out past its ends


_RINGS = ((1.0, 1.65), (0.5, 1.65), (0.0, 1.65), (0.0, 2.0), (0.0, 2.4))  # (air, height): the first that is clear is used


def _ring(half_w: float, half_h: float, size: float, stroke: float, air: float, tall: float) -> tuple[float, float]:
    """The half width and half height of a loop that rings a word whose box is half_w by half_h: `tall` times as
    tall as the word, and as wide as it has to be to pass outside the corners of the word's box (_LOOP_SHAPE; the
    taller the loop, the narrower it can be), with some room more at its ends for the `air` (0 to 1)."""
    ry = half_h * tall
    reach = (1 - (half_h / ry) ** _LOOP_SHAPE) ** (1 / _LOOP_SHAPE)
    return half_w / reach + (0.14 * size + stroke / 2) * air, ry


def _loop_points(cx, cy, rx, ry, tilt, rng, start: float, turns: float, grow: float):
    """One loop of the scribble: a rounded rectangle (_LOOP_SHAPE) that begins at angle `start` and goes round `turns`
    times, widening by `grow` as it goes so that its end passes outside its start, with a hand's wobble, turned by
    `tilt` degrees."""
    count = 64
    points = []
    for index in range(count + 1):
        share = index / count
        angle = start + 2 * math.pi * turns * share
        spread = 1 + grow * share
        cos, sin = math.cos(angle), math.sin(angle)
        across = math.copysign(abs(cos) ** (2 / _LOOP_SHAPE), cos)
        down = math.copysign(abs(sin) ** (2 / _LOOP_SHAPE), sin)
        points.append((cx + rx * spread * across, cy + ry * spread * down))
    return notes._turn(_wobbly(points, rng, 1.4), (cx, cy), tilt)


def _plan_circle(room, layout, recipe, copy, face, keepout, pal, bg, variant) -> list[_Item]:
    """A hand-drawn loop, an imperfect ellipse drawn twice and a little offset, around the emphasis word
    where it is in a note. There is nothing without an emphasis word, or when no note has it. It keeps off
    the face box, the keep-out boxes and the other notes (the note it rings is its own), and gives up the
    air around the word, down to 62% of it, to do that; with no room it is left out."""
    word = ((copy.get("extras") or {}).get("emphasis") or "").strip()
    if not word:
        return []
    for key in NOTE_KEYS:
        box = layout.get(key)
        if box is None or not box.lines or (key == "tr" and recipe["hero"] == "stats"):
            continue
        fragments = _word_ellipse(box, recipe, copy)
        if fragments is None:
            continue
        others = [(b.x, b.y, b.x + b.w, b.y + b.h) for name, b in layout.items() if name != key]
        others += [tuple(kept) for kept in keepout]
        items = []
        for center, word_w, word_h, tilt, size, stroke in fragments:
            turn = math.radians(tilt)
            # the face box with 20 px of its air given up first, and, where nothing else is clear, all of its air (the
            # box is the head and then 4% of the stage's width of air at its sides): never the head itself
            for given, (air_share, tall) in itertools.product((_FACE_AIR_GIVEN, round(_MARGIN * room.width)), _RINGS):
                kept_off = others + ([_inset(face, given, room.width)] if face is not None else [])
                hx, hy = _ring(word_w, word_h, size, stroke, air_share, tall)
                air = 0.06 * max(hx, hy) + 3 + stroke / 2  # (the second loop is a little bigger, and wanders)
                half_w = math.hypot(hx * math.cos(turn), hy * math.sin(turn)) + air
                half_h = math.hypot(hx * math.sin(turn), hy * math.cos(turn)) + air
                around = (center[0] - half_w, center[1] - half_h, center[0] + half_w, center[1] + half_h)
                if around[0] < _EDGE or around[1] < _EDGE or around[2] > room.width - _EDGE or around[3] > room.height - _EDGE:
                    continue
                if any(around[2] > a and around[0] < c and around[3] > b and around[1] < d for a, b, c, d in kept_off):
                    continue
                rng = random.Random(404 + len(items))
                loops = [
                    _loop_points(*center, hx, hy, tilt, rng, -0.5, 1.12, 0.08),
                    _loop_points(center[0] + 3, center[1] - 2, hx * 1.05, hy * 1.12, tilt, rng, 0.9, 1.1, 0.07),
                ]
                color, thick = _circle_ink(recipe)
                room.claim(around)
                others.append(around)
                items.append(_Item(
                    _note_group(key), lambda img, scale, loops=loops, color=color, thick=thick: _paint_lines(img, loops, color, thick, scale),
                    around,
                ))
                break
        if items:
            return items
    return []


_FACE_AIR_GIVEN = 20  # px of the air around the face box that a loop round a word may use first (not the head itself)


def _inset(box, by: float, width: int):
    """A box pulled in by `by` px at its left and right, where it is not at the stage's edge."""
    x0, y0, x1, y1 = box
    return x0 + by if x0 > 0 else x0, y0, x1 - by if x1 < width else x1, y1


def _circle_ink(recipe: dict) -> tuple[str, int]:
    """The color and the stroke of the scribble: the ink the ringed note's text is set in (the notes style's own,
    which the contrast guard measures, so it reads on what the note sits on: the ink for marker, handwritten and
    pixel notes, #111111 for comic ones, the dark color for the bold labels) and as thick as that style's arrows."""
    kind, pal = recipe["notes"], recipe["palette"]
    color = {"comic": notes._COMIC_TEXT, "label": pal["dark"]}.get(kind, pal["ink"])
    return color, notes.ARROW_STYLE[kind][1]


# --- speech bubble ----------------------------------------------------------------------------------------

_BUBBLE_SIZES = (46, 40, 34, 28)  # the lettering, biggest first
_BUBBLE_PAD = (24, 14)  # px at the sides and above and below the text
_BUBBLE_TAIL = 36  # px from the bubble's side to the tip of its tail
_BUBBLE_LINE = 4
_BUBBLE_MOUTH = 0.76  # how far down the face box (without its air) the mouth is


def _mouth_of(face, stage_height: int) -> tuple[float, float]:
    """About where the mouth is, from the face box alone: in the middle of it across, and 76% of the way
    from the top of the head to the chin, which is the box's bottom less its air."""
    x0, y0, x1, y1 = face
    chin = y1 - round(_FACE_AIR * stage_height)
    return (x0 + x1) / 2, y0 + _BUBBLE_MOUTH * (max(chin, y0 + 1) - y0)


def _bubble_shape(x: float, y: float, w: float, h: float, side: str, tip, radius: float):
    """The outline of a speech bubble: a rounded box at (x, y), w by h, with a tail on its `side` ("L" or
    "R") whose tip is `tip`, as one polygon, so that the outline runs round the tail and not across its
    base. The points go clockwise from the top right corner's arc, as notes._rounded gives them."""
    points = notes._rounded(x, y, x + w, y + h, radius)
    steps = len(points) // 4
    base = 11  # half the width of the tail where it leaves the box
    reach = min(max(tip[1], y + radius + base), y + h - radius - base)
    if side == "L":  # the left edge runs from the bottom left arc up to the top left arc
        tail = [(x, reach + base), tip, (x, reach - base)]
        at = 3 * steps
    else:  # the right edge runs from the top right arc down to the bottom right arc
        tail = [(x + w, reach - base), tip, (x + w, reach + base)]
        at = steps
    return points[:at] + tail + points[at:]


def _plan_bubble(room, layout, recipe, copy, face, keepout, pal, bg, variant) -> list[_Item]:
    """A speech bubble with the text of copy.extras.bubble, beside the face box at the height of the
    mouth, its tail pointing at the mouth and stopping at the edge of the box. On the right of the face
    first and else on the left; the lettering comes down in size to fit. Nothing when the text is
    empty or there is no room."""
    text = ((copy.get("extras") or {}).get("bubble") or "").strip()
    if not text or face is None:
        return []
    mouth = _mouth_of(face, room.height)
    x0, y0, x1, y1 = face
    for size in _BUBBLE_SIZES:
        fnt = font("marker", size)
        w = fnt.getlength(text) + 2 * _BUBBLE_PAD[0]
        h = size + 2 * _BUBBLE_PAD[1]
        if w > 330:
            continue
        for side in ("R", "L"):
            for nudge in _NUDGES:
                top = mouth[1] - h / 2 - 22 + nudge
                left = x1 + _BUBBLE_TAIL + 8 if side == "R" else x0 - _BUBBLE_TAIL - 8 - w
                box = (
                    (left - _BUBBLE_TAIL, top, left + w, top + h) if side == "R" else (left, top, left + w + _BUBBLE_TAIL, top + h)
                )
                if room.fits((box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3), margin=2):
                    tip = (x1 + 8, mouth[1]) if side == "R" else (x0 - 8, mouth[1])
                    room.claim(box, margin=5)
                    outline = _bubble_shape(left, top, w, h, "L" if side == "R" else "R", tip, min(h / 2, 22))
                    return [_Item(
                        _arrows_on,
                        lambda img, scale, outline=outline, text=text, size=size, left=left, top=top, w=w, h=h: _paint_bubble(
                            img, outline, text, size, (left, top, w, h), pal, scale
                        ),
                        box,
                    )]
    return []


def _paint_bubble(img, outline, text, size, box, pal, scale) -> None:
    left, top, w, h = box
    draw_polygon_aa(img, _scaled(outline, scale), _PAPER, pal["dark"], scale.px(_BUBBLE_LINE))
    fnt = font("marker", scale.font(size))
    pitch = scale.font(size)
    x = scale.x(left + w / 2) - fnt.getlength(text) / 2
    y = scale.y(top + h / 2) - pitch / 2
    draw_text(img, (x, y), [text], fnt, pal["dark"], line_h=pitch)


# --- name tag ---------------------------------------------------------------------------------------------

_TAG_BIG = 350  # px: the tag's width where the chest has room for it (1.4 x 250), within _TAG_BIG_DOWN px of the face
_TAG_BIG_DOWN = 60
_TAG_WIDTHS = (250, 210, 170, 140, 120)  # px, the fallbacks, biggest first; the tag is 0.68 as tall as it is wide
_TAG_TILT = -4
_TAG_BAND = 0.40  # the share of the tag's height that is the red band
_TAG_RED = "#d92b2b"
_TAG_ON_PERSON = 0.92  # how much of the tag's body must be on the person: it is stuck to their chest


def _plan_nametag(room, layout, recipe, copy, face, keepout, pal, bg, variant) -> list[_Item]:
    """A "HELLO my name is" sticker with the name of copy.name, on the person's chest just under the
    face box, a little tilted: the first place down the chest, nearest the face, where it is on the person
    and clear of the notes, at the biggest of its sizes. Nothing without a name, or without a chest."""
    name = (copy.get("name") or "").strip()
    if not name or face is None or room.person is None:
        return []
    x0, y0, x1, y1 = face
    cx0 = (x0 + x1) / 2
    steps = [(_TAG_BIG, nudge) for nudge in range(0, _TAG_BIG_DOWN + 1, 12)]
    steps += [(w, nudge) for nudge in range(0, 321, 12) for w in _TAG_WIDTHS]
    for w, nudge in steps:
        h = 0.68 * w
        for across in (0.0, -0.22, 0.22, -0.45, 0.45):
            cx, cy = cx0 + across * w, y1 + 6 + nudge + h / 2
            body = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            corners = notes._turn(notes._corners(*body), (cx, cy), _TAG_TILT)
            box = _box_of(corners, pad=3)
            if not room.fits(box, margin=3, person_max=None):
                continue
            share = room.person[
                max(int(body[1]), 0) : int(body[3]) + 1, max(int(body[0]), 0) : int(body[2]) + 1
            ].mean()
            if share >= _TAG_ON_PERSON:
                room.claim(box, margin=3)
                return [_Item(
                    _notes_on, lambda img, scale, cx=cx, cy=cy, w=w, h=h: _paint_tag(img, name, cx, cy, w, h, scale),
                    box,
                )]
    return []


def _paint_tag(img: Image.Image, name: str, cx: float, cy: float, w: float, h: float, scale: Scale) -> None:
    """The sticker as a tile of its own at the size it is drawn, turned and laid on the overlay: a white
    body with a thin dark line round it, a red band with "HELLO" and "my name is" in white, and the name in
    the hand font in dark ink, brought down in size until it fits."""
    k = scale.px(1.0)
    tw, th = round(w * k), round(h * k)
    pad = round(8 * k)
    tile = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    radius, line = 12 * k, 3 * k
    box = notes._rounded(pad, pad, pad + tw, pad + th, radius)
    draw_polygon_aa(tile, box, _PAPER, "#1a1a1a", line)
    band = th * _TAG_BAND
    top_part = [p for p in notes._rounded(pad, pad, pad + tw, pad + band + radius, radius)]
    band_box = [(x, min(y, pad + band)) for x, y in top_part]  # (the rounded top, cut flat at the band)
    draw_polygon_aa(tile, band_box, _TAG_RED)
    draw_polygon_aa(tile, box, None, "#1a1a1a", line)
    hello = font("label", round(0.20 * h * k))
    small = font("marker", round(0.105 * h * k))
    draw_text(tile, (pad + (tw - hello.getlength("HELLO")) / 2, pad + 0.06 * th), ["HELLO"], hello, _WHITE)
    draw_text(tile, (pad + (tw - small.getlength("my name is")) / 2, pad + 0.235 * th), ["my name is"], small, _WHITE)
    size = round(0.40 * h * k)
    fnt = font("hand", size)
    while fnt.getlength(name) > 0.86 * tw and size > round(0.14 * h * k):
        size -= 2
        fnt = font("hand", size)
    ink_y = pad + band + (th - band) / 2 - size * 0.62
    draw_text(tile, (pad + (tw - fnt.getlength(name)) / 2, ink_y), [name], fnt, "#1a1a1a")
    turned = tile.rotate(-_TAG_TILT, Image.Resampling.BICUBIC, expand=True)
    notes._lay(img, turned, round(scale.x(cx) - turned.width / 2), round(scale.y(cy) - turned.height / 2))


# --- REC timestamp ----------------------------------------------------------------------------------------

_REC_SIZE, _REC_STROKE, _REC_DOT = 56, 3.5, 12
_REC_TIME = "00:00:02"


def _plan_rec(room, layout, recipe, copy, face, keepout, pal, bg, variant) -> list[_Item]:
    """A camcorder's "REC" with a red dot in one top corner and its time in the other, in VT323 with a
    dark outline. Each piece takes its own corner, or the other one when that is taken (a note often sits
    in a top corner), and the first height down the side that is free. A 9:16 card's pieces start under
    the title room."""
    fnt = font("terminal", _REC_SIZE)
    margin = round(_MARGIN * room.width)
    top = margin
    for a, b, c, d in keepout:
        if a <= margin and c >= room.width - margin and b <= margin < d:
            top = max(top, d + 10)
    pieces = (
        ("rec", 2 * _REC_DOT + 12 + fnt.getlength("REC"), "L"),
        ("time", fnt.getlength(_REC_TIME), "R"),
    )
    height = _REC_SIZE * 0.78
    items = []
    for what, w, side in pieces:
        placed = False
        for use in (side, "R" if side == "L" else "L"):
            for nudge in range(0, 121, 8):
                left = margin if use == "L" else room.width - margin - w
                box = (left, top + nudge, left + w, top + nudge + height)
                if room.fits(box, margin=4, person_max=0.02):
                    room.claim(box, margin=4)
                    items.append(_Item(
                        None, lambda img, scale, what=what, box=box: _paint_rec(img, what, box, scale), box
                    ))
                    placed = True
                    break
            if placed:
                break
    return items


def _paint_rec(img: Image.Image, what: str, box, scale: Scale) -> None:
    size = scale.font(_REC_SIZE)
    fnt = font("terminal", size)
    left, top = scale.x(box[0]), scale.y(box[1])
    stroke = scale.px(_REC_STROKE)
    ascent, descent = fnt.getmetrics()
    text_top = top - (ascent + descent - scale.px(_REC_SIZE * 0.78)) / 2
    if what == "rec":
        radius = scale.px(_REC_DOT)
        middle = (left + radius, top + scale.px(_REC_SIZE * 0.78) / 2)
        draw_circle(img, middle, radius + stroke / 2, _OUTLINE, 0)
        draw_circle(img, middle, radius, _RED, 0)
        left += 2 * radius + scale.px(12)
        text = "REC"
    else:
        text = _REC_TIME
    _outlined_text(img, (left, text_top), text, fnt, _WHITE, _OUTLINE, stroke, ascent + descent)


def _outlined_text(img: Image.Image, xy, text: str, fnt, fill: str, outline: str, reach: float, line_h: int) -> None:
    """A line of text with a dark outline that grows outward only, `reach` px past the letters: the
    letters keep their full width (VT323's strokes are thin, and an outline centered on their edge, as
    notes._stroked_text draws it, would eat them). The letters go on a tile of their own, their mask is
    grown, and the outline color goes under the letters."""
    pad = math.ceil(reach) + 4
    width = math.ceil(fnt.getlength(text)) + 2 * pad
    height = line_h + 2 * pad
    left, top = math.floor(xy[0]), math.floor(xy[1])
    tile = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_text(tile, (xy[0] - left + pad, xy[1] - top + pad), [text], fnt, "#ffffff", line_h=line_h)
    mask = np.asarray(tile, np.float32)[..., 3] / 255
    for color, coverage in ((outline, dilate(mask, reach)), (fill, mask)):
        layer = np.empty((height, width, 4), np.uint8)
        layer[..., :3] = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
        layer[..., 3] = np.rint(np.clip(coverage, 0.0, 1.0) * 255)
        notes._lay(img, Image.fromarray(layer, "RGBA"), left - pad, top - pad)


# --- confetti ---------------------------------------------------------------------------------------------

_CONFETTI = (24, 12)  # a base count of 24 and up to 12 more, so 24 to 36 pieces
_CONFETTI_REACH = 15  # px: the half diagonal of the biggest piece, plus a little for its jitter
_CONFETTI_BRIGHT = 0.2  # the least luminance at which the dark or the texture color is bright enough for confetti
_CONFETTI_AIR = 12  # px that a piece keeps from the notes, the face and the other extras


def scatter(room: _Room, pal: dict, bg: str, variant: int = 0) -> list[tuple]:
    """The confetti pieces, each as (x, y, w, h, angle, color, dot, jitter x, jitter y, jitter angle): 24 to
    36 of them, as many as find a place within 600 tries (fewer only on a stage that has no room), on the
    background, never on the person, the face, the notes or what else is placed. They come from a fixed
    seed, so they are the same every time, and `variant` (the boil picture) moves each by up to 3 px and
    turns it by up to 15 degrees, a small re-scatter. Colors are the palette's light, edge and ink, and its dark or texture color only when that is bright (never the dark ones: they read as specks), that show on `bg`."""
    rng = random.Random(505)
    bright = [role for role in ("dark", "tex") if relative_luminance(pal[role]) >= _CONFETTI_BRIGHT]
    roles = [role for role in ("light", "edge", "ink", *bright) if contrast(pal[role], bg) >= 1.5]
    if not roles:
        return []
    wanted = _CONFETTI[0] + rng.randrange(_CONFETTI[1] + 1)
    boil = random.Random(900 + variant)
    pieces = []
    for _ in range(600):
        if len(pieces) == wanted:
            break
        x, y = rng.uniform(_EDGE, room.width - _EDGE), rng.uniform(_EDGE, room.height - _EDGE)
        dot = rng.random() < 0.3
        w, h = (rng.uniform(10, 14),) * 2 if dot else (rng.uniform(18, 26), rng.uniform(9, 13))
        angle, color = rng.uniform(0, 360), pal[roles[rng.randrange(len(roles))]]
        box = (x - _CONFETTI_REACH, y - _CONFETTI_REACH, x + _CONFETTI_REACH, y + _CONFETTI_REACH)
        if room.fits(box, margin=_CONFETTI_AIR, person_max=0.0):
            pieces.append((x, y, w, h, angle, color, dot, boil.uniform(-3, 3), boil.uniform(-3, 3), boil.uniform(-15, 15)))
    return pieces


def _plan_confetti(room, layout, recipe, copy, face, keepout, pal, bg, variant) -> list[_Item]:
    """The confetti, on from the first frame of the card (see scatter)."""
    pieces = scatter(room, pal, bg, variant)
    if not pieces:
        return []
    return [_Item(None, lambda img, scale, pieces=pieces: _paint_confetti(img, pieces, scale))]


def _paint_confetti(img: Image.Image, pieces, scale: Scale) -> None:
    for x, y, w, h, angle, color, dot, jx, jy, jangle in pieces:
        x, y = x + jx, y + jy
        if dot:
            draw_circle(img, (scale.x(x), scale.y(y)), scale.px(w / 2), color, 0)
            continue
        corners = notes._turn(notes._corners(x - w / 2, y - h / 2, x + w / 2, y + h / 2), (x, y), angle + jangle)
        draw_polygon_aa(img, _scaled(corners, scale), color)


# --- the order things are placed in, and painted in --------------------------------------------------------

# The order of placing: the circle (it rings a word that is already there), the REC text (it has the corners),
# the ones that stand for something (a name, a few words), then the doodles, and the scatter last.
_PLANNERS = {
    "circle": _plan_circle,
    "rec": _plan_rec,
    "nametag": _plan_nametag,
    "bubble": _plan_bubble,
    "crown": _plan_crown,
    "motion": _plan_motion,
    "hearts": _plan_hearts,
    "sparkles": _plan_sparkles,
    "confetti": _plan_confetti,
}
# And of painting, the first at the bottom: the scatter under everything, the text last.
_PAINT_ORDER = ("confetti", "circle", "hearts", "crown", "sparkles", "motion", "bubble", "nametag", "rec")


# --- the two that are part of the stage ---------------------------------------------------------------------

_VHS_LINES = (4, 1, 0.80)  # a line darker every 4 rows, 1 row thick, keeping this much of the light
_VHS_SHIFT = 1  # px each way: the red goes right and the green and blue go left, 2 px apart in all
_LEAK = (0.46, 0.60, "#ff7a1c", "#ff2d18")  # how far in it reaches (of the width), its strength, its two colors


def _slid(channel: np.ndarray, by: int) -> np.ndarray:
    """A channel moved `by` px along its rows (right for a positive number), the edge repeated."""
    if by == 0:
        return channel
    if by > 0:
        return np.concatenate([np.repeat(channel[:, :1], by, axis=1), channel[:, :-by]], axis=1)
    return np.concatenate([channel[:, -by:], np.repeat(channel[:, -1:], -by, axis=1)], axis=1)


def _vhs(img: np.ndarray, scale: Scale) -> np.ndarray:
    """Thin dark scanlines over the whole stage, and a slight red and cyan offset: the red channel 1 px to
    the right and the green and blue 1 px to the left (more at a bigger size, as every size in px is)."""
    shift = scale.n(_VHS_SHIFT, minimum=1)
    out = np.stack([_slid(img[..., 0], shift), _slid(img[..., 1], -shift), _slid(img[..., 2], -shift)], axis=-1)
    every, thick, kept = _VHS_LINES
    every, thick = scale.n(every, minimum=2), scale.n(thick, minimum=1)
    rows = np.where(np.arange(out.shape[0]) % every >= every - thick, kept, 1.0).astype(np.float32)
    return out * rows[:, None, None]


def _leak(img: np.ndarray, head) -> np.ndarray:
    """A soft warm glow, orange at the edge and red a little way in, bleeding in from the side of the stage
    that is farther from the head (the right one when nobody is there), screened over the picture so it only
    adds light. It is strongest a third of the way down and fades out across the picture."""
    height, width = img.shape[:2]
    reach, strength, near, far = _LEAK
    middle = (head[0] + head[2]) / 2 if head is not None else 0.0
    from_right = middle <= width / 2
    across = np.arange(width, dtype=np.float32)
    away = (width - 1 - across if from_right else across) / (reach * width)  # 0 at the edge, 1 where it ends
    fade = np.exp(-3.2 * np.clip(away, 0.0, None) ** 1.3).astype(np.float32)
    down = (np.arange(height, dtype=np.float32) / height - 0.35) / 0.55
    around = np.exp(-down**2).astype(np.float32)
    amount = strength * around[:, None] * fade[None, :]
    redder = np.clip(away, 0.0, 1.0).astype(np.float32)
    warm = np.array([int(near[i : i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)
    deep = np.array([int(far[i : i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)
    color = warm[None, None, :] + (deep - warm)[None, None, :] * redder[None, :, None]
    return screen(img, color, amount)


def stage_effects(recipe: dict, img: np.ndarray, head, scale: Scale = IDENTITY) -> np.ndarray:
    """The stage with the recipe's vhs and leak laid on it, in that order, when they are on (the stage
    as it is when neither is). img is the composed stage, floats of 0-1; head is the head's box (x0, y0,
    x1, y1) or None."""
    extras = recipe["extras"]
    if not (extras.get("vhs") or extras.get("leak")):
        return img
    if extras.get("vhs"):
        img = _vhs(img, scale)
    if extras.get("leak"):
        img = _leak(img, head)
    return np.clip(img, 0.0, 1.0)
