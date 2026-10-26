"""Tests for the note layout: the quadrants and the face box, fitting a note's text, the search that
gives every note of a card one size and a place, and the arrows from the notes to the person.

Two people: conftest's synthetic close shot at 1080 x 608, the size of the stage, and
tests/fixtures/close_alpha.png, the matte of a real close-up (a man in a beanie and a hoodie who
fills the frame) taken from docs/mockups. It is a silhouette only, and its use was approved. (A
third, tests/fixtures/vertical_alpha.png, his vertical photo at 1080 x 1920, is only used to read the
face box of a head with a hood, and a fourth, tests/fixtures/threequarter_alpha.png, the freeze of a
clip of him turned three quarters, to read the face box of a head with a hood on one side and to lay
out the notes around it.) The
notes are the sample copy in the marker style of the Bedroom Pop recipe: 56 px down to 34, capitals,
and a 22 px sticker edge. Widths and sizes come from the bundled fonts, so the tests ask the fonts
for them. The layout is also run on the same mattes moved, doubled, raised or stretched to a full
screen, because a picture is never framed exactly like the test one.

The checks that do not depend on how the layout works are written out the slow, plain way here
(pixel counts of boxes, a search of every spot, every way of setting a text), so that a mistake in
the layout's own tables and shortcuts cannot also be in the test.
"""

import dataclasses
import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mclib import layout
from mclib.analyze import person_boxes
from mclib.layout import (
    NoteStyle, apply_case, face_box, fit_note, plan_notes, prepare_matte, quadrants,
    silhouette_target, wrap,
)
from mclib.raster import dilate, font, quad_points

W, H = 1080, 608
STYLE = NoteStyle("marker", 56, 34, "upper", band=22)
QUIET = dataclasses.replace(STYLE, arrows=False)
TEXTS = {
    "tl": "Creator & AI educator",
    "tr": "@your_handle",
    "bl": "But overthinks every 6-second reel",
    "br": "Best friend: Clover (sorry, mom)",
}
KEYS = ["tl", "tr", "bl", "br"]
ELLIPSIS = chr(0x2026)  # built with chr(), not typed, so no tool can change it on the way in
CLOSE_ALPHA = Path(__file__).parent / "fixtures" / "close_alpha.png"
VERTICAL_ALPHA = Path(__file__).parent / "fixtures" / "vertical_alpha.png"
THREEQUARTER_ALPHA = Path(__file__).parent / "fixtures" / "threequarter_alpha.png"
TALL_KEEPOUT = [(0, 0, 1080, 300), (0, 1500, 1080, 1920), (940, 960, 1080, 1500)]
# the title room, the bottom band and the rail of buttons of a 9:16 card (what card._keepout gives)
CHIN = 506  # the lowest row of the beard in close_alpha.png, read off the photo it was cut from
MARGIN = round(0.04 * W)  # 43: the notes' gap to the edge of the stage and to the face
MAX_W = round(0.42 * W)  # 454: the widest a note's box gets
DEFAULT_FACE = (389, 49, 691, 334)  # the face with nobody there: (0.36 W, 0.08 H, 0.64 W, 0.55 H)
SIZES = [*range(STYLE.max_size, STYLE.min_size, -2), STYLE.min_size]  # 56, 54, ... 34


def close_alpha():
    """The real close-up's matte, float32 0-1, 1080 x 608."""
    with Image.open(CLOSE_ALPHA) as image:
        return np.asarray(image.convert("L"), np.float32) / 255


def vertical_alpha():
    """A vertical photo's matte (RVM), float32 0-1, 1080 x 1920: the same man, in the beanie and
    the hoodie, sitting back, so that his hood and shoulders widen the silhouette gradually."""
    with Image.open(VERTICAL_ALPHA) as image:
        return np.asarray(image.convert("L"), np.float32) / 255


def threequarter_alpha():
    """The matte (RVM) of the freeze of a clip of a man turned three quarters, float32 0-1, 1080 x 1920 (the
    freeze of proj_tall): in the beanie and the hoodie, looking out of a window. His head is 340 px wide
    (columns 234 to 572, widest at row 486), and the hood stands out to the right of it at row 598, where the
    right edge of the silhouette goes from 560 to 685 in two rows, and then widens into the shoulders, so
    that the person is never narrower than the head after the jaw: the rows at face height already hold the
    hood."""
    with Image.open(THREEQUARTER_ALPHA) as image:
        return np.asarray(image.convert("L"), np.float32) / 255


def three_quarter(ramp=1, width=1080, height=1920):
    """A head turned three quarters, with its hood on one side: an ellipse 336 wide and 420 tall (its top at
    row 286, its widest row 496, the numbers of the photo), and from row 598 a collar that stands out 120 px
    to the right of the head's right side, in `ramp` rows, and shoulders under it that widen the silhouette to
    the bottom of the stage. It is never narrower than the head after the jaw, so no neck shows."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32) + 0.5
    head = ((xx - 403) / 168) ** 2 + ((yy - 496) / 210) ** 2 <= 1
    collar = (yy >= 598) & (xx >= 280) & (xx <= 570 + 120 * np.clip((yy - 597) / ramp, 0, 1))
    shoulders = (yy >= 700) & (xx >= 270) & (xx <= 690 + np.clip((yy - 700) * 0.45, 0, 165))
    return (head | collar | shoulders).astype(np.float32)


def blob(cx, cy, rx, ry):
    """The pixels of an ellipse on the 1080 x 608 stage, as a boolean mask."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def block(rows, columns):
    """The pixels of a rectangle on the 1080 x 608 stage, from slices of rows and of columns."""
    mask = np.zeros((H, W), bool)
    mask[rows, columns] = True
    return mask


def with_parts(alpha, *parts):
    """A matte with boolean masks added to it."""
    return np.maximum(alpha, np.any(parts, axis=0).astype(np.float32))


def hooded(width=1080, height=1920, bottom_w=2.4):
    """A head, an ellipse 380 wide and 500 tall with its top at row 500, on a trapezoid that starts at
    the head's widest row with the head's own width and widens a little with every row, to bottom_w
    times that at the bottom of the picture: a hood and shoulders, with no neck and no row much wider
    than the one above it."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32) + 0.5
    cx, cy, rx, ry = width / 2, 750, 190, 250
    head = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    half = rx * (1 + (bottom_w - 1) * np.clip((yy - cy) / (height - cy), 0, 1))
    return (head | ((yy >= cy) & (np.abs(xx - cx) <= half))).astype(np.float32)


def plain_head(width=W, height=H):
    """A head, an ellipse 240 wide and 300 tall, on a short neck and then sharp shoulders: the width
    goes from the head's to the neck's to the shoulders' in a step, as a body does with no hood."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32) + 0.5
    head = ((xx - 540) / 120) ** 2 + ((yy - 200) / 150) ** 2 <= 1
    neck = (yy >= 340) & (yy < 400) & (np.abs(xx - 540) <= 50)
    shoulders = (yy >= 400) & (np.abs(xx - 540) <= 330)
    return (head | neck | shoulders).astype(np.float32)


def moved(alpha, right=0, up=0):
    """A matte moved right (left when negative), the empty stage showing behind it, and moved up
    with its last row repeated below, so that a body that ran off the bottom still does."""
    out = np.zeros_like(alpha)
    if right >= 0:
        out[:, right:] = alpha[:, : alpha.shape[1] - right]
    else:
        out[:, :right] = alpha[:, -right:]
    if up:
        out = np.concatenate([out[up:], np.repeat(out[-1:], up, axis=0)])
    return out


def two_people(person_alpha):
    """Two copies of the synthetic person side by side, each in half the stage's width."""
    both = np.zeros((H, W), np.float32)
    both[:, : W // 2] = person_alpha(W // 2, H)
    both[:, W // 2 :] = person_alpha(W // 2, H)
    return both


def raised_arms(alpha):
    """The synthetic person with both arms up: two bars from the shoulders to above the eyes."""
    out = alpha.copy()
    out[150:400, 250:330] = 1.0
    out[150:400, 750:830] = 1.0
    return out


def full_screen(alpha):
    """A 1080 x 608 matte on a 1080 x 1920 stage: its shape kept (it is 1080 wide already), in the
    middle, with bare stage above and below it (656 rows each)."""
    tall = np.zeros((1920, 1080), np.float32)
    tall[656 : 656 + alpha.shape[0]] = alpha
    return tall


def rect(box):
    """A NoteBox as (x0, y0, x1, y1)."""
    return box.x, box.y, box.x + box.w, box.y + box.h


def overlap(a, b):
    """The pixels two (x0, y0, x1, y1) boxes share."""
    return max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)


def inside(box, outer):
    """Whether an (x0, y0, x1, y1) box lies within another."""
    return outer[0] <= box[0] and outer[1] <= box[1] and box[2] <= outer[2] and box[3] <= outer[3]


def apart(a, b, size):
    """Whether two (x0, y0, x1, y1) notes are as far apart as notes are kept, written out the plain way: 8 px
    across, or ceil(0.4 x the font size) px up and down (which is what notes standing one above the other,
    sharing columns, have between them)."""
    across = max(b[0] - a[2], a[0] - b[2])
    down = max(b[1] - a[3], a[1] - b[3])
    return across >= 8 or down >= math.ceil(0.4 * size)


def gap_to_box(point, box):
    """How far a point is from an (x0, y0, x1, y1) box: 0 on or inside it."""
    return math.hypot(
        max(box[0] - point[0], 0, point[0] - box[2]), max(box[1] - point[1], 0, point[1] - box[3])
    )


def pixels(mask):
    """The (x, y) of every set pixel of a mask."""
    return np.argwhere(mask)[:, ::-1]


def distance_to(points, point):
    """How far a point is from the closest of some (x, y) points."""
    return float(np.hypot(*(points - point).T).min())


def audit(alpha, boxes, texts=TEXTS, style=STYLE):
    """Check what holds for every card on the 1080 x 608 stage, and count what this one holds, the
    plain way: the notes asked for are there in order, their numbers are whole, they are on the
    stage and no wider than 42% of it, and each note's clash flag says what the note really is
    (cut, on the face, on another note or too close to one, or out of its quadrant). Returns the
    notes that were cut, the pixels on the face, the pixels shared by notes, and the sizes."""
    face = face_box(alpha) or DEFAULT_FACE
    quads = quadrants(W, H, face, MARGIN)
    assert list(boxes) == [key for key in KEYS if texts.get(key, "").strip()]
    cut, on_face, on_notes = [], 0, 0
    for key, box in boxes.items():
        where = f"{key} at ({box.x}, {box.y})"
        assert box.key == key, where
        assert all(type(v) is int for v in (box.size, box.x, box.y, box.w, box.h)), where
        assert style.min_size <= box.size <= style.max_size, where
        assert inside(rect(box), (0, 0, W, H)) and box.w <= MAX_W, where
        is_cut = " ".join(box.lines) != apply_case(texts[key], style.case)
        face_px = overlap(rect(box), face)
        note_px = sum(overlap(rect(box), rect(o)) for k, o in boxes.items() if k != key)
        crowded = any(
            not apart(rect(box), rect(o), max(box.size, o.size)) for k, o in boxes.items() if k != key
        )  # an overlap is too close, too
        assert box.clash == bool(
            is_cut or face_px or crowded or not inside(rect(box), quads[key])
        ), where
        cut += [key] * is_cut
        on_face += face_px
        on_notes += note_px
    sizes = [box.size for box in boxes.values()]
    return {"cut": cut, "face": on_face, "notes": on_notes // 2, "sizes": sizes}


def near_mask(alpha, radius):
    """dilate(alpha, radius) > 0.5 the way the layout finds the person's surroundings for its soft
    cost: on the matte shrunk to a quarter of its size (the mean of each 4 x 4 block), and
    enlarged again. It is checked against the full-size mask below."""
    height, width = alpha.shape
    rows, columns = -(-height // 4), -(-width // 4)
    padded = np.pad(alpha, ((0, rows * 4 - height), (0, columns * 4 - width)), mode="edge")
    small = padded.reshape(rows, 4, columns, 4).mean(axis=(1, 3))
    return (dilate(small, radius / 4) > 0.5).repeat(4, axis=0).repeat(4, axis=1)[:height, :width]


def spot_price(key, quad, avoid, ring, near, box):
    """What a note costs where it sits, counted the slow, plain way, as (clash, soft). The clash is
    the pixels it shares with a box to avoid (the face, and the notes placed before it): a spot
    with none beats any spot with some, so it is compared first. The soft cost is 3 per pixel on
    the sticker edge, 1 per pixel near the person (with arrows only), and 0.02 per pixel between
    the box's outer corner and the quadrant's own (top left for tl, top right for tr, and so on)."""
    x0, y0, x1, y1 = rect(box)
    clash = sum(overlap(rect(box), other) for other in avoid)
    corner = (x0 if key[1] == "l" else x1, y0 if key[0] == "t" else y1)
    target = (quad[0] if key[1] == "l" else quad[2], quad[1] if key[0] == "t" else quad[3])
    soft = 3 * int(ring[y0:y1, x0:x1].sum()) + 0.02 * math.dist(corner, target)
    return clash, soft + (0 if near is None else int(near[y0:y1, x0:x1].sum()))


def assert_cheapest_spots(alpha, boxes, style):
    """Check that each note of a card kept in its quadrants sits on the cheapest spot of the 8 px
    grid in its quadrant, for the way its text is set. Every spot is priced by spot_price, with
    plain slicing of the masks and plain overlaps, nothing like the summed-area tables and the
    numpy pass the layout uses."""
    height, width = alpha.shape
    face = face_box(alpha)
    quads = quadrants(width, height, face, MARGIN)
    ring = (dilate(alpha, style.band) > 0.5) & ~(alpha > 0.5)
    near = near_mask(alpha, style.band + 8 + 56) if style.arrows else None
    placed = []
    for key, box in boxes.items():
        quad, avoid = quads[key], [face, *placed]
        cheapest = min(
            spot_price(key, quad, avoid, ring, near, dataclasses.replace(box, x=x, y=y))
            for y in range(quad[1], quad[3] - box.h + 1, 8)
            for x in range(quad[0], quad[2] - box.w + 1, 8)
        )
        price = spot_price(key, quad, avoid, ring, near, box)
        assert price[0] == cheapest[0] == 0, key
        assert price[1] == pytest.approx(cheapest[1]), key
        assert (box.x - quad[0]) % 8 == 0 and (box.y - quad[1]) % 8 == 0, key
        placed.append(rect(box))


def wrap_options(text, style, size, max_w):
    """Every way to set a text whole at exactly one size, as (lines, w, h), the widest first (so the
    fewest lines first): fit_note at max_w, and then at a width one px under the width of what it
    gave, until the text would have to be cut (a narrower box only cuts more)."""
    fixed = dataclasses.replace(style, max_size=size, min_size=size)
    cased = apply_case(text, style.case)
    options, width = [], max_w
    while width >= 1:
        lines, _, w, h = fit_note(text, fixed, width)
        if " ".join(lines) != cased:
            break
        options.append((lines, w, h))
        if w > width:  # a word wider than the width: nothing narrower is any different
            break
        width = w - 1
    return options


def has_clean_spot(w, h, quad, face):
    """Whether a w x h box has a spot on the 8 px grid of a quadrant that shares no pixel with the
    face."""
    return any(
        overlap((x, y, x + w, y + h), face) == 0
        for y in range(quad[1], quad[3] - h + 1, 8)
        for x in range(quad[0], quad[2] - w + 1, 8)
    )


def clean_ways(alpha, texts, style, size):
    """For each note, the ways of setting its text whole at this size that have a clean spot in its
    own quadrant. The notes of different quadrants never touch, so a card in which every note has
    one at some size is a card that can be laid out clean at that size."""
    face = face_box(alpha)
    quads = quadrants(W, H, face, MARGIN)
    return {
        key: [
            option for option in wrap_options(text, style, size, MAX_W)
            if has_clean_spot(option[1], option[2], quads[key], face)
        ]
        for key, text in texts.items()
    }


def assert_largest_size(alpha, boxes, texts, style):
    """Check, by trying everything, that a card kept in its quadrants has the biggest size it can:
    each note is set the widest way that has a clean spot at its size, and at every bigger size one
    of the notes has no way of being set whole on a clean spot of its quadrant."""
    size = next(iter(boxes.values())).size
    for key, ways in clean_ways(alpha, texts, style, size).items():
        assert (boxes[key].lines, boxes[key].w, boxes[key].h) == ways[0], key
    for bigger in (s for s in SIZES if s > size):
        assert not all(clean_ways(alpha, texts, style, bigger).values()), bigger


def least_overlap(text, face, style):
    """The fewest pixels a note can share with the face on its own: at any size, set any way, on
    any spot of the stage's 8 px grid. The least over the rows of spots times the least over the
    columns is the least over the spots, as an overlap is the product of the two."""
    best = math.inf
    for size in SIZES:
        for _, w, h in wrap_options(text, style, size, MAX_W):
            xs, ys = np.arange(0, W - w + 1, 8), np.arange(0, H - h + 1, 8)
            across = np.clip(np.minimum(xs + w, face[2]) - np.maximum(xs, face[0]), 0, None)
            down = np.clip(np.minimum(ys + h, face[3]) - np.maximum(ys, face[1]), 0, None)
            best = min(best, int(across.min()) * int(down.min()))
    return best


FACING = {"tl": (1, 1), "tr": (-1, 1), "bl": (1, -1), "br": (-1, -1)}  # which way the person is


def start_candidates(key, box):
    """Where an arrow may start: the corner of the note that faces the person, 10 px out along
    each axis; then points every 16 px (the ends too) along the two sides that face the person
    (tl: right and bottom, tr: left and bottom, bl: right and top, br: left and top), 8 px
    out."""
    x0, y0, x1, y1 = rect(box)
    sx, sy = FACING[key]
    side_x, side_y = (x1 if sx > 0 else x0), (y1 if sy > 0 else y0)

    def along(low, high):
        return sorted({*range(low, high + 1, 16), high})

    return (
        [(side_x + 10 * sx, side_y + 10 * sy)]
        + [(side_x + 8 * sx, y) for y in along(y0, y1)]
        + [(x, side_y + 8 * sy) for x in along(x0, x1)]
    )


def walk(start, aim, grown):
    """Where an arrow from start ends when it heads for aim, by plain steps: 2 px at a time, the
    last point before the grown person. None when it reaches aim without meeting the person."""
    dx, dy = aim[0] - start[0], aim[1] - start[1]
    length = math.hypot(dx, dy)
    last, step = start, 2.0
    while length and step <= length + 1e-9:
        point = (
            int(np.rint(start[0] + dx / length * step)),
            int(np.rint(start[1] + dy / length * step)),
        )
        if grown[point[1], point[0]]:
            return last
        last, step = point, step + 2
    return None


def bow(start, end, box, stage_centre):
    """The control point of an arrow: the middle of the chord pushed 18% of its length sideways,
    to the side that keeps the curve (16 points) farther from its own note, and on a tie to the
    side away from the center of the stage."""
    middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    best = None
    for side in (1, -1):
        normal = (-dy / length * side, dx / length * side)
        control = (middle[0] + 0.18 * length * normal[0], middle[1] + 0.18 * length * normal[1])
        clearance = min(gap_to_box(p, rect(box)) for p in quad_points(start, control, end, 16))
        away = normal[0] * (middle[0] - stage_centre[0]) + normal[1] * (middle[1] - stage_centre[1])
        if best is None or (round(clearance, 6), away) > best[0]:
            best = ((round(clearance, 6), away), control)
    return round(best[1][0]), round(best[1][1])


def curve_ok(alpha, note_rects, start, control, end):
    """Whether a curve, at 16 points, never enters the person (alpha over 0.5) and never touches a
    note box, its own included, except within 6 px of its start."""
    height, width = alpha.shape
    for x, y in quad_points(start, control, end, 16):
        if alpha[min(max(round(y), 0), height - 1), min(max(round(x), 0), width - 1)] > 0.5:
            return False
        if math.dist((x, y), start) > 6 and any(gap_to_box((x, y), r) == 0 for r in note_rects):
            return False
    return True


def assert_arrows(alpha, boxes, style):
    """Check every arrow. Each is None, or:
    - starts at one of the spots start_candidates gives, on bare background outside the person
      grown by band + 8;
    - ends on bare background (alpha 0) just outside that grown edge, and no farther than band +
      20 from the silhouette;
    - has a chord of 28 px to 180 k px, where k is the stage's height over 608, but at least 1
      and at most 2 (so a full-screen stage gets arrows twice as long);
    - bows its middle sideways by 18% of its length, to the side that keeps it farther from its own
      note (on a tie, away from the center of the stage);
    - never enters the person, and never touches a note box except within 6 px of its start.
    And it is the valid arrow whose chord is nearest 90 k px (ties: the end nearer the face box's
    center), found again here the slow way, or None only when no arrow is valid."""
    height, width = alpha.shape
    scale = min(max(height / 608, 1), 2)
    shortest, longest, best_chord = 28, 180 * scale, 90 * scale
    grown = dilate(alpha, style.band + 8) > 0.5
    grown_points = pixels(grown)
    person = pixels(alpha > 0.5)
    stage_centre = (width / 2, height / 2)
    face, bbox = face_box(alpha), person_boxes(alpha)["bbox"]
    face_centre = ((face[0] + face[2]) // 2, (face[1] + face[3]) // 2)
    aims = [((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2), face_centre]
    note_rects = [rect(box) for box in boxes.values()]

    def targets(start):
        nearest = grown_points[int(np.argmin(np.hypot(*(grown_points - start).T)))]
        return [*aims, (int(nearest[0]), int(nearest[1]))]

    def best_valid(key):
        best = None
        for start in start_candidates(key, boxes[key]):
            if not (0 <= start[0] < width and 0 <= start[1] < height) or grown[start[1], start[0]]:
                continue
            for aim in targets(start):
                end = walk(start, aim, grown)
                if end is None or not shortest <= math.dist(start, end) <= longest:
                    continue
                control = bow(start, end, boxes[key], stage_centre)
                if curve_ok(alpha, note_rects, start, control, end):
                    score = (
                        round(abs(math.dist(start, end) - best_chord), 6),
                        math.dist(end, face_centre),
                    )
                    best = score if best is None or score < best else best
        return best

    for key, box in boxes.items():
        best = best_valid(key)
        if box.arrow is None:
            assert best is None, key  # no arrow was valid, and the slow search finds none either
            continue
        assert best is not None, key
        start, control, end = box.arrow
        assert all(type(v) is int for point in box.arrow for v in point), key
        assert start in start_candidates(key, box), key
        assert alpha[start[1], start[0]] < 0.5 and not grown[start[1], start[0]], key
        assert alpha[end[1], end[0]] == 0 and not grown[end[1], end[0]], key
        assert style.band <= distance_to(person, end) <= style.band + 20, key

        chord = (end[0] - start[0], end[1] - start[1])
        length = math.hypot(*chord)
        assert shortest <= length <= longest, key
        # The walk stopped there because the next step was inside the grown person, and it was
        # heading for one of the three targets: the end sits on the line from the start to it
        ahead = [
            (round(end[0] + s * chord[0] / length), round(end[1] + s * chord[1] / length))
            for s in (2, 3, 4)
        ]
        assert any(grown[y, x] for x, y in ahead), key
        assert any(
            abs(chord[0] * (aim[1] - start[1]) - chord[1] * (aim[0] - start[0]))
            / math.hypot(aim[0] - start[0], aim[1] - start[1]) <= 1.5
            for aim in targets(start)
        ), key

        middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        push = (control[0] - middle[0], control[1] - middle[1])
        assert math.hypot(*push) == pytest.approx(0.18 * length, abs=1.5), key
        assert abs(push[0] * chord[0] + push[1] * chord[1]) <= 1.5 * length, key  # at right angles

        def clearance(bent):
            return min(gap_to_box(p, rect(box)) for p in quad_points(start, bent, end, 16))

        other_side = (middle[0] - push[0], middle[1] - push[1])
        chosen, other = clearance(control), clearance(other_side)
        assert chosen > 0 and chosen >= other - 1, key
        if abs(chosen - other) < 1e-6:  # a tie, as when both curves leave the note the same way
            outward = (middle[0] - stage_centre[0], middle[1] - stage_centre[1])
            assert push[0] * outward[0] + push[1] * outward[1] > 0, key
        assert curve_ok(alpha, note_rects, start, control, end), key
        assert (round(abs(length - best_chord), 6), math.dist(end, face_centre)) == pytest.approx(
            best
        ), key


# --- quadrants and the face -------------------------------------------------------------------


def test_quadrants_split_on_head_centre(person_alpha):
    head = (389, 49, 691, 261)  # centered on x = 540, the middle of the stage
    quads = quadrants(1080, 608, head, 43)
    assert quads == {
        "tl": (43, 43, 497, 304),
        "tr": (583, 43, 1037, 304),
        "bl": (43, 304, 497, 565),
        "br": (583, 304, 1037, 565),
    }
    # The left quadrants end before the head's center, the right ones start after it
    assert quads["tl"][2] < 540 < quads["tr"][0]
    assert quads["bl"][2] < 540 < quads["br"][0]

    # A head to the left takes the split with it, 43 px either side of its center (x = 250)
    left = quadrants(1080, 608, (100, 49, 400, 261), 43)
    assert left["tl"][2] == left["bl"][2] == 207
    assert left["tr"][0] == left["br"][0] == 293
    assert left["tl"][0] == left["bl"][0] == 43 and left["tr"][2] == left["br"][2] == 1037

    # With nobody to split on, the stage's own middle is the split. A head that is None is what
    # person_boxes gives for an empty matte.
    assert quadrants(1080, 608, None, 43) == quads
    assert all(type(v) is int for box in quadrants(1081, 609, None, 43).values() for v in box)

    # On a tall stage the rows keep the quadrants between two rows of it (plan_notes passes the
    # reach of an arrow above and below the person), cut in the middle of those rows
    tall = quadrants(1080, 1920, (332, 706, 739, 1225), 43, rows=(346, 1574))
    assert tall == {
        "tl": (43, 346, 492, 960),
        "tr": (578, 346, 1037, 960),
        "bl": (43, 960, 492, 1574),
        "br": (578, 960, 1037, 1574),
    }

    # The face box runs from the top of the person to below the neck. The synthetic person's head
    # ellipse ends at 0.34 H + 0.26 H and its body starts at 0.62 H, so the box ends between them.
    # Its columns are the head's, 0.40 W to 0.60 W, with 4% of W of air on either side.
    synthetic = person_alpha(W, H)
    face = face_box(synthetic)
    assert face[1] == person_boxes(synthetic)["bbox"][1]
    assert 0.60 * H <= face[3] <= 0.62 * H
    assert abs(face[0] - (0.40 * W - 43)) <= 2 and abs(face[2] - (0.60 * W + 43)) <= 2
    assert all(type(v) is int for v in face)

    # On the real close-up the narrowest row is y = 370, mid-face, where the raised hood starts to
    # widen the silhouette (its right edge jumps from x = 667 to 742 by y = 390). That is a cheek,
    # not a neck: the beard ends at y = 506. So the box ends below the chin, and its columns are
    # the beanie's, x = 375 to 694 with 43 px of air, not stretched to the hood.
    close = close_alpha()
    face = face_box(close)
    assert face == (332, 50, 739, 529)
    assert face[3] >= CHIN + 10

    # The head is where the face box is read from, so things wider than the head, which come
    # into the top of the picture when the person is framed higher or has their arms up, do not
    # stretch it. Raised 160 and 200 px, the close-up's head is cut by the top of the stage, and
    # the box is still no taller than 1.5 times the head's widest row (the beanie's).
    row_widths = (close > 0.5).sum(axis=1)
    head_w = int(row_widths[50:370].max())  # the beanie and face, down to where the hood widens
    assert 310 <= head_w <= 330
    for up in (160, 200):
        higher = face_box(moved(close, up=up))
        assert higher[1] == 0 and higher[3] - higher[1] <= 1.5 * head_w, up
        assert higher[0] == face[0] and higher[2] == face[2], up  # still the head's columns
    # Both arms up make the person's top as wide as 1.7 heads, and that is still not a head
    raised = face_box(raised_arms(synthetic))
    assert raised[3] - raised[1] <= 1.5 * 0.20 * W  # the synthetic head is 0.20 W wide
    assert raised[0] > 300 and raised[2] < 780  # the arms are 250 to 330 and 750 to 830

    # A head cut by the side of the stage looks narrower than it is, and the box would end above
    # the beard (at 497, with 3% of the stage's height of air). Past a side edge it gets 5%.
    for right in (-400, 400):
        clipped = face_box(moved(close, right=right))
        assert (clipped[0] == 0 or clipped[2] == W) and clipped[3] >= CHIN, right

    # Nobody in the matte: no face box (plan_notes then uses a default one)
    assert face_box(np.zeros((H, W), np.float32)) is None


def test_face_box_ends_the_head_where_a_hood_starts(person_alpha, monkeypatch):
    # A hood is not part of the head. The width of a hooded person grows a little with every row, so no
    # row is ever much wider than the widest so far, and the head hill followed it down over the
    # shoulders: the box ended 1.45 of those widths under the top. A row wider than 1.15 times the median
    # width just under the cap ends the head now. The old rule is the same code with that off.
    def old_rule(alpha):
        with monkeypatch.context() as patch:
            patch.setattr(layout, "_HOOD_GROWTH", math.inf)
            return face_box(alpha)

    # A vertical photo: the beanie and the face are 373 to 402 px wide from row 697 to 857, the
    # hood makes it 440 at row 897, 495 at 937, 724 at 1017 and 756 by row 1097, and the chin is at about
    # row 990. The box was (176, 497, 1030, 1651), his whole upper body, 854 px wide. It ends below the
    # chin now, and is the head's width with its air.
    vertical = vertical_alpha()
    top = person_boxes(vertical)["bbox"][1]
    assert old_rule(vertical) == (176, 497, 1030, 1651)
    face = face_box(vertical)
    assert face[1] == top == 497
    assert 1050 <= face[3] <= 1300  # near the neck (it was 1651), under the chin with a margin of air
    assert 450 <= face[2] - face[0] <= 650 and face[0] > 200 and face[2] < 900  # not the hood's 854
    assert face == (274, 497, 831, 1177)

    # The plate's close-up is not changed by it: its hood only starts to show at row 370, and the head
    # was already ended there
    close = close_alpha()
    assert face_box(close) == old_rule(close) == (332, 50, 739, 529)

    # A head with a hood written out, on trapezoids that widen to 2, 2.4 and 3 times the head's width:
    # the head is 380 wide and its bottom is at row 1000. The box ends under it, not on the shoulders,
    # and spans the head's columns with a little of the hood's first rows (the 1.15 times that ends
    # the head), where the old box went down to 1.45 times a head that had grown to the shoulders'.
    for bottom_w in (2.0, 2.4, 3.0):
        silhouette = hooded(bottom_w=bottom_w)
        x0, y0, x1, y1 = face_box(silhouette)
        assert y0 == 500, bottom_w
        assert 1000 <= y1 <= 1000 + 0.75 * 380, bottom_w  # the head's bottom, and not past the neck
        assert 380 <= x1 - x0 <= 1.5 * 380, bottom_w
        ox0, oy0, ox1, oy1 = old_rule(silhouette)
        assert oy1 > y1 + 50 and (ox1 - ox0) > 1.5 * 380, bottom_w  # the control: it was the shoulders

    # A plain head on sharp shoulders is read as it was: the same box with the rule on and off, and these
    # are the boxes of today (the synthetic person's, at the stage's size and on a full-screen stage)
    for alpha, box in (
        (plain_head(), (377, 50, 703, 416)),
        (person_alpha(W, H), (389, 49, 691, 369)),
        (person_alpha(1080, 1920), (389, 154, 691, 1168)),
    ):
        assert face_box(alpha) == old_rule(alpha) == box


def test_face_box_is_the_head_when_a_hood_stands_out_on_one_side(person_alpha, monkeypatch):
    # When the person is turned three quarters the hood stands out on ONE side of the head, at the height
    # of the chin, and the rows at face height, where the head's width is read from, already hold it: the
    # median width of the rows under the cap was the hood's, and the head went on down over the shoulders.
    # Now an edge that stands out ends the head. The old rule is the same code that does not look for one.
    def old_rule(alpha):
        with monkeypatch.context() as patch:
            patch.setattr(layout, "_LEDGE", math.inf)
            return face_box(alpha)

    # The three-quarter photo: the head is 234 to 572, and the box was (191, 286, 867, 1015), 676 px
    # wide for a 338 px head and down to row 1015 for a chin at about 745 (its widest row came out as
    # row 897, 463 px wide, inside the hood). Now it is the head's columns with 43 px of air, under the chin.
    photo = threequarter_alpha()
    face = face_box(photo)
    assert face == (191, 286, 616, 829)
    assert 180 <= face[0] <= 200 and 600 <= face[2] <= 630  # about the head, not up to the hood's 866
    assert old_rule(photo) == (191, 286, 867, 1015)

    # The same written out: a head 336 wide and a collar that stands out 120 px to the right of it, in
    # one row or in eight (a hood is not always square), with shoulders under it. The box is the head's
    # width and its air, centered on the head whichever side the hood is on; it was the hood's and more.
    for ramp in (1, 8):
        shape = three_quarter(ramp)
        assert face_box(shape) == (192, 286, 614, 831), ramp
        assert face_box(shape[:, ::-1]) == (W - 614, 286, W - 192, 831), ramp
        ox0, oy0, ox1, oy1 = old_rule(shape)
        assert (ox1 - ox0) > 1.5 * 336 and oy1 > 831 + 150, ramp

    # What stands out beside a head with the neck showing is the head's own, and is not touched: these are
    # the boxes before this rule (F4's probes). A wide brim and a hand at the cheek stand out at the
    # height of the face, raised arms above the shoulders, a profile's nose and hair at the sides.
    base = person_alpha(W, H)
    brim = with_parts(base, blob(540, 150, 300, 22), blob(540, 130, 125, 70))
    hand = with_parts(base, blob(686, 250, 42, 55), block(slice(290, 420), slice(658, 728)))
    profile = with_parts(base, blob(662, 215, 30, 18), blob(426, 190, 30, 70))
    for name, alpha, box in (
        ("wide brim", brim, (197, 49, 883, 608)),
        ("hand at the cheek", hand, (389, 49, 771, 409)),
        ("profile", profile, (353, 49, 735, 467)),
        ("raised arms", raised_arms(base), (396, 49, 684, 360)),
    ):
        assert face_box(alpha) == old_rule(alpha) == box, name

    # Below the head's widest row a row counts only as far as it stays within the head's width and its air
    # around the head's center (432 to 647, and 43 px each side): a hand that starts under the temples and
    # sticks out 70 px is cut there, where the box used to run out to the hand, to x 763
    assert face_box(with_parts(base, block(slice(250, 330), slice(650, 720)))) == (389, 49, 734, 369)

    # It is the neck that tells a hand from a hood: with a collar over the gap beside it, the same hand
    # stands out of a head with no neck, and is not the head's
    collared = with_parts(hand, block(slice(290, 380), slice(330, 750)))
    assert old_rule(collared) == (389, 49, 739, 409)
    assert face_box(collared) == (389, 49, 691, 380)


# --- fitting a note's text --------------------------------------------------------------------


def test_apply_case():
    text = "But overthinks every 6-second reel"
    assert apply_case(text, "upper") == "BUT OVERTHINKS EVERY 6-SECOND REEL"
    assert apply_case(text, "asis") == text
    with pytest.raises(ValueError, match="case"):
        apply_case(text, "title")


def test_fit_note_shrinks_then_wraps():
    max_w = 453

    # A short note keeps the biggest size on one line, and its box is just the text
    lines, size, w, h = fit_note("@your_handle", STYLE, max_w)
    assert (lines, size) == (["@YOUR_HANDLE"], 56)
    assert (w, h) == (font("marker", 56).getlength("@YOUR_HANDLE"), 56)
    assert type(w) is int and type(h) is int

    # A long one wraps and then shrinks until three lines are enough, never wider than max_w
    text = "But overthinks every single 6-second reel before it ever goes out"
    lines, size, w, h = fit_note(text, STYLE, max_w)
    fnt = font("marker", size)
    assert STYLE.min_size < size < STYLE.max_size and len(lines) <= 3
    assert " ".join(lines) == text.upper()  # all of it is there, in capitals
    assert all(fnt.getlength(line) <= max_w for line in lines)
    assert w == max(fnt.getlength(line) for line in lines) and w <= max_w
    assert h == len(lines) * size
    # It is the biggest size that works: one step up would not have fit in three lines
    one_up = NoteStyle("marker", size + 2, size + 2, "upper")
    assert fit_note(text, one_up, max_w)[0][-1].endswith(ELLIPSIS)

    # A word wider than the box is never split, so the text shrinks until the word fits. One still
    # too wide at the smallest size stays whole there, and the box is as wide as it.
    lines, size, w, h = fit_note("@your_handle_the_creator", STYLE, max_w)
    assert STYLE.min_size < size < STYLE.max_size and len(lines) == 1 and w <= max_w
    lines, size, w, h = fit_note("X" * 60, STYLE, max_w)
    assert (lines, size) == (["X" * 60], STYLE.min_size) and w > max_w

    # The padding goes on all four sides, and line_h sets how far apart the lines sit
    comic = NoteStyle("comic", 40, 28, "upper", line_h=1.05, pad=14)
    lines, size, w, h = fit_note("Best friend: Clover (sorry, mom)", comic, 400)
    fnt = font("comic", size)
    assert all(fnt.getlength(line) <= 400 - 2 * 14 for line in lines)
    assert w == max(fnt.getlength(line) for line in lines) + 2 * 14
    assert h == len(lines) * round(size * 1.05) + 2 * 14

    # Lines are measured in pixels, not characters: the same number of narrow letters stays on
    # one line, and wide ones need two
    flat = NoteStyle("marker", 40, 40, "asis")
    narrow, wide = "iiii iiii", "WWWW WWWW"
    assert len(narrow) == len(wide)
    assert fit_note(narrow, flat, 150)[0] == [narrow]
    assert fit_note(wide, flat, 150)[0] == ["WWWW", "WWWW"]  # and "asis" keeps the case it got

    # Press Start 2P is only crisp on an 8 px grid, so the pixel role goes 24 then 16
    pixel = NoteStyle("pixel", 24, 16, "upper", line_h=1.5)
    long_text = "But overthinks every single 6-second reel before posting it"
    assert fit_note("Creator", pixel, 400)[1] == 24
    assert fit_note(long_text, pixel, 400)[1] == 16
    assert {fit_note("word " * count, pixel, 400)[1] for count in range(1, 30)} == {24, 16}

    # VT323, the terminal role, goes in steps of 4 (52, 48 ... 32) and not on the grid of 8; every
    # other font keeps its steps of 2
    terminal = NoteStyle("terminal", 52, 32, "upper", line_h=1.0)
    steps = {52, 48, 44, 40, 36, 32}
    assert layout._sizes(terminal) == [52, 48, 44, 40, 36, 32] and layout._sizes(pixel) == [24, 16]
    assert layout._sizes(NoteStyle("terminal", 50, 33, "upper")) == [50, 46, 42, 38, 34, 33]
    assert layout._sizes(NoteStyle("terminal", 32, 32, "upper")) == [32]
    assert layout._sizes(STYLE) == SIZES
    assert layout._sizes(NoteStyle("hand", 52, 36, "asis"))[:3] == [52, 50, 48]
    assert fit_note("Creator", terminal, 400)[1] == 52
    assert fit_note(long_text, terminal, 400)[1] in steps - {52}
    found = set()
    for count in range(1, 30):  # always a step, and the biggest one that takes three lines
        size = fit_note("word " * count, terminal, 400)[1]
        found.add(size)
        if size < 52:
            one_up = NoteStyle("terminal", size + 4, size + 4, "upper")
            assert fit_note("word " * count, one_up, 400)[0][-1].endswith(ELLIPSIS), count
    assert found <= steps and {52, 32} <= found


def test_fit_note_ellipsizes():
    max_w = 453
    text = ("overthinking every reel " * 9)[:200]
    assert len(text) == 200
    lines, size, w, h = fit_note(text, STYLE, max_w)
    fnt = font("marker", size)

    # It could not fit at any size, so it is the smallest one, three lines, and the last one ends
    # in an ellipsis. Everything still fits the width, the ellipsis included.
    assert size == STYLE.min_size and len(lines) == 3
    assert lines[-1].endswith(ELLIPSIS)
    assert not any(ELLIPSIS in line for line in lines[:-1])
    assert all(fnt.getlength(line) <= max_w for line in lines)
    assert w <= max_w and h == 3 * size

    # The cut falls between whole words, as late as it can: one word more would not fit
    shown = lines[-1][:-1].split()
    rest = text.upper().split()[sum(len(line.split()) for line in lines[:2]) :]
    assert shown == rest[: len(shown)]
    assert fnt.getlength(" ".join(rest[: len(shown) + 1]) + ELLIPSIS) > max_w

    # A word wider than the whole line has no word break to cut at, so it is cut by characters
    small = NoteStyle("marker", 34, 34, "asis")
    lines, size, w, h = fit_note("Alpha Beta " + "X" * 40 + " Omega", small, 120)
    fnt = font("marker", 34)
    room = max(count for count in range(1, 40) if fnt.getlength("X" * count + ELLIPSIS) <= 120)
    assert lines == ["Alpha", "Beta", "X" * room + ELLIPSIS]
    assert room > 1 and fnt.getlength("X" * (room + 1) + ELLIPSIS) > 120

    # The pixel role gets there at 16, the smallest size on its grid, and the terminal role at 32
    pixel = NoteStyle("pixel", 24, 16, "upper", line_h=1.5)
    lines, size, w, h = fit_note(text, pixel, 400)
    assert size == 16 and len(lines) == 3 and lines[-1].endswith(ELLIPSIS)
    lines, size, w, h = fit_note(text, NoteStyle("terminal", 52, 32, "upper"), 400)
    assert size == 32 and len(lines) == 3 and lines[-1].endswith(ELLIPSIS)

    # wrap does the same on a budget of characters: greedy, no word split, three lines by default
    assert wrap("one two three four five", 9) == ["one two", "three", "four five"]
    assert wrap("extraordinarily short", 8) == ["extraordinarily", "short"]
    assert wrap("aaa bbb ccc ddd eee fff", 7, 2) == ["aaa bbb", "ccc" + ELLIPSIS]
    assert wrap("supercalifragilistic expialidocious", 8, 1) == ["superca" + ELLIPSIS]
    assert wrap("one two three four five six seven", 9)[-1].endswith(ELLIPSIS)
    assert wrap("", 10) == []
    with pytest.raises(ValueError, match="max_lines"):
        wrap("one two", 10, 0)


# --- placing the notes ------------------------------------------------------------------------


def test_plan_notes_avoid_head_and_stay_inside(person_alpha):
    synthetic = person_alpha(W, H)
    close = close_alpha()

    # The flag is off unless the layout turns it on, so a NoteBox built without it still works
    assert layout.NoteBox("tl", ["HI"], 56, 0, 0, 10, 10, None).clash is False

    sizes = {}
    for name, alpha in (("synthetic", synthetic), ("close-up", close)):
        face = face_box(alpha)
        quads = quadrants(W, H, face, MARGIN)
        boxes = plan_notes(alpha, TEXTS, STYLE)
        card = audit(alpha, boxes)

        for key, box in boxes.items():
            where = f"{name} {key}"
            assert " ".join(box.lines) == TEXTS[key].upper(), where  # all the words, in capitals
            assert overlap(rect(box), face) == 0, where  # off the face
            assert inside(rect(box), quads[key]), where  # and in its own quadrant
            assert not box.clash, where
        assert (card["cut"], card["face"], card["notes"]) == ([], 0, 0), name  # and off each other
        assert len(set(card["sizes"])) == 1, name  # one size for the whole card
        assert_cheapest_spots(alpha, boxes, STYLE)
        assert_largest_size(alpha, boxes, TEXTS, STYLE)
        sizes[name] = card["sizes"][0]

    # Nothing is in the synthetic person's way, so the notes keep the biggest size. Beside the
    # real close-up the strips left of the face box and right of it are about 290 px wide, and the
    # whole card shrinks, a size at a time, to the biggest at which every note fits its corner
    # (assert_largest_size tried every bigger size, and every way of setting each note).
    assert sizes == {"synthetic": 56, "close-up": 44}

    # The 42% of the stage (454 px) is wide enough for "BEST FRIEND: CLOVER" at 56 px, and too
    # narrow for "EVERY 6-SECOND REEL", so the notes break there
    boxes = plan_notes(synthetic, TEXTS, STYLE)
    marker = font("marker", 56)
    assert boxes["br"].lines[0] == "BEST FRIEND: CLOVER"
    assert boxes["br"].w == marker.getlength("BEST FRIEND: CLOVER")
    assert marker.getlength("EVERY 6-SECOND REEL") > 454
    assert boxes["bl"].lines[1] == "EVERY 6-SECOND"

    # The sticker edge is what moves a short note at the bottom left: without one it sits in its
    # corner, and with one it rides up out of the way, onto bare background above the shoulder
    edge = (dilate(synthetic, STYLE.band) > 0.5) & ~(synthetic > 0.5)

    def edge_under(box):
        return int(edge[box.y : box.y + box.h, box.x : box.x + box.w].sum())

    short = {"bl": "Hi there"}
    bare = plan_notes(synthetic, short, dataclasses.replace(QUIET, band=0))["bl"]
    avoiding = plan_notes(synthetic, short, QUIET)["bl"]
    corner = quadrants(W, H, face_box(synthetic), MARGIN)["bl"]
    assert bare.x == corner[0] and corner[3] - (bare.y + bare.h) < 8  # the outer corner
    assert edge_under(bare) > 0 and edge_under(avoiding) == 0
    assert avoiding.y < bare.y

    # With arrows the style also keeps notes away from the body, to leave room for the arrow: a
    # short note at the bottom right sits on the shoulder with no arrows, and clear of the person
    # (and of most of the near mask, the person grown by band + 8 + 56) with them. The near mask
    # is found at a quarter of the size, and is within 1% of the full-size one.
    full_near = dilate(synthetic, STYLE.band + 8 + 56) > 0.5
    near = near_mask(synthetic, STYLE.band + 8 + 56)
    assert (full_near != near).sum() < 0.01 * full_near.sum()
    person = synthetic > 0.5

    def under(mask, box):
        return int(mask[box.y : box.y + box.h, box.x : box.x + box.w].sum())

    short = {"br": "Hi there"}
    on_shoulder = plan_notes(synthetic, short, QUIET)["br"]
    off_body = plan_notes(synthetic, short, STYLE)["br"]
    assert under(person, on_shoulder) > 0 and under(person, off_body) == 0
    assert under(near, off_body) < under(near, on_shoulder)

    # A person well off to the right leaves the right-hand notes no room in their corners at any
    # size, so they leave them, for the room the left has. They still keep the whole text, the
    # one size, and off the face and each other, and each one that left is flagged.
    right = moved(synthetic, right=250)
    shoved = plan_notes(right, TEXTS, STYLE)
    card = audit(right, shoved)
    right_quads = quadrants(W, H, face_box(right), MARGIN)
    assert (card["cut"], card["face"], card["notes"]) == ([], 0, 0)
    assert len(set(card["sizes"])) == 1
    assert not all(inside(rect(box), right_quads[key]) for key, box in shoved.items())
    assert any(box.clash for box in shoved.values())

    # The close-up framed off-center. No word is dropped while a whole text has a clean spot
    # anywhere (the close-up 120 px to the left used to give "BUT OVERTHINKS EVERY..." for bl, with
    # the whole text clean elsewhere), notes keep off the face and off each other, and the card
    # keeps one size. When the notes cannot all stay in their corners at any size they leave them,
    # and are flagged. Placing them in another order is what gives 52 and 54 px at +120 and +150
    # (the default order gives 50 and 52). The space notes keep between them takes -120 from 56 px,
    # four notes edge to edge in one margin, to 52, and +120 from 52 to 50.
    floor = {-120: 52, 120: 50, 150: 54}
    for dx in (-120, 120, 150):
        alpha = moved(close, right=dx)
        boxes = plan_notes(alpha, TEXTS, STYLE)
        card = audit(alpha, boxes)
        assert card["cut"] == [] and card["face"] == 0 and card["notes"] == 0, dx
        assert all(" ".join(box.lines) == TEXTS[key].upper() for key, box in boxes.items()), dx
        assert len(set(card["sizes"])) == 1 and card["sizes"][0] >= floor[dx], dx

    # Two people side by side: the face box covers both heads, and the notes still keep off it
    # and off each other, with the whole text at one size
    both = two_people(person_alpha)
    boxes = plan_notes(both, TEXTS, STYLE)
    card = audit(both, boxes)
    assert (card["cut"], card["face"], card["notes"]) == ([], 0, 0)
    assert len(set(card["sizes"])) == 1

    # A note is cut only when its text cannot be set in three lines at the smallest size, and
    # then its flag says so. Nothing else is touched: the other notes keep the whole text, and
    # off the face and each other, in their corners. A text that ends in an ellipsis of its own
    # is not a cut one, so the flag is not decided by the last character.
    long_note = dict(TEXTS, tl=("overthinking every reel " * 9)[:200])
    boxes = plan_notes(close, long_note, STYLE)
    card = audit(close, boxes, long_note)
    assert card["cut"] == ["tl"] and boxes["tl"].lines[-1].endswith(ELLIPSIS)
    assert [key for key, box in boxes.items() if box.clash] == ["tl"]
    assert [key for key, box in boxes.items() if box.cut] == ["tl"]  # the flag a hero rule keeps
    assert card["notes"] == 0
    dots = dict(TEXTS, tr="Wait for it" + ELLIPSIS)
    boxes = plan_notes(close, dots, STYLE)
    card = audit(close, boxes, dots)
    assert boxes["tr"].lines == ["WAIT FOR IT" + ELLIPSIS] and card["cut"] == []
    assert not any(box.clash for box in boxes.values()) and not any(box.cut for box in boxes.values())

    # A single word wider than the whole stage fits nowhere, not in a corner and not inside the
    # margins. It still gets a box, with the word whole, at the left edge of the stage, and its
    # flag set, and the other notes stay clean and whole.
    huge = dict(TEXTS, tr="@" + "verylonghandle" * 6)
    boxes = plan_notes(synthetic, huge, STYLE)
    assert boxes["tr"].lines == [huge["tr"].upper()] and boxes["tr"].w > W and boxes["tr"].x == 0
    assert [key for key, box in boxes.items() if box.clash] == ["tr"]
    assert all(" ".join(boxes[key].lines) == TEXTS[key].upper() for key in ("tl", "bl", "br"))

    # A person who fills the stage leaves the notes no clean spot at any size. Then the card is the
    # one with the least overlap (the notes may differ in size here, and none is cut): each note
    # shares with the face no more than it must on its own, at any size, any way of setting it
    # and any spot, and no note is on another. All four go in a row along the strip above the
    # face, the smallest size, and every one is flagged.
    wall = np.zeros((H, W), np.float32)
    wall[60:, 150:930] = 1.0
    wall_face = face_box(wall)
    crowded = plan_notes(wall, TEXTS, STYLE)
    card = audit(wall, crowded)
    assert (card["cut"], card["notes"]) == ([], 0)
    assert card["face"] == sum(least_overlap(TEXTS[key], wall_face, STYLE) for key in KEYS) > 0
    assert card["sizes"] == [STYLE.min_size] * 4
    assert all(box.clash for box in crowded.values())

    # With nobody in the matte, the face is a box in the middle of the stage and the notes still
    # lay out around it; the box is (0.36 W, 0.08 H, 0.64 W, 0.55 H)
    nobody = plan_notes(np.zeros((H, W), np.float32), TEXTS, STYLE)
    card = audit(np.zeros((H, W), np.float32), nobody)
    assert (card["cut"], card["face"], card["notes"]) == ([], 0, 0)
    assert not any(box.clash for box in nobody.values())


def test_plan_notes_respect_positions(person_alpha):
    alpha = person_alpha(W, H)
    free = plan_notes(alpha, TEXTS, STYLE)

    moved_tl = plan_notes(alpha, TEXTS, STYLE, {"tl": [100, 90]})
    assert (moved_tl["tl"].x, moved_tl["tl"].y) == (100, 90)
    # It keeps the way its text is set, and the size of the card
    assert (moved_tl["tl"].lines, moved_tl["tl"].size) == (free["tl"].lines, free["tl"].size)
    assert all(moved_tl[key] == free[key] for key in ("tr", "bl", "br"))
    assert not any(box.clash for box in moved_tl.values())
    # Pinned right beside the head, its arrow is drawn again from the new place
    assert moved_tl["tl"].arrow is not None and moved_tl["tl"].arrow != free["tl"].arrow
    assert_arrows(alpha, moved_tl, STYLE)

    # What the project file holds: null is no override, and "burst" is not a note. A float is
    # rounded, and a place off the stage is pulled back onto it.
    saved = plan_notes(
        alpha, TEXTS, STYLE,
        {"tl": None, "tr": [5000, -30], "bl": [100.4, 300.6], "br": None, "burst": [10, 10]},
    )
    assert list(saved) == KEYS
    assert saved["tl"] == free["tl"] and saved["br"] == free["br"]
    assert (saved["tr"].x, saved["tr"].y) == (W - saved["tr"].w, 0)
    assert (saved["bl"].x, saved["bl"].y) == (100, 301)
    assert all(type(v) is int for v in (saved["bl"].x, saved["bl"].y, saved["tr"].x, saved["tr"].y))

    # A pinned note is fixed before the others are placed, so they keep off it: pinned in the
    # top left corner, tr pushes tl out of the corner, whatever the order the notes go in. The
    # card keeps one size, and the pinned note is set at it.
    corner = plan_notes(alpha, TEXTS, STYLE, {"tr": [43, 43]})
    assert (corner["tr"].x, corner["tr"].y) == (43, 43)
    assert (corner["tl"].x, corner["tl"].y) != (free["tl"].x, free["tl"].y)
    assert overlap(rect(corner["tl"]), rect(corner["tr"])) == 0
    assert len({box.size for box in corner.values()}) == 1
    # and no arrow runs over another note: the pinned tr used to draw one across tl
    assert_arrows(alpha, corner, STYLE)

    # A note put right on the face gets a valid arrow all the same, and says it is in the way. The
    # notes that were not pinned still keep off the face, and are not flagged.
    face = face_box(alpha)
    on_face = plan_notes(alpha, TEXTS, STYLE, {"tl": [440, 100]})
    assert 440 < 540 < on_face["tl"].x + on_face["tl"].w
    assert on_face["tl"].y < 155 < on_face["tl"].y + on_face["tl"].h
    assert on_face["tl"].arrow is not None and on_face["tl"].clash
    assert all(overlap(rect(on_face[key]), face) == 0 for key in ("tr", "bl", "br"))
    assert not any(on_face[key].clash for key in ("tr", "bl", "br"))
    assert_arrows(alpha, on_face, STYLE)

    # A note pinned on the body has every start inside the grown person: no arrow (and the slow
    # search in assert_arrows agrees that none is valid)
    buried = plan_notes(alpha, TEXTS, STYLE, {"tl": [300, 450]})
    assert buried["tl"].arrow is None and buried["tr"].arrow is not None
    assert_arrows(alpha, buried, STYLE)

    # A note with no text is skipped, even when it has a place
    only = plan_notes(alpha, {"tl": "", "tr": "  ", "bl": "Hi"}, STYLE, {"tl": [10, 10]})
    assert list(only) == ["bl"]

    # What a hero rule asks of the layout after it moves a box. clash_of says whether a box is in
    # the way: on the face box, on a keep-out box, on another box (the box itself is skipped, and a
    # box that only touches another is not on it), or off the stage. Boxes may be NoteBoxes or tuples.
    ctx = prepare_matte(alpha, STYLE)
    rest = [free[key] for key in ("tr", "bl", "br")]
    assert not layout.clash_of(free["tl"], ctx, rest, (W, H))
    assert not layout.clash_of(free["tl"], ctx, [free["tl"], *rest], (W, H))
    assert layout.clash_of(dataclasses.replace(free["tl"], x=face[0] + 10), ctx, rest, (W, H))
    assert layout.clash_of(dataclasses.replace(free["tl"], x=free["bl"].x, y=free["bl"].y), ctx, rest, (W, H))
    top = rect(free["tl"])
    assert not layout.clash_of((top[0], top[3], top[0] + 60, top[3] + 30), ctx, [top], (W, H))  # touches
    assert layout.clash_of((top[0], top[3] - 1, top[0] + 60, top[3] + 29), ctx, [top], (W, H))  # 1 px on it
    assert layout.clash_of(free["tl"], ctx, [rect(free["tl"])], (W, H))  # a tuple is another box
    off = dataclasses.replace(free["tr"], x=W - free["tr"].w + 1)
    on = dataclasses.replace(off, x=off.x - 1)
    assert layout.clash_of(off, ctx, [], (W, H)) and not layout.clash_of(on, ctx, [], (W, H))
    assert layout.clash_of(dataclasses.replace(free["tl"], y=-1), ctx, [], (W, H))
    kept = prepare_matte(alpha, STYLE, keepout=[rect(free["tl"])])
    assert layout.clash_of(free["tl"], kept, [], (W, H)) and not layout.clash_of(free["tr"], kept, [], (W, H))

    # arrow_for finds the arrow of a note among the other boxes as plan_notes did, and place_box the
    # spot plan_notes gave a note of that size (here the others are in other quadrants, so they do
    # not change it). With no clean spot at all it says so, and gives the one that overlaps least.
    for key in KEYS:
        others = [box for other, box in free.items() if other != key]
        assert layout.arrow_for(key, free[key], others, ctx) == free[key].arrow, key
        x, y, clean = layout.place_box(key, (free[key].w, free[key].h), ctx, STYLE, others)
        assert (x, y, clean) == (free[key].x, free[key].y, True), key
    assert layout.arrow_for("tl", free["tl"], rest, prepare_matte(alpha, QUIET)) is None
    smothered = prepare_matte(alpha, STYLE, keepout=[(0, 0, W, H)])
    x, y, clean = layout.place_box("tl", (free["tl"].w, free["tl"].h), smothered, STYLE, [])
    assert clean is False and 0 <= x <= W - free["tl"].w and 0 <= y <= H - free["tl"].h
    assert layout.place_box("tl", (W + 1, 40), ctx, STYLE, []) == (0, 0, False)  # as wide as no stage

    # What a drag in the studio does: the matte is prepared once, and each layout with a note
    # pinned somewhere new gives what it would without the preparation
    for pins in (None, {"tl": [100, 90]}, {"tr": [43, 43]}, {"tl": [440, 100], "br": [500, 300]}):
        assert plan_notes(alpha, TEXTS, STYLE, pins, ctx=ctx) == plan_notes(
            alpha, TEXTS, STYLE, pins
        ), pins


def test_notes_one_above_another_are_visibly_apart(person_alpha, monkeypatch):
    # Notes that stand one above the other, in a side margin or a quadrant, keep ceil(0.4 x the font size)
    # px between them, so that they do not read as one paragraph, and any two notes keep 8 px at their
    # sides. A note closer than that to another is flagged, as one that overlaps it is. The old planner
    # is the same code with no space kept, which is what packed() runs: _apart gives the boxes no room.
    def packed(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(layout, "_STACK_GAP", 0)
            patch.setattr(layout, "_NOTE_GAP", 0)
            return plan_notes(*args, **kwargs)

    alpha = person_alpha(W, H)
    quads = quadrants(W, H, face_box(alpha), MARGIN)

    # The distance, to the pixel: one line each, the top note on the last rows of its quadrant and the
    # bottom one on the first rows of its own, so that nothing else can flag them. At 56 px the space
    # is 23 (22.4 rounded up) and at 34 it is 14 (13.6): a px less and both notes are flagged, and any
    # more and they are clean.
    short, half = {"tl": "Hi", "bl": "Yo"}, quads["tl"][3]
    for size in (56, 34):
        style, need = dataclasses.replace(STYLE, max_size=size), math.ceil(0.4 * size)
        for gap in (0, need - 1, need, need + 40):
            boxes = plan_notes(alpha, short, style, {"tl": [43, half - size], "bl": [43, half + gap]})
            tl, bl = boxes["tl"], boxes["bl"]
            assert (tl.size, bl.size) == (size, size) and bl.y - (tl.y + tl.h) == gap, (size, gap)
            assert tl.clash == bl.clash == (gap < need), (size, gap)

    # With the top right note pinned in the top left corner, the top left one is pushed out of it, and
    # it used to take the rows right under it, touching. Now it keeps its distance, and so does every pair.
    corner = plan_notes(alpha, TEXTS, STYLE, {"tr": [43, 43]})
    size = corner["tl"].size
    assert rect(corner["tl"])[1] - rect(corner["tr"])[3] >= math.ceil(0.4 * size)
    assert all(apart(rect(a), rect(b), size) for a, b in itertools.combinations(corner.values(), 2))
    touching = packed(alpha, TEXTS, STYLE, {"tr": [43, 43]})
    assert rect(touching["tl"])[1] == rect(touching["tr"])[3]

    # At their sides: with the top left note pinned where the top right one would be tucked in, one grid
    # step from the corner of its quadrant, the free note sits 8 px or more from it, and touched it
    pin = {"tl": [quads["tr"][0] + 8 * 44 + fit_note("Yo", STYLE, MAX_W)[2], 43]}
    both = {"tl": "Hi", "tr": "Yo"}
    touching = packed(alpha, both, STYLE, pin)
    assert touching["tl"].x - (touching["tr"].x + touching["tr"].w) == 0
    beside = plan_notes(alpha, both, STYLE, pin)
    assert beside["tl"].x - (beside["tr"].x + beside["tr"].w) >= 8

    # Four notes in one margin: the close-up 120 px to the left leaves the right margin as the only room,
    # and the four notes used to fit there at 56 px, edge to edge. They keep their distance now, which
    # takes the size down to 52, and every pair is as far apart as notes are kept.
    close = moved(close_alpha(), right=-120)
    crowded = packed(close, TEXTS, STYLE)
    assert {box.size for box in crowded.values()} == {56}
    assert not all(apart(rect(a), rect(b), 56) for a, b in itertools.combinations(crowded.values(), 2))
    margin = plan_notes(close, TEXTS, STYLE)
    assert {box.size for box in margin.values()} == {52}
    assert all(apart(rect(a), rect(b), 52) for a, b in itertools.combinations(margin.values(), 2))


def test_notes_that_cannot_keep_their_distance_are_flagged_and_not_packed(monkeypatch):
    # On a stage 180 px tall with the person's block in the middle, the four notes cannot all keep their
    # distance at any size, so they are flagged. The old planner put them where they overlapped least, and
    # that left them touching, or a few px apart. Now each takes the spot where it crowds
    # least, and no two are closer than 10 px (up and down the space asks for 14), and none overlaps.
    stage = np.zeros((180, W), np.float32)
    stage[20:, 300:800] = 1.0

    def closest(boxes):
        return min(
            max(max(b[0] - a[2], a[0] - b[2]), max(b[1] - a[3], a[1] - b[3]))
            for a, b in itertools.combinations([rect(box) for box in boxes.values()], 2)
        )

    spread = plan_notes(stage, TEXTS, STYLE)
    assert {box.size for box in spread.values()} == {34} and all(box.clash for box in spread.values())
    assert not all(apart(rect(a), rect(b), 34) for a, b in itertools.combinations(spread.values(), 2))
    assert closest(spread) == 10
    with monkeypatch.context() as patch:
        patch.setattr(layout, "_STACK_GAP", 0)
        patch.setattr(layout, "_NOTE_GAP", 0)
        packed = plan_notes(stage, TEXTS, STYLE)
    assert closest(packed) == 0  # with no distance to keep, they pack flush, touching


def test_notes_around_a_three_quarter_head_use_the_room_beside_it():
    # The freeze of proj_tall on a 9:16 card: the face box was 675 px wide for a head 345 wide, so the top
    # quadrants had under 180 px of room each, and tl, tr and bl stood in the left margin as one block,
    # two of them flagged, with no arrows. The face box is the head now, and the notes use the room
    # beside it.
    photo = threequarter_alpha()
    ctx = prepare_matte(photo, STYLE, keepout=TALL_KEEPOUT)
    boxes = plan_notes(photo, TEXTS, STYLE, ctx=ctx)
    face = ctx.face
    assert face == (191, 286, 616, 829)

    # One size, every note whole and clean, and no two of them touching: each pair keeps its distance
    size = boxes["tl"].size
    assert {box.size for box in boxes.values()} == {size}
    assert [box.clash for box in boxes.values()] == [False] * 4
    assert all(" ".join(box.lines) == TEXTS[key].upper() for key, box in boxes.items())
    assert all(apart(rect(a), rect(b), size) for a, b in itertools.combinations(boxes.values(), 2))

    # The top notes are level with the head and beside it, one on each side, not over it. The strip left
    # of the head is 148 px wide, and what sets the size is its longest word: EDUCATOR fits it at 36 px
    # and not at 38.
    tl, tr = rect(boxes["tl"]), rect(boxes["tr"])
    assert tl[2] <= face[0] and tr[0] >= face[2]
    assert all(box[1] < face[3] and box[3] > face[1] for box in (tl, tr))
    strip = face[0] - MARGIN
    assert font("marker", 36).getlength("EDUCATOR") <= strip < font("marker", 38).getlength("EDUCATOR")
    assert size == 36

    # Arrows go to the notes where the search finds a way: three of the four, none over the person
    assert [key for key, box in boxes.items() if box.arrow is not None] == ["tl", "tr", "bl"]
    person = photo > 0.5
    for box in boxes.values():
        if box.arrow is not None:
            assert not any(person[round(y), round(x)] for x, y in quad_points(*box.arrow, 16))


# --- arrows -----------------------------------------------------------------------------------


def test_arrow_ends_outside_sticker_edge(person_alpha):
    synthetic = person_alpha(W, H)
    close = close_alpha()

    # Each note tries starts along the two sides that face the person, walks from each toward
    # three targets (the bbox center, the face box's center, and the nearest point of the grown
    # person), and keeps the valid arrow whose chord is nearest 90 px. assert_arrows holds every
    # arrow to the validity rules and checks the choice against that search done the slow way.
    # Both people, and the close-up again with a compact style (40 px down to 34).
    compact = NoteStyle("marker", 40, 34, "upper", band=22)
    arrowed = {}
    for name, alpha, style in (
        ("synthetic", synthetic, STYLE), ("close-up", close, STYLE), ("compact", close, compact),
    ):
        boxes = plan_notes(alpha, TEXTS, style)
        assert_arrows(alpha, boxes, style)
        arrowed[name] = [key for key, box in boxes.items() if box.arrow is not None]

    # The close-up fills the stage, and the notes beside the face used to be left with 12 to 14 px
    # of room for a straight arrow: it keeps arrows on at least 3 of its 4 notes (all four now)
    assert len(arrowed["close-up"]) >= 3
    assert arrowed["synthetic"] == arrowed["compact"] == KEYS

    # A full-screen stage is 1920 px tall, so its arrows may be twice as long (the chord limits and
    # the 90 px the choice is nearest to go with the height over 608, between 1 and 2), and its
    # notes stay within that reach of the person: the close-up on it, with bare stage above and
    # below, gets arrows, and some are longer than the 180 px of a 16:9 stage.
    tall = full_screen(close)
    boxes = plan_notes(tall, TEXTS, STYLE)
    assert sum(box.arrow is not None for box in boxes.values()) >= 2
    chords = [math.dist(box.arrow[0], box.arrow[2]) for box in boxes.values() if box.arrow]
    assert max(chords) > 180 and all(chord <= 360 for chord in chords)
    assert_arrows(tall, boxes, STYLE)

    # silhouette_target is the nearest point of the person grown by band + 8, found at half size:
    # on that grown person to within a pixel and a half, and as near to the start as a full-size
    # mask would give, give or take a couple of pixels
    grown = pixels(dilate(synthetic, STYLE.band + 8) > 0.5)
    for start in [(100, 100), (980, 120), (60, 560), (1000, 400), (540, 5)]:
        end = silhouette_target(synthetic, start, STYLE.band)
        assert type(end) is tuple and all(type(v) is int for v in end)
        assert synthetic[end[1], end[0]] == 0
        assert abs(math.dist(start, end) - distance_to(grown, start)) <= 3
        assert distance_to(grown, end) <= 1.5

    # A start inside the sticker edge, or even inside the person, is moved out to it
    for start in [(540, 300), (440, 150), (300, 450)]:
        end = silhouette_target(synthetic, start, STYLE.band)
        assert synthetic[end[1], end[0]] == 0
        assert distance_to(grown, end) <= 1.5

    # Nobody in the matte: the helper aims at the middle of the default face box,
    # (0.36 W, 0.08 H, 0.64 W, 0.55 H), but there is nothing to point at, so no note gets an arrow
    nobody = np.zeros((H, W), np.float32)
    assert silhouette_target(nobody, (100, 100), STYLE.band) == (540, 191)
    assert all(box.arrow is None for box in plan_notes(nobody, TEXTS, STYLE).values())

    # A speck under 1% of the picture is nobody to person_boxes, so it gets no arrows either
    speck = np.zeros((H, W), np.float32)
    speck[10:30, 10:30] = 1.0
    assert person_boxes(speck)["head"] is None
    assert all(box.arrow is None for box in plan_notes(speck, TEXTS, STYLE).values())


def test_no_arrows_when_style_says_so(person_alpha, monkeypatch):
    alpha = person_alpha(W, H)
    with_arrows = plan_notes(alpha, TEXTS, STYLE)
    without = plan_notes(alpha, TEXTS, QUIET)

    assert list(without) == KEYS and all(box.arrow is None for box in without.values())
    assert any(box.arrow is not None for box in with_arrows.values())
    # Only the arrows are different. The notes keep off the body when there are arrows to leave
    # room for, so the places can differ a little (the bottom right note slides up here). The text
    # and the sizes are the same.
    assert [(box.lines, box.size) for box in without.values()] == [
        (box.lines, box.size) for box in with_arrows.values()
    ]

    # What a style leaves out is arrows on, no padding, no sticker edge, and a line height of 1
    bare = NoteStyle("marker", 56, 34, "upper")
    assert (bare.line_h, bare.pad, bare.band, bare.arrows) == (1.0, 0, 0, True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bare.arrows = False

    # Dilating is the slow part of a layout, and it depends on the matte alone. A layout of a
    # matte not prepared does it three times, however many notes there are (not once per spot):
    # for the sticker edge (band), the person's grown edge that the arrows aim at (band + 8) and
    # the near mask (band + 8 + 56, found at a quarter of the size, so a quarter of the radius).
    # Without arrows, once: for the sticker edge.
    real, radii = layout.dilate, []

    def counting(mask, radius, *args, **kwargs):
        radii.append(radius)
        return real(mask, radius, *args, **kwargs)

    monkeypatch.setattr(layout, "dilate", counting)
    plan_notes(alpha, {"tl": TEXTS["tl"]}, STYLE)
    one_note = list(radii)
    radii.clear()
    plan_notes(alpha, TEXTS, STYLE)
    assert radii == one_note == [STYLE.band, STYLE.band + 8, (STYLE.band + 8 + 56) / 4]
    radii.clear()
    plan_notes(alpha, TEXTS, QUIET)
    assert radii == [STYLE.band]

    # Prepared once, the matte costs those dilations once, and no layout of it dilates again, with
    # notes pinned or not, with arrows or not (a context made for arrows serves a style without)
    radii.clear()
    ctx = prepare_matte(alpha, STYLE)
    assert radii == [STYLE.band, STYLE.band + 8, (STYLE.band + 8 + 56) / 4]
    radii.clear()
    layouts = [plan_notes(alpha, TEXTS, STYLE, ctx=ctx)]
    layouts.append(plan_notes(alpha, TEXTS, STYLE, {"tl": [100, 90]}, ctx=ctx))
    layouts.append(plan_notes(alpha, TEXTS, STYLE, {"br": [600, 400], "tr": [700, 60]}, ctx=ctx))
    layouts.append(plan_notes(alpha, TEXTS, QUIET, ctx=ctx))
    assert radii == []
    # and each gives what the same call without it does
    assert layouts[0] == with_arrows and layouts[3] == without
    assert layouts[1] == plan_notes(alpha, TEXTS, STYLE, {"tl": [100, 90]})
    assert layouts[2] == plan_notes(alpha, TEXTS, STYLE, {"br": [600, 400], "tr": [700, 60]})

    # Boxes no note may cover (the title room of a 9:16 card is one) go to prepare_matte, and the
    # layout takes them from the context. They are rounded to whole pixels, and cost no dilation
    # of their own. Given none, or one inside the face (which no note may cover anyway), nothing
    # changes at all.
    assert ctx.keepout == ()
    face = face_box(alpha)
    radii.clear()
    inside_face = [[face[0] + 10.2, face[1] + 10, face[2] - 10.4, face[1] + 59.6]]
    guarded = prepare_matte(alpha, STYLE, keepout=inside_face)
    assert radii == [STYLE.band, STYLE.band + 8, (STYLE.band + 8 + 56) / 4]
    assert guarded.keepout == ((face[0] + 10, face[1] + 10, face[2] - 10, face[1] + 60),)
    with pytest.raises(ValueError, match="keepout"):
        prepare_matte(alpha, STYLE, keepout=[(1, 2, 3)])
    empty = prepare_matte(alpha, STYLE, keepout=[])
    assert empty.keepout == ()
    assert plan_notes(alpha, TEXTS, STYLE, ctx=empty) == with_arrows
    assert plan_notes(alpha, TEXTS, STYLE, ctx=guarded) == with_arrows

    # They are priced like the face, hard: over the place where the top left note went, that note
    # (and no other) is somewhere else, whole and clean, and no note touches the box. Pinned onto
    # it, a note stays and is flagged, and the others keep off both.
    tl = with_arrows["tl"]
    keep = (tl.x - 12, tl.y - 12, tl.x + tl.w + 12, tl.y + tl.h + 12)
    ctx_keep = prepare_matte(alpha, STYLE, keepout=[keep])
    kept = plan_notes(alpha, TEXTS, STYLE, ctx=ctx_keep)
    assert all(overlap(rect(box), keep) == 0 for box in kept.values())
    assert not any(box.clash for box in kept.values())
    assert all(" ".join(box.lines) == TEXTS[key].upper() for key, box in kept.items())
    assert rect(kept["tl"]) != rect(tl) and kept != with_arrows
    pinned = plan_notes(alpha, TEXTS, STYLE, {"tl": [tl.x, tl.y]}, ctx=ctx_keep)
    assert (pinned["tl"].x, pinned["tl"].y) == (tl.x, tl.y) and pinned["tl"].clash
    assert all(overlap(rect(box), keep) == 0 and not box.clash for key, box in pinned.items() if key != "tl")
    # A box over the whole stage leaves no clean spot: every note is flagged, and still whole.
    smothered = plan_notes(alpha, TEXTS, STYLE, ctx=prepare_matte(alpha, STYLE, keepout=[(0, 0, W, H)]))
    assert all(box.clash for box in smothered.values())
    assert all(" ".join(box.lines) == TEXTS[key].upper() for key, box in smothered.items())

    # The use it is for: a title, 16 px of air around two lines, where the top quadrants begin. On a tall
    # stage with the person at the top, a note sits in that box unless it is kept out of it.
    tall = np.zeros((1920, 1080), np.float32)
    tall[:H] = alpha
    title = (189, 134, 891, 290)
    crowded = plan_notes(tall, TEXTS, STYLE)
    assert any(overlap(rect(box), title) > 0 for box in crowded.values())
    clear = plan_notes(tall, TEXTS, STYLE, ctx=prepare_matte(tall, STYLE, keepout=[title]))
    assert all(overlap(rect(box), title) == 0 for box in clear.values())
    assert all(" ".join(box.lines) == TEXTS[key].upper() for key, box in clear.items())

    # An arrow's curve may not enter a keep-out box either, its start included. A small box in the
    # middle of the top left arrow, away from the notes: the arrow without it runs through the
    # box, and with it that note's arrow goes another way (or none does), and no note moves.
    run = with_arrows["tl"].arrow
    middle = quad_points(*run, 16)[8]
    spot = (round(middle[0]) - 8, round(middle[1]) - 8, round(middle[0]) + 9, round(middle[1]) + 9)

    def enters(arrow, box):
        points = quad_points(*arrow, 16) if arrow is not None else ()
        return any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in points)

    around = plan_notes(alpha, TEXTS, STYLE, ctx=prepare_matte(alpha, STYLE, keepout=[spot]))
    assert enters(run, spot) and not enters(around["tl"].arrow, spot) and around["tl"].arrow != run
    assert all(rect(around[key]) == rect(with_arrows[key]) for key in KEYS)
    assert all(not enters(box.arrow, spot) for box in around.values())

    # On a tall stage the quadrants of a style with arrows stay within an arrow's reach (twice 180
    # px there) of the person, up and down. A style with no arrows has nothing to reach, and its
    # quadrants are the whole stage's.
    person = full_screen(alpha)
    tall_face, bbox = face_box(person), person_boxes(person)["bbox"]
    reached = quadrants(1080, 1920, tall_face, MARGIN, rows=(bbox[1] - 360, bbox[3] + 360))
    assert prepare_matte(person, STYLE).quads == reached
    assert prepare_matte(person, QUIET).quads == quadrants(1080, 1920, tall_face, MARGIN) != reached

    # What a context remembers of the prices is capped. With room for very little it forgets them
    # over and over, mid-layout, and still gives the same layouts.
    monkeypatch.setattr(layout, "_CACHE_BYTES", 20_000)
    capped = prepare_matte(alpha, STYLE)
    assert plan_notes(alpha, TEXTS, STYLE, ctx=capped) == with_arrows
    assert plan_notes(alpha, TEXTS, STYLE, {"tl": [100, 90]}, ctx=capped) == layouts[1]
    assert 0 < len(capped.cache) < len(ctx.cache)
    held = sum(part.nbytes for found in capped.cache.values() for part in found)
    assert capped.cache.held == held

    # A context is only good for the matte and the style it was made for
    with pytest.raises(ValueError, match="ctx"):
        plan_notes(person_alpha(540, 304), TEXTS, STYLE, ctx=ctx)  # another size
    with pytest.raises(ValueError, match="ctx"):
        plan_notes(alpha, TEXTS, dataclasses.replace(STYLE, band=10), ctx=ctx)  # another band
    with pytest.raises(ValueError, match="ctx"):
        plan_notes(alpha, TEXTS, STYLE, ctx=prepare_matte(alpha, QUIET))  # no arrows prepared


def test_notes_with_no_clean_spot_can_sit_flush_against_each_other():
    # A person who fills the stage leaves only the strip above the face, and these four notes fit
    # along it in one row only flush against each other: 1077 px of the stage's 1080 at the smallest
    # size. The spots are 8 px apart, so a row on them alone does not fit, and two notes overlapped.
    # Where no spot is clean, a note may also take the spots flush against the notes already placed
    # and against the edges of its field, and then the row fits with no note on another.
    wall = np.zeros((H, W), np.float32)
    wall[60:, 150:930] = 1.0
    texts = dict(TEXTS, tr="@your_handle", br="Best friend: Clover (sorry, mom)")
    boxes = plan_notes(wall, texts, STYLE)
    assert sum(box.w for box in boxes.values()) == 1077 and {box.size for box in boxes.values()} == {34}
    card = audit(wall, boxes, texts)
    assert (card["cut"], card["notes"]) == ([], 0)
