"""The style engine: puts a recipe's layers together into the stage and the notes of a card.

compose_stage makes the picture under the notes: the background, then the edge layers, then the
person, then the film grain when the recipe has it. style_layout lays out the notes (layout.
plan_notes), with the room a hero needs kept free, and then applies the recipe's hero rule, which
moves or adds boxes. draw_overlay draws the arrows, the notes, the hero and the extras onto an
RGBA layer the size of the stage, in the order of the approved mockup: the arrows go under the
notes, whose dark glow shades the first few pixels of an arrow where it leaves its note.

Each layer's options are drawn by the module of that layer (background, edges, notes), and an
id that is no option of the layer raises ValueError there.

The layout is made at the plan, 1080 px wide, and tuned there. compose_stage and draw_overlay can
draw it on a picture bigger than that (a render at the clip's own size), given a raster.Scale: the
layers multiply every size in px by its k, and the boxes of the layout are scaled for the drawing.
Nothing about the layout depends on the scale, so a render has the notes, arrows and extras the
studio showed. At the plan (IDENTITY) nothing changes, to the bit.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image

from ..analyze import person_boxes
from ..glyphs import missing as missing_glyphs
from ..layout import (
    MatteContext,
    NoteBox,
    arrow_for,
    clash_of,
    fit_note,
    fit_note_cut,
    place_box,
    plan_notes,
    prepare_matte,
    room_below,
    room_right,
)
from ..raster import IDENTITY, Scale, grain, over, screen
from ..project import NOTE_KEYS
from . import background, edges, extras, notes
from .recipes import unavailable

_SATURATION = 1.08  # the person's colors are pushed this far from gray, to stand out of the room
_FILM_GRAIN = 9  # levels of the soft-light grain over the whole stage
_BLENDS = {"normal": over, "screen": screen}


def compose_stage(
    recipe: dict, rgb: np.ndarray, alpha: np.ndarray, variant: int, scale: Scale = IDENTITY
) -> np.ndarray:
    """The stage of a card without its notes: float32, height x width x 3, in 0-1.

    rgb is the frozen frame (0-1) and alpha its cleaned matte, both of the stage's size, and
    variant the boil picture (0, 1 or 2): the dots, the grain and the wobble of the edge change
    with it. The stage is `scale` bigger than the plan the layout is made at (rgb and alpha are
    that size, and the matte was cleaned for it). The layers go on in this order:

    1. The background, drawn from the whole frame, the person included.
    2. The layers of the edge, under the person, each over the one before or screened.
    3. The person: the frame's pixels with their saturation raised by 8%, around the mean of
       the three channels, composited by alpha over the edge.
    4. With extras.vhs and extras.leak, the scanlines and channel offset and then the light leak
       (extras.stage_effects).
    5. With extras.grain, soft-light grain over everything: gray noise of 9 levels, strongest in
       the midtones and nothing in pure black or white. It uses the same seed as the background's
       own grain, which only makes the film grain a little stronger, never a pattern.
    """
    pal = recipe["palette"]
    head = person_boxes(alpha)["head"]
    img = background.render(recipe["bg"], rgb, alpha, pal, variant, head, scale)
    for color, mask, blend in edges.render(recipe["edge"], alpha, pal, variant, scale):
        img = _BLENDS[blend](img, color, mask)

    mean = rgb.mean(axis=-1, keepdims=True)
    person = np.clip(mean + (rgb - mean) * _SATURATION, 0.0, 1.0)
    img = np.clip(over(img, person, alpha), 0.0, 1.0)

    img = extras.stage_effects(recipe, img, head, scale)

    if recipe["extras"]["grain"]:
        noise = grain(np.zeros(img.shape[:2], np.float32), _FILM_GRAIN, variant, scale.noise_shape(img.shape))
        img = np.clip(img + noise[..., None] * (1.0 - np.abs(2.0 * img - 1.0)), 0.0, 1.0)
    return img.astype(np.float32)


def matte_context(
    recipe: dict, alpha: np.ndarray, keepout: list[tuple[int, int, int, int]] | None = None
) -> MatteContext:
    """What style_layout needs from the matte alone, for the recipe's notes style (see
    layout.prepare_matte). It is slow (a few dilations), so build it once for a matte, a notes
    style and a keep-out and pass it to every style_layout call. keepout is a list of
    (x0, y0, x1, y1) boxes in stage pixels that no note may cover, such as the title room of a
    9:16 card; none is the default."""
    return prepare_matte(alpha, notes.note_style(recipe), keepout)


# --- hero rules -------------------------------------------------------------------------------
#
# A hero's rule moves boxes after the notes are planned: it gets the recipe, the planned boxes, the
# matte context (ctx.quads are the room of each corner, ctx.face the face to keep clear), the copy
# and the pinned positions, and returns the boxes the card is drawn from. Add a hero here, with its
# drawing in notes._HEROES. A rule that moves a box must say again whether it is in the way
# (layout.clash_of, and the cut flag the planner gave the note), and takes the boxes of every
# other corner from ctx.quads, never from quadrants() with its default rows.

_BURST_SCALES = (1.0, 0.95, 0.90, 0.85, 0.80)  # the burst at full size, and what it shrinks to to fit
_BURST_GAP = 16  # px between the burst and the note that moves under it
_BURST_EDGE = 12  # px of the stage's right edge the burst keeps when it moves off the face
_BURST_AIR = 16  # px of air the notes keep at the sides of the burst and the note under it
_BURST_ROOM_BELOW = 32  # and under it: the arrow of the note below needs room to run
_HUD_SHIFT = 58  # px the tl note moves down for the label that takes its corner (24 and a gap of 34)
_HUD_EDGE = 20  # px between the bottom label and the bottom of the room (the stage's, or the safe area's)
_HUD_AIR = 16  # px of air the notes keep around the bottom label when it is among their rows


def _stage_size(ctx: MatteContext) -> tuple[int, int]:
    return ctx.shape[1], ctx.shape[0]


def _held(position, size: tuple[int, int], stage: tuple[int, int]) -> tuple[int, int]:
    """A pinned [x, y] from the project file as a corner, rounded and moved onto the stage."""
    (w, h), (width, height) = size, stage
    x, y = round(position[0]), round(position[1])
    return min(max(x, 0), max(width - w, 0)), min(max(y, 0), max(height - h, 0))


def _below_keepouts(ctx: MatteContext, x: int, y: int, size: tuple[int, int]) -> int:
    """The y a box at column x goes to so that it does not sit in a keep-out box: it is moved to the
    bottom of any box it is in, and again for the next one."""
    w, h = size
    moved = True
    while moved:
        moved = False
        for x0, y0, x1, y1 in ctx.keepout:
            if x < x1 and x + w > x0 and y < y1 and y + h > y0:
                y, moved = y1, True
    return y


def _checked(box: NoteBox, ctx: MatteContext, others) -> NoteBox:
    """A box that has been moved, with its clash flag worked out again: it is in the way when its
    text was cut, or when layout.clash_of says so."""
    return replace(box, clash=box.cut or clash_of(box, ctx, others, _stage_size(ctx)))


def _arrows_again(boxes: dict[str, NoteBox], ctx: MatteContext, keys) -> dict[str, NoteBox]:
    """The boxes with the arrows of the notes `keys` found again among all of them: the planner
    found them before the hero's boxes were there, so one could run through a box that is new."""
    result = dict(boxes)
    for key in keys:
        if key in result:
            others = [box for other, box in result.items() if other != key]
            result[key] = replace(result[key], arrow=arrow_for(key, result[key], others, ctx))
    return result


def _hero_none(recipe, boxes, ctx, copy, positions) -> dict[str, NoteBox]:
    """No hero element: the notes stay where they were planned."""
    return boxes


def _burst_spot(ctx: MatteContext, size: tuple[int, int], positions: dict) -> tuple[int, int]:
    """The top-left corner of a burst of `size` (w, h): where the person put it, or else the top of
    the top right room, below any keep-out box. When that box shares any pixel with the face box (a
    real overlap of the two rectangles, wherever the burst's own edges are), it moves right, to the
    face box's right side if the burst fits there, and else as far right as the stage lets it go. That
    may leave it on the air around the face, which it may use (see _tight), but never on the head: a
    burst that is on the head box is flagged."""
    w, h = size
    stage = _stage_size(ctx)
    if positions.get("burst") is not None:
        return _held(positions["burst"], (w, h), stage)
    x, y = ctx.quads["tr"][2] - w, ctx.quads["tr"][1]
    y = _below_keepouts(ctx, x, y, (w, h))
    face = ctx.face
    if x < face[2] and x + w > face[0] and y < face[3] and y + h > face[1]:
        x = min(face[2], stage[0] - _BURST_EDGE - w)
    return x, y


def _burst_at(ctx: MatteContext, positions: dict) -> tuple[int, int, int, int, float]:
    """The burst's box and its scale, as (x, y, w, h, scale). The burst takes its full size, 310 x
    220, where _burst_spot puts it. When that is on the head (judged as the burst is, against the
    face box without its air: see _tight) it shrinks, the box and its lettering together, to the
    first of 0.95, 0.90, 0.85 and 0.80 of its size at which it is not (it moves right as it
    narrows, and gets shorter). If none is, it stays at 0.80 and is flagged. A burst the person
    dragged keeps its size and where it was put."""
    positions = positions or {}
    if positions.get("burst") is not None:
        w, h = notes.burst_box()
        x, y = _burst_spot(ctx, (w, h), positions)
        return x, y, w, h, 1.0
    head, stage = _tight(ctx), _stage_size(ctx)
    for scale in _BURST_SCALES:
        w, h = notes.burst_box(scale)
        x, y = _burst_spot(ctx, (w, h), positions)
        if not clash_of((x, y, x + w, y + h), head, [], stage):
            break
    return x, y, w, h, scale


def _burst_room(recipe, ctx, copy, positions) -> list[tuple[int, int, int, int]]:
    """What the notes must keep out of so that the burst and the tr note under it fit: one box
    around the burst and the room that note needs under it (the room it needs at the biggest size,
    which is what it gets unless the card's size comes down). The notes are planned with it as a
    keep-out box, so none of them, and none of their arrows, is where the burst goes."""
    positions = positions or {}
    x, y, w, h, _ = _burst_at(ctx, positions)
    left, right, bottom = x, x + w, y + h
    text = (copy["notes"].get("tr") or "").strip()
    if text and positions.get("tr") is None:
        _, _, note_w, note_h = fit_note(text, notes.note_style(recipe), ctx.max_w)
        start = min(max(x + (w - note_w) // 2, 0), _stage_size(ctx)[0] - note_w)
        left, right, bottom = min(left, start), max(right, start + note_w), bottom + _BURST_GAP + note_h
    return [(left - _BURST_AIR, y, right + _BURST_AIR, bottom + _BURST_ROOM_BELOW)]


def _note_at_size(key: str, text: str, recipe: dict, ctx: MatteContext, boxes) -> NoteBox:
    """A note that the planner did not plan, set at the size of the ones it did (they all share
    one), or at the biggest size that fits when there are none: at the origin, for a rule to put."""
    style = notes.note_style(recipe)
    size = next((box.size for box in boxes), None)
    if size is not None:
        style = replace(style, max_size=size, min_size=size)
    lines, size, w, h, cut = fit_note_cut(text, style, ctx.max_w)
    cut = cut or bool(missing_glyphs(style.font, text, style.case))
    return NoteBox(key, lines, size, 0, 0, w, h, None, cut=cut)


def _hero_burst(recipe, boxes, ctx, copy, positions) -> dict[str, NoteBox]:
    """The name burst takes the top of the top right room, and the tr note goes under it, centered
    and with no arrow (it is looked for another place only if there is no room under the burst).
    The tr note was not planned, so that it does not take room from the others: it is set at their
    size here. A burst the person dragged stays where it was put, and a tr note that was dragged
    stays too (it was planned, like any note that is pinned).

    The burst is judged against the head box without its air (see _tight), and so is the tr note
    while it is under the burst, which is where the burst goes: they may be on the air around the
    face, never on the head. A tr note that had to look for another place, or was dragged, is judged
    against the whole face box, as every other note is. Where the full-size burst is on the head, it
    shrinks to fit (see _burst_at), and the tr note goes under the smaller one."""
    positions = positions or {}
    stage, head = _stage_size(ctx), _tight(ctx)
    text = notes.burst_text(copy)
    x, y, w, h, scale = _burst_at(ctx, positions)
    burst = NoteBox(
        "burst", [text], notes.burst_size(text, scale), x, y, w, h, None, cut=bool(missing_glyphs("comic", text))
    )

    others = [box for key, box in boxes.items() if key != "tr"]
    tr = boxes.get("tr")
    pinned = tr is not None
    note_text = (copy["notes"].get("tr") or "").strip()
    if tr is None and note_text:
        tr = _note_at_size("tr", note_text, recipe, ctx, others)
    judge = ctx
    if tr is not None:
        tr = replace(tr, arrow=None)
        if not pinned:
            tr = replace(
                tr,
                x=min(max(burst.x + (burst.w - tr.w) // 2, 0), stage[0] - tr.w),
                y=burst.y + burst.h + _BURST_GAP,
            )
            if clash_of(tr, head, [burst, *others], stage):  # no room under the burst: find another place
                x, y, _ = place_box("tr", (tr.w, tr.h), ctx, notes.note_style(recipe), [burst, *others])
                tr = replace(tr, x=x, y=y)
            else:
                judge = head
        tr = _checked(tr, judge, [burst, *others])
    result = {key: box for key, box in boxes.items() if key != "tr"}
    if tr is not None:
        result["tr"] = tr
    result["burst"] = _checked(burst, head, [*others, *([tr] if tr else [])])
    final = {key: result[key] for key in (*NOTE_KEYS, "burst") if key in result}
    return _arrows_again(final, ctx, ("tl", "bl", "br"))


def _hero_stats(recipe, boxes, ctx, copy, positions) -> dict[str, NoteBox]:
    """The stats panel takes the place of the tr note (which was not planned) and keeps its key, so
    that dragging "tr" moves the panel. It is as big as its rows make it, and it goes where a note
    of that size would (layout.place_box), unless the person dragged it."""
    positions = positions or {}
    rows = notes.stats_rows(copy)
    w, h = notes.stats_size(rows)
    others = [box for key, box in boxes.items() if key != "tr"]
    judge = ctx  # what the panel is judged by: the face box and its air, unless it had to give the air up
    if positions.get("tr") is not None:
        x, y = _held(positions["tr"], (w, h), _stage_size(ctx))
    else:
        style = notes.note_style(recipe)
        x, y, clean = place_box("tr", (w, h), ctx, style, others)
        if not clean:  # no room clear of the face and its air: give the air up, and keep clear of the head
            tight_x, tight_y, clean = place_box("tr", (w, h), _tight(ctx), style, others)
            if clean:
                x, y, judge = tight_x, tight_y, _tight(ctx)
    cut = any(missing_glyphs("terminal", label) for label, _ in rows)  # (the labels are in capitals already)
    panel = NoteBox("tr", [label for label, _ in rows], notes.STATS_SIZE, x, y, w, h, None, cut=cut)
    result = {**boxes, "tr": _checked(panel, judge, others)}
    final = {key: result[key] for key in NOTE_KEYS if key in result}
    return _arrows_again(final, ctx, ("tl", "bl", "br"))


def _hud_bottom(ctx: MatteContext) -> tuple[int, int, int, int]:
    """The box of the HUD's bottom label, "PRESS START": at the bottom right of the room the stage
    leaves, 20 px above its bottom and a margin left of its right edge. On a stage with no safe area
    (any shape but 9:16) that is the stage's own bottom right corner, in the margin under the notes. On
    a 9:16 card the room is the Reels safe area, so the label sits above the bottom band and left of
    the rail."""
    text, size = notes.HUD_BOTTOM
    width = notes.hud_width(text, size)
    y = room_below(ctx) - _HUD_EDGE - size
    return room_right(ctx, y, y + size) - ctx.margin - width, y, width, size


def _hud_room(recipe, ctx, copy, positions) -> list[tuple[int, int, int, int]]:
    """What the notes must keep out of so that the bottom label has its place: a box around it with
    some air, but only when the label is among the rows of the bottom notes (on a 9:16 card it
    is, since the notes' rows end at the bottom band; on any other stage it is in the margin under
    them, and the notes are planned as they always were)."""
    x, y, w, h = _hud_bottom(ctx)
    if y >= ctx.quads["br"][3]:
        return []
    return [(x - _HUD_AIR, y - _HUD_AIR, x + w + _HUD_AIR, y + h + _HUD_AIR)]


def _hero_hud(recipe, boxes, ctx, copy, positions) -> dict[str, NoteBox]:
    """The HUD's two labels: "PLAYER 1" in the top left corner of the room, where the tl note would
    be, which moves 58 px down (unless the person dragged it), and "PRESS START" at the bottom
    right of the room the stage leaves (see _hud_bottom). The tl note's arrow is found again where
    it is now."""
    positions = positions or {}
    (top_text, top_size), (bottom_text, bottom_size) = notes.HUD_TOP, notes.HUD_BOTTOM
    top_w = notes.hud_width(top_text, top_size)
    x, y = ctx.quads["tl"][0], ctx.quads["tl"][1]
    y = _below_keepouts(ctx, x, y, (top_w, top_size))
    bottom_x, bottom_y, bottom_w, _ = _hud_bottom(ctx)
    labels = {
        "hud_top": NoteBox("hud_top", [top_text], top_size, x, y, top_w, top_size, None),
        "hud_bottom": NoteBox(
            "hud_bottom", [bottom_text], bottom_size, bottom_x, bottom_y, bottom_w, bottom_size, None
        ),
    }
    moved = dict(boxes)
    shifted = boxes.get("tl") is not None and positions.get("tl") is None
    if shifted:
        moved["tl"] = replace(boxes["tl"], y=boxes["tl"].y + _HUD_SHIFT)
    everything = {**moved, **labels}
    again = (*labels, *(("tl",) if shifted else ()))  # the boxes that are new or moved: judged again
    result = {
        key: _checked(box, ctx, [other for name, other in everything.items() if name != key])
        if key in again else box
        for key, box in everything.items()
    }
    final = {key: result[key] for key in (*NOTE_KEYS, "hud_top", "hud_bottom") if key in result}
    return _arrows_again(final, ctx, ("tl",) if shifted else ())


_HERO_RULES = {"none": _hero_none, "burst": _hero_burst, "stats": _hero_stats, "hud": _hero_hud}
# What a hero keeps the notes out of while they are planned, for the heroes that need it:
# (recipe, ctx, copy, positions) -> a list of (x0, y0, x1, y1) boxes.
_HERO_ROOM = {"burst": _burst_room, "hud": _hud_room}


def _replaced(recipe: dict, positions: dict | None) -> str | None:
    """The note whose text is not planned, because the hero puts a box of its own where it would go:
    the stats panel takes the tr note's place, and the burst moves that note (unless the person
    pinned it, and then it is planned like any note that is pinned)."""
    hero = recipe["hero"]
    if hero == "stats" or (hero == "burst" and (positions or {}).get("tr") is None):
        return "tr"
    return None


def _changed(ctx: MatteContext, **changes) -> MatteContext:
    """The context with some of its fields changed. It has a price cache of its own, since what a
    spot costs depends on the face box and on the boxes to keep out of."""
    return replace(ctx, cache=type(ctx.cache)(), **changes)


def _tight(ctx: MatteContext) -> MatteContext:
    """The context with the face box taken without the air around it (a margin of the stage's
    width on either side), where that air is not cut off by the stage's edge: the head itself.

    The stats panel and the burst use it, and it is an accepted exception to the rule that a box
    keeps off the air around the face (the controller's ruling on the review, option (a), and
    ruling S2 for the burst). The panel is placed against the whole face box first. Where it does
    not fit there (on a close-up the room beside the head can be narrower than the panel: 341 px
    for 373 on the plate's matte) it falls back to this context: it may then overlap the air, but
    never the head, and that overlap is not flagged as a clash. The burst, and the tr note while it
    is under the burst, are judged against this context always: the burst is as big as it is, and
    where the head leaves less room than that it sits on the air. Every other box, the HUD's labels
    included, is judged against the whole face box with its air."""
    x0, y0, x1, y1 = ctx.face
    width = ctx.shape[1]
    left = x0 + ctx.margin if x0 > 0 else 0
    right = x1 - ctx.margin if x1 < width else width
    return _changed(ctx, face=(left, y0, right, y1))


def style_layout(
    recipe: dict,
    alpha: np.ndarray,
    copy: dict,
    positions: dict | None = None,
    ctx: MatteContext | None = None,
) -> dict[str, NoteBox]:
    """Where everything a recipe puts on the card goes, as {key: NoteBox} in stage pixels.

    The four notes are laid out by layout.plan_notes in the recipe's notes style (copy["notes"]
    holds their texts, and positions pins the notes the person moved), with the room a hero keeps
    free, and then the recipe's hero rule moves or adds boxes. (The tr note is not planned when a
    hero takes its place: the stats panel, or the burst, which moves it.) Pass ctx, from
    matte_context, to skip the slow work that depends on the matte alone: with it a layout takes
    tens of milliseconds, without it about a third of a second.
    The context also carries the boxes the notes must keep out of (the title room and the safe area,
    on a 9:16 card); a call without one keeps out of none. An id that is no hero is a ValueError.
    """
    try:
        hero_rule = _HERO_RULES[recipe["hero"]]
    except KeyError:
        raise unavailable("hero", recipe["hero"]) from None
    style = notes.note_style(recipe)
    if ctx is None:
        ctx = prepare_matte(alpha, style)
    planning = ctx
    if recipe["hero"] in _HERO_ROOM:
        room = _HERO_ROOM[recipe["hero"]](recipe, ctx, copy, positions)
        if room:  # (a changed context starts with a price cache of its own, so none is made for no room)
            planning = _changed(ctx, keepout=(*ctx.keepout, *room))
    texts = {key: text for key, text in copy["notes"].items() if key != _replaced(recipe, positions)}
    boxes = plan_notes(alpha, texts, style, positions, planning)
    return hero_rule(recipe, boxes, ctx, copy, positions)


def draw_overlay(
    img: Image.Image,
    recipe: dict,
    layout: dict[str, NoteBox],
    copy: dict,
    visible,
    face: tuple[int, int, int, int] | None = None,
    keepout: tuple[tuple[int, int, int, int], ...] = (),
    scale: Scale = IDENTITY,
    person: np.ndarray | None = None,
    variant: int = 0,
) -> None:
    """Draw the arrows, the notes, the hero and the extras onto an RGBA image the size of the
    stage, in place. layout is style_layout's, `visible` the groups on now (a set of "bottom",
    "top" and "arrows"), and face the face box (x0, y0, x1, y1) of the layout's context
    (MatteContext.face) for the extras that sit beside it; None draws none of those. keepout is
    the layout's context's (MatteContext.keepout): the extras keep out of those boxes as the notes
    do, so that a star or a piece of tape never lands on the title room.

    The layout, the face and the keep-out boxes are the ones made at the plan, whatever the size of
    the image: scale says how much bigger than the plan it is, and the drawing scales them. person is
    the plan's mask of the person (MatteContext.person) and variant the boil picture: the new extras
    (hearts, crown, confetti and the rest, see styles.extras) keep off the person and the confetti
    follows the boil. Without a person those extras that need one are left out."""
    notes.draw_arrows(img, layout, recipe, visible, scale)
    notes.draw_notes(img, layout, copy, recipe, visible, scale)
    notes.draw_hero(img, layout, copy, recipe, visible, scale)
    notes.draw_extras(img, layout, recipe, visible, face, keepout, scale)
    extras.draw(img, layout, recipe, copy, visible, face, keepout, scale, person, variant)
