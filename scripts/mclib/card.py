"""The card: the picture of the freeze. It is the stage (the frame turned into the style's picture) with
the notes, the arrows, the hero and the extras over it. The studio previews it, and the render encodes it.

The card is the clip's own frame (Amendment 1): there is nothing around it, no bars, no gradient and no
title. It is drawn at the plan, the size the project's frames and mattes are cut at (project["work"],
1080 px wide, in the clip's shape), or at any other size of that shape, which is what a final render
does at the clip's own size: every size in px of the style is multiplied by size width / plan width
(raster.Scale), and the layout of the notes, which is made at the plan whatever the size, from the plan's
matte, is scaled with it.

A 9:16 card keeps its notes, their arrows, its hero's boxes, its tape and its stars out of the title
room at the top, where the person adds the title in the app, and out of the Reels safe area's two boxes,
the bottom band and the right rail, where the app draws its caption and its buttons. _keepout is where
all of those boxes come from. A card of any other shape has none.

A card is drawn from a project folder, which holds the frames (frames/000060.jpg) and the mattes
(matte/000060.png) of the clip, and the project dict. The matte and the stage are costly to make
(a matte is cleaned up, a stage is composed from a few dilations and textures) and a render draws
about a dozen cards that differ only in which notes are on, so both are kept in small caches, and
so are the layout of the notes and the matte's context for it. Everything cached is read-only.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from . import matte
from .analyze import person_boxes
from .config import BOTTOM_BAND, RAIL_LEFT, RAIL_TOP, TALL_PLAN, TITLE_ROOM
from .layout import MatteContext, NoteBox
from .raster import IDENTITY, Scale, from_image, to_image
from .styles import engine, notes
from .styles.recipes import validate_recipe

# The groups of a card that can be on: the bottom notes, the top notes and the arrows. The timeline
# says which are visible on each card frame.
ALL_VISIBLE = frozenset({"bottom", "top", "arrows"})

_SHAPE_TOLERANCE = 0.03  # how far a size's shape may be from the plan's: rounding to even sizes, no more


# --- caches -----------------------------------------------------------------------------------


class _Cache:
    """A small least-recently-used cache that is safe to use from the studio's threads. Two threads
    that miss on the same key both make the value, and the last one stays."""

    def __init__(self, size: int = 8):
        self._size = size
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, make):
        """The value kept under key, or make() kept under it."""
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        value = make()
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_MATTES = _Cache()  # the cleaned matte of a frame, and its head box, at the plan and at the sizes drawn
_CONTEXTS = _Cache()  # what the layout works out from a matte alone, for one notes style
_LAYOUTS = _Cache()  # the boxes of the notes, arrows and hero, in px of the plan
_STAGES = _Cache()  # the stage as a picture, one for each boil variant and each size


def clear_caches() -> None:
    """Forget everything kept. Nothing needs this to stay right, since a changed file or setting
    is a new key; it is for freeing memory, and for tests that count the work."""
    for cache in (_MATTES, _CONTEXTS, _LAYOUTS, _STAGES):
        cache.clear()


def _stamp(path: Path) -> tuple[int, int]:
    """When a file was last written and how big it is: a change to either is a new picture."""
    info = path.stat()
    return info.st_mtime_ns, info.st_size


def _canonical(value) -> str:
    """JSON text of a recipe, the copy or the positions, the same for equal values, to key on."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# --- the plan, and what a card is drawn from ----------------------------------------------------


class FullFreeze(NamedTuple):
    """The freeze frame and its matte as they were cut from the clip at the size of a final render
    (render.py makes them), for a card to be drawn from at that size, instead of from the plan's own
    pictures enlarged. Both are files of exactly that size."""

    frame: Path
    matte: Path


def _plan_of(project: dict) -> tuple[int, int]:
    """The plan: the (width, height) the project's frames and mattes are cut at, and the layout is
    made at."""
    work = project["work"]
    return work["width"], work["height"]


def _scale_of(project: dict, size: tuple[int, int] | None) -> Scale:
    """How much bigger than the plan a picture of `size` is: IDENTITY for none, or for the plan's own
    size. ValueError for another shape than the plan's, but for the pixel that rounding to an even
    size costs."""
    plan = _plan_of(project)
    scale = Scale.between(plan, size)
    if not scale.identity and abs(scale.ky / scale.k - 1) > _SHAPE_TOLERANCE:
        raise ValueError(
            f"A card of {size[0]}x{size[1]} is not the shape of the project's {plan[0]}x{plan[1]}."
        )
    return scale


def _check_size(size: tuple[int, int], wanted: tuple[int, int], what: str) -> None:
    """Raise ValueError unless a picture of `size` (width, height) is the size the project is cut at.
    Frames are cut to the plan's size, so another size means the clip or its shape was changed after
    the frames were made."""
    if tuple(size) != tuple(wanted):
        raise ValueError(
            f"{what} is {size[0]}x{size[1]}, but the project is cut at {wanted[0]}x{wanted[1]}. "
            "Cut the frames and the matte again (run init on the clip)."
        )


def _check_source(source: FullFreeze, size: tuple[int, int]) -> None:
    """Raise ValueError, with a sentence, unless the frame and the matte of `source` are exactly `size`
    (width, height), the size of the card that is drawn from them. A picture of another size was not
    cut from the clip at the card's size, and enlarging or shrinking it would hide that. Both are looked
    at before any cache is, so that a wrong one is never let through by a picture that was kept."""
    for what, path in (("frame", source.frame), ("matte", source.matte)):
        with Image.open(path) as image:
            if image.size != tuple(size):
                raise ValueError(
                    f"The {what} is {image.size[0]}x{image.size[1]}, but the card is {size[0]}x{size[1]}."
                )


def _frame_path(folder: Path, frame: int) -> Path:
    return folder / "frames" / f"{frame:06d}.jpg"


class _Matte(NamedTuple):
    """A cleaned matte, as the caches keep it. key says which matte it is, and the other caches
    are keyed on it too, so that a new matte or a new edge setting starts them over."""

    key: tuple
    alpha: np.ndarray
    head: tuple[int, int, int, int] | None


def _matte(
    folder: Path, project: dict, size: tuple[int, int] | None = None, source: FullFreeze | None = None
) -> _Matte:
    """The freeze frame's matte, cleaned and with the project's edge setting applied, from the cache
    when the matte file, the setting and the size are the same as last time.

    size is None for the plan, where the file has to be the plan's size. Otherwise it is the size of a
    card drawn bigger than the plan: the matte is the one cut at that size (`source`'s), or else the
    project's own, enlarged, and its cleanup and its edge are those of that scale (matte.load_alpha).
    """
    frame = project["freeze"]["frame"]
    edge = project["matte"].get("edge", 0)
    path = matte.matte_path(folder, frame) if source is None else source.matte
    key = (str(path), frame, edge, size, _stamp(path))

    def make() -> _Matte:
        alpha = matte.load_alpha(folder, project, frame, size=size, path=None if source is None else path)
        alpha.setflags(write=False)
        return _Matte(key, alpha, person_boxes(alpha)["head"])

    found = _MATTES.get(key, make)
    _check_size(found.alpha.shape[::-1], size or _plan_of(project), "The matte")
    return found


def _keepout(project: dict) -> tuple[tuple[int, int, int, int], ...]:
    """Every box the notes, their arrows, the hero's boxes, the tape and the stars must keep out of,
    in px of the plan, and the one place they come from.

    A tall plan (9:16: at least 1.7 times as tall as it is wide) has three, shares of the plan from
    config: the title room across the top, where the person adds the title in the app; the bottom
    band, where the app draws the caption, the username and the audio line; and the rail of buttons at
    the right, from the middle of the height down to the bottom band. A note never sits where the
    title will be or where the app's own buttons and caption will be drawn over it. A plan of any
    other shape has none."""
    width, height = _plan_of(project)
    if height / width < TALL_PLAN:
        return ()
    floor = round(BOTTOM_BAND * height)
    return (
        (0, 0, width, round(TITLE_ROOM * height)),
        (0, floor, width, height),
        (round(RAIL_LEFT * width), round(RAIL_TOP * height), width, floor),
    )


def _context(found: _Matte, recipe: dict, keepout: tuple) -> MatteContext:
    """The layout's context for this matte, this recipe's notes style and these keep-out boxes."""
    key = (found.key, notes.note_style(recipe), keepout)
    return _CONTEXTS.get(key, lambda: engine.matte_context(recipe, found.alpha, list(keepout)))


def _boxes(found: _Matte, project: dict, recipe: dict) -> dict[str, NoteBox]:
    """The layout of the recipe's notes and hero on this matte (the plan's: the layout is the same
    whatever size the card is drawn at), kept while the recipe, the copy, the pinned positions and
    the keep-out boxes are the same. Cached boxes are shared, so callers must not change them."""
    copy, positions, keepout = project["copy"], project.get("positions"), _keepout(project)
    key = (
        found.key, tuple(found.alpha.shape), _canonical(recipe), _canonical(copy), _canonical(positions),
        keepout,
    )
    return _LAYOUTS.get(
        key,
        lambda: engine.style_layout(
            recipe, found.alpha, copy, positions, _context(found, recipe, keepout)
        ),
    )


def _stage(
    folder: Path,
    project: dict,
    recipe: dict,
    found: _Matte,
    variant: int,
    size: tuple[int, int] | None = None,
    source: FullFreeze | None = None,
    scale: Scale = IDENTITY,
) -> Image.Image:
    """The composed stage as a picture, kept for the recipe, the variant and the size. The frame and
    the matte are part of what it is made of, so it is keyed on both, and on the size, so that a stage
    drawn at one size is never given for another.

    At the plan (size None) the frame has to be the plan's size. At another size the frame is the one
    cut at that size (`source`'s, which render_card has checked is that size) or else the project's own,
    enlarged (Lanczos), and `found` is the matte for that size."""
    path = _frame_path(folder, project["freeze"]["frame"]) if source is None else source.frame
    key = (found.key, _stamp(path), _canonical(recipe), variant, size)

    def make() -> Image.Image:
        with Image.open(path) as image:
            picture = image.convert("RGB")
        if size is None:
            _check_size(picture.size, _plan_of(project), "The frame")
        elif picture.size != tuple(size):
            picture = picture.resize(tuple(size), Image.Resampling.LANCZOS)
        return to_image(engine.compose_stage(recipe, from_image(picture), found.alpha, variant, scale))

    return _STAGES.get(key, make)


# --- the card -----------------------------------------------------------------------------------


def render_card(
    project_dir: Path,
    project: dict,
    visible=ALL_VISIBLE,
    variant: int = 0,
    flash: float = 0.0,
    style: dict | None = None,
    size: tuple[int, int] | None = None,
    source: FullFreeze | None = None,
) -> Image.Image:
    """The card of the freeze frame, as an RGB picture of `size` (width, height): the stage, which is
    the frame turned into the style's picture, and over it the notes, arrows, hero and extras.

    size is None for the plan (the size of project["work"], 1080 wide, where the layout is made), or a
    picture of the plan's shape and any size, such as a render at the clip's own size. The layout and
    the matte that it is made from are the plan's, so a bigger card has the notes in the places the
    studio showed. The stage, the edge of the cutout and the overlay are drawn at the size asked for:
    every size in px of the style is multiplied by size width / plan width (raster.Scale). The freeze
    frame and its matte are `source`'s when it is given (cut from the clip at that size), and
    otherwise the project's own, enlarged.

    visible is a set of "bottom", "top" and "arrows": the bottom notes, the top notes and the
    arrows that are on. variant is the boil picture, 0, 1 or 2. Last comes a white flash of level
    `flash`, from 0 for none to 1 for all white. style is a recipe to draw instead of
    project["style"], to preview another look. On a 9:16 card the notes are laid out around the title
    room and inside the Reels safe area (_keepout).

    The cleaned matte, its context, the layout and the stage come from caches (see the top of this
    module), so the cards of one render, which differ in `visible` and `flash`, cost only the
    drawing of the notes.
    """
    visible = frozenset(visible)
    if not visible <= ALL_VISIBLE:
        wrong = ", ".join(sorted(visible - ALL_VISIBLE))
        raise ValueError(f"visible can hold bottom, top and arrows, not {wrong}")
    if not 0.0 <= flash <= 1.0:
        raise ValueError(f"flash must be from 0 to 1, got {flash!r}")
    if source is not None and size is None:
        raise ValueError("A card drawn from pictures cut at another size needs that size.")
    if source is not None:
        _check_source(source, size)
    folder = Path(project_dir).resolve()
    recipe = project["style"] if style is None else style
    validate_recipe(recipe)

    found = _matte(folder, project)  # at the plan: the layout is made from it, whatever the size
    boxes = _boxes(found, project, recipe)
    context = _context(found, recipe, _keepout(project))
    scale = _scale_of(project, size)
    if size is None or (scale.identity and source is None):  # the project's own pictures, at the plan
        picture = _stage(folder, project, recipe, found, variant)
    else:
        size = tuple(size)
        shown = _matte(folder, project, size, source)
        picture = _stage(folder, project, recipe, shown, variant, size, source, scale)

    layer = Image.new("RGBA", picture.size, (0, 0, 0, 0))
    engine.draw_overlay(
        layer, recipe, boxes, project["copy"], visible, context.face, context.keepout, scale, context.person,
        variant,
    )
    canvas = picture.convert("RGBA")
    canvas.alpha_composite(layer)
    card = canvas.convert("RGB")
    return Image.blend(card, Image.new("RGB", card.size, "white"), flash) if flash else card


def elements(project_dir: Path, project: dict) -> dict[str, NoteBox]:
    """The boxes of the card's notes, arrows and hero, as {key: NoteBox} in px of the plan (the
    card's own corner is 0, 0), for the studio to hit-test and drag. They are copies, so they can
    be changed. Each keeps the layout's clash flag (a note that is cut, overlaps the face, the
    title room, the safe area or another note, or left its corner), which the studio warns on and the
    card does not draw."""
    folder = Path(project_dir).resolve()
    recipe = project["style"]
    validate_recipe(recipe)
    boxes = _boxes(_matte(folder, project), project, recipe)
    return {key: replace(box, lines=list(box.lines)) for key, box in boxes.items()}
