"""Cutout edges: the sticker border around the person.

An edge is a list of layers, laid over the background in order and under the person. Each layer is
(color, mask, blend): a color of three floats in 0-1, a mask the size of the stage (0-1, 1 where the
layer is solid), and how it is laid on, "normal" for over and "screen" for a glow that only
lightens. A layer that moves from one boil variant to the next is seeded by the variant.

The sizes below are in px of the plan, 1080 wide. An edge drawn bigger (a render at the clip's own
size) is given a raster.Scale, and its radii, wobble, blurs and shifts are multiplied by the scale's
k (the shifts on whole pixels). The wobble is the plan's, enlarged. The 0.75 px ramp of dilate is the
antialiasing of the picture's own pixels, so it stays what it is.
"""

from __future__ import annotations

import numpy as np

from ..raster import IDENTITY, Scale, blur, dilate, hex_rgb
from .recipes import unavailable

_HANDCUT_RADIUS = 22  # px the sticker reaches out from the person
_HANDCUT_WOBBLE = 7  # px the edge wanders in or out of that, as if cut by hand
_COMIC_OUTER, _COMIC_INNER = 19, 12  # px: the black line reaches out 19 and the white one 12
_PAPER_RADIUS, _PAPER_WOBBLE, _PAPER_SCALE = 16, 3, 25  # px out, the wobble, and its size across
_PAPER_SHADOW = ("#2a1d10", 0.40, 9, (7, 10))  # its color, strength, blur and offset (right, down)
_NEON_HALO, _NEON_BLUR, _NEON_STRENGTH, _NEON_LINE = 10, 14, 0.95, 6  # px, px, share, px
_THIN_LINE = 5
_NONE_SHADOW = (0.35, 10, 8)  # a shadow with no edge: strength, blur and how far down

Layer = tuple[np.ndarray, np.ndarray, str]

# How far each edge reaches out from the person in px (its first radius, not counting a wobble or
# a glow). The notes keep off that much of the room around the person, and their arrows end just
# outside it, so it is every edge's, whichever notes style is drawn with it.
_REACH = {
    "handcut": _HANDCUT_RADIUS,
    "comic": _COMIC_OUTER,
    "paper": _PAPER_RADIUS,
    "neon": _NEON_HALO,
    "thin": _THIN_LINE,
    "none": 0,
}


def reach(kind: str) -> int:
    """How far an edge reaches out from the person, in px (see _REACH). An id that is no edge at
    all is a ValueError."""
    try:
        return _REACH[kind]
    except KeyError:
        raise unavailable("edge", kind) from None


def shifted(mask: np.ndarray, right: int, down: int) -> np.ndarray:
    """A mask moved `right` and `down` pixels (not negative), what comes in from beyond the edge
    of the picture being the edge itself, repeated. Shadows and misprinted plates use it."""
    height, width = mask.shape
    return np.pad(mask, ((down, 0), (right, 0)), mode="edge")[:height, :width]


_WOBBLE_FEATURES = 40  # px across: the size of the hand's wobbles (raster.dilate's default)


def _handcut(alpha: np.ndarray, pal: dict, variant: int, scale: Scale = IDENTITY) -> list[Layer]:
    """One layer: the edge color over the person grown by 22 px, with an edge that wobbles by up
    to 7 px. Each variant wobbles differently, so the cut seems to move as the card boils."""
    mask = dilate(
        alpha, scale.px(_HANDCUT_RADIUS), wobble=scale.px(_HANDCUT_WOBBLE), seed=variant,
        scale=scale.px(_WOBBLE_FEATURES), grid=scale.coarse(_WOBBLE_FEATURES),
    )
    return [(hex_rgb(pal["edge"]), mask, "normal")]


def _comic(alpha: np.ndarray, pal: dict, variant: int, scale: Scale = IDENTITY) -> list[Layer]:
    """Two straight lines, as a printed comic cuts out a figure: the dark color over the person
    grown by 19 px, and the edge color over the person grown by 12, which leaves a dark line 7 px
    wide around a white one."""
    return [
        (hex_rgb(pal["dark"]), dilate(alpha, scale.px(_COMIC_OUTER)), "normal"),
        (hex_rgb(pal["edge"]), dilate(alpha, scale.px(_COMIC_INNER)), "normal"),
    ]


def _paper(alpha: np.ndarray, pal: dict, variant: int, scale: Scale = IDENTITY) -> list[Layer]:
    """A paper cutout: the edge color over the person grown by 16 px, with an edge that wobbles by
    up to 3 px, and under it a shadow, lower and to the right, that is the cutout's shape blurred
    and moved 7 px right and 10 down."""
    mask = dilate(
        alpha, scale.px(_PAPER_RADIUS), wobble=scale.px(_PAPER_WOBBLE), seed=variant,
        scale=scale.px(_PAPER_SCALE), grid=scale.coarse(_PAPER_SCALE),
    )
    color, strength, blurred, (right, down) = _PAPER_SHADOW
    shadow = shifted(blur(mask, scale.px(blurred)), scale.n(right), scale.n(down))
    return [(hex_rgb(color), strength * shadow, "normal"), (hex_rgb(pal["edge"]), mask, "normal")]


def _neon(alpha: np.ndarray, pal: dict, variant: int, scale: Scale = IDENTITY) -> list[Layer]:
    """A glowing line: a halo of the texture color (the person grown by 10 px and blurred by 14,
    screened on at 0.95, so it only lights the room) and over it the edge color grown by 6 px."""
    halo = _NEON_STRENGTH * blur(dilate(alpha, scale.px(_NEON_HALO)), scale.px(_NEON_BLUR))
    return [
        (hex_rgb(pal["tex"]), halo, "screen"),
        (hex_rgb(pal["edge"]), dilate(alpha, scale.px(_NEON_LINE)), "normal"),
    ]


def _thin(alpha: np.ndarray, pal: dict, variant: int, scale: Scale = IDENTITY) -> list[Layer]:
    """A thin line: the edge color over the person grown by 5 px."""
    return [(hex_rgb(pal["edge"]), dilate(alpha, scale.px(_THIN_LINE)), "normal")]


def _no_edge(alpha: np.ndarray, pal: dict, variant: int, scale: Scale = IDENTITY) -> list[Layer]:
    """No edge, only the person's soft shadow: black at 0.35 where the person, blurred by 10 and
    moved 8 px down, is."""
    strength, blurred, down = _NONE_SHADOW
    shadow = strength * shifted(blur(alpha, scale.px(blurred)), 0, scale.n(down))
    return [(hex_rgb("#000000"), shadow, "normal")]


# One drawing function for each edge that is built. They all take what render takes, less the kind.
_KINDS = {
    "handcut": _handcut, "comic": _comic, "paper": _paper, "neon": _neon, "thin": _thin, "none": _no_edge,
}


def render(
    kind: str, alpha: np.ndarray, pal: dict, variant: int, scale: Scale = IDENTITY
) -> list[Layer]:
    """The layers of an edge, from the bottom up, as (rgb, a, blend) with blend "normal" or
    "screen". kind is a recipe's edge id, alpha the cleaned matte (height x width, 0-1), pal the
    palette as {role: "#rrggbb"}, variant the boil picture (0, 1 or 2) and scale how much bigger
    than the plan the matte is. An id that is no edge is a ValueError."""
    try:
        draw = _KINDS[kind]
    except KeyError:
        raise unavailable("edge", kind) from None
    return draw(alpha, pal, variant, scale)
