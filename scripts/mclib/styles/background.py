"""Backgrounds: what the room of the frozen frame turns into.

Each one draws from the whole frame, the person included (the person is pasted on top later), and
returns a float picture of the stage's size. Anything that moves from one boil variant to the next
is seeded by the variant, so a variant is the same picture every time it is drawn. The picture is
not clipped here: the grain can push a value past 0 or 1, and the engine clips once, at the end.
tone() gives the average color of a background, for the contrast guard (palette.text_contrast).

The numbers below are in px of the plan, 1080 wide. A background drawn bigger (a render at the
clip's own size) is given a raster.Scale: every size in px is multiplied by its k, a lattice that
lives on whole pixels (the scanlines, the shift of the boil) is scaled and rounded, the film grain is
drawn at the plan and enlarged, and the paper's blotches are the plan's, enlarged. At the plan
(IDENTITY) nothing changes, to the bit.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..config import plan_size
from ..raster import IDENTITY, Scale, grain, halftone, hex_rgb, lowfreq, luminance, mix, over
from .edges import shifted
from .recipes import unavailable

# dots: the frame's luminance, stretched (black at 0.04, white at 0.84) and bent, picks a point
# on the line from the dark color to the light one, and halftone dots of the texture color go
# over that, bigger where the room is dark.
_DOTS_FLOOR, _DOTS_RANGE, _DOTS_GAMMA = 0.04, 0.8, 0.8
_DOTS_LIFT, _DOTS_SPAN = 0.3, 0.7  # the line is walked from 0.3 to 1.0, so the room is never all dark
_DOTS_SPACING, _DOTS_ANGLE, _DOTS_GAIN = 9, 45, 0.9  # px between dots, their angle and their size
_DOTS_STRENGTH = 0.55  # how much of the texture color a full dot lays down
_DOTS_GRAIN = 7  # levels of film grain on the background


def _dots(rgb, alpha, pal, variant, head, scale=IDENTITY) -> np.ndarray:
    t = np.clip((luminance(rgb) - _DOTS_FLOOR) / _DOTS_RANGE, 0.0, 1.0) ** _DOTS_GAMMA
    dark, light = hex_rgb(pal["dark"]), hex_rgb(pal["light"])
    base = dark + (light - dark) * (_DOTS_LIFT + _DOTS_SPAN * t)[..., None]
    # The lattice of dots slides by one pixel a variant, which is what makes the card boil
    shift = scale.phase(variant)
    dots = halftone(1 - t, scale.px(_DOTS_SPACING), _DOTS_ANGLE, _DOTS_GAIN, phase=(shift, shift))
    return grain(
        over(base, hex_rgb(pal["tex"]), _DOTS_STRENGTH * dots), _DOTS_GRAIN, variant,
        scale.noise_shape(base.shape),
    )


def _center(rgb: np.ndarray, head) -> tuple[float, float]:
    """The point a background is lit or radiates from: the middle of the head's box, or the middle
    of the stage when nobody is in the picture."""
    if head is None:
        return rgb.shape[1] / 2, rgb.shape[0] / 2
    return (head[0] + head[2]) / 2, (head[1] + head[3]) / 2


# comic: 12 degree wedges around the head, the light color and the light color 40% toward white in
# turns, and red Ben-Day dots over them, bigger where the room is dark.
_COMIC_WEDGE = 12  # degrees
_COMIC_PALER = 0.4  # how far the second wedge color is mixed toward white
_COMIC_DOTS = (16, 0, 0.75)  # px between dots, their angle and their size
_COMIC_STRENGTH = 0.85  # how much of the texture color a full dot lays down


def _comic(rgb, alpha, pal, variant, head, scale=IDENTITY) -> np.ndarray:
    height, width = rgb.shape[:2]
    cx, cy = _center(rgb, head)
    rows = np.arange(height, dtype=np.float32)[:, None] - cy
    columns = np.arange(width, dtype=np.float32) - cx
    angle = np.degrees(np.arctan2(rows, columns))
    wedge = (np.floor((angle + 180) / _COMIC_WEDGE) % 2).astype(np.float32)
    light = hex_rgb(pal["light"])
    paler = hex_rgb(mix(pal["light"], "#ffffff", _COMIC_PALER))
    base = light + (paler - light) * wedge[..., None]
    spacing, dots_angle, gain = _COMIC_DOTS
    darkness = np.clip(1 - 1.2 * luminance(rgb), 0.0, 1.0)
    shift = scale.phase(variant)
    dots = halftone(darkness, scale.px(spacing), dots_angle, gain, phase=(shift, shift))
    return over(base, hex_rgb(pal["tex"]), _COMIC_STRENGTH * dots)


# riso: a sheet of paper (the light color, blotchy and grainy) printed in one ink, as halftone dots
# that are bigger where the room is dark, and a second plate of pink printed a little out of place.
_RISO_PAPER = (60, 0.035, 0.012 * 255)  # the blotches' size in px, their strength, the grain's levels
_RISO_DOTS = (6, 30)  # px between dots and their angle
_RISO_STRENGTH = 0.88  # how much of the ink a full dot lays down
_RISO_PLATE = ("#ff4f8b", 0.22, (6, 4))  # the second plate's color, strength and offset (right, down)


def _riso(rgb, alpha, pal, variant, head, scale=IDENTITY) -> np.ndarray:
    height, width = rgb.shape[:2]
    size, amount, noise = _RISO_PAPER
    blotches = lowfreq((height, width), scale.px(size), amount, variant, scale.coarse(size))
    paper = hex_rgb(pal["light"]) + blotches[..., None]
    paper = grain(paper, noise, variant, scale.noise_shape(paper.shape))
    ink = np.clip(1.1 * (1 - luminance(rgb)) - 0.08, 0.0, 1.0)
    spacing, angle = _RISO_DOTS
    shift = scale.phase(variant)
    dots = halftone(ink, scale.px(spacing), angle, phase=(shift, shift))
    img = over(paper, hex_rgb(pal["tex"]), _RISO_STRENGTH * dots)
    color, strength, (right, down) = _RISO_PLATE
    return over(
        img, hex_rgb(color), strength * shifted(ink, scale.n(right), scale.n(down)) * (1 - ink)
    )


# night: the frame in the dark and light colors, dimmed and lit by a glow of the texture color
# around the head, under scanlines.
_NIGHT_LIFT, _NIGHT_SPAN, _NIGHT_GAIN, _NIGHT_DIM = 0.15, 0.6, 1.3, 0.6  # see _night
_NIGHT_GLOW = (0.4, 420, 300)  # the texture color's strength at the head, and the glow's half sizes
_NIGHT_SCAN = (4, 2, 0.72)  # rows: a line every 4, from its third row, the light that is left
_NIGHT_GRAIN = 3  # levels of film grain


def _night(rgb, alpha, pal, variant, head, scale=IDENTITY) -> np.ndarray:
    height, width = rgb.shape[:2]
    dark, light = hex_rgb(pal["dark"]), hex_rgb(pal["light"])
    lit = _NIGHT_LIFT + _NIGHT_SPAN * np.clip(_NIGHT_GAIN * luminance(rgb), 0.0, 1.0)
    img = (dark + (light - dark) * lit[..., None]) * _NIGHT_DIM
    strength, half_w, half_h = _NIGHT_GLOW
    half_w, half_h = scale.px(half_w), scale.px(half_h)
    cx, cy = _center(rgb, head)
    across = ((np.arange(width, dtype=np.float32) - cx) / half_w) ** 2
    down = ((np.arange(height, dtype=np.float32) - cy) / half_h) ** 2
    img = img + hex_rgb(pal["tex"]) * (strength * np.exp(-(down[:, None] + across[None, :])))[..., None]
    every, first, kept = _NIGHT_SCAN
    if not scale.identity:  # whole rows: a line every `every` of them, from the middle of the period
        every = scale.n(every, minimum=2)
        first = round(every / 2)
    scan = np.where(np.arange(height) % every >= first, kept, 1.0).astype(np.float32)
    shaded = img * scan[:, None, None]
    return grain(shaded, _NIGHT_GRAIN, variant, scale.noise_shape(shaded.shape))


# flat: one color, lit from the head's center toward the light color, and a little grain.
_FLAT_CENTER, _FLAT_REACH, _FLAT_GRAIN = 0.2, 0.7, 4  # how far toward white; of the diagonal; levels


def _flat(rgb, alpha, pal, variant, head, scale=IDENTITY) -> np.ndarray:
    height, width = rgb.shape[:2]
    cx, cy = _center(rgb, head)
    across = np.arange(width, dtype=np.float32)[None, :] - cx
    down = np.arange(height, dtype=np.float32)[:, None] - cy
    away = np.hypot(across, down)
    share = np.clip(away / (_FLAT_REACH * math.hypot(width, height)), 0.0, 1.0)
    light = hex_rgb(pal["light"])
    middle = hex_rgb(mix(pal["light"], "#ffffff", _FLAT_CENTER))
    lit = middle + (light - middle) * share[..., None]
    return grain(lit, _FLAT_GRAIN, variant, scale.noise_shape(lit.shape))


# mono: the frame in black and white with more contrast, darker toward the corners.
_MONO_CONTRAST, _MONO_VIGNETTE, _MONO_FROM, _MONO_GRAIN = 1.25, 0.45, 0.55, 10  # see _mono


def _mono(rgb, alpha, pal, variant, head, scale=IDENTITY) -> np.ndarray:
    height, width = rgb.shape[:2]
    gray = np.clip((luminance(rgb) - 0.5) * _MONO_CONTRAST + 0.5, 0.0, 1.0)
    across = (np.arange(width, dtype=np.float32) - width / 2) / (width / 2)
    down = (np.arange(height, dtype=np.float32) - height / 2) / (height / 2)
    # 0 at the middle and 1 at the corners: the vignette starts 0.55 of the way out and takes 45% off
    # at the corners
    away = np.hypot(across[None, :], down[:, None]) / math.sqrt(2)
    shade = 1 - _MONO_VIGNETTE * np.clip((away - _MONO_FROM) / (1 - _MONO_FROM), 0.0, 1.0)
    shaded = np.repeat((gray * shade)[..., None], 3, axis=-1)
    return grain(shaded, _MONO_GRAIN, variant, scale.noise_shape(shaded.shape))


# One drawing function for each background that is built. They all take what render takes.
_KINDS = {"dots": _dots, "comic": _comic, "riso": _riso, "night": _night, "flat": _flat, "mono": _mono}


def render(
    kind: str,
    rgb: np.ndarray,
    alpha: np.ndarray,
    pal: dict,
    variant: int,
    head: tuple[int, int, int, int] | None,
    scale: Scale = IDENTITY,
) -> np.ndarray:
    """The background of a card, height x width x 3, floats around 0-1 (not clipped).

    kind is a recipe's bg id, rgb the frozen frame (0-1) and alpha its matte (a background may
    use it to find the room), pal the palette as {role: "#rrggbb"}, variant the boil picture
    (0, 1 or 2) and head the head's box (x0, y0, x1, y1), or None when nobody is in the picture,
    for a background that is lit around the head. scale is how much bigger than the plan the
    picture is (rgb and alpha are that size, and so is head). An id that is no background is a
    ValueError.
    """
    try:
        draw = _KINDS[kind]
    except KeyError:
        raise unavailable("bg", kind) from None
    return draw(rgb, alpha, pal, variant, head, scale)


# The tone is taken over a picture of the plan of a 16:9 clip (1080 x 608), not a small one: the night
# glow is 420 x 300 px, so on a picture 64 px across it would be at full strength everywhere, and the
# tone would come out lighter than the room the notes sit in (Player Select in Riso's palette: 3.4:1
# for its ink at 64 x 64, 2.7:1 on the stage).
_TONE_SIZE = plan_size(1920, 1080)  # (width, height)
_TONE_PHOTO = 0.5  # the luminance of the plain gray photo the background is drawn from


def tone(kind: str, pal: dict) -> str:
    """The average color of a background, as lowercase "#rrggbb": what a note that has no fill or
    glow of its own sits on. The background `kind` is drawn from the palette over a plain gray photo
    of luminance 0.5, as big as the plan of a 16:9 clip, with nobody in it (no head) and in the first
    boil variant, and the picture is clipped as the engine does and averaged. An id that is no
    background is a ValueError. The result is kept, so a tone is drawn once for each palette."""
    return _tone(kind, tuple(sorted(pal.items())))


@lru_cache(maxsize=256)
def _tone(kind: str, colors: tuple[tuple[str, str], ...]) -> str:
    width, height = _TONE_SIZE
    photo = np.full((height, width, 3), _TONE_PHOTO, np.float32)
    nobody = np.zeros((height, width), np.float32)
    picture = np.clip(render(kind, photo, nobody, dict(colors), 0, None), 0.0, 1.0)
    red, green, blue = (int(round(value)) for value in picture.reshape(-1, 3).mean(axis=0) * 255)
    return f"#{red:02x}{green:02x}{blue:02x}"
