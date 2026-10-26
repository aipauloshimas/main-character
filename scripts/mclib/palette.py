"""Colors: the contrast guard, and a palette taken from an image.

The guard measures how well a style's lettering reads on what it sits on, as the WCAG contrast ratio
(text_contrast), and project.validate warns when it is under config.CONTRAST_MIN. palette_from_image
takes the five colors of a style (dark, light, tex, edge and ink) from a picture, such as the
person's logo or a screenshot of their feed.

This module imports raster and the background renderer (for the tone the pixel notes sit on), and
nothing that imports it back (not the project file, the notes styles or the card), because the
project file and the notes styles both import it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from .raster import hex_rgb, luminance, mix
from .styles import background

# --- contrast ---------------------------------------------------------------------------------

# What the lettering of a notes style sits on, besides the palette (the same values the notes
# styles draw with, in styles/notes.py): a comic box is the light color mixed this far toward
# white, with lettering of this color, and a handwritten strip is paper of this color.
_COMIC_TEXT, _COMIC_FILL = "#111111", 0.45
_HAND_PAPER = "#fffdf5"


def _linear(value: float) -> float:
    """One sRGB channel (0-1) as linear light."""
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str) -> float:
    """The WCAG relative luminance of a "#rrggbb" color: 0 for black, 1 for white."""
    red, green, blue = (_linear(float(value)) for value in hex_rgb(color))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(a: str, b: str) -> float:
    """The WCAG contrast ratio of two "#rrggbb" colors, whichever is lighter: 1 for the same color,
    21 for black and white."""
    high, low = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def text_contrast(recipe: dict) -> float:
    """The contrast of a recipe's lettering on what it sits on, by its notes style:

    - marker: the ink on the dark color;
    - comic: #111111 on the light color mixed 45% toward white;
    - hand: the ink on the paper, #fffdf5;
    - label: the dark color on the edge color;
    - pixel: the ink on the tone of the recipe's background (background.tone): the notes have no
      fill and no glow, so they sit on whatever background the recipe picks.

    The marker is measured on the dark color because its two glows and its line under the lettering
    are drawn in that color (notes._marker), so it is what the marker sits on. A notes id that is
    none of these is a ValueError.
    """
    pal, notes = recipe["palette"], recipe["notes"]
    if notes == "marker":
        pair = pal["ink"], pal["dark"]
    elif notes == "comic":
        pair = _COMIC_TEXT, mix(pal["light"], "#ffffff", _COMIC_FILL)
    elif notes == "hand":
        pair = pal["ink"], _HAND_PAPER
    elif notes == "label":
        pair = pal["dark"], pal["edge"]
    elif notes == "pixel":
        pair = pal["ink"], background.tone(recipe["bg"], pal)
    else:
        raise ValueError(f"There is no contrast rule for the notes style {notes!r}.")
    return contrast(*pair)


def floor_contrast(ratio: float) -> str:
    """A contrast ratio as the sentences about it write it: cut to one decimal, never rounded. The guard
    turns down anything under config.CONTRAST_MIN, so a 2.95 has to read "2.9" and not "3.0", which is a
    figure that passes. (The studio's page cuts it the same way.)"""
    return f"{math.floor(ratio * 10) / 10:.1f}"


# --- a palette from an image ------------------------------------------------------------------

_SAMPLE = 64  # the picture is shrunk to this many pixels across and down
# A picture of more than _BIG pixels is shrunk to about _PRESHRINK on its long side before anything
# else is done with it. A photo of 100 megapixels is 400 MB once it is converted to RGBA, and the
# server reads whatever the person picks.
_BIG, _PRESHRINK = 4_000_000, 256
_MIN_ALPHA = 128  # pixels with less alpha than this are left out
# k-means with 5 clusters, one for each color of a palette. The clusters start on the pixels at
# these quantiles of the luminance (darkest first) and run for 10 rounds.
_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.99)
_ROUNDS = 10
# The ink is the dark or the light color when they are further apart than this contrast, and white
# when they are not. It is the dark one when the middle of the two is brighter than this (the
# relative luminance of mix(dark, light, 0.5)), else the light one.
_INK_CONTRAST = 4.0
_BRIGHT_MIDDLE = 0.35


class NoVisiblePixels(ValueError):
    """The image has nothing to take colors from: every pixel is see-through."""


def palette_from_image(path: Path | BinaryIO) -> dict[str, str]:
    """The five colors of a style, as {role: "#rrggbb"} in lowercase, taken from an image: a file's
    path, or an open binary file.

    The picture is shrunk to 64 x 64 and its see-through pixels (alpha under 128) are left out. K-means
    with 5 clusters finds its main colors, and the roles are given by the colors themselves:

    - dark and light are the darkest and the lightest, by relative luminance;
    - of the other three, tex is the most saturated (largest max - min of its channels) and edge the
      second;
    - ink is the dark or the light color when those two are further apart than 4:1 (the dark one when
      the middle of the two is bright, else the light one), and #ffffff when they are not.

    A picture of one color gives that color for four roles and white ink. NoVisiblePixels, a
    ValueError, when the image has no visible pixels; an image that cannot be read raises what Pillow
    raises. A big picture is shrunk first (see _visible_pixels).
    """
    colors = [_to_hex(center) for center in _kmeans(_visible_pixels(path))]
    colors.sort(key=relative_luminance)  # a stable sort: colors of the same luminance keep their order
    dark, light = colors[0], colors[-1]
    tex, edge = sorted(colors[1:-1], key=lambda color: -_saturation(color))[:2]
    return {"dark": dark, "light": light, "tex": tex, "edge": edge, "ink": _ink(dark, light)}


def _visible_pixels(path: Path | BinaryIO) -> np.ndarray:
    """The pixels of a 64 x 64 copy of the image as an (n, 3) float array of 0-255 values, in the
    order of the picture (row by row), without those with an alpha under 128.

    A picture of over 4 million pixels is first shrunk to about 256 on its long side (a JPEG is then
    decoded at a fraction of its size), so that it is never converted to RGBA at full size. Smaller
    pictures go to 64 x 64 in one step, as they always did.
    """
    with Image.open(path) as source:
        if source.width * source.height > _BIG:
            source.thumbnail((_PRESHRINK, _PRESHRINK), Image.Resampling.BOX)
        # BOX averages every pixel a small one stands for, so thin lines still count and no ringing
        # makes colors that are not in the picture. Pillow weights the colors by alpha as it shrinks.
        copy = source.convert("RGBA").resize((_SAMPLE, _SAMPLE), Image.Resampling.BOX)
    pixels = np.asarray(copy, np.float64).reshape(-1, 4)
    visible = pixels[pixels[:, 3] >= _MIN_ALPHA, :3]
    if not len(visible):
        raise NoVisiblePixels("The image has no visible pixels to take colors from.")
    return visible


def _kmeans(pixels: np.ndarray) -> np.ndarray:
    """The centers (5 x 3, 0-255) of 10 rounds of k-means over (n, 3) pixels.

    The start is fixed: the pixels that sit at the 0, 0.25, 0.5, 0.75 and 0.99 quantiles when they are
    sorted by luminance (Rec. 709 luma, as raster.luminance), where pixels of the same luminance keep
    the order of the picture. So the same picture always gives the same colors. A center that no
    pixel is nearest to stays where it is, and a pixel that two centers are equally near goes to the
    first.
    """
    by_light = np.argsort(luminance(pixels / 255), kind="stable")
    last = len(pixels) - 1
    centers = pixels[by_light[[int(share * last) for share in _QUANTILES]]]
    for _ in range(_ROUNDS):
        distance = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        nearest = distance.argmin(axis=1)
        for number in range(len(centers)):
            members = pixels[nearest == number]
            if len(members):
                centers[number] = members.mean(axis=0)
    return centers


def _to_hex(center) -> str:
    """A color of three 0-255 values as "#rrggbb", rounded to the nearest level (halves up)."""
    red, green, blue = (min(max(int(value + 0.5), 0), 255) for value in center)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _saturation(color: str) -> float:
    """How far a color is from gray, as the widest gap between its channels (0-1)."""
    channels = hex_rgb(color)
    return float(channels.max() - channels.min())


def _ink(dark: str, light: str) -> str:
    """The lettering color for a dark and a light color: white unless they are further apart than
    4:1, then the dark color over a bright middle and the light color over a dim one."""
    if contrast(light, dark) <= _INK_CONTRAST:
        return "#ffffff"
    return dark if relative_luminance(mix(dark, light, 0.5)) > _BRIGHT_MIDDLE else light
