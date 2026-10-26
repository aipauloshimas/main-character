"""Raster helpers for the look renderers: color math, halftone dots, grain, blur, the sticker-edge
dilation, and antialiased text, curves, arrows and polygons.

A picture is a float32 numpy array in 0-1: height x width for a mask, with a last axis of 3 for
color. Drawing goes straight onto a Pillow RGBA image. Coordinates are in pixels, with (0, 0) at
the top-left corner of the top-left pixel, as in CSS and SVG, so the numbers of the HTML mockups
carry over unchanged. There is no scipy here: the sticker edge estimates its distance field from a
blurred mask, and shapes are drawn supersampled on a small tile around them.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from .config import FONT_FILES, FONTS_DIR

_LANCZOS = Image.Resampling.LANCZOS
_BICUBIC = Image.Resampling.BICUBIC

# --- drawing at another size than the plan ----------------------------------------------------


@dataclass(frozen=True)
class Scale:
    """How big a picture is drawn against the plan it was laid out in (config.plan_size).

    A card is laid out at the plan, 1080 px wide, and everything is tuned there. To draw it bigger
    (a render at the clip's own size) every size in px is multiplied by k, the ratio of the widths,
    and a position is put at x * k and y * ky, with ky the ratio of the heights. The two differ by
    a hair, since a picture's size is rounded to whole even pixels: 608 is not 607.5.

    IDENTITY is the plan itself. Every method then gives back what it was given, not what k = 1.0
    would make of it, so a picture drawn at the plan is what it always was, bit for bit.
    """

    k: float = 1.0
    ky: float = 1.0
    plan: tuple[int, int] | None = None  # (width, height) of the plan, None for the plan itself

    @classmethod
    def between(cls, plan: tuple[int, int], size: tuple[int, int] | None) -> Scale:
        """The scale of a picture of `size` (width, height) against a plan: IDENTITY for none, or
        for the plan's own size."""
        if size is None or tuple(size) == tuple(plan):
            return IDENTITY
        return cls(size[0] / plan[0], size[1] / plan[1], (int(plan[0]), int(plan[1])))

    @property
    def identity(self) -> bool:
        return self.plan is None

    def px(self, value: float) -> float:
        """A size in plan px, as it is drawn."""
        return value if self.identity else value * self.k

    def n(self, value: int, minimum: int = 0) -> int:
        """A size that lives on whole pixels (a lattice: scanlines, the shift of a plate), as it is
        drawn: scaled and rounded, and never under `minimum`."""
        return value if self.identity else max(minimum, round(value * self.k))

    def phase(self, shift: float = 0) -> float:
        """The phase of a halftone lattice (halftone's `phase`, which reads pixel numbers) that has
        slid `shift` px of the plan. A picture k times as big has its pixel centers (k - 1) / 2 of its
        own px off from where the plan's were, and a lattice drawn without that is a quarter of a plan
        px off, which shows in every dot's edge."""
        return shift if self.identity else self.k * shift + (self.k - 1) / 2

    def x(self, value: float) -> float:
        """A position along x, as it is drawn."""
        return value if self.identity else value * self.k

    def y(self, value: float) -> float:
        """A position along y, as it is drawn."""
        return value if self.identity else value * self.ky

    def font(self, size: int, pixel: bool = False) -> int:
        """The size of a font, as it is drawn. A pixel font (Press Start 2P) is only crisp on its 8
        px grid, so it goes to the nearest multiple of 8, and never under 8."""
        if self.identity:
            return size
        if pixel:
            return max(8, round(size * self.k / 8) * 8)
        return max(1, round(size * self.k))

    def coarse(self, features: float) -> tuple[int, int] | None:
        """The grid of random values (lowfreq's `grid`) of a smooth field whose features are
        `features` plan px across, taken at the plan, so that a field drawn at any size is the field
        of the plan, enlarged. None at the plan itself."""
        if self.identity:
            return None
        return int(self.plan[1] // features) + 2, int(self.plan[0] // features) + 2

    def noise_shape(self, shape: tuple[int, ...]) -> tuple[int, int]:
        """The (height, width) film grain is drawn at for a picture of `shape`: the plan's, so that
        the grains are as big, in the picture, as they are at the plan (grain enlarges them)."""
        return tuple(shape[:2]) if self.identity else (self.plan[1], self.plan[0])


IDENTITY = Scale()

# --- color math -------------------------------------------------------------------------------


def _hex_ints(h: str) -> tuple[int, int, int]:
    digits = h.lstrip("#")
    if len(digits) != 6:
        raise ValueError(f"expected a color like #rrggbb, got {h!r}")
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def hex_rgb(h: str) -> np.ndarray:
    """A "#rrggbb" color as three float32 values in 0-1."""
    return np.array(_hex_ints(h), np.float32) / 255


def mix(a: str, b: str, t: float) -> str:
    """The hex color a fraction t of the way from a to b."""
    mixed = (round(x + (y - x) * t) for x, y in zip(_hex_ints(a), _hex_ints(b)))
    return "#%02x%02x%02x" % tuple(min(max(v, 0), 255) for v in mixed)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """The luma of RGB values on the last axis: Rec. 709 weights on the encoded values, as the
    mockup script used."""
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def duotone(
    lum: np.ndarray, dark: str, light: str, lift: float = 0.0, gamma: float = 1.0
) -> np.ndarray:
    """Map luminance (0-1) onto the line from the dark color to the light one.

    Gamma bends the ramp first (below 1 brightens the midtones). Lift then lets black start that
    fraction of the way toward light: the mix is lift + (1 - lift) * lum ** gamma. Returns
    height x width x 3.
    """
    along = lift + (1.0 - lift) * np.clip(lum, 0.0, 1.0) ** gamma
    start = hex_rgb(dark)
    return (start + (hex_rgb(light) - start) * along[..., None]).astype(np.float32)


# --- textures and blur ------------------------------------------------------------------------


def halftone(
    darkness: np.ndarray,
    spacing: float,
    angle: float,
    gain: float = 1.0,
    phase: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Dot coverage (0-1) of an amplitude-modulated halftone.

    The dots sit on a lattice `spacing` px apart, turned by `angle` degrees, and each is as big as
    the darkness under its center asks (0 is light, 1 is dark; gain scales it first). A dot of
    darkness t covers pi / 4 * t of its cell, with a 1 px antialiased edge. phase=(dx, dy) slides
    the whole lattice by that many pixels.
    """
    height, width = darkness.shape
    x = np.arange(width, dtype=np.float32) - np.float32(phase[0])
    y = (np.arange(height, dtype=np.float32) - np.float32(phase[1]))[:, None]
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    u = (x * cos + y * sin) / spacing
    v = (y * cos - x * sin) / spacing
    dist = np.hypot(u - np.round(u), v - np.round(v))  # 0 at a dot center, about 0.7 at a corner
    radius = 0.5 * np.sqrt(np.clip(darkness * gain, 0.0, 1.0))
    return np.clip((radius - dist) * spacing + 0.5, 0.0, 1.0).astype(np.float32)


def grain(img: np.ndarray, sigma: float, seed: int, plan: tuple[int, int] | None = None) -> np.ndarray:
    """Add film grain: Gaussian noise of `sigma` 8-bit levels (so sigma 8 is 8/255), the same on
    every channel. The result is not clipped. The same seed gives the same grain.

    plan is the (height, width) to draw the noise at, for a picture bigger than the plan it was laid
    out in (Scale.noise_shape). The noise is then enlarged to the picture, nearest neighbor, so its
    grains are as big, in the picture, as they are at the plan, and as strong.
    """
    rng = np.random.default_rng(seed)
    shape = img.shape[:2] if plan is None else tuple(plan)
    noise = rng.standard_normal(shape, dtype=np.float32) * np.float32(sigma / 255)
    if shape != img.shape[:2]:
        noise = np.asarray(
            Image.fromarray(noise).resize((img.shape[1], img.shape[0]), Image.Resampling.NEAREST)
        )
    return img + (noise[..., None] if img.ndim == 3 else noise)


def lowfreq(
    shape: tuple[int, int], scale: float, amp: float, seed: int, grid: tuple[int, int] | None = None
) -> np.ndarray:
    """A smooth random field of the given (height, width), in -amp to amp.

    Random values are laid on a grid `scale` px apart and enlarged bicubically, then scaled so the
    largest swing is exactly `amp`. grid is the number of random values (rows, columns) when it must
    not depend on the picture's size (Scale.coarse), so that a picture drawn bigger than its plan has
    the plan's field, enlarged, and not another one.
    """
    height, width = shape
    rng = np.random.default_rng(seed)
    grid = grid or (int(height // scale) + 2, int(width // scale) + 2)
    coarse = rng.standard_normal(grid, dtype=np.float32)
    field = np.asarray(Image.fromarray(coarse).resize((width, height), _BICUBIC))
    return (field / (np.abs(field).max() + 1e-6) * amp).astype(np.float32)


def blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a float array (0-1), through Pillow: height x width, or height x width x 3
    (each channel). Pillow blurs 8-bit images, so the result moves in steps of 1/255. That suits
    glows and shadows; dilate() needs more than that and blurs on its own."""
    return from_image(_quantize(arr).filter(ImageFilter.GaussianBlur(sigma)))


def _smooth(arr: np.ndarray, taps: np.ndarray, axis: int) -> np.ndarray:
    """Convolve a 2-D array along one axis with a symmetric kernel (taps are its center and one
    side), repeating the edge values."""
    reach = len(taps) - 1
    size = arr.shape[axis]
    padding = [(0, 0), (0, 0)]
    padding[axis] = (reach, reach)
    padded = np.pad(arr, padding, mode="edge")
    window = (lambda k: padded[k:k + size]) if axis == 0 else (lambda k: padded[:, k:k + size])

    out = window(reach) * taps[0]
    pair = np.empty_like(out)
    for k in range(1, reach + 1):
        np.add(window(reach - k), window(reach + k), out=pair)
        pair *= taps[k]
        out += pair
    return out


def _gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a 2-D float array in full float32 precision, with the edges repeated.

    Separable, with a kernel out to 4 sigma. dilate() reads the faint tail of a blurred mask, where
    one step of 1/255 is already a step of a pixel in distance, and that left its antialiased edge
    with 4 distinct alpha levels instead of a smooth ramp.
    """
    reach = math.ceil(4 * sigma)
    taps = np.exp(-0.5 * (np.arange(reach + 1) / sigma) ** 2)
    taps = (taps / (taps[0] + 2 * taps[1:].sum())).astype(np.float32)
    return _smooth(_smooth(arr.astype(np.float32), taps, 0), taps, 1)


# --- masks and compositing --------------------------------------------------------------------


def dilate(
    alpha: np.ndarray,
    radius: float,
    wobble: float = 0.0,
    seed: int = 0,
    scale: float = 40,
    grid: tuple[int, int] | None = None,
) -> np.ndarray:
    """Grow a mask by `radius` px: the sticker edge. Returns a soft mask, 1 inside the grown shape.

    No distance transform: the mask is thresholded at 0.5 and blurred with sigma = radius / 2. Near
    a straight edge the blurred value is a known function of the signed distance outside it, an
    erfc, so a lookup table turns the blur back into that distance. The grown edge is where the
    distance equals the radius, with a 1 px antialiased ramp.

    Compared with an exact distance, the edge of a convex shape comes out a little tight (0.75 px
    on a 50 px disk at radius 22, under 0.2 px at radius 8 or less), and a concave corner comes
    out rounded, which reads as cut with scissors.

    wobble adds a smooth random push (lowfreq, with features about `scale` px across, and its random
    grid when it has one) of up to that many pixels to the radius, which makes the edge look cut by
    hand. The table sees only 3 sigma out from the shape, so the edge cannot sit farther than 1.5 *
    radius - 0.75 px away: a wobble that would push it past that is cut off there.
    """
    if radius <= 0:
        return (alpha > 0.5).astype(np.float32)

    sigma = radius / 2
    blurred = _gaussian((alpha > 0.5).astype(np.float32), sigma)

    outside = np.linspace(-3 * sigma, 3 * sigma, 512)  # signed distance outside the edge
    # What a blurred straight edge reads at each distance: it falls from 1 (inside) to 0 (far out).
    profile = np.array([0.5 * math.erfc(r / (sigma * math.sqrt(2))) for r in outside])
    distance = np.interp(blurred, profile[::-1], outside[::-1])  # np.interp wants rising x

    edge_at = radius + (lowfreq(alpha.shape, scale, wobble, seed, grid) if wobble else 0.0)
    # Far from the shape the table can only say "3 sigma or more". An edge farther out than that
    # would turn all of that empty space into edge, so it stops where the table does.
    edge_at = np.minimum(edge_at, 3 * sigma - 0.75)
    return np.clip(edge_at - distance + 0.75, 0.0, 1.0).astype(np.float32)


def over(dst: np.ndarray, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Lay a color (3 values, or a picture the size of dst) over dst, as much as alpha (height x
    width, 0-1) says."""
    a = alpha[..., None]
    return dst * (1 - a) + rgb * a


def screen(dst: np.ndarray, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Screen a color over dst, as much as alpha says: it only lightens, so a glow adds light
    without hiding what is under it."""
    return dst + (1 - dst) * rgb * alpha[..., None]


def _quantize(arr: np.ndarray) -> Image.Image:
    """A float array (0-1) rounded to 8 bits, as the Pillow image its shape gives: "L" for height x
    width, "RGB" or "RGBA" for 3 or 4 channels."""
    return Image.fromarray(np.rint(np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))


def to_image(arr: np.ndarray, mode: str = "RGB") -> Image.Image:
    """A float array (0-1; height x width, or with 3 or 4 channels) as a Pillow image of `mode`."""
    img = _quantize(arr)
    return img if img.mode == mode else img.convert(mode)


def from_image(img: Image.Image) -> np.ndarray:
    """A Pillow image as float32 in 0-1: height x width for "L", with 3 or 4 channels for "RGB" and
    "RGBA". Any other mode is converted to RGBA first."""
    if img.mode not in ("L", "RGB", "RGBA"):
        img = img.convert("RGBA")
    return np.asarray(img, np.float32) / 255


# --- fonts and text ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def font(role: str, size: int) -> ImageFont.FreeTypeFont:
    """The bundled font of a FONT_FILES role at a pixel size, with its variable axes set.

    Cached by (role, size), so treat the result as read-only. Text is laid out by FreeType alone
    (no Raqm), so a line is the same width on every machine.
    """
    file_name, axes = FONT_FILES[role]
    path = str(FONTS_DIR / file_name)
    loaded = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.BASIC)
    if axes:
        loaded.set_variation_by_axes(axes)
    return loaded


def _rgba(color) -> tuple[int, int, int, int]:
    """A drawing color as (r, g, b, a) integers: a hex string ("#ff8800", or "#ff880080" with
    alpha) or a tuple of 3 or 4 values 0-255."""
    if isinstance(color, str):
        return ImageColor.getcolor(color, "RGBA")
    return tuple(int(v) for v in color) + (255,) * (4 - len(color))


def _tint(mask: Image.Image, color) -> Image.Image:
    """A layer of one color whose alpha is the "L" mask, scaled by the color's own alpha."""
    r, g, b, a = _rgba(color)
    if a < 255:
        mask = mask.point([v * a // 255 for v in range(256)])
    layer = Image.new("RGBA", mask.size, (r, g, b, 0))
    layer.putalpha(mask)
    return layer


def _composite(img: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    """Composite an RGBA layer onto img in place, its top-left corner at (x, y), clipped at the
    edges of img."""
    left, top = max(x, 0), max(y, 0)
    right, bottom = min(x + layer.width, img.width), min(y + layer.height, img.height)
    if right <= left or bottom <= top:
        return
    if (left, top, right, bottom) != (x, y, x + layer.width, y + layer.height):
        layer = layer.crop((left - x, top - y, right - x, bottom - y))
    img.alpha_composite(layer, (left, top))


def _require_rgba(img: Image.Image) -> None:
    if img.mode != "RGBA":
        raise ValueError(f"drawing needs an RGBA image, got {img.mode}")


def _floor(v: float) -> int:
    return math.floor(round(v, 6))  # round first: 59.99999999999999 is 60 for our purposes


def _ceil(v: float) -> int:
    return math.ceil(round(v, 6))


def _glows(glow) -> list[tuple]:
    """The glow argument of draw_text as a list of (color, radius) pairs: none for no glow or an
    empty list, one for a single (color, radius), or the list that was given."""
    if not glow:
        return []
    if len(glow) == 2 and isinstance(glow[1], numbers.Real):  # one (color, radius)
        return [tuple(glow)]
    return [tuple(one) for one in glow]


def draw_text(
    img: Image.Image,
    xy: tuple[float, float],
    lines: list[str],
    fnt: ImageFont.FreeTypeFont,
    fill,
    *,
    line_h: float | None = None,
    glow: tuple | list | None = None,
    shadow: tuple | None = None,
    rotate: float = 0.0,
    align: str = "left",
) -> tuple[int, int, int, int]:
    """Draw lines of text onto an RGBA image in place. Returns the text block's (x0, y0, x1, y1).

    The block is laid out like an HTML element with a line-height. Each line gets a box `line_h` px
    tall (default: the font's own ascent plus descent), stacked down from xy, and the glyphs sit
    centered in their box, so a smaller line_h packs the lines tighter, as in CSS. The block is as
    wide as its longest line, and `align` ("left" or "center") sets whether shorter lines start at
    its left edge or sit in its middle. xy is the block's top-left corner (to center a block on a
    point, subtract half of the widest fnt.getlength).

    rotate turns the block counter-clockwise by that many degrees about its center, as PIL's
    rotate does. That is the opposite sign of CSS rotate(). The box returned is the one around the
    block after the turn, in image pixels. The glow and the shadow are not part of it.

    Glyphs land on whole pixels, and a line is as wide as FreeType makes it (it rounds every advance
    to a pixel), which can differ from a browser's width by a few pixels on a long line. The
    baseline matches the browser's, as the half-leading rounds down in both.

    fill, and the colors below, are "#rrggbb" strings (or "#rrggbbaa") or tuples.
    glow=(color, radius): the text mask blurred by that Gaussian radius, in the color at full
    strength, under the text. A list of such pairs, [(color, radius), ...], draws each one in
    order, the first at the bottom, as the layers of a CSS text-shadow stack (a tight glow and a
    wide one, say). A list of one is that glow alone, and an empty list or None is no glow.
    shadow=(color, dx, dy): a hard copy of the text moved by (dx, dy), under the text and over the
    glows.
    """
    _require_rgba(img)
    if align not in ("left", "center"):
        raise ValueError(f'align must be "left" or "center", got {align!r}')
    if isinstance(lines, str):
        lines = [lines]

    ascent, descent = fnt.getmetrics()
    natural = ascent + descent
    pitch = natural if line_h is None else line_h
    half_leading = math.floor((pitch - natural) / 2)  # browsers round it down, so do we
    advances = [fnt.getlength(line) for line in lines]
    width = max(advances, default=0.0)
    height = pitch * len(lines)
    x, y = xy

    # The box of the block after the turn: the bounding box of a rotated rectangle.
    angle = math.radians(rotate)
    cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
    half_w, half_h = (width * cos + height * sin) / 2, (width * sin + height * cos) / 2
    center_x, center_y = x + width / 2, y + height / 2
    box = (
        _floor(center_x - half_w),
        _floor(center_y - half_h),
        _ceil(center_x + half_w),
        _ceil(center_y + half_h),
    )

    # Draw on a padded tile that holds the glow, the shadow and glyph parts that reach outside
    # their line box. The block's top-left sits at (pad + the fraction of xy) in the tile.
    glows = _glows(glow)
    glow_radius = max((radius for _, radius in glows), default=0)  # the room is the widest one's
    shift = max(abs(shadow[1]), abs(shadow[2])) if shadow else 0
    pad = math.ceil(0.3 * fnt.size + abs(natural - pitch) / 2 + 3 * glow_radius + shift) + 2
    left, top = math.floor(x), math.floor(y)
    tile = (math.ceil(width) + 2 * pad + 1, math.ceil(height) + 2 * pad + 1)

    def text_mask(dx: float = 0.0, dy: float = 0.0) -> Image.Image:
        # FreeType places glyphs on whole pixels, and Pillow would round a fractional x up but a
        # fractional y down, so round both the same way here.
        mask = Image.new("L", tile, 0)
        draw = ImageDraw.Draw(mask)
        for i, (line, advance) in enumerate(zip(lines, advances)):
            line_x = pad + (x - left) + (0.0 if align == "left" else (width - advance) / 2)
            baseline = pad + (y - top) + i * pitch + half_leading + ascent
            at = (math.floor(line_x + dx + 0.5), math.floor(baseline + dy + 0.5))
            draw.text(at, line, font=fnt, fill=255, anchor="ls")
        return mask

    plain = text_mask()
    layers = [(plain.filter(ImageFilter.GaussianBlur(radius)), color) for color, radius in glows]
    if shadow:
        layers.append((text_mask(shadow[1], shadow[2]), shadow[0]))
    layers.append((plain, fill))

    for mask, color in layers:
        origin_x, origin_y = left - pad, top - pad
        if rotate:
            turned = mask.rotate(rotate, _BICUBIC, expand=True)
            origin_x = round(origin_x + (mask.width - turned.width) / 2)
            origin_y = round(origin_y + (mask.height - turned.height) / 2)
            mask = turned
        _composite(img, _tint(mask, color), origin_x, origin_y)
    return box


# --- antialiased shapes -----------------------------------------------------------------------
#
# Each shape is drawn `ss` times larger on a tile that covers just its bounding box (never the
# whole image), shrunk with LANCZOS to an alpha mask, tinted and composited.


def _bounds(points, margin: float, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The integer box around the points, grown by margin and cut to an image of `size`. No points
    give an empty box."""
    if not points:
        return 0, 0, 0, 0
    xs, ys = zip(*points)
    return (
        max(_floor(min(xs) - margin), 0),
        max(_floor(min(ys) - margin), 0),
        min(_ceil(max(xs) + margin), size[0]),
        min(_ceil(max(ys) + margin), size[1]),
    )


def _shape_mask(box, ss: int, size: float, paint) -> Image.Image:
    """An "L" mask the size of box: a shape drawn ss times larger on a tile, shrunk with LANCZOS.

    paint(draw, at, size) fills the shape with 255, where at(point) turns image coordinates into
    tile coordinates and size is the stroke thickness or disk diameter in tile pixels. PIL draws a
    shape centered on a pixel only at an odd size, and it would drift half a pixel off center at an
    even one, so size is first split between the two odd sizes around it: the mask is their blend,
    which has the right weight and stays centered. (A 2 px line at ss 3 is 6 tile pixels, between 5
    and 7.)
    """
    x0, y0, x1, y1 = box
    narrow = max(1, 2 * math.floor((size - 1) / 2) + 1)
    share = min(max((size - narrow) / 2, 0.0), 1.0)  # how much of the next odd size to take

    def draw_at(odd: int) -> Image.Image:
        big = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
        paint(ImageDraw.Draw(big), lambda p: ((p[0] - x0) * ss, (p[1] - y0) * ss), odd)
        return big.resize((x1 - x0, y1 - y0), _LANCZOS)

    mask = draw_at(narrow)
    return Image.blend(mask, draw_at(narrow + 2), share) if share > 0.01 else mask


def _disk(draw: ImageDraw.ImageDraw, x: float, y: float, diameter: int, fill: int = 255) -> None:
    """A disk of odd diameter centered on the pixel that holds (x, y)."""
    r = diameter // 2
    cx, cy = math.floor(x), math.floor(y)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)


def _stroke(draw: ImageDraw.ImageDraw, pts, thick: int) -> None:
    """A polyline of odd thickness with round caps and joins. Each point lands on the pixel that
    holds it, which is also where PIL centers the line and where the end disks sit."""
    snapped = [(math.floor(x), math.floor(y)) for x, y in pts]
    if len(snapped) > 1:
        draw.line(snapped, fill=255, width=thick, joint="curve")
    for x, y in (snapped[0], snapped[-1]):
        _disk(draw, x, y, thick)


def quad_points(p0, c, p1, n: int = 32) -> list[tuple[float, float]]:
    """n points along the quadratic curve from p0 to p1 that bends toward the control point c,
    both ends included."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    start, control, end = np.array(p0, float), np.array(c, float), np.array(p1, float)
    curve = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t**2 * end
    return [(float(px), float(py)) for px, py in curve]


def draw_path(img: Image.Image, points, color, width: float, ss: int = 3) -> None:
    """Stroke a polyline `width` px thick onto an RGBA image in place, with round caps and joins."""
    _require_rgba(img)
    box = _bounds(points, width / 2 + 2, img.size)
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    mask = _shape_mask(
        box, ss, width * ss, lambda draw, at, thick: _stroke(draw, [at(p) for p in points], thick)
    )
    _composite(img, _tint(mask, color), box[0], box[1])


def draw_arrowhead(
    img: Image.Image, tip, from_pt, color, width: float, head: float = 24, ss: int = 3
) -> None:
    """An open arrowhead at `tip` for a shaft that arrives from from_pt (any point back along it,
    such as a curve's control point). Two strokes `head` px long make the barbs, half a radian
    either side of the shaft, joined at the tip like the rest of the arrow."""
    back = math.atan2(from_pt[1] - tip[1], from_pt[0] - tip[0])
    barbs = [
        (tip[0] + head * math.cos(back + side), tip[1] + head * math.sin(back + side))
        for side in (-0.5, 0.5)
    ]
    draw_path(img, [barbs[0], tip, barbs[1]], color, width, ss)


def draw_polygon_aa(
    img: Image.Image, points, fill, outline=None, width: float = 0, ss: int = 3
) -> None:
    """Draw a polygon onto an RGBA image in place: filled with `fill` (None for no fill) and
    outlined in `outline`, `width` px thick. The outline is centered on the polygon's edge, as an
    SVG stroke is, and its corners are round."""
    _require_rgba(img)
    stroked = outline is not None and width > 0
    box = _bounds(points, (width / 2 if stroked else 0) + 2, img.size)
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    if fill is not None and len(points) >= 3:  # fewer points enclose nothing

        def paint_fill(draw, at, _size):
            draw.polygon([(math.floor(x), math.floor(y)) for x, y in map(at, points)], fill=255)

        _composite(img, _tint(_shape_mask(box, ss, 1, paint_fill), fill), box[0], box[1])
    if stroked:
        closed = [*points, points[0]]

        def paint_outline(draw, at, thick):
            _stroke(draw, [at(p) for p in closed], thick)

        mask = _shape_mask(box, ss, width * ss, paint_outline)
        _composite(img, _tint(mask, outline), box[0], box[1])


def draw_circle(img: Image.Image, center, r: float, color, width: float, ss: int = 3) -> None:
    """Draw a circle onto an RGBA image in place: a ring `width` px thick centered on radius r (as
    an SVG circle with a stroke is), or a filled disk of radius r when width is 0."""
    _require_rgba(img)
    box = _bounds([center], r + width / 2 + 2, img.size)
    if box[2] <= box[0] or box[3] <= box[1]:
        return

    if width > 0:
        # A ring is two disks, drawn at the odd sizes mid + thick and mid - thick around a middle
        # diameter that is even, so the thickness is the size that gets blended and both disks stay
        # on one center.
        mid = 2 * round(r * ss)

        def paint(draw, at, thick):
            x, y = at(center)
            _disk(draw, x, y, mid + thick)
            if mid > thick:
                _disk(draw, x, y, mid - thick, fill=0)

        mask = _shape_mask(box, ss, width * ss, paint)
    else:
        mask = _shape_mask(
            box, ss, 2 * r * ss, lambda draw, at, diameter: _disk(draw, *at(center), diameter)
        )
    _composite(img, _tint(mask, color), box[0], box[1])
