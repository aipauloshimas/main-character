"""Tests for the raster helpers: fonts, color math, textures, the sticker-edge dilation and the
antialiased drawing."""

import math

import numpy as np
import pytest
from PIL import ImageFont

from mclib.config import FONT_FILES, FONTS_DIR
from mclib.raster import (
    dilate,
    draw_path,
    draw_text,
    duotone,
    font,
    from_image,
    grain,
    halftone,
    luminance,
    mix,
    quad_points,
    to_image,
)

TEXT = "Main character"
BACKGROUND = np.array([20, 20, 40], np.float32) / 255


def disk(size, radius):
    """A size x size mask with a filled disk of the given radius at its center."""
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2
    return (np.hypot(xx - center, yy - center) <= radius).astype(np.float32)


def blank(width, height):
    """An opaque RGBA image of one dark flat color."""
    return to_image(np.broadcast_to(BACKGROUND, (height, width, 3)), "RGBA")


def changed(img, before):
    """The pixels of img that differ from the float array it was before."""
    return (from_image(img) != before).any(axis=-1)


# --- fonts ------------------------------------------------------------------------------------


def test_font_roles_load():
    for role, (file_name, axes) in FONT_FILES.items():
        loaded = font(role, 40)
        assert isinstance(loaded, ImageFont.FreeTypeFont)
        assert loaded.size == 40

        # A variable font must really take its weight: the same text is wider than in the
        # font's default instance. A static font is the file as it is.
        default_width = ImageFont.truetype(str(FONTS_DIR / file_name), 40).getlength(TEXT)
        if axes:
            assert loaded.getlength(TEXT) != default_width, f"{role} ignored its axes {axes}"
        else:
            assert loaded.getlength(TEXT) == default_width

    assert font("hand", 40) is font("hand", 40)  # cached by role and size
    assert font("hand", 40) is not font("hand", 41)


# --- color math and textures ------------------------------------------------------------------


def test_duotone_endpoints():
    dark, light = "#7d2a72", "#f2a9de"
    dark_rgb = np.array([0x7D, 0x2A, 0x72], np.float32) / 255
    light_rgb = np.array([0xF2, 0xA9, 0xDE], np.float32) / 255

    # Black maps to dark and white to light, with the luminance taken from real pixels.
    lum = luminance(np.array([[[0, 0, 0], [1, 1, 1]]], np.float32))
    out = duotone(lum, dark, light)
    assert out.shape == (1, 2, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], dark_rgb, atol=1 / 255)
    np.testing.assert_allclose(out[0, 1], light_rgb, atol=1 / 255)

    # Lift lets black start part of the way toward light, and gamma leaves both ends alone.
    lifted = duotone(lum, dark, light, lift=0.3, gamma=0.8)
    np.testing.assert_allclose(lifted[0, 0], dark_rgb + 0.3 * (light_rgb - dark_rgb), atol=1 / 255)
    np.testing.assert_allclose(lifted[0, 1], light_rgb, atol=1 / 255)
    assert mix(dark, light, 0.3) == "#a05092"
    assert mix(dark, light, 0.0) == dark
    assert mix(dark, light, 1.0) == light


def test_halftone_coverage_tracks_darkness():
    shape = (360, 360)
    light = halftone(np.full(shape, 0.2, np.float32), 9, 45)
    dark = halftone(np.full(shape, 0.8, np.float32), 9, 45)

    for coverage in (light, dark):
        assert coverage.shape == shape
        assert 0.0 <= coverage.min() and coverage.max() <= 1.0
    assert light.mean() < dark.mean()
    # A dot of radius 0.5 * sqrt(t) cells covers pi/4 * t of its cell.
    assert light.mean() == pytest.approx(math.pi / 4 * 0.2, abs=0.02)
    assert dark.mean() == pytest.approx(math.pi / 4 * 0.8, abs=0.02)

    # Gain scales the darkness, and phase slides the dot lattice: a whole spacing is the same
    # lattice again, anything else is not.
    base = np.full(shape, 0.5, np.float32)
    assert halftone(base, 9, 0, gain=0.5).mean() < halftone(base, 9, 0).mean()
    np.testing.assert_allclose(halftone(base, 9, 0, phase=(9, 0)), halftone(base, 9, 0), atol=1e-4)
    assert np.abs(halftone(base, 9, 0, phase=(4, 0)) - halftone(base, 9, 0)).max() > 0.5


def test_grain_is_deterministic():
    flat = np.full((64, 64, 3), 0.5, np.float32)
    first, again, other = grain(flat, 8, seed=3), grain(flat, 8, seed=3), grain(flat, 8, seed=4)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.dtype == np.float32

    # Sigma counts 8-bit levels, the same noise lands on all three channels, the input stays as
    # it was, and a single-channel array works too.
    noise = first - flat
    assert noise.std() == pytest.approx(8 / 255, rel=0.1)
    np.testing.assert_array_equal(noise[..., 0], noise[..., 1])
    np.testing.assert_array_equal(noise[..., 0], noise[..., 2])
    assert (flat == 0.5).all()
    assert grain(flat[..., 0], 8, seed=3).shape == (64, 64)


# --- the sticker edge -------------------------------------------------------------------------


def test_dilate_grows_disk_by_radius():
    mask = disk(300, 50)
    grown = dilate(mask, 22)

    assert grown.shape == mask.shape
    assert grown.dtype == np.float32
    assert 0.0 <= grown.min() and grown.max() <= 1.0

    # A disk of radius 50 grown by 22 is 144 px across its middle row.
    assert (grown[150] > 0.5).sum() == pytest.approx(144, abs=8)
    assert grown[mask > 0.5].min() > 0.99  # the shape itself stays fully covered
    assert grown[0, 0] == 0.0  # and the far corner stays empty


def test_dilate_wobble_changes_edge():
    mask = disk(300, 50)
    plain = dilate(mask, 22)
    wobbly = dilate(mask, 22, wobble=7, seed=5)

    assert np.abs(wobbly - plain).max() > 0.5  # the edge moved
    np.testing.assert_array_equal(dilate(mask, 22, wobble=7, seed=5), wobbly)  # the seed rules it
    assert not np.array_equal(dilate(mask, 22, wobble=7, seed=6), wobbly)

    # How far each pixel is from the disk's center tells where the edge is. The wobbled edge
    # strays from the plain one by at most 9 px, in or out, all the way round.
    yy, xx = np.mgrid[0:300, 0:300]
    rho = np.hypot(xx - 149.5, yy - 149.5)
    plain_edge = rho[plain > 0.5].max()
    inside, outside = rho[wobbly > 0.5], rho[wobbly <= 0.5]
    assert inside.max() <= plain_edge + 9
    assert outside.min() >= plain_edge - 9
    assert inside.max() - outside.min() > 3  # and it really is uneven, not a smaller circle


# --- drawing ----------------------------------------------------------------------------------


def test_draw_text_glow_box():
    fnt = font("comic", 48)
    ascent, descent = fnt.getmetrics()
    line_h = ascent + descent
    lines = ["Main", "character"]

    img = blank(400, 200)
    before = from_image(img)
    box = draw_text(
        img, (60, 50), lines, fnt, "#ffe14d", glow=("#3ef0ff", 6), shadow=("#000000", 3, 3)
    )
    after = from_image(img)

    # The box is the text block: it starts at xy, is as wide as the longer line, and is two line
    # heights tall. It sits inside the image and the text changed the pixels in it.
    x0, y0, x1, y1 = box
    assert (x0, y0) == (60, 50)
    assert x1 - x0 == math.ceil(fnt.getlength("character"))
    assert y1 - y0 == 2 * line_h
    assert 0 <= x0 < x1 <= 400 and 0 <= y0 < y1 <= 200
    assert (after[y0:y1, x0:x1] != before[y0:y1, x0:x1]).any()

    # The fill color is there, the hard shadow is darker than the background, the glow spills
    # past the box in its own color (cyan adds blue) but not 30 px out, and the far corner stays
    # untouched.
    fill = np.array([0xFF, 0xE1, 0x4D], np.float32) / 255
    assert (np.abs(after[..., :3] - fill).max(axis=-1) < 0.02).sum() > 200
    assert (after[..., :3].sum(axis=-1) < BACKGROUND.sum() - 0.05).any()
    assert after[y0:y1, x0 - 2, 2].max() > BACKGROUND[2] + 0.1
    np.testing.assert_array_equal(after[:, x0 - 30], before[:, x0 - 30])
    np.testing.assert_array_equal(after[-1, -1], before[-1, -1])
    assert after[..., 3].min() == 1.0  # an opaque image stays opaque

    # A glow can also be a list of (color, radius), drawn in order with the first at the bottom,
    # like the layers of a CSS text-shadow. A list of one is that glow alone, an empty list is no
    # glow, and a glow drawn later lies over an earlier one (wherever both have some strength,
    # white over black is lighter than black over white by the product of the two strengths).
    def glowing(glow):
        out = blank(400, 200)
        draw_text(out, (60, 50), lines, fnt, "#ffe14d", glow=glow)
        return from_image(out)

    np.testing.assert_array_equal(glowing([("#3ef0ff", 6)]), glowing(("#3ef0ff", 6)))
    np.testing.assert_array_equal(glowing([]), glowing(None))
    white_then_black = glowing([("#ffffff", 3), ("#000000", 12)])
    black_then_white = glowing([("#000000", 12), ("#ffffff", 3)])
    lighter = black_then_white[..., :3].mean(axis=-1) - white_then_black[..., :3].mean(axis=-1)
    assert lighter.min() >= -1 / 255 and lighter.max() > 0.04

    # The room a glow gets is the widest one's, whichever is first, so a wide glow after a tight
    # one is not cut off by the tile: 25 px out it still lights the pixels, as it does alone.
    wide = glowing([("#3ef0ff", 3), ("#3ef0ff", 20)])
    only_wide = glowing([("#3ef0ff", 20)])
    far = x0 - 25
    assert (wide[:, far] != before[:, far]).any()
    np.testing.assert_array_equal(wide[:, far], only_wide[:, far])

    # Rotation is counter-clockwise, as PIL's rotate: the right end of a line rises. The box
    # grows to hold the tilted block.
    tilted = blank(400, 200)
    tx0, ty0, tx1, ty1 = draw_text(tilted, (60, 90), [TEXT], fnt, "#ffe14d", rotate=12)
    ys, xs = np.nonzero(changed(tilted, before))
    assert ys[xs < xs.mean()].mean() - ys[xs >= xs.mean()].mean() > 5
    assert 0 <= tx0 and tx1 <= 400 and 0 <= ty0 and ty1 <= 200
    assert ty1 - ty0 > line_h
    assert ys.min() >= ty0 and ys.max() < ty1 and xs.min() >= tx0 and xs.max() < tx1

    # Center alignment puts the shorter line under the middle of the longer one.
    def middle_gap(align):
        """How far apart the middles of the ink of the two lines are."""
        aligned = blank(400, 200)
        draw_text(aligned, (60, 50), lines, fnt, "#ffe14d", align=align)
        ink = changed(aligned, before)
        middles = []
        for top in (50, 50 + line_h):
            columns = np.nonzero(ink[top:top + line_h].any(axis=0))[0]
            middles.append((columns.min() + columns.max()) / 2)
        return abs(middles[0] - middles[1])

    assert middle_gap("center") < 4
    assert middle_gap("left") > 20
    with pytest.raises(ValueError, match="align"):
        draw_text(blank(400, 200), (60, 50), lines, fnt, "#ffe14d", align="right")


def test_draw_path_marks_pixels_along_curve():
    p0, control, p1 = (20, 160), (100, 20), (180, 160)
    points = quad_points(p0, control, p1)
    assert len(points) == 32
    assert points[0] == pytest.approx(p0) and points[-1] == pytest.approx(p1)

    img = to_image(np.zeros((200, 200, 4), np.float32), "RGBA")
    draw_path(img, points, "#ff0000", 6)

    # The curve passes through 0.25 * p0 + 0.5 * control + 0.25 * p1 = (100, 90), a point that
    # a 6 px stroke must cover. (LANCZOS rings a little, so full coverage can read 250 or so.)
    # Pixels well away from the curve are not touched.
    red, green, blue, alpha = img.getpixel((100, 90))
    assert (red, green, blue) == (255, 0, 0) and alpha >= 250
    assert img.getpixel((100, 70)) == (0, 0, 0, 0)
    assert img.getpixel((10, 10)) == (0, 0, 0, 0)
    assert img.getpixel((190, 190)) == (0, 0, 0, 0)

    # The stroke is antialiased: its edge has many in-between alphas.
    assert len(np.unique(np.asarray(img)[..., 3])) > 8

    # A straight 10 px line has round caps, which reach 5 px past its end: a flat end would leave
    # the pixels 3.5 px out empty, and the cap is long gone 7 px out (a little ringing aside).
    line = to_image(np.zeros((100, 200, 4), np.float32), "RGBA")
    draw_path(line, [(50, 50), (150, 50)], "#ff0000", 10)
    assert line.getpixel((46, 50))[3] >= 240
    assert line.getpixel((43, 50))[3] < 8
