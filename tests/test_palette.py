"""Tests for the contrast guard and the palette taken from an image.

The expected numbers are worked out here with the WCAG formula, or are WCAG's own examples, and the
contrast rule of each notes style is the table of the Global Constraints written out in full, with
its pixel row as ruling C1 amends it: the ink on the tone of the recipe's background.
"""

import io
import math
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mclib import palette as palette_module
from mclib.config import CONTRAST_MIN
from mclib.palette import NoVisiblePixels, contrast, palette_from_image, text_contrast
from mclib.project import default_project, validate
from mclib.raster import hex_rgb, mix
from mclib.styles import background, notes
from mclib.styles.recipes import BUILTINS, PALETTES, ROLES, validate_recipe

CLIP = {
    "path": "intro.mp4", "stem": "intro", "width": 1920, "height": 1080, "fps": 30.0,
    "duration": 6.0, "frames": 180, "orientation": "landscape",
}
# The picture of the brief: five equal blocks, as dark, light, red, blue and gray.
BLOCKS = ("#101010", "#f0f0f0", "#e8261c", "#1d4ed8", "#808080")


def recipe_with(notes_id, **colors):
    """A copy of Bedroom Pop with another notes style and some palette colors replaced."""
    recipe = deepcopy(BUILTINS["bedroom-pop"])
    recipe["notes"] = notes_id
    recipe["palette"].update(colors)
    return recipe


def bands(colors, width=64, height=64, alpha=255):
    """An RGBA picture of the colors as bands side by side, as equal as the width allows. At 64 x 64
    the picture is read as it is, with no shrinking to blur the edges between the bands."""
    picture = np.zeros((height, width, 4), np.uint8)
    for color, columns in zip(colors, np.array_split(np.arange(width), len(colors))):
        picture[:, columns] = [*(int(color[i : i + 2], 16) for i in (1, 3, 5)), alpha]
    return Image.fromarray(picture)


def taken_from(image, folder, name="picture.png"):
    """The palette that palette_from_image takes from a picture, saved in the folder first."""
    image.save(folder / name)
    return palette_from_image(folder / name)


def levels(color):
    """A "#rrggbb" color as three values, 0-255."""
    return np.rint(hex_rgb(color) * 255)


def test_contrast_black_white():
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#ffffff", "#000000") == pytest.approx(21.0)  # it does not matter which is lighter
    assert contrast("#7d2a72", "#7D2A72") == pytest.approx(1.0)  # the same color, in either case

    # WCAG's gray that just reaches 4.5:1 on white is #767676 (4.54); one step lighter falls short
    assert contrast("#767676", "#ffffff") == pytest.approx(4.54, abs=0.005)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.005)
    # The weights of the channels: red is 0.2126 of the light, blue 0.0722 and green 0.7152
    assert contrast("#ff0000", "#ffffff") == pytest.approx(1.05 / (0.2126 + 0.05))
    assert contrast("#0000ff", "#ffffff") == pytest.approx(1.05 / (0.0722 + 0.05))
    assert contrast("#00ff00", "#000000") == pytest.approx((0.7152 + 0.05) / 0.05)


def test_builtin_recipes_pass_guard():
    for recipe_id, recipe in BUILTINS.items():
        assert text_contrast(recipe) >= CONTRAST_MIN, recipe_id
    assert CONTRAST_MIN == 3.0

    # Each notes style is measured on what it draws on (the table of the Global Constraints). The
    # palette is made of colors far enough apart that a role taken for another gives another number.
    # The pixel notes have no fill and no glow, so they sit on the recipe's background: its tone
    # (the table said the middle of dark and light before ruling C1).
    colors = {"dark": "#203a8c", "light": "#d8c890", "tex": "#cc2222", "edge": "#f4d0d0", "ink": "#fff0a8"}
    measured = {
        "marker": (colors["ink"], colors["dark"]),
        "comic": ("#111111", mix(colors["light"], "#ffffff", 0.45)),
        "hand": (colors["ink"], "#fffdf5"),
        "label": (colors["dark"], colors["edge"]),
        "pixel": (colors["ink"], background.tone("dots", colors)),  # Bedroom Pop's background
    }
    assert set(measured) == {"marker", "comic", "hand", "label", "pixel"}
    ratios = {}
    for notes_id, pair in measured.items():
        ratios[notes_id] = text_contrast(recipe_with(notes_id, **colors))
        assert ratios[notes_id] == pytest.approx(contrast(*pair)), notes_id
    assert len({round(ratio, 2) for ratio in ratios.values()}) == len(ratios)
    assert measured["pixel"][1] != mix(colors["dark"], colors["light"], 0.5)
    with pytest.raises(ValueError, match="glitter"):
        text_contrast(recipe_with("glitter"))

    # Only the pixel notes read the background: on the night one they have another number, and the
    # other four notes styles keep theirs
    night = {"bg": "night"}
    assert text_contrast({**recipe_with("pixel", **colors), **night}) == pytest.approx(
        contrast(colors["ink"], background.tone("night", colors))
    )
    assert background.tone("night", colors) != background.tone("dots", colors)
    for notes_id in ("marker", "comic", "hand", "label"):
        assert text_contrast({**recipe_with(notes_id, **colors), **night}) == pytest.approx(ratios[notes_id])
    assert BUILTINS["player-select"]["notes"] == "pixel"  # so the loop above measured it on its night
    # What the guard measures against is what the notes draw: the comic box, the paper of a strip
    assert (palette_module._COMIC_TEXT, palette_module._COMIC_FILL, palette_module._HAND_PAPER) == (
        notes._COMIC_TEXT, notes._COMIC_FILL, notes._HAND_PAPER,
    )

    # The stats panel draws the cells that are on in the most visible of the texture, edge and ink
    # colors on the dark one. Riso's texture color is its dark blue, which would not show.
    riso = PALETTES["Riso"]
    assert riso["tex"] == riso["dark"] and contrast(riso["tex"], riso["dark"]) == pytest.approx(1.0)
    assert notes.stats_bar_color(riso) != riso["dark"]
    assert notes.stats_bar_color(riso) == riso["edge"]  # 7.0:1, against 2.3:1 for the ink
    for name, pal in PALETTES.items():
        bar = notes.stats_bar_color(pal)
        best = max(contrast(pal[role], pal["dark"]) for role in ("tex", "edge", "ink"))
        assert bar in (pal["tex"], pal["edge"], pal["ink"])
        assert contrast(bar, pal["dark"]) == pytest.approx(best) and best >= CONTRAST_MIN, name
    assert notes.stats_bar_color(PALETTES["Neon"]) == PALETTES["Neon"]["edge"]  # 16.5, ink 15.3, tex 14.4
    assert notes.stats_bar_color(PALETTES["Mono"]) == PALETTES["Mono"]["edge"]  # tex is the dark color
    # a tie goes to the first of tex, edge and ink (the same white, written in the two cases)
    tied = dict(PALETTES["Comic"], tex="#FFFFFF", edge="#ffffff")
    assert notes.stats_bar_color(tied) == "#FFFFFF"
    assert notes.stats_bar_color({**tied, "tex": "#ffffff", "edge": "#FFFFFF"}) == "#ffffff"

    # The panel's border keeps the texture color when that shows on the dark one (3:1, ruling C2),
    # and takes the bars' color when it does not. Riso and Mono have the dark color for texture,
    # and Magenta, Lagoon and Tangerine have a texture color that is darker still.
    for name, pal in PALETTES.items():
        shows = contrast(pal["tex"], pal["dark"]) >= CONTRAST_MIN
        assert notes.stats_border_color(pal) == (pal["tex"] if shows else notes.stats_bar_color(pal)), name
    assert [name for name, pal in PALETTES.items() if notes.stats_border_color(pal) != pal["tex"]] == [
        "Magenta", "Lagoon", "Tangerine", "Riso", "Mono",
    ]
    assert notes.stats_border_color(riso) == riso["edge"] != riso["tex"]
    assert notes.stats_border_color(PALETTES["Neon"]) == PALETTES["Neon"]["tex"]  # 14.4:1 on the dark one
    assert notes.stats_border_color(PALETTES["Comic"]) == PALETTES["Comic"]["tex"]  # 4.25:1: red is kept


def test_the_ratio_shown_in_a_sentence_is_floored_to_one_decimal():
    # "Text contrast 2.9:1 is hard to read": a ratio the guard turns down (under 3.0) is never shown as 3.0
    for ratio, shown in (
        (2.95, "2.9"), (2.96, "2.9"), (2.999999, "2.9"), (3.0, "3.0"), (3.04, "3.0"), (3.05, "3.0"),
        (1.0, "1.0"), (1.138, "1.1"), (7.0, "7.0"), (12.34, "12.3"), (21.0, "21.0"),
    ):
        assert palette_module.floor_contrast(ratio) == shown, ratio


def test_low_contrast_recipe_warns():
    # Pink ink on a pink dark color, in marker notes: 1.1:1
    recipe = recipe_with("marker", ink="#f2a9de", dark="#e0a0d0")
    validate_recipe(recipe)
    assert text_contrast(recipe) < CONTRAST_MIN
    assert text_contrast(recipe) == pytest.approx(1.138, abs=0.001)

    project = default_project(CLIP, "landscape", (1080, 608), recipe)
    warnings = validate(project)
    assert warnings == ["Text contrast 1.1:1 is hard to read on a phone."]
    assert "Text contrast" in warnings[0]

    # It follows the notes style of the project's own style: hand notes are measured on the paper
    # (ink #fffdf5 on #fffdf5 is 1.0:1), and the same colors are fine in marker notes
    paper = recipe_with("hand", ink="#fffdf5")
    assert validate(default_project(CLIP, "landscape", (1080, 608), paper)) == [
        "Text contrast 1.0:1 is hard to read on a phone."
    ]
    assert validate(default_project(CLIP, "landscape", (1080, 608), {**paper, "notes": "marker"})) == []

    # Every ready-made style reads, so none of them draws the warning. Next to one of the copy's own
    # warnings (a note over its soft limit), the contrast warning comes last.
    for recipe_id, ready_made in BUILTINS.items():
        assert validate(default_project(CLIP, "landscape", (1080, 608), ready_made)) == [], recipe_id
    project["copy"]["notes"]["tl"] = "x" * 31
    warnings = validate(project)
    assert len(warnings) == 2 and "soft limit" in warnings[0] and warnings[1].startswith("Text contrast")

    # Hard errors still come first: a recipe that is not valid raises, whatever its contrast
    project["style"]["palette"]["ink"] = "pink"
    with pytest.raises(ValueError, match="palette.ink"):
        validate(project)

    # Pixel notes have no fill and no glow, so they are measured on the tone of the background the
    # recipe picks (ruling C1). Player Select with Riso's palette is #1b1b1b ink on the night
    # background: 6.1:1 on the middle of dark and light, which is what the table said before, and
    # less than 3:1 on the dim blue that is drawn (about 1.7:1 under the notes on the real plate).
    riso = PALETTES["Riso"]
    riso_night = {**deepcopy(BUILTINS["player-select"]), "palette": dict(riso)}
    assert (riso_night["bg"], riso_night["notes"]) == ("night", "pixel")
    assert contrast(riso["ink"], mix(riso["dark"], riso["light"], 0.5)) > 6
    ratio = text_contrast(riso_night)
    assert 2.0 < ratio < CONTRAST_MIN
    assert validate(default_project(CLIP, "landscape", (1080, 608), riso_night)) == [
        f"Text contrast {math.floor(ratio * 10) / 10:.1f}:1 is hard to read on a phone."  # (cut, not rounded)
    ]
    # The background decides it: the same palette on paper-colored flat color reads (14:1), and the
    # same background with Neon's palette (the ready-made Player Select) reads too
    assert text_contrast({**riso_night, "bg": "flat"}) > 10
    assert validate(default_project(CLIP, "landscape", (1080, 608), {**riso_night, "bg": "flat"})) == []
    assert text_contrast(BUILTINS["player-select"]) > CONTRAST_MIN


def test_palette_from_image_blocks(tmp_path):
    # The picture of the brief: five equal blocks, 500 x 100, shrunk to 64 x 64 (a few pixels at the
    # edges of the blocks come out as a blend). Dark and light are the darkest and the lightest
    # block, and the texture color is the most saturated of the other three: the red.
    wide = taken_from(bands(BLOCKS, 500, 100), tmp_path, "wide.png")
    assert list(wide) == list(ROLES)
    for role, color in (("dark", "#101010"), ("light", "#f0f0f0"), ("tex", "#e8261c"), ("edge", "#1d4ed8")):
        assert np.abs(levels(wide[role]) - levels(color)).max() <= 8, role
    assert all(len(color) == 7 and color == color.lower() and color[0] == "#" for color in wide.values())
    validate_recipe({**BUILTINS["bedroom-pop"], "palette": wide})  # it is a palette a style can have

    # At 64 x 64 nothing is blended, so the colors come back exactly, whatever the order of the
    # bands: the edge color is the second most saturated (the blue; the gray has none), and the
    # ink is the light color, since the middle of dark and light (#808080) is not bright
    shuffled = ("#808080", "#e8261c", "#f0f0f0", "#101010", "#1d4ed8")
    exact = taken_from(bands(shuffled), tmp_path, "exact.png")
    assert exact == {"dark": "#101010", "light": "#f0f0f0", "tex": "#e8261c", "edge": "#1d4ed8", "ink": "#f0f0f0"}
    assert taken_from(bands(shuffled), tmp_path, "again.png") == exact  # the same picture, the same colors
    assert taken_from(bands(BLOCKS), tmp_path, "unshuffled.png") == exact

    # The ink: the dark color when the middle of dark and light is bright (#606060 to white is
    # #b0b0b0, a relative luminance of 0.43, over 0.35) and the two are far enough apart (6.3:1) ...
    bright = ("#606060", "#ffffff", "#c83c3c", "#5080c0", "#a0a0a0")
    pal = taken_from(bands(bright), tmp_path, "bright.png")
    assert pal == {"dark": "#606060", "light": "#ffffff", "tex": "#c83c3c", "edge": "#5080c0", "ink": "#606060"}
    # ... and white when they are no more than 4:1 apart (the darkest and lightest of these grays
    # are 1.55:1)
    grays = ("#808080", "#707070", "#909090", "#888888", "#787878")
    pal = taken_from(bands(grays), tmp_path, "grays.png")
    assert (pal["dark"], pal["light"], pal["ink"]) == ("#707070", "#909090", "#ffffff")
    assert contrast(pal["light"], pal["dark"]) < 4

    # A picture of one color must not crash: the clusters that get no pixels keep their center
    one = taken_from(Image.new("RGB", (40, 30), "#336699"), tmp_path, "one.png")
    assert one == {"dark": "#336699", "light": "#336699", "tex": "#336699", "edge": "#336699", "ink": "#ffffff"}

    # Pixels with alpha under 128 are left out, and 128 is kept
    half = bands(("#102030", "#102030", "#e0d0c0", "#e0d0c0", "#e0d0c0"))
    for alpha, light in ((127, "#102030"), (128, "#e0d0c0")):
        picture = np.asarray(half).copy()
        picture[:, 26:, 3] = alpha  # the right part, the light color
        pal = taken_from(Image.fromarray(picture), tmp_path, f"alpha{alpha}.png")
        assert (pal["dark"], pal["light"]) == ("#102030", light), alpha
    clear = bands(BLOCKS, alpha=0)
    with pytest.raises(ValueError, match="no visible pixels"):
        taken_from(clear, tmp_path, "clear.png")


def test_no_visible_pixels_is_its_own_error(tmp_path):
    with pytest.raises(NoVisiblePixels, match="no visible pixels"):
        taken_from(bands(BLOCKS, alpha=0), tmp_path, "clear.png")
    assert issubclass(NoVisiblePixels, ValueError)  # what a caller of the old kind already catches


def test_palette_from_an_open_file(tmp_path):
    shuffled = ("#808080", "#e8261c", "#f0f0f0", "#101010", "#1d4ed8")
    picture = bands(shuffled)
    buffer = io.BytesIO()
    picture.save(buffer, "PNG")
    buffer.seek(0)

    assert palette_from_image(buffer) == taken_from(picture, tmp_path)  # a path or a file: the same colors


@pytest.mark.parametrize("kind", ["PNG", "JPEG"])
def test_a_big_picture_is_shrunk_before_it_is_converted(tmp_path, monkeypatch, kind):
    # A photo of 48 megapixels (8000 x 6000: five bands of solid color). Converted to RGBA at full
    # size it would be 192 MB, and 100 megapixels would be 400 MB, so it is shrunk first.
    big = Image.new("RGB", (8000, 6000))
    draw = ImageDraw.Draw(big)
    for number, color in enumerate(BLOCKS):
        draw.rectangle((number * 1600, 0, number * 1600 + 1599, 5999), fill=color)
    path = tmp_path / f"big.{kind.lower()}"
    big.save(path, kind, **({"quality": 95} if kind == "JPEG" else {}))
    del big, draw

    converted = []
    real_convert = Image.Image.convert

    def convert(self, *args, **kwargs):
        converted.append(self.width * self.height)
        return real_convert(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", convert)

    found = palette_from_image(path)

    assert list(found) == list(ROLES) and all(len(color) == 7 for color in found.values())
    assert max(converted) <= 1_000_000, converted  # nothing near full size was ever converted
    for role, color in (("dark", "#101010"), ("light", "#f0f0f0"), ("tex", "#e8261c"), ("edge", "#1d4ed8")):
        assert np.abs(levels(found[role]) - levels(color)).max() <= 12, role
