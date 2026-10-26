"""Tests for the sizes: the plan a clip is laid out at, the size it is rendered at, and its shape.

The output is the clip's own frame (Amendment 1): 16:9 stays 16:9 and 9:16 stays 9:16. Everything is laid
out at the plan, 1080 px wide, and drawn at the size of the output. The numbers are written out here, so
that a wrong constant in config is caught.
"""

import pytest

import fetch_fonts
from mclib.config import FONT_FILES, FONTS_DIR, layout_name, output_size, plan_size


@pytest.mark.parametrize(
    ("clip", "plan"),
    [
        ((1920, 1080), (1080, 608)),  # 607.5 is not a whole number of pixels, and 608 is even
        ((1280, 720), (1080, 608)),
        ((3840, 2160), (1080, 608)),
        ((320, 180), (1080, 608)),
        ((1080, 1920), (1080, 1920)),
        ((720, 1280), (1080, 1920)),
        ((1440, 1080), (1080, 810)),  # 4:3
        ((1080, 1080), (1080, 1080)),  # 1:1
        ((1080, 1350), (1080, 1350)),  # 4:5
        ((1000, 1001), (1080, 1082)),  # 1081.08 is 1081, and an odd height is made even
        ((4096, 2160), (1080, 570)),  # 569.5 is 570
    ],
)
def test_the_plan_is_1080_wide_in_the_shape_of_the_clip(clip, plan):
    assert plan_size(*clip) == plan
    assert plan[0] == 1080 and plan[1] % 2 == 0


@pytest.mark.parametrize(
    ("clip", "output"),
    [
        ((1920, 1080), (1920, 1080)),  # the clip's own
        ((1080, 1920), (1080, 1920)),
        ((1280, 720), (1280, 720)),  # 1280 is over 1080 wide, so it stays
        ((3840, 2160), (3840, 2160)),  # 4K is the most, and it fits
        ((2160, 3840), (2160, 3840)),
        ((1440, 1080), (1440, 1080)),
        ((1080, 1080), (1080, 1080)),
        ((320, 180), (1080, 608)),  # a small clip is scaled up to 1080 wide: 607.5 is 608
        ((720, 1280), (1080, 1920)),
        ((540, 960), (1080, 1920)),
        ((1000, 1000), (1080, 1080)),
        ((4096, 2160), (3840, 2026)),  # over 3840 on the long side: scaled down to it, and 2025 made even
        ((7680, 4320), (3840, 2160)),
        ((4320, 7680), (2160, 3840)),
        ((5000, 5000), (3840, 3840)),
        ((1920, 1081), (1920, 1082)),  # an odd side is made even
    ],
)
def test_the_output_is_the_clips_own_size_within_1080_wide_and_4k(clip, output):
    found = output_size(*clip)
    assert found == output
    assert found[0] % 2 == 0 and found[1] % 2 == 0
    # it is the clip's shape, give or take the pixel that making it even costs
    assert found[0] / found[1] == pytest.approx(clip[0] / clip[1], rel=0.01)
    assert max(found) <= 3840


def test_a_clip_is_scaled_up_to_1080_wide_and_never_below_it():
    # at least 1080 wide: k is never under 1, so the notes are never smaller than they were drawn
    for width in (100, 320, 540, 1079, 1080):
        assert output_size(width, round(width * 9 / 16))[0] == 1080
    # a clip that is 1080 wide or more is not changed, up to 4K (the sides here are even already)
    for width, height in ((1082, 608), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)):
        assert output_size(width, height) == (width, height)


def test_a_clip_over_4k_is_scaled_down_to_3840_on_its_long_side_and_no_more():
    for clip in ((4096, 2160), (5000, 5000), (3841, 100), (2160, 4096), (8192, 4320)):
        assert max(output_size(*clip)) == 3840, clip
    assert output_size(3840, 100) == (3840, 100)  # 3840 itself is not over it
    # the scale is the same on both sides: the shape is kept
    assert output_size(8192, 4320) == (3840, 2026)


def test_the_bundled_fonts_are_the_fonts_the_edit_draws_with_and_nothing_more():
    # The title was the only text drawn in Nunito (Amendment 1 took the title out of the edit), so the font, its
    # license text and its place in the maintainer's download script are gone with it, and the folder, the table
    # of roles and the script say the same six
    assert set(FONT_FILES) == {"marker", "comic", "hand", "pixel", "terminal", "label"}
    wanted = {name for name, _ in FONT_FILES.values()}
    assert {path.name for path in FONTS_DIR.glob("*.ttf")} == wanted
    assert set(fetch_fonts.FAMILY_DIRS) == set(FONT_FILES)
    assert {path.name for path in FONTS_DIR.glob("OFL-*.txt")} == {
        f"OFL-{folder}.txt" for folder in fetch_fonts.FAMILY_DIRS.values()}
    assert not [path for path in FONTS_DIR.iterdir() if "nunito" in path.name.lower()]
    assert "Nunito" not in (FONTS_DIR.parent / "LICENSE-NOTES.md").read_text(encoding="utf-8")


def test_the_layout_names_the_shape_of_the_clip():
    assert layout_name(1920, 1080) == "landscape"
    assert layout_name(1080, 1080) == "landscape"  # a square one is landscape
    assert layout_name(1081, 1080) == "landscape"
    assert layout_name(1080, 1081) == "portrait"
    assert layout_name(1080, 1920) == "portrait"
    assert layout_name(320, 180) == "landscape"
