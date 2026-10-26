"""Tests for drawing a card at any size.

A card is laid out at the plan, 1080 px wide, where everything was tuned and the author approved the
four styles. A render draws the same card on the clip's own frame, which can be any size of that
shape, so every size in px is multiplied by k, the ratio of the widths (raster.Scale). Two oracles
hold the work to the approved look:

1. The look at the plan does not change. Four pictures in tests/fixtures (approved_landscape*.png,
   the stage of the card of the "plate" project in each built-in style) were drawn with the sample
   copy by code that matched the approved look byte for byte. The card at the plan is those pictures.
2. A card drawn bigger is the same card. At twice the size, shrunk by half, it is the card at the
   plan: the same colors, dots, edge and notes, and nothing about the layout moved.

The plate project is conftest's plate_maker: a real close-up matte over a drawn room.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from mclib import card, matte
from mclib.layout import NoteBox
from mclib.raster import IDENTITY, Scale, dilate, font, grain, halftone, lowfreq
from mclib.styles import engine, notes
from mclib.styles.recipes import BUILTINS, OPTIONS

FIXTURES = Path(__file__).parent / "fixtures"
APPROVED = {
    "bedroom-pop": "approved_landscape.png",
    "comic-pop": "approved_landscape_comic-pop.png",
    "scrapbook": "approved_landscape_scrapbook.png",
    "player-select": "approved_landscape_player-select.png",
}
PLAN = (1080, 608)
TWICE = (2160, 1216)


def blurred(image):
    """A picture as int16 levels after a Gaussian blur of 1 px, to compare two that differ in the
    way a bigger picture shrunk back does, by a pixel of an edge here and there."""
    return np.asarray(image.filter(ImageFilter.GaussianBlur(1)), np.int16)


def shrunk_difference(big, plan):
    """The mean absolute difference, in levels (of 255), between a picture drawn bigger, shrunk to
    the plan's size with Lanczos, and the picture drawn at the plan, both after a 1 px blur."""
    half = big.resize(plan.size, Image.Resampling.LANCZOS)
    return float(np.abs(blurred(half) - blurred(plan)).mean())


# --- oracle 1: the approved look -----------------------------------------------------------------


@pytest.mark.parametrize("style_id", list(APPROVED))
def test_the_approved_look_at_the_plan_does_not_change(plate_maker, tmp_path, style_id):
    folder, project = plate_maker(tmp_path, BUILTINS[style_id])

    drawn = np.asarray(card.render_card(folder, project)).astype(int)

    with Image.open(FIXTURES / APPROVED[style_id]) as approved:
        wanted = np.asarray(approved.convert("RGB")).astype(int)
    assert drawn.shape == wanted.shape == (608, 1080, 3)
    # byte for byte on the machine that made the pictures; the allowance is for another build of
    # Pillow or numpy, and is what the ruling asks of it: no channel off by more than 2 levels
    difference = np.abs(drawn - wanted)
    assert difference.max() <= 2 and difference.mean() < 0.1


# --- oracle 2: a bigger card is the same card -------------------------------------------------------


@pytest.mark.parametrize("style_id", list(APPROVED))
def test_a_card_drawn_at_twice_the_size_is_the_card_at_the_plan(plate_maker, tmp_path, style_id):
    folder, project = plate_maker(tmp_path, BUILTINS[style_id])
    plan = card.render_card(folder, project)

    big = card.render_card(folder, project, size=TWICE)

    assert big.size == TWICE and big.mode == "RGB"
    assert shrunk_difference(big, plan) < 3

    # The layout is the plan's, and a box at twice the size is exactly twice the box: the same
    # corner, the same width and height, the same arrow.
    boxes = card.elements(folder, project)
    twice = notes.scaled_layout(boxes, Scale.between(PLAN, TWICE))
    assert set(twice) == set(boxes)
    for key, box in boxes.items():
        found = twice[key]
        assert (found.x, found.y, found.w, found.h) == (2 * box.x, 2 * box.y, 2 * box.w, 2 * box.h), key
        assert found.lines == box.lines and (found.clash, found.cut) == (box.clash, box.cut), key
        wanted = None if box.arrow is None else tuple((2 * x, 2 * y) for x, y in box.arrow)
        assert found.arrow == wanted, key


@pytest.mark.slow
def test_a_card_drawn_at_4k_is_the_card_at_the_plan(plate_maker, tmp_path):
    # 3840 x 2160 is k = 3.56, the biggest a render goes: the stage, the edge and the notes are all
    # drawn at that size, in about ten seconds
    folder, project = plate_maker(tmp_path, BUILTINS["bedroom-pop"])
    plan = card.render_card(folder, project)

    big = card.render_card(folder, project, size=(3840, 2160))

    assert big.size == (3840, 2160)
    assert shrunk_difference(big, plan) < 3


# --- Scale ---------------------------------------------------------------------------------------


def test_the_plan_is_drawn_as_it_always_was():
    # IDENTITY gives back what it is given, the very same number, so that nothing is made of a
    # k of 1.0 that was not made before
    for scale in (IDENTITY, Scale.between(PLAN, None), Scale.between(PLAN, PLAN)):
        assert scale is IDENTITY and scale.identity
    assert type(IDENTITY.px(22)) is int and IDENTITY.px(22) == 22
    assert IDENTITY.px(2.5) == 2.5 and IDENTITY.n(4, minimum=9) == 4 and IDENTITY.phase(2) == 2
    assert IDENTITY.x(7.5) == 7.5 and IDENTITY.y(7.5) == 7.5
    assert IDENTITY.font(52) == 52 and IDENTITY.font(24, pixel=True) == 24
    assert IDENTITY.coarse(40) is None and IDENTITY.noise_shape((608, 1080, 3)) == (608, 1080)


def test_scale_multiplies_sizes_by_the_width_and_positions_by_each_axis():
    twice = Scale.between(PLAN, TWICE)
    assert (twice.k, twice.ky, twice.plan) == (2.0, 2.0, PLAN) and not twice.identity
    assert twice.px(22) == 44 and twice.px(2.5) == 5.0
    assert twice.x(10) == 20 and twice.y(10) == 20
    assert twice.font(56) == 112 and twice.font(24, pixel=True) == 48 and twice.font(16, pixel=True) == 32

    # A 16:9 clip of 1920x1080 is 1.778 times the plan across and 1.776 down, because 607.5 is not
    # a whole number of pixels
    wide = Scale.between(PLAN, (1920, 1080))
    assert wide.k == pytest.approx(1920 / 1080) and wide.ky == pytest.approx(1080 / 608)
    assert wide.k != wide.ky and wide.x(1080) == pytest.approx(1920) and wide.y(608) == pytest.approx(1080)

    # The biggest: 3840 x 2160 is 3.556 across. Lattices on whole pixels are rounded, and never under
    # their minimum; Press Start 2P goes to the nearest multiple of 8, and never under 8.
    huge = Scale.between(PLAN, (3840, 2160))
    assert huge.n(4, minimum=2) == 14 and huge.n(4) == 14 and huge.n(1, minimum=4) == 4
    assert huge.font(24, pixel=True) == 88 and huge.font(16, pixel=True) == 56
    assert huge.font(8, pixel=True) == 32 and huge.font(1, pixel=True) == 8
    assert huge.font(52) == 185 and huge.font(1) == 4


def test_a_lattice_is_shifted_so_that_it_lands_where_the_plans_does():
    # halftone reads pixel numbers, and the centers of a picture k times as big are (k - 1) / 2 of its
    # pixels from where the plan's were. With that, the plan's dots land on the bigger picture's.
    twice = Scale.between(PLAN, TWICE)
    assert twice.phase(0) == 0.5 and twice.phase(1) == 2.5 and twice.phase(2) == 4.5
    assert Scale.between(PLAN, (1080 * 3, 608 * 3)).phase(0) == 1.0

    # the dots of the dots style (9 px apart at 45 degrees), slid by the boil's 1 px, at the plan and at
    # twice the size, shrunk by half: with the phase they are the same dots, and without it they are
    # a quarter of a plan pixel off, ten times the error
    plan = halftone(np.full(PLAN[::-1], 0.5, np.float32), 9, 45, 0.9, phase=(1, 1))
    errors = {}
    for label, shift in (("with", twice.phase(1)), ("without", twice.px(1))):
        dots = halftone(np.full(TWICE[::-1], 0.5, np.float32), twice.px(9), 45, 0.9, phase=(shift, shift))
        errors[label] = float(np.abs(dots.reshape(608, 2, 1080, 2).mean(axis=(1, 3)) - plan).mean())
    assert errors["with"] < 0.02 and errors["without"] > 0.04, errors


def test_film_grain_is_drawn_at_the_plan_and_enlarged():
    plan = grain(np.zeros(PLAN[::-1], np.float32), 8, 3)
    big = grain(np.zeros(TWICE[::-1], np.float32), 8, 3, plan=PLAN[::-1])

    # every grain of the plan is a 2 x 2 block of the bigger picture, as strong
    assert big.shape == (1216, 2160)
    for row in (0, 1):
        for column in (0, 1):
            assert np.array_equal(big[row::2, column::2], plan)
    assert big.std() == pytest.approx(plan.std(), rel=1e-6)
    # drawn at the plan's own size, nothing is different, and an RGB picture gets the same on each channel
    assert np.array_equal(grain(np.zeros(PLAN[::-1], np.float32), 8, 3, plan=PLAN[::-1]), plan)
    rgb = grain(np.zeros((1216, 2160, 3), np.float32), 8, 3, plan=PLAN[::-1])
    assert np.array_equal(rgb[..., 0], big) and np.array_equal(rgb[..., 2], big)


def test_a_smooth_field_has_the_grid_of_the_plan():
    plan = lowfreq(PLAN[::-1], 40, 7, 5)
    big = lowfreq(TWICE[::-1], 80, 14, 5, grid=Scale.between(PLAN, TWICE).coarse(40))

    # the same random values, enlarged: twice as strong, and as smooth
    assert Scale.between(PLAN, TWICE).coarse(40) == (17, 29)
    half = np.asarray(Image.fromarray(big).resize(PLAN, Image.Resampling.BOX))
    assert np.abs(half - 2 * plan).max() < 0.3
    assert np.abs(big).max() == pytest.approx(14, rel=1e-3)
    assert np.abs(plan).max() == pytest.approx(7, rel=1e-3)
    # A grid of its own is another field, which is what a 1 px difference in a size would make of it
    assert not np.allclose(lowfreq(TWICE[::-1], 80, 14, 5, grid=(18, 29))[::2, ::2], plan * 2, atol=1)


def test_the_sticker_edge_grown_at_twice_the_size_is_the_plans_edge(person_alpha):
    alpha = person_alpha(*PLAN)
    big_alpha = np.kron(alpha, np.ones((2, 2), np.float32))
    grid = Scale.between(PLAN, TWICE).coarse(40)

    edge = dilate(alpha, 22, wobble=7, seed=2)
    bigger = dilate(big_alpha, 44, wobble=14, seed=2, scale=80, grid=grid)

    half = bigger.reshape(608, 2, 1080, 2).mean(axis=(1, 3))
    assert np.abs(half - edge).mean() < 0.01
    assert abs(float((half > 0.5).sum()) - float((edge > 0.5).sum())) / float((edge > 0.5).sum()) < 0.01


# --- the matte at another size -----------------------------------------------------------------------


def test_the_cleanup_of_a_matte_is_the_same_cleanup_at_twice_the_size(person_alpha):
    plan = person_alpha(*PLAN)
    plan[20:26, 20:26] = 1.0  # a speck of 6 x 6 px, off the person
    plan[300:305, 540:545] = 0.0  # a hole of 5 x 5 px, in the head
    big = np.kron(plan, np.ones((2, 2), np.float32))

    cleaned = matte.clean_alpha(plan)
    cleaned_big = matte.clean_alpha(big, 2.0)

    # the speck is gone and the hole is filled, at both sizes (a speck of 12 x 12 and a hole of 10 x 10
    # are the same share of the picture and of the person: they are cells of 8 px, not of 4)
    assert cleaned[20:26, 20:26].max() == 0 and cleaned_big[40:52, 40:52].max() == 0
    assert cleaned[300:305, 540:545].min() == 1.0 and cleaned_big[600:610, 1080:1090].min() == 1.0
    half = cleaned_big.reshape(608, 2, 1080, 2).mean(axis=(1, 3))
    assert np.abs(half - cleaned).mean() < 0.005
    # The cells are 4 px at the plan and never smaller, and k times that at a bigger size
    plain = matte.clean_alpha(plan, 1.0)
    assert np.array_equal(plain, cleaned)


def test_load_alpha_at_another_size(plate_maker, tmp_path):
    folder, project = plate_maker(tmp_path / "p", BUILTINS["bedroom-pop"])
    plan = matte.load_alpha(folder, project, 60)

    big = matte.load_alpha(folder, project, 60, size=TWICE)

    assert big.shape == (1216, 2160) and big.dtype == np.float32 and 0 <= big.min() and big.max() <= 1
    share = lambda alpha: float((alpha > 0.5).mean())  # noqa: E731
    assert share(big) == pytest.approx(share(plan), rel=0.01)
    # at the plan's own size it is the very same alpha
    assert np.array_equal(matte.load_alpha(folder, project, 60, size=PLAN), plan)

    # matte.edge is in px of the plan, and so many more px at a bigger size: shrinking by 3 takes the
    # same share off the person at both sizes
    project["matte"]["edge"] = -3
    shrunk = matte.load_alpha(folder, project, 60)
    shrunk_big = matte.load_alpha(folder, project, 60, size=TWICE)
    assert share(shrunk) < 0.99 * share(plan)
    assert share(shrunk_big) == pytest.approx(share(shrunk), rel=0.01)

    # a cutout of the bigger size of its own is read as it is, from its path
    own = tmp_path / "own.png"
    nobody = Image.new("L", TWICE, 0)
    nobody.save(own)
    assert not matte.load_alpha(folder, project, 60, size=TWICE, path=own).any()

    # and one of another size is refused, with a sentence, not enlarged to fit: it was not cut from the
    # frame that the card is drawn on (the project's own matte is for the plan, and is enlarged)
    with pytest.raises(ValueError, match=r"The matte is 1080x608, but the card is 2160x1216\."):
        matte.load_alpha(folder, project, 60, size=TWICE, path=folder / "matte" / "000060.png")


def test_the_edge_is_a_whole_number_of_px_at_any_size(plate_maker, tmp_path, monkeypatch):
    # matte.edge is in px of the plan, and round(edge * k) of them in a picture k times as big (not a
    # fraction of a px: a blur of 0.3 px and a dilation by 3.6 are not what the studio showed)
    folder, project = plate_maker(tmp_path, BUILTINS["bedroom-pop"])
    seen = []
    real = matte.adjust_edge
    monkeypatch.setattr(matte, "adjust_edge", lambda alpha, px: seen.append(px) or real(alpha, px))

    for edge, size, wanted in (
        (-3, (1296, 730), -4), (3, (1296, 730), 4), (2, (1296, 730), 2), (-3, TWICE, -6), (1, (1920, 1080), 2),
        (-1, (1280, 720), -1), (0, (1920, 1080), 0), (4, (3840, 2160), 14),
    ):
        project["matte"]["edge"] = edge
        matte.load_alpha(folder, project, 60, size=size)
        assert seen[-1] == wanted and isinstance(seen[-1], int), (edge, size, seen[-1])
    # at the plan it is the setting itself
    project["matte"]["edge"] = -2
    matte.load_alpha(folder, project, 60)
    assert seen[-1] == -2


# --- the card ---------------------------------------------------------------------------------------


def test_a_stage_is_kept_for_its_size(plate_maker, tmp_path, monkeypatch):
    folder, project = plate_maker(tmp_path, BUILTINS["bedroom-pop"])
    card.clear_caches()
    composed = []
    real = engine.compose_stage

    def counting(recipe, rgb, alpha, variant, scale=IDENTITY):
        composed.append((rgb.shape[:2], variant))
        return real(recipe, rgb, alpha, variant, scale)

    monkeypatch.setattr(engine, "compose_stage", counting)
    size = (1296, 730)

    plan = np.asarray(card.render_card(folder, project))
    bigger = np.asarray(card.render_card(folder, project, size=size))
    assert plan.shape == (608, 1080, 3) and bigger.shape == (730, 1296, 3)
    assert composed == [((608, 1080), 0), ((730, 1296), 0)]

    # Each size has its own stage, so that asking for the other size is not given the wrong picture,
    # and asking again, with other notes on, draws only the notes
    again = np.asarray(card.render_card(folder, project, frozenset({"bottom"})))
    other = np.asarray(card.render_card(folder, project, frozenset({"bottom"}), size=size))
    assert again.shape == plan.shape and other.shape == bigger.shape
    assert len(composed) == 2
    card.render_card(folder, project, variant=1, size=size)
    assert composed[-1] == ((730, 1296), 1) and len(composed) == 3
    assert np.array_equal(np.asarray(card.render_card(folder, project)), plan)  # still the plan's own


def test_a_card_can_be_drawn_from_the_pictures_cut_at_its_size(plate_maker, tmp_path):
    folder, project = plate_maker(tmp_path / "p", BUILTINS["bedroom-pop"])
    with Image.open(folder / "frames" / "000060.jpg") as frame:
        frame.convert("RGB").resize(TWICE, Image.Resampling.LANCZOS).save(tmp_path / "full.jpg", quality=97)
    with Image.open(folder / "matte" / "000060.png") as cutout:
        cutout.convert("L").resize(TWICE, Image.Resampling.BILINEAR).save(tmp_path / "full.png")
    source = card.FullFreeze(tmp_path / "full.jpg", tmp_path / "full.png")

    from_source = card.render_card(folder, project, size=TWICE, source=source)
    enlarged = card.render_card(folder, project, size=TWICE)

    # what the project has, enlarged, and what was cut at that size, when they are the same pictures,
    # make nearly the same card (the one is a JPEG of the other, and the matte is cleaned again)
    assert from_source.size == TWICE
    assert np.abs(np.asarray(from_source, np.int16) - np.asarray(enlarged, np.int16)).mean() < 2

    # pictures of another size than the card's are refused, not enlarged: the matte and the frame are each
    # looked at, so that the one of the right size does not let the other through
    plan_frame, plan_matte = folder / "frames" / "000060.jpg", folder / "matte" / "000060.png"
    with pytest.raises(ValueError, match=r"The matte is 1080x608, but the card is 2160x1216"):
        card.render_card(folder, project, size=TWICE, source=card.FullFreeze(tmp_path / "full.jpg", plan_matte))
    with pytest.raises(ValueError, match=r"The frame is 1080x608, but the card is 2160x1216"):
        card.render_card(folder, project, size=TWICE, source=card.FullFreeze(plan_frame, tmp_path / "full.png"))


def test_a_card_has_the_plans_shape(plate_maker, tmp_path):
    folder, project = plate_maker(tmp_path, BUILTINS["bedroom-pop"])

    # a size that is not the plan's shape is refused: the notes would be drawn out of place
    with pytest.raises(ValueError, match=r"1080x1920 is not the shape of the project's 1080x608"):
        card.render_card(folder, project, size=(1080, 1920))
    # but the pixel that rounding to an even size costs is not a different shape
    assert card.render_card(folder, project, size=(1920, 1080)).size == (1920, 1080)
    assert card.render_card(folder, project, size=(1296, 730)).size == (1296, 730)


def test_the_flash_and_the_boil_are_drawn_at_any_size(plate_maker, tmp_path):
    folder, project = plate_maker(tmp_path, BUILTINS["scrapbook"])
    size = (1296, 730)

    plain = np.asarray(card.render_card(folder, project, size=size), np.float32) / 255
    flashed = np.asarray(card.render_card(folder, project, flash=0.4, size=size), np.float32) / 255
    assert np.abs(flashed - (plain * 0.6 + 0.4)).max() <= 1.5 / 255
    # The three boil pictures differ (the dots slide, the paper's grain is another) and are each
    # the same picture every time they are drawn
    boil = [np.asarray(card.render_card(folder, project, variant=v, size=size)) for v in range(3)]
    assert not np.array_equal(boil[0], boil[1]) and not np.array_equal(boil[1], boil[2])
    assert np.array_equal(boil[1], np.asarray(card.render_card(folder, project, variant=1, size=size)))


def test_the_extras_are_where_the_plan_put_them(plate_maker, tmp_path):
    # Tape and stars are decided at the plan (what is left out, where a star lands), and drawn
    # bigger: so the overlay at twice the size is the plan's, enlarged. A star that barely fits at the
    # plan is not moved at another size because a box was rounded to a pixel.
    folder, project = plate_maker(tmp_path, BUILTINS["scrapbook"])
    recipe = {**BUILTINS["scrapbook"], "extras": {"tape": True, "stars": True, "grain": False}}
    boxes = card.elements(folder, project)
    context = card._context(card._matte(folder, project), recipe, card._keepout(project))

    def overlay(size):
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        scale = Scale.between(PLAN, size)
        notes.draw_extras(layer, boxes, recipe, card.ALL_VISIBLE, context.face, (), scale)
        return layer.getchannel("A")

    plan, big = overlay(PLAN), overlay(TWICE)
    assert np.asarray(plan).any(), "the tape and the stars are drawn"
    half = np.asarray(big.resize(PLAN, Image.Resampling.BOX), np.float32)
    # of 255: the same pieces, in the same places
    assert np.abs(half - np.asarray(plan, np.float32)).mean() < 1.0


# What a size that is not a multiple of the plan costs, as the mean difference in levels of the card shrunk back
# against the plan's, by style. It is the 3 allowed at twice the size, a little more: a lattice of whole pixels
# (the dots, the scanlines) cannot land on the plan's at 1.78 or 1.19 times its size, and Press Start 2P and the
# sizes of the text snap to grids of their own. The four styles measure 2.7 to 3.8, and 6.1 to 6.2 for Player
# Select, whose scanlines and pixel notes both have a grid of their own. The limits are what a render of a clip
# of 1920x1080 or 1280x720, the two usual sizes, may not be worse than.
NOT_A_MULTIPLE = {"bedroom-pop": 4, "comic-pop": 4, "scrapbook": 4, "player-select": 7}


@pytest.mark.parametrize("style_id", list(NOT_A_MULTIPLE))
def test_a_card_drawn_at_the_usual_sizes_is_the_card_at_the_plan(plate_maker, tmp_path, style_id):
    folder, project = plate_maker(tmp_path, BUILTINS[style_id])
    plan = card.render_card(folder, project)

    for size in ((1920, 1080), (1280, 720)):
        big = card.render_card(folder, project, size=size)

        assert big.size == size
        assert shrunk_difference(big, plan) < NOT_A_MULTIPLE[style_id], (style_id, size)


def test_every_option_draws_at_a_size_that_is_not_a_multiple(plate_maker, tmp_path):
    # Every id of every layer, and all three extras, at 1.2 times the plan: nothing assumes a whole k.
    # (How near the plan's card it is, is pinned for the four styles by the test above.)
    from test_styles import seeded_recipes

    recipes = seeded_recipes(6)
    for number, recipe in enumerate(recipes):
        if number % 2:
            recipe["extras"] = {"tape": True, "stars": True, "grain": True}
    for layer in ("bg", "edge", "notes", "arrows", "hero"):
        assert {recipe[layer] for recipe in recipes} == set(OPTIONS[layer]), layer

    for number, recipe in enumerate(recipes):
        folder, project = plate_maker(tmp_path / str(number), recipe)
        big = card.render_card(folder, project, variant=number % 3, size=(1296, 730))
        assert big.size == (1296, 730)
        assert np.asarray(big).std() > 20, recipe["id"]  # a picture, not a blank


# --- the notes' text ----------------------------------------------------------------------------------


# One recipe for each notes style: the four ready-made ones set the marker, comic, hand and pixel notes, and the
# bold labels are on none of them
NOTES_STYLES = {
    "marker": BUILTINS["bedroom-pop"],
    "comic": BUILTINS["comic-pop"],
    "hand": BUILTINS["scrapbook"],
    "pixel": BUILTINS["player-select"],
    "label": {**BUILTINS["bedroom-pop"], "id": "label-notes", "notes": "label"},
}
# The sizes of a render: twice, 1920x1080 and 4K, 1296x730 (1.2 times), 1280x720 and 1440x810, which take the
# most off the marker notes (3.4%) and 1296x730 off the pixel ones (3.4%)
FIT_SIZES = [TWICE, (1920, 1080), (3840, 2160), (1296, 730), (1280, 720), (1440, 810)]


@pytest.mark.parametrize("notes_style", list(NOTES_STYLES))
def test_note_text_drawn_bigger_stays_in_its_box(plate_maker, tmp_path, notes_style):
    # A text is not as wide at k times the size as k times its width: every advance of a glyph is
    # rounded to a pixel. A marker or a pixel note has nothing drawn behind it, and is set from the
    # corner of its box, so it would come out a few px past the room the layout kept for it, and the
    # others are centered on a fill that is as wide as the text and its padding. The size is brought
    # down a pixel at a time, all four notes alike, until it is not, by 3.5% at the most and never by more
    # than 5%; the comic, hand and label notes, which have room to spare, are not brought down at all.
    recipe = NOTES_STYLES[notes_style]
    folder, project = plate_maker(tmp_path, recipe)
    layout = card.elements(folder, project)
    # (the stats panel of Player Select is in the layout under "tr", and it is not a note)
    panel = "tr" if recipe["hero"] == "stats" else None
    keys = [key for key in notes.NOTE_KEYS if key in layout and key != panel]
    assert len(keys) >= 3
    assert len({layout[key].size for key in keys}) == 1

    for size in FIT_SIZES:
        scale = Scale.between(PLAN, size)
        style = notes._at(notes.note_style(recipe), scale)

        drawn = notes._fitted(notes.scaled_layout(layout, scale), style, recipe, scale)

        assert len({drawn[key].size for key in keys}) == 1, size
        exact = round(layout[keys[0]].size * scale.k)
        assert round(0.95 * exact) <= drawn[keys[0]].size <= exact, (size, drawn[keys[0]].size, exact)
        assert exact - drawn[keys[0]].size <= math.ceil(0.035 * exact), size  # 3.5% at the most, in whole px
        if notes_style in ("comic", "hand", "label"):
            assert drawn[keys[0]].size == exact, size
        for key in keys:
            face = font(style.font, drawn[key].size)
            for line in drawn[key].lines:
                assert face.getlength(line) <= drawn[key].w, (size, key, line, drawn[key].w)
                assert face.getbbox(line, anchor="ls")[2] <= drawn[key].w + 1, (size, key, line)

    # at the plan, nothing is changed
    plan_style = notes.note_style(recipe)
    assert notes._fitted(layout, plan_style, recipe, IDENTITY) is layout


def test_a_note_is_never_brought_down_by_more_than_5_percent():
    # A text that is far too wide for its box is not shrunk to nothing to fit: the size stops at 95% of what k
    # makes it, and the line is wider than its box, which is what a text that does not fit its note looks like.
    recipe = BUILTINS["bedroom-pop"]
    layout = {"tl": NoteBox("tl", ["WWWWWWWWWWWWWWWWWWWWWWWWW"], 50, 10, 10, 200, 60, None)}
    scale = Scale.between(PLAN, TWICE)
    style = notes._at(notes.note_style(recipe), scale)
    scaled = notes.scaled_layout(layout, scale)
    assert scaled["tl"].size == 100

    drawn = notes._fitted(scaled, style, recipe, scale)

    assert drawn["tl"].size == 95
    assert font(style.font, 95).getlength(drawn["tl"].lines[0]) > drawn["tl"].w


# The game HUD (Player Select's colors, with the HUD for its hero), on the 16:9 plan and on the 9:16 one
HUD = {**BUILTINS["player-select"], "id": "hud", "hero": "hud"}
HUD_SIZES = {
    False: [TWICE, (1920, 1080), (1280, 720), (3840, 2160)],
    True: [(1440, 2560), (2160, 3840)],
}


@pytest.mark.parametrize("portrait", [False, True])
def test_the_huds_labels_stay_in_their_boxes(plate_maker, tmp_path, portrait):
    # Press Start 2P is only crisp on its 8 px grid, so a label's size snaps to a multiple of 8, and the
    # label is then as wide as its text is at that size: not k times the width it had at the plan, which
    # it overflowed by 12% at 1920x1080 ("PRESS START" is 352 px there, in a box of 313). Its box is the
    # label as it is drawn. "PLAYER 1" keeps its left corner, and "PRESS START" its right edge, which is
    # where the layout set it, against the room's.
    folder, project = plate_maker(tmp_path, HUD, portrait=portrait)
    layout = card.elements(folder, project)
    assert {"hud_top", "hud_bottom"} <= set(layout)
    plan = (project["work"]["width"], project["work"]["height"])
    assert plan == ((1080, 1920) if portrait else PLAN)

    for size in HUD_SIZES[portrait]:
        scale = Scale.between(plan, size)
        scaled = notes.scaled_layout(layout, scale)
        for key in ("hud_top", "hud_bottom"):
            box = scaled[key]
            face = font("pixel", box.size)
            assert box.size % 8 == 0 and box.h == box.size, (size, key)
            assert box.w == math.ceil(face.getlength(box.lines[0])), (size, key)
            assert 0 <= box.x and box.x + box.w <= size[0], (size, key, box.x, box.w)
            assert 0 <= box.y and box.y + box.h <= size[1], (size, key, box.y, box.h)
            # What is drawn is in the box (the hard shadow falls a little right and down of it)
            _, right, down = notes._pixel_shadow(scale)
            layer = Image.new("RGBA", size, (0, 0, 0, 0))
            notes._hud(layer, {key: box}, project["copy"], HUD, {"top", "bottom"}, scale)
            left_edge, top_edge, right_edge, bottom_edge = layer.getchannel("A").getbbox()
            assert box.x <= left_edge and right_edge <= box.x + box.w + math.ceil(right), (size, key)
            assert box.y <= top_edge and bottom_edge <= box.y + box.h + math.ceil(down), (size, key)
        top, bottom = scaled["hud_top"], scaled["hud_bottom"]
        assert top.x == round(layout["hud_top"].x * scale.k), size
        was = layout["hud_bottom"]
        assert bottom.x + bottom.w == round((was.x + was.w) * scale.k), size
        # (the top of each is where the plan's is, in the picture's px)
        assert top.y == round(layout["hud_top"].y * scale.ky) and bottom.y == round(was.y * scale.ky), size


def test_a_scaled_layout_keeps_what_the_boxes_say():
    arrow = ((100, 200), (130, 230), (160, 240))
    layout = {
        "tl": NoteBox("tl", ["A", "B"], 48, 43, 43, 300, 100, arrow, clash=True, cut=True),
        "hud_top": NoteBox("hud_top", ["PLAYER 1"], 24, 43, 40, 192, 24, None),
        "hud_bottom": NoteBox("hud_bottom", ["PRESS START"], 16, 700, 560, 176, 16, None),
    }

    scaled = notes.scaled_layout(layout, Scale.between(PLAN, (3840, 2160)))

    k, ky = 3840 / 1080, 2160 / 608
    tl = scaled["tl"]
    assert (tl.x, tl.y, tl.w, tl.h) == (round(43 * k), round(43 * ky), round(300 * k), round(100 * ky))
    assert tl.size == 171
    assert tl.lines == ["A", "B"] and tl.clash and tl.cut and tl.key == "tl"
    assert tl.arrow == tuple((x * k, y * ky) for x, y in arrow)
    # the HUD's labels are Press Start 2P, which is crisp only on a multiple of 8, and a label's box is the label
    # as it is drawn at that size: a character is one em wide, and it is one em tall. "PLAYER 1" keeps its left
    # corner and "PRESS START" its right edge.
    top, bottom = scaled["hud_top"], scaled["hud_bottom"]
    assert (top.size, bottom.size) == (88, 56)
    assert (top.x, top.y, top.w, top.h) == (round(43 * k), round(40 * ky), 8 * 88, 88)
    assert (bottom.y, bottom.w, bottom.h) == (round(560 * ky), 11 * 56, 56)
    assert bottom.x + bottom.w == round((700 + 176) * k)
    assert top.arrow is None and bottom.arrow is None
    assert layout["tl"].x == 43 and layout["tl"].size == 48  # the layout it was made from is not touched
