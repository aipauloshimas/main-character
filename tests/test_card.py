"""Tests for the card: the stage with the notes and arrows over it, drawn from conftest's fake_project
(a synthetic clip frame and person, the sample copy and the Bedroom Pop recipe).

The card is the clip's own frame (Amendment 1), 1080x608 at the plan for a 16:9 clip, with nothing
around it: no bars, no gradient and no title. A note's box is in px of the card, so a box is looked up
in the card where it says. A 9:16 card is 1080x1920 and keeps out of the title room and the Reels safe
area; those tests use tests/fixtures/vertical_alpha.png, the matte of a vertical photo. Nothing
here depends on how the layout chose a place for a note, only on what the card draws where the layout
says. (How a card is drawn bigger, and the approved look, are test_scale's.)
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mclib import layout, matte
from mclib.analyze import person_boxes
from mclib.card import ALL_VISIBLE, _keepout, elements, render_card
from mclib.raster import from_image, quad_points, to_image
from mclib.styles import engine, notes
from mclib.styles.recipes import BUILTINS

INK = (0xF7, 0xE8, 0xB0)  # the ink of the Magenta palette
# What a 9:16 card keeps out of (ruling S1, amended), in px of its 1080x1920 plan, written out here: the title
# room, the top 15.6% of the height, which is where the person adds the title in the app; the bottom band, from
# 78.1% of the height down, where Instagram and TikTok draw the caption, the username and the audio line; and
# the right rail of the like, comment and share buttons, from 87% of the width, from half of the height down to
# the band. A card of any other shape has none.
TITLE_ROOM, BOTTOM_BAND, RIGHT_RAIL = (0, 0, 1080, 300), (0, 1500, 1080, 1920), (940, 960, 1080, 1500)
VERTICAL_ALPHA = Path(__file__).parent / "fixtures" / "vertical_alpha.png"  # a vertical photo, RVM, 1080x1920


def region(box):
    """The slice of the card under a note's box (px of the card)."""
    return slice(box.y, box.y + box.h), slice(box.x, box.x + box.w)


def hits(box, rect):
    """Whether a NoteBox shares any pixel with an (x0, y0, x1, y1) rectangle."""
    return box.x < rect[2] and box.x + box.w > rect[0] and box.y < rect[3] and box.y + box.h > rect[1]


def near(pixels, color, tolerance=8):
    """How many pixels of an (h, w, 3) uint8 picture are within `tolerance` levels of a color on
    every channel."""
    return int((np.abs(pixels.astype(int) - np.array(color)) <= tolerance).all(axis=-1).sum())


def as_array(image):
    return np.asarray(image)


def inside(point, rect):
    """Whether a point is in an (x0, y0, x1, y1) rectangle, its edges included."""
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def test_render_card_is_the_stage_with_the_notes_over_it(fake_project):
    folder, project = fake_project
    image = render_card(folder, project)
    assert image.size == (1080, 608) and image.mode == "RGB"
    pixels = as_array(image)

    # Nothing is around the stage: no black bars and no title. The picture goes to every edge of the card.
    assert pixels.mean() > 60
    for edge in (pixels[:6], pixels[-6:], pixels[:, :6], pixels[:, -6:]):
        assert edge.mean() > 30

    # The card is the stage, and then the notes over it, and nothing else: the stage as the engine makes
    # it, and the notes drawn on a layer of the card's size, put together here by hand, is the same
    # card to the pixel, for the notes that are on, the boil picture and the flash.
    recipe, copy = project["style"], project["copy"]
    alpha = matte.load_alpha(folder, project, 60)
    with Image.open(folder / "frames" / "000060.jpg") as frame:
        rgb = from_image(frame.convert("RGB"))
    boxes = engine.style_layout(recipe, alpha, copy, project["positions"])
    head = person_boxes(alpha)["head"]
    states = ((ALL_VISIBLE, 0, 0.0), (frozenset({"bottom"}), 1, 0.0), (ALL_VISIBLE, 2, 0.4))
    for visible, variant, flash in states:
        by_hand = to_image(engine.compose_stage(recipe, rgb, alpha, variant)).convert("RGBA")
        layer = Image.new("RGBA", (1080, 608), (0, 0, 0, 0))
        engine.draw_overlay(layer, recipe, boxes, copy, visible, head)
        by_hand.alpha_composite(layer)
        by_hand = by_hand.convert("RGB")
        if flash:
            by_hand = Image.blend(by_hand, Image.new("RGB", by_hand.size, "white"), flash)
        now = render_card(folder, project, visible, variant, flash)
        assert np.array_equal(as_array(by_hand), as_array(now)), (sorted(visible), variant, flash)

    # The picture of the freeze frame is not the card (the card has been through the style)
    first = as_array(Image.open(folder / "frames" / "000060.jpg").convert("RGB"))
    assert not np.array_equal(pixels, first)

    # Pictures of another size than the project is cut at are refused with a way out, not a broadcasting error
    wrong = deepcopy(project)
    wrong["work"] = {"width": 1080, "height": 1920}
    with pytest.raises(ValueError, match="matte is 1080x608, but the project is cut at 1080x1920"):
        render_card(folder, wrong)
    Image.new("RGB", (1920, 1080)).save(folder / "frames" / "000060.jpg")
    with pytest.raises(ValueError, match="frame is 1920x1080, but the project is cut at 1080x608"):
        render_card(folder, project)


def test_render_card_visibility_changes_quadrants(fake_project, monkeypatch):
    folder, project = fake_project
    loaded, composed = [], []
    real_load, real_compose = matte.load_alpha, engine.compose_stage

    def counting_load(project_dir, current, frame, **kwargs):
        loaded.append(frame)
        return real_load(project_dir, current, frame, **kwargs)

    def counting_compose(recipe, rgb, alpha, variant, *rest):
        composed.append(variant)
        return real_compose(recipe, rgb, alpha, variant, *rest)

    monkeypatch.setattr(matte, "load_alpha", counting_load)
    monkeypatch.setattr(engine, "compose_stage", counting_compose)

    boxes = elements(folder, project)
    assert sorted(boxes) == ["bl", "br", "tl", "tr"]
    for key, box in boxes.items():
        # In px of the card (0..1080 by 0..608), at the corner each is named for.
        assert 0 <= box.x and box.x + box.w <= 1080 and 0 <= box.y and box.y + box.h <= 608, key
        assert (box.y < 304) == (key[0] == "t") and (box.x < 540) == (key[1] == "l"), key
        assert not box.clash, key
    # The lines are the cased ones the layout made, which the card draws as they are.
    assert " ".join(boxes["bl"].lines) == "BUT OVERTHINKS EVERY 6-SECOND REEL"
    # The boxes are copies: changing them changes neither the next call nor the card.
    boxes["bl"].lines.append("CHANGED")
    boxes["bl"].x = 0
    assert elements(folder, project)["bl"].lines != boxes["bl"].lines
    assert elements(folder, project)["bl"].x != 0
    boxes = elements(folder, project)

    # Cards that differ only in what is visible cost one matte and one stage, and no more.
    everything = as_array(render_card(folder, project, ALL_VISIBLE))
    bottom = as_array(render_card(folder, project, frozenset({"bottom"})))
    top = as_array(render_card(folder, project, frozenset({"top"})))
    both = as_array(render_card(folder, project, frozenset({"bottom", "top"})))
    nothing = as_array(render_card(folder, project, frozenset()))
    arrows = as_array(render_card(folder, project, frozenset({"arrows"})))
    assert loaded == [60] and composed == [0]

    # Another boil variant, another recipe (the `style` override) and another edge setting each
    # make a stage of their own, and asking for any of them again does not make it twice.
    render_card(folder, project, ALL_VISIBLE, variant=1)
    assert composed == [0, 1]
    darker = {**project["style"], "palette": {**project["style"]["palette"], "dark": "#102030"}}
    assert not np.array_equal(as_array(render_card(folder, project, style=darker)), everything)
    assert composed == [0, 1, 0]
    softer = deepcopy(project)
    softer["matte"]["edge"] = 2
    render_card(folder, softer)
    assert loaded == [60, 60] and composed == [0, 1, 0, 0]
    for _ in range(2):
        render_card(folder, project, variant=1)
        render_card(folder, project, style=darker)
        render_card(folder, softer)
    assert loaded == [60, 60] and composed == [0, 1, 0, 0]

    # "bottom" shows the bottom notes only, and "top" the top ones: the ink of a note is there when
    # its group is on and gone when it is not, and a corner is the same whether or not the other
    # group and the arrows are on. (The ink is counted against the card without notes, because the
    # synthetic person's pixels can happen to be near the ink color.)
    for key in ("bl", "br"):
        area = region(boxes[key])
        assert near(everything[area], INK) - near(nothing[area], INK) > 100, key
        assert np.array_equal(everything[area], bottom[area]), key
        assert np.array_equal(top[area], nothing[area]), key
    for key in ("tl", "tr"):
        area = region(boxes[key])
        assert near(everything[area], INK) - near(nothing[area], INK) > 100, key
        assert np.array_equal(everything[area], top[area]), key
        assert np.array_equal(bottom[area], nothing[area]), key

    # "arrows" draws every arrow and nothing else, and none of them touches a note's box. Each one
    # is drawn from the coordinates of the layout, so its tip is ink at that point of the card, when
    # "arrows" is on and not otherwise, whichever notes are on.
    changed = (arrows != nothing).any(axis=2)
    assert changed.any() and (changed.sum() < 20_000)
    for box in boxes.values():
        assert not changed[region(box)].any()
    drawn = [(key, box.arrow) for key, box in boxes.items() if box.arrow is not None]
    assert len(drawn) >= 3, "the synthetic person has room for an arrow at (nearly) every note"
    for key, (start, control, end) in drawn:
        x, y = end
        tip = (slice(y - 1, y + 2), slice(x - 1, x + 2))
        for card_with_arrows in (arrows, everything):
            assert near(card_with_arrows[tip], INK, 8) >= 5, key
        for card_without in (nothing, bottom, top, both):
            assert near(card_without[tip], INK, 8) == 0, key

    # The marker's glow is the mockup's text-shadow: a tight dark glow and a wide one under the
    # lettering, and a hard dark line 3 px below it. Each of the four notes is set that way.
    drawn = []
    real_text = notes.draw_text

    def recording(*args, **kwargs):
        drawn.append(kwargs)
        return real_text(*args, **kwargs)

    with monkeypatch.context() as patch:  # only this one, the counting wrappers above stay
        patch.setattr(notes, "draw_text", recording)
        render_card(folder, project, ALL_VISIBLE)
    dark = project["style"]["palette"]["dark"]
    assert len(drawn) == 4
    assert all(call["glow"] == [(dark, 3), (dark, 10)] and call["shadow"] == (dark, 0, 3) for call in drawn)

    # With the arrows layer set to "none" the layout keeps no room for arrows, and "arrows" draws nothing.
    unarrowed = {**project, "style": {**project["style"], "arrows": "none"}}
    assert all(box.arrow is None for box in elements(folder, unarrowed).values())
    assert np.array_equal(
        as_array(render_card(folder, unarrowed)),
        as_array(render_card(folder, unarrowed, frozenset({"bottom", "top"}))),
    )

    # A group that is not bottom, top or arrows is refused rather than silently drawn as nothing.
    with pytest.raises(ValueError, match="bottm"):
        render_card(folder, project, frozenset({"bottm"}))
    # So is a recipe that is not valid. (test_styles checks the layers that are not built yet.)
    with pytest.raises(ValueError, match="^bg"):
        render_card(folder, project, style={**project["style"], "bg": "glitter"})
    with pytest.raises(ValueError, match="^hero"):
        elements(folder, {**project, "style": {**project["style"], "hero": "glitter"}})


def test_render_card_flash(fake_project):
    folder, project = fake_project
    plain = as_array(render_card(folder, project)).astype(np.float32) / 255
    for level in (0.85, 0.40):
        flashed = as_array(render_card(folder, project, flash=level)).astype(np.float32) / 255
        # out = card * (1 - f) + white * f, on every pixel
        assert np.abs(flashed - (plain * (1 - level) + level)).max() <= 1.5 / 255
        assert flashed.min() >= level - 1.5 / 255
    assert as_array(render_card(folder, project, flash=0.85)).mean() / 255 > 0.8
    unflashed = as_array(render_card(folder, project, flash=0.0))
    assert np.array_equal(unflashed, as_array(render_card(folder, project)))
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="flash"):
            render_card(folder, project, flash=bad)


def test_card_renders_with_empty_alpha(fake_project):
    folder, project = fake_project
    with_person = as_array(render_card(folder, project))  # the matte is cached after this
    Image.fromarray(np.zeros((608, 1080), np.uint8)).save(folder / "matte" / "000060.png")

    # The matte file changed, so the card is drawn from the new one, and nobody in it is no error.
    image = render_card(folder, project)
    assert image.size == (1080, 608)
    pixels = as_array(image)
    assert not np.array_equal(pixels, with_person)
    boxes = elements(folder, project)
    assert sorted(boxes) == ["bl", "br", "tl", "tr"]
    assert all(box.arrow is None for box in boxes.values()), "nobody to point at"

    # The notes are drawn all the same, around the middle top where a face would be, and only the
    # room is left behind them: no sticker edge, no person.
    bare = as_array(render_card(folder, project, frozenset()))
    for key, box in boxes.items():
        assert near(pixels[region(box)], INK) > 100, key
        assert near(bare[region(box)], INK) == 0, key
    assert near(pixels, (0xF5, 0xDF, 0x98), 6) == 0
    assert not matte.load_alpha(folder, project, 60).any(), "the matte is empty once cleaned up"


# --- a tall card: the title room and the Reels safe area ------------------------------------------


def vertical_project(fake_project, folder):
    """A 9:16 project folder on a vertical photo: its matte (RVM, 1080 x 1920) and the fixture's
    frame stretched to the plan. Returns the project that goes with it."""
    source, project = fake_project
    (folder / "frames").mkdir(parents=True)
    (folder / "matte").mkdir()
    with Image.open(source / "frames" / "000060.jpg") as frame:
        frame.convert("RGB").resize((1080, 1920)).save(folder / "frames" / "000060.jpg", quality=95)
    with Image.open(VERTICAL_ALPHA) as person:
        person.convert("L").save(folder / "matte" / "000060.png")
    tall = deepcopy(project)
    tall["layout"], tall["work"] = "portrait", {"width": 1080, "height": 1920}
    return tall


def test_a_tall_card_has_a_title_room_and_the_safe_area_and_no_other_shape_does(fake_project):
    _, project = fake_project
    assert _keepout(project) == ()  # 16:9, which is the whole of the card

    def planned(width, height):
        return _keepout({**project, "work": {"width": width, "height": height}})

    assert planned(1080, 1920) == (TITLE_ROOM, BOTTOM_BAND, RIGHT_RAIL)
    # The shares are of the plan's size: 15.625% down for the title room, the band from 78.125%, the rail
    # from 87.04% across and from half the height. A 9:16 clip at any size has them in the same places.
    assert planned(1080, 1900) == (
        (0, 0, 1080, 297), (0, 1484, 1080, 1900), (940, 950, 1080, 1484))
    # It applies from 1.7 times as tall as it is wide: 1836 is that, exactly, and 1835 is under it
    assert len(planned(1080, 1836)) == 3 and planned(1080, 1835) == ()
    # and no other shape has any: 4:5, 1:1, 4:3 and 16:9 are inside the safe area already
    for shape in ((1080, 1350), (1080, 1080), (1080, 810), (1080, 608)):
        assert planned(*shape) == ()


def test_a_tall_card_stays_out_of_the_title_room(fake_project, tmp_path, person_alpha):
    folder = tmp_path / "vertical"
    project = vertical_project(fake_project, folder)
    zones = (TITLE_ROOM, BOTTOM_BAND, RIGHT_RAIL)
    keepout = _keepout(project)
    assert keepout == zones
    cleaned = matte.load_alpha(folder, project, 60)

    # No note is on the title room and none of them has an arrow that runs into it (sampled as the layout
    # samples it). Without the keep-out the layout does put a note there (the top quadrants begin at the
    # very top), which is what it is for.
    boxes = elements(folder, project)
    assert sorted(boxes) == ["bl", "br", "tl", "tr"]
    assert not any(hits(box, TITLE_ROOM) for box in boxes.values())
    arrows = [box.arrow for box in boxes.values() if box.arrow is not None]
    assert len(arrows) == 2  # (the vertical photo is a big person: only tl and tr are in reach of one)
    assert all(y >= TITLE_ROOM[3] for arrow in arrows for _, y in quad_points(*arrow, 16))
    unguarded = engine.style_layout(project["style"], cleaned, project["copy"], project["positions"])
    assert any(hits(box, TITLE_ROOM) for box in unguarded.values())

    # The heroes keep out of the title room too, and off the head (the face box without its air, which
    # only the panel of stats may use): with the burst, the panel and the HUD on this card, no box enters
    # the room, none covers the head, and no arrow, sampled as the layout samples it, runs into it. (The
    # control: with no keep-out each of them does put a box in the room.)
    heroes = {
        "burst": BUILTINS["comic-pop"],
        "tr": {**BUILTINS["player-select"], "arrows": "curly"},  # the panel of stats is the "tr" box
        "hud_top": {**BUILTINS["player-select"], "hero": "hud", "arrows": "curly"},
    }
    arrowed = {"burst": 1, "tr": 2, "hud_top": 3}  # how many notes of the card have an arrow, with each hero
    for key, hero_recipe in heroes.items():
        shown = elements(folder, {**project, "style": hero_recipe})
        head = engine._tight(engine.matte_context(hero_recipe, cleaned, [TITLE_ROOM])).face
        unkept = engine.style_layout(hero_recipe, cleaned, project["copy"], project["positions"])
        assert any(hits(box, TITLE_ROOM) for box in unkept.values()), key
        assert key in shown, key
        if key == "tr":  # the panel of stats, not a note
            assert shown["tr"].lines == [row for row, _ in notes.stats_rows(project["copy"])]
        assert not any(hits(box, TITLE_ROOM) or hits(box, head) for box in shown.values()), key
        points = [point for box in shown.values() if box.arrow for point in quad_points(*box.arrow, 16)]
        assert len(points) == 16 * arrowed[key] and all(y >= TITLE_ROOM[3] for _, y in points), key

    # An arrow that would climb into the title room. With the person at the top of the frame (a head and
    # shoulders just under the room), a note pinned just under it has an arrow that heads up for the head.
    # The control: the arrow found with no keep-out does enter the room in most of these pins, and with
    # the keep-out it is another one, or none, and none of those enters the room.
    (folder / "high" / "frames").mkdir(parents=True)
    (folder / "high" / "matte").mkdir()
    with Image.open(folder / "frames" / "000060.jpg") as frame:
        frame.save(folder / "high" / "frames" / "000060.jpg", quality=95)
    at_top = np.zeros((1920, 1080), np.uint8)
    at_top[:608] = np.rint(person_alpha(1080, 608) * 255).astype(np.uint8)
    Image.fromarray(at_top).save(folder / "high" / "matte" / "000060.png")
    unkept = engine.matte_context(project["style"], matte.load_alpha(folder / "high", project, 60))
    climbing = []
    for x in range(414, 800, 64):
        for y in (301, 324):
            pinned = {**project, "positions": {**project["positions"], "tl": [x, y]}}
            around = elements(folder / "high", pinned)
            others = [box for key, box in around.items() if key != "tl"]
            run = layout.arrow_for("tl", around["tl"], others, unkept)  # found with no keep-out
            if run is not None and any(point[1] < TITLE_ROOM[3] for point in quad_points(*run, 16)):
                climbing.append(around["tl"].arrow)
    assert len(climbing) >= 7  # of the 14 pins, with nothing to keep the arrow out
    assert any(arrow is not None for arrow in climbing)  # it is found another way, not only dropped
    assert all(
        arrow is None or all(y >= TITLE_ROOM[3] for _, y in quad_points(*arrow, 16)) for arrow in climbing
    )

    # Tape and stars keep out of the title room too. card.render_card gives draw_overlay the context's
    # keep-out with its face box. Drawn with it, nothing of the extras is in the room: the layer is the one
    # with no extras there. Drawn with none (the control), a star lands in the room and the tape of the top
    # notes lies across its edge; and below the room both are still drawn. (On the person at the top of the
    # frame: the vertical photo's head is under the room, where no star goes.)
    high = folder / "high"
    context = engine.matte_context(project["style"], matte.load_alpha(high, project, 60), [TITLE_ROOM])
    bare = {**project["style"], "extras": {"tape": False, "stars": False, "grain": False}}
    boxes = elements(high, {**project, "style": bare})

    def alpha_of(recipe, keep):
        layer = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
        engine.draw_overlay(layer, recipe, boxes, project["copy"], ALL_VISIBLE, context.face, keep)
        return np.asarray(layer)[..., 3]

    base, room = alpha_of(bare, context.keepout), TITLE_ROOM[3]
    for extra in ("stars", "tape"):
        showy = {**bare, "extras": {**bare["extras"], extra: True}}
        free, kept = alpha_of(showy, ()), alpha_of(showy, context.keepout)
        assert (free[:room] != base[:room]).any(), extra  # the control: it is in the room
        assert (kept[:room] == base[:room]).all(), extra
        assert (kept[room:] != base[room:]).any(), extra  # and still drawn under it
    # The same on the card: the rows of the room are the card's without extras, rows under it differ
    both = {**bare, "extras": {"tape": True, "stars": True, "grain": False}}
    card = as_array(render_card(high, project, style=bare))
    with_extras = as_array(render_card(high, project, style=both))
    assert np.array_equal(with_extras[:room], card[:room]) and not np.array_equal(with_extras, card)

    # Pin the top right note into the room: it stays where it was put, and it is flagged, as a note that
    # sits where the title will be would be
    pinned = {**project, "positions": {**project["positions"], "tr": [700, 100]}}
    flagged = elements(folder, pinned)["tr"]
    assert (flagged.x, flagged.y) == (700, 100) and hits(flagged, TITLE_ROOM) and flagged.clash


def test_a_tall_card_stays_inside_the_reels_safe_area(fake_project, tmp_path):
    folder = tmp_path / "vertical"
    project = vertical_project(fake_project, folder)
    zones = (TITLE_ROOM, BOTTOM_BAND, RIGHT_RAIL)
    keepout = _keepout(project)
    assert keepout == zones  # the title room, and then the two boxes of the safe area
    cleaned = matte.load_alpha(folder, project, 60)

    # With each hero (none, the burst, the panel of stats and the HUD) and with the tape and the stars
    # on: no note and no hero box is in the title room, the bottom band or on the rail, no arrow (sampled
    # as the layout samples it) runs into any of them, and neither does a piece of tape or a star.
    extras = {"tape": True, "stars": True, "grain": False}
    recipes = {
        "none": {**BUILTINS["scrapbook"], "extras": extras},
        "burst": {**BUILTINS["comic-pop"], "extras": extras},
        "stats": {**BUILTINS["player-select"], "arrows": "curly", "extras": extras},
        "hud": {**BUILTINS["player-select"], "hero": "hud", "arrows": "curly", "extras": extras},
    }
    free_extras = {}
    for hero, recipe in recipes.items():
        boxes = elements(folder, {**project, "style": recipe})
        assert not any(hits(box, zone) for box in boxes.values() for zone in zones), hero
        points = [point for box in boxes.values() if box.arrow for point in quad_points(*box.arrow, 16)]
        assert len(points) >= 16, hero  # there are arrows to check
        assert not any(inside(point, zone) for point in points for zone in zones), hero

        context = engine.matte_context(recipe, cleaned, list(keepout))
        bare = {**recipe, "extras": {"tape": False, "stars": False, "grain": False}}

        def painted(style, keep):
            layer = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
            engine.draw_overlay(layer, style, boxes, project["copy"], ALL_VISIBLE, context.face, keep)
            return np.asarray(layer)[..., 3]

        base, kept, free = painted(bare, context.keepout), painted(recipe, context.keepout), painted(recipe, ())
        assert (kept != base).any(), hero  # the extras are drawn
        for x0, y0, x1, y1 in zones:
            assert (kept[y0:y1, x0:x1] == base[y0:y1, x0:x1]).all(), hero
            free_extras[hero] = free_extras.get(hero, False) or (free[y0:y1, x0:x1] != base[y0:y1, x0:x1]).any()
    assert any(free_extras.values())  # the control: drawn with no keep-out, a star does land on the rail

    # The HUD's "PRESS START" sits at the bottom right of the safe area, as it does at the bottom right
    # of a landscape card: 43 px (the margin) left of where the notes stop, which is the rail's edge, and
    # 20 px above the bottom band.
    hud = elements(folder, {**project, "style": recipes["hud"]})["hud_bottom"]
    assert (hud.x + hud.w, hud.y + hud.h) == (RIGHT_RAIL[0] - 43, BOTTOM_BAND[1] - 20)
    assert (hud.lines, hud.size) == (["PRESS START"], 16)

    # A note pinned into the bottom band, or onto the rail, stays where it was put and is flagged
    pinned = {**project, "style": recipes["none"], "positions": {"bl": [100, 1600], "br": [900, 1000]}}
    flagged = elements(folder, pinned)
    assert hits(flagged["bl"], BOTTOM_BAND) and flagged["bl"].clash
    assert hits(flagged["br"], RIGHT_RAIL) and flagged["br"].clash


def test_a_landscape_card_plans_without_a_keep_out(fake_project):
    # A 16:9 card is inside the safe area already, so it has none of its boxes and its layout is what it
    # was: the boxes of Bedroom Pop on the fixture, and the quadrants they were planned in
    folder, project = fake_project
    assert _keepout(project) == ()
    boxes = elements(folder, project)
    assert {key: (box.x, box.y, box.w, box.h, box.size, box.clash) for key, box in boxes.items()} == {
        "tl": (43, 43, 286, 112, 56, False), "tr": (703, 43, 331, 56, 56, False),
        "bl": (43, 392, 360, 168, 56, False), "br": (583, 408, 449, 112, 56, False),
    }
    assert {key: box.arrow for key, box in boxes.items()} == {
        "tl": ((329, 163), (355, 204), (401, 220)), "tr": ((735, 107), (718, 150), (677, 173)),
        "bl": ((155, 384), (200, 360), (250, 370)), "br": ((935, 400), (894, 376), (848, 384)),
    }
    context = engine.matte_context(project["style"], matte.load_alpha(folder, project, 60))
    assert context.keepout == () and context.quads == {
        "tl": (43, 43, 497, 304), "tr": (583, 43, 1037, 304), "bl": (43, 304, 497, 565), "br": (583, 304, 1037, 565),
    }


def test_the_quadrants_of_a_tall_card_are_the_room_the_title_and_the_safe_area_leave(
    fake_project, tmp_path, person_alpha
):
    # Where the face box is a head, the notes are planned inside the room the safe area leaves, not pushed
    # out of the boxes: the rows of the quadrants start under the title room (y 300) and end at the top of
    # the bottom band (y 1500), and the right quadrants stop at the rail's left side (x 940) where its rows
    # reach them. This is the synthetic person, a head and shoulders that fill the frame, on the same plan.
    folder, project = fake_project
    full = tmp_path / "synthetic"
    (full / "frames").mkdir(parents=True)
    (full / "matte").mkdir()
    with Image.open(folder / "frames" / "000060.jpg") as frame:
        frame.convert("RGB").resize((1080, 1920)).save(full / "frames" / "000060.jpg", quality=95)
    Image.fromarray(np.rint(person_alpha(1080, 1920) * 255).astype(np.uint8)).save(full / "matte" / "000060.png")
    project = deepcopy(project)
    project["layout"], project["work"] = "portrait", {"width": 1080, "height": 1920}
    cleaned = matte.load_alpha(full, project, 60)

    context = engine.matte_context(project["style"], cleaned, list(_keepout(project)))
    assert context.face == (389, 154, 691, 1168)
    assert context.quads == {
        "tl": (43, 300, 497, 900), "tr": (583, 300, 1037, 900),  # under the title room
        "bl": (43, 900, 497, 1500), "br": (583, 900, 940, 1500),  # to the bottom band, and to the rail
    }
    # A style with no arrows plans in the same room (its quadrants were the whole stage's)
    no_arrows = {**project["style"], "arrows": "none"}
    assert engine.matte_context(no_arrows, cleaned, list(_keepout(project))).quads == context.quads
    # With the title room alone the quadrants keep the rows they had (the person's reach)
    assert engine.matte_context(project["style"], cleaned, [_keepout(project)[0]]).quads["bl"][3] == 1877

    # Every hero lays out clean in that room: nothing cut, nothing flagged, the four notes in their own
    # quadrants (so the bottom ones are in the lower half, above the band) and the HUD's label in its place
    recipes = {
        "none": BUILTINS["bedroom-pop"], "burst": BUILTINS["comic-pop"], "stats": BUILTINS["player-select"],
        "hud": {**BUILTINS["player-select"], "hero": "hud"},
    }
    for hero, recipe in recipes.items():
        boxes = elements(full, {**project, "style": recipe})
        assert not any(box.clash or box.cut for box in boxes.values()), hero
        for key in ("bl", "br"):
            # (a note may end at the band's top row, where it touches it and is not in it)
            assert BOTTOM_BAND[1] >= boxes[key].y + boxes[key].h > 900 and boxes[key].y >= 900, (hero, key)
        for key in ("tl", "tr"):
            assert boxes[key].y >= 300, (hero, key)
        if hero == "hud":
            label = boxes["hud_bottom"]
            assert (label.x, label.y, label.w, label.h) == (721, 1464, 176, 16)
    # The labels' room is kept from the notes: planned with it, a bottom note never touches the label
    # (the box is 16 px of air around "PRESS START", which is among the rows of the bottom notes)
    label = (721 - 16, 1464 - 16, 897 + 16, 1480 + 16)
    assert not any(hits(box, label) for key, box in boxes.items() if key != "hud_bottom")


def test_the_burst_of_a_tall_card_shrinks_and_is_judged_against_the_head(fake_project, tmp_path):
    # Comic Pop on a vertical photo has no clash on any box. The full-size burst, 310 x 220, goes to the
    # top of the top right room, under the title room (y 300), and there it is on the head box: the head starts
    # at y 497 and the burst reaches 520. So it moves right, to where the stage lets it go (x 758), and the head
    # box without its air (which the burst is judged against) ends at x 788, so it is still on it. It shrinks,
    # the box and its lettering together, to the first of 0.95 (x 773, still on it) and 0.90 that is clear:
    # 279 x 198, its left edge at x 789, with lettering of 59 px (66 times 0.9).
    folder = tmp_path / "vertical"
    project = vertical_project(fake_project, folder)
    recipe = BUILTINS["comic-pop"]
    context = engine.matte_context(recipe, matte.load_alpha(folder, project, 60), list(_keepout(project)))
    head = engine._tight(context).face
    assert head[0] > context.face[0] and head[2] < context.face[2]  # there is air to tell from the head

    boxes = elements(folder, {**project, "style": recipe})
    assert not any(box.clash for box in boxes.values())
    burst = boxes["burst"]
    assert (burst.x, burst.y, burst.w, burst.h, burst.size) == (789, 300, 279, 198, 59)
    assert burst.x >= head[2] and not hits(burst, head)  # beside the head, under the title room
    assert burst.y == TITLE_ROOM[3] and not hits(burst, TITLE_ROOM)
    assert burst.y + burst.h > context.face[1]  # (it would be on the head, if it were above it)

    # A burst dragged onto the head keeps its full size and is flagged, whatever the air says
    dragged = {**project, "style": recipe, "positions": {**project["positions"], "burst": [450, 700]}}
    on_head = elements(folder, dragged)["burst"]
    assert (on_head.x, on_head.y, on_head.w, on_head.h) == (450, 700, 310, 220)
    assert hits(on_head, head) and on_head.clash
