"""Tests for the eleven newer little extras (styles/extras.py): hearts, a scribble circle, a doodle crown,
sparkles, motion lines, confetti, a REC timestamp, VHS lines, a light leak, a name tag and a speech bubble.

What is pinned, for each of them:

- Off, it draws nothing: the card is the card it was, pixel for pixel (the four approved looks are the
  oracles of test_scale, and every built-in has all eleven off).
- On, it changes the card in the place it is meant for, and it never touches the face box.
- On a 9:16 card the drawn ones keep out of the title room, the bottom band and the rail.
- It is the same card at twice the size, shrunk, as at the plan (test_scale's second oracle).
- A recipe saved before the eleven (without their keys) still works, with them off.

The scene is conftest's synthetic person (a head over rounded shoulders) on a stage of the plan's size,
drawn at the overlay level (engine.draw_overlay), with the person shifted down so that the head has
headroom for the crown; and the plate project (a real close-up) for the whole card.
"""

import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image, ImageFilter

from conftest import SAMPLE_COPY
from mclib import card, matte
from mclib import project as project_file
from mclib.card import ALL_VISIBLE
from mclib.raster import IDENTITY, Scale
from mclib.styles import engine, extras, recipes
from mclib.styles.recipes import BUILTINS, EXTRAS, LABELS, OPTIONS

NEW = EXTRAS[3:]  # the eleven, in the order the page shows them
DRAWN = ("hearts", "circle", "crown", "sparkles", "motion", "confetti", "rec", "nametag", "bubble")  # (over the card)
STAGE = ("vhs", "leak")  # (part of the stage)
PLAN = (1080, 608)
TWICE = (2160, 1216)
COPY = deepcopy(SAMPLE_COPY)
COPY["extras"]["bubble"] = "hm?"  # (the emphasis word is "overthinks", in the bl note)


def with_extras(recipe, *on):
    """A copy of the recipe with these extras on (and the other of the eleven off)."""
    return {**recipe, "extras": {**recipe["extras"], **{key: key in on for key in NEW}}}


def box_of(mask):
    """The (x0, y0, x1, y1) of the set pixels of a picture, x1 and y1 one past the last, or None."""
    ys, xs = np.nonzero(mask)
    return None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def groups(mask, gap=6, cell=4):
    """How many groups of set pixels there are, with a gap of up to `gap` px closing a group (a stroke that
    is a few px thick, a heart and a dash of confetti are one group each). Counted on a quarter-size
    picture (cells of `cell` px) of the mask grown by `gap` px, by flooding from each set cell over its 8
    neighbors."""
    grown = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * gap + 1))) > 0 if gap else mask
    rows, columns = grown.shape[0] // cell, grown.shape[1] // cell
    small = grown[: rows * cell, : columns * cell].reshape(rows, cell, columns, cell).any(axis=(1, 3))
    todo = {(int(y), int(x)) for y, x in zip(*np.nonzero(small))}
    found = 0
    while todo:
        found += 1
        stack = [todo.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                if neighbor in todo:
                    todo.remove(neighbor)
                    stack.append(neighbor)
    return found


class Scene:
    """The synthetic person on a stage of the plan's size, with the layout and the context of a recipe."""

    def __init__(self, person_alpha, recipe, shift=0, copy=COPY, size=PLAN, keepout=None):
        width, height = size
        alpha = np.zeros((height, width), np.float32)
        made = person_alpha(width, height)
        alpha[shift:] = made[: height - shift]
        self.alpha, self.recipe, self.copy = alpha, recipe, copy
        self.context = engine.matte_context(recipe, alpha, keepout)
        self.boxes = engine.style_layout(recipe, alpha, copy, None, self.context)
        self.size = size

    @property
    def face(self):
        return self.context.face

    def overlay(self, *on, visible=ALL_VISIBLE, scale=IDENTITY, variant=0, copy=None, recipe=None):
        """The overlay's four channels with these extras on and the rest of the eleven off."""
        style = with_extras(recipe or self.recipe, *on)
        size = self.size if scale.identity else (round(self.size[0] * scale.k), round(self.size[1] * scale.ky))
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        engine.draw_overlay(
            layer, style, self.boxes, copy or self.copy, visible, self.context.face, self.context.keepout, scale,
            self.context.person, variant,
        )
        return np.asarray(layer)

    def added(self, *on, **kwargs):
        """The pixels the extras add to the overlay: where it differs from the one with none of the eleven."""
        return (self.overlay(*on, **kwargs) != self.overlay(**kwargs)).any(axis=2)


@pytest.fixture
def scene(person_alpha):
    return Scene(person_alpha, BUILTINS["bedroom-pop"])


@pytest.fixture
def headroom(person_alpha):
    """The same person 80 px lower, so that there is room above their head for the crown and the motion lines."""
    return Scene(person_alpha, BUILTINS["bedroom-pop"], shift=80)


# --- the recipe: ids, labels, built-ins, old recipes -------------------------------------------------------


def test_the_extras_are_the_fourteen_in_the_order_the_page_shows_them():
    assert EXTRAS == (
        "tape", "stars", "grain", "hearts", "circle", "crown", "sparkles", "motion", "confetti", "rec", "vhs",
        "leak", "nametag", "bubble",
    )
    assert OPTIONS["extras"] == EXTRAS and set(LABELS["extras"]) == set(EXTRAS)
    assert [LABELS["extras"][key] for key in NEW] == [
        "Hearts", "Scribble circle", "Doodle crown", "Sparkles", "Motion lines", "Confetti", "REC timestamp",
        "VHS lines", "Light leak", "Name tag", "Speech bubble",
    ]


def test_every_built_in_has_the_eleven_off_and_the_first_three_as_they_were():
    for recipe in BUILTINS.values():
        assert set(recipe["extras"]) == set(EXTRAS)
        assert not any(recipe["extras"][key] for key in NEW), recipe["id"]
    assert {key for key in ("tape", "stars", "grain") if BUILTINS["scrapbook"]["extras"][key]} == {"tape", "stars"}
    assert {key for key, on in BUILTINS["bedroom-pop"]["extras"].items() if on} == {"grain"}


def test_a_recipe_from_before_the_eleven_is_valid_and_is_completed_with_them_off():
    old = deepcopy(BUILTINS["scrapbook"])
    old["extras"] = {"tape": True, "stars": True, "grain": False}

    recipes.validate_recipe(old)  # never refused
    full = recipes.complete(old)

    assert old["extras"] == {"tape": True, "stars": True, "grain": False}  # (the recipe handed in is not changed)
    assert list(full["extras"]) == ["tape", "stars", "grain", *NEW] or set(full["extras"]) == set(EXTRAS)
    assert full["extras"]["tape"] is True and not any(full["extras"][key] for key in NEW)
    assert recipes.complete(full) is full  # nothing to add
    assert recipes.complete("not a recipe") == "not a recipe"
    recipes.validate_recipe(full)


def test_the_extras_check_names_what_is_wrong():
    base = deepcopy(BUILTINS["bedroom-pop"])
    for extras_ in (
        {"tape": False, "stars": False},  # (the first three are required)
        {"tape": False, "stars": False, "grain": False, "glitter": True},
        {**base["extras"], "hearts": "yes"},
        {**base["extras"], "bubble": 1},
    ):
        with pytest.raises(ValueError, match="^extras"):
            recipes.validate_recipe({**base, "extras": extras_})
    recipes.validate_recipe({**base, "extras": {**base["extras"], "hearts": True, "leak": True}})


def test_a_saved_style_from_before_the_eleven_loads_with_them_off(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_HOME", str(tmp_path))
    old = deepcopy(BUILTINS["bedroom-pop"])
    old.update(id="u-night-owl", name="Night Owl", extras={"tape": False, "stars": True, "grain": True})
    folder = tmp_path / "styles"
    folder.mkdir()
    (folder / "night-owl.json").write_text(
        json.dumps({"id": "u-night-owl", "name": "Night Owl", "recipe": old}), encoding="utf-8"
    )

    styles = {style["id"]: style for style in recipes.list_styles()}
    got = recipes.get_style("u-night-owl")["recipe"]

    assert set(styles["u-night-owl"]["recipe"]["extras"]) == set(EXTRAS) == set(got["extras"])
    assert got["extras"]["stars"] is True and not any(got["extras"][key] for key in NEW)


def test_saving_a_style_that_lacks_the_eleven_keeps_them_off_in_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_HOME", str(tmp_path))
    old = deepcopy(BUILTINS["bedroom-pop"])
    old["extras"] = {"tape": False, "stars": False, "grain": True}

    saved = recipes.save_user_style(old, "Old Look")

    written = json.loads((tmp_path / "styles" / "old-look.json").read_text(encoding="utf-8"))
    assert set(written["recipe"]["extras"]) == set(EXTRAS) == set(saved["recipe"]["extras"])


def test_a_project_from_before_the_eleven_loads_with_them_off(tmp_path, project_maker):
    made = project_maker(tmp_path)
    old = json.loads((made.dir / "project.json").read_text(encoding="utf-8"))
    old["style"]["extras"] = {"tape": False, "stars": True, "grain": True}
    (made.dir / "project.json").write_text(json.dumps(old), encoding="utf-8")

    loaded = project_file.load(made.dir)

    assert set(loaded["style"]["extras"]) == set(EXTRAS) and loaded["style"]["extras"]["stars"] is True
    assert not any(loaded["style"]["extras"][key] for key in NEW)
    project_file.validate(loaded)


# --- the bubble's text: copy.extras.bubble -------------------------------------------------------------------


def test_the_bubble_text_is_text_or_null_with_a_soft_limit():
    assert project_file.empty_copy()["extras"]["bubble"] is None
    assert project_file.SOFT_LIMITS["bubble"] == 14

    project = project_file.default_project(
        {"path": "c.mp4", "stem": "c", "width": 1920, "height": 1080, "fps": 30.0, "duration": 4.0, "frames": 120,
         "orientation": "landscape"},
        "landscape", PLAN, BUILTINS["bedroom-pop"],
    )
    assert project_file.validate(project) == []
    project["copy"]["extras"]["bubble"] = "x" * 14
    assert project_file.validate(project) == []
    project["copy"]["extras"]["bubble"] = "x" * 15
    assert project_file.validate(project) == ["copy.extras.bubble is 15 characters; the soft limit is 14."]
    project["copy"]["extras"]["bubble"] = 3
    with pytest.raises(ValueError, match=r"copy\.extras\.bubble must be text or null"):
        project_file.validate(project)
    project["copy"]["extras"]["bubble"] = "wait \U0001F600"
    assert project_file.validate(project) == [
        "Removed emoji from copy.extras.bubble; the video fonts cannot draw them."
    ]
    assert project["copy"]["extras"]["bubble"] == "wait"


def test_a_project_from_before_the_bubble_has_none_and_a_patch_sets_it(tmp_path, project_maker):
    made = project_maker(tmp_path)
    old = json.loads((made.dir / "project.json").read_text(encoding="utf-8"))
    old["copy"]["extras"].pop("bubble", None)  # (a file from before the field has none)
    for key in "ABC":
        old["versions"][key]["extras"].pop("bubble", None)
    (made.dir / "project.json").write_text(json.dumps(old), encoding="utf-8")

    changed, warnings = project_file.update(made.dir, {"copy": {"extras": {"bubble": "hm?"}}})

    assert not [w for w in warnings if "bubble" in w] and changed["copy"]["extras"]["bubble"] == "hm?"
    assert project_file.load(made.dir)["copy"]["extras"]["bubble"] == "hm?"
    cleared, _ = project_file.update(made.dir, {"copy": {"extras": {"bubble": None}}})  # (a merge patch's null deletes it)
    assert cleared["copy"]["extras"].get("bubble") is None


# --- off draws nothing --------------------------------------------------------------------------------------


def test_with_the_eleven_off_the_overlay_and_the_stage_are_what_they_were(scene, person_alpha):
    old = deepcopy(scene.recipe)
    old["extras"] = {"tape": False, "stars": True, "grain": True}  # (a recipe from before them: no keys)
    layer = Image.new("RGBA", PLAN, (0, 0, 0, 0))
    engine.draw_overlay(layer, old, scene.boxes, scene.copy, ALL_VISIBLE, scene.face, scene.context.keepout)
    assert np.array_equal(np.asarray(layer), scene.overlay(recipe={**old, "extras": {**old["extras"]}}))

    rgb = np.full((608, 1080, 3), 0.5, np.float32)
    stage = engine.compose_stage(old, rgb, scene.alpha, 0)
    assert np.array_equal(stage, engine.compose_stage(with_extras(old), rgb, scene.alpha, 0))


@pytest.mark.parametrize("key", ["circle", "nametag", "bubble"])
def test_an_extra_that_needs_words_draws_nothing_without_them(scene, key):
    empty = deepcopy(scene.copy)
    empty["extras"].update(emphasis=None, bubble=None)
    empty["name"] = ""
    assert not scene.added(key, copy=empty).any()
    other = deepcopy(scene.copy)
    other["extras"]["emphasis"] = "unicorn"  # (a word no note has)
    assert not scene.added("circle", copy=other).any()
    assert scene.added(key).any()  # and with them, it is there


# --- hearts, circle, crown, sparkles, motion lines ----------------------------------------------------------------


def test_hearts_are_two_or_three_outlines_beside_the_head_and_come_in_with_the_arrows(scene):
    drawn = scene.added("hearts")

    x0, y0, x1, y1 = scene.face
    assert box_of(drawn) is not None and groups(drawn) in (2, 3)
    ys, xs = np.nonzero(drawn)
    assert not ((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).any()  # never on the face box
    assert (xs < x0).any() or (xs >= x1).any()  # beside it
    assert not scene.added("hearts", visible={"bottom", "top"}).any()  # with the arrows


def test_the_circle_rings_the_emphasis_word_in_its_note_and_comes_in_with_that_note(scene):
    drawn = scene.added("circle")

    note = scene.boxes["bl"]  # "But overthinks every 6-second reel"
    x0, y0, x1, y1 = box_of(drawn)
    assert note.x - 60 < x0 and x1 < note.x + note.w + 60 and note.y - 60 < y0 and y1 < note.y + note.h + 60
    assert groups(drawn) == 1  # one scribble, its two loops together
    assert not scene.added("circle", visible={"top", "arrows"}).any()  # bl is a bottom note
    assert scene.added("circle", visible={"bottom"}).any()
    # it may use the air the face box has at its sides (43 px of it), never the head inside that
    assert not (drawn[scene.face[1] : scene.face[3], scene.face[0] + 43 : scene.face[2] - 43]).any()


def test_the_circle_is_a_scribble_of_two_loops_a_little_off_each_other(scene):
    drawn = scene.added("circle")

    # one loop is a ring of 4 px strokes: across its middle a line crosses it twice, 8 px of ink. Two loops that
    # are offset and not the same size cross it four times (or twice, as thick as that), so more.
    middle = (np.nonzero(drawn.any(axis=1))[0][[0, -1]].mean()).astype(int)
    assert drawn[middle - 3 : middle + 4].sum(axis=1).max() > 8


def test_the_crown_stands_on_the_top_of_the_head_and_is_left_out_when_the_head_touches_the_top(headroom, scene):
    drawn = headroom.added("crown")

    x0, y0, x1, y1 = headroom.face
    box = box_of(drawn)
    assert box is not None and box[3] <= y0 and x0 < (box[0] + box[2]) / 2 < x1  # on the head, above the face box
    assert not headroom.added("crown", visible={"bottom", "top"}).any()  # with the arrows
    assert not scene.added("crown").any()  # the person at 0 has 49 px above the head: not enough room


def test_the_sparkles_are_three_to_five_near_the_outline_and_never_on_the_person(scene):
    drawn = scene.added("sparkles")

    assert 3 <= groups(drawn) <= 5
    person = scene.context.person
    assert not (drawn & person).any()
    near = Image.fromarray((person * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * 110 + 1))
    assert (np.asarray(near) > 0)[drawn].all()  # every pixel is within 110 px of the person
    ys, xs = np.nonzero(drawn)
    x0, y0, x1, y1 = scene.face
    assert not ((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).any()
    assert not scene.added("sparkles", visible={"bottom", "top"}).any()  # with the arrows


def test_the_motion_lines_are_three_strokes_beside_the_top_of_the_head(headroom):
    drawn = headroom.added("motion")

    x0, y0, x1, y1 = headroom.face
    assert groups(drawn, gap=0, cell=1) == 3  # (three strokes that do not touch)
    ys, xs = np.nonzero(drawn)
    assert not ((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).any()
    assert ys.max() < y0 + 0.5 * (y1 - y0)  # beside the top of the head, not its middle
    assert (xs >= x1).all() or (xs < x0).all()  # all on one side
    assert not headroom.added("motion", visible={"bottom", "top"}).any()


# --- confetti ------------------------------------------------------------------------------------------------------


def test_confetti_is_20_to_40_pieces_on_the_background_only(scene):
    room = extras._room(scene.boxes, scene.recipe, scene.face, scene.context.keepout, *PLAN, scene.context.person)
    pal = scene.recipe["palette"]

    pieces = extras.scatter(room, pal, "#7d2a72", 0)

    assert 20 <= len(pieces) <= 40
    drawn = scene.added("confetti")
    assert not (drawn & scene.context.person).any()  # never on the person
    x0, y0, x1, y1 = scene.face
    ys, xs = np.nonzero(drawn)
    assert not ((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).any()
    for box in scene.boxes.values():  # nor on a note
        assert not drawn[box.y : box.y + box.h, box.x : box.x + box.w].any(), box.key
    assert scene.added("confetti", visible=set()).any()  # on from the first frame, before any group
    colors = {tuple(c) for c in scene.overlay("confetti")[drawn][:, :3]}
    assert len(colors) >= 2  # the palette's colors


def test_confetti_boils_a_little_with_the_variant(scene):
    first, second, again = (scene.added("confetti", variant=v) for v in (0, 1, 0))

    assert np.array_equal(first, again)  # the same picture every time
    assert (first != second).any()  # it re-scatters
    shared = (first & second).sum() / first.sum()
    assert 0.15 < shared  # a small re-scatter: most pieces are still about where they were


# --- REC, name tag, bubble ----------------------------------------------------------------------------------------


def test_rec_is_a_red_dot_and_a_time_in_the_top_corners_for_the_whole_card(scene):
    drawn = scene.added("rec")
    overlay = scene.overlay("rec")

    assert scene.added("rec", visible=set()).any()  # always on
    ys, xs = np.nonzero(drawn)
    assert ys.max() < 0.4 * 608 and ys.min() < 130  # the top of the card (a piece goes down a corner when a note has it)
    red = (overlay[..., 0] > 230) & (overlay[..., 1] < 70) & (overlay[..., 2] < 70) & (overlay[..., 3] == 255)
    assert red.sum() > 200  # the dot
    dark_x, dark_y = np.nonzero((overlay[..., 3] == 255) & (overlay[..., :3].max(axis=2) < 40))[::-1]
    assert len(dark_x) > 100  # and the outline that lets the letters read
    assert groups(drawn) >= 2  # the dot and its word, and the time
    x0, y0, x1, y1 = scene.face
    assert not ((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)).any()


def test_the_name_tag_is_on_the_chest_under_the_face_with_the_name(scene):
    drawn = scene.added("nametag")
    overlay = scene.overlay("nametag")

    x0, y0, x1, y1 = box_of(drawn)
    assert y0 >= scene.face[3] and (scene.face[0] < (x0 + x1) / 2 < scene.face[2])  # under the face, on the chest
    assert (drawn & scene.context.person).sum() / drawn.sum() > 0.85  # stuck to the person
    band = (abs(overlay[..., 0].astype(int) - 0xD9) < 8) & (abs(overlay[..., 1].astype(int) - 0x2B) < 8) & (overlay[..., 3] == 255)
    assert band.sum() > 500  # the red band
    paper = (overlay[..., :3] == (0xFF, 0xFD, 0xF5)).all(axis=2) & (overlay[..., 3] == 255)
    assert paper.sum() > 1000  # and the white body
    assert scene.added("nametag", visible={"bottom"}).any() and scene.added("nametag", visible={"top"}).any()
    assert not scene.added("nametag", visible=set()).any()  # with the notes' groups
    named = deepcopy(scene.copy)
    named["name"] = "Sam"
    assert (scene.overlay("nametag", copy=named) != overlay).any()  # it writes the name


def test_the_speech_bubble_is_beside_the_face_at_the_mouth_with_its_text(scene):
    drawn = scene.added("bubble")
    overlay = scene.overlay("bubble")

    x0, y0, x1, y1 = scene.face
    box = box_of(drawn)
    mouth = y0 + 0.76 * (y1 - 18 - y0)
    assert box[2] <= x0 or box[0] >= x1  # beside the face box
    assert box[1] - 20 < mouth < box[3] + 20  # at the mouth's height
    assert not (drawn & scene.context.person).any()
    paper = (overlay[..., :3] == (0xFF, 0xFD, 0xF5)).all(axis=2) & (overlay[..., 3] == 255)
    assert paper.sum() > 1500
    other = deepcopy(scene.copy)
    other["extras"]["bubble"] = "wait"
    assert (scene.overlay("bubble", copy=other) != overlay).any()  # it says what the field says
    assert not scene.added("bubble", visible={"bottom", "top"}).any()  # with the arrows


# --- vhs and leak: part of the stage ------------------------------------------------------------------------------


def test_vhs_lays_scanlines_and_a_red_cyan_offset_over_the_whole_stage(scene):
    rgb = np.full((608, 1080, 3), 0.5, np.float32)
    rgb[:, 540:] = (0.9, 0.3, 0.3)  # an edge for the channels to come apart on
    base = engine.compose_stage(with_extras(scene.recipe), rgb, scene.alpha, 0)
    lined = engine.compose_stage(with_extras(scene.recipe, "vhs"), rgb, scene.alpha, 0)

    rows = lined.mean(axis=(1, 2)) / base.mean(axis=(1, 2))
    assert rows.min() < 0.9 and rows.max() > 0.98  # thin dark rows, the rest as they were
    assert (rows < 0.95).mean() < 0.4  # thin
    assert (lined[300, :, 0] != base[300, :, 0]).any()  # the red moved against the others
    across = np.abs(np.diff(lined[:, :, 1], axis=1)).argmax(axis=1)
    reds = np.abs(np.diff(lined[:, :, 0], axis=1)).argmax(axis=1)
    assert (reds - across).max() >= 1  # a pixel or two apart


def test_the_light_leak_adds_a_warm_glow_from_the_side_away_from_the_head(scene):
    rgb = np.full((608, 1080, 3), 0.25, np.float32)
    base = engine.compose_stage(with_extras(scene.recipe), rgb, scene.alpha, 0)
    glow = engine.compose_stage(with_extras(scene.recipe, "leak"), rgb, scene.alpha, 0)

    assert (glow >= base - 1e-6).all()  # it only adds light
    left, right = (glow - base)[:, :60].mean(), (glow - base)[:, -60:].mean()
    assert max(left, right) > 0.05 and min(left, right) < max(left, right) * 0.2  # from one edge only
    side = glow[:, :60] if left > right else glow[:, -60:]
    assert side[..., 0].mean() > side[..., 2].mean()  # warm: more red than blue


# --- the rules that hold for all of them -------------------------------------------------------------------------


def test_nothing_the_extras_draw_touches_the_face_box_or_the_notes(scene, headroom):
    for one in (scene, headroom):
        added = one.added(*DRAWN)
        x0, y0, x1, y1 = one.face
        # (the loop round a word may use the air the face box has at its sides, 43 px of it, never the head)
        assert not added[y0:y1, x0 + 43 : x1 - 43].any()
        assert not one.added(*[key for key in DRAWN if key != "circle"])[y0:y1, x0:x1].any()
        for key, box in one.boxes.items():
            if key == "bl":  # (the scribble circle rings a word of this note)
                continue
            assert not added[box.y + 6 : box.y + box.h - 6, box.x + 6 : box.x + box.w - 6].any(), key


def test_the_extras_keep_off_each_other(scene, headroom):
    for one in (scene, headroom):
        recipe = with_extras(one.recipe, *DRAWN)
        placed = extras.plan(one.boxes, recipe, one.copy, one.face, one.context.keepout, *PLAN, one.context.person)
        boxes = [(key, item.box) for key, items in placed.items() for item in items if item.box is not None]
        assert len(boxes) >= 8  # most of them found room
        for number, (first, a) in enumerate(boxes):
            for second, b in boxes[number + 1 :]:
                assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]), (first, second, a, b)


def test_the_layout_decides_where_an_extra_goes_not_what_is_visible(headroom):
    everything = headroom.added("hearts", "crown", "sparkles", "motion", "bubble")
    for visible in ({"arrows"}, {"bottom", "top", "arrows"}):
        part = headroom.added("hearts", "crown", "sparkles", "motion", "bubble", visible=visible)
        assert (part & ~everything).sum() == 0  # nothing lands where the full card has nothing


def test_on_a_9_16_card_every_drawn_extra_keeps_out_of_the_title_room_the_bottom_band_and_the_rail(plate_maker, tmp_path):
    folder, project = plate_maker(tmp_path / "tall", BUILTINS["scrapbook"], portrait=True)
    cleaned = matte.load_alpha(folder, project, 60)
    keepout = card._keepout(project)
    recipe = with_extras(BUILTINS["scrapbook"], *DRAWN)
    context = engine.matte_context(recipe, cleaned, list(keepout))
    copy = deepcopy(project["copy"])
    copy["extras"]["bubble"] = "hm?"
    boxes = engine.style_layout(recipe, cleaned, copy, None, context)

    def painted(style):
        layer = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
        engine.draw_overlay(layer, style, boxes, copy, ALL_VISIBLE, context.face, context.keepout, IDENTITY, context.person)
        return np.asarray(layer)

    base, drawn = painted(with_extras(recipe)), painted(recipe)

    added = (drawn != base).any(axis=2)
    assert added.sum() > 3000  # something is drawn
    assert len(keepout) == 3
    for x0, y0, x1, y1 in keepout:
        assert not added[y0:y1, x0:x1].any()
    # without the keep-out boxes the same extras do land in them: the REC text goes to the top of the card
    free = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    engine.draw_overlay(free, recipe, boxes, copy, ALL_VISIBLE, context.face, (), IDENTITY, context.person)
    assert (np.asarray(free) != base).any(axis=2)[: keepout[0][3]].any()


# --- the same card at twice the size ----------------------------------------------------------------------------


def blurred(image):
    return np.asarray(image.filter(ImageFilter.GaussianBlur(1)), np.int16)


def shrunk_difference(big, plan):
    return float(np.abs(blurred(big.resize(plan.size, Image.Resampling.LANCZOS)) - blurred(plan)).mean())


@pytest.mark.parametrize("key", [*NEW])
def test_an_extra_drawn_at_twice_the_size_is_the_card_at_the_plan(plate_maker, tmp_path, key):
    folder, project = plate_maker(tmp_path, BUILTINS["bedroom-pop"])
    project["copy"]["extras"]["bubble"] = "hm?"
    off, on = with_extras(BUILTINS["bedroom-pop"]), with_extras(BUILTINS["bedroom-pop"], key)

    plan_off, plan_on = (card.render_card(folder, project, style=style) for style in (off, on))
    big_off, big_on = (card.render_card(folder, project, style=style, size=TWICE) for style in (off, on))

    assert big_on.size == TWICE
    base = shrunk_difference(big_off, plan_off)
    assert shrunk_difference(big_on, plan_on) < 3
    assert shrunk_difference(big_on, plan_on) < base + 1.0  # the extra costs a part of the allowance, not all of it

    def shrunk(image):
        return np.asarray(image.resize(plan_on.size, Image.Resampling.LANCZOS), np.int16)

    def strong(a, b):
        return np.abs(a - b).max(axis=2) > 40

    mine = strong(np.asarray(plan_on, np.int16), np.asarray(plan_off, np.int16))
    theirs = strong(shrunk(big_on), shrunk(big_off))
    if key in STAGE:
        assert mine.any() and theirs.any()
    elif key in ("crown", "nametag"):
        # (the real close-up touches the top of the card and leaves no chest: both are left out, at both sizes)
        assert not mine.any() and not theirs.any(), key
    else:
        assert mine.any() and theirs.any(), key
        assert (mine & theirs).sum() >= 0.5 * min(mine.sum(), theirs.sum()), key  # the same extra, in the same place


def ink(one, key, scale=IDENTITY):
    """The alpha an extra adds to the overlay (the overlay with it, less the overlay without), as levels."""
    return np.clip(
        one.overlay(key, scale=scale)[..., 3].astype(np.int16) - one.overlay(scale=scale)[..., 3].astype(np.int16), 0, 255
    ).astype(np.uint8)


@pytest.mark.parametrize("key", DRAWN)
def test_the_drawn_extras_are_the_same_overlay_at_twice_the_size(request, key):
    """Where an extra goes is decided at the plan and only drawn bigger: at twice the size, shrunk, the ink
    it adds is the plan's (the crown and the motion lines need the head with headroom)."""
    one = request.getfixturevalue("headroom" if key in ("crown", "motion") else "scene")
    twice = Scale.between(PLAN, TWICE)

    plan = ink(one, key).astype(np.int16)
    big = np.asarray(Image.fromarray(ink(one, key, twice)).resize(PLAN, Image.Resampling.LANCZOS), np.int16)

    assert plan.any() and big.any(), key
    assert np.abs(big - plan).sum() < 0.35 * plan.sum(), key  # a hair's width of edge differs, not the shapes
    assert abs(float(big.sum()) / plan.sum() - 1) < 0.15, key  # the same amount of ink


def test_the_crown_and_the_motion_lines_are_in_the_same_place_at_twice_the_size(headroom):
    twice = Scale.between(PLAN, TWICE)
    for key in ("crown", "motion"):
        here = box_of(headroom.added(key))
        there = box_of(headroom.added(key, scale=twice))
        assert all(abs(a - b / 2) <= 3 for a, b in zip(here, there)), key


# --- taste review: circle ink, heart cluster, bigger tag, confetti colors ------------------------------------


def hearts_of(one):
    """The boxes of the hearts the planner placed, in px of the plan."""
    placed = extras.plan(
        one.boxes, with_extras(one.recipe, "hearts"), one.copy, one.face, one.context.keepout, *PLAN, one.context.person
    )
    return [item.box for item in placed["hearts"]]


def test_the_hearts_are_a_doodle_cluster_one_big_and_smaller_ones_offset_diagonally(scene, headroom):
    for one in (scene, headroom):
        boxes = hearts_of(one)
        assert 2 <= len(boxes) <= 3
        halves = sorted(((b[2] - b[0]) - 12) / 2 for b in boxes)  # (a box is 6 px wider each side than the heart)
        assert 1.4 <= halves[-1] / halves[0] <= 1.8  # sizes vary by about 1.6x
        assert len(set(round(h) for h in halves)) == len(halves)  # no two the same size
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        (ax, ay), (bx, by), *rest = centers
        for cx, cy in rest:  # not in a line: the triangle they make has some area
            assert abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) > 1500
        assert len({round(x) for x, _ in centers}) == len(centers)  # not a vertical column
        assert len({round(y) for _, y in centers}) == len(centers)  # nor a row


def test_the_circle_is_drawn_in_the_notes_text_ink_as_thick_as_the_arrows(person_alpha):
    from mclib.styles import notes

    for recipe_id, recipe in BUILTINS.items():
        one = Scene(person_alpha, recipe)
        overlay = one.overlay("circle", recipe=recipe)
        added = one.added("circle", recipe=recipe)
        if not added.any():  # (the hero of this style may leave the note no room)
            continue
        text = {"comic": "#111111", "label": recipe["palette"]["dark"]}.get(recipe["notes"], recipe["palette"]["ink"])
        want = tuple(int(text[i : i + 2], 16) for i in (1, 3, 5))
        solid = overlay[added & (overlay[..., 3] == 255)][:, :3]
        exact = (np.abs(solid.astype(int) - want).max(axis=1) <= 2).sum()
        assert exact > 300 and exact >= 0.4 * len(solid), recipe_id  # the loops are in the ink (their edges blend)
        width = notes.ARROW_STYLE[recipe["notes"]][1]
        thick = added.sum(axis=0)[np.nonzero(added.any(axis=0))[0]]  # ink in each column crossed by the loops
        assert np.median(np.sort(thick)[-len(thick) // 3 :]) >= width, recipe_id
        assert palette_contrast(text, recipe) >= 3.0, recipe_id  # it reads on what the note sits on


def palette_contrast(color, recipe):
    from mclib.palette import contrast
    from mclib.styles import background, notes
    from mclib.raster import mix

    pal = recipe["palette"]
    under = {
        "marker": pal["dark"],  # (the contrast guard measures a marker note on its dark glow)
        "comic": mix(pal["light"], "#ffffff", notes._COMIC_FILL),
        "hand": notes._HAND_PAPER,
        "label": pal["edge"],
    }.get(recipe["notes"], background.tone(recipe["bg"], pal))
    return contrast(color, under)


def test_the_name_tag_is_about_1_4_times_bigger_when_the_chest_has_room(person_alpha):
    small = Scene(person_alpha, BUILTINS["bedroom-pop"])
    # a smaller person, high on the stage, with the body running down to its bottom, leaves a deep chest
    alpha = np.zeros((608, 1080), np.float32)
    made = person_alpha(1080, 608)
    shrunk = np.asarray(Image.fromarray((made * 255).astype(np.uint8)).resize((648, 365)), np.float32) / 255
    alpha[30 : 30 + 365, 216 : 216 + 648] = shrunk
    alpha[30 + 365 :] = alpha[30 + 364]
    short = deepcopy(COPY)  # (short notes sit in the corners, and leave the chest to the tag)
    short["notes"] = {"tl": "AI", "tr": "@me", "bl": "But shy", "br": "Claude"}
    roomy = Scene.__new__(Scene)
    roomy.alpha, roomy.recipe, roomy.copy, roomy.size = alpha, BUILTINS["bedroom-pop"], short, PLAN
    roomy.context = engine.matte_context(roomy.recipe, alpha, None)
    roomy.boxes = engine.style_layout(roomy.recipe, alpha, short, None, roomy.context)

    def width(one):
        box = box_of(one.added("nametag"))
        return None if box is None else box[2] - box[0]

    assert width(small) is not None and width(small) <= 200  # the fallback: a size it had before
    assert width(roomy) is not None and 340 <= width(roomy) <= 380  # 1.4 x 250 (350 wide, tilted)


def test_confetti_uses_only_the_light_and_bright_colors_and_the_ink(scene):
    from mclib.palette import relative_luminance

    for recipe in BUILTINS.values():
        one = Scene.__new__(Scene)
        one.alpha, one.recipe, one.copy, one.size = scene.alpha, recipe, COPY, PLAN
        one.context = engine.matte_context(recipe, one.alpha, None)
        one.boxes = engine.style_layout(recipe, one.alpha, COPY, None, one.context)
        room = extras._room(one.boxes, recipe, one.face, one.context.keepout, *PLAN, one.context.person)
        pal = recipe["palette"]
        used = {piece[5] for piece in extras.scatter(room, pal, "#808080", 0)}
        allowed = {pal["light"], pal["edge"], pal["ink"]} | {
            pal[role] for role in ("dark", "tex") if relative_luminance(pal[role]) >= 0.2
        }
        assert used and used <= allowed, recipe["id"]
        darkest = min((pal[role] for role in ("dark", "tex")), key=relative_luminance)
        if relative_luminance(darkest) < 0.2 and darkest != pal["ink"]:
            assert darkest not in used, recipe["id"]
