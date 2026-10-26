"""Tests for the style recipes and the stage the engine composes.

The recipes are checked against the tables of the spec (section 8a) and of the plan's Global
Constraints, written out here in full so that a typo in recipes.py cannot also be in the test.
The stage is composed from conftest's fake_project: a synthetic clip frame, a synthetic person
and the Bedroom Pop recipe.
"""

import json
import math
import os
import random
import re
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from conftest import SAMPLE_COPY
from PIL import Image

from mclib import layout, matte
from mclib.card import ALL_VISIBLE, elements, render_card
from mclib.config import NOTE_MIN_PX
from mclib.layout import NoteStyle
from mclib.project import default_project, validate
from mclib.raster import (
    blur, dilate, duotone, font, from_image, grain, halftone, hex_rgb, lowfreq, luminance, mix, over,
    quad_points,
)
from mclib.styles import background, edges, engine, notes
from mclib.styles.engine import compose_stage
from mclib.styles.recipes import (
    BUILTINS,
    LABELS,
    OPTIONS,
    PALETTES,
    get_style,
    list_styles,
    save_user_style,
    unavailable,
    user_dir,
    validate_recipe,
)

E_ACUTE = chr(0xE9)  # built with chr(), not typed, so no tool can change it on the way in
POP = BUILTINS["bedroom-pop"]
NO_GRAIN = {**POP, "extras": {"tape": False, "stars": False, "grain": False}}
FRAME = 60
# The ids and the page labels of spec 8a. The hero "none" and the arrows "none" have no label of
# their own in the table: the page shows "None".
EXPECTED_LABELS = {
    "bg": {
        "dots": "Duotone + dots", "comic": "Comic print", "riso": "Riso on paper",
        "night": "Night scanlines", "flat": "Flat color", "mono": "Black & white",
    },
    "edge": {
        "handcut": "Hand-cut sticker", "comic": "Comic double line", "paper": "Paper + shadow",
        "neon": "Neon glow", "thin": "Thin line", "none": "No edge",
    },
    "notes": {
        "marker": "Marker", "comic": "Comic boxes", "hand": "Handwritten strips",
        "pixel": "Pixel", "label": "Bold labels",
    },
    "arrows": {"curly": "Curly", "doodle": "Doodle", "straight": "Line + dot", "none": "None"},
    "hero": {"none": "None", "burst": "Name burst", "stats": "Stats panel", "hud": "Game HUD"},
    "extras": {"tape": "Tape", "stars": "Stars", "grain": "Film grain"},
}

# Global Constraints: the 7 presets in role order (dark, light, tex, edge, ink).
ROLES = ("dark", "light", "tex", "edge", "ink")
EXPECTED_PALETTES = {
    "Magenta": ("#7d2a72", "#f2a9de", "#4c0f45", "#f5df98", "#f7e8b0"),
    "Lagoon": ("#1d4f5c", "#a3e6dc", "#0b2e36", "#ffe9a8", "#fff1c4"),
    "Tangerine": ("#8a3a12", "#ffb98a", "#4a1a06", "#fff0c8", "#fff6dc"),
    "Comic": ("#111111", "#ffd21f", "#e8261c", "#ffffff", "#111111"),
    "Riso": ("#2f4fa8", "#efe4cb", "#2f4fa8", "#fbf8f0", "#1b1b1b"),
    "Neon": ("#07061c", "#5a24c9", "#3ef0ff", "#9ff8ff", "#ffe14d"),
    "Mono": ("#141414", "#e9e6df", "#141414", "#ffe600", "#ffe600"),
}

# Global Constraints: the 4 built-in recipes, as (name, bg, edge, notes, arrows, hero, extras on,
# palette), in the table's order.
EXPECTED_BUILTINS = {
    "bedroom-pop": ("Bedroom Pop", "dots", "handcut", "marker", "curly", "none", {"grain"}, "Magenta"),
    "comic-pop": ("Comic Pop", "comic", "comic", "comic", "curly", "burst", set(), "Comic"),
    "scrapbook": ("Scrapbook", "riso", "paper", "hand", "doodle", "none", {"tape", "stars"}, "Riso"),
    "player-select": ("Player Select", "night", "neon", "pixel", "none", "stats", set(), "Neon"),
}


def recipe_with(**changes):
    """A fresh copy of Bedroom Pop with some fields replaced."""
    recipe = deepcopy(POP)
    recipe.update(changes)
    return recipe


def stage_inputs(fake_project):
    """The freeze frame as float RGB and its cleaned matte, as the card reads them."""
    folder, project = fake_project
    with Image.open(folder / "frames" / f"{FRAME:06d}.jpg") as image:
        rgb = from_image(image.convert("RGB"))
    return rgb, matte.load_alpha(folder, project, FRAME)


# --- recipes ----------------------------------------------------------------------------------


def test_builtins_are_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_HOME", str(tmp_path))  # no saved styles of the person's own

    # The layer ids and the labels are the spec's table, exactly.
    assert OPTIONS == {
        "bg": ("dots", "comic", "riso", "night", "flat", "mono"),
        "edge": ("handcut", "comic", "paper", "neon", "thin", "none"),
        "notes": ("marker", "comic", "hand", "pixel", "label"),
        "arrows": ("curly", "doodle", "straight", "none"),
        "hero": ("none", "burst", "stats", "hud"),
        "extras": ("tape", "stars", "grain"),
    }
    assert LABELS == EXPECTED_LABELS

    # The 7 presets, in the order of the table, each with its 5 roles.
    assert list(PALETTES) == list(EXPECTED_PALETTES)
    for name, colors in EXPECTED_PALETTES.items():
        assert PALETTES[name] == dict(zip(ROLES, colors)), name

    # The 4 recipes, in the order of the table. Each is valid and its palette is a copy of its preset.
    assert list(BUILTINS) == list(EXPECTED_BUILTINS)
    for recipe_id, (name, bg, edge, note_style, arrows, hero, extras, preset) in EXPECTED_BUILTINS.items():
        recipe = BUILTINS[recipe_id]
        validate_recipe(recipe)
        assert (recipe["id"], recipe["name"]) == (recipe_id, name)
        assert (recipe["bg"], recipe["edge"], recipe["notes"]) == (bg, edge, note_style)
        assert (recipe["arrows"], recipe["hero"]) == (arrows, hero)
        assert recipe["extras"] == {key: key in extras for key in ("tape", "stars", "grain")}
        assert recipe["palette"] == PALETTES[preset]
        assert recipe["palette"] is not PALETTES[preset]

    # What list_styles and get_style hand out is a copy: a caller cannot change a built-in.
    listed = list_styles()
    assert [style["id"] for style in listed] == list(BUILTINS)
    assert all(style["builtin"] for style in listed)
    listed[0]["recipe"]["palette"]["dark"] = "#000000"
    get_style("bedroom-pop")["recipe"]["bg"] = "mono"
    assert BUILTINS["bedroom-pop"]["palette"]["dark"] == "#7d2a72"
    assert BUILTINS["bedroom-pop"]["bg"] == "dots"
    assert get_style("scrapbook") == {
        "id": "scrapbook", "name": "Scrapbook", "builtin": True, "recipe": BUILTINS["scrapbook"],
    }
    with pytest.raises(ValueError, match="nothing-here"):
        get_style("nothing-here")


def test_validate_recipe_rejects_unknown_id():
    # An id that is not an option of its layer is refused, and the message starts with the layer.
    for layer in ("bg", "edge", "notes", "arrows", "hero"):
        with pytest.raises(ValueError, match=rf"^{layer}\b") as error:
            validate_recipe(recipe_with(**{layer: "glitter"}))
        assert "glitter" in str(error.value)
    with pytest.raises(ValueError, match="^bg"):
        validate_recipe(recipe_with(bg=["dots"]))  # a layer takes one id, never a list
    # The ids of another layer do not count: "marker" is a notes style, not a background.
    with pytest.raises(ValueError, match="^bg"):
        validate_recipe(recipe_with(bg="marker"))

    # extras has exactly tape, stars and grain, each true or false.
    for extras in (
        {"tape": False, "stars": False},
        {"tape": False, "stars": False, "grain": True, "glitter": True},
        {"tape": False, "stars": False, "grain": "yes"},
        {"tape": False, "stars": False, "grain": 1},
        None,
    ):
        with pytest.raises(ValueError, match="^extras"):
            validate_recipe(recipe_with(extras=extras))

    # palette has exactly the 5 roles, each #rrggbb in any case.
    trimmed = {role: color for role, color in POP["palette"].items() if role != "ink"}
    for palette in (trimmed, {**POP["palette"], "accent": "#ff0000"}, None):
        with pytest.raises(ValueError, match="^palette"):
            validate_recipe(recipe_with(palette=palette))
    for bad in ("#fff", "ffffff", "#gggggg", "#ffffffff", "red", None, 16777215):
        with pytest.raises(ValueError, match=r"^palette\.dark"):
            validate_recipe(recipe_with(palette={**POP["palette"], "dark": bad}))
    shouting = {role: color.upper() for role, color in POP["palette"].items()}
    validate_recipe(recipe_with(palette=shouting))

    # A name is not empty, and the recipe is an object with every key.
    for name in ("", "   ", None):
        with pytest.raises(ValueError, match="^style name"):
            validate_recipe(recipe_with(name=name))
    with pytest.raises(ValueError, match="recipe object"):
        validate_recipe(["not", "a", "recipe"])
    for key in ("id", "name", "bg", "edge", "notes", "arrows", "hero", "extras", "palette"):
        recipe = deepcopy(POP)
        del recipe[key]
        with pytest.raises(ValueError, match=re.escape(f"missing: {key}")):
            validate_recipe(recipe)

    # The project file asks the same questions of its style, now through the real check.
    clip = {"path": "clip.mp4", "stem": "clip", "width": 1920, "height": 1080, "fps": 30.0,
            "duration": 4.0, "frames": 120, "orientation": "landscape"}
    project = default_project(clip, "landscape", (1080, 608), POP)
    assert validate(project) == []
    project["style"]["bg"] = "glitter"
    with pytest.raises(ValueError, match="^bg"):
        validate(project)

    # Every option of every layer is built, so a valid recipe never raises NotImplementedError. An id
    # that is no option at all is a ValueError, and unavailable() is what tells the two apart.
    assert set(background._KINDS) == set(OPTIONS["bg"]) and set(edges._KINDS) == set(OPTIONS["edge"])
    assert set(notes._NOTES) == set(notes.NOTE_STYLES) == set(OPTIONS["notes"])
    assert {*notes._ARROWS, "none"} == set(OPTIONS["arrows"])
    assert set(notes._HEROES) == set(engine._HERO_RULES) == set(OPTIONS["hero"])
    assert isinstance(unavailable("bg", "dots"), NotImplementedError)
    assert isinstance(unavailable("hero", "glitter"), ValueError)
    rgb, alpha, pal = np.zeros((8, 8, 3), np.float32), np.zeros((8, 8), np.float32), POP["palette"]
    with pytest.raises(ValueError, match="glitter"):
        background.render("glitter", rgb, alpha, pal, 0, None)
    with pytest.raises(ValueError, match="glitter"):
        edges.render("glitter", alpha, pal, 0)


def test_user_style_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_HOME", str(tmp_path))
    assert user_dir() == tmp_path / "styles"
    assert [style["id"] for style in list_styles()] == list(BUILTINS)  # no folder yet is no styles

    recipe = {**deepcopy(BUILTINS["comic-pop"]), "bg": "flat"}
    saved = save_user_style(recipe, "My Neon!")
    assert saved["id"] == "u-my-neon" and saved["name"] == "My Neon!" and saved["builtin"] is False
    assert recipe["id"] == "comic-pop"  # the caller's recipe is left as it was

    # One file for the style, holding the id, the name and the recipe with that same id and name.
    stored_recipe = {**recipe, "id": "u-my-neon", "name": "My Neon!"}
    path = tmp_path / "styles" / "my-neon.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "id": "u-my-neon", "name": "My Neon!", "recipe": stored_recipe,
    }
    assert saved["recipe"] == stored_recipe
    assert list(path.parent.iterdir()) == [path]  # no temporary file left behind

    # Built-ins first, in the table's order, then the person's styles sorted by name (ignoring case).
    save_user_style(BUILTINS["scrapbook"], "alpha")
    save_user_style(BUILTINS["player-select"], "Zed")
    assert BUILTINS["scrapbook"]["id"] == "scrapbook" and BUILTINS["player-select"]["name"] == "Player Select"
    listed = list_styles()
    assert [style["id"] for style in listed] == [*BUILTINS, "u-alpha", "u-my-neon", "u-zed"]
    assert [style["builtin"] for style in listed] == [True] * 4 + [False] * 3
    assert listed[5] == {"id": "u-my-neon", "name": "My Neon!", "builtin": False, "recipe": stored_recipe}
    assert get_style("u-my-neon") == listed[5]

    # The same slug overwrites, whatever the spelling of the name: "my neon" is "u-my-neon" too.
    again = save_user_style(POP, "my neon")
    assert again["id"] == "u-my-neon"
    listed = list_styles()
    assert [style["id"] for style in listed] == [*BUILTINS, "u-alpha", "u-my-neon", "u-zed"]
    assert listed[5]["name"] == "my neon" and listed[5]["recipe"]["bg"] == "dots"

    # Slugs: lowercase a-z and 0-9, every other run becomes one hyphen, none at the ends.
    for name, slug in (
        ("  --Cafe  Leo!! ", "cafe-leo"),
        ("Caf" + E_ACUTE + " L" + E_ACUTE + "o", "caf-l-o"),  # letters outside a-z are other characters
        ("A_b 2", "a-b-2"),
        ("!!!", "style"),  # nothing left: the slug is "style"
    ):
        assert save_user_style(POP, name)["id"] == f"u-{slug}"
        assert (tmp_path / "styles" / f"{slug}.json").is_file()

    # A bad recipe or an empty name is refused, and nothing is written for it.
    before = sorted(path.name for path in (tmp_path / "styles").iterdir())
    with pytest.raises(ValueError, match="^bg"):
        save_user_style(recipe_with(bg="glitter"), "Broken")
    with pytest.raises(ValueError, match="name"):
        save_user_style(POP, "  ")
    assert sorted(path.name for path in (tmp_path / "styles").iterdir()) == before

    # A file that is not a style of the person's is skipped with a warning, never fatal: one bad
    # file must not take the whole style picker down.
    (tmp_path / "styles" / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "styles" / "renamed.json").write_text(
        json.dumps({"id": "u-other", "name": "Other", "recipe": stored_recipe}), encoding="utf-8"
    )
    with pytest.warns(UserWarning, match=r"\.json") as record:
        ids = [style["id"] for style in list_styles()]
        with pytest.raises(ValueError):
            get_style("u-other")
    assert "u-other" not in ids and "u-broken" not in ids and "u-my-neon" in ids
    said = " ".join(str(warning.message) for warning in record)
    assert "broken.json" in said and "renamed.json" in said


def test_saving_a_style_waits_out_a_file_lock(tmp_path, monkeypatch):
    # The gallery reads the saved styles while "Save as my style" writes one, and Windows refuses to
    # replace a file that is being read, for a moment. The save waits that out (fileio.retry_locked)
    monkeypatch.setenv("MC_HOME", str(tmp_path))
    real, refusals, pauses = os.replace, [], []

    def held_for_a_moment(source, destination):
        if len(refusals) < 2:
            refusals.append(destination)
            raise PermissionError(5, "Access is denied")
        return real(source, destination)

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", held_for_a_moment)
        patched.setattr(time, "sleep", pauses.append)
        saved = save_user_style(POP, "Night Owl")
    assert saved["id"] == "u-night-owl" and len(refusals) == 2 and pauses == [0.1, 0.1]
    assert get_style("u-night-owl")["name"] == "Night Owl"
    assert [path.name for path in (tmp_path / "styles").iterdir()] == ["night-owl.json"]

    # A lock that does not clear in ten tries is an error that names the file. The style that was
    # there is whole, and the temporary file is gone
    def held_for_good(source, destination):
        raise PermissionError(5, "Access is denied")

    pauses.clear()
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", held_for_good)
        patched.setattr(time, "sleep", pauses.append)
        with pytest.raises(PermissionError, match=r"night-owl\.json is in use by another program"):
            save_user_style(NO_GRAIN, "Night Owl")
    assert pauses == [0.1] * 9
    assert get_style("u-night-owl")["recipe"]["extras"]["grain"] is True
    assert [path.name for path in (tmp_path / "styles").iterdir()] == ["night-owl.json"]


def test_a_style_name_with_no_letters_or_digits_gets_a_file_of_its_own(tmp_path, monkeypatch):
    # "!!!" has nothing left for a slug but "style", and so does a name in another alphabet or a row of
    # stars: they used to be one file, and each save overwrote the one before
    monkeypatch.setenv("MC_HOME", str(tmp_path))
    names = ["!!!", "***", chr(0x2605) * 3, chr(0x65E5) + chr(0x672C) + chr(0x8A9E)]

    saved = [save_user_style(POP, name) for name in names]

    assert [style["id"] for style in saved] == ["u-style", "u-style-2", "u-style-3", "u-style-4"]
    assert sorted(path.name for path in (tmp_path / "styles").iterdir()) == [
        "style-2.json", "style-3.json", "style-4.json", "style.json"]
    assert [get_style(style["id"])["name"] for style in saved] == names  # each is still there, as itself

    # saving a name again is that style again, not a new one
    again = save_user_style(recipe_with(bg="flat"), "***")
    assert again["id"] == "u-style-2" and get_style("u-style-2")["recipe"]["bg"] == "flat"
    assert len(list((tmp_path / "styles").iterdir())) == 4

    # and a style that really is called "Style" is not overwritten by a name with no letters in it
    other = tmp_path / "other"
    monkeypatch.setenv("MC_HOME", str(other))
    assert save_user_style(POP, "Style")["id"] == "u-style"
    assert save_user_style(POP, "???")["id"] == "u-style-2"
    assert get_style("u-style")["name"] == "Style" and get_style("u-style-2")["name"] == "???"
    # (and the ids still match their file names, so the list shows both)
    assert [style["name"] for style in list_styles() if not style["builtin"]] == ["???", "Style"]


def test_a_long_style_name_makes_a_slug_of_about_60_characters(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_HOME", str(tmp_path))
    name = "The Midnight Arcade Remix Of My Favorite Bedroom Pop Look For The Spring Reels " * 3

    saved = save_user_style(POP, name)

    slug = saved["id"].removeprefix("u-")
    assert 50 <= len(slug) <= 60 and slug.startswith("the-midnight-arcade-remix") and not slug.endswith("-")
    assert (tmp_path / "styles" / f"{slug}.json").is_file()
    assert get_style(saved["id"])["name"] == name  # the name is kept whole
    # the cut never leaves a hyphen at the end: this name has a space where the 60th character is
    spaced = save_user_style(POP, "a" * 59 + " tail")
    assert spaced["id"] == "u-" + "a" * 59


def test_the_cut_check_of_a_note_the_planner_did_not_plan_is_the_planners_own(person_alpha):
    # The handle that goes under the burst is set apart from the planner's notes. A double space in it
    # is the same text, not a text that was cut (it used to be judged by comparing the strings)
    alpha = person_alpha(1080, 608)
    copy = deepcopy(SAMPLE_COPY)
    copy["notes"]["tr"] = "@your  handle"
    boxes = engine.style_layout(BUILTINS["comic-pop"], alpha, copy)

    assert boxes["tr"].lines == ["@YOUR HANDLE"] and not boxes["tr"].cut and not boxes["tr"].clash

    # and a text that is too long for three lines is cut, as it always was
    copy["notes"]["tr"] = "word " * 60
    assert engine.style_layout(BUILTINS["comic-pop"], alpha, copy)["tr"].cut


def test_fit_note_cut_is_fit_note_and_whether_the_text_was_cut():
    style = NoteStyle("marker", 56, 36, "upper", band=22)
    lines, size, w, h, cut = layout.fit_note_cut("Creator  &  AI educator", style, 454)

    assert (lines, size, w, h) == layout.fit_note("Creator  &  AI educator", style, 454)
    assert cut is False
    assert layout.fit_note_cut("word " * 60, style, 454)[4] is True


# --- the stage --------------------------------------------------------------------------------


def test_compose_stage_edge_color(fake_project):
    rgb, alpha = stage_inputs(fake_project)
    edge = hex_rgb(POP["palette"]["edge"])

    # 10 px above the top of the head is well inside the 22 px sticker edge (which wobbles by 7).
    column = 540
    top = int(np.flatnonzero(alpha[:, column] > 0.5)[0])
    assert top >= 20, "the synthetic head must leave room above it"
    row = top - 10

    stage = compose_stage(POP, rgb, alpha, 0)
    assert stage.shape == (608, 1080, 3) and stage.dtype == np.float32
    assert 0.0 <= stage.min() and stage.max() <= 1.0
    assert np.allclose(stage[row, column], edge, atol=0.08)
    patch = stage[row - 3 : row + 4, column - 3 : column + 4].reshape(-1, 3)
    assert np.allclose(patch.mean(axis=0), edge, atol=0.03)  # the film grain averages out

    # The edge reaches out 22 px from the head, give or take the 7 px it wobbles, and it wobbles
    # differently in each variant. (Measured up the column from the top of the head, where the
    # edge runs straight up.)
    def thickness(variant):
        bare = compose_stage(NO_GRAIN, rgb, alpha, variant)
        # Start 3 px above the head, past the 3 px ramp of its own matte, and count rows of edge up
        is_edge = np.abs(bare[: top - 3, column] - edge).max(axis=-1) < 0.02
        run = 0
        for flag in is_edge[::-1]:
            if not flag:
                break
            run += 1
        return run + 3

    widths = [thickness(variant) for variant in range(3)]
    assert all(14 <= width <= 30 for width in widths), widths
    assert len(set(widths)) > 1, widths

    # Without the grain the edge is exactly its color, and the person is the frame with its
    # saturation raised by 8% (m + (p - m) * 1.08 around the mean of the channels).
    plain = compose_stage(NO_GRAIN, rgb, alpha, 0)
    assert np.allclose(plain[row, column], edge, atol=1e-5)
    mean = rgb.mean(axis=-1, keepdims=True)
    boosted = np.clip(mean + (rgb - mean) * 1.08, 0.0, 1.0)
    inside = alpha >= 1.0
    assert inside.sum() > 10_000
    assert np.abs(boosted - rgb)[inside].max() > 0.01, "the frame must have color to boost"
    assert np.allclose(plain[inside], boosted[inside], atol=1e-5)
    empty = compose_stage(NO_GRAIN, rgb, np.zeros_like(alpha), 0)  # nobody in the matte
    assert np.abs(empty - edge).max(axis=-1).min() > 0.1, "no sticker edge without a person"

    # The room is magenta: a corner far from the person sits between the texture and the light
    # color, red and blue over green.
    corner = plain[:60, :60].reshape(-1, 3)
    low, high = hex_rgb(POP["palette"]["tex"]), hex_rgb(POP["palette"]["light"])
    assert (corner.mean(axis=0) > low - 0.05).all() and (corner.mean(axis=0) < high + 0.05).all()
    assert corner[:, 0].mean() > corner[:, 1].mean() + 0.1 and corner[:, 2].mean() > corner[:, 1].mean() + 0.1

    # The grain extra only moves pixels, softly: the stage with it differs a little from the stage
    # without it.
    shift = np.abs(stage - plain)
    assert 0.002 < shift.mean() < 0.03


def test_boil_variants_differ(fake_project):
    rgb, alpha = stage_inputs(fake_project)
    stages = [compose_stage(POP, rgb, alpha, variant) for variant in range(3)]

    # Each variant has its own halftone phase, grain and hand-cut edge, so the card boils.
    for first, second in ((0, 1), (1, 2), (0, 2)):
        assert np.abs(stages[first] - stages[second]).mean() > 0.002, (first, second)
    again = compose_stage(POP, rgb, alpha, 1)
    assert np.array_equal(again, stages[1])  # the same variant is the same picture, always

    # The edge is what wobbles: its outline differs from one variant to the next, by more than a
    # pixel in places, and stays within the wobble of 7 px around the radius of 22.
    outlines = [dilate(alpha, 22, wobble=7, seed=variant) for variant in range(3)]
    assert np.abs(outlines[0] - outlines[1]).max() > 0.5

    # The dots, written out: the frame's luminance picks a point on the line from the dark to the
    # light color (30% of the way at black), dots of the texture color are laid over it, and 7
    # levels of grain on top. The same seed gives the same grain, so the formula can be redone.
    pal = POP["palette"]
    lum = luminance(rgb)
    t = np.clip((lum - 0.04) / 0.8, 0, 1) ** 0.8
    base = duotone(np.clip((lum - 0.04) / 0.8, 0, 1), pal["dark"], pal["light"], lift=0.3, gamma=0.8)
    expected = over(base, hex_rgb(pal["tex"]), 0.55 * halftone(1 - t, 9, 45, 0.9, phase=(2, 2)))
    expected = grain(expected, 7, 2)
    assert np.allclose(background.render("dots", rgb, alpha, pal, 2, None), expected, atol=1e-5)
    assert np.abs(background.render("dots", rgb, alpha, pal, 0, None) - expected).mean() > 0.002


# --- the three other styles --------------------------------------------------------------------


def styled(fake_project, recipe_id, **changes):
    """The fake project drawn in a built-in style, with some fields of the recipe replaced. The
    project is a copy, so a test can change its copy and positions."""
    folder, project = fake_project
    project = deepcopy(project)
    project["style"] = {**deepcopy(BUILTINS[recipe_id]), **changes}
    return folder, project


def near(pixels, color, tolerance=8):
    """How many pixels of an (h, w, 3) uint8 picture are within `tolerance` levels of a color (a
    "#rrggbb" string or three 0-255 values) on every channel."""
    if isinstance(color, str):
        color = tuple(int(v) for v in np.rint(hex_rgb(color) * 255))
    return int((np.abs(pixels.astype(int) - np.array(color)) <= tolerance).all(axis=-1).sum())


def under(box, pixels):
    """The part of a card picture under a note's box (the card is the stage: a box is in px of it)."""
    return pixels[box.y : box.y + box.h, box.x : box.x + box.w]


def strongly(first, second, level=40):
    """Where two card pictures differ by more than `level` on some channel. (The edges of what is
    drawn ring a little, a few levels, a few px out, so "any difference" is wider than the shape.)"""
    return (np.abs(first.astype(int) - second.astype(int)).max(axis=2) > level)


def blobs(mask):
    """The groups of set pixels of a boolean picture that touch (8 neighbors), each as its box
    (x0, y0, x1, y1), x1 and y1 the last pixel, and the number of pixels, left to right."""
    todo = {(int(y), int(x)) for y, x in zip(*np.nonzero(mask))}
    found = []
    while todo:
        stack, group = [todo.pop()], []
        while stack:
            y, x = stack.pop()
            group.append((y, x))
            for neighbor in ((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                if neighbor in todo:
                    todo.remove(neighbor)
                    stack.append(neighbor)
        ys, xs = zip(*group)
        found.append((min(xs), min(ys), max(xs), max(ys), len(group)))
    return sorted(found)


def ink_near_box(box, card, bare, reach=14):
    """Whether what a note draws, taken as where the card differs from the card without notes by
    more than 40 levels, lies within `reach` px of the box the layout gave it. That box is not
    tilted, and the studio drags it, so a tilt must not carry the ink far from it."""
    top, left = max(box.y - 40, 0), max(box.x - 40, 0)
    window = (slice(top, box.y + box.h + 40), slice(left, box.x + box.w + 40))
    rows, columns = np.nonzero(strongly(card[window], bare[window]))
    return bool(
        rows.min() + top >= box.y - reach and rows.max() + top < box.y + box.h + reach
        and columns.min() + left >= box.x - reach and columns.max() + left < box.x + box.w + reach
    )


def test_comic_layout_has_burst_above_tr(fake_project):
    folder, project = styled(fake_project, "comic-pop")
    boxes = elements(folder, project)
    assert sorted(boxes) == ["bl", "br", "burst", "tl", "tr"]
    burst, tr = boxes["burst"], boxes["tr"]

    # The burst is 310 x 220 at the top of the top right room (its quadrant), and the tr note is
    # under it, centered on it, with no arrow: the burst takes the corner. Nothing is in the way.
    alpha = matte.load_alpha(folder, project, FRAME)
    quad = engine.matte_context(project["style"], alpha).quads["tr"]
    assert (burst.w, burst.h, burst.size, burst.lines) == (310, 220, 66, ["ROBIN!"])
    assert burst.y == quad[1] and burst.x + burst.w == quad[2] and burst.arrow is None
    assert burst.y + burst.h <= tr.y and tr.arrow is None
    assert abs((tr.x + tr.w / 2) - (burst.x + burst.w / 2)) <= 1
    assert not any(box.clash for box in boxes.values())

    # The other notes keep their arrows, found again once the burst and the note under it were
    # there, so no arrow runs through either of them
    assert [key for key, box in boxes.items() if box.arrow is None] == ["tr", "burst"]
    for key, box in boxes.items():
        for x, y in quad_points(*box.arrow, 16) if box.arrow else ():
            for other in (burst, tr):
                inside = other.x <= x < other.x + other.w and other.y <= y < other.y + other.h
                assert not inside, (key, other.key)

    # The notes are Bangers capitals at one size, from 46 down to 36 (the floor) in steps of 2
    sizes = {box.size for key, box in boxes.items() if key != "burst"}
    assert len(sizes) == 1 and sizes.pop() in range(NOTE_MIN_PX, 47, 2)
    assert " ".join(boxes["bl"].lines) == "BUT OVERTHINKS EVERY 6-SECOND REEL"

    # A burst that was dragged is where it was put, and the note under it follows. A tr note
    # that was dragged stays where it was put, and the burst keeps the corner.
    project["positions"]["burst"] = [100, 200]
    moved = elements(folder, project)
    assert (moved["burst"].x, moved["burst"].y) == (100, 200)
    assert moved["tr"].y == 200 + 220 + 16 and moved["tr"].arrow is None
    project["positions"]["burst"], project["positions"]["tr"] = None, [650, 400]
    pinned = elements(folder, project)
    assert (pinned["tr"].x, pinned["tr"].y) == (650, 400) and pinned["tr"].arrow is None
    assert (pinned["burst"].x, pinned["burst"].y) == (burst.x, burst.y)
    # A tr note that was dragged onto the burst is in its way, and the burst in the note's: both say so
    project["positions"]["tr"] = [burst.x + 40, burst.y + 60]
    on_burst = elements(folder, project)
    assert on_burst["tr"].clash and on_burst["burst"].clash
    project["positions"]["tr"] = None

    # The layout groups: the burst comes with the top notes, the HUD's two labels one with each
    assert [notes.group(key) for key in ("tl", "tr", "burst", "hud_top")] == ["top"] * 4
    assert [notes.group(key) for key in ("bl", "br", "hud_bottom")] == ["bottom"] * 3
    with pytest.raises(ValueError, match="middle"):
        notes.group("middle")


def test_burst_may_use_the_air_of_the_face_box(fake_project, person_alpha, tmp_path):
    # The burst is judged against the head box without its air, as the panel of stats is (see
    # engine._tight): it may sit on the air around the face when it cannot fit otherwise, never on the
    # head. The fixture's person is moved 92 px to the right, so that its head's columns end at 739:
    # the face box, with 43 px of air, ends at 783, and the head box at 740. The burst's home is the top
    # right corner (x 727), which is on the face box, and it is moved as far right as the stage lets it
    # (1080 - 12 - 310 = 758): off the head, on the air.
    folder, project = styled(fake_project, "comic-pop")
    near_right = tmp_path / "near_right"
    shutil.copytree(folder, near_right)
    moved = np.roll(person_alpha(1080, 608), 92, axis=1)
    Image.fromarray(np.rint(moved * 255).astype(np.uint8)).save(near_right / "matte" / f"{FRAME:06d}.png")
    context = engine.matte_context(project["style"], matte.load_alpha(near_right, project, FRAME))
    head = engine._tight(context)
    assert (context.face[2], head.face[2]) == (783, 740)

    boxes = elements(near_right, project)
    burst, tr = boxes["burst"], boxes["tr"]
    assert (burst.x, burst.y) == (1080 - 12 - 310, context.quads["tr"][1])
    assert head.face[2] <= burst.x < context.face[2]  # on the air, off the head
    assert not any(box.clash for box in boxes.values())
    # and the note under it, which goes where the burst goes, is not in the way for being on the air either
    assert tr.y == burst.y + burst.h + 16 and tr.x < context.face[2] and not tr.clash
    # The control: judged against the whole face box, as it was, the burst is in the way
    assert layout.clash_of(burst, context, [], (1080, 608))

    # It is still flagged on the head: a burst that was dragged onto the head is in the way, and one
    # dragged to the air beside it is not
    for position, flagged in (([400, 100], True), ([head.face[2] + 1, 100], False)):
        project["positions"]["burst"] = position
        assert elements(near_right, project)["burst"].clash == flagged, position
    project["positions"]["burst"] = None

    # The overlap that moves it is a real one, of the burst's box with the face box. Where the face box
    # begins inside the burst's width (the head at the far right, its columns from 812: the face box
    # from 769) the full-size burst, which the old test left at x 727 because its left edge was clear of
    # the face, is moved right of that spot too. It cannot clear the head from there, and shrinks as
    # far as it goes (see test_burst_shrinks_to_fit), and still says so.
    far = tmp_path / "far_right"
    shutil.copytree(folder, far)
    farther = np.roll(person_alpha(1080, 608), 380, axis=1)
    Image.fromarray(np.rint(farther * 255).astype(np.uint8)).save(far / "matte" / f"{FRAME:06d}.png")
    wide = engine.matte_context(project["style"], matte.load_alpha(far, project, FRAME))
    assert 727 < wide.face[0] < 727 + 310
    assert engine._burst_spot(wide, notes.burst_box(), {}) == (758, wide.quads["tr"][1])
    sitting = elements(far, project)["burst"]
    assert sitting.clash and (sitting.w, sitting.h) == (248, 176)


def test_burst_shrinks_to_fit(fake_project, person_alpha, tmp_path):
    # Where the full-size burst (310 x 220) would be on the head, it shrinks, the box and its lettering
    # together, to the first of 0.95, 0.90, 0.85 and 0.80 of its size at which it is not (it moves right
    # as it narrows, since it is right-aligned to the room, and gets shorter). Where none is, it stays at
    # 0.80 and is flagged. The fixture's person is moved right by 92 to 300 px, so that the head box ends
    # at x 740, 758 (where the full-size burst sits), 768, 783, 798, 813, 838 and 948.
    scales = (1.0, 0.95, 0.90, 0.85, 0.80)
    assert [notes.burst_box(scale) for scale in scales] == [(310, 220), (294, 209), (279, 198), (264, 187), (248, 176)]
    assert [notes.burst_size("ROBIN!", scale) for scale in scales] == [66, 63, 59, 56, 53]

    folder, project = styled(fake_project, "comic-pop")
    recipe, copy = project["style"], project["copy"]

    def laid_out(shift):
        moved_to = tmp_path / f"moved_{shift}"
        shutil.copytree(folder, moved_to, dirs_exist_ok=True)
        moved = np.roll(person_alpha(1080, 608), shift, axis=1)
        Image.fromarray(np.rint(moved * 255).astype(np.uint8)).save(moved_to / "matte" / f"{FRAME:06d}.png")
        head = engine._tight(engine.matte_context(recipe, matte.load_alpha(moved_to, project, FRAME))).face
        return moved_to, head, elements(moved_to, project)

    # shift: (the head box's right side, the burst's x, width, height, lettering, flagged)
    expected = {
        92: (740, 758, 310, 220, 66, False),   # it clears the head as it is
        110: (758, 758, 310, 220, 66, False),  # exactly at its edge
        120: (768, 774, 294, 209, 63, False),
        135: (783, 789, 279, 198, 59, False),
        150: (798, 804, 264, 187, 56, False),
        165: (813, 820, 248, 176, 53, False),  # the last one that fits
        190: (838, 820, 248, 176, 53, True),   # none does: 0.80 and the flag
        300: (948, 820, 248, 176, 53, True),
    }
    drawn = {}
    for shift, (head_right, x, w, h, size, flagged) in expected.items():
        where, head, boxes = laid_out(shift)
        burst = boxes["burst"]
        assert head[2] == head_right, shift
        assert (burst.x, burst.w, burst.h, burst.size, burst.clash) == (x, w, h, size, flagged), shift
        assert burst.y == 43 and burst.lines == ["ROBIN!"], shift  # still at the top of the room, saying the same
        if not flagged:  # it is off the head, and so is every other box
            assert burst.x >= head[2] or burst.y + burst.h <= head[1], shift
            assert not any(box.clash for box in boxes.values()), shift
        if boxes["tr"].y == burst.y + burst.h + 16:  # the tr note goes under the burst that is there
            assert abs(boxes["tr"].x + boxes["tr"].w / 2 - (burst.x + burst.w / 2)) <= 1, shift
        layer = Image.new("RGBA", (1080, 608), (0, 0, 0, 0))
        notes.draw_hero(layer, boxes, copy, recipe, ALL_VISIBLE)
        rows, columns = np.nonzero(np.asarray(layer)[..., 3] > 0)
        drawn[shift] = (columns.max() - columns.min() + 1, rows.max() - rows.min() + 1)
        assert burst.x <= columns.min() and columns.max() < burst.x + w + 6, shift  # the star is in its box

    # The star is drawn at the scale of its box: its ink is that much of the full-size star's, its points
    # and its outline and the lettering with it
    full_w, full_h = drawn[92]
    for shift, scale in ((120, 294 / 310), (135, 279 / 310), (150, 264 / 310), (165, 248 / 310)):
        assert abs(drawn[shift][0] / full_w - scale) < 0.02, shift
    assert abs(drawn[135][1] / full_h - 198 / 220) < 0.02

    # A burst that was dragged keeps its full size, and where it was put (it is flagged on the head)
    project["positions"]["burst"] = [400, 100]
    where, head, _ = laid_out(135)
    dragged = elements(where, project)["burst"]
    assert (dragged.x, dragged.y, dragged.w, dragged.h, dragged.size) == (400, 100, 310, 220, 66)
    assert dragged.clash


def test_burst_defaults_to_name(fake_project):
    folder, project = styled(fake_project, "comic-pop")
    extras = project["copy"]["extras"]

    # The burst says its own text, in capitals, or else the name and "!" (the sample's is "Robin")
    assert extras["burst"] == "Robin!"
    assert notes.burst_text(project["copy"]) == "ROBIN!"
    extras["burst"] = "claude code"
    assert elements(folder, project)["burst"].lines == ["CLAUDE CODE"]
    for unset in (None, "", "   "):
        extras["burst"] = unset
        assert elements(folder, project)["burst"].lines == ["ROBIN!"], repr(unset)
    project["copy"]["name"] = "Ana Maria"
    assert elements(folder, project)["burst"].lines == ["ANA MARIA!"]

    # and it is drawn with that text: another text is another picture, in the burst's box only
    project["copy"]["name"] = "Robin"
    box = elements(folder, project)["burst"]
    drawn = np.asarray(render_card(folder, project))
    extras["burst"] = "ok!"
    other = np.asarray(render_card(folder, project))
    changed = (drawn != other).any(axis=2)
    rows, columns = np.nonzero(changed)
    assert changed.sum() > 500
    assert rows.min() >= box.y and rows.max() < box.y + box.h
    assert columns.min() >= box.x and columns.max() < box.x + box.w


def test_comic_pop_renders(fake_project):
    folder, project = styled(fake_project, "comic-pop")
    recipe, pal = project["style"], project["style"]["palette"]
    image = render_card(folder, project)
    assert image.size == (1080, 608) and image.mode == "RGB"
    pixels = np.asarray(image)

    # The room, written out the plain way. Over a white frame there are no dots, so the stage is
    # the wedges alone: 12 degrees each, around the head's center, alternately the light color and
    # the light color 40% toward white (the middle of every one of the 30 wedges is checked).
    light, paler = hex_rgb(pal["light"]), hex_rgb(mix(pal["light"], "#ffffff", 0.4))
    nobody = np.zeros((608, 1080), np.float32)
    head = (400, 60, 680, 260)  # centered on (540, 160)
    wedges = background.render("comic", np.ones((608, 1080, 3), np.float32), nobody, pal, 0, head)
    for k in range(30):
        angle = math.radians(-180 + 12 * k + 6)
        x, y = round(540 + 120 * math.cos(angle)), round(160 + 120 * math.sin(angle))
        assert np.allclose(wedges[y, x], light if k % 2 == 0 else paler, atol=1e-5), k
    # With nobody in the picture the wedges turn around the middle of the stage instead (the
    # points are off the lattice of the dots: even a dot of nothing leaves a half pixel there)
    centered = background.render("comic", np.ones((608, 1080, 3), np.float32), nobody, pal, 0, None)
    assert np.allclose(centered[303, 540 + 101], light) and np.allclose(centered[305, 540 - 101], paler)

    # Over mid-gray the tex-colored Ben-Day dots (16 px apart, on the straight lattice, 0.85 of
    # tex at the middle of a dot) show on both colors of wedge, and the lattice slides a pixel a
    # variant: the middle of a dot is at (0, 0) in variant 0 and at (1, 1) in variant 1
    gray = np.full((608, 1080, 3), 0.4, np.float32)
    tex = hex_rgb(pal["tex"])
    dotted = [background.render("comic", gray, nobody, pal, variant, head) for variant in (0, 1)]
    for base in (light, paler):
        dot_middle = base * 0.15 + tex * 0.85
        assert (np.abs(dotted[0] - dot_middle).max(axis=-1) < 0.01).sum() > 200
    assert np.allclose(dotted[0][0, 0], paler * 0.15 + tex * 0.85, atol=1e-5)  # (0, 0) is in a pale wedge
    assert np.allclose(dotted[1][1, 1], paler * 0.15 + tex * 0.85, atol=1e-5)

    # The edge is two lines around the person: white up to 12 px out, then black up to 19 px out
    rgb, alpha = stage_inputs(fake_project)
    stage = compose_stage(recipe, rgb, alpha, 0)
    top = int(np.flatnonzero(alpha[:, 540] > 0.5)[0])
    assert np.allclose(stage[top - 6, 540], hex_rgb(pal["edge"]), atol=1e-4)
    assert np.allclose(stage[top - 16, 540], hex_rgb(pal["dark"]), atol=1e-4)
    outside = stage[top - 26, 540]
    for line in ("edge", "dark"):
        assert np.abs(outside - hex_rgb(pal[line])).max() > 0.1, line

    # The notes are boxes of the light color toward white, with a 5 px dark border and #111 text,
    # tilted a few degrees. Their ink stays within a hand of the box the layout gave them.
    boxes = elements(folder, project)
    fill = mix(pal["light"], "#ffffff", 0.45)
    bare = np.asarray(render_card(folder, project, frozenset()))
    only_bottom = np.asarray(render_card(folder, project, frozenset({"bottom"})))
    only_top = np.asarray(render_card(folder, project, frozenset({"top"})))
    for key in ("bl", "br"):
        area = under(boxes[key], pixels)
        assert near(area, fill) > 0.4 * boxes[key].w * boxes[key].h, key
        assert near(area, pal["dark"]) > 2 * (boxes[key].w + boxes[key].h) * 4, key
        assert ink_near_box(boxes[key], only_bottom, bare), key
    assert ink_near_box(boxes["tl"], only_top, bare)

    # The burst: a white star with a black outline and the name in the tex color, drawn with the
    # top notes (frame 2 of the card) and not before
    burst = boxes["burst"]
    star = under(burst, pixels)
    assert near(star, "#ffffff") > 18_000 and near(star, "#111111") > 2_500 and near(star, pal["tex"]) > 1_500
    assert not np.array_equal(star, under(burst, only_bottom)) and np.array_equal(
        under(burst, only_bottom), under(burst, bare)
    )
    on_top = np.asarray(render_card(folder, project, frozenset({"top"})))
    assert np.array_equal(under(burst, on_top), star)


# --- Scrapbook --------------------------------------------------------------------------------


def test_hand_notes_keep_natural_case(fake_project):
    folder, project = styled(fake_project, "scrapbook")
    boxes = elements(folder, project)
    assert sorted(boxes) == ["bl", "br", "tl", "tr"]

    # The handwritten notes keep the case the copy was written in (the lines are the copy's own
    # words, broken to fit), at one size from 52 down to 36 in steps of 2
    assert " ".join(boxes["bl"].lines) == "But overthinks every 6-second reel"
    for key, text in project["copy"]["notes"].items():
        assert " ".join(boxes[key].lines) == text, key
    sizes = {box.size for box in boxes.values()}
    assert len(sizes) == 1 and sizes.pop() in range(36, 53, 2)
    assert not any(box.clash for box in boxes.values())

    # Each note is a torn strip of #fffdf5 (its top and bottom edges zigzag) with the text in the
    # ink color: most of the box is paper, and there is ink in it
    pal = project["style"]["palette"]
    pixels = np.asarray(render_card(folder, project))
    for key, box in boxes.items():
        area = under(box, pixels)
        assert near(area, "#fffdf5") > 0.55 * box.w * box.h, key
        assert near(area, pal["ink"]) > 150, key

    # They are tilted a few degrees (and cast a shadow), and the ink stays within a hand of the
    # box the layout gave them, which is not tilted. (The tape and the stars are off here.)
    project["style"]["extras"] = {"tape": False, "stars": False, "grain": False}
    bare = np.asarray(render_card(folder, project, frozenset()))
    shown = {"bottom": ("bl", "br"), "top": ("tl", "tr")}
    for group, keys in shown.items():
        card = np.asarray(render_card(folder, project, frozenset({group})))
        for key in keys:
            assert ink_near_box(boxes[key], card, bare), key


def test_emphasis_missing_word_is_ignored(fake_project):
    folder, project = styled(fake_project, "scrapbook")
    extras = project["copy"]["extras"]
    box = elements(folder, project)["bl"]
    assert extras["emphasis"] == "overthinks"

    def card_with(word):
        extras["emphasis"] = word
        return np.asarray(render_card(folder, project))

    # A word that is not in the bl note, one that is only a part of a word of it, and no word at
    # all draw no underline and raise nothing: the card is the card without one
    plain = card_with(None)
    for missing in ("zebra", "overthink", "reel.", "overthinks e", "", "   "):
        assert np.array_equal(card_with(missing), plain), repr(missing)

    # The word that is there is underlined, whatever its case: a wavy line of the texture color,
    # 4 px thick, under that word alone (the first line of the note, which begins "But overthinks")
    # and nowhere else on the card
    fnt = font("hand", box.size)
    assert box.lines[0].lower().startswith("but overthinks") and len(box.lines) > 1
    left = box.x + box.w / 2 - max(fnt.getlength(text) for text in box.lines) / 2
    start, end = left + fnt.getlength("But "), left + fnt.getlength("But overthinks")
    for word in ("overthinks", "OVERTHINKS", "Overthinks"):
        marked = card_with(word)
        changed = strongly(marked, plain)
        rows, columns = np.nonzero(changed)
        assert changed.sum() > 200, word
        assert columns.min() >= start - 12 and columns.max() <= end + 12, word  # under the word
        assert columns.max() - columns.min() >= (end - start) - 12, word  # all of it
        assert rows.max() - rows.min() <= 18, word  # a wave and a line, tilted, a few px high
        assert rows.min() >= box.y + 0.3 * box.h, word  # under the first line, not above it
        assert near(marked[changed], project["style"]["palette"]["tex"], 12) > 0.6 * changed.sum(), word

    # A phrase that runs over the end of a line is underlined on both lines
    phrase = f"{box.lines[0].split()[-1]} {box.lines[1].split()[0]}"
    rows = np.flatnonzero(strongly(card_with(phrase), plain).any(axis=1))
    assert rows.max() - rows.min() > 0.8 * box.size, phrase


def test_scrapbook_renders(fake_project):
    folder, project = styled(fake_project, "scrapbook")
    recipe, pal = project["style"], project["style"]["palette"]
    image = render_card(folder, project)
    assert image.size == (1080, 608) and image.mode == "RGB"
    pixels = np.asarray(image)
    nobody = np.zeros((608, 1080), np.float32)
    white = np.ones((608, 1080, 3), np.float32)
    pink = hex_rgb("#ff4f8b")

    # The paper: the light color with a smooth blotchiness of 0.035 (60 px across) and a grain of
    # 0.012. Over a white frame, with no ink, the room is that and nothing else, except for the half
    # pixel that even a dot of nothing leaves in the middle of each of its places on the lattice
    paper = hex_rgb(pal["light"]) + lowfreq((608, 1080), 60, 0.035, 1)[..., None]
    paper = grain(paper, 0.012 * 255, 1)
    flat = background.render("riso", white, nobody, pal, 1, None)
    off_lattice = halftone(np.zeros((608, 1080), np.float32), 6, 30, phase=(1, 1)) == 0
    assert off_lattice.mean() > 0.95 and np.allclose(flat[off_lattice], paper[off_lattice], atol=1e-5)

    # Dark parts of the frame are ink of the texture color in halftone dots, 6 px apart at 30
    # degrees and laid on 0.88: the middle of a dot is the paper 88% of the way to the ink
    dark = np.full((608, 1080, 3), 0.2, np.float32)
    inked = background.render("riso", dark, nobody, pal, 0, None)
    middle = hex_rgb(pal["light"]) * 0.12 + hex_rgb(pal["tex"]) * 0.88
    assert (np.abs(inked - middle).max(axis=-1) < 0.1).mean() > 0.25

    # An edge between a dark half and a light one shows the pink plate, printed 6 px right and 4
    # down: 0.22 of pink where the ink was 6 px to the left and is not there, over the light side
    halves = np.ones((608, 1080, 3), np.float32)
    halves[:, :540] = 0.2
    plates = background.render("riso", halves, nobody, pal, 0, None)
    flat_paper = background.render("riso", white, nobody, pal, 0, None)
    ink_here = 1.1 * (1 - 0.2) - 0.08  # 0.8
    wanted = flat_paper[:, 540:546] * (1 - 0.22 * ink_here) + pink * 0.22 * ink_here
    assert np.allclose(plates[:, 541:546], wanted[:, 1:], atol=1e-4)
    assert np.allclose(plates[:, 560:], flat_paper[:, 560:], atol=1e-5)

    # The edge is a paper cutout 16 px out (wobbling by 3) with a shadow under it, lower and to the
    # right: 0.4 of #2a1d10 where the cutout's shape, blurred, moved 7 right and 10 down, is
    rgb, alpha = stage_inputs(fake_project)
    stage = compose_stage(recipe, rgb, alpha, 0)
    top = int(np.flatnonzero(alpha[:, 540] > 0.5)[0])
    assert np.allclose(stage[top - 8, 540], hex_rgb(pal["edge"]), atol=1e-4)
    (shadow_color, shadow, _), (edge_color, edge, _) = edges.render("paper", alpha, pal, 0)
    assert np.allclose(shadow_color, hex_rgb("#2a1d10")) and np.allclose(edge_color, hex_rgb(pal["edge"]))
    shape = blur(dilate(alpha, 16, wobble=3, seed=0, scale=25), 9)
    moved = np.pad(shape, ((10, 0), (7, 0)), mode="edge")[:608, :1080]
    assert np.allclose(shadow, 0.4 * moved, atol=1e-5) and np.allclose(edge, dilate(alpha, 16, 3, 0, 25))
    row = int(np.flatnonzero(alpha[:, 540] > 0.5)[0]) + 100  # a row of the head, well below its top
    columns = np.flatnonzero(alpha[row] > 0.5)
    assert shadow[row, columns[-1] + 24] - shadow[row, columns[0] - 24] > 0.1  # darker on the right

    # The doodle arrows wobble: the line is never far from the curve it follows, and it is off
    # that curve (by up to 4 px) at the swings of the wobble
    boxes = elements(folder, project)
    bare = np.asarray(render_card(folder, project, frozenset()))
    only_arrows = np.asarray(render_card(folder, project, frozenset({"arrows"})))
    drawn = strongly(only_arrows, bare, 60)
    off_curve = 0
    for key, box in boxes.items():
        assert box.arrow is not None, key
        points = quad_points(*box.arrow, 24)
        for x, y in points:
            around = drawn[round(y) - 5 : round(y) + 6, round(x) - 5 : round(x) + 6]
            assert around.any(), key
            off_curve += not drawn[round(y), round(x)]
    assert off_curve >= 12
    assert near(only_arrows[drawn], pal["ink"], 40) > 0.6 * drawn.sum()  # the ink is the notes' ink

    # Tape on the top corners of tl, tr and br, 120 x 36 px at 80% in two stripes of the texture
    # color toward cream, and three stars (22, 18 and 14 px) beside the face box. Each is drawn
    # alone: the tape comes in with its note (tl and tr with the top notes, br with the bottom
    # ones) and the stars come in with the arrows
    def card_with(visible=ALL_VISIBLE, **extras):
        other = deepcopy(project)
        other["style"]["extras"] = {**recipe["extras"], **extras}
        return np.asarray(render_card(folder, other, visible))

    tape = strongly(pixels, card_with(tape=False))
    corners = {"tl": boxes["tl"].x, "tr": boxes["tr"].x + boxes["tr"].w, "br": boxes["br"].x + boxes["br"].w}
    for key, corner in corners.items():
        y = boxes[key].y
        assert 2000 < tape[max(y - 50, 0) : y + 50, max(corner - 90, 0) : corner + 90].sum() < 4400, key
    assert 3 * 2000 < tape.sum() < 3 * 4400  # and nowhere else
    stripes = [hex_rgb(mix(pal["tex"], "#fff6c8", share)) * 255 for share in (0.6, 0.75)]
    under_tape = card_with(tape=False)[tape].astype(float)
    misses = [np.abs(pixels[tape] - (0.8 * color + 0.2 * under_tape)).max(axis=-1) for color in stripes]
    assert (np.minimum(*misses) < 8).mean() > 0.85  # each pixel is one stripe or the other, at 80%
    for visible, shown in ((frozenset({"bottom"}), {"br"}), (frozenset({"top"}), {"tl", "tr"})):
        found = strongly(card_with(visible), card_with(visible, tape=False))
        for key, corner in corners.items():
            y = boxes[key].y
            on = found[max(y - 50, 0) : y + 50, max(corner - 90, 0) : corner + 90].sum() > 2000
            assert on == (key in shown), key

    face = engine.matte_context(recipe, alpha).face
    stars = strongly(pixels, card_with(stars=False))
    found = blobs(stars)
    assert len(found) == 3
    nominal = (face[0] + 20, face[2] + 3, face[2] + 0.77 * (1080 - face[2]))  # where each star starts out
    for (x0, y0, x1, y1, count), radius, x in zip(found, (22, 18, 14), nominal):
        # five points: 1.9 radii wide and 1.8 high, and the 4 px line makes it a little more
        assert abs((x1 - x0) - (1.9 * radius + 4)) <= 4 and abs((y1 - y0) - (1.8 * radius + 4)) <= 4, radius
        assert abs((x0 + x1) / 2 - x) <= 3, radius  # a star may move up or down, never sideways
        assert count > 8 * radius
    both = frozenset({"bottom", "top"})
    assert np.array_equal(card_with(both), card_with(both, stars=False))  # no stars before the arrows


# --- Player Select and the options no built-in style uses --------------------------------------


def panel_size(labels):
    """The size of the stats panel, worked out here from the mockup's CSS: a 3 px border, 14 px at
    the sides, 12 px above and below, rows 34 px tall and 8 px apart; each row a label (VT323 at 34
    px), a 14 px gap and 10 cells of 13 px with 3 px between them."""
    widest = max(font("terminal", 34).getlength(label) for label in labels)
    return (
        math.ceil(widest) + 14 + (10 * 13 + 9 * 3) + 2 * 14 + 2 * 3,
        len(labels) * 34 + (len(labels) - 1) * 8 + 12 + 12 + 2 * 3,
    )


def panel_pixels(panel, pixels, pal):
    """What the card shows of a stats panel: how many pixels are the texture color (its border), how
    many are the color of the cells that are on, and how many are the low color #ff3ea5 (the cells
    of a stat of 2 or less). The cells that are on take the one of the texture, edge and ink colors
    that stands out most from the dark one, which in Neon is the edge color (16.5:1; the ink is
    15.3 and the texture 14.4), so that the bars show in Riso and Mono, where the texture color is
    the dark one."""
    area = under(panel, pixels)
    return near(area, pal["tex"], 0), near(area, pal["edge"], 0), near(area, "#ff3ea5", 0)


def test_stats_replace_tr(fake_project):
    folder, project = styled(fake_project, "player-select")
    boxes = elements(folder, project)
    assert sorted(boxes) == ["bl", "br", "tl", "tr"]
    panel, pal = boxes["tr"], project["style"]["palette"]

    # The panel takes the place of the tr note and keeps its key, so dragging "tr" moves it. Its
    # rows are the copy's stats (in capitals, as the terminal font is set), its size comes from
    # them, and the note's own text is nowhere on the card
    rows = notes.stats_rows(project["copy"])
    wanted = [(label.upper(), value) for label, value in project["copy"]["extras"]["stats"]]
    assert rows == wanted == [("AI SKILLS", 10), ("OVERTHINKING", 10), ("CHILL", 1)]
    assert panel.lines == ["AI SKILLS", "OVERTHINKING", "CHILL"] and panel.arrow is None
    assert (panel.w, panel.h) == panel_size(panel.lines) and panel.size == 34
    assert not any("YOUR_HANDLE" in " ".join(box.lines) for box in boxes.values())
    assert not any(box.clash for box in boxes.values())
    project["positions"]["tr"] = [300, 120]
    dragged = elements(folder, project)["tr"]
    assert (dragged.x, dragged.y, dragged.lines) == (300, 120, panel.lines)
    # and one dragged onto the tl note is in its way, and is flagged
    tl = boxes["tl"]
    project["positions"]["tr"] = [tl.x + 20, tl.y + 10]
    on_note = elements(folder, project)["tr"]
    assert (on_note.x, on_note.y) == (tl.x + 20, tl.y + 10) and on_note.clash
    project["positions"]["tr"] = None

    # It is drawn from those rows. The border is 3 px of the texture color, and the cells that are
    # on (two rows of 10) are the edge color, the one that shows best on the dark color; a row of 1
    # is a stat of 2 or less and so #ff3ea5. The other 9 cells of that row are white at 15%, and the
    # panel is the dark color at 82%.
    pixels = np.asarray(render_card(folder, project))
    tex, bars, low = panel_pixels(panel, pixels, pal)
    border = panel.w * panel.h - (panel.w - 6) * (panel.h - 6)
    assert abs(tex - border) <= 0.02 * tex and abs(bars - 20 * 13 * 18) <= 0.02 * bars
    assert abs(low - 13 * 18) <= 8
    behind = np.asarray(render_card(folder, project, frozenset()))
    inside = under(panel, pixels)[3:-3, 3:-3].astype(float)
    behind_it = under(panel, behind)[3:-3, 3:-3].astype(float)
    pure = np.zeros(inside.shape[:2], bool)  # the padding above and under the rows, and between them
    for first, last in ((0, 11), (46, 53), (88, 95), (130, 141)):
        pure[first : last + 1] = True
    dark = hex_rgb(pal["dark"]) * 255
    assert np.abs(inside[pure] - (0.82 * dark + 0.18 * behind_it[pure])).max() <= 3

    # It comes in with the top notes
    only_bottom = np.asarray(render_card(folder, project, frozenset({"bottom"})))
    only_top = np.asarray(render_card(folder, project, frozenset({"top"})))
    assert np.array_equal(under(panel, only_bottom), under(panel, behind))
    assert not np.array_equal(under(panel, only_top), under(panel, behind))

    # The border keeps the texture color only where that shows on the dark one (3:1). Riso's texture
    # color is its dark blue, so the border of its panel takes the color of the bars, the cream edge
    # color, all round (and not the blue, which is the panel's own color)
    riso_folder, riso = styled(fake_project, "player-select", palette=dict(PALETTES["Riso"]))
    riso_pal, riso_panel = riso["style"]["palette"], elements(riso_folder, riso)["tr"]
    assert contrast(riso_pal["tex"], riso_pal["dark"]) < 3.0 and riso_pal["tex"] == riso_pal["dark"]
    rim = np.zeros((riso_panel.h, riso_panel.w), bool)
    rim[:3], rim[-3:], rim[:, :3], rim[:, -3:] = True, True, True, True
    border_pixels = under(riso_panel, np.asarray(render_card(riso_folder, riso)))[rim]
    cream = np.rint(hex_rgb(riso_pal["edge"]) * 255).astype(int)
    blue = np.rint(hex_rgb(riso_pal["tex"]) * 255).astype(int)
    assert (border_pixels == cream).all() and not (border_pixels == blue).all(axis=-1).any()


def test_default_stats_without_extras(fake_project):
    folder, project = styled(fake_project, "player-select")
    pal = project["style"]["palette"]
    project["copy"]["extras"]["stats"] = None

    # Without stats the three rows are three of the notes, cut to 14 characters, at 9, 1 and 8
    rows = notes.stats_rows(project["copy"])
    assert rows == [("CREATOR & AI E", 9), ("BUT OVERTHINKS", 1), ("BEST FRIEND: C", 8)]
    assert all(len(label) <= 14 for label, _ in rows)
    boxes = elements(folder, project)
    assert boxes["tr"].lines == [label for label, _ in rows]
    assert (boxes["tr"].w, boxes["tr"].h) == panel_size(boxes["tr"].lines)
    for missing in ([], None):
        project["copy"]["extras"]["stats"] = missing
        assert notes.stats_rows(project["copy"]) == rows
    project["copy"]["notes"]["tl"] = "Hi"
    assert notes.stats_rows(project["copy"])[0] == ("HI", 9)

    # A stat of 2 or less is pink and one of 3 is not; as many cells are on as the stat says (a 0
    # has none): here 2 pink cells (A), 3 of the bars' color (B) and none (C)
    project["copy"]["extras"]["stats"] = [["A", 2], ["B", 3], ["C", 0]]
    panel = elements(folder, project)["tr"]
    tex, bars, low = panel_pixels(panel, np.asarray(render_card(folder, project)), pal)
    border = panel.w * panel.h - (panel.w - 6) * (panel.h - 6)
    assert abs(low - 2 * 13 * 18) <= 8
    assert abs(tex - border) <= 0.02 * tex and abs(bars - 3 * 13 * 18) <= 0.02 * bars


def test_hud_moves_tl_down(fake_project):
    folder, project = styled(fake_project, "player-select", hero="none")
    planned = elements(folder, project)
    assert sorted(planned) == ["bl", "br", "tl", "tr"]
    project["style"]["hero"] = "hud"
    boxes = elements(folder, project)
    assert sorted(boxes) == ["bl", "br", "hud_bottom", "hud_top", "tl", "tr"]

    # The tl note moves 58 px down for the label that takes its corner, and nothing else moves
    assert (boxes["tl"].x, boxes["tl"].y) == (planned["tl"].x, planned["tl"].y + 58)
    assert all(boxes[key] == planned[key] for key in ("tr", "bl", "br"))
    # "PLAYER 1" (pixel 24) in the top left corner of the room and "PRESS START" (pixel 16) at the
    # bottom right, in the margin under the notes, 20 px from the stage's edge
    top, bottom = boxes["hud_top"], boxes["hud_bottom"]
    assert (top.lines, top.size, top.x, top.y, top.w, top.h) == (["PLAYER 1"], 24, 43, 43, 192, 24)
    assert (bottom.lines, bottom.size, bottom.w, bottom.h) == (["PRESS START"], 16, 176, 16)
    assert (bottom.x + bottom.w, bottom.y + bottom.h) == (1080 - 43, 608 - 20)
    assert not any(box.clash for box in boxes.values())
    assert boxes["tl"].y >= top.y + top.h  # under the label
    # A tl note that was dragged stays where it was put
    project["positions"]["tl"] = [200, 150]
    dragged = elements(folder, project)["tl"]
    assert (dragged.x, dragged.y) == (200, 150)
    # A tl note that was dragged onto "PLAYER 1" is in the label's way: the label says so
    project["positions"]["tl"] = [top.x + 8, top.y]
    on_label = elements(folder, project)
    assert (on_label["tl"].x, on_label["tl"].y) == (top.x + 8, top.y) and on_label["hud_top"].clash
    project["positions"]["tl"] = None

    # The labels are ink with a hard black shadow 3 px down and right, each with its group: the top
    # one with the top notes and the bottom one with the bottom notes
    ink = project["style"]["palette"]["ink"]
    states = {"both": ALL_VISIBLE, "bottom": frozenset({"bottom"}), "top": frozenset({"top"})}
    states["none"] = frozenset()
    cards = {name: np.asarray(render_card(folder, project, groups)) for name, groups in states.items()}
    for box, shown in ((top, {"both", "top"}), (bottom, {"both", "bottom"})):
        for name, card in cards.items():
            assert (near(under(box, card), ink, 0) > 40) == (name in shown), (box.key, name)
    assert near(under(top, cards["both"]), "#000000", 0) > 80

    # Tape does not go across a label: the piece on the tl note, which would, is left out, and the
    # others are where they always are
    taped = {**project["style"], "extras": {"tape": True, "stars": False, "grain": False}}
    bare_style = {**taped, "extras": {"tape": False, "stars": False, "grain": False}}
    tape = strongly(
        np.asarray(render_card(folder, {**project, "style": taped})),
        np.asarray(render_card(folder, {**project, "style": bare_style})),
    )
    corners = {"tl": boxes["tl"].x, "tr": boxes["tr"].x + boxes["tr"].w, "br": boxes["br"].x + boxes["br"].w}
    for key, corner in corners.items():
        y = boxes[key].y
        assert (tape[max(y - 50, 0) : y + 50, max(corner - 90, 0) : corner + 90].sum() > 2000) == (key != "tl"), key

    # With arrows on, the moved note's arrow is found again for where it is now: it starts just
    # outside its box, and no arrow runs through a label
    project["style"] = {**project["style"], "arrows": "curly"}
    moved = elements(folder, project)
    start, tl = moved["tl"].arrow[0], moved["tl"]
    gap_x = max(tl.x - start[0], 0, start[0] - (tl.x + tl.w))
    gap_y = max(tl.y - start[1], 0, start[1] - (tl.y + tl.h))
    assert 0 < math.hypot(gap_x, gap_y) <= 10 * math.sqrt(2) + 0.01  # 10 px out from a corner, at most
    for key in ("tl", "tr", "bl", "br"):
        for x, y in quad_points(*moved[key].arrow, 16) if moved[key].arrow else ():
            for label in (moved["hud_top"], moved["hud_bottom"]):
                assert not (label.x <= x < label.x + label.w and label.y <= y < label.y + label.h), key


def test_player_select_renders(fake_project):
    folder, project = styled(fake_project, "player-select")
    recipe, pal = project["style"], project["style"]["palette"]
    image = render_card(folder, project)
    assert image.size == (1080, 608) and image.mode == "RGB"
    pixels = np.asarray(image)
    nobody = np.zeros((608, 1080), np.float32)

    # The room, written out the plain way: the frame's luminance (stretched 1.3 times) picks a
    # point on the line from the dark color to the light one, from 0.15 to 0.75 of the way, dimmed
    # to 0.6; a glow of the texture color (0.4 at the head's center, an ellipse of 420 by 300) is
    # added; every row of which y mod 4 is 2 or 3 is 0.72 as bright; and 3 levels of grain go on
    ys, xs = np.mgrid[0:608, 0:1080].astype(np.float32)
    frame = np.repeat(np.clip(0.2 + 0.6 * xs / 1080, 0, 1)[..., None], 3, axis=-1)
    head = (400, 60, 680, 260)  # centered on (540, 160)
    dark, light, tex = (hex_rgb(pal[role]) for role in ("dark", "light", "tex"))
    wanted = (dark + (light - dark) * (0.15 + 0.6 * np.clip(1.3 * luminance(frame), 0, 1))[..., None]) * 0.6
    glow = np.exp(-(((xs - 540) / 420) ** 2 + ((ys - 160) / 300) ** 2))
    wanted = (wanted + tex * (0.4 * glow)[..., None]) * np.where(ys % 4 >= 2, 0.72, 1.0)[..., None]
    wanted = grain(wanted, 3, 1)
    assert np.allclose(background.render("night", frame, nobody, pal, 1, head), wanted, atol=1e-4)
    assert not np.allclose(background.render("night", frame, nobody, pal, 1, None), wanted, atol=1e-2)

    # The edge is neon: a line of the edge color 6 px out, and a glow of the texture color around
    # it, screened on, that is brighter than the room, cyan and fades away with the distance
    rgb, alpha = stage_inputs(fake_project)
    stage = compose_stage(recipe, rgb, alpha, 0)
    room = compose_stage({**recipe, "edge": "none"}, rgb, alpha, 0)  # (its shadow is faint up there)
    top = int(np.flatnonzero(alpha[:, 540] > 0.5)[0])
    assert np.allclose(stage[top - 3, 540], hex_rgb(pal["edge"]), atol=1e-4)
    close_lift = (stage[top - 10, 520:560] - room[top - 10, 520:560]).mean(axis=0)
    far_lift = (stage[top - 50, 520:560] - room[top - 50, 520:560]).mean(axis=0)
    assert (close_lift > 0.05).all() and (close_lift > far_lift + 0.05).all()
    assert close_lift[1] > close_lift[0] + 0.1 and close_lift[2] > close_lift[0] + 0.1

    # The notes are VT323 capitals (the terminal font, narrow enough for a sentence to stay big on a
    # phone) from 52 px down to 36 (the floor) in steps of 4, in the ink color with a hard black
    # shadow 3 px down and right, and this style draws no arrows. Press Start 2P is for the HUD's
    # two labels.
    assert notes.NOTE_STYLES["pixel"] == NoteStyle("terminal", 52, 36, "upper", line_h=1.0)
    boxes = elements(folder, project)
    assert {box.size for key, box in boxes.items() if key != "tr"} <= {52, 48, 44, 40, 36}
    assert all(box.arrow is None for box in boxes.values())
    assert (notes.hud_width("PLAYER 1", 24), notes.hud_width("PRESS START", 16)) == (8 * 24, 11 * 16)
    assert " ".join(boxes["bl"].lines) == "BUT OVERTHINKS EVERY 6-SECOND REEL"
    for key in ("tl", "bl", "br"):
        area = under(boxes[key], pixels)
        assert near(area, pal["ink"], 0) > 60 and near(area, "#000000", 0) > 40, key
    # With no arrows to keep within reach of, the room is the whole stage's, and the panel of stats
    # stands in the top right, clear of the face
    context = engine.matte_context(recipe, alpha)
    assert context.quads["tl"][1] == 43 and boxes["tr"].y < 304
    assert boxes["tr"].x >= context.face[2] and boxes["tr"].x + boxes["tr"].w <= 1080
    # On the real close-up (the matte of the plate) the notes come out at 36 px or more (the
    # floor), which reads on a phone, where Press Start 2P fell to 16
    with Image.open(Path(__file__).parent / "fixtures" / "close_alpha.png") as image:
        close = np.asarray(image.convert("L"), np.float32) / 255
    plate = engine.style_layout(recipe, close, project["copy"])
    sizes = {box.size for key, box in plate.items() if key != "tr"}
    assert len(sizes) == 1 and min(sizes) >= NOTE_MIN_PX and not any(box.clash for box in plate.values())
    # The panel (373 px) does not fit in the 341 px beside the face box, so (the controller's ruling)
    # it may use the air of that box, the margin around the head, but never the head: it clears the
    # face box taken without its air, overlaps the box with it, and is not flagged for that
    ctx, stage, panel = engine.matte_context(recipe, close), (1080, 608), plate["tr"]
    tight = engine._tight(ctx)
    assert ctx.face == (332, 50, 739, 529) and tight.face == (375, 50, 696, 529)
    assert (panel.w, stage[0] - ctx.face[2]) == (373, 341)
    assert not layout.clash_of(panel, tight, [], stage) and panel.x >= tight.face[2]
    assert layout.clash_of(panel, ctx, [], stage) and not panel.clash and panel.x < ctx.face[2]


def test_notes_are_never_drawn_below_the_floor(fake_project):
    # Note text is never smaller than 36 px on the 1080 px plan (about 13 pt on a phone): the floor
    # is every notes style's smallest size, and the sizes step down to it. The labels used to go to
    # 24 px, which is fine print on a phone.
    assert NOTE_MIN_PX == 36
    table = {"marker": ("marker", 56), "comic": ("comic", 46), "hand": ("hand", 52),
             "pixel": ("terminal", 52), "label": ("label", 46)}
    assert set(notes.NOTE_STYLES) == set(table)
    for notes_id, (font_role, biggest) in table.items():
        style = notes.NOTE_STYLES[notes_id]
        assert (style.font, style.max_size, style.min_size) == (font_role, biggest, NOTE_MIN_PX), notes_id
        assert layout._sizes(style)[-1] == NOTE_MIN_PX == min(layout._sizes(style)), notes_id
    assert layout._sizes(notes.NOTE_STYLES["pixel"]) == [52, 48, 44, 40, 36]
    assert layout._sizes(notes.NOTE_STYLES["label"]) == [46, 44, 42, 40, 38, 36]
    # The hero's texts keep their own sizes: the stats rows and the HUD's labels are not notes
    assert notes.STATS_SIZE == 34 and (notes.HUD_TOP[1], notes.HUD_BOTTOM[1]) == (24, 16)

    # On the real close-up (the matte of the plate), every notes style sets the four notes at the
    # floor or above, whole (not cut). Sticker Pop (bold labels on flat color) is the one that used to
    # come out at 24 px there: its notes are 36 now, whole. Archivo Black is wide: three of its four
    # pills no longer fit the strips beside the head (@YOUR_HANDLE alone is 355 px wide at 36, with the
    # pill's padding, in strips of 289 and 298), so they are placed where they clash least and flagged.
    with Image.open(Path(__file__).parent / "fixtures" / "close_alpha.png") as image:
        close = np.asarray(image.convert("L"), np.float32) / 255
    copy = fake_project.project["copy"]
    for notes_id in table:
        recipe = {**deepcopy(BUILTINS["bedroom-pop"]), "notes": notes_id}
        boxes = engine.style_layout(recipe, close, copy)
        assert all(box.size >= NOTE_MIN_PX and not box.cut for box in boxes.values()), notes_id
    sticker = {**deepcopy(BUILTINS["scrapbook"]), "bg": "flat", "edge": "thin", "notes": "label",
               "arrows": "straight", "hero": "none", "palette": dict(PALETTES["Tangerine"]),
               "extras": {"tape": False, "stars": False, "grain": False}}
    boxes = engine.style_layout(sticker, close, copy)
    assert {box.size for box in boxes.values()} == {NOTE_MIN_PX}
    assert not any(box.cut for box in boxes.values())
    assert [(box.w, box.h) for box in boxes.values()][1] == (355, 75)  # the handle, at one 36 px line

    # A text that cannot be set whole at the floor is cut, with an ellipsis, and still drawn at the
    # floor: never smaller
    long_copy = deepcopy(copy)
    long_copy["notes"]["tl"] = "this note says far too much to fit " * 6
    for notes_id in ("marker", "label"):
        recipe = {**deepcopy(BUILTINS["bedroom-pop"]), "notes": notes_id}
        cut = engine.style_layout(recipe, close, long_copy)["tl"]
        assert cut.cut and cut.clash and cut.size == NOTE_MIN_PX, notes_id
        assert len(cut.lines) == 3 and cut.lines[-1].endswith(chr(0x2026)), notes_id


def contrast(first, second):
    """The WCAG contrast ratio of two #rrggbb colors."""

    def level(color):
        parts = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in hex_rgb(color)]
        return 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2]

    high, low = sorted((level(first), level(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def readable(recipe):
    """Whether a recipe's lettering has a contrast of 3 on what it sits on, as the style guard
    will measure it: the ink on the dark (marker), #111 on the light toward white (comic), the ink
    on paper (hand), the dark on the edge (label) and the ink on the tone of the recipe's background
    (pixel, which has no fill and no glow). A panel of stats needs the same of its bars, which are
    the most visible of the texture, edge and ink colors on the dark one."""
    pal = recipe["palette"]
    pair = {
        "marker": (pal["ink"], pal["dark"]),
        "comic": ("#111111", mix(pal["light"], "#ffffff", 0.45)),
        "hand": (pal["ink"], "#fffdf5"),
        "label": (pal["dark"], pal["edge"]),
        "pixel": (pal["ink"], background.tone(recipe["bg"], pal)),
    }[recipe["notes"]]
    bars = max(contrast(pal[role], pal["dark"]) for role in ("tex", "edge", "ink"))
    return contrast(*pair) >= 3.0 and (recipe["hero"] != "stats" or bars >= 3.0)


def seeded_recipes(count=40, seed=7):
    """`count` recipes with an option of every layer chosen by a seeded random: each layer's ids go
    round and are shuffled, so every id comes up (more than once, with 40), and each recipe has the
    one hero its hero layer gives. The palette is the first of the seven, in a random order, that
    the notes can be read on."""
    rng = random.Random(seed)
    columns = {}
    for layer in ("bg", "edge", "notes", "arrows", "hero"):
        ids = [OPTIONS[layer][i % len(OPTIONS[layer])] for i in range(count)]
        rng.shuffle(ids)
        columns[layer] = ids
    recipes = []
    for i in range(count):
        recipe = {layer: ids[i] for layer, ids in columns.items()}
        recipe["extras"] = {extra: rng.random() < 0.5 for extra in OPTIONS["extras"]}
        recipe.update(id=f"combo-{i + 1}", name=f"Combo {i + 1}")
        names = list(PALETTES)
        rng.shuffle(names)
        name = next(name for name in names if readable({**recipe, "palette": PALETTES[name]}))
        recipes.append({**recipe, "palette": dict(PALETTES[name])})
    return recipes


def test_every_option_renders(fake_project):
    folder, project = fake_project
    recipes = seeded_recipes()
    for recipe in recipes:
        validate_recipe(recipe)
    for layer in ("bg", "edge", "notes", "arrows", "hero"):
        assert {recipe[layer] for recipe in recipes} == set(OPTIONS[layer]), layer
    for extra in OPTIONS["extras"]:
        assert {recipe["extras"][extra] for recipe in recipes} == {True, False}, extra

    # The options that no ready-made style uses draw what the table says. Flat color is a circle
    # of light from the head's center, the light color toward white, to the light color 0.7 of the
    # stage's diagonal away, and it takes nothing from the frame; black and white is the frame's
    # luminance with 1.25 times the contrast, and a vignette that takes 45% off at the corners.
    rgb, alpha = stage_inputs(fake_project)
    pal, nobody = POP["palette"], np.zeros((608, 1080), np.float32)
    head = (400, 60, 680, 260)
    flat = background.render("flat", rgb, nobody, pal, 0, head)
    white = np.ones((608, 1080, 3), np.float32)
    assert np.array_equal(flat, background.render("flat", white, nobody, pal, 0, head))
    light, middle = hex_rgb(pal["light"]), hex_rgb(mix(pal["light"], "#ffffff", 0.2))
    for x, y in ((540, 160), (700, 160), (100, 500), (1000, 40)):
        share = min(math.hypot(x - 540, y - 160) / (0.7 * math.hypot(1080, 608)), 1.0)
        patch = flat[y - 8 : y + 9, x - 8 : x + 9].mean(axis=(0, 1))
        assert np.allclose(patch, middle + (light - middle) * share, atol=0.01), (x, y)
    for level, gray in ((0.2, 0.125), (0.8, 0.875)):
        mono = background.render("mono", np.full((608, 1080, 3), level, np.float32), nobody, pal, 0, None)
        assert np.array_equal(mono[..., 0], mono[..., 1]) and np.array_equal(mono[..., 1], mono[..., 2])
        assert abs(mono[296:313, 532:549].mean() - gray) < 0.01
        for corner in (mono[:8, :8], mono[-8:, -8:]):
            assert abs(corner.mean() - gray * 0.55) < 0.02

    # The tone of a background, which the pixel notes are measured on, is its mean color over the
    # stage (1080 x 608) drawn from the palette over a plain mid-gray photo (luminance 0.5) with
    # nobody in it, in the first variant. Flat color's is its middle color, and the share of the way
    # to the light one that its circle has on average; black and white's is the photo's 0.5 under
    # the mean of its vignette, the same on the three channels.
    ys, xs = np.mgrid[0:608, 0:1080]
    mean_share = np.clip(np.hypot(xs - 540, ys - 304) / (0.7 * math.hypot(1080, 608)), 0.0, 1.0).mean()
    flat_tone = hex_rgb(background.tone("flat", pal)) * 255
    assert np.abs(flat_tone - (middle + (light - middle) * mean_share) * 255).max() <= 1
    away = np.hypot((xs - 540) / 540, (ys - 304) / 304) / math.sqrt(2)
    vignette = (1 - 0.45 * np.clip((away - 0.55) / 0.45, 0.0, 1.0)).mean()
    gray_tone = hex_rgb(background.tone("mono", pal)) * 255
    assert gray_tone.max() - gray_tone.min() < 0.01 and abs(gray_tone[0] - 0.5 * vignette * 255) <= 1
    palette_before = dict(pal)
    for kind in OPTIONS["bg"]:  # every kind: lowercase #rrggbb, the same each time, the palette left alone
        tone = background.tone(kind, pal)
        assert re.fullmatch(r"#[0-9a-f]{6}", tone) and tone == background.tone(kind, pal), kind
    assert pal == palette_before
    with pytest.raises(ValueError, match="glitter"):
        background.tone("glitter", pal)

    # The thin edge is a line of the edge color 5 px out, and no edge is a soft black shadow of
    # the person, 8 px lower (0.35 of its mask blurred by 10), and nothing around it
    (color, mask, blend), = edges.render("thin", alpha, pal, 0)
    assert blend == "normal" and np.allclose(color, hex_rgb(pal["edge"]))
    assert np.allclose(mask, dilate(alpha, 5))
    (black, shadow, blend), = edges.render("none", alpha, pal, 0)
    wanted = 0.35 * np.pad(blur(alpha, 10), ((8, 0), (0, 0)), mode="edge")[:608]
    assert blend == "normal" and not black.any() and np.allclose(shadow, wanted, atol=1e-5)
    top = int(np.flatnonzero(alpha[:, 540] > 0.5)[0])
    plain = {**POP, "extras": {"tape": False, "stars": False, "grain": False}}
    thin = compose_stage({**plain, "edge": "thin"}, rgb, alpha, 0)
    assert np.allclose(thin[top - 3, 540], hex_rgb(pal["edge"]), atol=1e-4)
    assert np.abs(thin[top - 9, 540] - hex_rgb(pal["edge"])).max() > 0.05

    # Bold labels are a pill for each line (the edge color, round corners 10 px), the text in the
    # dark color, tilted a little, and the straight arrow is a line 3 px wide that ends in an open
    # circle of radius 9 (in the arrows' color, which is the edge's, so there is no edge here).
    lagoon = {**deepcopy(BUILTINS["scrapbook"]), "bg": "night", "edge": "none", "notes": "label",
              "arrows": "straight", "hero": "none", "extras": {"tape": False, "stars": False, "grain": False},
              "palette": dict(PALETTES["Lagoon"])}
    labels = {**project, "style": lagoon}
    boxes = elements(folder, labels)
    lagoon_pal = lagoon["palette"]
    card = np.asarray(render_card(folder, labels))
    assert {box.size for box in boxes.values()} <= set(range(NOTE_MIN_PX, 47, 2))
    for key, box in boxes.items():
        area = under(box, card)
        assert " ".join(box.lines) == project["copy"]["notes"][key].upper(), key
        assert 0.2 * box.w * box.h < near(area, lagoon_pal["edge"]) < 0.9 * box.w * box.h, key
        assert near(area, lagoon_pal["dark"]) > 100, key
    bare = np.asarray(render_card(folder, labels, frozenset()))
    for group, keys in {"bottom": ("bl", "br"), "top": ("tl", "tr")}.items():
        shown = np.asarray(render_card(folder, labels, frozenset({group})))
        assert all(ink_near_box(boxes[key], shown, bare) for key in keys), group  # tilted, not far off
    drawn = strongly(np.asarray(render_card(folder, labels, frozenset({"arrows"}))), bare, 60)
    for key, box in boxes.items():
        (sx, sy), _, (ex, ey) = box.arrow
        length = math.hypot(ex - sx, ey - sy)
        along, across = ((ex - sx) / length, (ey - sy) / length), (-(ey - sy) / length, (ex - sx) / length)

        def at(distance, sideways=0.0):
            return drawn[round(ey - along[1] * distance + across[1] * sideways),
                         round(ex - along[0] * distance + across[0] * sideways)]

        assert all(at(distance) for distance in (20, 0.5 * length, length - 12)), key  # the line is straight
        assert at(9, 0) and at(0, 9) and at(0, -9), key  # a ring around the end, radius 9
        assert not at(0, 4.5) and not at(2, 4.5), key  # and it is open: nothing in it but the line
        assert not at(0.5 * length, 8) and not at(0.5 * length, -8), key  # which is thin (3 px)

    # Every one of them renders, in 1080 x 608, with no error
    for recipe in recipes:
        image = render_card(folder, project, style=recipe)
        assert image.size == (1080, 608) and image.mode == "RGB", recipe["id"]


def test_tone_is_drawn_once_for_each_background_and_palette(monkeypatch):
    # The tone of a background is drawn over a picture of the stage's size (about 40 to 100 ms), and
    # project.validate asks for it every time a pixel recipe is checked. So it is kept for each
    # background and palette (the roles in any order are one palette), and a palette or a background that
    # has not been asked for is drawn once and then kept too.
    from mclib.palette import text_contrast

    drawn = []
    real = background.render

    def counting(kind, *args):
        drawn.append(kind)
        return real(kind, *args)

    monkeypatch.setattr(background, "render", counting)
    background._tone.cache_clear()
    pal = dict(PALETTES["Riso"])
    first = background.tone("night", pal)
    assert background.tone("night", pal) == background.tone("night", dict(reversed(pal.items()))) == first
    assert drawn == ["night"]
    background.tone("dots", pal)
    background.tone("night", dict(PALETTES["Neon"]))
    background.tone("dots", pal)
    assert drawn == ["night", "dots", "night"]

    # The guard asks for it through the recipe: one more pixel recipe on the same background and palette,
    # the project's own check included, draws nothing
    recipe = {**deepcopy(BUILTINS["player-select"]), "palette": dict(PALETTES["Neon"])}
    text_contrast(recipe)
    text_contrast(recipe)
    validate(default_project({"path": "x.mp4", "stem": "x", "width": 1920, "height": 1080, "fps": 30.0,
                              "duration": 6.0, "frames": 180, "orientation": "landscape"},
                             "landscape", (1080, 608), recipe))
    assert drawn == ["night", "dots", "night"]
    # An id that is no background is an error, and nothing is kept for it
    for _ in range(2):
        with pytest.raises(ValueError, match="glitter"):
            background.tone("glitter", pal)
    assert drawn == ["night", "dots", "night", "glitter", "glitter"]
