"""Tests for the glyph table: which characters each bundled font draws, and what the card does about the
ones it can't.

The table (scripts/mclib/glyphs.json) is made from the fonts by scripts/make_glyphs.py, so one test reads
the fonts again and holds the table to them, and another asks FreeType, through Pillow, what it really
draws for a few hundred characters, so that a mistake in the cmap reader cannot also be in the table. The
rest is what the table is for: a note in a script its font lacks is cut (so it has a FLAG line and the
studio's amber outline), the burst and the stats panel too, and patch warns in words.
"""

import json
from copy import deepcopy

import make_glyphs
import mc
import pytest
from conftest import SAMPLE_COPY
from PIL import ImageFont

from mclib import glyphs
from mclib import project as project_file
from mclib.config import FONT_FILES, FONTS_DIR
from mclib.glyphs import TABLE
from mclib.layout import NoteStyle, plan_notes
from mclib.styles import engine
from mclib.styles.recipes import BUILTINS

CYRILLIC_YA = chr(0x42F)  # Cyrillic capital Ya: Caveat and Press Start 2P have it, the other four fonts do not
STAR = chr(0x2B50)  # the star emoji: no bundled font has it, and the emoji strip does not cover it
HEART = chr(0x2665)  # the heart suit: Patrick Hand SC and Press Start 2P have it, VT323 does not
ZWJ = chr(0x200D)  # the joiner of emoji sequences
LOWER_F_HOOK = chr(0x192)  # f with a hook: in every font, but its capital (U+0191) is in none

W, H = 1080, 608
MARKER = NoteStyle("marker", 56, 36, "upper", band=22)
TEXTS = dict(SAMPLE_COPY["notes"])


# --- the table, and the fonts it is made from -------------------------------------------------


def test_the_table_is_what_the_fonts_say_today():
    # A font changed without the table (or the table by hand) would let a note through that the font
    # cannot draw, or flag one that it can. The file is what the script writes, to the byte.
    on_disk = TABLE.read_text(encoding="utf-8")
    built = make_glyphs.build()

    assert json.loads(on_disk) == built
    assert on_disk == make_glyphs.as_text(built)
    assert set(built) == {name for name, _ in FONT_FILES.values()}  # every bundled font, and no other
    for name, found in built.items():
        assert found and all(first <= last for first, last in found), name
        assert all(found[i][1] + 1 < found[i + 1][0] for i in range(len(found) - 1)), name  # (merged)


def test_the_ranges_are_made_from_sets_of_code_points():
    assert make_glyphs.ranges({5, 3, 4, 9, 10, 20}) == [[3, 5], [9, 10], [20, 20]]
    assert make_glyphs.ranges(set()) == []


def test_a_cmap_in_a_format_the_reader_does_not_know_is_an_error():
    with pytest.raises(ValueError, match="format 14"):
        make_glyphs._subtable(bytes([0, 14]) + bytes(40), 0)


# The characters that are asked of FreeType: the Latin blocks, Greek and Cyrillic, the punctuation and
# the symbols a note could hold, and the characters the emoji strip leaves (a star, a clock, a joiner)
SAMPLE = [
    *range(0x21, 0x250), *range(0x370, 0x500), *range(0x2010, 0x2070), *range(0x20A0, 0x20C0),
    *range(0x2190, 0x21A0), *range(0x2600, 0x2700), 0x2122, 0x2B50, 0x23F0, 0x231A, 0x200D, 0xFE0F,
    0x1F600, 0x1F44D, 0x65E5, 0x4E2D, 0x5B57, 0x627, 0x5D0,
]


@pytest.mark.parametrize("role", list(FONT_FILES))
def test_the_table_agrees_with_what_freetype_draws(role):
    file_name, _ = FONT_FILES[role]
    pil = ImageFont.truetype(str(FONTS_DIR / file_name), 48, layout_engine=ImageFont.Layout.BASIC)
    gap = pil.getmask(chr(0xFFFF))  # no font maps it: this is the glyph a font draws for "not here"
    for code in SAMPLE:
        char = chr(code)
        if char.isspace():  # (a space has a glyph with nothing in it, which no picture tells from "not here")
            continue
        shown = pil.getmask(char)
        drawn = shown.size != gap.size or bytes(shown) != bytes(gap)
        assert glyphs.covers(role, char) == drawn, (role, hex(code))


# --- asking the table -------------------------------------------------------------------------


def test_missing_names_each_character_a_font_cannot_draw_once_in_order():
    assert glyphs.missing("marker", TEXTS["tl"]) == []
    assert glyphs.missing("marker", "Caf" + chr(0xE9) + " cr" + chr(0xE8) + "me, na" + chr(0xEF) + "ve") == []
    assert glyphs.missing("marker", f"{CYRILLIC_YA} creator {STAR} {CYRILLIC_YA}!") == [CYRILLIC_YA, STAR]
    assert glyphs.missing("marker", "   \n\t ") == []  # spaces are no glyphs
    assert glyphs.missing("marker", "") == []

    # each font has its own characters
    assert glyphs.missing("hand", CYRILLIC_YA) == []  # Caveat has Cyrillic
    assert glyphs.missing("pixel", f"{CYRILLIC_YA} {HEART}") == []  # and so has Press Start 2P
    assert glyphs.missing("terminal", f"x {HEART}") == [HEART]  # VT323 has no heart
    assert glyphs.missing("marker", HEART) == []
    assert glyphs.missing("comic", CYRILLIC_YA) == glyphs.missing("label", CYRILLIC_YA) == [CYRILLIC_YA]
    # emoji that the strip leaves behind (it only takes the ranges of the pictographs and dingbats)
    assert glyphs.missing("marker", f"A{ZWJ}B") == [ZWJ]

    assert glyphs.covers("marker", "a") and not glyphs.covers("marker", STAR)
    assert glyphs.covers("pixel", CYRILLIC_YA) and not glyphs.covers("label", CYRILLIC_YA)


def test_missing_asks_about_the_letters_as_the_style_draws_them():
    # f with a hook is in every font and its capital is in none: set in capitals, the note has a hole
    assert glyphs.missing("marker", LOWER_F_HOOK, "asis") == []
    assert glyphs.missing("marker", LOWER_F_HOOK, "upper") == [LOWER_F_HOOK]  # (the letter as written)
    with pytest.raises(ValueError, match="case must be"):
        glyphs.missing("marker", "a", "title")


# --- the layout flags it ----------------------------------------------------------------------


def test_a_note_with_characters_its_font_lacks_is_cut_and_clashes(person_alpha):
    alpha = person_alpha(W, H)
    clean = plan_notes(alpha, TEXTS, MARKER)
    assert not any(box.cut or box.clash for box in clean.values())

    boxes = plan_notes(alpha, {**TEXTS, "tl": f"{CYRILLIC_YA} {STAR} educator"}, MARKER)

    assert boxes["tl"].cut and boxes["tl"].clash
    assert not any(boxes[key].cut for key in ("tr", "bl", "br"))
    # the same text in a font that has Cyrillic (Caveat, the "hand" role) is a note like any other
    drawn_by_caveat = plan_notes(alpha, {**TEXTS, "tl": CYRILLIC_YA + " educator"},
                                 NoteStyle("hand", 52, 36, "asis", pad=16))
    assert not drawn_by_caveat["tl"].cut


def test_the_burst_and_the_stats_panel_and_the_note_under_the_burst_are_cut_too(person_alpha):
    alpha = person_alpha(W, H)
    copy = deepcopy(SAMPLE_COPY)

    # the name burst (Bangers): the name the person gave, or the burst text they wrote
    pop = BUILTINS["comic-pop"]
    assert not any(box.cut for box in engine.style_layout(pop, alpha, copy).values())
    copy["extras"]["burst"] = CYRILLIC_YA + "!"
    boxes = engine.style_layout(pop, alpha, copy)
    assert boxes["burst"].cut and boxes["burst"].clash and not boxes["tl"].cut
    copy["extras"]["burst"] = None
    copy["name"] = CYRILLIC_YA
    assert engine.style_layout(pop, alpha, copy)["burst"].cut

    # the tr note moves under the burst: it is still judged against its own font
    copy = deepcopy(SAMPLE_COPY)
    copy["notes"]["tr"] = f"{CYRILLIC_YA}{CYRILLIC_YA}{CYRILLIC_YA}"
    boxes = engine.style_layout(pop, alpha, copy)
    assert boxes["tr"].cut and boxes["tr"].clash and not boxes["burst"].cut

    # the stats panel (VT323): a label it lacks a character for
    player = BUILTINS["player-select"]
    copy = deepcopy(SAMPLE_COPY)
    assert not engine.style_layout(player, alpha, copy)["tr"].cut
    copy["extras"]["stats"] = [["AI skills", 10], [f"I {HEART} it", 10], ["Chill", 1]]
    panel = engine.style_layout(player, alpha, copy)["tr"]
    assert panel.cut and panel.clash
    # with no stats of its own the panel shows the notes (in VT323 too)
    copy["extras"]["stats"] = None
    copy["notes"]["tl"] = f"I {HEART} it"
    assert engine.style_layout(player, alpha, copy)["tr"].cut


# --- patch says so in words, card flags it ----------------------------------------------------


def valid(fake_project, **changes):
    """A copy of the fake project with its recipe and copy changed, which is a project validate can read."""
    project = deepcopy(fake_project.project)
    for key, value in changes.items():
        project[key] = value
    return project


def test_validate_warns_with_the_characters_and_the_font(fake_project):
    project = valid(fake_project)
    project["copy"]["notes"]["tl"] = f"{CYRILLIC_YA} {STAR} creator {CYRILLIC_YA}"
    warnings = project_file.validate(project)

    assert "copy.notes.tl has characters the Marker font can't draw: " + CYRILLIC_YA + ", " + STAR + "." in warnings
    assert project["copy"]["notes"]["tl"] == f"{CYRILLIC_YA} {STAR} creator {CYRILLIC_YA}"  # the text is never cut

    # a character nobody can see is named by its code point
    project = valid(fake_project)
    project["copy"]["notes"]["bl"] = f"But A{ZWJ}B"
    assert "copy.notes.bl has characters the Marker font can't draw: U+200D." in project_file.validate(project)

    # emoji are still taken out by the strip, with its own warning, and then they are not flagged twice
    project = valid(fake_project)
    project["copy"]["notes"]["br"] = "Best friend " + chr(0x1F916)
    warnings = project_file.validate(project)
    assert "Removed emoji from copy.notes.br; the video fonts cannot draw them." in warnings
    assert not [warning for warning in warnings if "can't draw" in warning]

    # the text that is fine has no warning, and they come before the contrast one (which is last)
    assert not [warning for warning in project_file.validate(valid(fake_project)) if "can't draw" in warning]
    project = valid(fake_project)
    project["copy"]["notes"]["tl"] = STAR
    project["style"]["palette"]["ink"] = project["style"]["palette"]["dark"]
    warnings = project_file.validate(project)
    assert "can't draw" in warnings[-2] and warnings[-1].startswith("Text contrast")


def test_validate_names_the_font_of_the_notes_style_and_of_the_hero(fake_project):
    def warned(recipe, **copy_changes):
        project = valid(fake_project, style=deepcopy(BUILTINS[recipe]))
        for where, value in copy_changes.items():
            target = project["copy"]
            *path, last = where.split(".")
            for step in path:
                target = target[step]
            target[last] = value
        return [warning for warning in project_file.validate(project) if "can't draw" in warning]

    f = CYRILLIC_YA
    assert warned("comic-pop", **{"notes.tl": f}) == [f"copy.notes.tl has characters the Comic boxes font can't draw: {f}."]
    assert warned("player-select", **{"notes.bl": f}) == [f"copy.notes.bl has characters the Pixel font can't draw: {f}."]
    # Caveat has Cyrillic; it has no star
    assert warned("scrapbook", **{"notes.tl": f}) == []
    assert warned("scrapbook", **{"notes.tl": STAR}) == [f"copy.notes.tl has characters the Handwritten strips font can't draw: {STAR}."]
    # the burst: its text, or the name when it has none (the name is shown only by the burst)
    assert warned("comic-pop", **{"extras.burst": f + "!"}) == [f"copy.extras.burst has characters the Name burst font can't draw: {f}."]
    assert warned("comic-pop", **{"extras.burst": None, "name": f}) == [f"copy.name has characters the Name burst font can't draw: {f}."]
    assert warned("bedroom-pop", **{"extras.burst": f + "!", "name": f}) == []  # (no burst on this card)
    # the stats panel takes the tr note's place: that note is not drawn, and its labels are
    assert warned("player-select", **{"notes.tr": f}) == []
    assert warned("player-select", **{"extras.stats": [["ok", 3], [f, 4], ["fine", 5]]}) == [
        f"copy.extras.stats[1] has characters the Stats panel font can't draw: {f}."]
    assert warned("comic-pop", **{"extras.stats": [["ok", 3], [f, 4], ["fine", 5]]}) == []  # (no panel on this card)


def test_card_flags_a_note_its_font_cannot_draw_and_patch_warns(fake_project, tmp_path, capsys):
    folder, _ = fake_project
    patch = tmp_path / "patch.json"
    patch.write_text(json.dumps({"copy": {"notes": {"tl": f"{CYRILLIC_YA} {STAR} educator"}}}), encoding="utf-8")

    assert mc.main(["patch", str(folder), "--file", str(patch)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert f"WARNING copy.notes.tl has characters the Marker font can't draw: {CYRILLIC_YA}, {STAR}." in printed

    assert mc.main(["card", str(folder), "--out", str(tmp_path / "card.png")]) == 0
    assert "FLAG tl cut" in capsys.readouterr().out.splitlines()
