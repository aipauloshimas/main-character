"""Tests for the constants and the project file."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from mclib import config
from mclib import project as project_file
from mclib.project import (
    NOTE_KEYS,
    PROJECT_FILE,
    SCHEMA,
    SOFT_LIMITS,
    UPDATE_LOCK,
    default_project,
    load,
    merge_patch,
    save,
    update,
    validate,
)

# Emoji and symbols are built from code points so no editor or tool can change them.
ROBOT = chr(0x1F916)  # robot face, in the pictograph block
SPARKLES = chr(0x2728)  # sparkles, in the dingbats block
VS16 = chr(0xFE0F)  # emoji variation selector
EM_DASH = chr(0x2014)  # not an emoji, so it must survive

# Bedroom Pop written out: validate checks the style with styles.recipes.validate_recipe, so it must be valid.
RECIPE = {
    "id": "bedroom-pop",
    "name": "Bedroom Pop",
    "bg": "dots",
    "edge": "handcut",
    "notes": "marker",
    "arrows": "curly",
    "hero": "none",
    "extras": {"tape": False, "stars": False, "grain": True},
    "palette": {
        "dark": "#7d2a72",
        "light": "#f2a9de",
        "tex": "#4c0f45",
        "edge": "#f5df98",
        "ink": "#f7e8b0",
    },
}

CLIP = {
    "path": "C:/clips/intro.mp4",
    "stem": "intro",
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "duration": 6.0,
    "frames": 180,
    "orientation": "landscape",
}

# The examples of RFC 7386, section 3, that start from an object.
RFC_7386_CASES = [
    ({"a": "b"}, {"a": "c"}, {"a": "c"}),
    ({"a": "b"}, {"b": "c"}, {"a": "b", "b": "c"}),
    ({"a": "b"}, {"a": None}, {}),
    ({"a": "b", "b": "c"}, {"a": None}, {"b": "c"}),
    ({"a": ["b"]}, {"a": "c"}, {"a": "c"}),
    ({"a": "c"}, {"a": ["b"]}, {"a": ["b"]}),
    ({"a": {"b": "c"}}, {"a": {"b": "d", "c": None}}, {"a": {"b": "d"}}),
    ({"a": [{"b": "c"}]}, {"a": [1]}, {"a": [1]}),
    ({"e": None}, {"a": 1}, {"e": None, "a": 1}),
    ({}, {"a": {"bb": {"ccc": None}}}, {"a": {"bb": {}}}),
]

# Bad values for the fixed choices and the shape of the file:
# (path into the project, the bad value, text the error must mention).
BAD_SETTINGS = [
    (("schema",), 3, "schema"),
    (("schema",), 1, "schema"),  # a project of the first schema is migrated by load: validate never sees one
    (("clip",), "intro.mp4", "clip"),
    (("layout",), "letterbox", "layout"),  # the first schema's name for a shape
    (("work", "width"), 0, "work"),
    (("freeze", "frame"), -1, "freeze.frame"),
    (("freeze", "suggested"), 62, "freeze.suggested"),
    (("audio", "track"), 7, "audio.track"),
    (("audio", "onsets"), "none", "audio.onsets"),
    (("style_source",), 3, "style_source"),
    (("matte", "detail"), "ultra", "matte.detail"),
    (("matte", "edge"), 5, "matte.edge"),
    (("matte", "edge"), -4.5, "matte.edge"),
    (("matte", "edge"), 0.5, "matte.edge"),  # a whole number: the preview and the sized renders must agree
    (("matte", "edge"), 2.0, "matte.edge"),
    (("matte", "edge"), "2", "matte.edge"),
    (("matte", "edge"), True, "matte.edge"),
    (("matte", "edge"), None, "matte.edge"),
    (("picked",), "D", "picked"),
    (("notice",), 7, "notice"),
    (("positions", "tl"), [10], "positions.tl"),
    (("positions", "tl"), [float("nan"), 80], "positions.tl"),
    (("positions", "burst"), [120, float("inf")], "positions.burst"),
    (("positions", "br"), "center", "positions.br"),
    (("style_request",), "make it pop", "style_request"),
    (("status", "outputs"), "a.mp4", "status"),
    (("copy", "notes"), {"tl": "x"}, "copy.notes"),
    (("copy", "name"), 7, "copy.name"),
    (("copy", "extras", "burst"), 7, "copy.extras.burst"),
    (("versions", "B"), None, "versions.B"),
]


def make_project():
    return default_project(CLIP, "landscape", (1080, 608), RECIPE)


def test_default_project_has_timing_defaults():
    p = make_project()
    assert list(p) == [
        "schema", "clip", "layout", "work", "freeze", "timing", "audio", "style", "style_source",
        "matte", "versions", "picked", "copy", "positions", "style_request", "status",
    ]
    assert p["schema"] == SCHEMA == 2
    # The numbers are written out here, so a wrong constant in config is caught.
    assert p["timing"] == {
        "freeze_at": 2.07,
        "hold": 3.0,
        "head_speed": 1.0,
        "tail": 1.2,
        "release_from": 0.0,
        "reveal": "staggered",
        "boil": True,
        "shutter": False,
    }
    assert p["timing"] == config.TIMING_DEFAULTS
    assert p["layout"] == "landscape"
    assert p["work"] == {"width": 1080, "height": 608}
    assert p["clip"] == CLIP and p["clip"] is not CLIP
    assert p["style"] == RECIPE and p["style"] is not RECIPE
    assert p["style_source"] == "bedroom-pop"
    assert p["picked"] is None

    # Everything else starts empty. The copy is the four notes, the name and the extras: there is no title (the
    # person adds that in the app), so no "hook" anywhere, and no limit for one.
    empty_copy = {
        "notes": {"tl": "", "tr": "", "bl": "", "br": ""},
        "name": "",
        "extras": {"burst": None, "emphasis": None, "stats": None},
    }
    assert p["copy"] == empty_copy
    assert p["versions"] == {"A": empty_copy, "B": empty_copy, "C": empty_copy}
    assert project_file.empty_copy() == empty_copy and "hook" not in SOFT_LIMITS
    assert set(SOFT_LIMITS) == {"tl", "tr", "bl", "br", "name"}
    assert p["freeze"] == {"frame": 0, "suggested": []}
    assert p["audio"] == {"track": None, "onsets": []}
    assert p["matte"] == {"detail": "auto", "edge": 0}
    assert p["positions"] == {"tl": None, "tr": None, "bl": None, "br": None, "burst": None}
    assert p["style_request"] is None
    assert p["status"] == {"rendered_at": None, "outputs": [], "done_at": None}

    # Editing one project must not leak into the shared defaults, another project or another version.
    p["timing"]["hold"] = 5.0
    p["versions"]["A"]["notes"]["tl"] = "changed"
    assert config.TIMING_DEFAULTS["hold"] == 3.0
    assert make_project()["timing"]["hold"] == 3.0
    assert p["versions"]["B"]["notes"]["tl"] == ""
    assert p["copy"]["notes"]["tl"] == ""

    # A fresh project is valid and has nothing to warn about.
    assert validate(make_project()) == []


def test_merge_patch_replaces_nested_and_deletes_null():
    target = {"a": {"b": 1, "c": 2}}
    patch = {"a": {"b": None, "d": 3}}
    assert merge_patch(target, patch) == {"a": {"c": 2, "d": 3}}
    # The inputs are not touched.
    assert target == {"a": {"b": 1, "c": 2}}
    assert patch == {"a": {"b": None, "d": 3}}

    for original, change, expected in RFC_7386_CASES:
        assert merge_patch(original, change) == expected

    # A list is replaced whole, and the result does not share it with the patch.
    patch = {"outputs": ["a.mp4"]}
    result = merge_patch({"outputs": []}, patch)
    assert result == {"outputs": ["a.mp4"]}
    result["outputs"].append("b.mp4")
    assert patch == {"outputs": ["a.mp4"]}

    # The result does not share anything with the target either.
    target = {"a": {"b": [1]}}
    result = merge_patch(target, {"c": 1})
    result["a"]["b"].append(2)
    assert target == {"a": {"b": [1]}}

    # A null deletes the key, so a project that clears "picked" this way loses the key. It is
    # still a valid project.
    p = merge_patch(make_project(), {"picked": "B"})
    assert p["picked"] == "B"
    p = merge_patch(p, {"picked": None, "style_request": None})
    assert "picked" not in p and "style_request" not in p
    assert validate(p) == []


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    folder = tmp_path / "pasta com acentuação"  # spaces and accents must work in a path
    p = make_project()
    p["copy"]["name"] = "José"
    p["copy"]["notes"]["tl"] = "Criador & educador de IA"
    p["positions"]["tl"] = [120, 80]

    save(folder, p)

    assert load(folder) == p
    assert [entry.name for entry in folder.iterdir()] == [PROJECT_FILE]
    assert not list(folder.glob("*.tmp"))  # (the temp file has the process's number in its name: look for any)
    text = (folder / PROJECT_FILE).read_bytes().decode("utf-8")
    assert "José" in text and "u00e9" not in text  # real UTF-8, not ASCII escapes

    # Saving again replaces the file that is already there.
    p["picked"] = "B"
    save(folder, p)
    assert load(folder)["picked"] == "B"
    assert [entry.name for entry in folder.iterdir()] == [PROJECT_FILE]

    # NaN is not valid JSON and the page could not read it, so save refuses it and leaves the file alone.
    p["positions"]["tl"] = [float("nan"), 80]
    with pytest.raises(ValueError):
        save(folder, p)
    assert load(folder)["positions"]["tl"] == [120, 80]
    assert [entry.name for entry in folder.iterdir()] == [PROJECT_FILE]
    p["positions"]["tl"] = [120, 80]

    # A save that fails at the last step leaves the previous file whole, because the write goes
    # to a temp file first and only os.replace puts it in place. The temp file does not stay behind
    # (its name is this process's, so a later save would not write over it).
    def failing_replace(source, destination):
        raise OSError("disk went away")

    p["picked"] = "C"
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            save(folder, p)
    assert load(folder)["picked"] == "B"
    assert not list(folder.glob("*.tmp")) and [entry.name for entry in folder.iterdir()] == [PROJECT_FILE]


def test_save_writes_a_temp_file_of_its_own_process(tmp_path, monkeypatch):
    # The command line (patch, style) and the studio's server are two programs that save the same project.json.
    # They must not share a temp file, or one could replace the file with the other's half-written text: the
    # temp file is named for the process, project.json.<pid>.tmp
    seen = []
    real = os.replace

    def watching(source, destination):
        seen.append((Path(source).name, Path(destination).name, Path(source).read_text(encoding="utf-8")))
        return real(source, destination)

    p = make_project()
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", watching)
        save(tmp_path, p)

    [(source, destination, text)] = seen
    assert source == f"{PROJECT_FILE}.{os.getpid()}.tmp" and destination == PROJECT_FILE
    assert json.loads(text) == p  # the whole file was in the temp file when it was put in place
    assert [entry.name for entry in tmp_path.iterdir()] == [PROJECT_FILE]


def test_a_project_of_schema_1_is_migrated_when_it_is_loaded(tmp_path):
    # Schema 1 had the letterbox and the full screen, and a hook. Schema 2 is the clip's own shape: the
    # letterbox is "landscape" and the full screen is "portrait", the work size and the positions (in px of the
    # plan, which is what they were for those two) stay, and so does everything else. The hook does not: the
    # edit has no title, so the file is read as if it had never had one.
    for old, new, work in (("letterbox", "landscape", (1080, 608)), ("fullscreen", "portrait", (1080, 1920))):
        folder = tmp_path / old
        p = default_project(CLIP, old, work, RECIPE)
        p["schema"] = 1
        p["copy"]["hook"] = ["this is your sign to", "post it"]
        p["positions"]["tl"] = [120, 80]
        p["freeze"]["frame"] = 17
        folder.mkdir()
        (folder / PROJECT_FILE).write_text(json.dumps(p), encoding="utf-8")

        loaded = load(folder)

        assert loaded["schema"] == SCHEMA == 2 and loaded["layout"] == new
        assert validate(loaded) == []
        assert loaded["work"] == {"width": work[0], "height": work[1]}
        assert "hook" not in loaded["copy"]
        assert loaded["positions"]["tl"] == [120, 80] and loaded["freeze"]["frame"] == 17
        assert {key: value for key, value in loaded.items() if key not in ("schema", "layout", "copy")} == {
            key: value for key, value in p.items() if key not in ("schema", "layout", "copy")}
        assert loaded["copy"] == {key: value for key, value in p["copy"].items() if key != "hook"}
        # the file is not touched by a read, and it is a schema 2 file once the project is saved
        assert json.loads((folder / PROJECT_FILE).read_text(encoding="utf-8"))["schema"] == 1
        save(folder, loaded)
        assert json.loads((folder / PROJECT_FILE).read_text(encoding="utf-8"))["schema"] == 2

    # Changing a project of schema 1 through update saves it as schema 2
    folder = tmp_path / "updated"
    p = default_project(CLIP, "letterbox", (1080, 608), RECIPE)
    p["schema"] = 1
    folder.mkdir()
    (folder / PROJECT_FILE).write_text(json.dumps(p), encoding="utf-8")
    merged, _ = update(folder, {"timing": {"hold": 4}})
    assert merged["schema"] == 2 and merged["layout"] == "landscape"
    assert json.loads((folder / PROJECT_FILE).read_text(encoding="utf-8"))["layout"] == "landscape"

    # A schema 2 project, or a file of any other schema, comes back as it is
    again = tmp_path / "again"
    q = make_project()
    save(again, q)
    assert load(again) == q
    future = tmp_path / "future"
    future.mkdir()
    (future / PROJECT_FILE).write_text(json.dumps({**q, "schema": 3}), encoding="utf-8")
    assert load(future)["schema"] == 3
    with pytest.raises(ValueError, match="schema must be 2, got 3"):
        validate(load(future))


def test_load_drops_the_title_of_an_older_file(tmp_path):
    # The edit has no title (Amendment 1). A file that has one, whatever its schema (the clip's-own-shape round kept
    # it in the file), is read without it, in the live copy and in every version, and silently: no warning
    folder = tmp_path / "older"
    p = make_project()
    p["copy"]["notes"]["tl"] = "Creator & AI educator"
    p["versions"]["B"]["notes"]["bl"] = "But naps"
    for block in (p["copy"], *p["versions"].values()):
        block["hook"] = ["this is your sign to", "post it"]
    folder.mkdir()
    (folder / PROJECT_FILE).write_text(json.dumps(p), encoding="utf-8")

    loaded = load(folder)

    assert "hook" not in loaded["copy"]
    assert all("hook" not in version for version in loaded["versions"].values())
    assert loaded["copy"]["notes"]["tl"] == "Creator & AI educator" and loaded["versions"]["B"]["notes"]["bl"] == "But naps"
    assert validate(loaded) == []  # (before validate: a project with a hook is one the validation never sees)
    assert json.loads((folder / PROJECT_FILE).read_text(encoding="utf-8"))["copy"]["hook"]  # a read leaves the file

    # a change saves the file without it, and a patch on the older file is applied to the project as it is read
    merged, warnings = update(folder, {"copy": {"name": "Pauli"}})
    assert warnings == [] and "hook" not in merged["copy"]
    saved = json.loads((folder / PROJECT_FILE).read_text(encoding="utf-8"))
    assert "hook" not in saved["copy"] and all("hook" not in version for version in saved["versions"].values())
    assert saved["copy"]["name"] == "Pauli"

    # what is not a copy block is left as it is: nothing breaks on a file that lacks the parts (validate says so)
    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / PROJECT_FILE).write_text(json.dumps({"schema": 2, "copy": "none", "versions": ["A"]}), encoding="utf-8")
    assert load(odd) == {"schema": 2, "copy": "none", "versions": ["A"]}
    (odd / PROJECT_FILE).write_text("[1, 2]", encoding="utf-8")
    assert load(odd) == [1, 2]


def test_load_makes_the_edge_of_an_older_file_a_whole_number(tmp_path):
    # matte.edge is a whole number now (validate says so). The studio's slider used to move in halves, so a project
    # that was saved with 1.5 is read as 2, and is a valid project, rather than one that cannot be changed
    folder = tmp_path / "halves"
    folder.mkdir()
    for edge, expected in ((1.5, 2), (-0.5, 0), (2.0, 2), (-3.5, -4), (0.4, 0), (3, 3), (0, 0)):
        p = make_project()
        p["matte"]["edge"] = edge
        (folder / PROJECT_FILE).write_text(json.dumps(p), encoding="utf-8")

        loaded = load(folder)

        assert loaded["matte"]["edge"] == expected and isinstance(loaded["matte"]["edge"], int), edge
        assert validate(loaded) == []
    # the rest of the matte stays, and a value that is not a number is left for validate to refuse
    p = make_project()
    p["matte"] = {"detail": "high", "edge": "soft"}
    (folder / PROJECT_FILE).write_text(json.dumps(p), encoding="utf-8")
    assert load(folder)["matte"] == {"detail": "high", "edge": "soft"}
    # and one that is out of range is still out of range once it is whole
    p["matte"] = {"detail": "auto", "edge": 7.4}
    (folder / PROJECT_FILE).write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(ValueError, match="matte.edge must be a whole number from -4 to 4, got 7"):
        validate(load(folder))


def test_the_edge_has_a_plain_sentence_when_it_is_not_a_whole_number():
    for bad, shown in ((1.5, "1.5"), (5, "5"), ("2", "'2'"), (True, "True")):
        p = make_project()
        p["matte"]["edge"] = bad
        with pytest.raises(ValueError) as error:
            validate(p)
        assert str(error.value) == f"matte.edge must be a whole number from -4 to 4, got {shown}"
    for good in range(-4, 5):
        p = make_project()
        p["matte"]["edge"] = good
        assert validate(p) == []


def test_a_notice_is_text_or_not_there():
    # init leaves one for the studio's page (see pipeline.POSITIONS_RESET); a patch with a null takes it away again
    p = make_project()
    assert "notice" not in p and validate(p) == []
    p["notice"] = "Your note positions were reset because the clip's shape changed."
    assert validate(p) == []
    assert "notice" not in merge_patch(p, {"notice": None})


def test_save_waits_out_a_file_lock(tmp_path, monkeypatch):
    # The waiter, the studio's page or a virus scan can have project.json open at the moment of the
    # save, and Windows then refuses to replace it. That lasts a moment, so the save waits for it.
    p = make_project()
    save(tmp_path, p)
    p["picked"] = "B"
    real, refusals, pauses = os.replace, [], []

    def held_for_a_moment(source, destination):
        if len(refusals) < 2:
            refusals.append(destination)
            raise PermissionError(5, "Access is denied")
        return real(source, destination)

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", held_for_a_moment)
        patched.setattr(time, "sleep", pauses.append)
        save(tmp_path, p)
    assert len(refusals) == 2 and pauses == [0.1, 0.1]
    assert load(tmp_path)["picked"] == "B"
    assert [entry.name for entry in tmp_path.iterdir()] == [PROJECT_FILE]

    # A lock that does not clear in ten tries is an error that says which file, and the file that
    # was there is whole.
    def held_for_good(source, destination):
        raise PermissionError(5, "Access is denied")

    pauses.clear()
    p["picked"] = "C"
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", held_for_good)
        patched.setattr(time, "sleep", pauses.append)
        with pytest.raises(PermissionError, match=r"project\.json is in use by another program"):
            save(tmp_path, p)
    assert pauses == [0.1] * 9
    assert load(tmp_path)["picked"] == "B"
    assert not list(tmp_path.glob("*.tmp"))  # and the save that gave up leaves no temp file behind


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to replace a file that is open")
def test_save_waits_for_a_project_file_that_is_really_open(tmp_path):
    save(tmp_path, make_project())
    p = make_project()
    p["picked"] = "B"

    with open(tmp_path / PROJECT_FILE, encoding="utf-8") as reader:  # the waiter is reading it
        threading.Timer(0.3, reader.close).start()
        save(tmp_path, p)

    assert load(tmp_path)["picked"] == "B"


# --- update: how a program other than the studio server changes the file -------------------------


def test_update_applies_the_patch_then_the_function_and_saves(tmp_path):
    save(tmp_path, make_project())

    project, warnings = update(
        tmp_path, {"copy": {"name": "Pauli"}, "picked": "B"}, mutate=lambda p: p["matte"].update(detail="high")
    )

    assert warnings == []
    assert project["copy"]["name"] == "Pauli" and project["picked"] == "B" and project["matte"]["detail"] == "high"
    assert load(tmp_path) == project

    # A null in the patch deletes the key, and the music track stays in the file as a null, as the studio keeps it
    update(tmp_path, {"audio": {"track": "music.mp3"}})
    update(tmp_path, {"picked": None, "audio": {"track": None}})
    assert "picked" not in load(tmp_path) and load(tmp_path)["audio"]["track"] is None


def test_update_returns_the_warnings_of_validate(tmp_path):
    save(tmp_path, make_project())

    project, warnings = update(tmp_path, {"copy": {"notes": {"tl": "T" * 31}}})

    assert warnings == ["copy.notes.tl is 31 characters; the soft limit is 30."]
    assert load(tmp_path)["copy"]["notes"]["tl"] == "T" * 31  # kept whole


def test_update_saves_nothing_when_nothing_changes_or_the_result_is_not_valid(tmp_path, monkeypatch):
    save(tmp_path, make_project())
    before = (tmp_path / PROJECT_FILE).read_bytes()
    saves = []
    monkeypatch.setattr(project_file, "save", lambda *args: saves.append(args))

    update(tmp_path)  # no patch and no function
    update(tmp_path, {"copy": {"name": ""}})  # the value it has already
    assert saves == []

    with pytest.raises(ValueError, match=r"timing\.hold must be a number from 1\.0 to 6\.0"):
        update(tmp_path, {"timing": {"hold": 99}})

    def refuses(project):
        raise ValueError("not this one")

    with pytest.raises(ValueError, match="not this one"):  # the function can refuse
        update(tmp_path, mutate=refuses)
    assert saves == [] and (tmp_path / PROJECT_FILE).read_bytes() == before


def test_update_waits_out_a_lock_on_the_read(tmp_path, monkeypatch):
    # Another program is replacing project.json at the moment of the read, and Windows refuses it for a moment
    save(tmp_path, make_project())
    real, refusals, pauses = load, [], []

    def held_for_a_moment(folder):
        if len(refusals) < 2:
            refusals.append(folder)
            raise PermissionError(5, "Access is denied")
        return real(folder)

    monkeypatch.setattr(project_file, "load", held_for_a_moment)
    monkeypatch.setattr(time, "sleep", pauses.append)

    update(tmp_path, {"picked": "B"})

    assert len(refusals) == 2 and pauses == [0.1, 0.1]
    assert real(tmp_path)["picked"] == "B"


def test_update_says_when_the_file_cannot_be_read(tmp_path):
    (tmp_path / PROJECT_FILE).write_text("{ not whole", encoding="utf-8")

    with pytest.raises(ValueError, match=r"project\.json cannot be read \(.*\)\. Fix it, or move it away"):
        update(tmp_path, {"picked": "B"})
    with pytest.raises(FileNotFoundError):
        update(tmp_path / "nowhere", {"picked": "B"})
    assert (tmp_path / PROJECT_FILE).read_text(encoding="utf-8") == "{ not whole"


def test_update_keeps_a_changed_freeze_frame_inside_the_clip(tmp_path):
    save(tmp_path, make_project())  # the clip has 180 frames

    with pytest.raises(ValueError, match=r"The freeze frame must be between 0 and 179: the clip has 180 frames\."):
        update(tmp_path, {"freeze": {"frame": 180}})
    assert load(tmp_path)["freeze"]["frame"] == 0
    update(tmp_path, {"freeze": {"frame": 179}})
    assert load(tmp_path)["freeze"]["frame"] == 179

    # A project that is past the end already (an old clip) is not stuck: what else is changed goes through
    stuck = make_project()
    stuck["freeze"]["frame"] = 500
    save(tmp_path, stuck)
    update(tmp_path, {"copy": {"name": "Pauli"}})
    assert load(tmp_path)["freeze"]["frame"] == 500 and load(tmp_path)["copy"]["name"] == "Pauli"


def test_update_takes_the_update_lock(tmp_path):
    save(tmp_path, make_project())
    done = threading.Event()

    def work():
        update(tmp_path, {"picked": "B"})
        done.set()

    with UPDATE_LOCK:  # the studio's patch, or the render's record of its status, is between its read and its save
        worker = threading.Thread(target=work)
        worker.start()
        assert not done.wait(0.3)
    worker.join(30)

    assert done.is_set() and load(tmp_path)["picked"] == "B"


def test_validate_rejects_out_of_range_timing():
    # The ranges are written out here, so a loose range in config is caught.
    assert config.TIMING_RANGES == {
        "freeze_at": (0.5, 5.0),
        "hold": (1.0, 6.0),
        "head_speed": (0.75, 1.5),
        "tail": (0.0, 3.0),
        "release_from": (0.0, 1.0),
    }

    p = make_project()
    p["timing"]["head_speed"] = 2.0
    with pytest.raises(ValueError, match="head_speed"):
        validate(p)

    # Every ranged setting takes its bounds and refuses values just outside them.
    for key, (low, high) in config.TIMING_RANGES.items():
        for bad in (low - 0.1, high + 0.1):
            p = make_project()
            p["timing"][key] = bad
            with pytest.raises(ValueError, match=key):
                validate(p)
        for good in (low, high):
            p = make_project()
            p["timing"][key] = good
            assert validate(p) == []

    # Reveal is one of two words, a flag is a real boolean and a number is not text or a boolean.
    for key, bad in [("reveal", "sideways"), ("boil", 1), ("shutter", "yes"), ("hold", True), ("tail", "1.2")]:
        p = make_project()
        p["timing"][key] = bad
        with pytest.raises(ValueError, match=key):
            validate(p)

    # The other fixed choices and the shape of the file are guarded the same way.
    for path, bad, expected in BAD_SETTINGS:
        p = make_project()
        holder = p
        for step in path[:-1]:
            holder = holder[step]
        holder[path[-1]] = bad
        with pytest.raises(ValueError) as error:
            validate(p)
        assert expected in str(error.value)
    p = make_project()
    p["layout"], p["picked"] = "portrait", "C"
    p["matte"] = {"detail": "max", "edge": -4}
    p["positions"]["tl"] = [120, 80.5]
    p["audio"] = {"track": "song.mp3", "onsets": [0.5, 1.25]}
    assert validate(p) == []

    # A missing section is named.
    for key in ("timing", "versions", "copy", "style"):
        p = make_project()
        del p[key]
        with pytest.raises(ValueError, match=key):
            validate(p)
    p = make_project()
    del p["versions"]["B"]
    with pytest.raises(ValueError, match="versions"):
        validate(p)
    with pytest.raises(ValueError, match="project"):
        validate(["not", "a", "project"])

    # A half-written style recipe is refused by name.
    p = make_project()
    del p["style"]["palette"]
    with pytest.raises(ValueError, match="palette"):
        validate(p)


def test_validate_warns_on_soft_limits():
    p = make_project()
    p["copy"]["notes"]["bl"] = "x" * 40
    warnings = validate(p)
    assert len(warnings) == 1
    assert "bl" in warnings[0] and "40" in warnings[0]
    assert p["copy"]["notes"]["bl"] == "x" * 40  # a soft limit warns; it never edits the text

    # Each field has its own limit: exactly at it is quiet, one over it warns and names the field.
    p = make_project()
    for key in NOTE_KEYS:
        p["copy"]["notes"][key] = "x" * SOFT_LIMITS[key]
    p["copy"]["name"] = "x" * SOFT_LIMITS["name"]
    assert validate(p) == []

    for key in NOTE_KEYS:
        p = make_project()
        p["copy"]["notes"][key] = "x" * (SOFT_LIMITS[key] + 1)
        warnings = validate(p)
        assert len(warnings) == 1 and f"notes.{key}" in warnings[0]

    # The notes come first, in the order of the card, then the name
    p = make_project()
    p["copy"]["notes"]["br"] = "x" * (SOFT_LIMITS["br"] + 1)
    p["copy"]["name"] = "x" * (SOFT_LIMITS["name"] + 1)
    warnings = validate(p)
    assert len(warnings) == 2
    assert "notes.br" in warnings[0] and "name" in warnings[1]

    # There is no title to be over a limit: a hook that is somehow in the copy is not looked at
    p = make_project()
    p["copy"]["hook"] = ["x" * 100, "x" * 100]
    assert validate(p) == []


def test_the_contrast_sentence_shows_the_ratio_floored_to_one_decimal(monkeypatch):
    # The guard turns down anything under 3.0, so a 2.95 must never read "3.0:1 is hard to read": the sentence
    # cuts the ratio to one decimal, it does not round it (the studio's page does the same)
    p = make_project()
    for ratio, shown in ((2.95, "2.9"), (2.999, "2.9"), (2.5, "2.5"), (2.0, "2.0"), (1.04, "1.0"), (1.138, "1.1")):
        monkeypatch.setattr(project_file, "text_contrast", lambda style, ratio=ratio: ratio)
        assert validate(p) == [f"Text contrast {shown}:1 is hard to read on a phone."], ratio
    for ratio in (3.0, 3.04, 12.97):
        monkeypatch.setattr(project_file, "text_contrast", lambda style, ratio=ratio: ratio)
        assert validate(p) == [], ratio  # (3.0 and over passes)


@pytest.mark.parametrize("key", ["hud_top", "hud_bottom", "tc", "BL", "burst2"])
def test_positions_refuse_a_key_that_no_element_can_be_pinned_by(key, tmp_path):
    # A FLAG line can name hud_top, and a patch that pinned it changed nothing at all: the layout only
    # reads tl, tr, bl, br and burst. So the patch is refused, with the keys that work.
    p = make_project()
    p["positions"][key] = [100, 200]

    with pytest.raises(ValueError) as error:
        validate(p)
    assert str(error.value) == f"positions.{key} can't be set: only tl, tr, bl, br and burst can be moved."

    # through a patch (what the command line and the studio do), nothing is saved
    save(tmp_path, make_project())
    before = (tmp_path / PROJECT_FILE).read_bytes()
    with pytest.raises(ValueError, match=rf"positions\.{key} can't be set"):
        update(tmp_path, {"positions": {key: [100, 200]}})
    assert (tmp_path / PROJECT_FILE).read_bytes() == before


def test_positions_still_take_the_five_keys_that_work():
    p = make_project()
    p["positions"] = {"tl": [10, 20], "tr": None, "bl": [30.5, 40], "br": [0, 0], "burst": [500, 100]}
    assert validate(p) == []

    # and fewer of them: a file written before the burst had a key of its own
    p["positions"] = {"tl": [10, 20]}
    assert validate(p) == []
    p["positions"] = {}
    assert validate(p) == []

    # it is still an object
    p["positions"] = [[10, 20]]
    with pytest.raises(ValueError, match="positions must be an object"):
        validate(p)
def test_validate_strips_emoji_with_warning():
    p = make_project()
    p["copy"]["notes"]["br"] = f"Best friend: Clover {ROBOT}"
    warnings = validate(p)
    assert p["copy"]["notes"]["br"] == "Best friend: Clover"
    assert len(warnings) == 1
    assert "emoji" in warnings[0] and "notes.br" in warnings[0]

    # Every copy string is cleaned: notes, name, extras and all three versions.
    p = make_project()
    live = p["copy"]
    live["notes"]["tl"] = f"Creator{VS16}"
    live["notes"]["tr"] = f"Hello {ROBOT} world"  # the gap the emoji leaves is closed
    live["notes"]["br"] = f"Best friend: Clover {SPARKLES}"
    live["name"] = f"Robin {ROBOT}"
    live["extras"]["burst"] = f"ROBIN{ROBOT}"
    live["extras"]["emphasis"] = f"{ROBOT}overthinks"
    live["extras"]["stats"] = [[f"Overthinking {ROBOT}", 10], ["Snacks", 7], ["Sleep", 2]]
    p["versions"]["B"]["notes"]["bl"] = f"But {ROBOT} naps"
    warnings = validate(p)
    assert live["notes"]["br"] == "Best friend: Clover"
    assert live["notes"]["tl"] == "Creator"
    assert live["notes"]["tr"] == "Hello world"
    assert live["name"] == "Robin"
    assert live["extras"]["burst"] == "ROBIN"
    assert live["extras"]["emphasis"] == "overthinks"
    assert live["extras"]["stats"] == [["Overthinking", 10], ["Snacks", 7], ["Sleep", 2]]
    assert p["versions"]["B"]["notes"]["bl"] == "But naps"
    assert len(warnings) == 8 and all("emoji" in warning for warning in warnings)
    assert "versions.B.notes.bl" in " ".join(warnings)
    assert validate(p) == []  # nothing left to clean the second time

    # Text without emoji is left exactly as written, accents and dashes included.
    p = make_project()
    p["copy"]["notes"]["tl"] = f"  Café {EM_DASH} née & fine  "
    assert validate(p) == []
    assert p["copy"]["notes"]["tl"] == f"  Café {EM_DASH} née & fine  "

    # A hard error leaves the project untouched, so the emoji is still there.
    p = make_project()
    p["copy"]["notes"]["br"] = f"Claude {ROBOT}"
    p["timing"]["hold"] = 99
    with pytest.raises(ValueError):
        validate(p)
    assert p["copy"]["notes"]["br"] == f"Claude {ROBOT}"


def test_validate_rejects_stat_value_over_10():
    p = make_project()
    p["copy"]["extras"]["stats"] = [["Overthinking", 11], ["Snacks", 7], ["Sleep", 2]]
    with pytest.raises(ValueError, match="stats"):
        validate(p)

    # 0 and 10 are fine, and so are fractions.
    p["copy"]["extras"]["stats"] = [["Overthinking", 10], ["Snacks", 0], ["Sleep", 7.5]]
    assert validate(p) == []

    # Just over 10, below 0, text and the wrong number of stats are refused, in a version too.
    for bad in (
        [["A", 10.01], ["B", 1], ["C", 1]],
        [["A", -1], ["B", 1], ["C", 1]],
        [["A", "10"], ["B", 1], ["C", 1]],
        [["A", 1], ["B", 1]],
    ):
        p = make_project()
        p["versions"]["C"]["extras"]["stats"] = bad
        with pytest.raises(ValueError, match="stats"):
            validate(p)
