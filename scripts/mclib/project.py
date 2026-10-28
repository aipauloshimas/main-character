"""The project file: its default shape, atomic load and save, merge patch and validation.

Everything the studio page and Claude change lives in one JSON file, project.json, in the
clip's project folder.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import unicodedata
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from . import config, glyphs
from .fileio import retry_locked
from .palette import floor_contrast, text_contrast
from .styles.recipes import complete as complete_recipe
from .styles.recipes import validate_recipe

SCHEMA = 2  # 2 is the clip's own shape (Amendment 1); 1 had the letterbox, the full screen and a title (the hook)
PROJECT_FILE = "project.json"
LAYOUTS = ("landscape", "portrait")  # the shape of a clip: wide or square, or tall (plan_size has its size)
_SCHEMA_1_LAYOUTS = {"letterbox": "landscape", "fullscreen": "portrait"}
# Held from the read to the save by whatever reads project.json, changes it and writes it back, so that two of
# them in this program (the studio's patches, the render's record of its status) cannot each save a file that
# lacks the other's change. It is one lock for the whole program, and re-entrant: the code that holds it may call
# the code that takes it again.
UPDATE_LOCK = threading.RLock()
NOTE_KEYS = ("tl", "tr", "bl", "br")  # top left, top right, bottom left, bottom right
VERSION_KEYS = ("A", "B", "C")
# Soft limits, in characters. Going over only warns; the text is never cut. (bubble is the speech bubble's
# text, copy.extras.bubble.)
SOFT_LIMITS = {"tl": 30, "tr": 24, "bl": 34, "br": 30, "name": 16, "bubble": 14}

_POSITION_KEYS = NOTE_KEYS + ("burst",)
_POSITION_NAMES = ", ".join(_POSITION_KEYS[:-1]) + f" and {_POSITION_KEYS[-1]}"  # "tl, tr, bl, br and burst"
_REVEAL_MODES = ("staggered", "all")
_MATTE_DETAILS = ("auto", "high", "max")
_MATTE_EDGE = (-4, 4)
_STAT_RANGE = (0, 10)
_REMOVED_AUDIO_KEYS = ("track", "onsets")  # the music: an older file has them, and load drops them
# "picked", "style_request" and "notice" are not listed: a merge patch (RFC 7386) deletes a key when it
# sets it to null, so "nothing picked" can reach the file as a missing key.
_REQUIRED_KEYS = (
    "schema", "clip", "layout", "work", "freeze", "timing", "audio", "style", "style_source",
    "matte", "versions", "copy", "positions", "status",
)

# Emoji for this project, as code point ranges: pictographs U+1F000-U+1FAFF, symbols and
# dingbats U+2600-U+27BF, and the emoji variation selector U+FE0F. The note fonts cannot draw them.
_EMOJI_RANGES = ((0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0xFE0F, 0xFE0F))
_EMOJI = re.compile("[" + "".join(f"{chr(first)}-{chr(last)}" for first, last in _EMOJI_RANGES) + "]")


def empty_copy() -> dict:
    """A copy block with no text yet: four notes, a name and the extras. There is no title (a hook): the
    person adds that in the app, and the skill only suggests one in the caption (Amendment 1)."""
    return {
        "notes": {key: "" for key in NOTE_KEYS},
        "name": "",
        "extras": {"burst": None, "emphasis": None, "stats": None, "bubble": None},
    }


def empty_positions() -> dict:
    """The positions of a project whose notes are all where the layout puts them: none is pinned. The
    pins are in px of the plan, so a project that is cut at another plan starts with these."""
    return {key: None for key in _POSITION_KEYS}


def default_project(clip: dict, layout: str, work: tuple[int, int], style: dict) -> dict:
    """A new project for a probed clip: default timing, empty copy, nothing picked yet.

    The clip and the style recipe are copied in, so changing the project never changes them.
    The freeze frame starts at 0 until the analysis suggests one, and style_source starts as
    the recipe's id.
    """
    return {
        "schema": SCHEMA,
        "clip": deepcopy(clip),
        "layout": layout,
        "work": {"width": int(work[0]), "height": int(work[1])},
        "freeze": {"frame": 0, "suggested": []},
        "timing": dict(config.TIMING_DEFAULTS),
        "audio": {"original": True},
        "style": deepcopy(style),
        "style_source": str(style.get("id", "")),
        "matte": {"detail": "auto", "edge": 0},
        "versions": {key: empty_copy() for key in VERSION_KEYS},
        "picked": None,
        "copy": empty_copy(),
        "positions": empty_positions(),
        "style_request": None,
        "status": {"rendered_at": None, "outputs": [], "done_at": None},
    }


def load(project_dir: Path) -> dict:
    """Read project.json from a project folder. An older file comes back as a project is now (see
    _migrated); the file itself is changed when the project is next saved."""
    with open(Path(project_dir) / PROJECT_FILE, encoding="utf-8-sig") as handle:
        return _migrated(json.load(handle))


def _migrated(project):
    """A project from an older file as this version holds it, quietly:

    - A project of schema 1 is schema 2. The letterbox is the "landscape" shape and the full screen is
      "portrait", and the work size and the positions of the notes, which are in px of the plan, are the
      plan's of a 16:9 and a 9:16 clip, which is what those two were. A project of another shape of clip
      was cut at a size that is not its plan: init cuts it again, and render says so.
    - The title is gone, whatever the schema: the edit has none (Amendment 1), so the hook of the copy and
      of each version is dropped.
    - The clip keeps its own sound unless the file says otherwise: a file from before the setting has
      audio.original on.
    - There is no music and no trend timing: the skill only makes the freeze-frame effect, and the
      person adds the sound in their own app. timing.freeze_at, audio.track and audio.onsets are
      dropped from a file that has them.
    - The edge of the cutout is a whole number: the studio's slider used to move in halves, so a 1.5 is
      read as 2 (validate would refuse it, and the project could not be changed at all).
    - The style has every little extra: a project from before the eleven new ones has its style with
      only tape, stars and grain, and the rest are off (recipes.complete).

    Everything else stays as it is. Anything that is not a project (a dict) is given back as it is."""
    if not isinstance(project, dict):
        return project
    if project.get("schema") == 1:
        project["schema"] = SCHEMA
        project["layout"] = _SCHEMA_1_LAYOUTS.get(project.get("layout"), project.get("layout"))
    versions = project.get("versions")
    for block in (project.get("copy"), *(versions.values() if isinstance(versions, dict) else ())):
        if isinstance(block, dict):
            block.pop("hook", None)
    audio = project.get("audio")
    if isinstance(audio, dict):
        audio.setdefault("original", True)
        for key in _REMOVED_AUDIO_KEYS:
            audio.pop(key, None)
    timing = project.get("timing")
    if isinstance(timing, dict):
        timing.pop("freeze_at", None)
    matte = project.get("matte")
    if isinstance(matte, dict) and isinstance(matte.get("edge"), float) and math.isfinite(matte["edge"]):
        matte["edge"] = round(matte["edge"])
    if "style" in project:
        project["style"] = complete_recipe(project["style"])
    return project


def save(project_dir: Path, project: dict) -> None:
    """Write project.json atomically: to project.json.<pid>.tmp first, then os.replace over the file.

    A failed write leaves the previous file whole, and a reader never sees half a file. The temp file
    is named for this process, because the command line (patch, style, refine) and the studio's server
    are two programs that save the same file, and they must not write the same temp file; and it does
    not stay behind when the save fails. The folder is created if it is missing. NaN and infinity are
    refused (ValueError), because the studio page reads the file as strict JSON.

    On Windows the waiter, the studio's page or a virus scan can have project.json open at the
    very moment of the replace, and the system refuses it for an instant, so a refusal is waited
    out (see fileio.retry_locked): ten tries 0.1 s apart. A file that stays locked is a
    PermissionError that names it, and the file that was there is left whole.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(project, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temp = project_dir / f"{PROJECT_FILE}.{os.getpid()}.tmp"
    target = project_dir / PROJECT_FILE
    try:
        with open(temp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        retry_locked(lambda: os.replace(temp, target), target)
    except BaseException:
        temp.unlink(missing_ok=True)  # (os.replace moves it, so only a save that failed has one to remove)
        raise


def update(
    project_dir: Path, patch: dict | None = None, mutate: Callable[[dict], None] | None = None
) -> tuple[dict, list[str]]:
    """Change project.json on the disk, as the studio does, and return the project with the
    warnings of validate. This is how the command line writes the file (patch, style and refine).

    Under UPDATE_LOCK, the file is read again (a read waits out a moment in which Windows has the
    file locked, see fileio.retry_locked), then `patch` (a JSON merge patch) is applied, then
    `mutate` (a function that edits the project in place: for what a merge patch cannot say, such as
    a null that stays), and the result is validated and saved. ValueError, with a sentence, when the
    file cannot be read or the result is not a valid project; nothing is saved then. A change of
    the freeze frame has to stay inside the clip. The file is not written when nothing changed.
    """
    project_dir = Path(project_dir)
    path = project_dir / PROJECT_FILE
    with UPDATE_LOCK:
        try:
            current = retry_locked(lambda: load(project_dir), path)
        except ValueError as error:  # not JSON, or not text; a missing file is a FileNotFoundError, as it was
            raise ValueError(
                f"{path} cannot be read ({error}). Fix it, or move it away to start a new project."
            ) from error
        merged = merge_patch(current, patch or {})
        if mutate is not None:
            mutate(merged)
        audio = merged.get("audio")
        if isinstance(audio, dict):
            audio.setdefault("original", True)  # a null here puts the default back: the clip keeps its sound
        warnings = validate(merged)
        _check_freeze_in_clip(current, merged)
        if merged != current:
            save(project_dir, merged)
        return merged, warnings


def _check_freeze_in_clip(before: dict, after: dict) -> None:
    """ValueError unless a freeze frame that was changed is a frame of the clip. A project that is
    past the end already is let through, so that it is not stuck: only a change is looked at.
    `after` has been validated."""
    frame, frames = after["freeze"]["frame"], after["clip"].get("frames")
    old = before.get("freeze")
    changed = frame != (old.get("frame") if isinstance(old, dict) else None)
    if changed and _is_int(frames) and frame >= frames:
        raise ValueError(f"The freeze frame must be between 0 and {frames - 1}: the clip has {frames} frames.")


def merge_patch(target: dict, patch: dict) -> dict:
    """Apply a JSON Merge Patch (RFC 7386) and return the result as a new dict.

    A null in the patch deletes the key, objects merge, and any other value replaces what was
    there. Neither argument is changed, and the result shares nothing with either.
    """
    return _apply_patch(deepcopy(target), patch)


def _apply_patch(target, patch):
    if not isinstance(patch, dict):
        return deepcopy(patch)
    if not isinstance(target, dict):
        target = {}
    for key, value in patch.items():
        if value is None:
            target.pop(key, None)
        else:
            target[key] = _apply_patch(target.get(key), value)
    return target


def validate(project: dict) -> list[str]:
    """Check a project before it is saved or rendered.

    Raises ValueError on the first hard error and leaves the project untouched. Otherwise it
    removes emoji from every copy string in place (the live copy and versions A, B and C) and
    returns the warnings: one for each field it cleaned, one for each note or name of the live copy
    that is over its soft limit, one for each text of the live copy that the card draws in a font
    which can't draw all of its characters (see _glyph_warnings), and, last, one when the style's
    lettering is hard to read on a phone (its text contrast is under config.CONTRAST_MIN).
    """
    _require(isinstance(project, dict), "project must be an object")
    missing = [key for key in _REQUIRED_KEYS if key not in project]
    _require(not missing, "project is missing: " + ", ".join(missing))

    _check_basics(project)
    _check_freeze(project["freeze"])
    _check_audio(project["audio"])
    _check_matte(project["matte"])
    _check_positions(project["positions"])
    _check_timing(project["timing"])
    validate_recipe(project["style"])
    versions = project["versions"]
    _require(
        isinstance(versions, dict) and all(key in versions for key in VERSION_KEYS),
        "versions must have A, B and C",
    )
    blocks = [("copy", project["copy"])] + [(f"versions.{key}", versions[key]) for key in VERSION_KEYS]
    for where, block in blocks:
        _check_copy(block, where)

    warnings: list[str] = []
    for where, block in blocks:
        _clean_copy(block, where, warnings)
    warnings.extend(_soft_limit_warnings(project["copy"]))
    warnings.extend(_glyph_warnings(project["style"], project["copy"]))
    warnings.extend(_contrast_warnings(project["style"]))
    return warnings


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_number(value) -> bool:
    """A real number: an int or a finite float, never a bool."""
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_basics(project: dict) -> None:
    _require(project["schema"] == SCHEMA, f"schema must be {SCHEMA}, got {project['schema']!r}")
    _require(isinstance(project["clip"], dict), "clip must be an object")
    _require(
        project["layout"] in LAYOUTS,
        f'layout must be "landscape" or "portrait", got {project["layout"]!r}',
    )
    work = project["work"]
    _require(
        isinstance(work, dict) and all(_is_int(work.get(key)) and work[key] > 0 for key in ("width", "height")),
        "work must have a positive width and height",
    )
    _require(isinstance(project["style_source"], str), "style_source must be text")
    picked = project.get("picked")
    _require(picked in (None,) + VERSION_KEYS, f'picked must be "A", "B", "C" or null, got {picked!r}')
    notice = project.get("notice")  # a sentence that init leaves for the studio's page to show once (pipeline.py)
    _require(notice is None or isinstance(notice, str), "notice must be text or null")
    request = project.get("style_request")
    _require(request is None or isinstance(request, dict), "style_request must be an object or null")
    status = project["status"]
    _require(
        isinstance(status, dict) and isinstance(status.get("outputs", []), list),
        "status must be an object with a list of outputs",
    )


def _check_freeze(freeze: dict) -> None:
    _require(
        isinstance(freeze, dict) and _is_int(freeze.get("frame")) and freeze["frame"] >= 0,
        "freeze.frame must be a frame number, 0 or more",
    )
    _require(isinstance(freeze.get("suggested"), list), "freeze.suggested must be a list")


def _check_audio(audio: dict) -> None:
    _require(isinstance(audio, dict), "audio must be an object")
    for key in audio:  # (a key nothing reads is a patch that changes nothing: Claude is told, as with positions)
        _require(key == "original", f"audio.{key} can't be set: only audio.original is a setting.")
    _require(isinstance(audio.get("original", True), bool), "audio.original must be true or false")


def _check_matte(matte: dict) -> None:
    _require(isinstance(matte, dict), "matte must be an object")
    detail, edge = matte.get("detail"), matte.get("edge")
    _require(detail in _MATTE_DETAILS, f'matte.detail must be "auto", "high" or "max", got {detail!r}')
    low, high = _MATTE_EDGE
    # A whole number: the preview draws the edge as it is and a render at the clip's size rounds it, so a
    # fraction would be a different edge in each
    _require(
        _is_int(edge) and low <= edge <= high,
        f"matte.edge must be a whole number from {low} to {high}, got {edge!r}",
    )


def _check_positions(positions: dict) -> None:
    _require(isinstance(positions, dict), "positions must be an object")
    for key in positions:  # (a key the layout does not read is a patch that changes nothing: Claude is told)
        _require(key in _POSITION_KEYS, f"positions.{key} can't be set: only {_POSITION_NAMES} can be moved.")
    for key in _POSITION_KEYS:
        point = positions.get(key)
        is_point = isinstance(point, (list, tuple)) and len(point) == 2 and all(_is_number(v) for v in point)
        _require(point is None or is_point, f"positions.{key} must be [x, y] or null, got {point!r}")


def _check_timing(timing: dict) -> None:
    _require(isinstance(timing, dict), "timing must be an object")
    for key in timing:  # (a key nothing reads, such as the old freeze_at, is a patch that changes nothing)
        _require(key in config.TIMING_DEFAULTS, f"timing.{key} can't be set: the settings are {', '.join(config.TIMING_DEFAULTS)}.")
    for key in config.TIMING_DEFAULTS:
        _require(key in timing, f"timing.{key} is missing")
    for key, (low, high) in config.TIMING_RANGES.items():
        value = timing[key]
        _require(
            _is_number(value) and low <= value <= high,
            f"timing.{key} must be a number from {low} to {high}, got {value!r}",
        )
    _require(
        timing["reveal"] in _REVEAL_MODES,
        f'timing.reveal must be "staggered" or "all", got {timing["reveal"]!r}',
    )
    for key in ("boil", "shutter"):
        _require(isinstance(timing[key], bool), f"timing.{key} must be true or false, got {timing[key]!r}")


def _check_copy(block: dict, where: str) -> None:
    """Hard checks on one copy block: the shape, and stats values from 0 to 10."""
    _require(isinstance(block, dict), f"{where} must be an object")
    notes = block.get("notes")
    _require(
        isinstance(notes, dict) and all(isinstance(notes.get(key), str) for key in NOTE_KEYS),
        f"{where}.notes must have text for {', '.join(NOTE_KEYS)}",
    )
    _require(isinstance(block.get("name"), str), f"{where}.name must be text")
    extras = block.get("extras")
    _require(isinstance(extras, dict), f"{where}.extras must be an object")
    for key in ("burst", "emphasis", "bubble"):
        _require(
            extras.get(key) is None or isinstance(extras[key], str),
            f"{where}.extras.{key} must be text or null",
        )
    stats = extras.get("stats")
    if stats is None:
        return
    _require(
        isinstance(stats, list)
        and len(stats) == 3
        and all(isinstance(pair, (list, tuple)) and len(pair) == 2 and isinstance(pair[0], str) for pair in stats),
        f"{where}.extras.stats must be three [label, value] pairs or null",
    )
    low, high = _STAT_RANGE
    for label, value in stats:
        _require(
            _is_number(value) and low <= value <= high,
            f"{where}.extras.stats value for {label!r} must be from {low} to {high}, got {value!r}",
        )


def _strip_emoji(text: str) -> str:
    """The text without emoji. Text that has none comes back exactly as it was."""
    if not _EMOJI.search(text):
        return text
    return " ".join(_EMOJI.sub("", text).split())  # also closes the gap the emoji left


def _clean_copy(block: dict, where: str, warnings: list[str]) -> None:
    """Remove emoji from every string of a copy block in place, with one warning per field."""

    def warn(label):
        warnings.append(f"Removed emoji from {where}.{label}; the video fonts cannot draw them.")

    def clean(container, key, label):
        text = container[key]
        cleaned = _strip_emoji(text)
        if cleaned != text:
            container[key] = cleaned
            warn(label)

    for key in NOTE_KEYS:
        clean(block["notes"], key, f"notes.{key}")
    clean(block, "name", "name")
    extras = block["extras"]
    for key in ("burst", "emphasis", "bubble"):
        if extras.get(key) is not None:
            clean(extras, key, f"extras.{key}")
    for index, (label, value) in enumerate(extras.get("stats") or []):
        cleaned = _strip_emoji(label)
        if cleaned != label:
            extras["stats"][index] = [cleaned, value]
            warn(f"extras.stats[{index}]")


def _shown(char: str) -> str:
    """A character as a sentence names it: as it is, or as its code point when it has nothing to look
    at (a joiner, a control) or only marks the letter before it."""
    if char.isprintable() and not unicodedata.category(char).startswith("M"):
        return char
    return f"U+{ord(char):04X}"


def _glyph_warnings(style: dict, copy: dict) -> list[str]:
    """A warning for each text of the live copy that this style draws in a font that can't draw all
    of its characters (glyphs.json), naming them and the font: "copy.notes.tl has characters the
    Marker font can't draw: <a Cyrillic letter>, <a star>." The text is not touched: the font draws
    a gap for those characters, and the person decides what to write instead. Only what the card
    draws counts (styles.notes.drawn_texts), so the burst's text is looked at with the Name burst
    hero and not with another."""
    from .styles.notes import drawn_texts  # (notes imports this module, so it is not loaded at the top)

    warnings = []
    for where, text, role, case, layer in drawn_texts(style, copy):
        found = glyphs.missing(role, text, case)
        if found:
            warnings.append(
                f"{where} has characters the {layer} font can't draw: {', '.join(_shown(char) for char in found)}."
            )
    return warnings


def _contrast_warnings(style: dict) -> list[str]:
    """A warning when the lettering of the style (a valid recipe) is under the contrast guard's
    minimum, else none."""
    ratio = text_contrast(style)
    if ratio >= config.CONTRAST_MIN:
        return []
    return [f"Text contrast {floor_contrast(ratio)}:1 is hard to read on a phone."]


def _soft_limit_warnings(block: dict) -> list[str]:
    fields = [(f"copy.notes.{key}", block["notes"][key], SOFT_LIMITS[key]) for key in NOTE_KEYS]
    fields.append(("copy.name", block["name"], SOFT_LIMITS["name"]))
    fields.append(("copy.extras.bubble", block["extras"].get("bubble") or "", SOFT_LIMITS["bubble"]))
    return [
        f"{label} is {len(text)} characters; the soft limit is {limit}."
        for label, text, limit in fields
        if len(text) > limit
    ]
