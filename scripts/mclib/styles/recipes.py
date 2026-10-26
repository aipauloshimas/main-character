"""Recipes: the options of each layer, the palettes, the four ready-made styles, and the styles the
person saves.

A recipe is a dict with an id, a name, one option id for each of the layers bg, edge, notes, arrows
and hero, the little extras (tape, stars and grain, each on or off) and a palette of five colors:

    {"id": "bedroom-pop", "name": "Bedroom Pop", "bg": "dots", "edge": "handcut",
     "notes": "marker", "arrows": "curly", "hero": "none",
     "extras": {"tape": False, "stars": False, "grain": True},
     "palette": {"dark": "#7d2a72", "light": "#f2a9de", "tex": "#4c0f45", "edge": "#f5df98",
                 "ink": "#f7e8b0"}}

A recipe has one hero at most, because "hero" is one id. This module must not import the project
file, which imports it.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from copy import deepcopy
from pathlib import Path

from ..config import mc_home
from ..fileio import retry_locked

ROLES = ("dark", "light", "tex", "edge", "ink")  # what the five colors of a palette are for
EXTRAS = ("tape", "stars", "grain")
LAYERS = ("bg", "edge", "notes", "arrows", "hero")  # the layers that take exactly one option
_RECIPE_KEYS = ("id", "name", *LAYERS, "extras", "palette")

# The ids of every layer, and the labels the page shows for them (spec 8a). The extras are listed
# as a layer here too, so that the page can build all six rows from these two tables.
OPTIONS: dict[str, tuple[str, ...]] = {
    "bg": ("dots", "comic", "riso", "night", "flat", "mono"),
    "edge": ("handcut", "comic", "paper", "neon", "thin", "none"),
    "notes": ("marker", "comic", "hand", "pixel", "label"),
    "arrows": ("curly", "doodle", "straight", "none"),
    "hero": ("none", "burst", "stats", "hud"),
    "extras": EXTRAS,
}
LABELS: dict[str, dict[str, str]] = {
    "bg": {
        "dots": "Duotone + dots",
        "comic": "Comic print",
        "riso": "Riso on paper",
        "night": "Night scanlines",
        "flat": "Flat color",
        "mono": "Black & white",
    },
    "edge": {
        "handcut": "Hand-cut sticker",
        "comic": "Comic double line",
        "paper": "Paper + shadow",
        "neon": "Neon glow",
        "thin": "Thin line",
        "none": "No edge",
    },
    "notes": {
        "marker": "Marker",
        "comic": "Comic boxes",
        "hand": "Handwritten strips",
        "pixel": "Pixel",
        "label": "Bold labels",
    },
    "arrows": {"curly": "Curly", "doodle": "Doodle", "straight": "Line + dot", "none": "None"},
    "hero": {"none": "None", "burst": "Name burst", "stats": "Stats panel", "hud": "Game HUD"},
    "extras": {"tape": "Tape", "stars": "Stars", "grain": "Film grain"},
}

# The seven presets, each as {role: "#rrggbb"}.
PALETTES: dict[str, dict[str, str]] = {
    name: dict(zip(ROLES, colors))
    for name, colors in {
        "Magenta": ("#7d2a72", "#f2a9de", "#4c0f45", "#f5df98", "#f7e8b0"),
        "Lagoon": ("#1d4f5c", "#a3e6dc", "#0b2e36", "#ffe9a8", "#fff1c4"),
        "Tangerine": ("#8a3a12", "#ffb98a", "#4a1a06", "#fff0c8", "#fff6dc"),
        "Comic": ("#111111", "#ffd21f", "#e8261c", "#ffffff", "#111111"),
        "Riso": ("#2f4fa8", "#efe4cb", "#2f4fa8", "#fbf8f0", "#1b1b1b"),
        "Neon": ("#07061c", "#5a24c9", "#3ef0ff", "#9ff8ff", "#ffe14d"),
        "Mono": ("#141414", "#e9e6df", "#141414", "#ffe600", "#ffe600"),
    }.items()
}


def _builtin(recipe_id, name, bg, edge, notes, arrows, hero, extras, palette) -> dict:
    """A ready-made recipe: `extras` names the ones that are on, and the palette is a copy of a preset."""
    return {
        "id": recipe_id,
        "name": name,
        "bg": bg,
        "edge": edge,
        "notes": notes,
        "arrows": arrows,
        "hero": hero,
        "extras": {key: key in extras for key in EXTRAS},
        "palette": dict(PALETTES[palette]),
    }


# The four ready-made styles, in the order the page shows them. The columns are _builtin's.
_READY_MADE = (
    ("bedroom-pop", "Bedroom Pop", "dots", "handcut", "marker", "curly", "none", ("grain",), "Magenta"),
    ("comic-pop", "Comic Pop", "comic", "comic", "comic", "curly", "burst", (), "Comic"),
    ("scrapbook", "Scrapbook", "riso", "paper", "hand", "doodle", "none", ("tape", "stars"), "Riso"),
    ("player-select", "Player Select", "night", "neon", "pixel", "none", "stats", (), "Neon"),
)
BUILTINS: dict[str, dict] = {row[0]: _builtin(*row) for row in _READY_MADE}


# --- checking a recipe ------------------------------------------------------------------------

_COLOR = re.compile(r"#[0-9a-fA-F]{6}")


def validate_recipe(recipe) -> None:
    """Raise ValueError, naming the layer or field, unless `recipe` is a whole, valid recipe.

    It needs every key, a name that is not empty, an option id of its layer for each of bg, edge,
    notes, arrows and hero, exactly the extras tape, stars and grain as true or false, and exactly
    the five palette roles as "#rrggbb" colors (in any case). Other keys are left alone.
    """
    if not isinstance(recipe, dict):
        raise ValueError("style must be a recipe object")
    missing = [key for key in _RECIPE_KEYS if key not in recipe]
    if missing:
        raise ValueError("style recipe is missing: " + ", ".join(missing))
    if not isinstance(recipe["id"], str):
        raise ValueError(f"style id must be text, got {recipe['id']!r}")
    _check_name(recipe["name"])
    for layer in LAYERS:
        if recipe[layer] not in OPTIONS[layer]:
            raise ValueError(f"{layer} must be one of {', '.join(OPTIONS[layer])}, got {recipe[layer]!r}")
    _check_extras(recipe["extras"])
    _check_palette(recipe["palette"])


def _check_name(name) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"style name must not be empty, got {name!r}")


def _check_extras(extras) -> None:
    wanted = ", ".join(EXTRAS)
    if not isinstance(extras, dict) or set(extras) != set(EXTRAS):
        raise ValueError(f"extras must have exactly {wanted}, each true or false, got {extras!r}")
    for key in EXTRAS:
        if not isinstance(extras[key], bool):
            raise ValueError(f"extras.{key} must be true or false, got {extras[key]!r}")


def _check_palette(palette) -> None:
    wanted = ", ".join(ROLES)
    if not isinstance(palette, dict):
        raise ValueError(f"palette must be an object with the colors {wanted}, got {palette!r}")
    problems = []
    missing = [role for role in ROLES if role not in palette]
    if missing:
        problems.append("missing: " + ", ".join(missing))
    unknown = [str(role) for role in palette if role not in ROLES]
    if unknown:
        problems.append("not a color role: " + ", ".join(unknown))
    if problems:
        raise ValueError(f"palette must have exactly the colors {wanted} ({'; '.join(problems)})")
    for role in ROLES:
        color = palette[role]
        if not isinstance(color, str) or not _COLOR.fullmatch(color):
            raise ValueError(f"palette.{role} must be a color like #rrggbb, got {color!r}")


def unavailable(layer: str, kind) -> Exception:
    """The error for an option that a layer's drawing code does not have: NotImplementedError when
    it is a real option of the layer that has not been built yet, ValueError when it is no option
    at all. The caller raises it."""
    if kind in OPTIONS[layer]:
        return NotImplementedError(f"The {layer} option {kind!r} is not built yet.")
    return ValueError(f"{layer} must be one of {', '.join(OPTIONS[layer])}, got {kind!r}")


# --- the person's own styles ------------------------------------------------------------------


def user_dir() -> Path:
    """Where the person's saved styles live: one JSON file each, in <MC_HOME>/styles."""
    return mc_home() / "styles"


_MAX_SLUG = 60  # the longest a slug is: a name can be a paragraph, and the slug is a file name


def _slug(name: str) -> str:
    """The file name of a style: its name in lowercase, every run of characters other than a-z and
    0-9 as one hyphen, no hyphen at either end, cut to _MAX_SLUG characters, and "style" when
    nothing is left."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:_MAX_SLUG].strip("-") or "style"


def _own_slug(folder: Path, name: str) -> str:
    """The slug of a name that has no letter or digit to make one of (a row of stars, a name in
    another alphabet): "style", or "style-2", "style-3" and so on, the first that holds no other
    name's style. The file of this same name is the one that is written over."""
    number = 1
    while True:
        slug = "style" if number == 1 else f"style-{number}"
        path = folder / f"{slug}.json"
        try:
            taken = json.loads(path.read_text(encoding="utf-8-sig"))["name"] != name if path.exists() else False
        except (OSError, ValueError, KeyError, TypeError):  # (a file that is no style is not ours to write over)
            taken = True
        if not taken:
            return slug
        number += 1


def _entry(recipe_id: str, name: str, builtin: bool, recipe: dict) -> dict:
    """A style as the page and the CLI see it, with a copy of its recipe."""
    return {"id": recipe_id, "name": name, "builtin": builtin, "recipe": deepcopy(recipe)}


def save_user_style(recipe: dict, name: str) -> dict:
    """Save a recipe as one of the person's styles and return it as list_styles shows it.

    The style is called `name`, and its id is "u-" and a slug of the name (see _slug), so "My
    Neon!" is "u-my-neon". It is written to <styles folder>/<slug>.json as {"id", "name",
    "recipe"}, with the id and the name also in the recipe, over the file of the same slug if there
    is one. The recipe handed in is not changed. A recipe that is not valid, or a name that is
    empty, raises ValueError and writes nothing.
    """
    if not isinstance(recipe, dict):
        raise ValueError("style must be a recipe object")
    _check_name(name)
    folder = user_dir()
    slug = _slug(name) if re.search(r"[a-z0-9]", name.lower()) else _own_slug(folder, name)
    stored = deepcopy(recipe)
    stored["id"], stored["name"] = f"u-{slug}", name
    validate_recipe(stored)

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug}.json"
    temp = folder / f"{slug}.json.tmp"
    text = json.dumps({"id": stored["id"], "name": name, "recipe": stored}, indent=2, ensure_ascii=False)
    try:
        with open(temp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
        # A reader never sees half a file. The gallery reads the styles while "Save as my style"
        # writes one, and Windows refuses the replace for a moment then, so it is waited out.
        retry_locked(lambda: os.replace(temp, path), path)
    finally:
        temp.unlink(missing_ok=True)  # gone already after a good save
    return _entry(stored["id"], name, False, stored)


def _user_styles() -> list[dict]:
    """The person's saved styles, in no particular order. A file that cannot be read, is not a
    valid style, or holds an id that is not its own file name is skipped with a warning, so that
    one bad file cannot take the whole list down."""
    folder = user_dir()
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob("*.json")):
        try:
            with open(path, encoding="utf-8-sig") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or data.get("id") != f"u-{path.stem}":
                raise ValueError(f"its id is not u-{path.stem}")
            if "name" not in data or "recipe" not in data:
                raise ValueError("it has no name or no recipe")
            validate_recipe(data["recipe"])
            _check_name(data["name"])
        except (OSError, ValueError) as error:  # json.JSONDecodeError is a ValueError
            warnings.warn(f"Skipped the saved style {path.name}: {error}", UserWarning, stacklevel=3)
            continue
        found.append(_entry(data["id"], data["name"], False, data["recipe"]))
    return found


def list_styles() -> list[dict]:
    """Every style, as {id, name, builtin, recipe}: the four ready-made ones in the order of the
    table, then the person's own, sorted by name (ignoring case). The recipes are copies."""
    ready = [_entry(recipe["id"], recipe["name"], True, recipe) for recipe in BUILTINS.values()]
    own = sorted(_user_styles(), key=lambda style: (style["name"].casefold(), style["id"]))
    return ready + own


def get_style(style_id: str) -> dict:
    """One style as list_styles shows it, by its id. ValueError when there is none."""
    if style_id in BUILTINS:
        recipe = BUILTINS[style_id]
        return _entry(recipe["id"], recipe["name"], True, recipe)
    for style in _user_styles():
        if style["id"] == style_id:
            return style
    raise ValueError(f"There is no style {style_id!r}.")
