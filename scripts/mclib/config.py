"""Constants for /main-character: the plan, the size of the output, timing, fonts and paths.

Times are in seconds and sizes in pixels. Every value here comes from the design spec, so
change it there first.
"""

from __future__ import annotations

import os
from pathlib import Path

# The output is the clip's own frame, in its own shape, at the quality of the clip (Amendment 1): 16:9
# stays 16:9 and 9:16 stays 9:16, with no canvas and no bars around it.
#
# Everything is laid out at the plan, a size 1080 px wide that has the clip's shape, which is where the
# styles were tuned and approved (plan_size: 1080x608 for 16:9, 1080x1920 for 9:16). The studio, the
# frames and the mattes of a project are all at the plan. A render draws the same card at the size of the
# output (output_size), with every size in px multiplied by output width / plan width (raster.Scale).
#
# FPS is the rate the frame counts in this file are written at (STAGGER below), and what the helpers use
# when they are given no rate. The clip's own is media.output_fps.
FPS = 30
PLAN_WIDTH = 1080  # the width of the plan, and the least width of the output
MAX_LONG_SIDE = 3840  # the most the output is across its longer side


def _even(number: int) -> int:
    """A size rounded up to an even number: the video's pixels come in pairs (yuv420p)."""
    return number + number % 2


def plan_size(width: int, height: int) -> tuple[int, int]:
    """The size a clip of width x height (as a player shows it, after any rotation) is laid out at:
    1080 wide, as tall as the clip's shape makes it, to an even number of px. 1920x1080 gives
    1080x608, 1080x1920 gives 1080x1920, 1440x1080 gives 1080x810."""
    return PLAN_WIDTH, _even(round(PLAN_WIDTH * height / width))


def output_size(width: int, height: int) -> tuple[int, int]:
    """The size of the video a clip of width x height is rendered at: the clip's own, to even px,
    but at least 1080 wide (a smaller clip is scaled up, so the notes stay sharp) and at most 3840
    on the longer side (a bigger one is scaled down). 1920x1080 stays 1920x1080, 320x180 is 1080x608
    and 4096x2160 is 3840x2026. The shape is the clip's, give or take a pixel of rounding."""
    scale = max(1, PLAN_WIDTH / width)
    if max(width, height) * scale > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / max(width, height)
    return _even(round(width * scale)), _even(round(height * scale))


def layout_name(width: int, height: int) -> str:
    """The shape of a clip as the project names it: "landscape" when it is as wide as it is tall or wider
    (a square one too), and "portrait" when it is taller. It only names the shape: the size is plan_size's."""
    return "landscape" if width >= height else "portrait"


# The Reels safe area (ruling S1, amended by Amendment 1), for a plan that is tall: at least this many
# times taller than wide, which is a 9:16 clip. Instagram and TikTok draw the caption, the username and
# the audio line over the bottom of the frame, and the like, comment and share buttons in a rail at its
# right, and the person adds the title in the app, at the top. Notes, arrows, hero boxes, tape and stars
# stay out of all three, which are these shares of the plan: the title room is the top TITLE_ROOM of its
# height, the bottom band is from BOTTOM_BAND of its height down, and the rail is from RAIL_LEFT of its
# width to its right edge, from RAIL_TOP of its height down to the bottom band. Any other shape has none:
# card._keepout is where they become boxes.
TALL_PLAN = 1.7
TITLE_ROOM = 0.15625
BOTTOM_BAND = 0.78125
RAIL_LEFT, RAIL_TOP = 0.8704, 0.5

# Timing controls. "hold" is how long the card stays, "tail" how much clip plays after it (at most the
# rest of the clip; the default, the
# longest clip there is, plays all of it), "release_from" how far past the freeze
# frame the clip resumes. "reveal" is "staggered" or "all". "boil" wobbles the card, and
# "shutter" adds a click with a two-frame white flash on the freeze.
TIMING_DEFAULTS = {
    "hold": 3.0,
    "head_speed": 1.0,
    "tail": 30.0,
    "release_from": 0.0,
    "reveal": "staggered",
    "boil": True,
    "shutter": False,
}
TIMING_RANGES = {
    "hold": (1.0, 6.0),
    "head_speed": (0.75, 1.5),
    "tail": (0.0, 30.0),
    "release_from": (0.0, 1.0),
}

# Card animation, in card frames at 30 fps (FPS); timeline.card_states turns them into frames of
# the clip's own rate by time. The bottom notes are on at frame 0, the top notes at
# "top_after" and the arrows at "arrows_after". Everything is off for the last "out_before"
# frames. The boil picks variant (frame // boil_every) % boil_variants. "flash" is the white
# level of the shutter flash on the first two card frames.
STAGGER = dict(top_after=2, arrows_after=5, out_before=6, boil_every=7, boil_variants=3, flash=(0.85, 0.40))

CLIP_RANGE = (2.0, 30.0)  # shortest and longest clip accepted
MIN_BITRATE = 10_000_000  # bits per second: a final render never encodes under 10 Mbps (Q1)
CONTRAST_MIN = 3.0  # lowest text contrast (WCAG ratio) the style guard allows
NOTE_MIN_PX = 36  # smallest size a note's text is drawn at, on the 1080 px plan: about 13 pt on a phone

# Folders of the skill, found from this file: <skill>/scripts/mclib/config.py.
SKILL_DIR = Path(__file__).resolve().parents[2]
FONTS_DIR = SKILL_DIR / "fonts"
STUDIO_DIR = SKILL_DIR / "studio"

# The Robust Video Matting model. It is downloaded only when the person agrees.
MODEL_PATH = Path.home() / ".cache" / "rvm" / "rvm_mobilenetv3_fp32.onnx"
MODEL_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
# The SHA-256 of that file (14,975,696 bytes), the official v1.0.0 release's. A download is kept only if it
# hashes to this, so a proxy's error page or a wrong file can never become the model.
MODEL_SHA256 = "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"

# The fonts of the notes and the hero elements: role -> (file in fonts/, variable-font axis values or None).
# The edit draws no title (Amendment 1), so there is no font for one.
FONT_FILES = {
    "marker": ("PatrickHandSC-Regular.ttf", None),
    "comic": ("Bangers-Regular.ttf", None),
    "hand": ("Caveat[wght].ttf", [700]),
    "pixel": ("PressStart2P-Regular.ttf", None),
    "terminal": ("VT323-Regular.ttf", None),
    "label": ("ArchivoBlack-Regular.ttf", None),
}


def mc_home() -> Path:
    """The folder for the person's saved styles. MC_HOME overrides it, and it is read on every
    call so tests can point it somewhere else."""
    return Path(os.environ.get("MC_HOME", Path.home() / ".main-character"))
