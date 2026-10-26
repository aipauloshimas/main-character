"""Looking at a clip: where the person is, and which frame to freeze on.

Frames are the JPEGs media.extract_frames writes (frames_dir/000123.jpg is frame 123) and mattes
the 8-bit gray PNGs matte.run_rvm writes under the same numbers (matte_dir/000123.png, 255 is
person). An alpha is a float32 array, height x width, 0-1, as matte.load_alpha returns it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import FPS
from .raster import from_image

_FRAME_FILES = "[0-9]" * 6 + ".jpg"  # the names extract_frames gives the frames

# person_boxes
_PERSON = 0.5  # alpha over this is person
_MIN_AREA = 0.01  # a matte with less person than this share of the picture has nobody in it
_HEAD_SHARE = 0.38  # the head is the top 38% of the person's box
_HEAD_PAD = 0.04  # and its sides are padded by this share of the picture's width

# suggest_freeze
_ENTRANCE = 1.0  # seconds; before this the person is still coming in
_FREEZE_MIN_AREA = 0.05  # a candidate fills at least this share of the picture
_FREEZE_MAX_STEP = 0.02  # and its area is within this of the frame before
_FREEZE_GAP = 6  # frames; picks are at least this far apart


# --- where the person is ----------------------------------------------------------------------


def person_boxes(alpha: np.ndarray) -> dict:
    """Where the person is in an alpha: {"bbox", "head", "area"}.

    The person is where alpha is over 0.5, and area is the share of the picture that covers.
    bbox is the box around the person as (x0, y0, x1, y1) in whole pixels, with x1 and y1 one past
    the last pixel, as in a slice. head is the top 38% of that box in height and, across, the
    columns the person covers in those rows, with 4% of the picture's width added on each side
    (up to the picture's edge): the part the notes have to stay off. With less than 1% of the
    picture covered there is nobody, and bbox and head are both None.
    """
    mask = np.asarray(alpha) > _PERSON
    area = float(mask.mean())
    if area < _MIN_AREA:
        return {"bbox": None, "head": None, "area": area}

    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1

    head_y1 = y0 + max(1, round(_HEAD_SHARE * (y1 - y0)))  # at least one row, for a tiny person
    across = np.flatnonzero(mask[y0:head_y1].any(axis=0))
    width = mask.shape[1]
    pad = round(_HEAD_PAD * width)
    head = (max(int(across[0]) - pad, 0), y0, min(int(across[-1]) + 1 + pad, width), head_y1)
    return {"bbox": (x0, y0, x1, y1), "head": head, "area": area}


# --- the best frame to freeze -----------------------------------------------------------------


def sharpness(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """How much fine detail a picture has inside box = (x0, y0, x1, y1): the variance of its
    4-neighbor Laplacian (up, down, left and right of a pixel, minus four times the pixel). An
    out-of-focus or smeared picture has little of it, a flat one none. gray is luminance, height x
    width. Only the pixels in the box count, and a box under 3 x 3 pixels has no detail to measure,
    so it gives 0.
    """
    x0, y0, x1, y1 = box
    crop = np.asarray(gray[y0:y1, x0:x1], np.float32)
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return 0.0
    laplacian = (
        crop[:-2, 1:-1] + crop[2:, 1:-1] + crop[1:-1, :-2] + crop[1:-1, 2:] - 4 * crop[1:-1, 1:-1]
    )
    return float(laplacian.var())


def _read_alpha(matte_dir: Path, number: int) -> np.ndarray:
    """The raw alpha of frame number. It is not cleaned the way matte.load_alpha does it, which
    takes tens of milliseconds a frame, and boxes and areas do not need that."""
    with Image.open(matte_dir / f"{number:06d}.png") as image:
        return from_image(image.convert("L"))


def _read_gray(frames_dir: Path, number: int) -> np.ndarray:
    """Frame number as luminance, float32 in 0-255, height x width."""
    with Image.open(frames_dir / f"{number:06d}.jpg") as image:
        return np.asarray(image.convert("L"), np.float32)


def suggest_freeze(frames_dir: Path, matte_dir: Path, fps: int = FPS, top: int = 3) -> list[int]:
    """The best frames to freeze on, best first: sharp, and with the person standing still.

    A frame is a candidate when it comes at or after the first second (frame round(1.0 * fps), so
    the person has come in), the person fills at least 5% of the picture, and that share is within
    0.02 of the frame before. Its score is the sharpness of the head times (1 - motion), where
    motion is how much the picture changed since the frame before inside the person's box: the
    mean absolute difference in luminance over 255. The best `top` are picked from the highest
    score down, skipping any within 6 frames of a pick already made, so there are at most `top`.
    Frames that tie go to the earlier one.

    When no frame is a candidate it gives [the frame with the most person] instead (the first one,
    on a tie), whichever second it falls in. The mattes are read raw, and every frame and matte is
    read once, so 300 frames of 1080 x 608 take a few seconds. Raises FileNotFoundError when
    frames_dir has no frames.
    """
    frames_dir, matte_dir = Path(frames_dir), Path(matte_dir)
    count = sum(1 for _ in frames_dir.glob(_FRAME_FILES))
    if count == 0:
        raise FileNotFoundError(f"There are no frames (000000.jpg, ...) in {frames_dir}.")

    first = round(_ENTRANCE * fps)
    areas: list[float] = []
    scored: list[tuple[int, float]] = []
    previous = None  # the luminance of the frame before this one
    for number in range(count):
        found = person_boxes(_read_alpha(matte_dir, number))
        area = found["area"]
        areas.append(area)
        gray = _read_gray(frames_dir, number)
        if (
            previous is not None
            and number >= first
            and area >= _FREEZE_MIN_AREA
            and abs(area - areas[number - 1]) < _FREEZE_MAX_STEP
        ):
            x0, y0, x1, y1 = found["bbox"]
            motion = float(np.abs(gray[y0:y1, x0:x1] - previous[y0:y1, x0:x1]).mean()) / 255
            scored.append((number, sharpness(gray, found["head"]) * (1 - motion)))
        previous = gray

    if not scored:
        return [int(np.argmax(areas))]

    picks: list[int] = []
    for number, _ in sorted(scored, key=lambda item: -item[1]):  # stable: a tie keeps the earlier
        if len(picks) >= top:
            break
        if all(abs(number - other) >= _FREEZE_GAP for other in picks):
            picks.append(number)
    return picks
