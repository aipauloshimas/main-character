"""Tests for the analysis of a clip: where the person is, and which frame makes the best freeze.

The freeze tests run on small clips written into tmp_path, named the way extract_frames and
run_rvm name their files, so a frame number a test expects is a file name. A noise texture in a
box stands in for the person: blurred it is a person out of focus, and shifted a little from one
frame to the next it is a person who is walking.
"""

import numpy as np
import pytest
from PIL import Image, ImageFilter

from mclib.analyze import person_boxes, sharpness, suggest_freeze

W, H = 192, 108  # a clip frame in miniature, the shape of the 1080 x 608 plan
TEXTURE = np.random.default_rng(5).integers(40, 216, size=(H, W), dtype=np.uint8)
STILL = (64, 20, 128, 84)  # the person's box: 64 x 64 px, a fifth of the frame
HEAD_BAND = (64, 20, 128, 44)  # the top 38% of STILL (24 of its 64 rows): where the head is
BELOW_HEAD = (64, 44, 128, 84)  # and the rest of it


def write_clip(folder, shots):
    """Write a clip with a frame for each shot: frames/000000.jpg on and matte/000000.png on.

    A shot is (box, blur) or (box, blur, detail). The person is the box (x0, y0, x1, y1), or
    nobody when box is None. The box is filled with TEXTURE on flat gray, or, when detail is given,
    only the part of the box inside detail is, and the rest of the person is flat gray. blur is the
    radius of a Gaussian blur over the picture (0 is none). The matte is 255 inside the whole box
    and 0 around it. Returns (frames_dir, matte_dir).
    """
    frames, mattes = folder / "frames", folder / "matte"
    frames.mkdir(parents=True)
    mattes.mkdir()
    for number, shot in enumerate(shots):
        box, blur = shot[:2]
        picture = np.full((H, W), 128, np.uint8)
        matte = np.zeros((H, W), np.uint8)
        if box:
            x0, y0, x1, y1 = box
            matte[y0:y1, x0:x1] = 255
            dx0, dy0, dx1, dy1 = shot[2] if len(shot) > 2 else box
            picture[dy0:dy1, dx0:dx1] = TEXTURE[: dy1 - dy0, : dx1 - dx0]
        image = Image.fromarray(picture)
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(blur))
        image.convert("RGB").save(frames / f"{number:06d}.jpg", quality=95)
        Image.fromarray(matte).save(mattes / f"{number:06d}.png")
    return frames, mattes


def sharp(count):
    """Shots of a person standing still and in focus."""
    return [(STILL, 0)] * count


def blurred(count):
    """Shots of a person standing still, out of focus."""
    return [(STILL, 2.5)] * count


def moving(count):
    """Shots of a person walking 2 px a frame, out of focus."""
    return [((10 + 2 * step, 20, 74 + 2 * step, 84), 2.5) for step in range(count)]


# --- person_boxes -----------------------------------------------------------------------------


def test_person_boxes_head_above_body(person_alpha):
    w, h = 1080, 608
    alpha = person_alpha(w, h)
    found = person_boxes(alpha)
    x0, y0, x1, y1 = found["bbox"]
    hx0, hy0, hx1, hy1 = found["head"]

    # The fixture's shoulders reach from 0.19 w to 0.81 w and run off the bottom of the picture,
    # and its head tops out at 0.08 h. y1 is exclusive, so it is h itself.
    assert abs(x0 - 0.19 * w) <= 2 and abs(x1 - 0.81 * w) <= 2
    assert abs(y0 - 0.08 * h) <= 2 and y1 == h
    assert 0.1 < found["area"] < 0.6
    assert found["area"] == pytest.approx((alpha > 0.5).mean())  # the mask, not the soft alpha

    # The head is the top 38% of the box, so it ends above the middle of the person. It is as wide
    # as the fixture's head, 0.10 w either side of the middle, with 4% of w added on each side.
    assert hy0 == y0
    assert abs((hy1 - hy0) - 0.38 * (y1 - y0)) <= 1
    assert hy1 < (y0 + y1) / 2
    assert abs(hx0 - 0.36 * w) <= 2 and abs(hx1 - 0.64 * w) <= 2

    # A plain rectangle shows the conventions exactly. x1 and y1 are one past the person, as in a
    # slice. The head is round(0.38 * 80) = 30 rows, and its sides are padded by round(0.04 * 1080)
    # = 43 px.
    plain = np.zeros((h, w), np.float32)
    plain[100:180, 200:290] = 1.0
    boxed = person_boxes(plain)
    assert boxed["bbox"] == (200, 100, 290, 180)
    assert boxed["head"] == (157, 100, 333, 130)
    # Plain ints and a plain float, which the project file can hold as JSON.
    assert all(type(value) is int for box in (boxed["bbox"], boxed["head"]) for value in box)
    assert type(boxed["area"]) is float

    # The padding stops at the edges of the picture, on both sides.
    plain = np.zeros((h, w), np.float32)
    plain[100:180, 20:] = 1.0
    assert person_boxes(plain)["head"] == (0, 100, 1080, 130)


def test_person_boxes_empty():
    nobody = np.zeros((608, 1080), np.float32)
    assert person_boxes(nobody) == {"bbox": None, "head": None, "area": 0.0}

    # The mask is alpha > 0.5, so an alpha of exactly 0.5 is not the person, and 0.51 is.
    half = np.full((608, 1080), 0.5, np.float32)
    assert person_boxes(half) == {"bbox": None, "head": None, "area": 0.0}
    assert person_boxes(half + 0.01)["bbox"] == (0, 0, 1080, 608)

    # Under 1% of the picture there is nobody to box, but the area is still told, because the
    # freeze suggestion falls back on it. In a 100 x 100 picture 99 pixels are 0.99%, and 100
    # pixels are exactly 1%, which is enough, even for a person one row tall.
    strip = np.zeros((100, 100), np.float32)
    strip[0, :99] = 1.0
    found = person_boxes(strip)
    assert found["bbox"] is None and found["head"] is None
    assert found["area"] == pytest.approx(0.0099)

    strip[0, :] = 1.0
    found = person_boxes(strip)
    assert found["bbox"] == (0, 0, 100, 1)
    assert found["head"] == (0, 0, 100, 1)
    assert found["area"] == pytest.approx(0.01)


# --- sharpness and suggest_freeze -------------------------------------------------------------


def test_suggest_freeze_prefers_sharp_still_frame(tmp_path):
    # The sharpness is the variance of the 4-neighbor Laplacian inside the box. Flat is 0, and in a
    # checkerboard of 0 and 255 every pixel's Laplacian is 4 * 255 = +-1020.
    flat = np.full((20, 30), 90.0)
    checker = (np.indices((20, 30)).sum(axis=0) % 2) * 255.0
    assert sharpness(flat, (0, 0, 30, 20)) == 0.0
    assert sharpness(checker, (0, 0, 30, 20)) == pytest.approx(1020.0**2)
    both = flat.copy()
    both[:, :15] = checker[:, :15]
    assert sharpness(both, (0, 0, 10, 20)) == pytest.approx(1020.0**2)
    assert sharpness(both, (20, 0, 30, 20)) == 0.0
    # A box under 3 pixels across has no detail to measure.
    assert sharpness(checker, (0, 0, 2, 20)) == 0.0
    assert sharpness(checker, (0, 0, 30, 2)) == 0.0

    # A person who walks out of focus until frame 44, then stands still and in focus. Frames 46 to
    # 59 are one picture, so they tie and the earlier wins. Frame 45 is that picture too, but it
    # follows a frame that moved, so its motion is not 0 and it ranks below them. The picks are 6
    # frames apart or more. The folder has an accent and a space in its name, as a person's do.
    odd_folder = tmp_path / ("caf" + chr(0xE9) + " clip") / "walk"
    frames, mattes = write_clip(odd_folder, moving(45) + sharp(15))
    assert suggest_freeze(frames, mattes) == [46, 52, 58]

    # Nobody moves here, so only the sharpness tells the frames apart. Frames 30 to 44 are out of
    # focus, and the 30 before them are in focus but come before the first second.
    frames, mattes = write_clip(tmp_path / "focus", sharp(30) + blurred(15) + sharp(15))
    assert suggest_freeze(frames, mattes) == [46, 52, 58]
    assert suggest_freeze(frames, mattes, top=1) == [46]
    # fps says where the first second ends: at 10 fps, frame 10 is the first that counts.
    assert suggest_freeze(frames, mattes, fps=10) == [10, 16, 22]

    # Where the detail is matters: it is the head that is measured, the top 38% of the box. Up to
    # frame 44 the detail is only below the head, and from frame 45 on only in it. Measured over
    # the whole box, the whole frame or the bottom of the box, the lower part (the bigger one)
    # would win and the first frames would be picked, [30, 36, 42]. On the head it is the face
    # that comes into focus at frame 45 that wins, and frame 45 itself moves.
    face_late = [(STILL, 0, BELOW_HEAD)] * 45 + [(STILL, 0, HEAD_BAND)] * 15
    frames, mattes = write_clip(tmp_path / "face", face_late)
    assert suggest_freeze(frames, mattes) == [46, 52, 58]


def test_suggest_freeze_without_person(tmp_path):
    speck = (80, 40, 105, 65)  # 25 x 25 px: 3% of the frame, under the 5% a candidate needs
    shots = [(None, 0)] * 60
    shots[30] = (speck, 0)
    frames, mattes = write_clip(tmp_path / "speck", shots)
    assert suggest_freeze(frames, mattes) == [30]

    # A person who is small but still and in focus is no candidate either. The frame with the most
    # person is 12, where the box is a pixel bigger on each side. It comes before the first second,
    # which does not matter to the fallback: it looks at every frame.
    shots = [(speck, 0)] * 60
    shots[12] = ((80, 40, 106, 66), 0)
    frames, mattes = write_clip(tmp_path / "small", shots)
    assert suggest_freeze(frames, mattes) == [12]

    # The area must also hold steady. Here it swings by 3% of the frame from one frame to the next,
    # more than the 2% allowed, so nothing qualifies although every frame has a big person.
    small, big, bigger = (64, 20, 109, 66), (60, 18, 112, 70), (56, 16, 114, 73)
    shots = [small, big] * 30
    shots[41] = bigger
    frames, mattes = write_clip(tmp_path / "swing", [(box, 0) for box in shots])
    assert suggest_freeze(frames, mattes) == [41]

    # The same swing at 1.6% is within the limit, so now there are candidates: three picks. The
    # person is 6.1% and 7.6% of the frame, just over the 5% a candidate needs.
    small, big = (64, 20, 99, 56), (62, 20, 98, 64)
    frames, mattes = write_clip(tmp_path / "steady", [(box, 0) for box in [small, big] * 30])
    picks = suggest_freeze(frames, mattes)
    assert len(picks) == 3 and min(picks) >= 30
    assert all(abs(a - b) >= 6 for a in picks for b in picks if a != b)

    # With no frames at all there is nothing to suggest.
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no frames"):
        suggest_freeze(tmp_path / "empty", tmp_path / "empty")
