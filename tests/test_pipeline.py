"""Tests for init: turning a clip into a project folder.

RVM is replaced by conftest's fake_matte, a runner that writes the cutout of a synthetic person (or
of nobody) for every frame it is asked for, and keeps a list of its calls. That makes init quick
and lets a test see whether the cutout was made, for which frames and at what detail. The slow
parts, cutting the frames and looking at them, are real.

Most tests look at what one init made, which is made once (initialized) or on a small clip
(small_clip).
"""

import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from conftest import FakeMatte
from PIL import Image

from mclib import card, matte, media, pipeline, render
from mclib import project as project_file
from mclib.matte import ModelMissing
from mclib.styles.recipes import BUILTINS


def make_clip(path, seconds, rate, size="320x180"):
    """A small test-pattern clip written with ffmpeg: the length in seconds, the rate and the size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size={size}:rate={rate}:duration={seconds}", "-c:v", "libx264", "-preset",
         "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def jpegs(folder):
    return sorted(Path(folder).glob("*.jpg"))


def pngs(folder):
    return sorted(Path(folder).glob("*.png"))


def sizes(paths):
    """The set of sizes of the pictures at these paths."""
    found = set()
    for path in paths:
        with Image.open(path) as image:
            found.add(image.size)
    return found


class Initialized(NamedTuple):
    """The synthetic clip after one init: the project folder, the warnings and the runner."""

    dir: Path
    warnings: list
    runner: object


@pytest.fixture(scope="module")
def initialized(tmp_path_factory, synthetic_clip, fake_matte):
    runner = fake_matte()
    folder, warnings = pipeline.init_project(
        synthetic_clip, tmp_path_factory.mktemp("init") / "proj", matte_runner=runner)
    return Initialized(folder, warnings, runner)


@pytest.fixture(scope="module")
def small_clip(tmp_path_factory):
    """2.4 s at 10 fps, 320x180: 24 frames, quick to cut however it is laid out."""
    return make_clip(tmp_path_factory.mktemp("small") / "small.mp4", 2.4, 10)


@pytest.fixture(scope="module")
def small_portrait_clip(tmp_path_factory):
    """The same, 180x320: 9:16, laid out at 1080x1920."""
    return make_clip(tmp_path_factory.mktemp("small-portrait") / "tall.mp4", 2.4, 10, size="180x320")


# --- the five tests of the task -------------------------------------------------------------


def test_init_builds_project(initialized, synthetic_clip):
    folder, warnings, runner = initialized

    project = project_file.load(folder)
    assert project_file.validate(project) == []  # a whole project, and no soft warnings
    assert warnings == []
    assert folder.is_dir() and project["layout"] == "landscape"
    assert project["work"] == {"width": 1080, "height": 608}
    assert project["clip"]["frames"] == 120
    assert len(jpegs(folder / "frames")) == len(pngs(folder / "matte")) == 120
    assert sizes(jpegs(folder / "frames")) == sizes(pngs(folder / "matte")) == {(1080, 608)}

    # The freeze is the first of the suggestions, all from after the first second
    freeze = project["freeze"]
    assert freeze["frame"] == freeze["suggested"][0]
    assert 1 <= len(freeze["suggested"]) <= 3 and min(freeze["suggested"]) >= 30

    # A new project is a Bedroom Pop one with nothing written yet
    assert project["style"]["id"] == project["style_source"] == "bedroom-pop"
    assert project["style"] == BUILTINS["bedroom-pop"]
    assert project["copy"] == project_file.empty_copy() and project["picked"] is None

    # and it says what the clip is, and what it was read from, so that a later init can tell
    clip, found = project["clip"], synthetic_clip.stat()
    assert clip["path"] == str(synthetic_clip.resolve()) and clip["stem"] == "synthetic"
    assert (clip["width"], clip["height"], clip["orientation"]) == (1920, 1080, "landscape")
    assert clip["fps"] == 30.0 and clip["duration"] == pytest.approx(4.0, abs=0.05)
    assert isinstance(clip["bit_rate"], int) and clip["bit_rate"] > 0
    # the modified time is text: 19 digits are more than a JavaScript number keeps exactly
    assert (clip["size"], clip["mtime"]) == (found.st_size, str(found.st_mtime_ns))

    # The cutout ran over the first second, to see how big the person is, and then went on
    assert runner.calls == [(0.533, 0, 30), (0.533, 30, 120)]

    # The project is one the card can be drawn from, with the freeze frame and its matte
    assert card.render_card(folder, project).size == (1080, 608)


def test_init_rejects_too_long_clip(tmp_path, fake_matte):
    long_clip = make_clip(tmp_path / "long.mp4", 31, 5)
    short_clip = make_clip(tmp_path / "short.mp4", 1, 5)

    for clip in (long_clip, short_clip):
        with pytest.raises(ValueError, match=r"between 2 and 30 seconds"):
            pipeline.init_project(clip, tmp_path / "proj", matte_runner=fake_matte())
    with pytest.raises(ValueError, match="31.0 seconds long"):
        pipeline.init_project(long_clip, tmp_path / "proj", matte_runner=fake_matte())
    assert not (tmp_path / "proj").exists()  # nothing was started


def test_init_reuses_or_rebuilds_cache(small_clip, small_portrait_clip, tmp_path, fake_matte):
    runner = fake_matte()
    folder, _ = pipeline.init_project(small_clip, tmp_path / "proj", matte_runner=runner)
    assert len(runner.calls) == 2  # the first second, and the rest
    first = (folder / "frames" / "000000.jpg").stat().st_mtime_ns

    # The same clip at the same size: the frames and the cutout are kept, and not made again
    pipeline.init_project(small_clip, folder, matte_runner=runner)
    assert len(runner.calls) == 2
    assert (folder / "frames" / "000000.jpg").stat().st_mtime_ns == first

    # A project that was cut at another size than its clip's plan (this is what the full screen made of a
    # 16:9 clip, before the clip's own format) has both made again at the plan, and says so
    stale = project_file.load(folder)
    stale["layout"], stale["work"] = "portrait", {"width": 1080, "height": 1920}
    project_file.save(folder, stale)
    pipeline.init_project(small_clip, folder, matte_runner=runner)
    assert len(runner.calls) > 2
    assert len(jpegs(folder / "frames")) == len(pngs(folder / "matte")) == 24
    assert sizes(jpegs(folder / "frames")) == sizes(pngs(folder / "matte")) == {(1080, 608)}
    project = project_file.load(folder)
    assert (project["layout"], project["work"]) == ("landscape", {"width": 1080, "height": 608})

    # and it is the cache of that size now
    made = len(runner.calls)
    pipeline.init_project(small_clip, folder, matte_runner=runner)
    assert len(runner.calls) == made

    # A clip of another shape is laid out at its own plan: 9:16 is 1080x1920, and the frames and the cutout are
    # cut at that, in a folder of its own
    tall, _ = pipeline.init_project(small_portrait_clip, tmp_path / "tall", matte_runner=runner)
    tall_project = project_file.load(tall)
    assert (tall_project["layout"], tall_project["work"]) == ("portrait", {"width": 1080, "height": 1920})
    assert sizes(jpegs(tall / "frames")) == sizes(pngs(tall / "matte")) == {(1080, 1920)}


def test_init_keeps_copy_on_rerun(small_clip, tmp_path, fake_matte):
    folder, _ = pipeline.init_project(small_clip, tmp_path / "proj", matte_runner=fake_matte())
    project = project_file.load(folder)

    # What the person did with the project: wrote the copy and versions, chose another style,
    # pinned a note, changed the timing and the cutout settings, and picked a freeze frame
    project["copy"]["name"] = "Robin"
    project["copy"]["notes"]["tl"] = "Creator & AI educator"
    project["versions"]["B"]["name"] = "Robin"
    project["picked"] = "B"
    project["style"], project["style_source"] = dict(BUILTINS["comic-pop"]), "comic-pop"
    project["positions"]["tr"] = [700, 40]
    project["timing"]["hold"] = 4.5
    project["freeze"]["frame"] = 17
    project["matte"] = {"detail": "high", "edge": 2}
    project_file.save(folder, project)
    kept = ("copy", "versions", "picked", "style", "style_source", "positions", "timing", "matte")
    runner = fake_matte()

    pipeline.init_project(small_clip, folder, matte_runner=runner)

    again = project_file.load(folder)
    assert runner.calls == []  # nothing was made again
    for key in kept:
        assert again[key] == project[key], key
    assert again["freeze"]["frame"] == 17  # their freeze frame, with the suggestions beside it
    assert again["freeze"]["suggested"] == project["freeze"]["suggested"]

    # Frames cut at another size than the plan make the frames and the cutout again, and keep all of it but
    # the pins: a note is pinned in px of a plan, and where the person put it on another is not where it is here
    project = project_file.load(folder)
    project["layout"], project["work"] = "portrait", {"width": 1080, "height": 1920}
    project_file.save(folder, project)
    pipeline.init_project(small_clip, folder, matte_runner=runner)
    after = project_file.load(folder)
    assert runner.calls != [] and after["layout"] == "landscape"
    assert after["work"] == {"width": 1080, "height": 608}
    for key in kept:
        if key != "positions":
            assert after[key] == project[key], key
    assert after["positions"] == project_file.empty_positions()
    assert after["freeze"]["frame"] == 17

    # A project of the first schema (the letterbox, with a hook) is read as it is now, and kept: its frames
    # and cutout are the plan's already, so nothing is cut again. The edit has no title any more, so the hook
    # (of the copy and of the versions) is not kept, and the file init saves does not have it
    old = project_file.load(folder)
    old["schema"], old["layout"] = 1, "letterbox"
    old["positions"] = project["positions"]  # pinned again, on the plan that it is cut at now
    for block in (old["copy"], *old["versions"].values()):
        block["hook"] = ["this is your sign to", "post it"]
    folder.joinpath(project_file.PROJECT_FILE).write_text(
        json.dumps(old), encoding="utf-8")
    again_runner = fake_matte()
    pipeline.init_project(small_clip, folder, matte_runner=again_runner)
    migrated = project_file.load(folder)
    assert again_runner.calls == []
    assert (migrated["schema"], migrated["layout"]) == (2, "landscape")
    for key in kept:
        assert migrated[key] == project[key], key
    assert "hook" not in folder.joinpath(project_file.PROJECT_FILE).read_text(encoding="utf-8")


def test_init_clears_the_pinned_positions_when_the_plan_changes(
    small_clip, small_portrait_clip, tmp_path, fake_matte
):
    folder, _ = pipeline.init_project(small_clip, tmp_path / "proj", matte_runner=fake_matte())
    project = project_file.load(folder)
    assert project["positions"] == project_file.empty_positions()
    project["positions"].update(tl=[100, 80], burst=[500, 300])  # pinned in px of the 1080x608 plan
    project["copy"]["notes"]["tl"] = "Creator & AI educator"
    project_file.save(folder, project)

    # the same clip is the same plan: the notes stay where they were put
    pipeline.init_project(small_clip, folder, matte_runner=fake_matte())
    assert project_file.load(folder)["positions"] == project["positions"]

    # a clip of another shape in the folder is a plan of 1080x1920, where those px are somewhere else:
    # the notes are free again, and what the person wrote is kept
    pipeline.init_project(small_portrait_clip, folder, matte_runner=fake_matte())
    tall = project_file.load(folder)
    assert tall["work"] == {"width": 1080, "height": 1920}
    assert tall["positions"] == project_file.empty_positions()
    assert set(tall["positions"].values()) == {None}
    assert tall["copy"]["notes"]["tl"] == "Creator & AI educator"


def test_init_says_when_it_clears_pinned_positions(small_clip, small_portrait_clip, tmp_path, fake_matte):
    sentence = "Your note positions were reset because the clip's shape changed."
    assert pipeline.POSITIONS_RESET == sentence
    folder, warnings = pipeline.init_project(small_clip, tmp_path / "proj", matte_runner=fake_matte())
    assert warnings == [] and "notice" not in project_file.load(folder)

    # A clip of another shape, and nothing pinned: no note was moved, so there is nothing lost to say
    _, warnings = pipeline.init_project(small_portrait_clip, folder, matte_runner=fake_matte())
    assert warnings == [] and "notice" not in project_file.load(folder)

    # A pinned note is where the person put it on one plan only: on another, init sets every note free
    # again and says so. The sentence is a warning (the command line prints it as a WARNING line), and it
    # is left in the file as the project's notice, which the studio's page shows as a toast, once
    project = project_file.load(folder)
    project["positions"].update(tl=[100, 80], burst=[500, 300])  # on the 1080x1920 plan
    project["copy"]["notes"]["tl"] = "Creator & AI educator"
    project_file.save(folder, project)
    _, warnings = pipeline.init_project(small_clip, folder, matte_runner=fake_matte())
    assert warnings == [sentence]
    reset = project_file.load(folder)
    assert reset["positions"] == project_file.empty_positions() and reset["notice"] == sentence
    assert reset["copy"]["notes"]["tl"] == "Creator & AI educator"  # and nothing else was touched
    assert project_file.validate(reset) == []

    # The same plan again is no reason to say it: the pins stay and init is quiet
    reset["positions"]["bl"] = [200, 300]
    del reset["notice"]  # (the page showed it and took it away)
    project_file.save(folder, reset)
    _, warnings = pipeline.init_project(small_clip, folder, matte_runner=fake_matte())
    assert warnings == [] and project_file.load(folder)["positions"]["bl"] == [200, 300]
    assert "notice" not in project_file.load(folder)

    # Said next to the other warning init has
    stale = project_file.load(folder)
    stale["positions"]["tl"] = [20, 30]
    project_file.save(folder, stale)
    _, warnings = pipeline.init_project(small_portrait_clip, folder, matte_runner=fake_matte(nobody=True))
    assert len(warnings) == 2 and warnings[0].startswith("No person found") and warnings[1] == sentence


def test_the_share_of_a_matte_that_is_person_is_public(small_clip, tmp_path, fake_matte):
    # render cuts the freeze frame again at the size of the video, and sets how closely by this share
    # of the matte it has, as init does: the one function, with no underscore
    folder, _ = pipeline.init_project(small_clip, tmp_path / "proj", matte_runner=fake_matte(area=0.25))
    assert pipeline.person_area(folder / "matte", 0) == pytest.approx(0.25, abs=0.01)
    assert render.person_area is pipeline.person_area
    assert not hasattr(pipeline, "_person_area")


def test_init_warns_without_person(small_clip, tmp_path, fake_matte):
    folder, warnings = pipeline.init_project(
        small_clip, tmp_path / "proj", matte_runner=fake_matte(nobody=True))

    assert len(warnings) == 1 and warnings[0].startswith("No person found in most frames")
    project = project_file.load(folder)  # and the project is saved all the same
    project_file.validate(project)
    assert len(jpegs(folder / "frames")) == 24
    assert project["freeze"]["suggested"] == [0] and project["freeze"]["frame"] == 0

    # The median decides: a person who fills 1.9% of the frame is nobody, and 2.1% is somebody
    _, warnings = pipeline.init_project(
        small_clip, tmp_path / "other", matte_runner=fake_matte(area=0.019))
    assert len(warnings) == 1
    _, warnings = pipeline.init_project(
        small_clip, tmp_path / "third", matte_runner=fake_matte(area=0.021))
    assert warnings == []


class TwoPeopleMatte(FakeMatte):
    """A cutout of two people side by side: a taller one on the left and a shorter one on the right, with
    a gap between them. `second` is the share of the picture's width the second one is across."""

    def __init__(self, person_alpha, second=0.25):
        super().__init__(person_alpha)
        self._second = second

    def _png(self, width, height):
        alpha = np.zeros((height, width), np.float32)
        alpha[round(0.15 * height):, round(0.10 * width):round(0.35 * width)] = 1.0  # 25% across, 85% tall
        alpha[round(0.35 * height):, round(0.60 * width):round((0.60 + self._second) * width)] = 1.0
        buffer = io.BytesIO()
        Image.fromarray(np.rint(alpha * 255).astype(np.uint8)).save(buffer, "PNG")
        return buffer.getvalue()


def test_init_warns_when_two_people_are_in_the_freeze_frame(small_clip, tmp_path, person_alpha):
    # The notes keep clear of one face. With two people side by side one of them gets notes on its face.
    sentence = "Two people are in the frame; the notes avoid only one face."

    folder, warnings = pipeline.init_project(
        small_clip, tmp_path / "two", matte_runner=TwoPeopleMatte(person_alpha))

    assert warnings == [sentence]
    project_file.validate(project_file.load(folder))  # and the project is made all the same

    # A second region that is a small part of the person (a hand across the frame, a bit of a chair)
    # is not a person: it has to be over 15% of what the cutout holds
    _, warnings = pipeline.init_project(
        small_clip, tmp_path / "hand", matte_runner=TwoPeopleMatte(person_alpha, second=0.03))
    assert warnings == []

    # One person, however big, is one person
    _, warnings = pipeline.init_project(small_clip, tmp_path / "one", matte_runner=FakeMatte(person_alpha))
    assert warnings == []

    # The same clip again, from the cutouts it kept, says it again: the warning is about the frame
    _, warnings = pipeline.init_project(
        small_clip, tmp_path / "two", matte_runner=TwoPeopleMatte(person_alpha))
    assert warnings == [sentence]


def test_the_person_regions_of_a_matte_are_their_shares_of_the_person(person_alpha):
    alpha = np.zeros((608, 1080), np.float32)
    alpha[100:, 0:400] = 1.0  # 400 x 508
    alpha[300:, 600:800] = 1.0  # 200 x 308: 200 * 308 / (400 * 508 + 200 * 308) is 23% of the person
    shares = matte.person_regions(alpha)

    assert shares == pytest.approx([0.7673, 0.2327], abs=0.02)
    assert matte.person_regions(person_alpha(1080, 608)) == [1.0]  # one person
    assert matte.person_regions(np.zeros((608, 1080), np.float32)) == []  # nobody


# --- the cutout's detail, the clip's own rate, and what a rerun does ---------------------------


def test_init_sets_the_detail_from_how_big_the_person_is(small_clip, small_portrait_clip, tmp_path, fake_matte):
    # A 9:16 clip is laid out at 1080x1920, and shows the model 576 / 1920 = 0.3 of the picture, and a person
    # who fills over 45% of the frame needs 0.5. The first second (10 frames of this clip) tells which.
    big = fake_matte(area=0.6)
    pipeline.init_project(small_portrait_clip, tmp_path / "big", matte_runner=big)
    assert big.calls == [(0.3, 0, 10), (0.5, 0, 24)]  # then all of it again, at the closer look

    small = fake_matte(area=0.1)
    folder, _ = pipeline.init_project(small_portrait_clip, tmp_path / "small", matte_runner=small)
    assert small.calls == [(0.3, 0, 10), (0.3, 10, 24)]  # and it goes on from there

    # The detail a project was set to is kept when it is cut again: "high" is 1.6 times as much. (A clip
    # of another shape is another plan, 1080 wide and 608 tall: 576 / 1080 times 1.6.)
    project = project_file.load(folder)
    project["matte"]["detail"] = "high"
    project_file.save(folder, project)
    again = fake_matte(area=0.1)
    pipeline.init_project(small_clip, folder, matte_runner=again)
    assert again.calls == [(0.853, 0, 10), (0.853, 10, 24)]


def test_init_counts_the_frames_of_a_24_fps_clip_at_24(synthetic_clip_24, tmp_path, fake_matte):
    runner = fake_matte()

    folder, warnings = pipeline.init_project(synthetic_clip_24, tmp_path / "proj", matte_runner=runner)

    # 24 x 4 frames, as many as the clip has: the count is what was cut, not a count at 30 fps
    project = project_file.load(folder)
    assert project["clip"]["fps"] == 24.0 and project["clip"]["frames"] == 96
    assert len(jpegs(folder / "frames")) == len(pngs(folder / "matte")) == 96
    assert runner.calls == [(0.533, 0, 24), (0.533, 24, 96)]  # the first second is 24 frames
    freeze = project["freeze"]
    assert 24 <= freeze["frame"] < 96 and set(freeze["suggested"]) <= set(range(24, 96))
    assert warnings == []


def test_init_cuts_again_when_the_clip_or_its_rate_changed(small_clip, tmp_path, fake_matte):
    clip = tmp_path / "clip.mp4"
    shutil.copy(small_clip, clip)
    runner = fake_matte()
    folder, _ = pipeline.init_project(clip, tmp_path / "proj", matte_runner=runner)
    made = len(runner.calls)
    pipeline.init_project(clip, folder, matte_runner=runner)
    assert len(runner.calls) == made  # the same clip: kept

    # the clip was changed, as its modified time says
    found = clip.stat()
    os.utime(clip, ns=(found.st_atime_ns, found.st_mtime_ns + 2_000_000_000))
    pipeline.init_project(clip, folder, matte_runner=runner)
    assert len(runner.calls) > made
    assert project_file.load(folder)["clip"]["mtime"] == str(found.st_mtime_ns + 2_000_000_000)

    # the frame rate it was cut at is part of what is kept: the same clip, read as 23.976 fps before,
    # comes out as 24000/1001 then and 10 fps now, which numbers the frames differently
    made = len(runner.calls)
    project = project_file.load(folder)
    project["clip"]["fps"] = 23.976
    project_file.save(folder, project)
    pipeline.init_project(clip, folder, matte_runner=runner)
    assert len(runner.calls) > made
    assert project_file.load(folder)["clip"]["fps"] == 10.0

    # a different clip, which is another path: cut again
    made = len(runner.calls)
    other = tmp_path / "other.mp4"
    shutil.copy(small_clip, other)
    pipeline.init_project(other, folder, matte_runner=runner)
    assert len(runner.calls) > made and project_file.load(folder)["clip"]["path"] == str(other.resolve())


def test_init_stores_the_modified_time_as_text_and_still_reads_a_number(small_clip, tmp_path, fake_matte):
    clip = tmp_path / "clip.mp4"
    shutil.copy(small_clip, clip)
    # A modified time in nanoseconds that a float cannot hold: this is what a JavaScript round trip
    # through the studio does to a number of this size, and it would look like a changed clip
    found = clip.stat()
    os.utime(clip, ns=(found.st_atime_ns, 1_761_456_789_123_456_789))
    ns = clip.stat().st_mtime_ns
    assert ns > 2**53 and int(float(ns)) != ns

    runner = fake_matte()
    folder, _ = pipeline.init_project(clip, tmp_path / "proj", matte_runner=runner)
    project = project_file.load(folder)
    assert project["clip"]["mtime"] == str(ns)  # every digit
    assert isinstance(project["clip"]["size"], int)
    made = len(runner.calls)

    # a project made before it was text has the number, and its frames and cutout are still kept
    project["clip"]["mtime"] = ns
    project_file.save(folder, project)
    pipeline.init_project(clip, folder, matte_runner=runner)
    assert len(runner.calls) == made
    assert project_file.load(folder)["clip"]["mtime"] == str(ns)  # and it is text from now on

    # a number the page has rounded is another time, so that is a clip that changed
    project = project_file.load(folder)
    project["clip"]["mtime"] = int(float(ns))
    project_file.save(folder, project)
    pipeline.init_project(clip, folder, matte_runner=runner)
    assert len(runner.calls) > made


def test_init_cuts_again_when_the_frames_are_not_there_or_not_whole(small_clip, tmp_path, fake_matte):
    runner = fake_matte()
    folder, _ = pipeline.init_project(small_clip, tmp_path / "proj", matte_runner=runner)
    made = len(runner.calls)

    (folder / "matte" / "000007.png").unlink()  # a cutout that stopped halfway
    pipeline.init_project(small_clip, folder, matte_runner=runner)
    assert len(runner.calls) > made and len(pngs(folder / "matte")) == 24

    made = len(runner.calls)  # and frames of another size, which an init that failed left behind
    for path in jpegs(folder / "frames"):
        Image.new("RGB", (1080, 1920)).save(path)
    pipeline.init_project(small_clip, folder, matte_runner=runner)
    assert len(runner.calls) > made
    assert sizes(jpegs(folder / "frames")) == {(1080, 608)}


def test_init_gives_a_different_clip_its_own_freeze_and_clamps_a_stale_one(
        small_clip, tmp_path, fake_matte):
    # The same clip keeps the person's frame, and a frame past the end is put on the last one
    folder, _ = pipeline.init_project(small_clip, tmp_path / "proj", matte_runner=fake_matte())
    project = project_file.load(folder)
    project["freeze"]["frame"] = 500
    project["copy"]["name"] = "Robin"
    project_file.save(folder, project)
    runner = fake_matte()
    pipeline.init_project(small_clip, folder, matte_runner=runner)
    assert runner.calls == []
    assert project_file.load(folder)["freeze"]["frame"] == 23  # the clip has frames 0 to 23

    # A different clip is different footage, so it gets the suggestion, and the copy stays
    project = project_file.load(folder)
    project["freeze"]["frame"] = 20
    project_file.save(folder, project)
    shorter = make_clip(tmp_path / "shorter.mp4", 2.0, 10)
    pipeline.init_project(shorter, folder, matte_runner=fake_matte())
    after = project_file.load(folder)
    assert after["clip"]["frames"] == 20 and after["copy"]["name"] == "Robin"
    assert after["freeze"]["frame"] == after["freeze"]["suggested"][0] < 20


def test_init_makes_the_default_folder_from_a_path_with_accents_and_spaces(
        small_clip, tmp_path, monkeypatch, fake_matte):
    clip = tmp_path / ("V" + chr(0xED) + "deos de Jo" + chr(0xE3) + "o") / "my clip.mp4"
    clip.parent.mkdir()
    shutil.copy(small_clip, clip)
    working = tmp_path / "working folder"
    working.mkdir()
    monkeypatch.chdir(working)

    folder, _ = pipeline.init_project(clip, matte_runner=fake_matte())

    assert folder == (working / "main-character" / "my clip").resolve()
    project = project_file.load(folder)
    assert project["clip"]["path"] == str(clip.resolve()) and project["clip"]["stem"] == "my clip"
    assert len(jpegs(folder / "frames")) == len(pngs(folder / "matte")) == 24


def test_init_cuts_out_with_rvm_and_needs_the_model_before_anything_else(
        small_clip, tmp_path, monkeypatch, fake_matte):
    def missing(*args, **kwargs):
        raise ModelMissing("The matte model is missing: it should be at somewhere.")

    called = []
    monkeypatch.setattr(matte, "ensure_model", missing)
    monkeypatch.setattr(matte, "run_rvm", lambda *args, **kwargs: called.append(args))

    with pytest.raises(ModelMissing, match="matte model is missing"):
        pipeline.init_project(small_clip, tmp_path / "proj")
    assert called == [] and not (tmp_path / "proj" / "frames").exists()  # no frames cut for nothing

    # With the model there, the runner that is run is matte.run_rvm
    runner = fake_matte()
    monkeypatch.setattr(matte, "ensure_model", lambda *args, **kwargs: Path("model.onnx"))
    monkeypatch.setattr(matte, "run_rvm", runner)
    pipeline.init_project(small_clip, tmp_path / "proj")
    assert runner.calls != []

    # and a project whose frames and cutout are kept does not need the model at all
    monkeypatch.setattr(matte, "ensure_model", missing)
    pipeline.init_project(small_clip, tmp_path / "proj")


def test_init_reports_progress(small_clip, tmp_path, fake_matte):
    steps = []

    pipeline.init_project(small_clip, tmp_path / "proj", matte_runner=fake_matte(area=0.1),
                          progress=lambda *step: steps.append(step))

    assert steps[0] == ("frames", 1, 1) and steps[-1] == ("freeze", 1, 1)
    # the cutout counts across its two runs, the first second and the rest
    assert [step for step in steps if step[0] == "matte"] == [("matte", n, 24) for n in range(1, 25)]
    assert [step[0] for step in steps] == ["frames"] + ["matte"] * 24 + ["freeze"]


def test_init_says_plainly_what_it_cannot_do(small_clip, tmp_path, fake_matte):
    with pytest.raises(FileNotFoundError, match=r"There is no clip at .*nope\.mp4"):
        pipeline.init_project(tmp_path / "nope.mp4", tmp_path / "proj", matte_runner=fake_matte())

    # A project file that cannot be read holds the person's copy, so it is not replaced
    folder = tmp_path / "damaged"
    folder.mkdir()
    (folder / "project.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="project.json cannot be read"):
        pipeline.init_project(small_clip, folder, matte_runner=fake_matte())
    assert (folder / "project.json").read_text(encoding="utf-8") == "{not json"
    assert not (folder / "frames").exists()


@pytest.mark.slow
def test_init_and_render_keep_a_23_976_fps_clip_exact(synthetic_clip_ntsc, tmp_path, fake_matte):
    folder, _ = pipeline.init_project(
        synthetic_clip_ntsc, tmp_path / "proj", matte_runner=fake_matte())
    project = project_file.load(folder)
    assert project["clip"]["fps"] == pytest.approx(24000 / 1001) and project["clip"]["frames"] == 48
    assert len(jpegs(folder / "frames")) == len(pngs(folder / "matte")) == 48

    # a short card, and the freeze on a frame with some clip before it
    project["freeze"]["frame"] = 12
    project["timing"].update(hold=1.0, tail=0.0)
    project_file.save(folder, project)
    result = render.render(folder, matte_runner=fake_matte())

    # 12 frames of clip (0.5 s) and 24 of card (1 s), in frames of 1001/24000 s, at exactly that rate. The
    # clip is 1280x720, so the video is too: the clip's own size, and the freeze frame is cut at it
    rate = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "csv=p=0", str(result.video)], check=True, capture_output=True, text=True).stdout
    assert rate.strip() == "24000/1001"
    assert result.frames == media.count_frames(result.video) == 12 + 24
    assert result.freeze_out == pytest.approx(12 * 1001 / 24000)
    assert result.notes == []


@pytest.mark.slow
def test_init_and_render_a_portrait_clip_in_its_own_format(tmp_path, fake_matte):
    # A clip that is taller than wide stays 9:16, full frame: it is laid out at 1080x1920 from its shape
    clip = make_clip(tmp_path / "portrait.mp4", 2.4, 10, size="540x960")
    folder, warnings = pipeline.init_project(clip, tmp_path / "proj", matte_runner=fake_matte())

    project = project_file.load(folder)
    assert (project["layout"], project["work"]) == ("portrait", {"width": 1080, "height": 1920})
    assert project["clip"]["orientation"] == "portrait" and warnings == []
    assert sizes(jpegs(folder / "frames")) == sizes(pngs(folder / "matte")) == {(1080, 1920)}

    project["freeze"]["frame"] = 5
    project["timing"].update(hold=1.0, tail=0.3)
    project_file.save(folder, project)
    scaled_up = ["Scaled up to 1080x1920 so the notes stay sharp; your clip is 540x960."]

    # 5 frames of clip (0.5 s), 10 of card, 3 of tail, at the clip's 10 fps, 1080x1920 as a preview: the
    # plan's size, so that is the size of the video as well
    preview = render.render(folder, preview=True)
    assert preview.frames == media.count_frames(preview.video) == 5 + 10 + 3
    info = media.probe(preview.video)
    assert (info["width"], info["height"], info["fps"]) == (1080, 1920, 10.0)
    assert preview.notes == scaled_up

    # and the final render of a 540x960 clip is scaled up to 1080 wide, which is the plan's size too, so
    # the freeze frame it is drawn on is the project's own (no cutout is made again)
    runner = fake_matte()
    final = render.render(folder, matte_runner=runner)
    info = media.probe(final.video)
    assert (info["width"], info["height"], info["fps"], info["frames"]) == (1080, 1920, 10.0, 18)
    assert final.notes == scaled_up and runner.calls == []
    # the clip fills the whole frame: there are no bars where a letterbox would have them
    shown = tmp_path / "first.png"
    media.run(["-i", str(final.video), "-frames:v", "1", str(shown)])
    with Image.open(shown) as frame:
        for point in ((540, 150), (540, 700), (540, 1000), (540, 1800)):
            assert sum(frame.convert("RGB").getpixel(point)) > 150, point


def test_a_rotated_phone_clip_is_laid_out_and_rendered_at_its_display_size(
    rotated_phone_clip, tmp_path, fake_matte
):
    # Stored 2560x1440 with a display rotation of 90, a player shows it as 1440x2560, and that is the size
    # of the clip: its plan is the 9:16 one (1080x1920), and its video is 1440x2560, over 1080 wide, so its own
    assert media.probe(rotated_phone_clip, count=False)["rotation"] in (90, 270)
    folder, _ = pipeline.init_project(rotated_phone_clip, tmp_path / "proj", matte_runner=fake_matte())
    project = project_file.load(folder)
    assert (project["clip"]["width"], project["clip"]["height"]) == (1440, 2560)
    assert (project["layout"], project["work"]) == ("portrait", {"width": 1080, "height": 1920})
    assert sizes(jpegs(folder / "frames")) == sizes(pngs(folder / "matte")) == {(1080, 1920)}

    project["freeze"]["frame"] = 15  # half a second in, for a short render
    project["timing"].update(hold=1.0, tail=0.0, boil=False)
    project_file.save(folder, project)
    result = render.render(folder, matte_runner=fake_matte())

    shown = media.probe(result.video)
    assert (shown["width"], shown["height"]) == (1440, 2560)
    with Image.open(result.cover) as cover:
        assert cover.size == (1440, 2560)
    assert result.notes == []  # between 1080 wide and 4K: not scaled, and nothing to say about it
