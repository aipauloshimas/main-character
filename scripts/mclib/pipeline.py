"""Starting a project: init_project turns one clip into a project folder that the studio and the
render can work from.

The folder gets the clip's frames (frames/000000.jpg, ...), the cutout of every frame (matte/
000000.png, ...) and project.json. Cutting out the person takes the longest, so the frames and the
mattes are kept, and a second init of the same clip only looks at them again.
"""

from __future__ import annotations

import shutil
import statistics
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from PIL import Image

from . import matte, media
from . import project as project_file
from .analyze import person_boxes, suggest_freeze
from .config import CLIP_RANGE, layout_name, plan_size
from .raster import from_image
from .styles.recipes import BUILTINS

NO_PERSON_AREA = 0.02  # a clip whose person fills less of the frame than this, in most frames, has none
_NO_PERSON = (
    "No person found in most frames. Check that the clip shows a person, and that they are "
    "not too far from the camera."
)
POSITIONS_RESET = "Your note positions were reset because the clip's shape changed."
TWO_PEOPLE = "Two people are in the frame; the notes avoid only one face."
SECOND_PERSON = 0.15  # a piece of the cutout is a person when it is over this share of the person's area
DETAILS = ("auto", "high", "max")  # how closely the person is cut out (matte.detail)
BAD_DETAIL = 'The detail has to be "auto", "high" or "max".'
RERUN_BEFORE = 12  # frames before the freeze that refine_matte cuts out again, and 1 after it
_FRAMES = "frames"
_MATTE = "matte"


def init_project(
    clip: Path,
    project_dir: Path | None = None,
    matte_runner: Callable | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[Path, list[str]]:
    """Make, or update, the project of a clip. Returns the project folder and the warnings, which
    are sentences for the person.

    The clip has to be between config.CLIP_RANGE seconds long, or ValueError says so. The folder is
    project_dir, or ./main-character/<clip name> in the current folder.

    The frames are cut at the plan (config.plan_size: 1080 wide, in the clip's own shape, which is
    1080x608 for 16:9 and 1080x1920 for 9:16) and at the clip's own frame rate (media.output_fps), and
    the mattes are made by matte_runner, which is matte.run_rvm unless a test gives another. The
    runner is first run over the first second of frames, to measure how much of the picture the
    person fills, and that sets the detail of the cutout for the rest (matte.downsample_for): a
    person who fills the frame gets a closer look. Then the best frames to freeze on are suggested
    and the first is the freeze, and a warning says so when most frames have nobody in them.

    A folder that already has a project for this same clip (its path, size, modified time and
    frame rate), cut at this size, keeps its frames and mattes and they are not made again.
    Whatever else it has changed, the copy, the three versions, the style, the positions, the
    timing, the audio and the matte settings, is kept too, and so is its freeze frame, put back
    within the clip's frames when it is past the end. A different clip, or frames that were cut at
    another size, start the frames and the mattes again, and a different clip gets the suggested
    freeze. The positions are pinned in px of the plan, so when the plan is another one (the clip
    is another shape, or the project was cut at another size) they are cleared: no note is pinned.
    When a note was pinned, POSITIONS_RESET says so: it is one of the warnings, and it is left in the
    project as its `notice` too, which the studio's page shows as a toast and then takes away.

    progress, if given, is called as progress(stage, done, total): "frames" once the frames are
    cut, "matte" after each frame of the cutout, and "freeze" once the suggestion is made.
    """
    clip = Path(clip).resolve()
    if not clip.is_file():
        raise FileNotFoundError(f"There is no clip at {clip}.")
    info = media.probe(clip, count=False)
    low, high = CLIP_RANGE
    if not low <= info["duration"] <= high:
        raise ValueError(
            f"The clip is {info['duration']:.1f} seconds long. This skill works with clips "
            f"between {low:g} and {high:g} seconds: trim it and try again."
        )
    work = plan_size(info["width"], info["height"])
    layout = layout_name(info["width"], info["height"])
    rate = media.output_fps(info)
    project_dir = Path(project_dir) if project_dir else Path.cwd() / "main-character" / clip.stem
    project_dir = project_dir.resolve()

    stat = clip.stat()
    record = {
        "path": str(clip),
        "stem": clip.stem,
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "duration": info["duration"],
        "frames": 0,  # the count of the frames cut, set below
        "orientation": media.orientation(info["width"], info["height"]),
        "bit_rate": info["bit_rate"],
        "size": stat.st_size,
        # Nanoseconds since 1970, which is 19 digits: more than a JavaScript number holds exactly,
        # so it is text, and the studio's page can carry it through without changing a digit
        "mtime": str(stat.st_mtime_ns),
    }
    previous = _read_project(project_dir)
    same_clip = previous is not None and _same_clip(previous["clip"], record, rate)
    frames_dir, matte_dir = project_dir / _FRAMES, project_dir / _MATTE
    if same_clip and _cached(previous, work, frames_dir, matte_dir):
        record["frames"] = previous["clip"]["frames"]
    else:
        runner = matte_runner
        if runner is None:
            matte.ensure_model()  # no use cutting frames the cutout cannot be made of
            runner = matte.run_rvm
        for folder in (frames_dir, matte_dir):
            shutil.rmtree(folder, ignore_errors=True)
        record["frames"] = media.extract_frames(clip, frames_dir, work, rate)
        if progress:
            progress("frames", 1, 1)
        detail = previous["matte"].get("detail", "auto") if previous else "auto"
        _cut_out(runner, frames_dir, matte_dir, work, detail, record["frames"], rate, progress)

    count = record["frames"]
    picks = suggest_freeze(frames_dir, matte_dir, fps=rate)
    if progress:
        progress("freeze", 1, 1)
    warnings = []
    areas = [person_area(matte_dir, number) for number in range(count)]
    if statistics.median(areas) < NO_PERSON_AREA:
        warnings.append(_NO_PERSON)

    if previous is None:
        project = project_file.default_project(record, layout, work, BUILTINS["bedroom-pop"])
        project["freeze"]["frame"] = picks[0]
    else:
        project = previous
        replanned = project.get("work") != {"width": work[0], "height": work[1]}
        project["clip"], project["layout"] = record, layout
        project["work"] = {"width": work[0], "height": work[1]}
        if replanned:  # a note pinned on another plan is not where the person put it on this one
            pinned = any(point is not None for point in project["positions"].values())
            project["positions"] = project_file.empty_positions()
            if pinned:  # (with nothing pinned nothing was lost, so there is nothing to say)
                warnings.append(POSITIONS_RESET)
                project["notice"] = POSITIONS_RESET
        old_frame = project["freeze"].get("frame", 0)
        project["freeze"]["frame"] = min(max(old_frame, 0), count - 1) if same_clip else picks[0]
    project["freeze"]["suggested"] = picks
    # The notes are laid out around one face. Two people side by side are two big pieces of the cutout.
    freeze = matte.load_alpha(project_dir, project, project["freeze"]["frame"])
    if sum(share > SECOND_PERSON for share in matte.person_regions(freeze)) >= 2:
        warnings.append(TWO_PEOPLE)
    project_file.validate(project)
    project_file.save(project_dir, project)
    return project_dir, warnings


def refine_matte(
    project_dir: Path, project: dict, detail: str, matte_runner: Callable | None = None
) -> int:
    """Cut the person out again around the freeze frame at a finer detail, save it as matte.detail
    and return the number of frames cut. The studio's Cutout control and `mc.py refine` both do this.

    The frames from RERUN_BEFORE before the freeze frame to the one after it are cut by the same
    runner as the first cutout (matte_runner, which is matte.run_rvm unless a test gives another),
    at the ratio matte.downsample_for gives for the detail and for the share of the picture the
    person fills in the freeze frame's own matte (a frame with no matte yet is cut as if the share
    were not known). `project` is the project as it was read, for the freeze frame and the work size.
    matte.detail is saved, in the file as it is then, when the cutout is made, so a cutout that fails
    changes nothing.

    ValueError for a detail that is not one of DETAILS. matte.ModelMissing, a sentence about the
    missing model, when no runner is given and the model is not on this machine; that is raised
    before anything is touched.
    """
    if detail not in DETAILS:
        raise ValueError(BAD_DETAIL)
    runner = matte_runner
    if runner is None:
        matte.ensure_model()
        runner = matte.run_rvm
    project_dir = Path(project_dir)
    frame, work = project["freeze"]["frame"], project["work"]
    try:
        area = person_area(project_dir / _MATTE, frame)
    except OSError:  # the freeze frame has no cutout to measure
        area = None
    ratio = matte.downsample_for(work["width"], work["height"], detail, area)
    cut = runner(
        project_dir / _FRAMES, project_dir / _MATTE, ratio, start=max(frame - RERUN_BEFORE, 0), end=frame + 1
    )
    project_file.update(project_dir, mutate=lambda current: current["matte"].update(detail=detail))
    return cut


def _read_project(project_dir: Path) -> dict | None:
    """The project in a folder, or None when there is none. A project file that cannot be read is
    an error, not a reason to start over, since it holds the person's copy."""
    path = project_dir / project_file.PROJECT_FILE
    if not path.is_file():
        return None
    try:
        project = project_file.load(project_dir)
        project_file.validate(project)
    except ValueError as error:  # json.JSONDecodeError is a ValueError
        raise ValueError(
            f"{path} cannot be read ({error}). Fix it, or move it away to start a new project."
        ) from error
    return project


def _same_clip(old: dict, new: dict, rate: Fraction) -> bool:
    """Whether a project's clip is the same footage as the clip being initialized: the same path,
    size and modified time, read at the same frame rate. A rate that comes out differently, such
    as 24 fps and 23.976 fps, means the frames would be numbered differently. The modified time
    is text, and a project made before it was has it as a number, which counts as the same."""
    return (
        old.get("path") == new["path"]
        and old.get("size") == new["size"]
        and str(old.get("mtime")) == new["mtime"]
        and "fps" in old
        and media.output_fps(old) == rate
    )


def _cached(project: dict, work: tuple[int, int], frames_dir: Path, matte_dir: Path) -> bool:
    """Whether the frames and mattes the project says it has are on disk, whole, and cut to this
    work size: one of each for every frame, and the first of each the size of the work. The size
    is looked at because an init that stopped halfway may have cut them at another size."""
    count = project["clip"].get("frames")
    if project.get("work") != {"width": work[0], "height": work[1]} or not count:
        return False
    for folder, pattern in ((frames_dir, "*.jpg"), (matte_dir, "*.png")):
        names = sorted(folder.glob(pattern)) if folder.is_dir() else []
        if len(names) != count:
            return False
        with Image.open(names[0]) as first:
            if first.size != tuple(work):
                return False
    return True


def _cut_out(
    runner: Callable,
    frames_dir: Path,
    matte_dir: Path,
    work: tuple[int, int],
    detail: str,
    count: int,
    rate: Fraction,
    progress: Callable[[str, int, int], None] | None,
) -> None:
    """Make the matte of every frame with runner, at a detail that fits how big the person is.

    The first second of frames is cut out at the detail the settings give, and the person's area
    there (the median share of the frame they cover) sets the detail for the rest. When that is the
    same, the cutout goes on from where it stopped. When a person who fills the frame needs a closer
    look, the first second is made again at that detail, with all the others.
    """
    first = min(count, max(1, round(rate)))

    def report(offset: int) -> Callable[[int, int], None] | None:
        if progress is None:
            return None
        return lambda done, total: progress("matte", offset + done, count)

    ratio = matte.downsample_for(*work, detail)
    runner(frames_dir, matte_dir, ratio, start=0, end=first, progress=report(0))
    area = statistics.median(person_area(matte_dir, number) for number in range(first))
    wanted = matte.downsample_for(*work, detail, area)
    if wanted == ratio:
        if first < count:
            runner(frames_dir, matte_dir, ratio, start=first, end=None, progress=report(first))
    else:
        runner(frames_dir, matte_dir, wanted, start=0, end=None, progress=report(0))


def person_area(matte_dir: Path, number: int) -> float:
    """The share of the picture that frame number's matte calls person (alpha over 0.5)."""
    with Image.open(matte_dir / f"{number:06d}.png") as image:
        return person_boxes(from_image(image.convert("L")))["area"]
