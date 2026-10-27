"""The render: a project folder in, the finished videos out.

render() plans the timeline, draws the card pictures, encodes each segment, joins the parts, adds
the sound (the clip's own, the click and the music) and writes the cover. It follows the quality of the clip (ruling Q1):

- The frame rate is the clip's own (media.output_fps), up to 60. Every frame count, here and in the
  timeline, is in frames of that rate, so no frame is made up and none is repeated.
- There is exactly one lossy encode. The parts are lossless intermediates, and the join encodes
  the final video once, at a bitrate of at least the clip's own and never under 10 Mbps.
- The size is the clip's own (config.output_size), in its own shape with nothing around it: at least
  1080 wide, and at most 3840 on the longer side. A clip that is smaller is scaled up, and one that is
  bigger is scaled down, and the person is told (output_notes).

The head and the tail are the clip itself, and the card is a still of its freeze frame. So only the
freeze frame and its cutout have to be at the size of the output: the render cuts them from the clip at
that size, with RVM, and draws the card on them (card.FullFreeze), while the layout of the notes stays
the plan's, which is what the studio showed. A preview is at the plan's size and needs none of that.

Everything is made in <project>/renders/_work, and the finished files are moved to <project>/renders
only when all of them are done, so a render that fails leaves the last good files alone. _work is
removed at the end, and when the render fails: its parts are lossless, and large.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from PIL import Image

from . import card, matte, media, sfx
from . import onsets as beats
from . import project as project_file
from .config import MAX_LONG_SIDE, PLAN_WIDTH, TIMING_RANGES, output_size, plan_size
from .fileio import retry_locked
from .pipeline import RERUN_BEFORE, person_area
from .timeline import Timeline, build_timeline, card_states, snap_to_onsets

_RENDERS = "renders"
_WORK = "_work"
_CLIP_KEYS = ("path", "stem", "width", "height", "fps", "duration")  # what a render reads of the clip


@dataclass(frozen=True)
class RenderResult:
    """What a render made.

    video is the finished video (renders/preview.mp4 for a preview) and with_music the one with the
    track added, or None. cover is the card as a still, or None for a preview, which makes only the
    video. frames is the length of the video in frames, and freeze_out and release_out say when the
    card starts and ends, in seconds. warning is a sentence for the person when the freeze could not
    land where they asked (the timeline's), otherwise None. notes are the other things the person
    should know about how the clip was made (see output_notes), one sentence each.
    """

    video: Path
    with_music: Path | None
    cover: Path | None
    frames: int
    freeze_out: float
    release_out: float
    warning: str | None
    notes: list[str] = field(default_factory=list)


def timeline_for(project: dict, hits: list[float] | None = None) -> Timeline:
    """The timeline a render of this project follows, counted in frames of the clip's own rate.

    The clip's length, the freeze frame as a time (freeze.frame divided by the rate) and the timing
    go to build_timeline. With a music track, "freeze at" and "hold" first snap to the track's
    nearest hits, and are then put back inside their allowed ranges, since a hit close to the
    freeze can make the hold shorter than allowed. hits are the track's hits in seconds, by default
    the project's own audio.onsets (see render, which finds them when there are none).
    """
    clip, timing, audio = project["clip"], dict(project["timing"]), project["audio"]
    fps = float(media.output_fps(clip))
    if hits is None:
        hits = audio.get("onsets") or []
    if audio.get("track"):
        freeze_at, hold = snap_to_onsets(timing["freeze_at"], timing["hold"], hits)
        timing["freeze_at"] = _clamp(freeze_at, TIMING_RANGES["freeze_at"])
        timing["hold"] = _clamp(hold, TIMING_RANGES["hold"])
    return build_timeline(clip["duration"], project["freeze"]["frame"] / fps, timing, fps)


def output_notes(clip: dict, rate: Fraction) -> list[str]:
    """What the person should know about how the clip was made, one sentence each, in the order
    they apply. clip is the project's clip and rate the rate media.output_fps picked for it.

    - The clip is under 1080 px wide: it is scaled up to 1080 wide, so that the notes stay sharp.
    - The clip is over 3840 px on its longer side (4K): it is scaled down to that.
    - The clip is faster than 60 fps: it is rendered at 60.
    """
    notes = []
    width, height = output_size(clip["width"], clip["height"])
    shown = f"your clip is {clip['width']}x{clip['height']}"
    if max(clip["width"], clip["height"]) > MAX_LONG_SIDE:
        notes.append(f"Rendered at {width}x{height}; {shown}, bigger than 4K.")
    elif clip["width"] < PLAN_WIDTH:
        notes.append(f"Scaled up to {width}x{height} so the notes stay sharp; {shown}.")
    if float(clip["fps"]) > float(rate) * (1 + media.RATE_TOLERANCE):
        notes.append(
            f"Your clip runs at {clip['fps']:g} fps; rendered at {float(rate):g} fps, "
            "the highest this skill makes."
        )
    return notes


def render(
    project_dir: Path,
    preview: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
    matte_runner: Callable | None = None,
) -> RenderResult:
    """Render a project folder and return what was made.

    The finished video is renders/<clip>-main-character.mp4. Its sound is the clip's own (when
    audio.original is on, and the clip has sound), silent over the card and back in step with the
    picture when the video lets go, with the shutter click on the freeze when timing.shutter is on.
    A track in audio.track also gives renders/<clip>-with-music.mp4 (the track and the click, without
    the clip's sound), and the card as a still gives renders/<clip>-cover.png.
    The freeze and the release snap to the track's hits: audio.onsets, or, when the project has
    none, the hits found in the track now. status.rendered_at, status.outputs (the files, as
    paths inside the project folder) and the moments the card starts and ends, status.freeze_out
    and status.release_out (seconds), are saved to project.json, which is read again just before,
    so a change made while the render ran is kept.

    A preview is at the plan's size and fast: only renders/preview.mp4, with the same sound as the
    finished video (the clip's own, and the click) but no music, no cover and no change to the
    project. A final render is at the size of the clip (config.output_size), and
    when that is not the plan's it cuts the freeze frame out of the clip again at that size, with
    matte_runner (matte.run_rvm unless a test gives another), at the detail the project says: that
    needs the matte model, and ModelMissing says so before the clip is cut.

    progress, if given, is called as progress(stage, done, total) after each piece of work: the
    stages are "cutout" (when the freeze frame is cut at the output's size), "cards", "parts",
    "join", "sound" and "cover", with the pieces of the stage done and in all. Raises
    FileNotFoundError when the clip or the track is missing, ValueError for a project that is not
    valid or a freeze frame past the end of the clip, and MediaError when the joined video does not
    have the frames the plan has (a click placed by the plan would be off). A render with no track
    deletes the with-music file that an earlier render left, which is out of date, once the new
    files are in place (a render that fails keeps it, as status.outputs still names it).

    On Windows a file that a video player or a virus scan has open cannot be replaced for a moment.
    Publishing the files, clearing _work and saving project.json wait that out (fileio.retry_locked,
    about a second), and a file that stays locked is a PermissionError that names it. renders/_work
    is cleared whether the render worked or failed (a render that was killed, or whose folder a
    program has open, leaves it, and the next render removes it). A render owns that folder, so two
    renders of one project must not run at the same time.
    """
    project_dir = Path(project_dir).resolve()
    project = project_file.load(project_dir)
    project_file.validate(project)
    source, track = _inputs(project_dir, project)
    clip, timing = project["clip"], project["timing"]
    rate = media.output_fps(clip)
    timeline = timeline_for(project, _hits(project, track))
    renders = project_dir / _RENDERS
    work = renders / _WORK
    _remove_tree(work)  # what a failed run left behind, if nothing has a file of it open
    work.mkdir(parents=True, exist_ok=True)  # and if something does, it is used as it is
    try:
        def report(stage: str, done: int, total: int) -> None:
            if progress is not None:
                progress(stage, done, total)

        plan = (project["work"]["width"], project["work"]["height"])
        size = plan if preview else output_size(clip["width"], clip["height"])
        full = None
        if size != plan:
            full = _cut_freeze(project_dir, project, source, work, size, rate, matte_runner, report)
        parts = _encode_parts(project_dir, project, timeline, source, work, rate, preview, size, full, report)
        joined = work / "joined.mp4"
        bitrate = None if preview else media.target_bitrate(clip.get("bit_rate"))
        card_at = round(timeline.freeze_out * float(rate))
        still = (card_at, round(timeline.release_out * float(rate)) - 1)
        media.concat(parts, joined, fps=rate, bitrate=bitrate, still=still)
        _check_join(joined, timeline.total_frames)
        report("join", 1, 1)

        def result(video: Path, with_music: Path | None, cover: Path | None) -> RenderResult:
            return RenderResult(
                video, with_music, cover, timeline.total_frames, timeline.freeze_out,
                timeline.release_out, timeline.warning, output_notes(clip, rate),
            )

        # The clip's own sound, laid out on the timeline (silent over the card), unless the person muted it
        # or the clip has none. The shutter click goes over it.
        voice = None
        if project["audio"].get("original", True):
            voice = media.clip_sound(source, timeline.segments, rate, work / "clip-sound.wav")
        click = sfx.shutter_click(work / "click.wav") if timing["shutter"] else None

        if preview:
            video = renders / "preview.mp4"
            if voice or click:
                heard = work / "preview.mp4"
                _add_sound(joined, heard, None, click, None, timeline.freeze_out, voice)
                _publish({heard: video})
            else:
                _publish({joined: video})
            return result(video, None, None)

        stem = clip["stem"]
        video, with_music, cover = (
            work / f"{stem}-main-character.mp4",
            work / f"{stem}-with-music.mp4",
            work / f"{stem}-cover.png",
        )
        _add_sound(joined, video, with_music, click, track, timeline.freeze_out, voice)
        if click or track or voice:
            report("sound", 1, 1)
        card.render_card(project_dir, project, size=size, source=full).save(cover)
        report("cover", 1, 1)

        finished = {video: renders / video.name, cover: renders / cover.name}
        if track:
            finished[with_music] = renders / with_music.name
        _publish(finished)
        if not track:  # the with-music file of an earlier render is out of date: it goes once the new files are in
            _remove_file(renders / with_music.name)
        _record(
            project_dir, [finished[path] for path in (video, with_music, cover) if path in finished],
            timeline.freeze_out, timeline.release_out,
        )
        return result(finished[video], finished.get(with_music), finished[cover])
    finally:
        _remove_tree(work)  # the parts of a render are large: not left behind by one that failed


def _inputs(project_dir: Path, project: dict) -> tuple[Path, Path | None]:
    """The clip and the music track (None when there is none) of a project, after checking that a
    render can start: the clip is described, the frames were cut at the size its shape needs, the
    clip and the track are files, and the freeze frame is one the clip has and its picture and
    cutout are in the folder. A problem is raised as an error with a sentence for the person, before
    anything is made."""
    clip, frame = project["clip"], project["freeze"]["frame"]
    missing = [key for key in _CLIP_KEYS if key not in clip]
    if missing:
        raise ValueError(f"The project's clip has no {', '.join(missing)}. Run init on the clip again.")
    plan, work = plan_size(clip["width"], clip["height"]), project["work"]
    if (work["width"], work["height"]) != plan:
        raise ValueError(
            f"The frames of this project were cut at {work['width']}x{work['height']}, and a clip of "
            f"{clip['width']}x{clip['height']} needs {plan[0]}x{plan[1]}. Run init on the clip again."
        )
    source = _file_in(project_dir, clip["path"], "clip")
    track = _file_in(project_dir, project["audio"].get("track"), "music track")  # (a null in a patch deletes it)
    if clip.get("frames") is not None and frame >= clip["frames"]:
        raise ValueError(
            f"The freeze frame is {frame}, but the clip has {clip['frames']} frames "
            f"(the last is {clip['frames'] - 1})."
        )
    for picture in (project_dir / "frames" / f"{frame:06d}.jpg", matte.matte_path(project_dir, frame)):
        if not picture.is_file():
            raise FileNotFoundError(f"{picture} is missing. Run init on the clip again.")
    return source, track


def _cut_freeze(
    project_dir: Path,
    project: dict,
    source: Path,
    work: Path,
    size: tuple[int, int],
    rate: Fraction,
    matte_runner: Callable | None,
    report: Callable[[str, int, int], None],
) -> card.FullFreeze:
    """The freeze frame and its cutout, cut from the clip at `size`, the output's, in work/freeze.

    The frame and the frames before it that the matte warms up on (as many as refine_matte cuts again,
    pipeline.RERUN_BEFORE) are cut from the clip at that size, and the matte is run on them at the
    ratio matte.downsample_for gives at that size, for the project's detail and for the share of the
    picture the person fills in the freeze frame's own matte. The project's pictures at the plan are
    only what the layout was made from: the card is drawn on these.
    """
    frame = project["freeze"]["frame"]
    first = max(frame - RERUN_BEFORE, 0)
    frames, mattes = work / "freeze" / "frames", work / "freeze" / "matte"
    runner = matte_runner
    if runner is None:
        matte.ensure_model()  # no use cutting frames the cutout cannot be made of
        runner = matte.run_rvm
    media.extract_frames(source, frames, size, rate, first=first, last=frame)
    try:
        area = person_area(project_dir / "matte", frame)
    except OSError:  # the freeze frame has no cutout to measure
        area = None
    ratio = matte.downsample_for(size[0], size[1], project["matte"].get("detail", "auto"), area)
    shown = frame - first  # which of the frames cut is the freeze frame: they are numbered from 0
    runner(
        frames, mattes, ratio, start=shown, end=shown + 1,
        progress=lambda done, total: report("cutout", done, total),
    )
    return card.FullFreeze(frames / f"{shown:06d}.jpg", mattes / f"{shown:06d}.png")


def _first_frame(
    project_dir: Path, source: Path, work: Path, size: tuple[int, int], rate: Fraction, out: Path
) -> None:
    """The clip's first frame as a picture of the output's size, saved as a PNG at `out` (to be held
    as a card is). It is the project's own frame 0 when that is already the size, and else it is cut
    from the clip at that size, as the freeze frame is."""
    frame = project_dir / "frames" / "000000.jpg"
    with Image.open(frame) as plan_frame:
        if plan_frame.size != tuple(size):
            media.extract_frames(source, work / "first", size, rate, first=0, last=0)
            frame = work / "first" / "000000.jpg"
    with Image.open(frame) as image:
        image.convert("RGB").save(out, compress_level=1)


def _encode_parts(
    project_dir: Path,
    project: dict,
    timeline: Timeline,
    source: Path,
    work: Path,
    rate: Fraction,
    preview: bool,
    size: tuple[int, int],
    full: card.FullFreeze | None,
    report: Callable[[str, int, int], None],
) -> list[Path]:
    """Draw the pictures and encode each segment of the timeline as a video part in work, in order,
    every part at `size`.

    The pictures are the card in each of its states (one picture for each distinct look, however many
    frames it lasts, drawn on `full`, the freeze frame cut at that size, when there is one) and, for a
    head that has to hold the first frame, that frame. The head and the tail are the clip. Each
    segment is encoded by what it is, never by where it stands: a freeze on frame 0 has the card
    first, and a head that is only the held frame has no clip. The parts of a final render are
    lossless, since the join is its one lossy encode.
    """
    clip, timing = project["clip"], project["timing"]

    (hold,) = [segment.frames for segment in timeline.segments if segment.kind == "card"]
    states = card_states(hold, timing["reveal"], timing["boil"], timing["shutter"], float(rate))
    looks = list(dict.fromkeys((state.visible, state.variant, state.flash) for state in states))
    pictures = {}
    for number, look in enumerate(looks):
        pictures[look] = work / f"card-{number:02d}.png"
        card.render_card(project_dir, project, *look, size=size, source=full).save(
            pictures[look], compress_level=1
        )
        report("cards", number + 1, len(looks))
    fill = work / "fill.png"
    if any(segment.kind == "fill" for segment in timeline.segments):
        _first_frame(project_dir, source, work, size, rate, fill)

    src_size = (clip["width"], clip["height"])
    lossless = not preview
    parts = []
    for number, segment in enumerate(timeline.segments):
        part = work / f"part-{number:02d}.mp4"
        if segment.kind == "clip":
            media.encode_clip(
                source, part, segment, src_size, size, preview=preview, fps=rate, lossless=lossless,
            )
        else:
            if segment.kind == "fill":
                stills = [(fill, segment.frames)]
            else:
                stills = [
                    (pictures[(state.visible, state.variant, state.flash)], state.frames)
                    for state in states
                ]
            media.encode_stills(stills, part, preview=preview, fps=rate, lossless=lossless)
        parts.append(part)
        report("parts", number + 1, len(timeline.segments))
    return parts


def _add_sound(
    joined: Path,
    video: Path,
    with_music: Path | None,
    click: Path | None,
    track: Path | None,
    at: float,
    voice: Path | None = None,
) -> None:
    """Make the finished video from the silent joined one. Its sound is `voice`, the clip's own sound
    (media.clip_sound), with the click over it at `at` seconds when there is one; with no voice, the
    click alone, and with neither the picture as it is. With a track, also with_music: the track (and
    the click, over it) added to the picture, without the clip's sound. The picture is copied and not
    encoded again."""
    if track:
        if click:
            media.mix_click(joined, click, at, with_music, music=track)
        else:
            media.mux_audio(joined, track, with_music)
    if voice:
        # (the voice has faded out at its end already, so it is not faded again)
        if click:
            media.mix_click(joined, click, at, video, music=voice, fade_out=0)
        else:
            media.mux_audio(joined, voice, video, fade_out=0)
    elif click:
        media.mix_click(joined, click, at, video)
    else:
        shutil.copyfile(joined, video)


def _record(project_dir: Path, outputs: list[Path], freeze_out: float, release_out: float) -> None:
    """Save when the render finished and which files it made to project.json: status.rendered_at, an
    ISO time in UTC, status.outputs, the files as paths inside the project folder, and the moments
    of the video where the card starts and ends, in seconds, as status.freeze_out and
    status.release_out (what the render's result says), so that whoever checks the video knows where
    to look. The project is read again first, so that nothing changed while the render ran is lost,
    and it is read, changed and saved under project.UPDATE_LOCK, so that an edit the studio saves in
    the same moment is not overwritten by this, nor this by the edit."""
    with project_file.UPDATE_LOCK:
        latest = project_file.load(project_dir)
        latest["status"]["rendered_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        latest["status"]["outputs"] = [path.relative_to(project_dir).as_posix() for path in outputs]
        latest["status"]["freeze_out"] = freeze_out
        latest["status"]["release_out"] = release_out
        project_file.save(project_dir, latest)


def _hits(project: dict, track: Path | None) -> list[float]:
    """The hits of the music track, in seconds: the project's audio.onsets, or, when it has none
    (the track was put in by hand, or has no clear hits), the ones found in the track itself. No
    track has none. A track that opens with silence reports one hit at its very start, which is
    nowhere near the freeze or the release, and so is left alone."""
    if track is None:
        return []
    stored = project["audio"].get("onsets")
    return list(stored) if stored else beats.detect_onsets(beats.load_audio_mono(track))


def _clamp(value: float, limits: tuple[float, float]) -> float:
    low, high = limits
    return min(max(value, low), high)


def _file_in(project_dir: Path, name: str | None, what: str) -> Path | None:
    """The file a project names (the clip, the track): the path as it is when it is absolute,
    otherwise inside the project folder. None when there is no name. FileNotFoundError, saying
    what is missing, when there is no such file."""
    if not name:
        return None
    path = Path(name)
    if not path.is_absolute():
        path = project_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"The {what} {path} is missing.")
    return path


def _check_join(joined: Path, planned: int) -> None:
    """Raise MediaError unless the joined video has the frames the plan has. The freeze, the card
    and the click are placed by the plan, so a video that came out a frame short or long would
    have them in the wrong place. Nothing has been kept yet when this is raised."""
    got = media.count_frames(joined)
    if got != planned:
        raise media.MediaError(
            f"The video came out with {got} frames, not the {planned} planned, so the freeze, the "
            "card and the click would not line up. Nothing was saved: render again."
        )


def _publish(finished: dict[Path, Path]) -> None:
    """Move each finished file to its place in renders/, replacing the one from an earlier render.
    A file that a player has open is waited for (see fileio.retry_locked); if one stays locked, the
    error names it."""
    for source, dest in finished.items():
        retry_locked(lambda source=source, dest=dest: os.replace(source, dest), dest)


def _remove_file(path: Path) -> None:
    """Delete a file if it is there, waiting out a lock on it."""
    retry_locked(lambda: path.unlink(missing_ok=True), path)


def _remove_tree(folder: Path) -> None:
    """Delete a folder and what is in it. On Windows a file ffmpeg wrote can stay locked for a
    moment after it exits (a virus scan, the indexer), and a folder whose last file is still being
    let go of is "not empty", so any refusal is waited out (see fileio.retry_locked). If the folder
    still will not go, what can be deleted is, and the rest is left: it is only a temporary folder,
    and the next render clears it."""

    def remove() -> None:
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass  # already gone

    try:
        retry_locked(remove, folder, errors=OSError)
    except PermissionError:
        shutil.rmtree(folder, ignore_errors=True)
