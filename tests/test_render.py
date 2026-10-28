"""Tests for the render: the plan, the whole pipeline on the small fake project, the sound, and what
the render owes the quality of the clip (its frame rate, one lossy encode, its bitrate, its size, the notes).

The fake project (conftest) is a 4 s, 30 fps clip of 1920x1080 frozen on frame 60, so a default render
is 210 frames: 60 of clip (from its first frame to the freeze), 90 of card and 60 of tail (the rest of the clip). Its video is the clip's
own frame, 1920x1080, with nothing around it (a preview is at the plan's size, 1080x608). Several tests
look at one render of it, made once (rendered). The others change something first and render again,
usually with a short timeline (small) because a render is the slow part of a test.

The card of a final render is drawn on the freeze frame cut from the clip at the size of the video, and
its matte: a fake runner stands in for RVM (conftest's fake_matte). Pictures are read back the way a
viewer meets them: frame_rgb decodes a frame as a BT.709 player does, and a card is the same picture as
card.render_card gives from the same freeze (frozen), within the little that a lossy encode changes.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from conftest import BT709, color_tags, rms, x264_options
from PIL import Image

from mclib import card, fileio, matte, media, render
from mclib import project as project_file
from mclib.card import ALL_VISIBLE
from mclib.timeline import card_states

SOUND_RATE = 44100
OUT = (1920, 1080)  # what the video of the fake project is: the clip's own size
PLAN = (1080, 608)  # what a preview is, and what the frames of the project are cut at
NOTES_ON = {  # the card's visible layers by what a frame shows, to tell one state from another
    "bottom": frozenset({"bottom"}),
    "bottom and top": frozenset({"bottom", "top"}),
    "all": ALL_VISIBLE,
    "none": frozenset(),
}


# --- helpers --------------------------------------------------------------------------------


class Rendered(NamedTuple):
    """The fake project rendered once with its defaults: what render() returned, the project
    folder, the project as it was rendered, and every ffmpeg call the render made."""

    result: render.RenderResult
    dir: Path
    project: dict
    calls: list


def record_ffmpeg(monkeypatch_or_patch):
    """Make every ffmpeg call media.run runs also be written to the list that is returned. Takes
    a pytest MonkeyPatch (the fixture, or MonkeyPatch.context())."""
    calls = []
    real = media.run

    def run(args):
        calls.append(list(args))
        return real(args)

    monkeypatch_or_patch.setattr(media, "run", run)
    return calls


def ffprobe(path, entries, stream="v:0"):
    """Some entries of a file's streams from ffprobe as text, such as "r_frame_rate"."""
    shown = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
         f"stream={entries}", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(shown)["streams"]


def frame_rgb(video, number, size=OUT):
    """One frame of a video as a BT.709 limited-range player shows it: floats 0-1, height x width x 3."""
    raw = video.with_suffix(f".{number}.rgb")
    media.run(["-i", str(video), "-vf",
               f"select=eq(n\\,{number}),scale=in_color_matrix=bt709:in_range=tv:out_range=pc:"
               "flags=accurate_rnd+full_chroma_int,format=rgb24",
               "-fps_mode", "passthrough", "-frames:v", "1", "-f", "rawvideo", str(raw)])
    return np.fromfile(raw, dtype=np.uint8).reshape(size[1], size[0], 3) / 255.0


def motion(video, first, count, crop):
    """How much a video's picture changes from each of `count` frames, starting at frame `first`,
    to the next: the mean absolute difference (0-255) of the gray picture inside crop, which is
    (x, y, width, height). It is a list of count - 1 numbers. A frame that repeats the one before it
    comes out near 0 (a lossy encode changes a still picture only a little), and two different
    frames of a moving picture do not."""
    x, y, width, height = crop
    raw = video.with_suffix(f".{first}.gray")
    media.run(["-i", str(video), "-vf",
               f"select=between(n\\,{first}\\,{first + count - 1}),crop={width}:{height}:{x}:{y},"
               "scale=270:152:flags=area,format=gray",
               "-fps_mode", "passthrough", "-f", "rawvideo", str(raw)])
    frames = np.fromfile(raw, dtype=np.uint8).reshape(-1, 152, 270).astype(float)
    assert len(frames) == count
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2)).tolist()


def frozen(folder, project, tmp_path, fake_matte, size=OUT):
    """The freeze frame and its matte cut at `size`, as a final render cuts them (and with the same fake
    matte), for a card to be drawn from. A render makes them in renders/_work and clears it afterward."""
    clip = Path(project["clip"]["path"])
    rate = media.output_fps(project["clip"])
    return render._cut_freeze(
        folder, project, clip, tmp_path, size, rate, fake_matte(), lambda *step: None)


def card_rgb(folder, project, visible, variant=0, flash=0.0, size=None, source=None):
    """The card as render_card draws it: floats 0-1, height x width x 3."""
    drawn = card.render_card(folder, project, visible, variant, flash, size=size, source=source)
    return np.asarray(drawn, float) / 255


def distance(a, b):
    """How far apart two pictures are: the mean absolute difference, 0 to 1."""
    return float(np.abs(a - b).mean())


def closest(frame, candidates):
    """Which of the candidate pictures (name -> PIL image) a decoded frame is nearest to, with the
    gap to each. The candidates are resized to the size of the frame, for a preview's."""
    height, width = frame.shape[:2]
    gaps = {name: distance(frame, np.asarray(image.convert("RGB").resize((width, height)), float) / 255)
            for name, image in candidates.items()}
    return min(gaps, key=gaps.get), gaps


def mono_samples(path):
    """The sound of a file as mono floats at 44.1 kHz, decoded by ffmpeg."""
    raw = path.with_suffix(".pcm")
    media.run(["-i", str(path), "-vn", "-ac", "1", "-ar", str(SOUND_RATE), "-f", "s16le", str(raw)])
    return np.frombuffer(raw.read_bytes(), dtype="<i2") / 32768.0


def peak(samples, start, end):
    """The loudest sample between two times, in seconds, as a share of full scale."""
    return float(np.abs(samples[int(start * SOUND_RATE):int(end * SOUND_RATE)]).max())


def small(folder, project, **clip):
    """Shorten the project's timeline, so that a render of it is quick: the freeze on frame 15 (0.5 s
    into the clip), a card of 1 s and no tail, and no boil (one stage drawn, not three). That is 15
    frames of clip then 30 of card. Keyword arguments set keys of the project's clip. The project is
    saved and returned."""
    freeze_on(folder, project, 15)
    project["timing"].update(hold=1.0, tail=0.0, boil=False)
    project["clip"].update(clip)
    project_file.save(folder, project)
    return project


def freeze_on(folder, project, frame):
    """Put the freeze on a frame of the fake project, giving that frame the picture and the matte
    of frame 60, which are the only ones the project has. Saves the project."""
    shutil.copy(folder / "frames" / "000060.jpg", folder / "frames" / f"{frame:06d}.jpg")
    shutil.copy(folder / "matte" / "000060.png", folder / "matte" / f"{frame:06d}.png")
    project["freeze"]["frame"] = frame
    project_file.save(folder, project)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, project_maker, fake_matte):
    """The fake project, rendered once with its defaults."""
    folder, project = project_maker(tmp_path_factory.mktemp("rendered"))
    with pytest.MonkeyPatch.context() as patch:
        calls = record_ffmpeg(patch)
        result = render.render(folder, matte_runner=fake_matte())
    return Rendered(result, folder, project, calls)


@pytest.fixture(scope="module")
def small_final(tmp_path_factory, project_maker, fake_matte):
    """A short final render of a project whose clip is said to have a bitrate of 20 Mbps, and which has
    no shutter click. Its result, its ffmpeg calls and its progress calls are kept, as
    (Rendered, progress)."""
    folder, project = project_maker(tmp_path_factory.mktemp("small_final"))
    small(folder, project, bit_rate=20_000_000)
    progress = []
    with pytest.MonkeyPatch.context() as patch:
        calls = record_ffmpeg(patch)
        result = render.render(
            folder, progress=lambda *step: progress.append(step), matte_runner=fake_matte())
    return Rendered(result, folder, project, calls), progress


# --- the six tests of the task --------------------------------------------------------------


def test_render_frame_count_and_size(rendered):
    result, folder, project, _ = rendered

    assert media.count_frames(result.video) == 210  # 60 of clip + 90 of card + 60 of tail (the rest of the clip)
    info = media.probe(result.video)
    assert (info["width"], info["height"], info["fps"]) == (1920, 1080, 30.0)  # the clip's own frame
    assert result.frames == 210
    assert result.video == folder.resolve() / "renders" / "synthetic-main-character.mp4"
    assert result.freeze_out == pytest.approx(60 / 30)
    assert result.release_out == pytest.approx(150 / 30)
    assert result.notes == []

    # H.264 in yuv420p, with the clip's own sound (see the tests of the sound), tagged BT.709 limited range,
    # and nothing left in _work
    assert [(s["codec_name"], s["pix_fmt"]) for s in ffprobe(result.video, "codec_name,pix_fmt")] == [
        ("h264", "yuv420p")]
    assert [s["codec_name"] for s in ffprobe(result.video, "codec_name", stream="a")] == ["aac"]
    assert color_tags(result.video) == {"stream": BT709, "frame": BT709}
    assert not (folder / "renders" / "_work").exists()


def test_render_card_segment_matches_card_png(rendered, tmp_path, fake_matte):
    result, folder, project, _ = rendered
    # the card of the render was drawn on the freeze frame cut at the size of the video
    source = frozen(folder, project, tmp_path, fake_matte)
    states = card_states(90, "staggered", True)
    # CardState.start counts from the card's first frame, so the video's frame is 60 more
    first_card_frame = round(result.freeze_out * 30)
    assert first_card_frame == 60

    # Card frame 10 is output frame 70: every note is on, and the boil is on its second picture
    state = next(s for s in states if s.start <= 10 < s.start + s.frames)
    assert (state.visible, state.variant) == (ALL_VISIBLE, 1)
    expected = card_rgb(folder, project, state.visible, state.variant, state.flash, OUT, source)
    assert distance(frame_rgb(result.video, 70), expected) < 0.02

    # The notes come in by the stagger and go out before the card does: at card frames 1, 3, 10
    # and 86 the video is closest to the card with those notes on, out of the four that exist
    for card_frame, shown in ((1, "bottom"), (3, "bottom and top"), (10, "all"), (86, "none")):
        variant = (card_frame // 7) % 3
        frame = frame_rgb(result.video, first_card_frame + card_frame)
        gaps = {name: distance(frame, card_rgb(folder, project, visible, variant, 0.0, OUT, source))
                for name, visible in NOTES_ON.items()}
        assert min(gaps, key=gaps.get) == shown, (card_frame, gaps)
        assert gaps[shown] < 0.02, (card_frame, gaps)


def test_render_preview_is_at_the_plans_size(fake_project):
    folder, project = fake_project
    before = project_file.load(folder)

    result = render.render(folder, preview=True)

    # the plan's size, 1080x608, made fast (and no cutout at the video's size: it needs no model)
    info = media.probe(result.video)
    assert (info["width"], info["height"], info["fps"]) == (1080, 608, 30.0)
    assert info["frames"] == result.frames == 210
    assert result.video == folder.resolve() / "renders" / "preview.mp4"
    assert result.cover is None
    # Nothing but the preview (and the render lock's file) is written, and the project is not touched
    assert sorted(path.name for path in (folder / "renders").iterdir()) == [".render-lock", "preview.mp4"]
    assert project_file.load(folder) == before


def test_render_writes_cover_and_status(rendered, tmp_path, fake_matte):
    result, folder, project, _ = rendered

    # the cover is the card at the size of the video, on the freeze frame cut at that size
    assert result.cover == folder.resolve() / "renders" / "synthetic-cover.png"
    expected = card.render_card(
        folder, project, ALL_VISIBLE, size=OUT, source=frozen(folder, project, tmp_path, fake_matte))
    with Image.open(result.cover) as cover:
        assert cover.size == OUT
        assert np.array_equal(np.asarray(cover.convert("RGB")), np.asarray(expected))

    status = project_file.load(folder)["status"]
    assert status["outputs"] == [
        "renders/synthetic-main-character.mp4", "renders/synthetic-cover.png"]
    assert all((folder / path).is_file() for path in status["outputs"])
    # It records its own moments too, in seconds, so that whoever checks the video knows where to look
    assert status["freeze_out"] == result.freeze_out == pytest.approx(60 / 30)
    assert status["release_out"] == result.release_out == pytest.approx(150 / 30)
    when = datetime.fromisoformat(status["rendered_at"])
    assert when.utcoffset() == timedelta(0)  # UTC, and it says so
    assert abs(datetime.now(timezone.utc) - when) < timedelta(minutes=30)
    assert status["done_at"] is None  # that is the studio's, not the render's


def test_render_with_shutter(fake_project, fake_matte):
    folder, project = fake_project
    project["timing"]["shutter"] = True
    project["audio"]["original"] = False  # (the clip's own sound is tested apart: here the video has the click alone)
    project_file.save(folder, project)

    result = render.render(folder, matte_runner=fake_matte())

    assert result.frames == 210
    assert project_file.load(folder)["status"]["outputs"] == [
        "renders/synthetic-main-character.mp4", "renders/synthetic-cover.png"]

    # The video has a sound: the click, 120 ms of it on the freeze (frame 60, 2.0 s)
    (audio,) = ffprobe(result.video, "codec_name", stream="a")
    assert audio["codec_name"] == "aac"
    clicked = mono_samples(result.video)
    assert peak(clicked, 1.99, 2.12) > 0.5
    assert peak(clicked, 0.0, 1.9) < 0.01 and peak(clicked, 2.3, 6.9) < 0.01

    # The two-frame white flash: card frames 0 and 1 (output 60 and 61) are the card blended with white, at 0.85
    # and 0.40, and then it is gone (frame 62 is the card as it is, and 59 is the clip's last frame)
    means = [frame_rgb(result.video, number).mean() for number in (59, 60, 61, 62)]
    plain = means[3]
    assert means[1] == pytest.approx(0.85 + 0.15 * plain, abs=0.03), means
    assert means[2] == pytest.approx(0.40 + 0.60 * plain, abs=0.03), means
    assert means[0] < means[2] - 0.15, means


# --- the clip's own sound -------------------------------------------------------------------


def audio_duration(path):
    """The length of a file's sound in seconds, as its audio stream states it."""
    (stream,) = ffprobe(path, "duration", stream="a")
    return float(stream["duration"])


def test_the_video_keeps_the_clips_own_sound_and_the_card_is_silent(rendered):
    result, *_ = rendered

    # one sound as long as the video (210 frames, 7.0 s), within a frame
    assert audio_duration(result.video) == pytest.approx(210 / 30, abs=1 / 30)
    sound = mono_samples(result.video)
    tone = 0.125 / 2 ** 0.5  # the clip's sine is at 1/8 of full scale, so this is its rms
    # the head (the clip from 0 to 2.0 s) and the tail (the clip from 2.0 s on, after the card) are the
    # clip's sound, and the card (2.0 s to 5.0 s) is silent
    assert rms(sound, 0.3, 1.9) == pytest.approx(tone, rel=0.2)
    assert rms(sound, 2.2, 4.9) < 0.001
    assert rms(sound, 5.2, 6.7) == pytest.approx(tone, rel=0.2)
    assert rms(sound, 6.85, 6.98) < 0.5 * tone  # and it fades out with the video


def test_the_video_has_no_sound_when_the_original_is_off(fake_project, fake_matte):
    folder, project = fake_project
    small(folder, project)
    project["audio"]["original"] = False  # and no shutter click: nothing to hear
    project["timing"]["shutter"] = False
    project_file.save(folder, project)

    result = render.render(folder, matte_runner=fake_matte())

    assert ffprobe(result.video, "codec_type", stream="a") == []
    assert media.count_frames(result.video) == 45


def test_the_shutter_click_is_heard_over_the_clips_sound(fake_project, fake_matte):
    folder, project = fake_project
    small(folder, project)
    project["timing"]["shutter"] = True
    project_file.save(folder, project)

    result = render.render(folder, matte_runner=fake_matte())

    sound = mono_samples(result.video)
    assert audio_duration(result.video) == pytest.approx(45 / 30, abs=1 / 30)
    assert rms(sound, 0.1, 0.45) > 0.05  # the clip's sound in the head
    assert peak(sound, 0.5, 0.62) > 0.5  # the click, on the freeze (15 frames in, 0.5 s)
    assert rms(sound, 0.8, 1.4) < 0.001  # and the rest of the card is silent

    # With the original off, the click is all there is
    project["audio"]["original"] = False
    project_file.save(folder, project)
    only = render.render(folder, matte_runner=fake_matte())
    alone = mono_samples(only.video)
    assert peak(alone, 0.5, 0.62) > 0.5 and rms(alone, 0.1, 0.45) < 0.001


def test_a_preview_has_the_clips_sound_too(fake_project):
    folder, project = fake_project
    small(folder, project)

    result = render.render(folder, preview=True)

    assert audio_duration(result.video) == pytest.approx(45 / 30, abs=1 / 30)
    sound = mono_samples(result.video)
    assert rms(sound, 0.1, 0.45) > 0.05 and rms(sound, 0.8, 1.4) < 0.001

    # and with the original off it is silent again, as a preview was before the clip kept its sound
    project["audio"]["original"] = False
    project_file.save(folder, project)
    assert ffprobe(render.render(folder, preview=True).video, "codec_type", stream="a") == []


def test_a_clip_without_sound_renders_as_it_did(tmp_path, project_maker, synthetic_clip, fake_matte):
    silent = tmp_path / "silent.mp4"
    media.run(["-i", str(synthetic_clip), "-an", "-c:v", "copy", str(silent)])
    folder, project = project_maker(tmp_path, clip=silent)
    small(folder, project)

    result = render.render(folder, matte_runner=fake_matte())
    preview = render.render(folder, preview=True)

    assert ffprobe(result.video, "codec_type", stream="a") == []
    assert ffprobe(preview.video, "codec_type", stream="a") == []
    assert media.count_frames(result.video) == 45

    # with a shutter click it has the click alone, as before
    project["timing"]["shutter"] = True
    project_file.save(folder, project)
    clicked = mono_samples(render.render(folder, matte_runner=fake_matte()).video)
    assert peak(clicked, 0.5, 0.62) > 0.5 and rms(clicked, 0.1, 0.45) < 0.001


def test_the_head_at_a_speed_has_sound_as_long_as_its_picture(fake_project, fake_matte):
    folder, project = fake_project
    small(folder, project)
    project["timing"]["head_speed"] = 1.43
    project_file.save(folder, project)

    result = render.render(folder, matte_runner=fake_matte())

    frames = media.count_frames(result.video)
    assert audio_duration(result.video) == pytest.approx(frames / 30, abs=1 / 30)
    sound = mono_samples(result.video)
    assert rms(sound, 0.1, result.freeze_out - 0.05) > 0.05  # the head's sound, sped up
    assert rms(sound, result.freeze_out + 0.1, frames / 30 - 0.05) < 0.001  # and the card is silent


def test_a_freeze_early_in_the_clip_just_lands_early(fake_project, fake_matte):
    folder, project = fake_project
    freeze_on(folder, project, 20)  # only 20 frames of clip before the freeze: the head is those 20
    project["timing"].update(hold=1.0, tail=0.0)  # and a short card, to make the render quick
    project_file.save(folder, project)

    result = render.render(folder, matte_runner=fake_matte())

    assert result.freeze_out == pytest.approx(20 / 30)
    assert result.frames == media.count_frames(result.video) == 20 + 30
    assert not hasattr(result, "warning") and not hasattr(result, "with_music")


def test_an_older_project_with_music_and_trend_timing_renders_without_them(fake_project):
    # A project.json from before the skill dropped the music has audio.track, audio.onsets and timing.freeze_at.
    # It loads without them, and the render is the clip's own sound and the freeze frame, nothing else.
    folder, project = fake_project
    small(folder, project)
    older = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    older["audio"].update(track="track.wav", onsets=[0.5, 1.5])
    older["timing"]["freeze_at"] = 2.07
    (folder / "project.json").write_text(json.dumps(older), encoding="utf-8")

    result = render.render(folder, preview=True)

    assert result.freeze_out == pytest.approx(15 / 30)  # the freeze lands at the freeze frame, not at 2.07 s
    assert result.frames == media.count_frames(result.video) == 15 + 30


# --- the plan -------------------------------------------------------------------------------


def test_timeline_for_follows_the_clips_rate(fake_project):
    folder, project = fake_project
    plan = render.timeline_for(project)
    assert [(s.kind, s.frames) for s in plan.segments] == [("clip", 60), ("card", 90), ("clip", 60)]
    assert plan.total_frames == 210

    # The same project on a 24 fps clip: the freeze frame is a time of 2.0 s, everything is counted
    # in 24ths of a second, and the tail is the 2.0 s of clip left
    project["clip"]["fps"] = 24.0
    project["freeze"]["frame"] = 48
    plan = render.timeline_for(project)
    assert [(s.kind, s.frames) for s in plan.segments] == [("clip", 48), ("card", 72), ("clip", 48)]
    assert plan.freeze_out == pytest.approx(48 / 24)

    # and at 23.976, which is 24000/1001 and not 24: the freeze frame is 2.002 s into the clip
    project["clip"]["fps"] = float(Fraction(24000, 1001))
    plan = render.timeline_for(project)
    assert [segment.frames for segment in plan.segments] == [48, 72, 48]
    assert plan.freeze_out == pytest.approx(48 * 1001 / 24000)


# --- the quality of the clip (ruling Q1) ----------------------------------------------------


@pytest.mark.slow
def test_render_follows_a_24_fps_clip(tmp_path, project_maker, synthetic_clip_24, fake_frames_24, fake_matte):
    (tmp_path / "project").mkdir()
    folder, project = project_maker(
        tmp_path / "project", clip=synthetic_clip_24, frames=fake_frames_24, fps=24.0, count=96, freeze=48,
        stem="synthetic24")
    project["timing"]["hold"] = 1.0  # a short card, to make the render quick
    project_file.save(folder, project)

    result = render.render(folder, matte_runner=fake_matte())

    # The rate stays 24, and the length is the plan's at 24: 48 of clip + 24 of card + 48 of tail
    assert [s["r_frame_rate"] for s in ffprobe(result.video, "r_frame_rate")] == ["24/1"]
    assert result.frames == media.count_frames(result.video) == 120
    assert result.freeze_out == pytest.approx(48 / 24) and result.release_out == pytest.approx(72 / 24)
    assert result.video.name == "synthetic24-main-character.mp4"
    assert color_tags(result.video) == {"stream": BT709, "frame": BT709}

    # No frame is made up: the stage changes from every one of the clip's 48 frames before the
    # freeze to the next, and from every one of the 28 after it. The clip is one picture of moving
    # test pattern, so a repeated frame would be a step of about 0, as it is when the same clip is
    # converted to 30 fps.
    whole = (0, 0, *OUT)  # (the clip fills the whole frame)
    assert min(motion(result.video, 0, 48, whole)) > 0.5
    assert min(motion(result.video, 72, 28, whole)) > 0.5
    converted = tmp_path / "converted.mp4"
    media.run(["-i", str(synthetic_clip_24), "-vf", "fps=30", "-c:v", "libx264", "-preset",
               "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p", str(converted)])
    assert min(motion(converted, 0, 60, (0, 0, 1920, 1080))) < 0.1

    # The notes arrive by the stagger in time: at 24 fps the top notes join on card frame 2 and the
    # arrows on frame 4 (at 30 fps they would be 2 and 5), so card frame 3 has no arrows yet
    source = frozen(folder, project, tmp_path / "freeze", fake_matte)
    for card_frame, shown in ((1, "bottom"), (3, "bottom and top"), (4, "all")):
        frame = frame_rgb(result.video, 48 + card_frame)
        gaps = {name: distance(frame, card_rgb(folder, project, visible, (card_frame // 6) % 3, 0.0, OUT, source))
                for name, visible in NOTES_ON.items()}
        assert min(gaps, key=gaps.get) == shown, (card_frame, gaps)


def final_join(calls):
    """The second pass of the join (the encode that writes the video), and its first pass."""
    passes = [args for args in calls if "libx264" in args and "-qp" not in args]
    first = [args for args in passes if args[args.index("-pass") + 1] == "1"]
    second = [args for args in passes if args[args.index("-pass") + 1] == "2"]
    assert len(first) == len(second) == 1
    return second[0], first[0]


def test_final_render_encodes_once_at_the_clips_bitrate(rendered, small_final):
    # The parts are lossless intermediates, and exactly one encode is not: the join, in two passes.
    # A single pass at an average bitrate spends the budget as it goes and can leave the end of
    # the video starved (a real intro: 2.6 Mbps for the last second of a 10.7 Mbps clip), so a
    # first pass reads the whole video and the second spreads the bits where the picture needs them.
    for calls, parts in ((rendered.calls, 3), (small_final[0].calls, 2)):
        encodes = [args for args in calls if "libx264" in args]
        lossless = [args for args in encodes if "-qp" in args]
        assert len(lossless) == parts and len(encodes) == parts + 2
        assert not any("-crf" in args for args in encodes)
        for args in lossless:  # the parts: quantizer 0, the fastest preset
            assert args[args.index("-qp") + 1] == "0"
            assert args[args.index("-preset") + 1] == "ultrafast"
        join, first = final_join(calls)  # and the join is the one that is not
        for args in (join, first):
            assert args[args.index("-f") + 1] == "concat"
            assert args[args.index("-preset") + 1] == "medium"
            assert args[args.index("-passlogfile") + 1] == join[join.index("-passlogfile") + 1]
        assert first[-1] == os.devnull and first[first.index("-f", first.index("-pass")) + 1] == "null"
        assert "colour_primaries=1" in join[join.index("-bsf:v") + 1]  # the BT.709 tags
        # and the card, a still with grain, gets a quarter of its share, so the footage has the bits
        zones = {args[args.index("-x264-params") + 1] for args in (join, first)}
        (zone,) = zones
        start, end, weight = zone.removeprefix("zones=").split(",")
        assert weight == f"b={media.STILL_WEIGHT}" and 0 < int(start) < int(end)

    # The fake project has no bitrate of its clip, so it is the floor of 10 Mbps, in both passes
    join, first = final_join(rendered.calls)
    for args in (join, first):
        assert [args[args.index(key) + 1] for key in ("-b:v", "-maxrate", "-bufsize")] == [
            "10000000", "15000000", "20000000"]

    # and a clip of 20 Mbps gets 20 Mbps, with a ceiling of 30 and a buffer of 40
    join, _ = final_join(small_final[0].calls)
    assert [join[join.index(key) + 1] for key in ("-b:v", "-maxrate", "-bufsize")] == [
        "20000000", "30000000", "40000000"]

    # That is what the finished file says about itself too, and it is still BT.709
    options = x264_options(small_final[0].result.video)
    assert {key: options[key] for key in ("rc", "bitrate", "vbv_maxrate", "vbv_bufsize")} == {
        "rc": "2pass", "bitrate": "20000", "vbv_maxrate": "30000", "vbv_bufsize": "40000"}
    assert color_tags(small_final[0].result.video) == {"stream": BT709, "frame": BT709}


def test_render_is_the_size_of_the_clip_and_says_so_when_it_is_not(small_final):
    result = small_final[0].result  # a clip of 1920x1080: its own size, so nothing to say
    assert result.notes == []
    assert result.frames == 45  # 15 frames of clip and 30 of card
    info = media.probe(result.video)
    assert (info["width"], info["height"]) == OUT

    # A clip under 1080 wide is scaled up to 1080 wide, so that the notes stay sharp, and one over 4K
    # (3840 on its long side) is scaled down to it. Each is told, with the size it is rendered at. Fast
    # clips are noted too.
    note = render.output_notes
    assert note({"width": 1920, "height": 1080, "fps": 30.0}, Fraction(30)) == []
    assert note({"width": 1080, "height": 1920, "fps": 30.0}, Fraction(30)) == []
    assert note({"width": 3840, "height": 2160, "fps": 30.0}, Fraction(30)) == []  # 4K itself is the most
    assert note({"width": 320, "height": 180, "fps": 30.0}, Fraction(30)) == [
        "Scaled up to 1080x608 so the notes stay sharp; your clip is 320x180."]
    assert note({"width": 720, "height": 1280, "fps": 30.0}, Fraction(30)) == [
        "Scaled up to 1080x1920 so the notes stay sharp; your clip is 720x1280."]
    assert note({"width": 4096, "height": 2160, "fps": 30.0}, Fraction(30)) == [
        "Rendered at 3840x2026; your clip is 4096x2160, bigger than 4K."]
    assert note({"width": 4320, "height": 7680, "fps": 120.0}, Fraction(60)) == [
        "Rendered at 2160x3840; your clip is 4320x7680, bigger than 4K.",
        "Your clip runs at 120 fps; rendered at 60 fps, the highest this skill makes."]
    # 60.2 fps is 60, and 59.94 is 60000/1001: no sentence for either
    assert note({"width": 1920, "height": 1080, "fps": 60.2}, Fraction(60)) == []
    assert note({"width": 1920, "height": 1080, "fps": 59.94}, Fraction(60000, 1001)) == []


def test_render_makes_a_120_fps_clip_at_60(fake_project):
    folder, project = fake_project
    small(folder, project, fps=120.0)
    freeze_on(folder, project, 30)  # half a second in, counted in the 60 fps frames the render makes

    result = render.render(folder, preview=True)

    assert result.notes == [
        "Your clip runs at 120 fps; rendered at 60 fps, the highest this skill makes."]
    assert [s["r_frame_rate"] for s in ffprobe(result.video, "r_frame_rate")] == ["60/1"]
    # the plan at 60 fps: 30 frames of head (0.5 s), 60 of card (1 s), no tail
    assert result.frames == media.count_frames(result.video) == 90
    assert result.freeze_out == pytest.approx(0.5)


def test_render_reports_progress(small_final):
    progress = small_final[1]
    stages = []
    for stage, done, total in progress:
        assert 1 <= done <= total
        if stage not in stages:
            stages.append(stage)
    # (the freeze frame is cut at the size of the video, which is not the plan's: that is the "cutout")
    assert stages == ["cutout", "cards", "parts", "join", "sound", "cover"]
    assert [step for step in progress if step[0] == "cutout"] == [("cutout", 1, 1)]
    for stage in stages:  # each stage ends having done all of its pieces
        last = [step for step in progress if step[0] == stage][-1]
        assert last[1] == last[2]
    assert [step for step in progress if step[0] == "parts"][-1] == ("parts", 2, 2)


# --- the parts, by what each one is --------------------------------------------------------


def test_render_encodes_parts_by_kind_when_the_card_comes_first(fake_project):
    folder, project = fake_project
    freeze_on(folder, project, 0)  # nothing before the freeze: the card is the first part
    project["timing"].update(hold=1.0, tail=0.3)
    project_file.save(folder, project)
    assert [s.kind for s in render.timeline_for(project).segments] == ["card", "clip"]

    result = render.render(folder, preview=True)

    assert result.frames == media.count_frames(result.video) == 30 + 9
    # the video opens on the card, with the bottom notes on, as the card's first frame has them
    candidates = {name: card.render_card(folder, project, visible)
                  for name, visible in NOTES_ON.items()}
    assert closest(frame_rgb(result.video, 0, size=PLAN), candidates)[0] == "bottom"


# --- the work folder, and what goes wrong -----------------------------------------------------


def test_a_render_that_finds_another_running_is_turned_away_and_leaves_its_work_alone(fake_project):
    # The studio and a terminal once rendered one project at the same time: the second cleared the
    # first's renders/_work, and the first lost its freeze frames halfway through.
    folder, _ = fake_project
    renders = folder.resolve() / "renders"
    theirs = renders / "_work" / "freeze" / "frames" / "000001.jpg"
    theirs.parent.mkdir(parents=True)
    theirs.write_bytes(b"the other render's frame")
    with fileio.held_alone(renders / ".render-lock", "held by the other render"):
        with pytest.raises(fileio.Busy, match="Another render of this project is running"):
            render.render(folder, preview=True)
    assert theirs.read_bytes() == b"the other render's frame"
    assert not (renders / "preview.mp4").exists()
    assert render.render(folder, preview=True).video.exists()  # and once it is done, a render runs


def test_render_clears_the_work_folder_even_when_it_is_locked_for_a_moment(fake_project, monkeypatch):
    folder, project = fake_project
    small(folder, project)
    work = folder / "renders" / "_work"
    work.mkdir(parents=True)
    (work / "left-by-a-failed-render.txt").write_text("old", encoding="utf-8")

    # Windows refuses to delete a file ffmpeg wrote for a moment after it exits
    real, refusals, naps = shutil.rmtree, [], []

    def flaky(path, *args, **kwargs):
        if (Path(path) / "part-00.mp4").exists() and len(refusals) < 2:
            refusals.append(path)
            raise PermissionError(5, "Access is denied")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(render.shutil, "rmtree", flaky)
    monkeypatch.setattr(time, "sleep", naps.append)

    result = render.render(folder, preview=True)

    assert len(refusals) == 2 and len(naps) == 2  # asked three times, waiting between
    assert not work.exists()
    assert result.video.is_file()


def test_render_does_not_fail_when_the_work_folder_cannot_be_removed(fake_project, monkeypatch):
    folder, project = fake_project
    small(folder, project)
    real = shutil.rmtree

    def locked(path, ignore_errors=False, **kwargs):
        if not (Path(path) / "part-00.mp4").exists():
            return real(path, ignore_errors=ignore_errors, **kwargs)
        if ignore_errors:
            return None  # what the real one does with errors it was told to ignore
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(render.shutil, "rmtree", locked)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    result = render.render(folder, preview=True)  # the video is made, the folder is left over

    assert result.video.is_file() and (folder / "renders" / "_work").is_dir()
    monkeypatch.undo()
    render.render(folder, preview=True)  # and the next render clears it
    assert not (folder / "renders" / "_work").exists()


def test_render_goes_on_when_a_stale_work_folder_cannot_be_removed(fake_project, monkeypatch):
    folder, project = fake_project
    small(folder, project)
    work = folder / "renders" / "_work"
    work.mkdir(parents=True)
    (work / "stale.txt").write_text("left by a render that failed, and something has it open", encoding="utf-8")
    real = shutil.rmtree

    def stuck(path, ignore_errors=False, **kwargs):
        if not (Path(path) / "stale.txt").exists():
            return real(path, ignore_errors=ignore_errors, **kwargs)
        if ignore_errors:
            return None  # what the real one does with errors it was told to ignore
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(render.shutil, "rmtree", stuck)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    result = render.render(folder, preview=True)  # the folder survives the clean-up, and is used as it is

    assert result.video.is_file() and (work / "stale.txt").is_file()
    monkeypatch.undo()
    render.render(folder, preview=True)  # once nothing holds it, the next render clears it
    assert not work.exists()


@pytest.mark.parametrize("preview", [False, True])
def test_a_render_that_fails_clears_its_work_folder(fake_project, fake_matte, monkeypatch, preview):
    # The parts of a final render are lossless, and large. A render that stops halfway leaves nothing of them.
    folder, project = fake_project
    small(folder, project)
    work = folder / "renders" / "_work"
    seen = []

    def stops_after_the_first_part(*args, **kwargs):
        seen.append(sorted(path.name for path in work.iterdir()))  # (the work folder is in use at this point)
        raise media.MediaError("ffmpeg failed: the disk is full")

    monkeypatch.setattr(media, "encode_stills", stops_after_the_first_part)
    monkeypatch.setattr(media, "encode_clip", stops_after_the_first_part)

    with pytest.raises(media.MediaError, match="the disk is full"):
        render.render(folder, preview=preview, matte_runner=fake_matte())

    assert seen and any(name.startswith("card-") for name in seen[0])
    assert not work.exists()
    assert not (folder / "renders" / "synthetic-main-character.mp4").exists()
    assert project_file.load(folder)["status"]["rendered_at"] is None


def test_a_render_that_fails_still_says_why_when_the_work_folder_cannot_be_cleared(fake_project, monkeypatch):
    folder, project = fake_project
    small(folder, project)
    real = shutil.rmtree

    def locked(path, ignore_errors=False, **kwargs):
        if not (Path(path) / "card-00.png").exists():
            return real(path, ignore_errors=ignore_errors, **kwargs)
        if ignore_errors:
            return None  # what the real one does with errors it was told to ignore
        raise PermissionError(5, "Access is denied")

    def stops(*args, **kwargs):
        raise media.MediaError("ffmpeg failed: the disk is full")

    monkeypatch.setattr(render.shutil, "rmtree", locked)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(media, "encode_stills", stops)
    monkeypatch.setattr(media, "encode_clip", stops)

    # the error that stopped the render is the one that is raised, not the folder that would not go
    with pytest.raises(media.MediaError, match="the disk is full"):
        render.render(folder, preview=True)
    assert (folder / "renders" / "_work").is_dir()  # (left over, and the next render clears it)


def test_inputs_give_the_clip(fake_project):
    folder, project = fake_project

    source = render._inputs(folder, project)

    assert source.is_file()


def test_the_record_of_a_render_takes_the_projects_lock(fake_project):
    # The studio saves project.json for each edit, and a render saves its status into it: one lock, so that
    # neither overwrites the other
    folder, _ = fake_project
    done = threading.Event()

    def record():
        render._record(folder, [folder / "renders" / "a.mp4"], 2.0, 5.0)
        done.set()

    with project_file.UPDATE_LOCK:  # an edit is being saved
        threading.Thread(target=record).start()
        assert not done.wait(0.4)  # the record waits for it
    assert done.wait(10)
    status = project_file.load(folder)["status"]
    assert status["outputs"] == ["renders/a.mp4"] and (status["freeze_out"], status["release_out"]) == (2.0, 5.0)


def test_render_publishes_and_records_through_a_lock_that_clears(fake_project, fake_matte, monkeypatch):
    folder, project = fake_project
    small(folder, project)
    real, refusals, naps = os.replace, {}, []

    def held_for_a_moment(source, destination, *args, **kwargs):
        # a video player, or a virus scan, has the file open: Windows says "Access is denied"
        name = Path(destination).name
        if refusals.get(name, 0) < 2:
            refusals[name] = refusals.get(name, 0) + 1
            raise PermissionError(5, "Access is denied")
        return real(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", held_for_a_moment)
    monkeypatch.setattr(time, "sleep", naps.append)

    result = render.render(folder, matte_runner=fake_matte())

    # the video, the cover and project.json were each refused twice, 0.1 s apart, and then went through
    assert refusals == {"synthetic-main-character.mp4": 2, "synthetic-cover.png": 2, "project.json": 2}
    assert naps == [0.1] * 6
    assert result.video.is_file() and result.cover.is_file()
    assert not (folder / "renders" / "_work").exists()
    status = project_file.load(folder)["status"]
    assert status["outputs"] == ["renders/synthetic-main-character.mp4", "renders/synthetic-cover.png"]
    assert status["rendered_at"] is not None


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to replace a file that is open")
def test_render_publishes_over_a_video_that_a_player_really_has_open(fake_project, fake_matte):
    folder, project = fake_project
    small(folder, project)
    old = folder / "renders" / "synthetic-main-character.mp4"
    old.parent.mkdir()
    old.write_bytes(b"the video of an earlier render")
    player = open(old, "rb")  # the studio's player, or a virus scan, has it open

    def let_go_soon(stage, done, total):
        if stage == "cover":  # the render is about to move its files into place
            threading.Timer(0.35, player.close).start()

    try:
        result = render.render(folder, progress=let_go_soon, matte_runner=fake_matte())
    finally:
        player.close()

    assert result.video == old.resolve() and media.count_frames(old) == 45  # the new one is there
    assert project_file.load(folder)["status"]["rendered_at"] is not None
    assert not (old.parent / "_work").exists()


def test_render_names_the_file_when_a_lock_never_clears(fake_project, fake_matte, monkeypatch):
    folder, project = fake_project
    small(folder, project)
    renders = folder / "renders"
    real, naps = os.replace, []

    def player_has_it(source, destination, *args, **kwargs):
        if Path(destination).name == "synthetic-main-character.mp4":
            raise PermissionError(5, "Access is denied")
        return real(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", player_has_it)
    monkeypatch.setattr(time, "sleep", naps.append)

    # ten tries, 0.1 s apart, and then a sentence that says which file to let go of
    with pytest.raises(PermissionError, match=r"synthetic-main-character\.mp4 is in use by another program"):
        render.render(folder, matte_runner=fake_matte())
    assert naps == [0.1] * 9
    assert not (renders / "synthetic-main-character.mp4").exists()  # nothing was half moved
    assert project_file.load(folder)["status"]["rendered_at"] is None  # and nothing says it worked
    assert not (renders / "_work").exists()  # the work folder, with its large parts, is not left behind

    # Let go of the file, and the next render is made from the start
    monkeypatch.undo()
    result = render.render(folder, matte_runner=fake_matte())
    assert result.video.is_file() and not (renders / "_work").exists()
    assert project_file.load(folder)["status"]["rendered_at"] is not None


def test_render_stops_when_the_join_has_the_wrong_number_of_frames(fake_project, fake_matte, monkeypatch):
    folder, project = fake_project
    small(folder, project)  # 15 frames of clip and 30 of card
    real = media.count_frames
    monkeypatch.setattr(
        media, "count_frames", lambda path: real(path) - (1 if Path(path).name == "joined.mp4" else 0))

    with pytest.raises(media.MediaError, match="came out with 44 frames, not the 45 planned"):
        render.render(folder, matte_runner=fake_matte())

    # a click placed by the plan would be a frame off, so nothing was kept (and nothing is left in _work)
    assert not (folder / "renders" / "synthetic-main-character.mp4").exists()
    assert not (folder / "renders" / "_work").exists()
    assert project_file.load(folder)["status"]["rendered_at"] is None


def test_render_names_what_is_missing(fake_project):
    folder, project = fake_project

    project["clip"]["path"] = str(folder / "gone.mp4")
    project_file.save(folder, project)
    with pytest.raises(FileNotFoundError, match=r"clip .*gone\.mp4"):
        render.render(folder)

    project["clip"]["path"] = str(folder / "frames" / "000000.jpg")  # something that does exist
    project["freeze"]["frame"] = 120  # the clip has frames 0 to 119
    project_file.save(folder, project)
    with pytest.raises(ValueError, match="freeze frame is 120.*last is 119"):
        render.render(folder)

    # a freeze frame whose picture or cutout is not in the folder, and a clip that is not described
    project["freeze"]["frame"] = 61
    project_file.save(folder, project)
    with pytest.raises(FileNotFoundError, match=r"000061\.jpg is missing. Run init"):
        render.render(folder)
    (folder / "frames" / "000061.jpg").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"000061\.png is missing. Run init"):
        render.render(folder)
    project["freeze"]["frame"] = 60
    del project["clip"]["fps"], project["clip"]["width"]
    project_file.save(folder, project)
    with pytest.raises(ValueError, match="clip has no width, fps. Run init"):
        render.render(folder)
    assert not (folder / "renders").exists()  # none of them started anything


def test_render_wants_frames_cut_at_the_size_its_clip_needs(fake_project):
    folder, project = fake_project
    project["work"] = {"width": 1080, "height": 1920}  # what the full screen made of a 16:9 clip, before
    project_file.save(folder, project)

    with pytest.raises(ValueError, match=r"cut at 1080x1920, and a clip of 1920x1080 needs 1080x608\. Run init"):
        render.render(folder)
    assert not (folder / "renders").exists()


def test_a_final_render_cuts_the_freeze_frame_again_at_the_size_of_the_video(fake_project, fake_matte):
    folder, project = fake_project
    small(folder, project)
    runner = fake_matte()
    progress = []

    result = render.render(folder, progress=lambda *step: progress.append(step), matte_runner=runner)

    # RVM is run on the freeze frame (15) after the 12 before it, which are its warmup, at the size of
    # the video, at the ratio the model gets at that size: 576 / 1920 = 0.3, the same picture it sees at
    # the plan. (The freeze is the 13th of the frames cut: 3 to 15.)
    assert runner.calls == [(0.3, 12, 13)]
    assert media.probe(result.video)["width"] == 1920
    assert ("cutout", 1, 1) in progress

    # The detail the project says is the model's: "high" is 1.6 times as much (at most 1), and a person
    # who fills the frame needs a closer look. A freeze near the start has fewer frames before it.
    project["matte"]["detail"] = "high"
    freeze_on(folder, project, 5)
    again = fake_matte()
    render.render(folder, preview=False, matte_runner=again)
    assert again.calls == [(0.48, 5, 6)]


def test_a_final_render_needs_the_matte_model_when_it_cuts_the_freeze_frame_again(fake_project, monkeypatch):
    folder, project = fake_project
    small(folder, project)

    def missing(*args, **kwargs):
        raise matte.ModelMissing("The matte model is missing: it should be at somewhere.")

    monkeypatch.setattr(matte, "ensure_model", missing)

    with pytest.raises(matte.ModelMissing, match="matte model is missing"):
        render.render(folder)
    # nothing was published, and a preview, which is at the plan, needs no cutout and no model
    assert not list((folder / "renders").glob("*.mp4")) and not list((folder / "renders").glob("*.png"))
    assert render.render(folder, preview=True).video.is_file()
    assert project_file.load(folder)["status"]["rendered_at"] is None


def test_the_head_and_the_tail_are_the_clip_at_its_own_size(rendered, tmp_path):
    result, folder, project, _ = rendered
    # frames 0 to 59 of the video are frames 0 to 59 of the clip, with nothing around them and nothing
    # drawn on them, at the size of the video
    frames = tmp_path / "frames"
    media.extract_frames(Path(project["clip"]["path"]), frames, OUT)
    for number, clip_frame in ((0, 0), (28, 28), (59, 59)):
        with Image.open(frames / f"{clip_frame:06d}.jpg") as jpeg:
            wanted = np.asarray(jpeg.convert("RGB"), float) / 255
        assert distance(frame_rgb(result.video, number), wanted) < 0.02, (number, clip_frame)
    # and the tail picks the clip up after the card: frame 150 is frame 60 of the clip
    with Image.open(frames / "000060.jpg") as jpeg:
        wanted = np.asarray(jpeg.convert("RGB"), float) / 255
    assert distance(frame_rgb(result.video, 150), wanted) < 0.02
