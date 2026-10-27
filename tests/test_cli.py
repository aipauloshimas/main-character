"""Tests for mc.py, the command line, and for the waiter that its wait command runs.

The commands run in this process (mc.main, with the slow or outside parts replaced where a test is
about the words a command prints) and, for the ones that matter most, as a program of their own,
the way Claude runs them.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

import mc
from mclib import analyze, card, doctor, matte, media, palette, pipeline, render, server, wait
from mclib import project as project_file
from mclib.raster import from_image
from mclib.render import RenderResult
from mclib.styles import recipes

MC = Path(__file__).resolve().parents[1] / "scripts" / "mc.py"
LONG_AGO = "2020-01-01T00:00:00+00:00"
STAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}(Z|\+00:00)")  # UTC, to the millisecond


@pytest.fixture(autouse=True)
def private_home(tmp_path_factory, monkeypatch):
    """No test of the commands reads the styles in the real home folder of whoever runs them."""
    monkeypatch.setenv("MC_HOME", str(tmp_path_factory.mktemp("mc-home")))


def run_mc(*args, env=None):
    """Run mc.py as a program, the way Claude does, and return what it did (stdout, stderr and the
    exit code), with the text read as UTF-8."""
    return subprocess.run(
        [sys.executable, str(MC), *[str(arg) for arg in args]], capture_output=True,
        encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, env=env,
    )


def stamp(offset=0.0):
    """The time now, plus some seconds, as an ISO text with milliseconds, in UTC."""
    when = datetime.now(timezone.utc) + timedelta(seconds=offset)
    return when.isoformat(timespec="milliseconds")


def change_project(folder, change):
    """Read project.json, let change(project) edit it, and write it back, as the studio does. The
    waiter reads the file now and then, and Windows does not let a file be replaced in the instant
    it is being read: project.save waits that out."""
    project = project_file.load(folder)
    change(project)
    project_file.save(folder, project)


def later(folder, change, delay=0.3):
    """Change the project after a delay, from another thread, as the studio does while the waiter
    waits. The thread is started and returned."""
    def work():
        time.sleep(delay)
        change_project(folder, change)

    thread = threading.Thread(target=work)
    thread.start()
    return thread


# --- the two tests of the task ----------------------------------------------------------------


def test_wait_returns_done_and_style(tmp_path, fake_project):
    folder, _ = fake_project

    # What happened before the call is not news
    def old_news(project):
        project["status"]["done_at"] = LONG_AGO
        project["style_request"] = {"at": LONG_AGO, "text": "vaporwave sunset"}

    change_project(folder, old_news)
    with pytest.raises(TimeoutError):
        wait.wait_event(folder, poll=0.05, timeout=0.4)

    # Done is stamped while it waits, and the stamp comes back as the file has it
    writer = later(folder, lambda project: project["status"].update(done_at=stamp()))
    event, done_at = wait.wait_event(folder, poll=0.05, timeout=30)
    writer.join()
    assert event == "DONE" and done_at == project_file.load(folder)["status"]["done_at"]
    assert wait.parse_time(done_at) > datetime.now(timezone.utc) - timedelta(seconds=30)

    # Then a style request: counting from the Done that was just heard, so that it is not told again
    writer = later(folder, lambda project: project["style_request"].update(at=stamp()))
    event, style_at = wait.wait_event(folder, since=done_at, poll=0.05, timeout=30)
    writer.join()
    assert event == "STYLE" and style_at == project_file.load(folder)["style_request"]["at"]
    assert wait.parse_time(style_at) > wait.parse_time(done_at)


def test_cli_card_writes_png(fake_project, tmp_path):
    folder, _ = fake_project
    out = tmp_path / ("my caf" + chr(0xE9) + " cards") / "card.png"  # a new folder, with a space and an accent

    done = run_mc("card", folder, "--out", out)

    assert done.returncode == 0 and done.stderr == ""
    assert Path(done.stdout.strip()) == out.resolve()  # it says where the card is
    with Image.open(out) as image:
        assert image.size == (1080, 608) and image.mode == "RGB"  # the plan: the clip's own frame

    # With no --out the card is in the project folder, and --style draws another style's
    plain = run_mc("card", folder)
    assert Path(plain.stdout.strip()) == (folder / "card.png").resolve()
    comic = run_mc("card", folder, "--out", tmp_path / "comic.png", "--style", "comic-pop")
    assert comic.returncode == 0
    with Image.open(folder / "card.png") as bedroom, Image.open(tmp_path / "comic.png") as other:
        assert bedroom.tobytes() != other.tobytes()
    unknown = run_mc("card", folder, "--style", "no-such-style")
    assert unknown.returncode == 1 and unknown.stdout == ""
    assert unknown.stderr.strip() == "ERROR There is no style 'no-such-style'."


# --- the waiter ---------------------------------------------------------------------------------


def test_wait_returns_the_earlier_event_first_and_only_what_is_newer(fake_project):
    folder, _ = fake_project
    early, late = "2030-01-01T00:00:00.000+00:00", "2030-01-01T00:00:05.250Z"  # either spelling of UTC

    def both(project):
        project["status"]["done_at"] = late
        project["style_request"] = {"at": early, "text": "x"}

    change_project(folder, both)
    since = "2029-12-31T23:59:00+00:00"

    # Both are newer than since: the one that came first, and then the other, and then none
    assert wait.wait_event(folder, since=since, timeout=0) == ("STYLE", early)
    assert wait.wait_event(folder, since=early, timeout=0) == ("DONE", late)  # the Z as written
    with pytest.raises(TimeoutError, match="Nothing happened"):
        wait.wait_event(folder, since=late, timeout=0)

    # An event at the very moment of since is the one that woke Claude, not a new one. Another
    # zone is the same moment: 01:00:05.25 at +01:00 is 00:00:05.25 in UTC.
    with pytest.raises(TimeoutError):
        wait.wait_event(folder, since="2030-01-01T01:00:05.250+01:00", timeout=0)

    # Events that come together are told Done first
    def together(project):
        project["status"]["done_at"] = early
    change_project(folder, together)
    assert wait.wait_event(folder, since=since, timeout=0) == ("DONE", early)

    # A stamp that is not a time is no event, and neither is a stamp that is not there
    def broken(project):
        project["status"]["done_at"] = "yesterday, I think"
        project["style_request"] = None
    change_project(folder, broken)
    with pytest.raises(TimeoutError):
        wait.wait_event(folder, since=since, timeout=0)
    assert wait.parse_time("2030-01-01T00:00:00").tzinfo is not None  # no zone means UTC


def test_wait_keeps_waiting_while_the_file_cannot_be_read(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    # no project yet, and then half of one, as a reader can meet a file that is being written
    threading.Timer(0.2, (folder / "project.json").write_text, ["{ not whole", "utf-8"]).start()
    threading.Timer(0.5, (folder / "project.json").write_text, [json.dumps(
        {"status": {"done_at": "2999-01-01T00:00:00+00:00"}}), "utf-8"]).start()

    assert wait.wait_event(folder, poll=0.05, timeout=30) == ("DONE", "2999-01-01T00:00:00+00:00")


# --- the commands -----------------------------------------------------------------------------


def test_cli_init_prints_the_project_the_warnings_and_the_freeze(fake_project, monkeypatch, capsys):
    folder, project = fake_project
    project["freeze"]["suggested"] = [60, 75, 90]
    project_file.save(folder, project)
    seen = []

    def init_project(clip, project_dir=None, matte_runner=None, progress=None):
        seen.append((clip, project_dir, matte_runner))
        return folder, ["The first thing.", "The second thing."]

    monkeypatch.setattr(pipeline, "init_project", init_project)

    code = mc.main(["init", "my clip.mp4", "--project-dir", str(folder)])

    output = capsys.readouterr()
    assert code == 0
    assert output.out.splitlines() == [
        f"PROJECT {folder}", "WARNING The first thing.", "WARNING The second thing.",
        "FREEZE 60 SUGGESTED 60,75,90"]
    assert seen == [(Path("my clip.mp4"), folder, None)]  # RVM, as the default

    seen.clear()
    assert mc.main(["init", "other.mp4"]) == 0
    assert seen == [(Path("other.mp4"), None, None)]


def test_cli_init_says_when_a_clip_of_another_shape_resets_the_note_positions(tmp_path, monkeypatch, capsys, fake_matte):
    # The real init, with the cutout replaced by the fake one, on a clip and then on a clip of another shape in the
    # same folder: the notes the person had pinned are free again, and the command says so on a WARNING line
    monkeypatch.setattr(matte, "ensure_model", lambda *args, **kwargs: Path("model.onnx"))
    monkeypatch.setattr(matte, "run_rvm", fake_matte())
    clips = {}
    for name, size in (("wide", "320x180"), ("tall", "180x320")):
        clips[name] = tmp_path / f"{name}.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=10:duration=2.4",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(clips[name])],
            check=True, capture_output=True,
        )
    folder = tmp_path / "proj"
    assert mc.main(["init", str(clips["wide"]), "--project-dir", str(folder)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"PROJECT {folder.resolve()}" and not any(line.startswith("WARNING") for line in lines)

    change_project(folder, lambda project: project["positions"].update(tl=[100, 80]))
    assert mc.main(["init", str(clips["tall"]), "--project-dir", str(folder)]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[1:-1] == ["WARNING Your note positions were reset because the clip's shape changed."]
    assert lines[-1].startswith("FREEZE ")
    assert project_file.load(folder)["positions"] == project_file.empty_positions()


def test_cli_render_prints_the_files_and_every_warning(tmp_path, monkeypatch, capsys, fake_project):
    folder, _ = fake_project
    seen = []
    warnings = ["Rendered at 3840x2026; your clip is 4096x2160, bigger than 4K.",
                "Your clip runs at 120 fps; rendered at 60 fps, the highest this skill makes."]

    def fake_render(project_dir, preview=False, progress=None):
        seen.append((project_dir, preview))
        renders = Path(project_dir) / "renders"
        if preview:
            return RenderResult(renders / "preview.mp4", None, None, 188, 2.0667, 5.0667, None)
        return RenderResult(
            renders / "a-main-character.mp4", renders / "a-with-music.mp4", renders / "a-cover.png",
            188, 2.0667, 5.0667, "The freeze lands at 0.67 s, so the trend's sound will not line up.",
            warnings)

    monkeypatch.setattr(render, "render", fake_render)
    renders = folder.resolve() / "renders"

    assert mc.main(["render", str(folder)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"VIDEO {renders / 'a-main-character.mp4'}", f"WITH_MUSIC {renders / 'a-with-music.mp4'}",
        f"COVER {renders / 'a-cover.png'}", "FRAMES 188 FREEZE 2.07 RELEASE 5.07",
        "WARNING The freeze lands at 0.67 s, so the trend's sound will not line up.",
        f"WARNING {warnings[0]}", f"WARNING {warnings[1]}"]

    assert mc.main(["render", str(folder), "--preview"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"VIDEO {renders / 'preview.mp4'}", "FRAMES 188 FREEZE 2.07 RELEASE 5.07"]
    assert seen == [(folder.resolve(), False), (folder.resolve(), True)]


def test_cli_wait_prints_the_event_as_one_line(fake_project, capsys):
    folder, _ = fake_project
    change_project(folder, lambda project: project["status"].update(done_at="2999-05-06T07:08:09+00:00"))

    assert mc.main(["wait", str(folder), "--since", "2999-01-01T00:00:00Z"]) == 0

    assert capsys.readouterr().out == "DONE 2999-05-06T07:08:09+00:00\n"


def test_cli_exits_non_zero_with_one_line_when_it_fails(tmp_path, fake_project, capsys, monkeypatch):
    folder, _ = fake_project
    nowhere = tmp_path / "no project here"
    long_clip, small_clip = tmp_path / "long.mp4", tmp_path / "small.mp4"
    for clip, seconds in ((long_clip, 31), (small_clip, 2.4)):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             f"testsrc2=size=160x90:rate=5:duration={seconds}", "-pix_fmt", "yuv420p", str(clip)],
            check=True, capture_output=True)

    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (160, 90), "red").save(photo)

    failures = [
        (["render", str(nowhere)], f"ERROR There is no project in {nowhere.resolve()}. Run init first."),
        (["card", str(nowhere)], "ERROR There is no project in"),
        (["wait", str(nowhere)], "ERROR There is no project in"),
        (["init", str(tmp_path / "missing.mp4")], "ERROR There is no clip at"),
        (["init", str(long_clip)], "ERROR The clip is 31.0 seconds long."),
        (["init", str(photo)],
         "ERROR That's a picture, not a video. Give me a clip of 2 to 30 seconds."),
        (["wait", str(folder), "--since", "when I said"],
         "ERROR --since must be a time like 2026-10-26T05:00:00Z."),
    ]
    for argv, message in failures:
        assert mc.main(argv) == 1, argv
        output = capsys.readouterr()
        assert output.out == "" and output.err.startswith(message), (argv, output.err)
        assert output.err.count("\n") == 1  # one line, no traceback

    # A command line that makes no sense is argparse's, which exits with 2 (the format is the clip's own now,
    # so init has no --layout any more)
    for argv in (["render"], ["frobnicate"], ["init", "x.mp4", "--layout", "sideways"], []):
        with pytest.raises(SystemExit) as stopped:
            mc.main(argv)
        assert stopped.value.code == 2, argv
    capsys.readouterr()

    # A failure that is not about the person's files is an error line too: the model is missing
    def no_model(*args, **kwargs):
        raise matte.ModelMissing("The matte model is missing: it should be at nowhere.")

    monkeypatch.setattr(matte, "ensure_model", no_model)
    assert mc.main(["init", str(small_clip), "--project-dir", str(tmp_path / "p")]) == 1
    assert capsys.readouterr().err.startswith("ERROR The matte model is missing")


def test_cli_doctor_prints_the_report_and_exits_by_it(monkeypatch, capsys):
    good = {"python": "3.13.2", "packages": {"numpy": "2.2.6", "PIL": "10.4.0", "onnxruntime": "1.20.1"},
            "ffmpeg": "8.1", "ffprobe": "8.1", "hdr": "zscale", "libx264": True, "fonts_missing": [],
            "model": True, "ok": True}
    bad = dict(good, ffmpeg=None, ok=False)
    found = []
    monkeypatch.setattr(doctor, "check", lambda *args, **kwargs: found[0])

    found.append(good)
    assert mc.main(["doctor"]) == 0
    assert capsys.readouterr().out == doctor.report(good) + "\n"

    found[0] = bad  # not ready: the report says so, and so does the exit code
    assert mc.main(["doctor"]) == 1
    output = capsys.readouterr().out
    assert output == doctor.report(bad) + "\n" and "ffmpeg: not found on the PATH" in output

    assert mc.main(["doctor", "--json"]) == 1
    assert json.loads(capsys.readouterr().out) == bad


def test_cli_doctor_downloads_the_model_only_when_asked(monkeypatch, tmp_path, capsys):
    target = tmp_path / "rvm" / "model.onnx"
    monkeypatch.setattr(doctor, "MODEL_PATH", target)
    downloads = []

    def download_model(dest, *args, **kwargs):
        downloads.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"a model")
        return dest

    monkeypatch.setattr(matte, "download_model", download_model)

    mc.main(["doctor"])  # it asks for nothing: a report that says the model is missing
    first = capsys.readouterr()
    assert downloads == [] and "matte model: missing" in first.out
    assert "python mc.py doctor --download-model" in first.out

    mc.main(["doctor", "--download-model"])
    second = capsys.readouterr()
    assert downloads == [target] and f"Downloaded the matte model to {target}" in second.err
    assert f"matte model: OK, {target}" in second.out  # and the report that follows sees it

    mc.main(["doctor", "--download-model"])  # it is there now: nothing is downloaded again
    third = capsys.readouterr()
    assert downloads == [target] and "already at" in third.err

    # a download that fails is an error with the reason
    target.unlink()

    def fails(dest, *args, **kwargs):
        raise matte.ModelMissing("The download from the web was only 312 bytes.")

    monkeypatch.setattr(matte, "download_model", fails)
    assert mc.main(["doctor", "--download-model"]) == 1
    assert "ERROR The download from the web was only 312 bytes." in capsys.readouterr().err


def test_cli_doctor_works_on_a_machine_with_none_of_the_packages(tmp_path):
    # The first thing the skill runs is doctor, before anything is installed. Here the three
    # packages are replaced by ones that fail to import, ahead of the real ones on the path.
    blocked = tmp_path / "blocked"
    for name in ("numpy", "PIL", "onnxruntime"):
        (blocked / name).mkdir(parents=True)
        (blocked / name / "__init__.py").write_text(
            f'raise ImportError("{name} is not installed here")', encoding="utf-8")
    # No web either, so that nothing could be downloaded if a test went wrong, and a home folder
    # that has no model in it
    dead_end = "http://127.0.0.1:9"
    env = dict(os.environ, PYTHONPATH=str(blocked), HTTP_PROXY=dead_end, HTTPS_PROXY=dead_end,
               http_proxy=dead_end, https_proxy=dead_end, HOME=str(tmp_path),
               USERPROFILE=str(tmp_path))

    done = run_mc("doctor", "--json", env=env)

    assert done.returncode == 1 and done.stderr == ""  # not ready, and no traceback
    result = json.loads(done.stdout)
    assert result["packages"] == {"numpy": None, "PIL": None, "onnxruntime": None}
    assert result["ok"] is False and result["ffmpeg"] is not None
    text = run_mc("doctor", env=env).stdout
    assert "numpy: not installed" in text and "Pillow: not installed" in text
    assert "\n  python -m pip install numpy pillow onnxruntime\n" in text

    # and asking for the model before the packages are there says what to do first
    asked = run_mc("doctor", "--download-model", env=env)
    assert asked.returncode == 1
    assert "ERROR The packages are not installed yet" in asked.stderr
    assert "python -m pip install numpy pillow onnxruntime" in asked.stderr


@pytest.mark.slow
def test_cli_init_then_render_a_24_fps_clip(synthetic_clip_24, tmp_path, monkeypatch, capsys, fake_matte):
    # The whole way through, as Claude runs it, with the cutout replaced by a fake one, and a clip
    # and a project folder that have a space and an accent in their names
    runner = fake_matte()
    monkeypatch.setattr(matte, "ensure_model", lambda *args, **kwargs: Path("model.onnx"))
    monkeypatch.setattr(matte, "run_rvm", runner)
    name = "v" + chr(0xED) + "deo do Jo" + chr(0xE3) + "o"
    clip = tmp_path / "clips" / f"{name}.mp4"
    clip.parent.mkdir()
    shutil.copy(synthetic_clip_24, clip)
    folder = tmp_path / f"my {name} project"

    assert mc.main(["init", str(clip), "--project-dir", str(folder)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"PROJECT {folder.resolve()}" and not any(line.startswith("WARNING") for line in lines)
    project = project_file.load(folder)
    assert lines[-1].startswith(f"FREEZE {project['freeze']['frame']} SUGGESTED ")

    # a short card, and the freeze on a frame with a second of clip before it
    project["freeze"]["frame"] = 48
    project["timing"]["hold"] = 1.0
    project_file.save(folder, project)
    plan = render.timeline_for(project)

    assert mc.main(["render", str(folder)]) == 0
    lines = capsys.readouterr().out.splitlines()
    video = Path(lines[0].removeprefix("VIDEO "))
    assert video == folder.resolve() / "renders" / f"{name}-main-character.mp4"
    assert [line.split(" ")[0] for line in lines] == ["VIDEO", "COVER", "FRAMES"]  # and no warning
    info = media.probe(video)
    # the video is the clip's own frame: 1920x1080 stays 1920x1080
    assert (info["width"], info["height"], info["fps"], info["frames"]) == (1920, 1080, 24.0, plan.total_frames)
    assert plan.total_frames == 2 + 48 + 24 + 48  # the tail is the 2 s of clip left
    assert lines[2] == f"FRAMES {plan.total_frames} FREEZE {plan.freeze_out:.2f} RELEASE {plan.release_out:.2f}"
    assert project_file.load(folder)["status"]["outputs"] == [
        f"renders/{name}-main-character.mp4", f"renders/{name}-cover.png"]

    # and a preview, which is at the plan's size, has no cover, and leaves the project as the render left it
    before = project_file.load(folder)
    assert mc.main(["render", str(folder), "--preview"]) == 0
    lines = capsys.readouterr().out.splitlines()
    preview = Path(lines[0].removeprefix("VIDEO "))
    assert preview == folder.resolve() / "renders" / "preview.mp4"
    assert [line.split(" ")[0] for line in lines] == ["VIDEO", "FRAMES"]
    assert (media.probe(preview)["width"], media.probe(preview)["height"]) == (1080, 608)
    assert project_file.load(folder) == before


# --- serve ------------------------------------------------------------------------------------


def start_serve(folder, *args):
    """Run `mc.py serve` as a program of its own, and return it with the first line it printed. A
    program that never prints one is killed after 60 seconds."""
    process = subprocess.Popen(
        [sys.executable, str(MC), "serve", str(folder), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
    )
    watchdog = threading.Timer(60, process.kill)
    watchdog.start()
    try:
        return process, process.stdout.readline().strip()
    finally:
        watchdog.cancel()


def stop(process):
    process.terminate()
    process.communicate(timeout=30)  # (also closes the pipes)


def test_cli_serve_prints_the_url_and_serves_the_project(fake_project):
    import urllib.request

    folder, project = fake_project
    process, line = start_serve(folder)
    try:
        assert line.startswith("URL http://127.0.0.1:") and line.endswith("/")
        port = int(line.rsplit(":", 1)[1].rstrip("/"))
        assert port >= 8340  # the first free port from 8340
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/project", timeout=30) as answer:
            assert json.loads(answer.read())["style"] == project["style"]
    finally:
        stop(process)

    # A port that is asked for is the port that is used
    wanted = port + 40
    process, line = start_serve(folder, "--port", str(wanted))
    try:
        assert line == f"URL http://127.0.0.1:{wanted}/"
    finally:
        stop(process)


def test_cli_serve_says_what_is_wrong(tmp_path, fake_project):
    import socket

    folder, _ = fake_project
    nothing = run_mc("serve", tmp_path / "nowhere")
    assert nothing.returncode == 1 and nothing.stdout == ""
    assert nothing.stderr.startswith("ERROR There is no project in")

    for port in ("0", "70000", "abc"):
        wrong = run_mc("serve", folder, "--port", port)
        assert wrong.returncode == 2 and "from 1 to 65535" in " ".join(wrong.stderr.split())

    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen()
    try:
        busy = run_mc("serve", folder, "--port", held.getsockname()[1])
    finally:
        held.close()
    assert busy.returncode == 1 and busy.stdout == ""
    assert busy.stderr.startswith("ERROR Port ") and " is in use." in busy.stderr


# --- the commands that SKILL.md relies on: patch, the flags of card, refine, style ----------------

ROBOT = chr(0x1F916)  # an emoji, built from its code point so no editor or tool can change it

# A recipe as Claude writes one for "Ask Claude" (SKILL.md, step 8). Its pixel notes are measured on
# the night background, so a light ink passes the contrast guard.
SUNSET = {
    "id": "custom", "name": "Sunset Arcade", "bg": "night", "edge": "neon", "notes": "pixel",
    "arrows": "none", "hero": "hud", "extras": {"tape": False, "stars": True, "grain": False},
    "palette": {"dark": "#1a0b2e", "light": "#ff6ad5", "tex": "#26c6da", "edge": "#ffd1f7", "ink": "#fff3a8"},
}


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def fails(argv, capsys, message):
    """Run a command that has to fail the way every command does: nothing on stdout, one ERROR line
    on stderr that starts with `message`, and the exit code 1."""
    assert mc.main([str(arg) for arg in argv]) == 1, argv
    output = capsys.readouterr()
    assert output.out == "" and output.err.startswith(message), (argv, output.err)
    assert output.err.count("\n") == 1  # one line, no traceback


def test_cli_patch_merges_validates_saves_and_prints_the_contrast(fake_project, tmp_path, capsys):
    folder, project = fake_project
    contrast = f"CONTRAST {palette.floor_contrast(palette.text_contrast(project['style']))}"
    patch_file = tmp_path / "patch.json"

    # A merge patch: objects merge, a null deletes the key, and anything else replaces what was there
    write_json(patch_file, {
        "copy": {"name": "Pauli", "notes": {"br": "Best friend: Clover"}},
        "picked": None,
        "timing": {"hold": 4.0},
    })
    assert mc.main(["patch", str(folder), "--file", str(patch_file)]) == 0

    assert capsys.readouterr().out == contrast + "\n"  # no warning, so the contrast alone
    saved = project_file.load(folder)
    assert saved["copy"]["name"] == "Pauli" and saved["copy"]["notes"]["br"] == "Best friend: Clover"
    assert saved["copy"]["notes"]["tl"] == project["copy"]["notes"]["tl"]  # the rest of the copy stays
    assert "picked" not in saved  # the null deleted the key
    assert saved["timing"]["hold"] == 4.0 and saved["timing"]["tail"] == project["timing"]["tail"]
    assert saved["clip"] == project["clip"]

    # Over a soft limit only warns, and the text is kept whole
    write_json(patch_file, {"copy": {"notes": {"tl": "T" * 31}}})
    assert mc.main(["patch", str(folder), "--file", str(patch_file)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "WARNING copy.notes.tl is 31 characters; the soft limit is 30.", contrast]
    assert project_file.load(folder)["copy"]["notes"]["tl"] == "T" * 31

    # The video fonts cannot draw emoji: they are taken out, with a warning
    write_json(patch_file, {"copy": {"notes": {"tl": f"Creator {ROBOT} educator"}}})
    assert mc.main(["patch", str(folder), "--file", str(patch_file)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "WARNING Removed emoji from copy.notes.tl; the video fonts cannot draw them.", contrast]
    assert project_file.load(folder)["copy"]["notes"]["tl"] == "Creator educator"

    # Lettering that is hard to read is a warning for a patch (the style command refuses it)
    dark = project["style"]["palette"]["dark"]
    write_json(patch_file, {"style": {"palette": {"ink": dark}}})  # marker lettering in its own background color
    assert mc.main(["patch", str(folder), "--file", str(patch_file)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "WARNING Text contrast 1.0:1 is hard to read on a phone.", "CONTRAST 1.0"]
    assert project_file.load(folder)["style"]["palette"]["ink"] == dark

    # A patch that changes nothing is fine, and leaves the file alone
    before = (folder / "project.json").read_bytes()
    write_json(patch_file, {"copy": {"name": "Pauli"}})
    assert mc.main(["patch", str(folder), "--file", str(patch_file)]) == 0
    assert (folder / "project.json").read_bytes() == before


@pytest.mark.parametrize("patch, message", [
    ({"clip": {"fps": 60.0}}, "ERROR A patch can't touch clip or status"),
    ({"status": None}, "ERROR A patch can't touch clip or status"),
    # (a good edit next to a status is refused as a whole: nothing is saved)
    ({"copy": {"name": "Pauli"}, "status": {"done_at": "2999-01-01T00:00:00+00:00"}},
     "ERROR A patch can't touch clip or status"),
    ({"timing": {"hold": 99}}, "ERROR timing.hold must be a number from 1.0 to 6.0, got 99"),
    ({"style": {"hero": "everything"}}, "ERROR hero must be one of none, burst, stats, hud, got 'everything'"),
    ({"matte": {"edge": 1.5}}, "ERROR matte.edge must be a whole number from -4 to 4, got 1.5"),
    ({"freeze": {"frame": 500}}, "ERROR The freeze frame must be between 0 and 119: the clip has 120 frames."),
])
def test_cli_patch_that_cannot_be_applied_is_an_error_and_saves_nothing(
    fake_project, tmp_path, capsys, patch, message
):
    folder, _ = fake_project
    before = (folder / "project.json").read_bytes()

    fails(["patch", folder, "--file", write_json(tmp_path / "patch.json", patch)], capsys, message)

    assert (folder / "project.json").read_bytes() == before
    assert not list(folder.glob("*.tmp"))  # (no temp file of the save, whatever its name)


def test_cli_patch_says_what_is_wrong_with_the_patch_file(fake_project, tmp_path, capsys):
    folder, _ = fake_project
    not_json = tmp_path / "broken.json"
    not_json.write_text("{ not whole", encoding="utf-8")
    a_list = write_json(tmp_path / "list.json", [1, 2])
    bom = tmp_path / "bom.json"  # a file saved by a program that puts a byte order mark first
    bom.write_bytes(b"\xef\xbb\xbf" + json.dumps({"copy": {"name": "Bom"}}).encode("utf-8"))

    fails(["patch", folder, "--file", tmp_path / "nothing.json"], capsys, "ERROR There is no patch file at")
    fails(["patch", folder, "--file", not_json], capsys, "ERROR The patch file")
    fails(["patch", folder, "--file", a_list], capsys, f"ERROR The patch file {a_list} must hold one JSON object.")
    fails(["patch", tmp_path / "no project here", "--file", bom], capsys, "ERROR There is no project in")
    assert mc.main(["patch", str(folder), "--file", str(bom)]) == 0
    assert project_file.load(folder)["copy"]["name"] == "Bom"
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:  # a command line that makes no sense is argparse's 2
        mc.main(["patch", str(folder)])
    assert stopped.value.code == 2


def test_cli_card_prints_a_flag_for_each_note_that_does_not_fit(fake_project, tmp_path, capsys):
    folder, _ = fake_project
    out = tmp_path / "card.png"
    where = f"{out.resolve()}\n"

    # The sample copy fits the fake person cleanly: the path alone
    assert mc.main(["card", str(folder), "--out", str(out)]) == 0
    assert capsys.readouterr().out == where

    # A note too long for its box is cut. It clashes as well, and the line says cut. (The notes that
    # have to make room for it may clash too.)
    change_project(folder, lambda project: project["copy"]["notes"].update(tr="word " * 60))
    assert mc.main(["card", str(folder), "--out", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == str(out.resolve()) and "FLAG tr cut" in lines and "FLAG tr clash" not in lines
    assert all(re.fullmatch(r"FLAG (tl|tr|bl|br) (cut|clash)", line) for line in lines[1:])

    # Two notes pinned on top of each other clash
    def pinned(project):
        project["copy"]["notes"]["tr"] = "@your_handle"
        project["positions"].update(tl=[300, 100], bl=[310, 110])

    change_project(folder, pinned)
    assert mc.main(["card", str(folder), "--out", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == str(out.resolve()) and sorted(lines[1:]) == ["FLAG bl clash", "FLAG tl clash"]


def test_cli_refine_cuts_the_frames_around_the_freeze_again(fake_project, fake_frames, fake_matte, monkeypatch, capsys):
    folder, _ = fake_project
    for frame in fake_frames.glob("*.jpg"):  # the fake project has two frames; give it them all
        (folder / "frames" / frame.name).write_bytes(frame.read_bytes())
    runner, checked = fake_matte(), []
    monkeypatch.setattr(matte, "ensure_model", lambda *args, **kwargs: checked.append(True) or Path("model.onnx"))
    monkeypatch.setattr(matte, "run_rvm", runner)
    with Image.open(folder / "matte" / "000060.png") as image:
        area = analyze.person_boxes(from_image(image.convert("L")))["area"]

    assert mc.main(["refine", str(folder), "--detail", "high"]) == 0

    assert capsys.readouterr().out == "REFINED 13 frames\n"
    (ratio, start, end), = runner.calls
    assert (start, end) == (48, 61)  # from 12 frames before the freeze to the one after it
    assert ratio == pytest.approx(matte.downsample_for(1080, 608, "high", area), abs=1e-3)
    assert project_file.load(folder)["matte"]["detail"] == "high"
    assert checked == [True]  # the model was looked for first

    # Near the start of the clip the run begins at frame 0
    change_project(folder, lambda project: project["freeze"].update(frame=3))
    assert mc.main(["refine", str(folder), "--detail", "max"]) == 0
    assert capsys.readouterr().out == "REFINED 4 frames\n"
    assert runner.calls[-1][1:] == (0, 4)
    assert project_file.load(folder)["matte"]["detail"] == "max"


def test_cli_refine_that_fails_changes_nothing(fake_project, fake_matte, monkeypatch, capsys):
    folder, _ = fake_project
    runner = fake_matte()
    monkeypatch.setattr(matte, "run_rvm", runner)
    monkeypatch.setattr(matte, "ensure_model", lambda *args, **kwargs: Path("model.onnx"))

    fails(["refine", folder, "--detail", "huge"], capsys, 'ERROR The detail has to be "auto", "high" or "max".')
    assert runner.calls == []

    def missing(*args, **kwargs):
        raise matte.ModelMissing("The matte model is missing: put it here.")

    monkeypatch.setattr(matte, "ensure_model", missing)  # said before anything is cut
    fails(["refine", folder, "--detail", "high"], capsys, "ERROR The matte model is missing: put it here.")
    assert runner.calls == []

    def stops(*args, **kwargs):
        raise RuntimeError("The cutout stopped.")

    monkeypatch.setattr(matte, "ensure_model", lambda *args, **kwargs: Path("model.onnx"))
    monkeypatch.setattr(matte, "run_rvm", stops)
    fails(["refine", folder, "--detail", "high"], capsys, "ERROR The cutout stopped.")

    assert project_file.load(folder)["matte"]["detail"] == "auto"  # the detail is saved when the cutout is made
    with pytest.raises(SystemExit) as stopped:  # --detail has to be given
        mc.main(["refine", str(folder)])
    assert stopped.value.code == 2


def test_cli_style_saves_the_recipe_and_answers_the_open_request(fake_project, tmp_path, capsys):
    folder, project = fake_project
    assert palette.text_contrast(SUNSET) >= 3.0
    asked = server.Studio(folder).ask_for_a_style("a sunset arcade look", None)["style_request"]
    assert asked["answered_at"] is None
    recipe_file = write_json(tmp_path / "style.json", SUNSET)
    before = datetime.now(timezone.utc)

    assert mc.main(["style", str(folder), "--file", str(recipe_file)]) == 0

    after = datetime.now(timezone.utc)
    assert capsys.readouterr().out.splitlines() == [
        f"CONTRAST {palette.floor_contrast(palette.text_contrast(SUNSET))}", "STYLE Sunset Arcade"]
    saved = project_file.load(folder)
    # (the file's recipe has the first three extras only: the newer ones are saved, off)
    assert saved["style"] == recipes.complete(SUNSET) and saved["style_source"] == "custom"
    assert saved["style"]["extras"] == {**SUNSET["extras"], **{key: False for key in recipes.EXTRAS[3:]}}
    assert saved["copy"] == project["copy"] and saved["positions"] == project["positions"]  # nothing else moved
    request = saved["style_request"]
    answered = request.pop("answered_at")
    assert request == {key: value for key, value in asked.items() if key != "answered_at"}  # the ask is as it was
    assert STAMP.fullmatch(answered)  # stamped as the studio stamps: UTC, to the millisecond
    assert before - timedelta(milliseconds=1) <= wait.parse_time(answered) <= after

    # The answer is not an event of its own. A waiter that counts from the request is not woken by it
    # (SKILL.md restarts the waiter that way), and is by the next request, which is later than the answer
    with pytest.raises(TimeoutError):
        wait.wait_event(folder, since=asked["at"], timeout=0)
    time.sleep(0.05)
    again = server.Studio(folder).ask_for_a_style("something calmer", None)["style_request"]
    assert wait.wait_event(folder, since=answered, poll=0.05, timeout=5) == ("STYLE", again["at"])


def test_cli_style_answers_with_the_servers_own_stamp(fake_project, tmp_path, monkeypatch):
    folder, _ = fake_project
    server.Studio(folder).ask_for_a_style("anything", None)
    monkeypatch.setattr(server, "now_stamp", lambda: "2999-01-02T03:04:05.678+00:00")

    assert mc.main(["style", str(folder), "--file", str(write_json(tmp_path / "style.json", SUNSET))]) == 0

    assert project_file.load(folder)["style_request"]["answered_at"] == "2999-01-02T03:04:05.678+00:00"


def test_cli_style_leaves_alone_a_request_that_is_answered_or_missing(fake_project, tmp_path):
    folder, _ = fake_project
    recipe_file = write_json(tmp_path / "style.json", SUNSET)

    # No request: the style is saved, and nothing is made up
    assert project_file.load(folder)["style_request"] is None
    assert mc.main(["style", str(folder), "--file", str(recipe_file)]) == 0
    assert project_file.load(folder)["style_request"] is None

    # A request that is answered keeps its answer
    answered = {"text": "x", "image": None, "at": LONG_AGO, "answered_at": "2020-01-01T00:00:05.000+00:00"}
    change_project(folder, lambda project: project.update(style_request=dict(answered)))
    assert mc.main(["style", str(folder), "--file", str(recipe_file)]) == 0
    assert project_file.load(folder)["style_request"] == answered

    # A request that has no answered_at at all (a patch deleted it) is open
    change_project(folder, lambda project: project.update(style_request={"text": "x", "at": LONG_AGO}))
    assert mc.main(["style", str(folder), "--file", str(recipe_file)]) == 0
    request = project_file.load(folder)["style_request"]
    assert request["at"] == LONG_AGO and STAMP.fullmatch(request["answered_at"])


def test_cli_style_refuses_what_is_not_a_recipe_or_not_readable_on_a_phone(fake_project, tmp_path, capsys):
    folder, _ = fake_project
    server.Studio(folder).ask_for_a_style("anything", None)
    before = (folder / "project.json").read_bytes()
    low = dict(SUNSET, notes="marker", palette=dict(SUNSET["palette"], ink=SUNSET["palette"]["dark"]))
    two_heroes = dict(SUNSET, hero=["burst", "hud"])  # a layer takes one id
    not_a_color = dict(SUNSET, palette=dict(SUNSET["palette"], ink="yellow"))
    refused = [
        (low, "ERROR Text contrast 1.0:1 is hard to read on a phone. Change the text or background colors."),
        (dict(SUNSET, hero="everything"), "ERROR hero must be one of none, burst, stats, hud, got 'everything'"),
        (two_heroes, "ERROR hero must be one of none, burst, stats, hud, got ['burst', 'hud']"),
        (not_a_color, "ERROR palette.ink must be a color like #rrggbb, got 'yellow'"),
        ({"id": "x"}, "ERROR style recipe is missing: name, bg, edge, notes, arrows, hero, extras, palette"),
        ([SUNSET], "ERROR The style file"),
    ]

    for recipe, message in refused:
        fails(["style", folder, "--file", write_json(tmp_path / "style.json", recipe)], capsys, message)
        assert (folder / "project.json").read_bytes() == before  # no style, and the request is still open

    fails(["style", folder, "--file", tmp_path / "nothing.json"], capsys, "ERROR There is no style file at")
    fails(["style", tmp_path / "no project here", "--file", tmp_path / "style.json"], capsys,
          "ERROR There is no project in")


def test_cli_style_prints_its_name_on_one_line(fake_project, tmp_path, capsys):
    # "STYLE <name>" is a line that Claude reads, so a name with a line break, a tab or a run of spaces in it is
    # still one line, with single spaces. The recipe itself is saved as it was written
    folder, _ = fake_project
    written = dict(SUNSET, name="  Sunset \n\tArcade\r\n  Two  ")

    assert mc.main(["style", str(folder), "--file", str(write_json(tmp_path / "style.json", written))]) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"CONTRAST {palette.floor_contrast(palette.text_contrast(SUNSET))}", "STYLE Sunset Arcade Two"]
    assert project_file.load(folder)["style"]["name"] == written["name"]


def test_cli_style_and_patch_show_the_contrast_floored_in_their_sentences(fake_project, tmp_path, capsys, monkeypatch):
    # The guard turns down under 3.0, so a 2.95 reads "2.9:1", never "3.0:1 is hard to read"
    folder, _ = fake_project
    monkeypatch.setattr(palette, "text_contrast", lambda recipe: 2.95)
    before = (folder / "project.json").read_bytes()

    fails(["style", folder, "--file", write_json(tmp_path / "style.json", SUNSET)], capsys,
          "ERROR Text contrast 2.9:1 is hard to read on a phone. Change the text or background colors.")
    assert (folder / "project.json").read_bytes() == before

    monkeypatch.setattr(project_file, "text_contrast", lambda recipe: 2.95)  # the one validate asks, for a patch
    patch_file = write_json(tmp_path / "patch.json", {"copy": {"name": "Pauli"}})
    assert mc.main(["patch", str(folder), "--file", str(patch_file)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "WARNING Text contrast 2.9:1 is hard to read on a phone." in lines
    assert not any("3.0:1" in line for line in lines)


def test_cli_contrast_line_is_floored_the_way_the_page_shows_it(fake_project, tmp_path, capsys, monkeypatch):
    # A ratio of 9.16 is 9.1:1 on the studio's page, which cuts it. The line printed 9.2, which rounds it.
    folder, _ = fake_project
    monkeypatch.setattr(palette, "text_contrast", lambda recipe: 9.16)

    patch_file = write_json(tmp_path / "patch.json", {"copy": {"name": "Pauli"}})
    assert mc.main(["patch", str(folder), "--file", str(patch_file)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "CONTRAST 9.1"

    assert mc.main(["style", str(folder), "--file", str(write_json(tmp_path / "style.json", SUNSET))]) == 0
    assert capsys.readouterr().out.splitlines() == ["CONTRAST 9.1", "STYLE Sunset Arcade"]

    # and a ratio of exactly 3.0, the least the guard lets through, still reads 3.0
    monkeypatch.setattr(palette, "text_contrast", lambda recipe: 3.0)
    assert mc.main(["style", str(folder), "--file", str(write_json(tmp_path / "style.json", SUNSET))]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "CONTRAST 3.0"


def test_cli_card_has_no_flag_for_the_game_huds_labels_only_one_note(fake_project, tmp_path, capsys, monkeypatch):
    # A FLAG line names a key to move with positions, and the HUD's labels have none (the patch is refused). So
    # the labels are not flagged: one line says there is no room, and what to do about it.
    folder, _ = fake_project
    out = tmp_path / "card.png"
    note = "NOTE The Game HUD has no room here; pick another hero if it covers something."

    def hud_and_a_note_on_its_label(project):
        project["style"] = dict(SUNSET)
        project["positions"]["tl"] = [43, 43]  # on top of "PLAYER 1"

    # with room, the card says nothing but where it is
    change_project(folder, lambda project: project.update(style=dict(SUNSET)))
    assert mc.main(["card", str(folder), "--out", str(out)]) == 0
    assert capsys.readouterr().out == f"{out.resolve()}\n"

    # the note that was put on the label is flagged, as any note is, and the label is one line of its own
    change_project(folder, hud_and_a_note_on_its_label)
    assert mc.main(["card", str(folder), "--out", str(out)]) == 0
    assert capsys.readouterr().out.splitlines() == [str(out.resolve()), "FLAG tl clash", note]

    # both labels in the way is still one line, and a label that is "cut" is not a flag either
    boxes = card.elements(folder, project_file.load(folder))
    crowded = {key: replace(box, clash=True, cut=key == "hud_top") if key.startswith("hud") else box
               for key, box in boxes.items()}
    monkeypatch.setattr(card, "elements", lambda *args: crowded)
    assert mc.main(["card", str(folder), "--out", str(out)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed.count(note) == 1 and not [line for line in printed if "hud" in line]


def test_cli_style_marks_only_the_request_it_was_asked_about(fake_project, tmp_path, capsys):
    # Claude works on a request for a minute. The person can ask again (the new request replaces the old one,
    # with a later stamp) or cancel meanwhile, and the style that was made for the first must not answer that.
    folder, _ = fake_project
    recipe_file = write_json(tmp_path / "style.json", SUNSET)
    contrast = f"CONTRAST {palette.floor_contrast(palette.text_contrast(SUNSET))}"
    note = "NOTE The request changed; the page shows the newest one."
    first = server.Studio(folder).ask_for_a_style("a sunset arcade look", None)["style_request"]
    time.sleep(0.01)
    second = server.Studio(folder).ask_for_a_style("something calmer", None)["style_request"]
    assert second["at"] > first["at"]

    # asked again: the style is saved, the new request stays open, and the line says so
    assert mc.main(["style", str(folder), "--file", str(recipe_file), "--answers", first["at"]]) == 0
    assert capsys.readouterr().out.splitlines() == [contrast, "STYLE Sunset Arcade", note]
    saved = project_file.load(folder)
    assert saved["style"] == recipes.complete(SUNSET) and saved["style_source"] == "custom"
    assert saved["style_request"] == second and saved["style_request"]["answered_at"] is None

    # the stamp of the request that is open answers it, with no note, whatever way the time is written
    elsewhere = wait.parse_time(second["at"]).astimezone(timezone(timedelta(hours=2)))
    assert mc.main(["style", str(folder), "--file", str(recipe_file),
                    "--answers", elsewhere.isoformat(timespec="milliseconds")]) == 0
    assert capsys.readouterr().out.splitlines() == [contrast, "STYLE Sunset Arcade"]
    assert STAMP.fullmatch(project_file.load(folder)["style_request"]["answered_at"])

    # cancelled (the request is gone): saved, and nothing is made up in its place
    change_project(folder, lambda project: project.update(style_request=None))
    assert mc.main(["style", str(folder), "--file", str(recipe_file), "--answers", second["at"]]) == 0
    assert capsys.readouterr().out.splitlines() == [contrast, "STYLE Sunset Arcade", note]
    assert project_file.load(folder)["style_request"] is None

    # a request that was answered already is left as it is: no second answer, and nothing changed to say
    answered = {"text": "x", "image": None, "at": LONG_AGO, "answered_at": "2020-01-01T00:00:05.000+00:00"}
    change_project(folder, lambda project: project.update(style_request=dict(answered)))
    assert mc.main(["style", str(folder), "--file", str(recipe_file), "--answers", LONG_AGO]) == 0
    assert capsys.readouterr().out.splitlines() == [contrast, "STYLE Sunset Arcade"]
    assert project_file.load(folder)["style_request"] == answered

    # without --answers it is what it was: the request that is open is the one answered
    asked = server.Studio(folder).ask_for_a_style("anything", None)["style_request"]
    assert mc.main(["style", str(folder), "--file", str(recipe_file)]) == 0
    assert capsys.readouterr().out.splitlines() == [contrast, "STYLE Sunset Arcade"]
    assert STAMP.fullmatch(project_file.load(folder)["style_request"]["answered_at"])
    assert project_file.load(folder)["style_request"]["at"] == asked["at"]


def test_cli_style_answers_must_be_a_time(fake_project, tmp_path, capsys):
    folder, _ = fake_project
    server.Studio(folder).ask_for_a_style("anything", None)
    before = (folder / "project.json").read_bytes()

    for stamp in ("that one", ""):
        fails(["style", folder, "--file", write_json(tmp_path / "style.json", SUNSET), "--answers", stamp], capsys,
              "ERROR --answers must be a time like 2026-10-26T05:00:00Z.")
    assert (folder / "project.json").read_bytes() == before  # nothing is saved for a stamp that is not one


def test_cli_patch_and_style_run_as_programs(fake_project, tmp_path):
    folder, _ = fake_project

    done = run_mc("patch", folder, "--file", write_json(tmp_path / "patch.json", {"copy": {"name": "Pauli"}}))
    assert done.returncode == 0 and done.stderr == ""
    *warnings, contrast = done.stdout.splitlines()  # (the sample's br note is over its soft limit: a warning)
    assert re.fullmatch(r"CONTRAST \d+\.\d", contrast) and all(line.startswith("WARNING ") for line in warnings)
    refused = run_mc("patch", folder, "--file", write_json(tmp_path / "patch.json", {"status": {}}))
    assert refused.returncode == 1 and refused.stdout == ""
    assert refused.stderr.startswith("ERROR A patch can't touch clip or status")

    done = run_mc("style", folder, "--file", write_json(tmp_path / "style.json", SUNSET))
    assert done.returncode == 0 and done.stderr == ""
    assert re.fullmatch(r"CONTRAST \d+\.\d\nSTYLE Sunset Arcade", done.stdout.strip())
    assert project_file.load(folder)["style"] == recipes.complete(SUNSET)
