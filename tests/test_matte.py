"""Tests for the matte: the downsample ratio, the model check, the run over a folder of frames, and
the cleanup and edge controls that every loaded matte goes through.

The cleanup and edge tests start from person_alpha, a matte of a person in a close shot, and draw
specks and holes into it. A real matte of a test pattern cannot say which frame a matte came from,
which frames warmed the state, or what the alpha floor did, so run_rvm is also tried on flat gray
frames with a stand-in for the onnxruntime session. The real model then checks that a run leaves a
frame-sized PNG for every frame.
"""

import hashlib
import io
import os
import ssl
import subprocess
import time
import urllib.error
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mclib import matte
from mclib.config import MODEL_PATH, MODEL_SHA256, MODEL_URL
from mclib.matte import (
    ModelMissing,
    adjust_edge,
    clean_alpha,
    downsample_for,
    ensure_model,
    load_alpha,
    matte_path,
    run_rvm,
)
from mclib.raster import blur

# Flat frames for the stand-in: frame n is gray level 40 + 6 n, so a level says which frame it was.
FLAT_LEVELS = [40 + 6 * n for n in range(30)]


def write_flat_frames(folder, levels):
    """Tiny JPEGs named 000000.jpg on, frame n flat gray at levels[n] (quality 100 is exact)."""
    folder.mkdir()
    for n, level in enumerate(levels):
        image = Image.new("RGB", (32, 24), (level,) * 3)
        image.save(folder / f"{n:06d}.jpg", quality=100, subsampling=0)
    return folder


def frame_number(level):
    """The frame a gray level in FLAT_LEVELS stands for."""
    return (level - 40) // 6


def png_level(path):
    """The gray level of a flat matte PNG."""
    with Image.open(path) as image:
        pixels = np.asarray(image)
    assert (pixels == pixels[0, 0]).all()
    return int(pixels[0, 0])


class StandIn:
    """Takes the place of the onnxruntime session. The alpha it returns is the frame's mean
    brightness, so a matte shows which frame it came from. It counts its steps in the first
    recurrent state, so a state that was reset or skipped a step shows too. calls lists
    (gray level of the frame, steps taken before it) in the order the frames were run."""

    def __init__(self):
        self.calls = []
        self.ratios = set()

    def run(self, names, feeds):
        src, steps = feeds["src"], feeds["r1i"]
        if not self.calls:  # the state starts as zeros of shape (1, 1, 1, 1)
            assert steps.shape == (1, 1, 1, 1) and steps.dtype == np.float32
        self.calls.append((round(float(src.mean()) * 255), int(steps.item())))
        self.ratios.add(float(feeds["downsample_ratio"][0]))
        outputs = {
            "pha": np.full((1, 1, *src.shape[2:]), src.mean(), np.float32),
            "r1o": steps + 1, "r2o": feeds["r2i"], "r3o": feeds["r3i"], "r4o": feeds["r4i"],
        }
        return [outputs[name] for name in names]


# --- downsample_for ---------------------------------------------------------------------------


def test_downsample_for_values():
    # auto: the picture the model sees is 576 px on its long side
    assert downsample_for(1080, 608, "auto") == pytest.approx(0.5333, abs=1e-4)
    assert downsample_for(1080, 1920, "auto") == pytest.approx(0.3)
    assert downsample_for(1080, 1920) == pytest.approx(0.3)  # auto is the default
    # a person who fills the frame (more than 0.45 of it) gets at least 0.5, never less than auto
    assert downsample_for(1080, 1920, "auto", 0.5) == pytest.approx(0.5)
    assert downsample_for(1080, 1920, "auto", 0.45) == pytest.approx(0.3)
    assert downsample_for(1080, 608, "auto", 0.9) == pytest.approx(0.5333, abs=1e-4)
    # high and max multiply auto, boost included, and stop at 1
    assert downsample_for(1080, 1920, "high") == pytest.approx(0.48)
    assert downsample_for(1080, 1920, "max") == pytest.approx(0.72)
    assert downsample_for(1080, 1920, "high", 0.6) == pytest.approx(0.8)
    assert downsample_for(1080, 608, "max") == 1.0
    assert downsample_for(400, 300, "auto") == 1.0
    with pytest.raises(ValueError, match="detail"):
        downsample_for(1080, 608, "ultra")


# --- clean_alpha ------------------------------------------------------------------------------


def test_clean_alpha_removes_specks(person_alpha):
    person = person_alpha(1080, 608)
    marked = person.copy()
    marked[20:23, 1000:1003] = 1.0  # a 3x3 speck far from the person
    marked[560:584, 40:64] = 0.9  # a 24x24 blob, small next to a person
    # Faint detail by the person's left edge, whose 0.5 contour is at x = 205: alpha 2 px away
    # keeps most of its alpha, the same 20 px away goes.
    marked[540:580, 202:204] = 0.3
    marked[540:580, 183:185] = 0.3
    untouched = marked.copy()

    out = clean_alpha(marked)

    assert out.dtype == np.float32 and out.shape == marked.shape
    assert np.array_equal(marked, untouched)  # the input is left alone
    # Far from the person everything goes to exactly 0.
    assert out[20:23, 1000:1003].max() == 0
    assert out[560:584, 40:64].max() == 0
    assert out[540:580, 183:185].max() == 0
    assert (out[540:580, 202:204] > 0.7 * 0.3).all()  # the cut fades in only near its edge
    # The person is as it was: exactly inside, and within a few percent of alpha at the soft edge.
    assert np.array_equal(out[450:600, 300:800], person[450:600, 300:800])
    assert np.array_equal(out[150:300, 500:580], person[150:300, 500:580])
    elsewhere = np.ones(person.shape, bool)
    elsewhere[540:580, 183:204] = False  # the detail above
    assert np.abs(out - person)[elsewhere].max() < 0.05

    # A matte that fades out slowly, as RVM's does around dark hair, is cut without a step: where
    # the source itself is smooth, neighboring pixels never differ by much.
    hazy = blur(person, 10)
    hazy_out = clean_alpha(hazy)
    assert (hazy - hazy_out).max() > 0.1  # something is cut, or this proves nothing
    for axis in (0, 1):
        source_step = np.abs(np.diff(hazy, axis=axis))
        step = np.abs(np.diff(hazy_out, axis=axis))
        assert step[source_step < 0.06].max() < 0.15

    # A size that is not a multiple of the 4 px cell works too.
    odd = person_alpha(1081, 607)
    assert clean_alpha(odd).shape == odd.shape
    assert np.abs(clean_alpha(odd) - odd).max() < 0.05


def test_clean_alpha_fills_holes(person_alpha):
    person = person_alpha(1080, 608)
    rows, cols = np.mgrid[0:608, 0:1080]
    hole = (slice(431, 451), slice(523, 543))  # 20x20, off the 4 px grid the cleanup uses
    with_rim = (slice(427, 455), slice(519, 547))  # the hole and a ring one cell (4 px) wide
    # An enclosed gap 4% of the person's size, like the loop between an arm and the torso with a
    # hand on the hip, and a rim of its own 4 px wide.
    loop = ((cols - 760) / 32) ** 2 + ((rows - 500) / 80) ** 2 <= 1
    loop_rim = ((cols - 760) / 36) ** 2 + ((rows - 500) / 84) ** 2 <= 1
    assert 0.035 < loop.sum() / (person >= 0.5).sum() < 0.045

    holed = person.copy()
    holed[with_rim] = 0.7  # a soft rim: see-through, but above the 0.5 that counts as person
    holed[hole] = 0.0
    holed[500:512, 700:712] = 0.3  # a see-through patch
    holed[548:608, 600:620] = 0.0  # a gap that runs off the bottom of the frame: open, not a hole
    holed[loop_rim] = 0.7
    holed[loop] = 0.0

    out = clean_alpha(holed)

    assert out[with_rim].min() >= 0.99  # the hole and its rim, corners too
    assert out[500:512, 700:712].min() >= 0.99
    assert out[548:608, 600:620].max() == 0
    # The loop is background seen through the cutout, not a hole in it: it stays open, and its
    # rim is not grown over.
    assert out[loop].max() <= 1 / 255
    assert np.allclose(out[loop_rim & ~loop], 0.7, atol=0.05)
    # Nothing else changed beyond a few percent of alpha at the soft edge: the person with only
    # the open gaps cut out of it.
    expected = person.copy()
    expected[548:608, 600:620] = 0.0
    expected[loop_rim] = 0.7
    expected[loop] = 0.0
    assert np.abs(out - expected).max() < 0.05

    # A hole that touches the outside only at a corner is filled, but that background is not
    # pulled into the person. On the 4 px cells: solid all over, one empty cell (the hole) at
    # (8, 8), and an empty row from (9, 9) to the right edge, which meets the hole diagonally.
    pinched = np.ones((64, 64), np.float32)
    pinched[32:36, 32:36] = 0.0
    pinched[36:40, 36:] = 0.0
    out = clean_alpha(pinched)
    assert out[32:36, 32:36].min() >= 0.99
    assert out[36:40, 36:40].max() == 0


# --- adjust_edge and load_alpha ---------------------------------------------------------------


def test_adjust_edge_shrinks_and_softens(person_alpha, tmp_path):
    alpha = person_alpha(1080, 608)
    area = alpha.sum()

    def steepest(a):
        return max(np.abs(np.diff(a, axis=0)).max(), np.abs(np.diff(a, axis=1)).max())

    assert adjust_edge(alpha, 0) is alpha
    shrunk, soft = adjust_edge(alpha, -3), adjust_edge(alpha, 3)
    for result in (shrunk, soft):
        assert result.dtype == np.float32 and result.shape == alpha.shape
    # 3 px off an outline of about 1800 px is some 2.7% of the area: it shrinks, but by that much
    assert 0.95 < shrunk.sum() / area < 0.985
    assert abs(soft.sum() - area) < 0.05 * area
    assert steepest(soft) < 0.75 * steepest(alpha)

    # load_alpha reads matte/000123.png for frame 123, cleans it, then applies the project's edge.
    project_dir = tmp_path / "project"
    drawn = alpha.copy()
    drawn[20:23, 1000:1003] = 1.0  # a speck for the cleanup to take out
    drawn[431:451, 523:543] = 0.0  # and a hole for it to fill
    png = matte_path(project_dir, 123)
    assert png == project_dir / "matte" / "000123.png"
    assert matte_path(project_dir, 7).name == "000007.png"
    png.parent.mkdir(parents=True)
    Image.fromarray(np.rint(drawn * 255).astype(np.uint8)).save(png)
    with Image.open(png) as image:
        stored = np.asarray(image, np.float32) / 255
    cleaned = clean_alpha(stored)
    assert not np.array_equal(cleaned, stored)
    for edge in (0, -3, 3):
        loaded = load_alpha(project_dir, {"matte": {"edge": edge}}, 123)
        assert np.array_equal(loaded, adjust_edge(cleaned, edge))
    assert np.array_equal(load_alpha(project_dir, {"matte": {}}, 123), cleaned)  # no edge: 0


# --- the model --------------------------------------------------------------------------------


def test_ensure_model_missing(tmp_path):
    missing = tmp_path / "rvm" / "model.onnx"
    with pytest.raises(ModelMissing) as caught:
        ensure_model(missing)
    message = str(caught.value)
    assert str(missing) in message
    assert "15 MB" in message
    assert MODEL_URL in message
    assert isinstance(caught.value, RuntimeError)

    present = tmp_path / "model.onnx"
    present.write_bytes(b"not a real model")
    assert ensure_model(present) == present


# --- downloading the model --------------------------------------------------------------------
#
# Nothing here reaches the web: urlopen, curl and the rename are stood in for, and the model is a few
# hundred bytes whose SHA-256 is handed to download_model (the real pin is checked on its own, below).

FAKE_MODEL = b"not the real model, but it is the same bytes every time " * 20
FAKE_SHA256 = hashlib.sha256(FAKE_MODEL).hexdigest()
URL = "https://example.invalid/releases/model.onnx"
CERTIFICATE = ("Python couldn't check the download's certificate. On a Mac with Python from python.org, "
               'run "Install Certificates.command" in the Python folder, then try again.')


class Download(io.BytesIO):
    """What urlopen gives: a file of the bytes that works in a with statement."""

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()


def cert_error():
    """What urllib raises on a Mac whose Python has no certificates: a URLError holding the SSL error."""
    return urllib.error.URLError(ssl.SSLCertVerificationError(
        1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate"))


@pytest.fixture
def small_model(monkeypatch):
    """The size floor lowered so that the stand-in model is big enough, and no curl on this machine."""
    monkeypatch.setattr(matte, "_MIN_MODEL_BYTES", 100)
    monkeypatch.setattr(matte.shutil, "which", lambda name, *args, **kwargs: None)


def serve(monkeypatch, content=FAKE_MODEL):
    """Make urlopen give `content`, and return the list of the (url, timeout) it was asked for."""
    asked = []

    def urlopen(url, timeout=None):
        asked.append((url, timeout))
        return Download(content)

    monkeypatch.setattr(matte.urllib.request, "urlopen", urlopen)
    return asked


def test_download_model_keeps_the_file_after_checking_its_hash(tmp_path, monkeypatch, small_model):
    asked = serve(monkeypatch)
    dest = tmp_path / "rvm" / "model.onnx"

    assert matte.download_model(dest, URL, FAKE_SHA256) == dest

    assert dest.read_bytes() == FAKE_MODEL
    assert [path.name for path in dest.parent.iterdir()] == ["model.onnx"]  # no .part left
    assert asked == [(URL, 60)]


def test_download_model_refuses_a_file_that_is_not_the_model(tmp_path, monkeypatch, small_model):
    dest = tmp_path / "model.onnx"

    # an error page, a few KB, is too small
    monkeypatch.setattr(matte, "_MIN_MODEL_BYTES", 10_000)
    serve(monkeypatch, b"<html>Not Found</html>")
    with pytest.raises(ModelMissing, match=r"was only 22 bytes, too small for the matte model"):
        matte.download_model(dest, URL, FAKE_SHA256)
    assert list(tmp_path.iterdir()) == []

    # big enough and the wrong bytes: the hash is the pin
    monkeypatch.setattr(matte, "_MIN_MODEL_BYTES", 100)
    serve(monkeypatch, FAKE_MODEL + b"!")
    with pytest.raises(ModelMissing) as caught:
        matte.download_model(dest, URL, FAKE_SHA256)
    message = str(caught.value)
    assert message.startswith(f"The file from {URL} is not the matte model I expected")
    assert FAKE_SHA256[:12] in message and hashlib.sha256(FAKE_MODEL + b"!").hexdigest()[:12] in message
    assert "was not kept" in message and "try the download again" in message.lower()
    assert list(tmp_path.iterdir()) == []  # not the model, and not the .part either

    # and a model that is there already is left alone by a download that fails
    dest.write_bytes(b"the model that was there")
    with pytest.raises(ModelMissing):
        matte.download_model(dest, URL, FAKE_SHA256)
    assert dest.read_bytes() == b"the model that was there" and [p.name for p in tmp_path.iterdir()] == ["model.onnx"]


def test_the_pinned_hash_is_the_official_models():
    assert len(MODEL_SHA256) == 64 and set(MODEL_SHA256) <= set("0123456789abcdef")
    assert matte.download_model.__defaults__[2] == MODEL_SHA256  # the default of the argument
    if MODEL_PATH.is_file():  # (on a machine that has the model: it is the official v1.0.0 file)
        assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256


@pytest.mark.parametrize("raised", ["wrapped", "bare"])
def test_download_model_falls_back_to_curl_when_python_cannot_check_the_certificate(
    tmp_path, monkeypatch, small_model, raised
):
    # python.org's Python on a Mac fails with CERTIFICATE_VERIFY_FAILED. curl uses the system's certificates.
    error = cert_error() if raised == "wrapped" else ssl.SSLCertVerificationError(1, "certificate verify failed")

    def urlopen(url, timeout=None):
        raise error

    monkeypatch.setattr(matte.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(matte.shutil, "which", lambda name, *args, **kwargs: "/usr/bin/curl" if name == "curl" else None)
    dest, commands = tmp_path / "model.onnx", []

    def curl(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(FAKE_MODEL)  # curl writes the file it is told
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(matte.subprocess, "run", curl)

    assert matte.download_model(dest, URL, FAKE_SHA256) == dest

    assert commands == [["/usr/bin/curl", "-L", "--fail", "-o", str(dest) + ".part", URL]]
    assert dest.read_bytes() == FAKE_MODEL and [p.name for p in tmp_path.iterdir()] == ["model.onnx"]


def test_download_model_checks_what_curl_brought_too(tmp_path, monkeypatch, small_model):
    def urlopen(url, timeout=None):
        raise cert_error()

    monkeypatch.setattr(matte.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(matte.shutil, "which", lambda name, *args, **kwargs: "curl")

    def curl(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(FAKE_MODEL + b"?")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(matte.subprocess, "run", curl)

    with pytest.raises(ModelMissing, match="is not the matte model I expected"):
        matte.download_model(tmp_path / "model.onnx", URL, FAKE_SHA256)
    assert list(tmp_path.iterdir()) == []

    # curl that fails says why in one line, and leaves nothing
    def failing_curl(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"<html>")
        return subprocess.CompletedProcess(
            command, 22, "", "  % Total    % Received\ncurl: (22) The requested URL returned error: 404\n")

    monkeypatch.setattr(matte.subprocess, "run", failing_curl)
    with pytest.raises(ModelMissing) as caught:
        matte.download_model(tmp_path / "model.onnx", URL, FAKE_SHA256)
    assert str(caught.value).startswith("curl couldn't download the matte model")
    assert "curl: (22) The requested URL returned error: 404" in str(caught.value)
    assert "Total" not in str(caught.value) and list(tmp_path.iterdir()) == []

    # curl that stalls is a sentence too
    def stalls(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 900)

    monkeypatch.setattr(matte.subprocess, "run", stalls)
    with pytest.raises(ModelMissing, match="curl couldn't download the matte model"):
        matte.download_model(tmp_path / "model.onnx", URL, FAKE_SHA256)


def test_download_model_without_curl_says_how_to_fix_the_certificate(tmp_path, monkeypatch, small_model):
    calls = []

    def urlopen(url, timeout=None):
        raise cert_error()

    monkeypatch.setattr(matte.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(matte.subprocess, "run", lambda *args, **kwargs: calls.append(args))  # (no curl: never run)

    with pytest.raises(ModelMissing) as caught:
        matte.download_model(tmp_path / "model.onnx", URL, FAKE_SHA256)

    assert str(caught.value) == CERTIFICATE
    assert calls == [] and list(tmp_path.iterdir()) == []


def test_download_model_says_what_is_wrong_when_the_web_cannot_be_reached(tmp_path, monkeypatch, small_model):
    monkeypatch.setattr(matte.shutil, "which", lambda name, *args, **kwargs: "curl")
    monkeypatch.setattr(matte.subprocess, "run", lambda *args, **kwargs: pytest.fail("curl is for certificates"))

    for failure, said in (
        (urllib.error.URLError(OSError(11001, "getaddrinfo failed")), "getaddrinfo failed"),
        (urllib.error.HTTPError(URL, 404, "Not Found", {}, None), "Not Found"),
        (TimeoutError("The read operation timed out"), "The read operation timed out"),
        (ConnectionResetError("Connection reset by peer"), "Connection reset by peer"),
    ):
        def urlopen(url, timeout=None, failure=failure):
            raise failure

        monkeypatch.setattr(matte.urllib.request, "urlopen", urlopen)
        with pytest.raises(ModelMissing) as caught:
            matte.download_model(tmp_path / "model.onnx", URL, FAKE_SHA256)
        message = str(caught.value)
        assert message.startswith(f"Couldn't download the matte model from {URL}:"), message
        assert said in message and "internet connection" in message and "Traceback" not in message
        assert list(tmp_path.iterdir()) == []


def test_download_model_waits_out_a_virus_scan_that_holds_the_new_file(tmp_path, monkeypatch, small_model):
    # On Windows, Defender scans the file that was just written and the rename is refused for a moment
    serve(monkeypatch)
    real, refused, pauses = os.replace, [], []

    def held_for_a_moment(source, destination):
        if len(refused) < 3:
            refused.append(destination)
            raise PermissionError(5, "Access is denied")
        return real(source, destination)

    monkeypatch.setattr(os, "replace", held_for_a_moment)
    monkeypatch.setattr(time, "sleep", pauses.append)
    dest = tmp_path / "model.onnx"

    assert matte.download_model(dest, URL, FAKE_SHA256) == dest

    assert dest.read_bytes() == FAKE_MODEL and len(refused) == 3 and pauses == [0.1] * 3

    # a lock that stays gives the sentence that names the file, and the .part is gone
    dest.unlink()

    def held_for_good(source, destination):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(os, "replace", held_for_good)
    pauses.clear()
    with pytest.raises(PermissionError, match=r"model\.onnx is in use by another program"):
        matte.download_model(dest, URL, FAKE_SHA256)
    assert pauses == [0.1] * 9 and list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="RVM model not present")
def test_run_rvm_writes_one_png_per_frame(tmp_path, frames_dir_20, monkeypatch):
    flat = write_flat_frames(tmp_path / "flat", FLAT_LEVELS)
    with monkeypatch.context() as patched:
        stand_in = StandIn()
        patched.setattr(matte, "_open_session", lambda: stand_in)

        # A range in the middle: the 4 frames before it warm the state, then its own frames run
        # on that same state, and each matte is named for the frame it was made from.
        part = tmp_path / "part"
        ticks = []
        done = run_rvm(flat, part, 0.25, start=10, end=13, warmup=4,
                       progress=lambda finished, total: ticks.append((finished, total)))
        assert done == 3
        assert [frame_number(level) for level, _ in stand_in.calls] == [6, 7, 8, 9, 10, 11, 12]
        assert [steps for _, steps in stand_in.calls] == list(range(7))
        assert stand_in.ratios == {0.25}
        assert sorted(path.name for path in part.iterdir()) == [
            "000010.png", "000011.png", "000012.png"]
        made_from = [frame_number(png_level(part / f"{n:06d}.png")) for n in (10, 11, 12)]
        assert made_from == [10, 11, 12]
        assert ticks == [(1, 3), (2, 3), (3, 3)]  # the warm-up frames are not counted

        # From frame 0 the first frames warm the state, then the run starts again at frame 0.
        stand_in.calls.clear()
        assert run_rvm(flat, tmp_path / "head", 0.25, end=2, warmup=3) == 2
        assert [frame_number(level) for level, _ in stand_in.calls] == [0, 1, 2, 0, 1]
        assert [steps for _, steps in stand_in.calls] == list(range(5))

        # Near the start the warm-up stops at frame 0, and a range past the end stops at the end.
        stand_in.calls.clear()
        assert run_rvm(flat, tmp_path / "tail", 0.25, start=28, end=99, warmup=40) == 2
        assert [frame_number(level) for level, _ in stand_in.calls] == [*range(28), 28, 29]
        stand_in.calls.clear()
        assert run_rvm(flat, tmp_path / "none", 0.25, start=30) == 0
        assert stand_in.calls == []

        # Misuse is refused before any work: no frames to read, or a start before frame 0.
        with pytest.raises(FileNotFoundError, match="no frames"):
            run_rvm(tmp_path / "missing", tmp_path / "none", 0.25)
        with pytest.raises(ValueError, match="start"):
            run_rvm(flat, tmp_path / "none", 0.25, start=-1)
        assert stand_in.calls == []

        # The alpha floor: under 0.03 is cut, then it ramps back up to the raw alpha at 0.10.
        # Gray levels 5, 8, 17, 25, 26 and 200 are alpha 0.020, 0.031, 0.067, 0.098, 0.102, 0.784.
        ramp = write_flat_frames(tmp_path / "ramp", [5, 8, 17, 25, 26, 200])
        run_rvm(ramp, tmp_path / "ramp-matte", 0.25, warmup=0)
        levels = [png_level(tmp_path / "ramp-matte" / f"{n:06d}.png") for n in range(6)]
        assert levels == [0, 0, 9, 24, 26, 200]

    # The real model: one PNG for every frame, as big as the frame, in 8-bit gray.
    project_dir = tmp_path / "project"
    ticks.clear()
    written = run_rvm(frames_dir_20, project_dir / "matte", downsample_for(1080, 608),
                      progress=lambda finished, total: ticks.append((finished, total)))
    assert written == 20
    assert ticks == [(n, 20) for n in range(1, 21)]
    pngs = sorted((project_dir / "matte").iterdir())
    assert [png.name for png in pngs] == [f"{n:06d}.png" for n in range(20)]
    for png in pngs:
        with Image.open(png) as image:
            assert (image.mode, image.size) == ("L", (1080, 608))
    assert load_alpha(project_dir, {"matte": {"edge": 2}}, 19).shape == (608, 1080)
