"""Tests for the setup check. They run against this machine, which has what the tests need (ffmpeg,
the three packages, the fonts), and make it lack things one at a time."""

import json
import platform
import re
import shutil
import subprocess

import pytest

from mclib import doctor, media
from mclib.config import FONT_FILES

KEYS = {"python", "packages", "ffmpeg", "ffprobe", "hdr", "libx264", "fonts_missing", "model", "ok"}


@pytest.fixture
def model_present(tmp_path, monkeypatch):
    """The matte model is where it is kept, whether or not it is on this machine."""
    model = tmp_path / "rvm_mobilenetv3_fp32.onnx"
    model.write_bytes(b"not really a model")
    monkeypatch.setattr(doctor, "MODEL_PATH", model)
    return model


def package_versions(monkeypatch, **versions):
    """Make doctor find these package versions (None is a package that cannot be imported), and
    the real ones for any package left out."""
    real = doctor._package_version
    monkeypatch.setattr(
        doctor, "_package_version", lambda name: versions.get(name, real(name)))


# --- what it finds ----------------------------------------------------------------------------


def test_doctor_reports_missing_ffmpeg(monkeypatch):
    real = shutil.which
    monkeypatch.setattr(
        doctor.shutil, "which", lambda name, *args, **kwargs: None if name == "ffmpeg"
        else real(name, *args, **kwargs))

    result = doctor.check()

    assert result["ok"] is False
    assert result["ffmpeg"] is None and result["ffprobe"] is not None
    text = doctor.report(result)
    assert "ffmpeg: not found on the PATH" in text
    assert "ffprobe: not found" not in text
    assert "winget install Gyan.FFmpeg" in text and "brew install ffmpeg" in text
    # what ffmpeg can do is not asked of an ffmpeg that is not there: no more lines to fix
    assert result["hdr"] is None and result["libx264"] is False
    assert "HDR clips: not checked" in text and "libx264: not checked" in text
    assert "ffmpeg-full" not in text and text.count("libx264") == 1  # no fix for either: install ffmpeg


def test_doctor_does_not_ask_a_missing_ffmpeg_what_it_can_do(monkeypatch, model_present):
    real = shutil.which
    monkeypatch.setattr(
        doctor.shutil, "which", lambda name, *args, **kwargs: None if name == "ffmpeg"
        else real(name, *args, **kwargs))

    def never(*args, **kwargs):
        raise AssertionError("ffmpeg is not there to be asked")

    monkeypatch.setattr(media, "hdr_support", never)
    monkeypatch.setattr(media, "has_encoder", never)

    assert doctor.check()["ok"] is False


def test_doctor_is_ok_when_everything_is_there(model_present):
    result = doctor.check()

    assert set(result) == KEYS
    assert result["ok"] is True
    assert set(result["packages"]) == {"numpy", "PIL", "onnxruntime"}
    assert all(isinstance(version, str) and version for version in result["packages"].values())
    assert result["fonts_missing"] == [] and result["model"] is True
    assert result["python"] == platform.python_version()
    json.dumps(result)  # it is all plain data: the CLI prints it as JSON

    # The tools' versions are what their own first line says
    for tool in ("ffmpeg", "ffprobe"):
        first = subprocess.run([tool, "-version"], capture_output=True, text=True,
                               check=True).stdout.splitlines()[0]
        assert result[tool] == re.search(r"version (\S+)", first).group(1)

    lines = doctor.report(result).splitlines()
    assert lines[0] == f"Python {platform.python_version()}: OK"
    assert [line.split(" ")[0] for line in lines[1:4]] == ["numpy", "Pillow", "onnxruntime"]
    assert all(line.endswith(": OK") for line in lines[:3])
    assert lines[-1] == "Everything is ready."
    assert "To fix" not in "\n".join(lines)


def test_doctor_gives_the_exact_fixes(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "MODEL_PATH", tmp_path / "gone.onnx")
    package_versions(monkeypatch, numpy=None, onnxruntime=None)

    result = doctor.check()
    text = doctor.report(result)

    assert result["ok"] is False and result["model"] is False
    assert result["packages"]["numpy"] is None and result["packages"]["onnxruntime"] is None
    assert result["packages"]["PIL"] is not None
    assert "numpy: not installed" in text and "onnxruntime: not installed" in text
    assert "\n  python -m pip install numpy pillow onnxruntime\n" in text
    assert "\n  python mc.py doctor --download-model\n" in text
    assert "matte model: missing (a download of about 15 MB from https://github.com/" in text
    assert "Ask the person" in text  # these install and download things
    # in the order the lines are given: the checks first, then the fixes
    assert text.index("numpy: not installed") < text.index("To fix:") < text.index("python -m pip")


def test_doctor_wants_pillow_9_1_or_newer(monkeypatch, model_present):
    # Image.Resampling and ImageFont.Layout came with 9.1
    package_versions(monkeypatch, PIL="9.0.1")
    result = doctor.check()
    text = doctor.report(result)
    assert result["ok"] is False and result["packages"]["PIL"] == "9.0.1"
    assert "Pillow 9.0.1: 9.1 or newer is needed" in text
    assert "\n  python -m pip install --upgrade pillow\n" in text
    assert "pip install numpy pillow onnxruntime" not in text  # nothing is missing

    for version in ("9.1.0", "9.5.0", "10.4.0", "11.0.0rc1", "12.1.2"):
        package_versions(monkeypatch, PIL=version)
        assert doctor.check()["ok"] is True, version
    package_versions(monkeypatch, PIL="9.0.99")
    assert doctor.check()["ok"] is False
    # a Pillow that does not say what version it is has imported, so it is taken to be new enough
    package_versions(monkeypatch, PIL="of unknown version")
    assert doctor.check()["ok"] is True


def test_doctor_wants_python_3_10_or_newer(monkeypatch, model_present):
    monkeypatch.setattr(doctor.platform, "python_version", lambda: "3.9.7")

    result = doctor.check()

    assert result["ok"] is False and result["python"] == "3.9.7"
    assert doctor.report(result).splitlines()[0] == "Python 3.9.7: 3.10 or newer is needed"
    monkeypatch.setattr(doctor.platform, "python_version", lambda: "3.10.0")
    assert doctor.check()["ok"] is True


def test_doctor_finds_missing_fonts(tmp_path, model_present):
    files = sorted({name for name, _ in FONT_FILES.values()})
    assert len(files) == 6  # the notes' fonts and the HUD's: there is no title, so no title font (Nunito)
    assert "Nunito[wght].ttf" not in files and "hook" not in FONT_FILES
    fonts = tmp_path / "skill" / "fonts"
    fonts.mkdir(parents=True)
    for name in files[:3]:
        (fonts / name).write_bytes(b"x")

    result = doctor.check(skill_dir=tmp_path / "skill")

    assert result["fonts_missing"] == files[3:]
    assert result["ok"] is False
    text = doctor.report(result)
    assert f"fonts: missing {', '.join(files[3:])}" in text
    assert "Copy the whole main-character folder again" in text

    # no fonts folder at all is all of them, and the real skill folder has every one
    assert doctor.check(skill_dir=tmp_path / "nowhere")["fonts_missing"] == files
    assert doctor.check()["fonts_missing"] == []
    assert "fonts: OK, all 6 found" in doctor.report(doctor.check())


def test_doctor_counts_a_package_that_will_not_load_as_missing(monkeypatch, model_present):
    real = doctor.importlib.import_module

    def broken(name, *args, **kwargs):
        if name == "onnxruntime":  # installed, and fails while loading its binary
            raise OSError("DLL load failed while importing onnxruntime_pybind11_state")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(doctor.importlib, "import_module", broken)

    result = doctor.check()

    assert result["packages"]["onnxruntime"] is None and result["ok"] is False
    assert "onnxruntime: not installed" in doctor.report(result)


# --- what ffmpeg can do: HDR clips and H.264 -------------------------------------------------


def test_doctor_has_a_line_for_hdr_clips_and_one_for_libx264(model_present):
    # This machine's ffmpeg is the full build: zscale for HDR, and libx264
    result = doctor.check()

    assert result["hdr"] == "zscale" and result["libx264"] is True and result["ok"] is True
    lines = doctor.report(result).splitlines()
    assert "HDR clips: OK (converted with zscale)" in lines
    assert "libx264: OK" in lines
    # after the two tools, before the fonts
    order = [next(i for i, line in enumerate(lines) if line.startswith(start))
             for start in ("ffprobe", "HDR clips", "libx264", "fonts")]
    assert order == sorted(order)


def test_doctor_takes_ffmpegs_own_color_management_for_hdr(monkeypatch, model_present):
    # Homebrew's ffmpeg has no zscale; from ffmpeg 8 the scale filter converts HDR by itself
    monkeypatch.setattr(media, "has_filter", lambda name: name != "zscale")

    result = doctor.check()

    assert result["hdr"] == "swscale" and result["ok"] is True
    text = doctor.report(result)
    assert "HDR clips: OK (converted by ffmpeg 8 or newer)" in text
    assert "To fix" not in text and text.endswith("Everything is ready.")


def test_doctor_says_what_to_install_when_hdr_cannot_be_converted(monkeypatch, model_present):
    monkeypatch.setattr(media, "has_filter", lambda name: name != "zscale")
    monkeypatch.setattr(media, "_swscale_tone_maps", lambda: False)

    result = doctor.check()

    assert result["hdr"] is None and result["ok"] is False
    text = doctor.report(result)
    assert "HDR clips: this ffmpeg can't convert them (an iPhone shoots HDR by default)" in text
    fix = text.split("To fix:")[1]
    assert "winget install Gyan.FFmpeg" in fix  # Windows: the full build
    assert "brew install ffmpeg-full" in fix and "PATH" in fix  # a Mac: the keg on the PATH
    assert "ffmpeg 8" in fix  # or a version that does it itself
    assert "libx264" not in text.split("To fix:")[0].replace("libx264: OK", "")


def test_doctor_wants_an_ffmpeg_with_libx264(monkeypatch, model_present):
    # An LGPL build of ffmpeg passes every other check and fails at the first preview
    monkeypatch.setattr(media, "has_encoder", lambda name: name != "libx264")

    result = doctor.check()

    assert result["libx264"] is False and result["ok"] is False and result["hdr"] == "zscale"
    text = doctor.report(result)
    assert "libx264: missing (the videos are written with it)" in text
    fix = text.split("To fix:")[1]
    assert "libx264" in fix and "winget install Gyan.FFmpeg" in fix and "brew install ffmpeg" in fix


def test_doctor_counts_an_ffmpeg_that_cannot_list_its_filters_as_unable(monkeypatch, model_present):
    def broken(*args, **kwargs):
        raise media.MediaError("ffmpeg failed: boom")

    monkeypatch.setattr(media, "hdr_support", broken)
    monkeypatch.setattr(media, "has_encoder", broken)

    result = doctor.check()

    assert result["hdr"] is None and result["libx264"] is False and result["ok"] is False
