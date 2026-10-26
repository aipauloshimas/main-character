"""The setup check: is everything the skill needs on this machine?

check() looks at Python, the three packages, ffmpeg and ffprobe (and what this ffmpeg can do: bring
an iPhone's HDR clips down to normal video, and write H.264 with libx264), the bundled fonts and the
matte model, and report() says what it found, one line each, with the exact commands that fix what
is wrong. It must work on a machine that has none of the packages, so it imports only the standard
library, config.py and media.py (which needs nothing else), and it imports a package to test it, not
to use it.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import re
import shutil
import subprocess
from pathlib import Path

from . import media
from .config import FONT_FILES, MODEL_PATH, MODEL_URL, SKILL_DIR

PYTHON_MIN = (3, 10)
PILLOW_MIN = (9, 1)  # Image.Resampling and ImageFont.Layout, which the drawing code uses
PIP_FIX = "python -m pip install numpy pillow onnxruntime"
PILLOW_FIX = "python -m pip install --upgrade pillow"
MODEL_FIX = "python mc.py doctor --download-model"
_PACKAGES = {"numpy": "numpy", "PIL": "Pillow", "onnxruntime": "onnxruntime"}  # import -> name
_FFMPEG_FIX = (
    "Install ffmpeg (Windows: winget install Gyan.FFmpeg; macOS: brew install ffmpeg), then "
    "open a new terminal so that ffmpeg and ffprobe run from it."
)
_HDR_FIX = f"Install an ffmpeg that can convert HDR video: {media.HDR_FIX}."
_X264_FIX = (
    "Install an ffmpeg build that includes libx264 (Windows: winget install Gyan.FFmpeg; macOS: "
    "brew install ffmpeg), then open a new terminal so that ffmpeg and ffprobe run from it."
)
_HDR_LINES = {  # what the report says of how this ffmpeg converts HDR clips (check's "hdr")
    "zscale": "HDR clips: OK (converted with zscale)",
    "swscale": "HDR clips: OK (converted by ffmpeg 8 or newer)",
    None: "HDR clips: this ffmpeg can't convert them (an iPhone shoots HDR by default)",
}


def check(skill_dir: Path = SKILL_DIR) -> dict:
    """What this machine has, as {python, packages, ffmpeg, ffprobe, hdr, libx264, fonts_missing,
    model, ok}.

    python is the version as text. packages maps numpy, PIL and onnxruntime to the version found,
    or None when the package cannot be imported. ffmpeg and ffprobe are the version found on the
    PATH, or None. hdr is how this ffmpeg brings an iPhone's HDR clips down to normal video:
    "zscale", "swscale" (ffmpeg 8's own color management) or None when it cannot, and libx264 is
    whether it has the encoder the videos are written with. Both are asked of an ffmpeg that is
    there, and are None and False when it is not. fonts_missing lists the files of the skill's fonts
    folder (under skill_dir) that are not there, and model is whether the matte model is where it is
    kept. ok is true only when everything is right: Python 3.10 or newer, all three packages with
    Pillow 9.1 or newer, both tools, HDR and libx264 there, no font missing and the model there.
    """
    fonts = Path(skill_dir) / "fonts"
    ffmpeg = _tool_version("ffmpeg")
    hdr, libx264 = _ffmpeg_abilities() if ffmpeg is not None else (None, False)
    result = {
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in _PACKAGES},
        "ffmpeg": ffmpeg,
        "ffprobe": _tool_version("ffprobe"),
        "hdr": hdr,
        "libx264": libx264,
        "fonts_missing": sorted(
            {name for name, _ in FONT_FILES.values() if not (fonts / name).is_file()}
        ),
        "model": MODEL_PATH.is_file(),
    }
    result["ok"] = not _fixes(result)
    return result


def report(result: dict) -> str:
    """The result of check as text: one line for each thing checked, then the commands that fix
    what is wrong (or "Everything is ready." when nothing is)."""
    packages, fonts = result["packages"], result["fonts_missing"]
    lines = [
        _line("Python", result["python"], _python_ok(result["python"]), "3.10 or newer is needed")
    ]
    for name, label in _PACKAGES.items():
        version = packages[name]
        too_old = name == "PIL" and version is not None and not _pillow_ok(version)
        lines.append(_line(label, version, version is not None and not too_old,
                           "9.1 or newer is needed" if too_old else "not installed"))
    for tool in ("ffmpeg", "ffprobe"):
        lines.append(_line(tool, result[tool], result[tool] is not None, "not found on the PATH"))
    if result["ffmpeg"] is None:  # (there is no ffmpeg to ask what it can do)
        lines += ["HDR clips: not checked (no ffmpeg)", "libx264: not checked (no ffmpeg)"]
    else:
        lines.append(_HDR_LINES[result["hdr"]])
        lines.append(
            "libx264: OK" if result["libx264"] else "libx264: missing (the videos are written with it)"
        )
    lines.append(
        f"fonts: missing {', '.join(fonts)}" if fonts
        else f"fonts: OK, all {len({name for name, _ in FONT_FILES.values()})} found"
    )
    lines.append(
        f"matte model: OK, {MODEL_PATH}" if result["model"]
        else f"matte model: missing (a download of about 15 MB from {MODEL_URL})"
    )

    fixes = _fixes(result)
    if not fixes:
        return "\n".join([*lines, "", "Everything is ready."])
    return "\n".join([
        *lines, "", "To fix:", *[f"  {fix}" for fix in fixes], "",
        "Ask the person before running a command that installs or downloads something.",
    ])


def _fixes(result: dict) -> list[str]:
    """The commands, and the one instruction, that fix what is wrong in a result of check, or an
    empty list when nothing is."""
    packages = result["packages"]
    fixes = []
    if any(version is None for version in packages.values()):
        fixes.append(PIP_FIX)
    if packages["PIL"] is not None and not _pillow_ok(packages["PIL"]):
        fixes.append(PILLOW_FIX)
    if not _python_ok(result["python"]):
        fixes.append("Install Python 3.10 or newer from https://www.python.org/downloads/")
    if result["ffmpeg"] is None or result["ffprobe"] is None:
        fixes.append(_FFMPEG_FIX)
    if result["ffmpeg"] is not None:
        if not result["libx264"]:
            fixes.append(_X264_FIX)
        if result["hdr"] is None:
            fixes.append(_HDR_FIX)
    if result["fonts_missing"]:
        fixes.append("Copy the whole main-character folder again: its fonts folder is incomplete.")
    if not result["model"]:
        fixes.append(MODEL_FIX)
    return fixes


def _line(label: str, found: str | None, good: bool, otherwise: str) -> str:
    """One line of the report: "numpy 2.1.0: OK", "numpy: not installed" or "Pillow 9.0: too old"."""
    if good:
        return f"{label} {found}: OK"
    return f"{label}: {otherwise}" if found is None else f"{label} {found}: {otherwise}"


def _version_tuple(version: str) -> tuple[int, ...]:
    """The leading numbers of a version as text: "10.4.0" is (10, 4, 0), "3.13.2rc1" is (3, 13, 2).
    A version with no numbers in it is ()."""
    return tuple(int(number) for number in re.findall(r"\d+", version.split("+")[0])[:3])


def _python_ok(version: str) -> bool:
    return _version_tuple(version) >= PYTHON_MIN


def _pillow_ok(version: str) -> bool:
    """Whether a Pillow version is new enough. One that says nothing is taken to be, since the
    import worked."""
    numbers = _version_tuple(version)
    return not numbers or numbers >= PILLOW_MIN


def _package_version(name: str) -> str | None:
    """The version of a package after importing it, or None when it cannot be imported. The import
    is the test: a package can be installed and still fail to load, and any error counts."""
    try:
        module = importlib.import_module(name)
    except Exception:  # an ImportError, but also what a broken binary raises while loading
        return None
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(_PACKAGES[name])
        except importlib.metadata.PackageNotFoundError:
            version = "of unknown version"
    return str(version)


def _ffmpeg_abilities() -> tuple[str | None, bool]:
    """How this ffmpeg converts HDR clips (media.hdr_support) and whether it has libx264, as check's
    hdr and libx264. An ffmpeg that cannot even list what it has is taken to have neither."""
    try:
        hdr = media.hdr_support()
    except media.MediaError:
        hdr = None
    try:
        libx264 = media.has_encoder("libx264")
    except media.MediaError:
        libx264 = False
    return hdr, libx264


def _tool_version(name: str) -> str | None:
    """The version of ffmpeg or ffprobe as the first line of its -version says it, or None when the
    tool is not on the PATH or does not run."""
    path = shutil.which(name)
    if path is None:
        return None
    try:
        shown = subprocess.run(
            [path, "-version"], stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
            encoding="utf-8", errors="replace",
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.search(r"version (\S+)", shown)
    return found.group(1) if found else None
