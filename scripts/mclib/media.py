"""Media helpers: the only module that talks to ffmpeg and ffprobe.

Every call is an argument list run without a shell, so paths with spaces, accents or quotes are
safe, and a failing call raises MediaError with the tool's own message. The cutout and the render
lean on three promises kept here:

- Frames are counted at one rate, the fps the caller gives (config.FPS, 30, when it gives none;
  output_fps picks the rate of a clip, which is the clip's own). extract_frames and encode_clip
  both begin with frame_filter, so frame i of one is frame i of the other, whatever frame rate the
  clip was shot at.
- encode_clip and encode_stills write exactly the number of frames asked for, or raise.
- concat joins parts without losing or doubling a frame. The parts of one video are all the same
  size: the clip's own frame (render.py gives each the size it is to have), with nothing around it.
- Every video written here is normal (SDR) BT.709 limited-range video, tagged that way, so the
  parts match and a player shows the colors as drawn. A clip's own pixels pass through as they
  are. Art drawn in RGB (a card, an overlay) is converted with the BT.709 matrix. A JPEG made by
  extract_frames decodes in Pillow to the clip's true colors. An HDR clip (HLG or PQ, which
  iPhones shoot by default) is tone-mapped to SDR first, by both functions, in the same way.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from .config import CLIP_RANGE, FPS, MIN_BITRATE
from .timeline import Segment

_FADE_OUT = 0.4  # seconds the sound fades out at the end of the video
_CUT_FADE = 0.010  # seconds the clip's sound fades in and out at each cut, so that a cut does not click
_SOUND_RATE = 48000  # the rate of the clip's sound as clip_sound writes it
STILL_WEIGHT = 0.25  # of its share of the bits, what the card gets in a two-pass final encode
_FRAME_FILES = "[0-9]" * 6 + ".jpg"  # the file names extract_frames writes, as a glob pattern

# Color. ffmpeg 8 drops the -color_primaries and -color_trc options, so every encode writes its
# tags with this bitstream filter instead: BT.709 primaries, transfer and matrix, limited range.
_BT709_TAGS = [
    "-bsf:v",
    "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:"
    "video_full_range_flag=0",
]
_RGB_TO_BT709 = "scale=out_color_matrix=bt709:out_range=tv"  # RGB art to BT.709 limited range
# A JPEG is BT.601 full range, which is what Pillow decodes. The clip is read as BT.709 even
# when it carries no tags, as every player does for HD video.
_TO_JPEG_COLORS = "scale=in_color_matrix=bt709:out_color_matrix=bt470bg:out_range=pc"
_HDR_TRANSFERS = ("arib-std-b67", "smpte2084")  # HLG and PQ
# An HDR clip is brought down to normal video one of two ways. With the zscale filter (the build has
# to include libzimg: Gyan's full build on Windows does, Homebrew's ffmpeg does not) the light is made
# linear, the primaries BT.709, the highlights rolled off by Hable's curve, and the transfer BT.709
# again. Without it, ffmpeg 8 and newer can do the whole thing in the scale filter, with swscale's own
# color management: the transfer, the primaries and the matrix of the output are BT.709, and the
# perceptual intent rolls the highlights off.
_TONE_MAP = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)
_SWSCALE_TONE_MAP = (
    "scale=out_transfer=bt709:out_primaries=bt709:out_color_matrix=bt709:out_range=tv:"
    "intent=perceptual,format=yuv420p"
)
# What to install for HDR clips, for the doctor's fix and for the error a clip gets. On a Mac the
# full build is keg-only, so it has to be put first on the PATH to be the ffmpeg that runs.
HDR_FIX = (
    "on Windows, the full build (winget install Gyan.FFmpeg); on a Mac, ffmpeg 8 or newer "
    "(brew install ffmpeg), or brew install ffmpeg-full with its bin folder first on the PATH "
    "(brew --prefix ffmpeg-full shows where it is)"
)
_HDR_ERROR = (
    "This clip is HDR, and this ffmpeg build can't convert HDR to normal video. Turn off HDR "
    f"video in your camera settings, or install an ffmpeg that can: {HDR_FIX}."
)
_NOT_A_VIDEO = "That's a picture, not a video. Give me a clip of {low:g} to {high:g} seconds."
_NO_LENGTH = "{name} has no length I can read. Give me a clip of {low:g} to {high:g} seconds."
_IMAGE_FORMATS = ("image2", "ico")  # and every "<kind>_pipe" (jpeg_pipe, png_pipe, ...): ffprobe's names for a picture
_VARIABLE_RATE = 1.05  # an r_frame_rate over this times the average is a variable rate's fastest gap

# The rates a video is made at. output_fps snaps a probed rate to the nearest of these when it is
# within RATE_TOLERANCE of it, so a clip that probes as 29.97 is 30000/1001, never 29.97 or 30.
_STANDARD_RATES = (
    Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
    Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60),
)
RATE_TOLERANCE = 0.005  # half a percent, as a share of the rate
MAX_FPS = 60  # the highest rate a render is made at


class MediaError(RuntimeError):
    """ffmpeg or ffprobe failed, or a file is not what was expected."""


def missing_tool(name: str) -> MediaError:
    """The error for a tool (ffmpeg or ffprobe) that is not on this computer: it names the tool and
    says where to look. The caller raises it."""
    return MediaError(
        f"{name} isn't installed, or isn't on the PATH. Run python mc.py doctor to see what to install."
    )


def _exec(command: list[str]) -> str:
    """Run a command without a shell and return what it printed. A failure raises MediaError
    with the tool's message, and a tool that is not there says so (see missing_tool). stdin is
    closed, so a tool can never stop to wait for an answer."""
    try:
        done = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError:
        raise missing_tool(command[0]) from None
    if done.returncode != 0:
        raise MediaError(f"{command[0]} failed: {done.stderr.strip()}")
    return done.stdout


def run(args: list[str]) -> None:
    """Run ffmpeg with args. It prints only errors, and it overwrites the output file."""
    _exec(["ffmpeg", "-v", "error", "-y", *args])


def _parse_filter_names(listing: str) -> frozenset[str]:
    """The filter names in the text of `ffmpeg -filters`. A filter line is flags, name, "inputs->
    outputs" and a description, but the flags are a few letters that differ from one ffmpeg version
    to the next (they could be in more columns, or hold a letter nobody has used yet). So a line is
    read from the one column that is always the same: the name is the word before the first one with
    an arrow in it. The legend above the list has no such column."""
    names = set()
    for line in listing.splitlines():
        fields = line.split()
        for index, field in enumerate(fields):
            if "->" in field:
                if index >= 2:  # (the flags, then the name)
                    names.add(fields[index - 1])
                break
    return frozenset(names)


@functools.cache
def _filter_names() -> frozenset[str]:
    """The names of the filters this ffmpeg build has. Asked once, then kept."""
    return _parse_filter_names(_exec(["ffmpeg", "-hide_banner", "-filters"]))


def has_filter(name: str) -> bool:
    """Whether this ffmpeg build has the filter, for example "zscale"."""
    return name in _filter_names()


_ENCODER_FLAGS = re.compile(r"[VAS][.A-Z]{3,}")  # six letters today: the kind, then the flags


def _parse_encoder_names(listing: str) -> frozenset[str]:
    """The encoder names in the text of `ffmpeg -encoders`: lines of flags, name and description.
    The legend above the list has an "=" where a name would be."""
    names = set()
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] != "=" and _ENCODER_FLAGS.fullmatch(fields[0]):
            names.add(fields[1])
    return frozenset(names)


@functools.cache
def _encoder_names() -> frozenset[str]:
    """The names of the encoders this ffmpeg build has. Asked once, then kept."""
    return _parse_encoder_names(_exec(["ffmpeg", "-hide_banner", "-encoders"]))


def has_encoder(name: str) -> bool:
    """Whether this ffmpeg build has the encoder, for example "libx264". An LGPL build leaves it
    out, and then no video the skill makes can be written."""
    return name in _encoder_names()


@functools.cache
def _swscale_tone_maps() -> bool:
    """Whether this build brings HLG down to normal video with swscale's own color management (the
    scale filter's out_transfer and out_primaries, from ffmpeg 8). It is found by trying it on one
    tiny frame, so it is true exactly when a clip would be converted, whatever the version string
    says (a git build has none that can be read). Asked once, then kept."""
    probe = (
        "color=c=gray:s=16x16:r=1:d=1,format=yuv420p10le,"
        "setparams=colorspace=bt2020nc:color_primaries=bt2020:color_trc=arib-std-b67:range=tv,"
        + _SWSCALE_TONE_MAP
    )
    try:
        _exec(["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", probe,
               "-frames:v", "1", "-f", "null", "-"])
    except MediaError:
        return False
    return True


def hdr_support() -> str | None:
    """How this ffmpeg build brings an HDR clip down to normal video: "zscale" (the zscale and
    tonemap filters), "swscale" (ffmpeg 8's own color management, in the scale filter) or None when
    it can do neither. zscale comes first, since it is the way that has always been used."""
    if has_filter("zscale") and has_filter("tonemap"):
        return "zscale"
    return "swscale" if _swscale_tone_maps() else None


# --- looking at a file ----------------------------------------------------------------------


def probe(path: Path, count: bool = True) -> dict:
    """The first video stream of a file, as {width, height, fps, duration, bit_rate, frames,
    rotation, color_transfer, hdr}. With count False there is no frames, and no decoding: it is
    quick, for a caller that only needs to know what the clip is.

    Width and height are the picture as a player shows it, so a clip stored sideways with a
    display rotation of 90 comes back as portrait, and rotation says that turn in degrees (0, 90,
    180 or 270). fps is the stream's own frame rate (the average one, for a clip whose frame rate
    varies, as a phone's can) and duration its length in seconds. A picture is no clip, and a
    file with no length of its own raises MediaError, saying so. bit_rate
    is the video's bitrate in bits per second, an int, or None when the file states none (a
    container that does not state it for the stream gives the whole file's, a little more). frames
    is counted by decoding the whole stream, never read from the header. color_transfer is the
    transfer the stream is tagged with (a string, None when it has none) and hdr is True for HLG
    and PQ, the two kinds of HDR phones shoot.
    """
    return _probe(path, count=count)


def count_frames(path: Path) -> int:
    """How many frames the first video stream holds, counted by decoding it."""
    return probe(path)["frames"]


def _probe(path: Path, count: bool) -> dict:
    """probe, without its slow frame count when count is False."""
    shown = _exec([
        "ffprobe", "-v", "error", "-select_streams", "v:0", *(["-count_frames"] if count else []),
        "-show_streams", "-show_format", "-of", "json", str(path),
    ])
    found = json.loads(shown)
    if not found.get("streams"):
        raise MediaError(f"{Path(path).name} has no video stream.")
    stream = found["streams"][0]

    rotation = _rotation(stream)
    width, height = stream["width"], stream["height"]
    if rotation in (90, 270):
        width, height = height, width
    info = {
        "width": width,
        "height": height,
        "fps": _frame_rate(stream),
        "duration": _length(stream, found.get("format") or {}, path),
        "bit_rate": _bit_rate(stream, found.get("format") or {}),
    }
    if count:
        info["frames"] = int(stream["nb_read_frames"])
    info["rotation"] = rotation
    transfer = stream.get("color_transfer")
    info["color_transfer"] = None if transfer in (None, "", "unknown") else transfer
    info["hdr"] = info["color_transfer"] in _HDR_TRANSFERS
    return info


def _ratio(text) -> float:
    """A frame rate as ffprobe writes it ("30000/1001") as a number; 0.0 for one it leaves out, writes
    as "0/0" (unknown) or garbles."""
    try:
        return float(Fraction(text))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _frame_rate(stream: dict) -> float:
    """The frame rate of a video stream in frames per second: its r_frame_rate. A phone that varies
    its frame rate states r_frame_rate as its fastest gap between two frames (60 for a clip that runs
    at 30 most of the time), and read as the clip's rate that doubles it. So when r_frame_rate is more
    than 5% over avg_frame_rate, the clip is a variable-rate one and the average is its rate."""
    rate, average = _ratio(stream.get("r_frame_rate")), _ratio(stream.get("avg_frame_rate"))
    if average and rate > average * _VARIABLE_RATE:
        return average
    return rate or average


def _length(stream: dict, container: dict, path: Path) -> float:
    """The length in seconds of a video stream, from the stream or else from its container
    (Matroska and WebM keep it there). A file that states none is not a clip: a picture has no
    length at all, and the person is told so, with the lengths a clip may be."""
    for found in (stream, container):
        try:
            length = float(found["duration"])
        except (KeyError, TypeError, ValueError):
            continue
        if length >= 0:  # (a NaN is neither)
            return length
    low, high = CLIP_RANGE
    kinds = str(container.get("format_name", "")).split(",")
    if any(kind.endswith("_pipe") or kind in _IMAGE_FORMATS for kind in kinds):
        raise MediaError(_NOT_A_VIDEO.format(low=low, high=high))
    raise MediaError(_NO_LENGTH.format(name=Path(path).name, low=low, high=high))


def _bit_rate(stream: dict, container: dict) -> int | None:
    """The bits per second a video stream states for itself, or else its container states for the
    whole file, or None. ffprobe prints them as text, and leaves them out when it does not know."""
    for found in (stream, container):
        try:
            rate = int(found["bit_rate"])
        except (KeyError, TypeError, ValueError):
            continue
        if rate > 0:
            return rate
    return None


def _rotation(stream: dict) -> int:
    """A stream's display rotation in degrees (0, 90, 180 or 270), from its Display Matrix side
    data. The older "rotate" tag is not read: ffmpeg 8 does not turn the picture for it, and the
    size probe reports has to be the size of the frames ffmpeg really decodes."""
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            return round(side_data["rotation"]) % 360
    return 0


def _duration(path: Path) -> float:
    """The length of a file's first video stream in seconds, without counting its frames."""
    return _probe(path, count=False)["duration"]


def orientation(w: int, h: int) -> str:
    """The shape of a picture: "landscape" when it is at least 1.2 times wider than tall,
    "portrait" when at least 1.2 times taller than wide, otherwise "square"."""
    # w / h >= 1.2 is w * 5 >= h * 6, which is exact for whole pixels where the division is not
    if w * 5 >= h * 6:
        return "landscape"
    if h * 5 >= w * 6:
        return "portrait"
    return "square"


def output_fps(info: dict) -> Fraction:
    """The frame rate a clip is made into: its own, never a converted one. info is anything with an
    "fps" number, such as what probe returns or a project's clip.

    A probed rate within half a percent of a standard rate (24000/1001, 24, 25, 30000/1001, 30, 48,
    50, 60000/1001 or 60) is that rate, as an exact fraction. Any other rate is rounded to a whole
    number, at least 1. Nothing goes above MAX_FPS (60): a faster clip is made at 60, which only
    drops frames, and the caller says so.
    """
    fps = float(info["fps"])
    nearest = min(_STANDARD_RATES, key=lambda rate: abs(float(rate) - fps))
    if abs(float(nearest) - fps) <= RATE_TOLERANCE * float(nearest):
        return nearest
    return Fraction(min(max(round(fps), 1), MAX_FPS))


def _rate(fps: Fraction | int | float) -> Fraction:
    """A frame rate as an exact fraction, which ffmpeg reads as it is: str(Fraction(30000, 1001)) is
    "30000/1001" and str(Fraction(30)) is "30". A float is turned into the nearest fraction with a
    denominator of at most 1001, so 29.97 does not become a 50-digit binary fraction."""
    return Fraction(fps).limit_denominator(1001)


def target_bitrate(clip_bit_rate: int | None) -> int:
    """The bitrate of a final render, in bits per second: the clip's own, and never under
    MIN_BITRATE (10 Mbps). None, for a clip that states none, is the floor."""
    return max(int(clip_bit_rate or 0), MIN_BITRATE)


# --- frames out of a clip, and back in ------------------------------------------------------


def frame_filter(
    src_w: int, src_h: int, work: tuple[int, int], fps: Fraction | int = FPS, focus_y: float = 0.42
) -> str:
    """The ffmpeg filters that turn a clip into work-sized frames at fps frames per second (a
    whole number or an exact Fraction such as 30000/1001). The work size is the plan's (config.
    plan_size) for the frames of a project, and the output's (config.output_size) for the video.

    The picture is scaled until both sides are at least the work size, keeping its shape, then
    cut to it: centered across, and down so that the part of the source at focus_y (0 is the top,
    1 the bottom) stays at the same relative height, kept inside the picture. Both sizes have the
    clip's shape, so this only takes off the pixel or two that rounding to an even size costs.
    Every number is worked out here rather than by an ffmpeg expression, so the same input always
    gives the same string. src_w and src_h are the size as a player shows the clip, after any
    rotation, because ffmpeg turns the picture upright when it decodes.
    """
    width, height = work
    # The tight side lands exactly on the work size and the other is rounded up, so the picture
    # is never smaller than the work size. -(-a // b) is a divided by b, rounded up.
    if width * src_h >= height * src_w:  # the width is the tight side
        scaled_w, scaled_h = width, -(-width * src_h // src_w)
    else:
        scaled_w, scaled_h = -(-height * src_w // src_h), height
    x = (scaled_w - width) // 2
    y = min(max(round(focus_y * (scaled_h - height)), 0), scaled_h - height)
    return f"fps={_rate(fps)},scale={scaled_w}:{scaled_h},crop={width}:{height}:{x}:{y}"


def _tone_map(info: dict) -> list[str]:
    """The filters that bring an HDR clip down to normal video, to go first in a chain, or none
    for a clip that is not HDR. They are zscale's when this build has it, and else ffmpeg 8's own
    color management in the scale filter (see hdr_support). Raises MediaError, saying what to
    install, when this build can do neither."""
    if not info["hdr"]:
        return []
    method = hdr_support()
    if method == "zscale":
        return [_TONE_MAP]
    if method == "swscale":
        return [_SWSCALE_TONE_MAP]
    raise MediaError(_HDR_ERROR)


def extract_frames(
    clip: Path,
    out_dir: Path,
    work: tuple[int, int],
    fps: Fraction | int = FPS,
    first: int = 0,
    last: int | None = None,
) -> int:
    """Write every frame of a clip, at fps frames per second, as a work-sized JPEG (-q:v 2) named
    000000.jpg, 000001.jpg and so on, and return how many were written. A render passes the clip's
    own rate (output_fps), so the frames are the clip's own, none dropped and none made up.

    first and last (frame numbers, both written) cut only that stretch of the clip, numbered from
    000000 on: the freeze frame of a final render and the frames the matte warms up on, at the size
    of the output. The frames after last are never decoded.

    The JPEGs hold the clip's true colors: Pillow, which reads a JPEG as BT.601 full range, gives
    the colors a BT.709 player shows for the clip. An HDR clip is tone-mapped first. JPEGs left in
    out_dir by an earlier run are removed first, so the folder and the count agree.
    """
    out_dir = Path(out_dir)
    info = _probe(clip, count=False)
    chain = [*_tone_map(info), frame_filter(info["width"], info["height"], work, fps)]
    stop = []
    if first or last is not None:  # (frame numbers are those of frame_filter's rate, as encode_clip's are)
        end = "" if last is None else f":end_frame={last + 1}"
        chain += [f"trim=start_frame={first}{end}", "setpts=PTS-STARTPTS"]
        stop = [] if last is None else ["-frames:v", str(last - first + 1)]
    chain.append(_TO_JPEG_COLORS)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(_FRAME_FILES):
        stale.unlink()
    # A percent sign in the folder name is doubled, or ffmpeg reads it as part of the name pattern
    pattern = os.path.join(str(out_dir).replace("%", "%%"), "%06d.jpg")
    run(["-i", str(clip), "-vf", ",".join(chain), "-q:v", "2", *stop, "-start_number", "0", pattern])
    return sum(1 for _ in out_dir.glob(_FRAME_FILES))


def _encoder(preview: bool, lossless: bool = False) -> list[str]:
    """The libx264 settings: crf 16 at preset medium, or for a preview crf 28 at ultrafast, or,
    when lossless, quantizer 0 at ultrafast, which keeps every pixel of what the filters made (the
    file is large, and only a part of a render, joined and deleted). The video is tagged BT.709
    limited range."""
    if lossless:
        return ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p",
                *_BT709_TAGS]
    preset, crf = ("ultrafast", "28") if preview else ("medium", "16")
    return ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p",
            *_BT709_TAGS]


def _final_encoder(bitrate: int) -> list[str]:
    """The libx264 settings of a render's one lossy encode: preset medium at an average of bitrate
    bits per second, with a ceiling of 1.5 times that and a buffer of 2 times. Tagged BT.709
    limited range."""
    return ["-c:v", "libx264", "-preset", "medium", "-b:v", str(bitrate),
            "-maxrate", str(round(bitrate * 1.5)), "-bufsize", str(bitrate * 2),
            "-pix_fmt", "yuv420p", *_BT709_TAGS]


def _check_frames(path: Path, wanted: int) -> None:
    """Raise MediaError unless the video holds exactly the frames wanted. ffmpeg exits normally
    on an empty result, for example when a clip is trimmed past its end, so the count is checked."""
    try:
        got = count_frames(path)
    except MediaError:  # an empty file has no video stream
        got = 0
    if got != wanted:
        raise MediaError(f"{Path(path).name} has {got} frames, expected {wanted}.")


def encode_clip(
    clip: Path,
    out: Path,
    seg: Segment,
    src_size: tuple[int, int],
    size: tuple[int, int],
    preview: bool = False,
    fps: Fraction | int = FPS,
    lossless: bool = False,
) -> None:
    """Encode one run of a clip (a "clip" segment) as a video part of exactly seg.frames frames, at
    fps frames per second (see output_fps), and `size` (the output's, or for a preview the plan's):
    the clip's own frames and nothing around them. The segment's source times are seconds, so they
    stay right at any rate. A part that concat is going to join can be lossless (see _encoder), so
    that the join is the only lossy encode.

    An HDR clip is tone-mapped first. Then comes a relabel (no conversion) of the clip as BT.709,
    which is what every player takes HD video without tags to be. Then the filters run in this
    order: frame_filter (which brings the picture to `size`), the trim to the segment's first
    frame, the speed change, fps again (the speed change leaves the frames at odd times), and
    cloning of the last frame, which keeps the count exact when rounding leaves the clip a frame
    short. The pixels of the clip are otherwise as they are (a clip stored in full range is brought
    to the limited range every video here has). src_size is the clip's size as a player shows it,
    from probe. The part has no sound and none of the clip's metadata. A preview is encoded fast.
    """
    rate = _rate(fps)
    chain = [
        *_tone_map(_probe(clip, count=False)),
        "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709",
        frame_filter(*src_size, size, rate),
        f"trim=start_frame={round(seg.src_start * float(rate))}",
        f"setpts=(PTS-STARTPTS)/{seg.speed:.6f}",
        f"fps={rate}",
        "tpad=stop=-1:stop_mode=clone",
        "scale=out_range=tv,format=yuv420p",
    ]
    run([
        "-i", str(clip), "-vf", ",".join(chain), "-an", "-map_metadata", "-1", "-r", str(rate),
        "-frames:v", str(seg.frames), *_encoder(preview, lossless), str(out),
    ])
    _check_frames(out, seg.frames)


def encode_stills(
    stills: list[tuple[Path, int]],
    out: Path,
    preview: bool = False,
    fps: Fraction | int = FPS,
    lossless: bool = False,
) -> None:
    """Encode pictures, each held for a number of frames, as a video part at fps frames per second.
    stills is a list of (picture, frames), at least one, in order. The part has exactly the frames
    added up, and the size of the pictures, which have to be all the one size (the part is joined to
    the others, which are the clip's own size). lossless is as for encode_clip.

    The pictures go through the concat demuxer, which needs each one's duration and the last one
    listed again so that it is shown for its whole time. Three details put every change on the
    right frame:

    - "option framerate" makes the demuxer count in frames of the output rate (30ths of a second
      at 30 fps). Its default is 25 fps, which moves a change by up to 20 ms, so some land a
      frame early.
    - The fps filter holds each picture until the next one arrives. The sync that -r does on its
      own puts a picture that follows a long gap one frame early, so -r only keeps the rate fixed.
    - -frames:v cuts off the repeat of the last picture.

    The pictures are RGB, so they are converted with the BT.709 matrix to limited range, to match
    the clip parts.
    """
    rate = _rate(fps)
    entries = []
    for picture, frames in stills:
        entries += [
            f"file {_quote(picture)}", f"option framerate {rate}",
            f"duration {frames / float(rate):.6f}",
        ]
    entries += [f"file {_quote(stills[-1][0])}", f"option framerate {rate}"]

    chain = [f"fps={rate}", _RGB_TO_BT709, "format=yuv420p"]
    total = sum(frames for _, frames in stills)
    _run_concat(entries, [
        "-vf", ",".join(chain), "-r", str(rate), "-frames:v", str(total),
        *_encoder(preview, lossless), str(out),
    ])
    _check_frames(out, total)


def concat(
    parts: list[Path], out: Path, fps: Fraction | int = FPS, bitrate: int | None = None,
    still: tuple[int, int] | None = None,
) -> None:
    """Join video parts, in order, into one video at fps frames per second. The parts hold no
    sound. They are encoded again (libx264, preset medium, yuv420p) so the joins are clean, and
    tagged BT.709 limited range like the parts. The quality is crf 18, or, when bitrate (bits per
    second) is given, that average with a ceiling of 1.5 times it and a buffer of 2 times it: the
    one lossy encode of a render whose parts are lossless.

    At a bitrate the encode takes two passes. A single pass at an average spends the budget as it
    goes and can leave the end of the video starved: a real intro, a 10.7 Mbps clip, came out with
    2.6 Mbps for the wink in its last second. The first pass reads the whole video and writes only
    its log; the second spreads the bits where the picture needs them.

    still is the card, as (first frame, last frame) of the output, when there is one. Left to
    itself the two-pass encode treats the card's grain and boiling dots as the hardest picture and
    gives it most of the bits (15 Mbps of 10.6 in a real intro, and 5.6 for the footage). A zone
    weights the card at STILL_WEIGHT of its share, so the footage gets the bits and the average is
    still the bitrate."""
    entries = [f"file {_quote(part)}" for part in parts]
    rate = ["-r", str(_rate(fps))]
    if not bitrate:
        _run_concat(entries, [
            *rate, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            *_BT709_TAGS, str(out),
        ])
        return
    quality = _final_encoder(bitrate)
    if still is not None and still[1] >= still[0]:
        quality += ["-x264-params", f"zones={still[0]},{still[1]},b={STILL_WEIGHT}"]
    with tempfile.TemporaryDirectory(prefix="main-character-pass-") as folder:
        log = ["-passlogfile", str(Path(folder) / "x264")]
        _run_concat(entries, [*rate, *quality, "-pass", "1", *log, "-an", "-f", "null", os.devnull])
        _run_concat(entries, [*rate, *quality, "-pass", "2", *log, str(out)])


def _quote(path: Path) -> str:
    """A path as a concat list wants it: absolute, with forward slashes, in single quotes. A
    single quote inside the path is closed, escaped and opened again. Without an absolute path
    the demuxer would look for the file next to the list."""
    return "'" + Path(path).resolve().as_posix().replace("'", "'\\''") + "'"


def _run_concat(entries: list[str], output_args: list[str]) -> None:
    """Run ffmpeg on a concat list made of entries, followed by output_args. The list is a UTF-8
    file with LF line ends in a temporary folder that is removed afterward. -safe 0 lets it name
    absolute paths."""
    with tempfile.TemporaryDirectory(prefix="main-character-") as folder:
        listing = Path(folder) / "list.txt"
        listing.write_bytes(("\n".join(entries) + "\n").encode("utf-8"))
        run(["-f", "concat", "-safe", "0", "-i", str(listing), *output_args])


# --- sound ----------------------------------------------------------------------------------


def _fade_out(duration: float, seconds: float) -> str:
    """The afade filter that takes the last seconds of a sound that is duration long to silence."""
    return f"afade=t=out:st={max(0.0, duration - seconds):.6f}:d={seconds:.6f}"


def mux_audio(video: Path, audio: Path, out: Path, fade_out: float = _FADE_OUT) -> None:
    """Add sound to a video. The picture is copied as it is. The sound is the first audio track
    of audio, encoded as AAC 192k, cut to the length of the video and faded out over its last
    fade_out seconds (no fade when that is 0)."""
    duration = _duration(video)
    fade = ["-af", _fade_out(duration, fade_out)] if fade_out > 0 else []
    run([
        "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", *fade, "-t", f"{duration:.6f}", str(out),
    ])


def mix_click(
    video: Path, click: Path, at: float, out: Path, music: Path | None = None,
    fade_out: float = _FADE_OUT,
) -> None:
    """Add sound to a video: the click sound starting at "at" seconds, over the music when there
    is any. The picture is copied as it is. The sound is padded or cut to the length of the video
    and encoded as AAC 192k. The music is cut to the video, not turned down, and fades out over
    its last fade_out seconds (0.4 s, like the sound mux_audio makes, so the ending is the same
    with a click or not; none when fade_out is 0, for a sound that has faded already, such as
    clip_sound's). The mix lasts as long as the longer of the two, so a track that ends before the
    click does not take the click with it.
    """
    duration = _duration(video)
    whole = f"apad=whole_dur={duration:.6f}"
    inputs = ["-i", str(video), "-i", str(click)]
    graph = f"[1:a]adelay={round(at * 1000)}:all=1[click]"
    if music is None:
        graph += f";[click]{whole}[sound]"
    else:
        inputs += ["-i", str(music)]
        fade = f",{_fade_out(duration, fade_out)}" if fade_out > 0 else ""
        graph += (
            f";[2:a]atrim=end={duration:.6f}{fade}[music]"
            f";[music][click]amix=inputs=2:duration=longest:normalize=0,{whole}[sound]"
        )
    run([
        *inputs, "-filter_complex", graph, "-map", "0:v:0", "-map", "[sound]", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.6f}", str(out),
    ])


def _audio_channels(path: Path) -> int:
    """How many channels the first audio stream of a file has, 0 when it has no audio."""
    shown = _exec([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
        "-of", "csv=p=0", str(path),
    ]).strip()
    return int(shown) if shown else 0


def has_audio(path: Path) -> bool:
    """True when a file has at least one audio stream."""
    return _audio_channels(path) > 0


def clip_sound(clip: Path, segments, fps: Fraction | int | float, out: Path) -> Path | None:
    """Write the clip's own sound, laid out on the timeline, as one WAV at `out` (48 kHz stereo,
    16-bit), and return `out`. None, with nothing written, when the clip has no sound.

    segments are the timeline's (timeline.Segment), in order, and fps the rate the video is counted
    in (see output_fps). The sound follows them one by one:

    - a "fill" or a "card" is silence for frames / fps seconds: the held first frame and the card
      have no sound of their own;
    - a "clip" is the clip's sound from the segment's src_start, for frames * speed / fps seconds of
      the clip, played at the segment's speed (atempo, which changes the speed and leaves the pitch
      alone), so that it lasts frames / fps seconds, as the picture does. The tail therefore starts
      at its own source time, and the sound comes back in step with the picture when the card lets go.
      A piece of the clip fades in and out over 10 ms, so that a cut does not click.

    The whole sound fades out over its last 0.4 s, like the music does, and is padded or cut to
    exactly total frames / fps seconds, the length of the video (a clip whose sound is shorter than
    its picture is padded with silence). The tail of a clip with a different length of sound and
    picture is therefore never a frame off the picture.
    """
    channels = _audio_channels(clip)
    if not channels:
        return None
    # A mono sound is copied to both sides. The resampler's own mono-to-stereo is 3 dB quieter.
    stereo = "pan=stereo|c0=c0|c1=c0," if channels == 1 else ""
    rate = _rate(fps)
    seconds = [Fraction(segment.frames) / rate for segment in segments]
    total = float(sum(seconds))
    clips = [number for number, segment in enumerate(segments) if segment.kind == "clip"]

    graph = []
    if clips:  # one copy of the clip's sound for each piece of it that is used
        names = "".join(f"[a{number}]" for number in clips)
        graph.append(f"[0:a]asplit={len(clips)}{names}")
    for number, (segment, length) in enumerate(zip(segments, map(float, seconds))):
        exact = f"apad=whole_dur={length:.9f},atrim=end={length:.9f}"
        if segment.kind == "clip":
            start = segment.src_start
            speed = f",atempo={segment.speed:.6f}" if segment.speed != 1.0 else ""
            fade = min(_CUT_FADE, length / 2)
            graph.append(
                f"[a{number}]atrim=start={start:.9f}:end={start + length * segment.speed:.9f},"
                f"asetpts=PTS-STARTPTS{speed},{stereo}aresample={_SOUND_RATE},aformat=channel_layouts=stereo,"
                f"afade=t=in:d={fade:.6f},afade=t=out:st={length - fade:.9f}:d={fade:.6f},"
                f"{exact}[p{number}]"
            )
        else:
            graph.append(
                f"anullsrc=r={_SOUND_RATE}:cl=stereo,{exact}[p{number}]"
            )
    joined = "".join(f"[p{number}]" for number in range(len(segments)))
    graph.append(
        f"{joined}concat=n={len(segments)}:v=0:a=1,{_fade_out(total, _FADE_OUT)},"
        f"apad=whole_dur={total:.9f},atrim=end={total:.9f}[sound]"
    )
    inputs = ["-i", str(clip)] if clips else []
    run([
        *inputs, "-filter_complex", ";".join(graph), "-map", "[sound]", "-c:a", "pcm_s16le",
        "-ar", str(_SOUND_RATE), "-t", f"{total:.9f}", str(out),
    ])
    return Path(out)
