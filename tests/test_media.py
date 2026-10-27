"""Tests for the ffmpeg helpers: probing, frame extraction, encoding, concatenating and audio.

The clips are generated, so nothing depends on a file in the repo, and frame counts are asserted
exactly. To check which frames an encode kept, the tests use a "code clip" in which frame N is
drawn as 8 black or white blocks (the bits of N). frame_codes() reads the numbers back from an
encoded video, so a trim that starts a frame late, or a speed change that picks the wrong frames,
shows up as a different list of numbers.

A video here is the clip's own frame and nothing around it (Amendment 1): the clip parts are the clip
at the size they are given, and the card parts are pictures of that size. The tests make them at 1080x608,
the plan of a 16:9 clip, and a few at the sizes a render makes (1920x1080, and a small clip scaled up).

Colors are checked the way a viewer meets them. player_rgb() decodes a frame as a BT.709 player
does and the result is compared with the colors that were drawn (flat patches and palette
swatches). The color tags are read twice, from the container and from the decoded picture,
because the two can disagree. test_hdr_clip_is_tone_mapped uses a clip shot as HLG.
"""

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from conftest import BT709, color_tags, rms, x264_options
from PIL import Image, ImageDraw

from mclib import media
from mclib.media import MediaError
from mclib.timeline import Segment

WORK = (1080, 608)  # the size the videos here are made at: the plan of a 16:9 clip
FRAME = (0, 0, 1080, 608)  # the whole frame, as a box for frame_codes
SOUND_RATE = 44100  # the rate mono_samples decodes at
# What every video the helpers write must carry, both as the container states it and as the
# decoder reads it from the pictures. The two can disagree, and players follow one or the other.
BT709_TAGS = {"stream": BT709, "frame": BT709}


def hex_rgb(code):
    """A "#rrggbb" color as an (r, g, b) tuple."""
    return tuple(int(code[i:i + 2], 16) for i in (1, 3, 5))


# Palette colors of the styles, drawn as squares along the bottom like the art of a card (over the palette
# patches of the camera clip, and clear of the rows where the person's are)
SWATCHES = [hex_rgb(code) for code in (
    "#7d2a72", "#f2a9de", "#f3e3a8", "#ffd21f", "#e8261c", "#2f4fa8", "#5a24c9", "#3ef0ff")]
SWATCH_BOXES = [(140 + 115 * i, 500, 219 + 115 * i, 579) for i in range(len(SWATCHES))]
# The flat patches of a camera clip, in a grid of 4 columns and 4 rows: the first 8 are a person
# (skin tones, hair, gray, pure red), the rest palette colors
PATCHES = [hex_rgb(code) for code in (
    "#ffdbac", "#f1c27d", "#e0ac69", "#c68642", "#8d5524", "#5c3c28", "#808080", "#ff0000",
    "#7d2a72", "#f3e3a8", "#ffd21f", "#e8261c", "#2f4fa8", "#5a24c9", "#3ef0ff", "#ffffff")]
PERSON = 8
LEVELS = 3  # how far a color may be off in 8-bit levels, after one trip through the helpers
JPEG_LEVELS = 5  # and after the longer trip through a JPEG (lossy, and full range is coarser)


# --- helpers --------------------------------------------------------------------------------


def streams(path):
    """Every stream of a file as ffprobe reports it, read without going through mclib.media."""
    shown = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,pix_fmt,width,height,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(shown)["streams"]


def format_tags(path):
    """The metadata tags of a file's container (title, location and so on), from ffprobe."""
    shown = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(shown).get("format", {}).get("tags", {})


def assert_x264(path, crf, preset):
    """Check that a video was encoded by libx264 at this crf and preset (medium or ultrafast).

    The encoder leaves its settings as text in the stream. The preset is not named there, but
    cabac is on at medium and off at ultrafast.
    """
    options = x264_options(path)
    cabac = {"medium": "1", "ultrafast": "0"}[preset]
    assert (options["crf"], options["cabac"]) == (f"{crf}.0", cabac)


def frame_codes(video, box):
    """The frame numbers a video shows in the code blocks inside box (x, y, width, height).

    One ffmpeg call crops the box, averages it down to 8 pixels and writes one byte per pixel.
    A pixel above 128 is a set bit.
    """
    left, top, width, height = box
    raw = video.with_suffix(".codes")
    media.run(["-i", str(video), "-vf",
               f"crop={width}:{height}:{left}:{top},scale=8:1:flags=area,format=gray",
               "-f", "rawvideo", str(raw)])
    data = raw.read_bytes()
    return [sum((data[i + bit] > 128) << bit for bit in range(8)) for i in range(0, len(data), 8)]


def code_png(path, number):
    """A picture of the size of the frames showing number in its code blocks, like the code clip."""
    width, height = WORK
    image = Image.new("RGB", WORK, "black")
    draw = ImageDraw.Draw(image)
    for bit in range(8):
        if number >> bit & 1:
            draw.rectangle((width * bit // 8, 0, width * (bit + 1) // 8 - 1, height - 1), fill="white")
    image.save(path)
    return path


def frame_colors(video):
    """The color of each frame of a video made of flat red and flat blue pictures."""
    raw = video.with_suffix(".rgb")
    media.run(["-i", str(video), "-vf", "scale=1:1:flags=area", "-f", "rawvideo",
               "-pix_fmt", "rgb24", str(raw)])
    data = raw.read_bytes()
    colors = []
    for i in range(0, len(data), 3):
        red, _, blue = data[i:i + 3]
        colors.append("red" if red > 128 > blue else "blue" if blue > 128 > red else "other")
    return colors


def mono_samples(path):
    """The sound of a file as mono floats at 44.1 kHz, decoded by ffmpeg."""
    raw = path.with_suffix(".pcm")
    media.run(["-i", str(path), "-vn", "-ac", "1", "-ar", str(SOUND_RATE), "-f", "s16le", str(raw)])
    return np.frombuffer(raw.read_bytes(), dtype="<i2") / 32768.0


def make_code_clip(path, fps, seconds):
    """A 640x360 clip where frame N shows N as 8 black or white blocks, bit 0 on the left."""
    bit = "if(mod(floor(N/pow(2,floor(X/80))),2),235,16)"  # a block is 80 px wide
    media.run(["-f", "lavfi", "-i",
               f"color=black:size=640x360:rate={fps}:duration={seconds},format=yuv420p,"
               f"geq=lum='{bit}':cb=128:cr=128",
               "-frames:v", str(fps * seconds), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "16", "-pix_fmt", "yuv420p", str(path)])
    return path


def image_code(path):
    """The frame number shown in the code blocks of a 1080x608 work-sized picture."""
    with Image.open(path) as image:
        gray = image.convert("L")
        return sum((gray.getpixel((135 * bit + 67, 300)) > 128) << bit for bit in range(8))


def draw_swatches(image):
    """Draw the palette swatches along the bottom of an RGB or RGBA image of the size of the frames."""
    draw = ImageDraw.Draw(image)
    alpha = (255,) if image.mode == "RGBA" else ()
    for color, box in zip(SWATCHES, SWATCH_BOXES):
        draw.rectangle(box, fill=color + alpha)


def swatch_error(picture):
    """How far the palette swatches in a picture are from their true colors: the largest
    difference in any channel. picture is the frame as RGB floats, shape (rows, columns, 3)."""
    worst = 0.0
    for color, (x0, y0, x1, y1) in zip(SWATCHES, SWATCH_BOXES):
        x, y = (x0 + x1) // 2, (y0 + y1) // 2
        shown = picture[y - 14:y + 15, x - 14:x + 15].reshape(-1, 3).mean(axis=0)
        worst = max(worst, float(np.abs(shown - color).max()))
    return worst


def player_rgb(video, frame):
    """One frame of a video of the work size as a BT.709 limited-range player shows it, as floats."""
    width, height = WORK
    raw = video.with_suffix(".player")
    media.run(["-i", str(video), "-vf",
               f"select=eq(n\\,{frame}),scale=in_color_matrix=bt709:in_range=tv:out_range=pc:"
               "flags=accurate_rnd+full_chroma_int,format=rgb24",
               "-fps_mode", "passthrough", "-frames:v", "1", "-f", "rawvideo", str(raw)])
    return np.fromfile(raw, dtype=np.uint8).reshape(height, width, 3).astype(float)


def stored_planes(video, frame):
    """The Y, Cb and Cr planes of one frame of a video of the work size, exactly as it stores them."""
    width, height = WORK
    raw = video.with_suffix(".yuv")
    media.run(["-i", str(video), "-vf", f"select=eq(n\\,{frame})", "-fps_mode", "passthrough",
               "-frames:v", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", str(raw)])
    data = np.fromfile(raw, dtype=np.uint8).astype(float)
    luma, chroma = data[:width * height], data[width * height:].reshape(2, height // 2, width // 2)
    return luma.reshape(height, width), chroma[0], chroma[1]


def patch_center(i, top=0):
    """The center of patch i, for a picture whose 1080x608 patch grid starts top rows down."""
    return 270 * (i % 4) + 135, top + 152 * (i // 4) + 76


def patch_rgb(picture, i, top=0):
    """The mean color of patch i in an RGB picture (floats, shape (rows, columns, 3))."""
    x, y = patch_center(i, top)
    return picture[y - 14:y + 15, x - 14:x + 15].reshape(-1, 3).mean(axis=0)


def patch_ycc(planes, i, top=0):
    """The mean Y, Cb and Cr of patch i in the planes of a frame, as stored."""
    x, y = patch_center(i, top)
    luma, cb, cr = planes
    return np.array([luma[y - 14:y + 15, x - 14:x + 15].mean(),
                     cb[y // 2 - 7:y // 2 + 8, x // 2 - 7:x // 2 + 8].mean(),
                     cr[y // 2 - 7:y // 2 + 8, x // 2 - 7:x // 2 + 8].mean()])


def make_patch_clip(path, kind):
    """A 1080x608 clip of 6 frames of flat patches in known colors, stored as a camera could:
    "tagged" is BT.709 limited range with its tags, "untagged" the same picture with no color
    tags at all, and "full-range" is BT.709 full range, tagged as such."""
    grid = Image.new("RGB", WORK)
    draw = ImageDraw.Draw(grid)
    for i, color in enumerate(PATCHES):
        x, y = patch_center(i, 0)
        draw.rectangle((x - 135, y - 76, x + 134, y + 75), fill=color)
    grid_png = path.with_suffix(".png")
    grid.save(grid_png)

    full = kind == "full-range"
    pix_fmt = "yuvj420p" if full else "yuv420p"
    convert = f"scale=out_color_matrix=bt709:out_range={'pc' if full else 'tv'},format={pix_fmt}"
    tags = ["-colorspace", "bt709", "-color_range", "pc" if full else "tv"]
    if kind == "untagged":
        convert += ",setparams=colorspace=unknown:color_primaries=unknown:color_trc=unknown"
        tags = []
    media.run(["-loop", "1", "-framerate", "30", "-i", str(grid_png), "-vf", convert, *tags,
               "-frames:v", "6", "-c:v", "libx264", "-preset", "veryfast", "-crf", "10",
               "-pix_fmt", pix_fmt, str(path)])
    return path


def freeze_flow(kind, folder):
    """What a render does around the freeze, for a camera clip of the given kind: the head from
    encode_clip, the card from the frozen frame (extract_frames, Pillow, encode_stills) with
    palette art over it, and concat. Returns the head, the card, the joined video and the frozen
    frame as RGB floats. In the joined video, frame 3 is the head's last and frame 4 the card's
    first."""
    folder.mkdir()
    camera = make_patch_clip(folder / "camera.mp4", kind)
    head, card, final = (folder / f"{name}.mp4" for name in ("head", "card", "final"))
    media.encode_clip(camera, head, Segment("clip", 4, 0.0, 1.0), WORK, WORK)

    media.extract_frames(camera, folder / "frames", WORK)
    with Image.open(folder / "frames" / "000003.jpg") as jpeg:
        frozen = jpeg.convert("RGB")
    art = frozen.copy()
    draw_swatches(art)  # the card: the frozen frame, with palette art over it
    art.save(folder / "card.png")
    media.encode_stills([(folder / "card.png", 4)], card)

    media.concat([head, card], final)
    return head, card, final, np.asarray(frozen, dtype=float)


@pytest.fixture(scope="module")
def code_clip(tmp_path_factory):
    """A 4 s code clip at 30 fps (120 frames): frame N shows N."""
    return make_code_clip(tmp_path_factory.mktemp("code") / "code.mp4", 30, 4)


# --- probe ----------------------------------------------------------------------------------


def test_probe_synthetic(synthetic_clip, tmp_path):
    info = media.probe(synthetic_clip)

    assert (info["width"], info["height"]) == (1920, 1080)
    assert info["fps"] == 30.0
    assert info["frames"] == 120  # counted by decoding, not read from the header
    assert info["duration"] == pytest.approx(4.0, abs=0.05)
    assert info["rotation"] == 0
    assert (info["color_transfer"], info["hdr"]) == (None, False)  # a normal (SDR) clip
    assert media.count_frames(synthetic_clip) == 120

    # 1.2 is where a picture stops being square; the test is exact, with no float rounding
    assert media.orientation(info["width"], info["height"]) == "landscape"
    assert media.orientation(1200, 1000) == "landscape"
    assert media.orientation(1000, 1200) == "portrait"
    assert media.orientation(1199, 1000) == "square"
    assert media.orientation(1000, 1199) == "square"
    assert media.orientation(1080, 1080) == "square"

    # a failing call is a MediaError that carries ffmpeg's own message
    with pytest.raises(MediaError, match="missing.mp4"):
        media.probe(tmp_path / "missing.mp4")
    with pytest.raises(MediaError, match="nope.mp4"):
        media.run(["-i", str(tmp_path / "nope.mp4"), str(tmp_path / "out.mp4")])
    # and a file with sound but no picture says so, instead of crashing on the missing stream
    sound_only = tmp_path / "sound.m4a"
    media.run(["-i", str(synthetic_clip), "-vn", "-c:a", "copy", str(sound_only)])
    with pytest.raises(MediaError, match="no video"):
        media.probe(sound_only)


def test_probe_applies_rotation(portrait_rotated_clip, synthetic_clip, tmp_path):
    info = media.probe(portrait_rotated_clip)

    # stored as 1920x1080 with a display rotation of 90, shown as 1080x1920
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["rotation"] in (90, 270)
    assert media.orientation(info["width"], info["height"]) == "portrait"

    # ffmpeg turns the picture upright when it decodes, so frames cut at the rotated size come out
    # upright. The stored picture is sideways, and a display rotation of 90 is counterclockwise.
    stored = tmp_path / "stored.png"
    media.run(["-noautorotate", "-i", str(portrait_rotated_clip), "-frames:v", "1", str(stored)])
    frames = tmp_path / "frames"
    assert media.extract_frames(portrait_rotated_clip, frames, (1080, 1920)) > 0
    with Image.open(stored) as sideways, Image.open(frames / "000000.jpg") as extracted:
        upright = np.asarray(sideways.convert("RGB").rotate(90, expand=True), dtype=int)
        cut = np.asarray(extracted.convert("RGB"), dtype=int)
    assert cut.shape == (1920, 1080, 3)
    assert np.abs(cut - upright).mean() < 3
    assert np.abs(cut - np.rot90(upright, 2)).mean() > 30  # the wrong direction would not match

    # What probe reports is always the size of the frames ffmpeg really decodes. ffmpeg turns the
    # picture for a display rotation, but not for the old "rotate" tag, so probe ignores that tag.
    decoded = tmp_path / "decoded.png"
    media.run(["-i", str(portrait_rotated_clip), "-frames:v", "1", str(decoded)])
    with Image.open(decoded) as image:
        assert image.size == (info["width"], info["height"])
    tag_only = tmp_path / "tag-only.mkv"
    media.run(["-i", str(synthetic_clip), "-t", "1", "-an", "-c", "copy", "-metadata:s:v:0",
               "rotate=90", str(tag_only)])
    media.run(["-i", str(tag_only), "-frames:v", "1", str(decoded)])
    tagged = media.probe(tag_only)
    with Image.open(decoded) as image:
        assert image.size == (tagged["width"], tagged["height"]) == (1920, 1080)
    assert tagged["rotation"] == 0


# --- extract_frames -------------------------------------------------------------------------


def test_extract_frames_size_and_count(synthetic_clip, tmp_path):
    # the crop is plain numbers: the picture is scaled to cover the work size, centered across,
    # and cut so that focus_y of the source stays at the same relative height, clamped to the edges
    cut = media.frame_filter
    assert cut(1920, 1080, (1080, 608)) == "fps=30,scale=1081:608,crop=1080:608:0:0"
    assert cut(1920, 1080, (1080, 1920)) == "fps=30,scale=3414:1920,crop=1080:1920:1167:0"
    assert cut(1080, 608, (1080, 608)) == "fps=30,scale=1080:608,crop=1080:608:0:0"
    assert cut(1080, 1920, (1080, 608)) == "fps=30,scale=1080:1920,crop=1080:608:0:551"
    assert cut(1000, 1777, (1080, 608)) == "fps=30,scale=1080:1920,crop=1080:608:0:551"  # 1919.16
    assert cut(1080, 1920, (1080, 608), focus_y=0.0).endswith("crop=1080:608:0:0")
    assert cut(1080, 1920, (1080, 608), focus_y=1.0).endswith("crop=1080:608:0:1312")
    assert cut(1080, 1920, (1080, 608), focus_y=1.7).endswith("crop=1080:608:0:1312")
    assert cut(1080, 1920, (1080, 608), focus_y=-0.5).endswith("crop=1080:608:0:0")
    assert cut(1920, 1080, (1080, 608), fps=24).startswith("fps=24,scale=")

    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "000500.jpg").write_bytes(b"left over from an earlier run")

    count = media.extract_frames(synthetic_clip, frames, WORK)

    assert count == 120
    # the leftover is gone, so the folder and the count agree
    assert sorted(path.name for path in frames.iterdir()) == [f"{n:06d}.jpg" for n in range(120)]
    for path in frames.iterdir():
        with Image.open(path) as image:
            assert image.size == (1080, 608)


def test_extract_frames_path_with_spaces_and_accents(synthetic_clip, tmp_path):
    folder = tmp_path / "Vídeos de João"
    folder.mkdir()
    clip = folder / "my clip.mp4"
    shutil.copy(synthetic_clip, clip)
    # a percent sign in the folder would read as part of the file name pattern if left alone
    frames = tmp_path / "Quadros 100% de João" / "frames"

    assert media.probe(clip)["frames"] == 120
    assert media.extract_frames(clip, frames, WORK) == 120
    assert len(list(frames.glob("*.jpg"))) == 120


def test_extract_frames_can_cut_a_stretch_of_the_clip_at_any_size(code_clip, tmp_path):
    # A final render cuts the freeze frame, and the frames the matte warms up on, at the size of the output.
    # first and last are frame numbers, both written, numbered from 000000 on; the code clip says which
    # frame each picture is, and the frames after the last are never decoded.
    stretch = tmp_path / "stretch"
    assert media.extract_frames(code_clip, stretch, WORK, first=48, last=60) == 13
    assert sorted(path.name for path in stretch.iterdir()) == [f"{n:06d}.jpg" for n in range(13)]
    assert [image_code(stretch / f"{n:06d}.jpg") for n in range(13)] == list(range(48, 61))

    # the freeze frame on its own, which is also how the first frame is cut; and a stretch that runs out
    one = tmp_path / "one"
    assert media.extract_frames(code_clip, one, WORK, first=0, last=0) == 1
    assert [path.name for path in one.iterdir()] == ["000000.jpg"] and image_code(one / "000000.jpg") == 0
    past = tmp_path / "past"
    assert media.extract_frames(code_clip, past, WORK, first=118, last=125) == 2
    assert [image_code(path) for path in sorted(past.iterdir())] == [118, 119]
    # without first and last it is every frame, as it was
    assert media.extract_frames(code_clip, tmp_path / "all", WORK) == 120

    # at the size of an output that is not the plan's: 1920x1080 here, which the clip (640x360) is
    # scaled up to, and the same frames come out
    big = tmp_path / "big"
    assert media.extract_frames(code_clip, big, (1920, 1080), first=10, last=12) == 3
    with Image.open(big / "000000.jpg") as image:
        assert image.size == (1920, 1080)
        gray = image.convert("L")
        assert sum((gray.getpixel((240 * bit + 120, 540)) > 128) << bit for bit in range(8)) == 10


# --- encode_clip ----------------------------------------------------------------------------


def test_encode_clip_exact_frames(synthetic_clip, code_clip, tmp_path):
    seg = Segment("clip", 62, 0.9333, 1.0)
    out = tmp_path / "head.mp4"

    media.encode_clip(synthetic_clip, out, seg, (1920, 1080), WORK)

    # exactly 62 frames of H.264, yuv420p, 1080x608 at 30 fps, and no sound
    assert media.count_frames(out) == 62
    info = media.probe(out)
    assert (info["width"], info["height"], info["fps"]) == (1080, 608, 30.0)
    assert [(s["codec_type"], s["codec_name"], s["pix_fmt"]) for s in streams(out)] == [
        ("video", "h264", "yuv420p")
    ]
    assert_x264(out, 16, "medium")
    # a normal video: BT.709 limited range, tagged so on every part
    assert color_tags(out) == BT709_TAGS
    assert (media.probe(out)["color_transfer"], media.probe(out)["hdr"]) == ("bt709", False)

    # the clip fills the frame, with nothing around it: the first frame is frame 28 of the clip (0.9333 s in),
    # cut to the same size, and there are no bars at any edge
    frames = tmp_path / "frames"
    media.extract_frames(synthetic_clip, frames, WORK)
    with Image.open(frames / "000028.jpg") as jpeg:
        wanted = np.asarray(jpeg.convert("RGB"), dtype=float)
    shown = player_rgb(out, 0)
    assert np.abs(shown - wanted).mean() < 3
    for edge in (shown[:6], shown[-6:], shown[:, :6], shown[:, -6:]):
        assert edge.mean() > 30

    # it is the right stretch of the clip: the head starts at source frame 28 (0.9333 s) and goes
    # on one frame at a time. The code clip, in a preview, tells which frames were kept.
    def preview_of(segment):
        preview = tmp_path / f"preview-{segment.frames}-{segment.src_start}-{segment.speed}.mp4"
        media.encode_clip(code_clip, preview, segment, (640, 360), WORK, preview=True)
        return preview

    head = preview_of(seg)
    info = media.probe(head)
    assert (info["width"], info["height"], info["fps"]) == (1080, 608, 30.0)  # (a preview is made fast, not small)
    assert_x264(head, 28, "ultrafast")
    assert color_tags(head) == BT709_TAGS
    assert frame_codes(head, FRAME) == list(range(28, 90))

    # a clip that runs out a frame short gets its last frame cloned, so the count stays exact
    one_short = preview_of(Segment("clip", 31, 3.0, 1.0))  # frames 90 to 119 are 30 frames
    assert frame_codes(one_short, FRAME) == list(range(90, 120)) + [119]
    far_short = preview_of(Segment("clip", 20, 3.7, 1.0))  # 9 frames of clip, 11 clones
    assert frame_codes(far_short, FRAME) == list(range(111, 120)) + [119] * 11

    # none of the clip's metadata reaches the part (a phone clip carries where it was shot)
    tagged = tmp_path / "tagged.mp4"
    media.run(["-i", str(code_clip), "-c", "copy", "-metadata", "title=Private title",
               "-metadata", "location=+52.5200+013.4050/", str(tagged)])
    private = tmp_path / "private.mp4"
    media.encode_clip(tagged, private, Segment("clip", 5, 0.0, 1.0), (640, 360), WORK, preview=True)
    assert {"title", "location"} <= format_tags(tagged).keys()
    assert not {"title", "location"} & format_tags(private).keys()

    # a start past the end of the clip leaves nothing to clone: an error, not an empty file
    with pytest.raises(MediaError, match="expected 10"):
        preview_of(Segment("clip", 10, 4.0, 1.0))


def test_encode_clip_is_the_clip_at_the_size_it_is_given(code_clip, tmp_path):
    # The size is the output's: the clip's own, or scaled up to 1080 wide (this one is 640x360), and the
    # frames are the same ones whatever it is
    for size in ((1920, 1080), (1080, 608), (320, 180)):
        out = tmp_path / f"part-{size[0]}.mp4"
        media.encode_clip(code_clip, out, Segment("clip", 12, 0.5, 1.0), (640, 360), size, preview=True)
        info = media.probe(out)
        assert (info["width"], info["height"], info["frames"]) == (*size, 12), size
        assert frame_codes(out, (0, 0, *size)) == list(range(15, 27)), size

    # a 9:16 clip is the clip's own shape, and is not put in a canvas
    tall = tmp_path / "tall.mp4"
    media.run(["-f", "lavfi", "-i", "color=blue:size=360x640:rate=30:duration=1", "-c:v", "libx264",
               "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(tall)])
    part = tmp_path / "tall-part.mp4"
    media.encode_clip(tall, part, Segment("clip", 10, 0.0, 1.0), (360, 640), (1080, 1920), preview=True)
    info = media.probe(part)
    assert (info["width"], info["height"]) == (1080, 1920)
    assert frame_colors(part) == ["blue"] * 10


def test_encode_clip_speed(synthetic_clip, code_clip, tmp_path):
    out = tmp_path / "fast.mp4"

    media.encode_clip(synthetic_clip, out, Segment("clip", 62, 0.0, 1.45), (1920, 1080), WORK)

    assert media.count_frames(out) == 62

    # played at 1.45x, output frame i shows source frame 1.45 * i, give or take a frame or two
    # (setpts and fps keep whole frames, so the pick is never closer than that)
    fast = tmp_path / "fast-preview.mp4"
    media.encode_clip(code_clip, fast, Segment("clip", 62, 0.0, 1.45), (640, 360), WORK, preview=True)
    codes = frame_codes(fast, FRAME)
    assert len(codes) == 62
    assert all(abs(code - 1.45 * i) <= 2.0 for i, code in enumerate(codes))
    steps = {after - before for before, after in zip(codes, codes[1:])}
    assert min(steps) >= 1 and 2 in steps  # always forward, and some source frames skipped
    assert codes[-1] >= 86

    # slower than normal, from 1.0 s in, repeats frames and keeps the start
    slow = tmp_path / "slow-preview.mp4"
    media.encode_clip(code_clip, slow, Segment("clip", 30, 1.0, 0.75), (640, 360), WORK, preview=True)
    codes = frame_codes(slow, FRAME)
    assert len(codes) == 30
    assert all(abs(code - (30 + 0.75 * i)) <= 1.0 for i, code in enumerate(codes))
    assert len(set(codes)) < 30  # some frames play twice

    # A clip shot at 60 fps is brought down to 30 first. extract_frames and encode_clip number the
    # frames of that 30 fps clip alike, so frame i is the same picture in both, and a time in
    # seconds means the same frame everywhere.
    sixty = make_code_clip(tmp_path / "sixty.mp4", 60, 2)
    frames = tmp_path / "frames"
    assert media.extract_frames(sixty, frames, WORK) == 60
    for i in (0, 10, 25):
        part = tmp_path / f"sixty-{i}.mp4"
        media.encode_clip(sixty, part, Segment("clip", 1, i / 30, 1.0), (640, 360), WORK, preview=True)
        (shown,) = frame_codes(part, FRAME)
        assert shown == image_code(frames / f"{i:06d}.jpg")
        assert abs(shown - 2 * i) <= 1  # frame 2 * i at 60 fps is the time i / 30


# --- encode_stills --------------------------------------------------------------------------


def test_encode_stills_durations(tmp_path):
    folder = tmp_path / "Cartões d'água"  # an accent, a space and a quote, all in the concat list
    folder.mkdir()
    red, blue = folder / "João's red card.png", folder / "blue card.png"
    Image.new("RGB", WORK, (255, 0, 0)).save(red)
    Image.new("RGB", WORK, (0, 0, 255)).save(blue)
    out = tmp_path / "card.mp4"

    media.encode_stills([(red, 10), (blue, 20)], out)

    assert media.count_frames(out) == 30
    info = media.probe(out)
    assert (info["width"], info["height"], info["fps"]) == (1080, 608, 30.0)
    assert_x264(out, 16, "medium")
    assert color_tags(out) == BT709_TAGS
    assert frame_colors(out) == ["red"] * 10 + ["blue"] * 20  # and the cut is on frame 10

    # short runs, like card states that last 2 or 3 frames, change on exactly the right frame
    runs = [(red, 2), (blue, 3), (red, 7), (blue, 7), (red, 7), (blue, 7), (red, 6)]
    many = tmp_path / "many.mp4"
    media.encode_stills(runs, many)
    names = {red: "red", blue: "blue"}
    assert frame_colors(many) == [names[still] for still, frames in runs for _ in range(frames)]

    # a preview has the same frames at the same size (it is made fast, not small), and a single still can
    # be held for any length
    preview = tmp_path / "card-preview.mp4"
    media.encode_stills([(red, 10), (blue, 20)], preview, preview=True)
    info = media.probe(preview)
    assert (info["width"], info["height"], info["frames"]) == (1080, 608, 30)
    assert_x264(preview, 28, "ultrafast")
    assert color_tags(preview) == BT709_TAGS
    assert frame_colors(preview) == ["red"] * 10 + ["blue"] * 20
    held = tmp_path / "held.mp4"
    media.encode_stills([(blue, 7)], held)
    assert frame_colors(held) == ["blue"] * 7

    # palette art in a card lands on its hex, as a BT.709 player shows it
    palette = Image.new("RGB", WORK, "black")
    draw_swatches(palette)
    palette.save(tmp_path / "palette.png")
    swatched = tmp_path / "swatched.mp4"
    media.encode_stills([(tmp_path / "palette.png", 3)], swatched)
    assert swatch_error(player_rgb(swatched, 0)) <= LEVELS


# --- concat, mux_audio and mix_click --------------------------------------------------------


def test_concat_and_mux(synthetic_clip, code_clip, tmp_path):
    # two 30-frame parts of different kinds, as a render makes them: a clip, then a held picture
    part_a = tmp_path / "part-a.mp4"
    part_b = tmp_path / "part-b.mp4"
    media.encode_clip(code_clip, part_a, Segment("clip", 30, 0.0, 1.0), (640, 360), WORK)
    media.encode_stills([(code_png(tmp_path / "card.png", 200), 30)], part_b)
    video = tmp_path / "video.mp4"

    media.concat([part_a, part_b], video)

    assert media.count_frames(video) == 60
    info = media.probe(video)
    assert (info["width"], info["height"], info["fps"]) == (1080, 608, 30.0)
    assert [(s["codec_type"], s["codec_name"], s["pix_fmt"]) for s in streams(video)] == [
        ("video", "h264", "yuv420p")
    ]
    # each part keeps its own frames and the join is on frame 30. The parts were made at crf 16,
    # and the join is a new encode at crf 18.
    assert frame_codes(video, FRAME) == list(range(30)) + [200] * 30
    assert_x264(part_a, 16, "medium")
    assert_x264(part_b, 16, "medium")
    assert_x264(video, 18, "medium")
    # the parts and the joined video carry the same tags: BT.709 limited range
    assert color_tags(part_a) == color_tags(part_b) == color_tags(video) == BT709_TAGS

    # mux_audio copies the picture and adds AAC sound cut to the video's length, fading out
    muxed = tmp_path / "muxed.mp4"
    media.mux_audio(video, synthetic_clip, muxed)
    kinds = {s["codec_type"]: s for s in streams(muxed)}
    assert sorted(kinds) == ["audio", "video"]
    assert kinds["video"]["codec_name"] == "h264"
    assert media.count_frames(muxed) == 60
    assert kinds["audio"]["codec_name"] == "aac"
    assert float(kinds["audio"]["duration"]) == pytest.approx(2.0, abs=0.05)  # the source has 4 s
    assert color_tags(muxed) == BT709_TAGS  # the picture is copied, tags and all
    sound = mono_samples(muxed)
    body = rms(sound, 0.2, 0.4)
    assert body > 0.05
    assert rms(sound, 1.95, 2.0) < 0.2 * body
    flat = tmp_path / "flat.mp4"
    media.mux_audio(video, synthetic_clip, flat, fade_out=0.0)
    assert rms(mono_samples(flat), 1.95, 2.0) > 0.8 * body  # no fade, the sound runs to the end

    # mix_click puts the click on the second it was asked for. On its own it is silence around it.
    click = tmp_path / "click.wav"
    media.run(["-f", "lavfi", "-i", f"sine=frequency=1000:sample_rate={SOUND_RATE}:duration=0.1",
               str(click)])
    clicked = tmp_path / "clicked.mp4"
    media.mix_click(video, click, 1.0, clicked)
    kinds = {s["codec_type"]: s for s in streams(clicked)}
    assert kinds["video"]["codec_name"] == "h264" and media.count_frames(clicked) == 60
    assert color_tags(clicked) == BT709_TAGS
    assert kinds["audio"]["codec_name"] == "aac"
    assert float(kinds["audio"]["duration"]) == pytest.approx(2.0, abs=0.05)
    sound = mono_samples(clicked)
    assert rms(sound, 0.0, 0.95) < 0.001
    assert rms(sound, 1.0, 1.1) > 0.05
    assert rms(sound, 1.15, 2.0) < 0.001

    # with music, the click rides on top of the music, which is cut to the video and fades out
    with_music = tmp_path / "with-music.mp4"
    media.mix_click(video, click, 1.0, with_music, music=synthetic_clip)
    assert float({s["codec_type"]: s for s in streams(with_music)}["audio"]["duration"]) == (
        pytest.approx(2.0, abs=0.05))
    sound = mono_samples(with_music)
    assert rms(sound, 0.2, 0.4) == pytest.approx(body, rel=0.1)  # the music is not turned down
    assert rms(sound, 1.0, 1.1) > 1.2 * body  # music and click together
    assert rms(sound, 1.95, 2.0) < 0.2 * body
    assert color_tags(with_music) == BT709_TAGS

    # The freeze, for the three ways a camera can store a clip. Whatever the clip, the head, the
    # card and the joined video carry the same BT.709 tags. The frozen frame, as Pillow reads the
    # JPEG, has the clip's true colors. The head shows the clip's colors too (a full-range clip
    # is converted to limited range). The person keeps its colors across the cut, and palette
    # art lands on its hex through the card.
    for kind in ("tagged", "untagged", "full-range"):
        head, card, final, frozen = freeze_flow(kind, tmp_path / kind)

        assert color_tags(head) == color_tags(card) == color_tags(final) == BT709_TAGS, kind
        for i, color in enumerate(PATCHES):
            assert np.abs(patch_rgb(frozen, i) - color).max() <= JPEG_LEVELS, (kind, "jpeg", i)

        # frame 3 is the head's last, frame 4 the card's first
        before_cut, after_cut = player_rgb(final, 3), player_rgb(final, 4)
        for i, color in enumerate(PATCHES):
            assert np.abs(patch_rgb(before_cut, i) - color).max() <= LEVELS, (kind, "head", i)
        assert swatch_error(after_cut) <= LEVELS, kind

        last_of_head, first_of_card = stored_planes(final, 3), stored_planes(final, 4)
        for i in range(PERSON):
            seam = np.abs(patch_ycc(last_of_head, i) - patch_ycc(first_of_card, i))
            assert seam.max() <= LEVELS, (kind, "seam", i, seam)

    # concat tags what it writes whatever the parts carried: here the parts have no tags at all
    camera = tmp_path / "untagged" / "camera.mp4"
    assert color_tags(camera)["frame"]["color_space"] == "unknown"
    joined = tmp_path / "joined.mp4"
    media.concat([camera, camera], joined)
    assert color_tags(joined) == BT709_TAGS


# --- HDR ------------------------------------------------------------------------------------


# The centers of the four patches of the HLG clip (red and yellow on top, cyan and gray below), in a
# picture of the work size
HLG_CENTERS = {"red": (270, 152), "yellow": (810, 152), "cyan": (270, 456), "gray": (810, 456)}


def patch_colors(picture):
    """The mean color of each of the HLG clip's four patches in a work-sized picture (floats)."""
    return {name: picture[y - 14:y + 15, x - 14:x + 15].reshape(-1, 3).mean(axis=0)
            for name, (x, y) in HLG_CENTERS.items()}


def extracted_colors(clip, folder):
    """The colors of the four patches in the 10th frame media.extract_frames cuts of the clip."""
    assert media.extract_frames(clip, folder, WORK) == 30
    with Image.open(folder / "000010.jpg") as jpeg:
        return patch_colors(np.asarray(jpeg.convert("RGB"), dtype=float))


def make_pq_clip(hlg_clip, path):
    """The HLG clip's picture converted to PQ (HDR10), the other kind of HDR: 3 frames."""
    media.run(["-i", str(hlg_clip), "-vf", "zscale=t=smpte2084:npl=100,format=yuv420p10le",
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p10le",
               "-frames:v", "3", str(path)])
    return path


def test_hdr_clip_is_tone_mapped(hlg_clip, tmp_path, monkeypatch):
    info = media.probe(hlg_clip)
    assert info["hdr"] is True
    assert info["color_transfer"] == "arib-std-b67"  # HLG, which iPhones shoot by default
    # PQ (HDR10) is HDR too: the same picture converted to that transfer
    pq = make_pq_clip(hlg_clip, tmp_path / "pq.mp4")
    assert (media.probe(pq)["color_transfer"], media.probe(pq)["hdr"]) == ("smpte2084", True)

    # With neither way to convert HDR (zscale and tonemap, or ffmpeg 8's own color management in the
    # scale filter), the person is told what to do, and nothing is written. Both functions say it,
    # in the same words, and the words say what to install on each system.
    head = Segment("clip", 4, 0.0, 1.0)
    said = set()
    for missing in ({"zscale"}, {"tonemap"}, {"zscale", "tonemap"}):  # either one is enough
        with monkeypatch.context() as patched:
            patched.setattr(media, "has_filter", lambda name, gone=missing: name not in gone)
            patched.setattr(media, "_swscale_tone_maps", lambda: False)
            assert media.hdr_support() is None
            with pytest.raises(MediaError) as error:
                media.extract_frames(hlg_clip, tmp_path / "none", WORK)
            said.add(str(error.value))
            with pytest.raises(MediaError) as error:
                media.encode_clip(hlg_clip, tmp_path / "none.mp4", head, (640, 360), WORK)
            said.add(str(error.value))
    assert len(said) == 1
    (message,) = said
    assert message.startswith(
        "This clip is HDR, and this ffmpeg build can't convert HDR to normal video. Turn off HDR "
        "video in your camera settings, or install ")
    assert "winget install Gyan.FFmpeg" in message  # Windows: the full build has zscale
    assert "brew install ffmpeg-full" in message and "PATH" in message  # a Mac: the keg on the PATH
    assert "ffmpeg 8" in message  # or a version that converts HDR by itself
    assert not (tmp_path / "none").exists() and not (tmp_path / "none.mp4").exists()
    assert media.has_filter("zscale") and media.has_filter("tonemap")
    assert not media.has_filter("no-such-filter")
    assert media.hdr_support() == "zscale"  # zscale is the way when this build has it

    # The clip is four patches: red and yellow on top, cyan and gray below. Converted to normal
    # video, saturated colors stay saturated. Read as plain BT.709 video (no conversion), the
    # red would come out (142, 61, 31) and the cyan (106, 149, 158), washed out.
    seen = extracted_colors(hlg_clip, tmp_path / "frames")
    assert seen["red"][0] > 100 and seen["red"][1:].max() < 25
    assert seen["cyan"][0] < 60 and seen["cyan"][1:].min() > 100
    assert seen["yellow"][:2].min() > 90 and seen["yellow"][2] < 35
    assert seen["gray"].max() - seen["gray"].min() < 8

    # encode_clip makes a normal part of it, tagged BT.709 limited range, that shows the same
    # picture as the extracted frame because both start with the same conversion
    part = tmp_path / "part.mp4"
    media.encode_clip(hlg_clip, part, Segment("clip", 8, 0.0, 1.0), (640, 360), WORK)
    assert media.count_frames(part) == 8
    assert color_tags(part) == BT709_TAGS
    assert (media.probe(part)["color_transfer"], media.probe(part)["hdr"]) == ("bt709", False)
    assert [(s["codec_name"], s["pix_fmt"]) for s in streams(part)] == [("h264", "yuv420p")]
    in_part = patch_colors(player_rgb(part, 3))
    for name in HLG_CENTERS:
        assert np.abs(in_part[name] - seen[name]).max() <= 8, (name, in_part[name], seen[name])


def test_hdr_clip_without_zscale_is_converted_by_ffmpegs_own_color_management(
    hlg_clip, tmp_path, monkeypatch
):
    # Homebrew's ffmpeg has no zscale, and an iPhone shoots HLG by default. ffmpeg 8 and newer can
    # bring the clip down all the same, in the scale filter (swscale's own color management). Here
    # zscale is hidden from this build, which has it: the picture has to come out right.
    through_zscale = extracted_colors(hlg_clip, tmp_path / "zscale")
    with monkeypatch.context() as patched:
        patched.setattr(media, "has_filter", lambda name: name != "zscale")
        assert media.hdr_support() == "swscale"

        seen = extracted_colors(hlg_clip, tmp_path / "swscale")
        # Gray stays gray, within the tolerance the zscale path has, and is neither black nor blown out
        assert seen["gray"].max() - seen["gray"].min() < 8
        assert 60 < seen["gray"].mean() < 110
        assert abs(seen["gray"].mean() - through_zscale["gray"].mean()) < 15  # the same picture
        # Converted, not relabeled: read as plain BT.709 video the cyan would keep a red of 106 and
        # the red a green of 61. The tone curve is not zscale's, so the colors are not as deep.
        assert seen["red"][0] > 100 and seen["red"][0] - seen["red"][1:].max() > 70
        assert seen["cyan"][0] < 90 and seen["cyan"][1:].min() > 100
        assert seen["yellow"][:2].min() > 90 and seen["yellow"][2] < 60

        # encode_clip makes a normal part of it, tagged BT.709 limited range, with the same picture
        part = tmp_path / "part-swscale.mp4"
        media.encode_clip(hlg_clip, part, Segment("clip", 8, 0.0, 1.0), (640, 360), WORK)
        assert media.count_frames(part) == 8
        assert color_tags(part) == BT709_TAGS
        assert (media.probe(part)["color_transfer"], media.probe(part)["hdr"]) == ("bt709", False)
        assert [(s["codec_name"], s["pix_fmt"]) for s in streams(part)] == [("h264", "yuv420p")]
        in_part = patch_colors(player_rgb(part, 3))
        for name in HLG_CENTERS:
            assert np.abs(in_part[name] - seen[name]).max() <= 8, (name, in_part[name], seen[name])

        # PQ (HDR10) goes the same way
        pq = make_pq_clip(hlg_clip, tmp_path / "pq.mp4")
        assert media.extract_frames(pq, tmp_path / "pq-frames", WORK) == 3
        with Image.open(tmp_path / "pq-frames" / "000001.jpg") as jpeg:
            gray = patch_colors(np.asarray(jpeg.convert("RGB"), dtype=float))["gray"]
        assert gray.max() - gray.min() < 8 and 40 < gray.mean() < 110


def test_hdr_support_is_zscale_when_the_build_has_it_and_swscale_otherwise(monkeypatch):
    monkeypatch.setattr(media, "has_filter", lambda name: True)
    monkeypatch.setattr(media, "_swscale_tone_maps", lambda: True)
    assert media.hdr_support() == "zscale"  # both ways there: zscale is the one that has been used
    monkeypatch.setattr(media, "has_filter", lambda name: name == "tonemap")
    assert media.hdr_support() == "swscale"
    monkeypatch.setattr(media, "has_filter", lambda name: name == "zscale")  # zscale alone is no way
    assert media.hdr_support() == "swscale"
    monkeypatch.setattr(media, "_swscale_tone_maps", lambda: False)
    assert media.hdr_support() is None


def test_swscale_tone_mapping_is_found_by_trying_it(monkeypatch):
    # This build is ffmpeg 8.1, so the real try works. A build whose scale filter does not know the
    # options (before ffmpeg 8) or whose ffmpeg fails is told "no", never an error.
    media._swscale_tone_maps.cache_clear()
    try:
        assert media._swscale_tone_maps() is True

        def refuse(command):
            raise MediaError("ffmpeg failed: Option out_transfer not found.")

        media._swscale_tone_maps.cache_clear()
        monkeypatch.setattr(media, "_exec", refuse)
        assert media._swscale_tone_maps() is False
    finally:
        monkeypatch.undo()
        media._swscale_tone_maps.cache_clear()


FILTER_LEGEND = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
"""


@pytest.mark.parametrize("listing", [
    # ffmpeg 8: three flag letters
    FILTER_LEGEND + " ... zscale            V->V       Apply resizing and color space conversion.\n"
                    " T.C tonemap           V->V       Conversion to different dynamic ranges.\n"
                    " ... amovie            |->N       Read audio from a movie source.\n",
    # one more column of flags, as a later ffmpeg could add
    FILTER_LEGEND + " .... zscale           V->V       Apply resizing and color space conversion.\n"
                    " T.C. tonemap          V->V       Conversion to different dynamic ranges.\n"
                    " .... amovie           |->N       Read audio from a movie source.\n",
    # a flag letter nobody has used yet, and the I/O column wider
    FILTER_LEGEND + " TSCX zscale           VV->VV     Apply resizing and color space conversion.\n"
                    " T.CQ tonemap          V->V       Conversion to different dynamic ranges.\n"
                    " ...Z amovie           |->N       Read audio from a movie source.\n",
    # flag letters a space apart
    FILTER_LEGEND + " T S zscale            V->V       Apply resizing and color space conversion.\n"
                    " T . C tonemap         V->V       Conversion to different dynamic ranges.\n"
                    " . . . amovie          |->N       Read audio from a movie source.\n",
], ids=["three-flags", "four-flags", "unknown-letters", "spaced-flags"])
def test_the_filter_list_is_read_whatever_the_flag_columns_are(listing):
    names = media._parse_filter_names(listing)

    assert {"zscale", "tonemap", "amovie"} <= names
    # the legend is not filters, and a description is not a name
    assert not {"Timeline", "support", "Source", "Apply", "Filters:", "="} & names


def test_the_filter_list_of_this_build_has_the_filters_this_module_leans_on():
    assert media.has_filter("scale") and media.has_filter("fps") and media.has_filter("setparams")
    assert media.has_filter("tpad") and media.has_filter("afade") and media.has_filter("amix")
    assert not media.has_filter("no-such-filter")


def test_encoders_are_found_in_the_list_and_libx264_is_there():
    listing = """Encoders:
 V..... = Video
 A..... = Audio
 S..... = Subtitle
 .F.... = Frame-level multithreading
 ..S... = Slice-level multithreading
 ...X.. = Codec is experimental
 ....B. = Supports draw_horiz_band
 .....D = Supports direct rendering method 1
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 VFS... mpeg4                MPEG-4 part 2
 V.....X newcodec            A flag column wider than today's
 A....D aac                  AAC (Advanced Audio Coding)
"""
    names = media._parse_encoder_names(listing)

    assert names == {"libx264", "mpeg4", "newcodec", "aac"}
    assert media.has_encoder("libx264") and media.has_encoder("aac")
    assert not media.has_encoder("no-such-encoder")


def test_a_missing_ffmpeg_or_ffprobe_is_a_plain_error(synthetic_clip, tmp_path, monkeypatch):
    # With no ffmpeg or ffprobe on the PATH, the person is told which one is missing and where to
    # look, not "[WinError 2] The system cannot find the file specified"
    monkeypatch.setenv("PATH", str(tmp_path))  # a folder with nothing in it
    for tool, call in (
        ("ffmpeg", lambda: media.run(["-version"])),
        ("ffprobe", lambda: media.probe(synthetic_clip, count=False)),
    ):
        with pytest.raises(MediaError) as error:
            call()
        assert str(error.value).startswith(f"{tool} isn't installed, or isn't on the PATH."), tool
        assert "python mc.py doctor" in str(error.value)
        assert "WinError" not in str(error.value) and "Errno" not in str(error.value)
    assert str(media.missing_tool("ffprobe")).startswith("ffprobe isn't installed")


def test_a_picture_is_not_a_clip(tmp_path):
    # A photo given as the clip used to be a KeyError: 'duration'
    for suffix in (".jpg", ".png", ".webp", ".bmp"):
        photo = tmp_path / f"photo{suffix}"
        Image.new("RGB", (64, 48), "red").save(photo)
        for count in (False, True):
            with pytest.raises(MediaError) as error:
                media.probe(photo, count=count)
            assert str(error.value) == (
                "That's a picture, not a video. Give me a clip of 2 to 30 seconds."), suffix


def test_a_file_with_no_length_that_is_no_picture_says_so_too(tmp_path, monkeypatch):
    nameless = {"streams": [{"width": 640, "height": 360, "r_frame_rate": "30/1"}],
                "format": {"format_name": "mpegts"}}
    monkeypatch.setattr(media, "_exec", lambda command: json.dumps(nameless))

    with pytest.raises(MediaError, match=r"^clip\.ts has no length I can read\. Give me a clip of 2 to 30"):
        media.probe(tmp_path / "clip.ts", count=False)


def canned_probe(monkeypatch, **stream):
    """Make probe read this stream (over a 640x360 one at 30 fps, 2 s long) as if ffprobe said it."""
    shown = {"streams": [{"width": 640, "height": 360, "r_frame_rate": "30/1", "duration": "2.0",
                          **stream}], "format": {"duration": "2.0"}}
    monkeypatch.setattr(media, "_exec", lambda command: json.dumps(shown))
    return lambda: media.probe(Path("canned.mp4"), count=False)


def test_a_variable_frame_rate_clip_is_read_at_its_average_not_its_fastest_gap(monkeypatch):
    # A phone that varies its frame rate states r_frame_rate as its fastest gap (60 here) and the
    # average as it really runs (30000/1001). Read as 60, the clip came out doubled.
    probe = canned_probe(monkeypatch, r_frame_rate="60/1", avg_frame_rate="30000/1001")
    assert probe()["fps"] == pytest.approx(30000 / 1001)
    assert media.output_fps(probe()) == Fraction(30000, 1001)

    # A steady clip, a stream that states no average, and the "0/0" ffprobe writes for an unknown
    # one, are read at r_frame_rate as before
    for stream in ({"avg_frame_rate": "30/1"}, {}, {"avg_frame_rate": "0/0"}, {"avg_frame_rate": "x/y"}):
        assert canned_probe(monkeypatch, **stream)()["fps"] == 30.0, stream

    # 5%: r_frame_rate more than 5% above the average is the fastest gap of a variable rate;
    # within that it is rounding noise (a stream of 30 fps whose average is 29.1 is a 30 fps stream)
    assert canned_probe(monkeypatch, r_frame_rate="30/1", avg_frame_rate="29100/1000")()["fps"] == 30.0
    assert canned_probe(monkeypatch, r_frame_rate="30/1", avg_frame_rate="28500/1000")()["fps"] == 28.5
    # an average above r_frame_rate is never used
    assert canned_probe(monkeypatch, r_frame_rate="30/1", avg_frame_rate="60/1")()["fps"] == 30.0


@pytest.fixture(scope="module")
def vfr_clip(tmp_path_factory):
    """A 4 s clip with jittered timestamps, as a phone writes when it varies its frame rate: a 60 fps
    picture of which every even frame and every 20th odd frame is kept, so the gaps are 1/60 s and
    1/30 s in no steady beat. ffprobe calls its r_frame_rate 60 (the fastest gap) and its average
    about 33.1."""
    path = tmp_path_factory.mktemp("vfr") / "vfr.mp4"
    media.run(["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=60:duration=4",
               "-vf", "select='not(mod(n,2))+eq(mod(n,20),1)'", "-fps_mode", "vfr",
               "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)])
    return path


def test_a_clip_with_jittered_timestamps_is_not_doubled_to_60_fps(vfr_clip, tmp_path):
    shown = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,avg_frame_rate", "-of", "json", str(vfr_clip)],
        check=True, capture_output=True, text=True).stdout
    stream = json.loads(shown)["streams"][0]
    assert stream["r_frame_rate"] == "60/1"  # the file really is the case: 60 for its fastest gap
    average = float(Fraction(stream["avg_frame_rate"]))
    assert 30 < average < 36

    info = media.probe(vfr_clip, count=False)

    assert info["fps"] == pytest.approx(average)
    assert media.output_fps(info) == Fraction(round(average))  # about 33, never 60
    assert media.probe(vfr_clip)["fps"] == info["fps"]
    # and the frames a render cuts are counted at that rate, so the clip is not stretched
    frames = media.extract_frames(vfr_clip, tmp_path / "frames", (1080, 608), media.output_fps(info))
    assert frames == pytest.approx(info["duration"] * round(average), abs=2)


def test_mix_click_keeps_the_click_when_the_music_ends_early(synthetic_clip, tmp_path):
    # The click lands at 3.0 s, after a track that is only 1 s long has ended. It used to be dropped,
    # because the mix lasted as long as its first input, the music.
    click = tmp_path / "click.wav"
    media.run(["-f", "lavfi", "-i", f"sine=frequency=1000:sample_rate={SOUND_RATE}:duration=0.1",
               str(click)])
    short_music = tmp_path / "short.wav"
    media.run(["-f", "lavfi", "-i", f"sine=frequency=330:sample_rate={SOUND_RATE}:duration=1",
               str(short_music)])
    mixed = tmp_path / "mixed.mp4"

    media.mix_click(synthetic_clip, click, 3.0, mixed, music=short_music)

    assert float({s["codec_type"]: s for s in streams(mixed)}["audio"]["duration"]) == (
        pytest.approx(4.0, abs=0.05))  # the video's length
    sound = mono_samples(mixed)
    assert rms(sound, 0.1, 0.8) > 0.05  # the music, while it lasts
    assert rms(sound, 1.2, 2.9) < 0.001  # then silence
    assert rms(sound, 3.0, 3.1) > 0.05  # and the click, at its second
    assert rms(sound, 3.2, 3.9) < 0.001


# --- clip_sound: the clip's own sound, laid out on the timeline ----------------------------


@pytest.fixture(scope="module")
def beep_clip(tmp_path_factory):
    """A 4 s, 30 fps clip whose sound says where it is: a quiet 440 Hz tone all the way through (amplitude
    0.2) and, from 3.0 s to 3.2 s of the clip, a loud 1 kHz beep on top of it (0.5)."""
    path = tmp_path_factory.mktemp("beepclip") / "beep.mp4"
    media.run([
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=4",
        "-f", "lavfi", "-i",
        "aevalsrc='0.2*sin(2*PI*440*t)+if(between(t,3,3.2),0.5*sin(2*PI*1000*t),0)':s=44100:d=4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
        "-frames:v", "120", str(path),
    ])
    return path


def wav_info(path):
    """A WAV's (rate, channels, samples per channel), read without going through mclib.media."""
    import wave

    with wave.open(str(path), "rb") as handle:
        return handle.getframerate(), handle.getnchannels(), handle.getnframes()


def wav_mono(path):
    """A WAV as mono floats at its own rate: the mean of its channels."""
    import wave

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2") / 32768.0
        return raw.reshape(-1, channels).mean(axis=1), handle.getframerate()


# The timeline of the tests below, at 30 fps: half a second of the held first frame, 2 s of the clip from
# its 1.0 s, a card of 1 s, and a tail that plays from the clip's 3.0 s (where the beep is) for 1 s.
LAID_OUT = (
    Segment("fill", 15), Segment("clip", 60, 1.0, 1.0), Segment("card", 30), Segment("clip", 30, 3.0),
)


def test_clip_sound_follows_the_timeline(beep_clip, tmp_path):
    out = tmp_path / "sound.wav"

    made = media.clip_sound(beep_clip, LAID_OUT, 30, out)

    assert made == out
    rate, channels, samples = wav_info(out)
    assert (rate, channels) == (48000, 2)
    assert samples == round(135 / 30 * rate)  # exactly the video's length: 4.5 s
    sound, rate = wav_mono(out)
    tone = 0.2 / 2 ** 0.5  # the rms of the clip's quiet tone
    assert rms(sound, 0.0, 0.45, rate) == pytest.approx(0.0, abs=0.001)  # the held first frame: silence
    assert rms(sound, 0.6, 2.4, rate) == pytest.approx(tone, rel=0.15)  # the head: the clip's own sound
    assert rms(sound, 2.6, 3.4, rate) < 0.001  # the card: silence
    # and when the video lets go, the clip's sound comes back from where the tail starts (the clip's
    # 3.0 s), so the beep that is there is heard 0.5 s into the tail, at once: 3.5 s of the video
    assert rms(sound, 3.55, 3.65, rate) > 0.3
    assert rms(sound, 3.8, 4.3, rate) == pytest.approx(tone, rel=0.15)
    loud = int(np.argmax(np.abs(sound[int(3.0 * rate):]) > 0.35)) / rate + 3.0
    assert loud == pytest.approx(3.5, abs=0.01)  # in sync with the picture, within a frame


def test_clip_sound_fades_at_the_cuts_and_at_the_end(beep_clip, tmp_path):
    out = tmp_path / "sound.wav"
    media.clip_sound(beep_clip, LAID_OUT, 30, out)
    sound, rate = wav_mono(out)

    # each piece of the clip comes in and goes out over about 10 ms, so a cut does not click
    first = int(0.5 * rate)  # the head starts here
    assert np.abs(sound[first:first + 40]).max() < 0.5 * np.abs(sound[first + 1000:first + 1500]).max()
    tail = int(3.5 * rate)  # and the card's silence is followed by the tail
    assert np.abs(sound[tail:tail + 40]).max() < 0.1
    # the last 0.4 s fades out, as the music does
    assert rms(sound, 4.45, 4.5, rate) < 0.2 * rms(sound, 3.8, 4.0, rate)


def test_clip_sound_at_a_speed_is_as_long_as_the_picture(beep_clip, tmp_path):
    out = tmp_path / "sound.wav"
    # the head at 1.43x: 43 frames of output take 43 * 1.43 / 30 s of the clip
    segments = (Segment("clip", 43, 0.5, 1.43), Segment("card", 30))

    media.clip_sound(beep_clip, segments, 30, out)

    rate, _, samples = wav_info(out)
    assert samples == round(73 / 30 * rate)
    sound, rate = wav_mono(out)
    assert rms(sound, 0.2, 1.2, rate) > 0.05  # the clip's sound, sped up
    assert rms(sound, 1.5, 2.4, rate) < 0.001


def test_clip_sound_is_exact_at_a_fractional_rate(beep_clip, tmp_path):
    out = tmp_path / "sound.wav"
    rate_ntsc = Fraction(24000, 1001)

    media.clip_sound(beep_clip, (Segment("clip", 47, 0.0), Segment("card", 49)), rate_ntsc, out)

    rate, _, samples = wav_info(out)
    assert samples == pytest.approx(96 / float(rate_ntsc) * rate, abs=1)


def test_clip_sound_pads_a_sound_that_is_shorter_than_the_picture(beep_clip, tmp_path):
    out = tmp_path / "sound.wav"
    # 40 frames from the clip's 3.5 s: only 0.5 s of sound is left there, and 1.33 s is asked for
    media.clip_sound(beep_clip, (Segment("clip", 40, 3.5),), 30, out)

    rate, _, samples = wav_info(out)
    assert samples == round(40 / 30 * rate)


def test_clip_sound_gives_nothing_for_a_clip_without_sound(synthetic_clip_ntsc, tmp_path):
    out = tmp_path / "sound.wav"

    assert media.clip_sound(synthetic_clip_ntsc, (Segment("clip", 24, 0.0),), 24, out) is None
    assert not out.exists()
    assert not media.has_audio(synthetic_clip_ntsc)


def test_has_audio(beep_clip):
    assert media.has_audio(beep_clip)


# --- the clip's own frame rate (ruling Q1) --------------------------------------------------

NTSC = Fraction(24000, 1001)  # 23.976 fps, a rate that only a fraction states exactly


def rate_of(path):
    """The frame rate a video's first stream states, as ffprobe prints it ("24/1", "24000/1001")."""
    shown = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return shown.strip()


def make_ntsc_code_clip(path):
    """A 640x360 code clip at 24000/1001 fps, 48 frames: make_code_clip at a fractional rate."""
    bit = "if(mod(floor(N/pow(2,floor(X/80))),2),235,16)"
    media.run(["-f", "lavfi", "-i",
               f"color=black:size=640x360:rate=24000/1001,format=yuv420p,"
               f"geq=lum='{bit}':cb=128:cr=128",
               "-frames:v", "48", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
               "-pix_fmt", "yuv420p", str(path)])
    return path


def test_output_fps_snaps_to_the_clips_own_rate():
    standard = {
        float(Fraction(24000, 1001)): Fraction(24000, 1001), 24.0: Fraction(24),
        25.0: Fraction(25), float(Fraction(30000, 1001)): Fraction(30000, 1001),
        30.0: Fraction(30), 48.0: Fraction(48), 50.0: Fraction(50),
        float(Fraction(60000, 1001)): Fraction(60000, 1001), 60.0: Fraction(60),
    }
    for fps, rate in standard.items():
        found = media.output_fps({"fps": fps})
        assert found == rate and isinstance(found, Fraction), fps

    # A probe a little off is still the standard rate, up to half a percent
    assert media.output_fps({"fps": 29.9}) == Fraction(30000, 1001)
    assert media.output_fps({"fps": 24.05}) == 24
    assert media.output_fps({"fps": 60.2}) == 60
    # Any other rate is rounded to a whole number, at least 1
    assert [media.output_fps({"fps": fps}) for fps in (15.0, 12.4, 26.0, 14.985, 0.2)] == [
        15, 12, 26, 15, 1]
    # Nothing goes above 60: a faster clip loses frames to 60, it never keeps a higher rate
    for fps in (61.0, 90.0, 119.88, 120.0, 240.0):
        assert media.output_fps({"fps": fps}) == 60, fps
    # It reads a project's clip as well as a probe
    assert media.output_fps({"path": "a.mp4", "width": 1920, "fps": 25.0}) == 25


def test_frames_are_cut_at_the_clips_own_rate(synthetic_clip_24, synthetic_clip_ntsc, tmp_path):
    assert media.frame_filter(1920, 1080, WORK, fps=24).startswith("fps=24,scale=")
    assert media.frame_filter(1920, 1080, WORK, fps=NTSC).startswith("fps=24000/1001,scale=")

    # 24 x duration frames from the 24 fps clip, and 48 from the 2 s clip at 24000/1001: every frame
    # of the clip, none made up and none dropped.
    for clip, frames, rate in (
        (synthetic_clip_24, 96, "24/1"), (synthetic_clip_ntsc, 48, "24000/1001")
    ):
        info = media.probe(clip)
        assert info["frames"] == frames
        assert rate_of(clip) == rate
        folder = tmp_path / clip.stem
        assert media.extract_frames(clip, folder, WORK, media.output_fps(info)) == frames
        assert len(list(folder.glob("*.jpg"))) == frames

    # Cut at 30 fps instead, the 24 fps clip would be converted: 120 frames, 24 of them repeats.
    assert media.extract_frames(synthetic_clip_24, tmp_path / "converted", WORK) == 120


def test_parts_and_the_join_keep_the_rate_they_are_given(tmp_path):
    code = make_code_clip(tmp_path / "code24.mp4", 24, 2)  # 48 frames, frame N shows N
    head, card, joined = (tmp_path / f"{name}.mp4" for name in ("head", "card", "joined"))
    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    Image.new("RGB", WORK, (255, 0, 0)).save(red)
    Image.new("RGB", WORK, (0, 0, 255)).save(blue)

    # The segment's start is in seconds: 0.5 s is frame 12 at 24 fps, so the part shows 12 to 31
    media.encode_clip(code, head, Segment("clip", 20, 0.5, 1.0), (640, 360), WORK, preview=True, fps=24)
    assert media.count_frames(head) == 20 and rate_of(head) == "24/1"
    assert frame_codes(head, FRAME) == list(range(12, 32))

    # and a change of picture lands on the frame it was asked for
    runs = [(red, 5), (blue, 7), (red, 3)]
    media.encode_stills(runs, card, preview=True, fps=24)
    assert media.count_frames(card) == 15 and rate_of(card) == "24/1"
    assert frame_colors(card) == ["red"] * 5 + ["blue"] * 7 + ["red"] * 3

    media.concat([head, card], joined, fps=24)
    assert media.count_frames(joined) == 35 and rate_of(joined) == "24/1"
    assert color_tags(joined) == BT709_TAGS


def test_a_fractional_rate_stays_exact_through_the_parts_and_the_join(tmp_path):
    code = make_ntsc_code_clip(tmp_path / "code2398.mp4")
    assert rate_of(code) == "24000/1001"
    head, card, joined = (tmp_path / f"{name}.mp4" for name in ("head", "card", "joined"))
    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    Image.new("RGB", WORK, (255, 0, 0)).save(red)
    Image.new("RGB", WORK, (0, 0, 255)).save(blue)

    # frame 12 of the clip is 12 x 1001 / 24000 s in
    media.encode_clip(code, head, Segment("clip", 12, 12 * 1001 / 24000, 1.0), (640, 360), WORK,
                      preview=True, fps=NTSC)
    assert media.count_frames(head) == 12 and rate_of(head) == "24000/1001"
    assert frame_codes(head, FRAME) == list(range(12, 24))

    # The concat demuxer counts the pictures' durations in frames of 1001/24000 s, so every change
    # lands on its frame and not one early or late.
    runs = [(red, 2), (blue, 3), (red, 7), (blue, 1), (red, 6)]
    media.encode_stills(runs, card, preview=True, fps=NTSC)
    assert rate_of(card) == "24000/1001"
    assert frame_colors(card) == (
        ["red"] * 2 + ["blue"] * 3 + ["red"] * 7 + ["blue"] + ["red"] * 6)

    media.concat([head, card], joined, fps=NTSC)
    assert media.count_frames(joined) == 31 and rate_of(joined) == "24000/1001"
    assert frame_codes(joined, FRAME)[:12] == list(range(12, 24))


# --- one lossy encode, at the clip's bitrate (ruling Q1) ------------------------------------


def profile_of(path):
    """The H.264 profile of a video's first stream, as ffprobe names it."""
    shown = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=profile",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return shown.strip()


def test_probe_reports_the_clips_bitrate(synthetic_clip, tmp_path):
    info = media.probe(synthetic_clip)
    assert isinstance(info["bit_rate"], int) and info["bit_rate"] > 0
    # the video's own rate, within the little the sound adds to the file
    assert info["bit_rate"] == pytest.approx(
        synthetic_clip.stat().st_size * 8 / info["duration"], rel=0.2)

    # Matroska states no rate for a stream, so the file's is the best there is
    mkv = tmp_path / "clip.mkv"
    media.run(["-i", str(synthetic_clip), "-c", "copy", str(mkv)])
    assert media.probe(mkv)["bit_rate"] == pytest.approx(
        mkv.stat().st_size * 8 / media.probe(mkv)["duration"], rel=0.2)


def test_target_bitrate_is_the_clips_own_and_never_under_the_floor():
    assert media.target_bitrate(None) == 10_000_000
    assert media.target_bitrate(3_000_000) == 10_000_000
    assert media.target_bitrate(10_000_000) == 10_000_000
    assert media.target_bitrate(20_000_000) == 20_000_000
    assert media.target_bitrate(19_876_543) == 19_876_543


def test_lossless_parts_are_joined_by_one_lossy_encode(code_clip, tmp_path):
    part_a, part_b, joined = (tmp_path / f"{name}.mp4" for name in ("a", "b", "joined"))
    media.encode_clip(code_clip, part_a, Segment("clip", 30, 0.0, 1.0), (640, 360), WORK, lossless=True)
    media.encode_stills([(code_png(tmp_path / "card.png", 200), 30)], part_b, lossless=True)

    # The parts lose nothing: constant quantizer 0, which is the lossless profile of x264, with
    # the same exact frames and BT.709 tags as before
    for part in (part_a, part_b):
        options = x264_options(part)
        assert (options["rc"], options["qp"]) == ("cqp", "0")
        assert profile_of(part) == "High 4:4:4 Predictive"
        assert media.count_frames(part) == 30
        assert color_tags(part) == BT709_TAGS
        assert [s["pix_fmt"] for s in streams(part)] == ["yuv420p"]

    # and the join is the one lossy encode, in two passes: a target bitrate with a ceiling of 1.5x
    # and a buffer of 2x, the profile players expect, still tagged BT.709. The card (the still part)
    # is a zone weighted at STILL_WEIGHT, so the footage gets the bits.
    media.concat([part_a, part_b], joined, bitrate=20_000_000, still=(30, 59))
    options = x264_options(joined)
    assert options["rc"] == "2pass" and "crf" not in options
    assert options["zones"] == f"30,59,b={media.STILL_WEIGHT}"
    assert (options["bitrate"], options["vbv_maxrate"], options["vbv_bufsize"]) == (
        "20000", "30000", "40000")  # kbit/s
    assert profile_of(joined) == "High"
    assert color_tags(joined) == BT709_TAGS
    assert frame_codes(joined, FRAME) == list(range(30)) + [200] * 30

    # Without a bitrate the join is what it always was, and so is a preview
    plain = tmp_path / "plain.mp4"
    media.concat([part_a, part_b], plain)
    assert_x264(plain, 18, "medium")


def test_probe_without_counting_frames_is_the_same_but_for_the_count(synthetic_clip):
    quick, full = media.probe(synthetic_clip, count=False), media.probe(synthetic_clip)

    assert "frames" not in quick
    assert quick == {key: value for key, value in full.items() if key != "frames"}
