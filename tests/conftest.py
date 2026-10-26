"""Shared fixtures. The clips are generated with ffmpeg, so no test depends on a file in the repo.

The fixtures call ffmpeg and ffprobe directly instead of going through mclib.media, so a bug in
that module cannot break the clips the tests use to check it. The one exception is frames_dir_20,
which exists to hand the matte its frames exactly as extract_frames writes them.

The helpers that more than one test file needs (color_tags, x264_options, rms) are here too, as
plain functions: a test file takes them with "from conftest import ...".
"""

import http.client
import io
import json
import re
import shutil
import subprocess
import threading
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from PIL import Image, ImageDraw

# The sample copy, in natural case (the notes styles apply their own case when they draw). There is
# no title in it: the person adds that in the app (Amendment 1).
SAMPLE_COPY = {
    "notes": {
        "tl": "Creator & AI educator",
        "tr": "@your_handle",
        "bl": "But overthinks every 6-second reel",
        "br": "Best friend: Clover (sorry, mom)",
    },
    "name": "Robin",
    "extras": {
        "burst": "Robin!",
        "emphasis": "overthinks",
        "stats": [["AI skills", 10], ["Overthinking", 10], ["Chill", 1]],
    },
}


# The color tags every video the skill writes must carry: BT.709, limited range.
BT709 = {"color_range": "tv", "color_space": "bt709", "color_transfer": "bt709",
         "color_primaries": "bt709"}


def color_tags(path):
    """The color tags of a video: as its first stream states them, and as the decoder reads them
    from the first picture ("unknown" for a tag that is not there). The two can disagree, and
    players follow one or the other, so a test compares both with {"stream": BT709, "frame": BT709}."""
    shown = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
         "-show_entries", "stream=color_range,color_space,color_transfer,color_primaries:"
         "frame=color_range,color_space,color_transfer,color_primaries", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    found = json.loads(shown)
    return {layer: {key: found[layer + "s"][0].get(key, "unknown") for key in BT709}
            for layer in ("stream", "frame")}


def x264_options(path):
    """The settings libx264 left as text in a video, as a dict: rc, crf, qp, bitrate and so on."""
    found = re.search(rb"options: ([ -~]+)", path.read_bytes())
    return dict(pair.split("=", 1) for pair in found.group(1).decode().split() if "=" in pair)


def rms(samples, start, end, sr=44100):
    """The root mean square of the samples between two times, in seconds, at sr samples a second."""
    window = samples[int(start * sr):int(end * sr)]
    return float(np.sqrt(np.mean(window ** 2)))


def _ffmpeg(*args):
    """Run ffmpeg quietly. A failure raises RuntimeError with ffmpeg's own message."""
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *args], stdin=subprocess.DEVNULL, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip())


@pytest.fixture(scope="session")
def synthetic_clip(tmp_path_factory):
    """A 4 s clip, 1920x1080 at 30 fps (120 frames): the testsrc2 pattern and a 440 Hz sine, AAC.

    The picture is BT.709 limited range and tagged as such, like a phone clip.
    """
    path = tmp_path_factory.mktemp("clips") / "synthetic.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=4",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-colorspace", "bt709", "-color_range", "tv",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-frames:v", "120", str(path),
    )
    return path


@pytest.fixture(scope="session")
def synthetic_clip_24(tmp_path_factory):
    """A 4 s clip, 1920x1080 at 24 fps (96 frames), made like synthetic_clip: the testsrc2 pattern
    and a 440 Hz sine, AAC, BT.709 limited range."""
    path = tmp_path_factory.mktemp("clips24") / "synthetic24.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=24:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=4",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-colorspace", "bt709", "-color_range", "tv",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-frames:v", "96", str(path),
    )
    return path


@pytest.fixture(scope="session")
def synthetic_clip_ntsc(tmp_path_factory):
    """A 2 s clip, 1280x720 at 24000/1001 fps (48 frames), the rate of a film shot for NTSC: the
    one that a fraction of a frame rate has to be exact for."""
    path = tmp_path_factory.mktemp("clips2398") / "synthetic2398.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24000/1001",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-colorspace", "bt709", "-color_range", "tv",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-frames:v", "48", str(path),
    )
    return path


@pytest.fixture(scope="session")
def portrait_rotated_clip(synthetic_clip, tmp_path_factory):
    """The first second of the synthetic clip stored sideways (1920x1080) with a display rotation
    of 90 degrees, which is how a phone stores a portrait video. Players show it as 1080x1920."""
    path = tmp_path_factory.mktemp("rotated") / "rotated.mp4"
    _ffmpeg(
        "-display_rotation", "90", "-i", str(synthetic_clip), "-t", "1", "-c", "copy", str(path)
    )

    # Confirm the rotation is really in the file, so a test of it cannot pass by accident
    shown = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json",
         str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    side_data = json.loads(shown)["streams"][0].get("side_data_list", [])
    assert any(abs(entry.get("rotation", 0)) == 90 for entry in side_data), (
        "ffprobe reports no rotation for the fixture"
    )
    return path


@pytest.fixture(scope="session")
def rotated_phone_clip(synthetic_clip, tmp_path_factory):
    """A 3 s clip stored sideways, 2560x1440 at 30 fps (90 frames), with a display rotation of 90 degrees,
    as a phone stores a portrait video: players show it as 1440x2560, which is over 1080 wide, so its
    plan (1080x1920) and its video (1440x2560) are not the same size. Long enough for init (2 s)."""
    folder = tmp_path_factory.mktemp("rotated3")
    stored = folder / "stored.mp4"
    _ffmpeg(
        "-i", str(synthetic_clip), "-t", "3", "-vf", "scale=2560:1440,format=yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", str(stored),
    )
    path = folder / "phone.mp4"
    _ffmpeg("-display_rotation", "90", "-i", str(stored), "-c", "copy", str(path))

    # Confirm the rotation is really in the file, so a test of it cannot pass by accident
    shown = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(shown)["streams"][0]
    assert any(abs(entry.get("rotation", 0)) == 90 for entry in stream.get("side_data_list", [])), (
        "ffprobe reports no rotation for the fixture"
    )
    assert (stream["width"], stream["height"]) == (2560, 1440)
    return path


@pytest.fixture(scope="session")
def hlg_clip(tmp_path_factory):
    """A 1 s clip, 640x360 at 30 fps (30 frames), as an iPhone shoots by default: 10-bit BT.2020
    HLG. The picture is four flat patches, saturated red and yellow on top, cyan and gray below,
    that zscale converts from SDR to HLG, and the HLG tags are written into the stream. Read as
    plain BT.709 video the colors wash out, so it tells a real conversion from a relabel. The
    test is skipped where ffmpeg cannot make the clip."""
    folder = tmp_path_factory.mktemp("hlg")
    patches = Image.new("RGB", (640, 360))
    draw = ImageDraw.Draw(patches)
    for color, box in (("#ff0000", (0, 0, 319, 179)), ("#ffd21f", (320, 0, 639, 179)),
                       ("#3ef0ff", (0, 180, 319, 359)), ("#808080", (320, 180, 639, 359))):
        draw.rectangle(box, fill=color)
    patches.save(folder / "patches.png")

    path = folder / "hlg.mp4"
    try:
        _ffmpeg(
            "-loop", "1", "-framerate", "30", "-i", str(folder / "patches.png"),
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                   "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv,"
                   "zscale=t=arib-std-b67:m=2020_ncl:p=2020:r=tv:npl=100,format=yuv420p10le",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "8", "-pix_fmt", "yuv420p10le",
            "-bsf:v", "h264_metadata=colour_primaries=9:transfer_characteristics=18:"
                      "matrix_coefficients=9:video_full_range_flag=0",
            "-frames:v", "30", str(path),
        )
    except RuntimeError as error:
        pytest.skip(f"this ffmpeg cannot make an HLG test clip: {error}")
    return path


@pytest.fixture(scope="session")
def frames_dir_20(synthetic_clip, tmp_path_factory):
    """The first 20 frames of the synthetic clip as the matte reads them: 1080x608 JPEGs named
    000000.jpg to 000019.jpg, written by mclib.media.extract_frames. Tests only read the folder."""
    from mclib import media

    folder = tmp_path_factory.mktemp("frames20")
    media.extract_frames(synthetic_clip, folder, (1080, 608))
    for extra in sorted(folder.glob("*.jpg"))[20:]:
        extra.unlink()
    return folder


@pytest.fixture(scope="session")
def person_alpha():
    """A factory of test mattes: person_alpha(w, h) is a new float32 array (h x w, 0-1) of a person
    in a close shot, as big as one in a 1080x608 frame, scaled to any w and h.

    An elliptical head sits over rounded shoulders that run off the bottom of the frame, joined by
    a neck. The edge falls from 1 to 0 over 3 px like a real matte's, so it has a 0.5 contour for a
    cleanup to find and a ramp for a blur to soften. Each call makes a fresh array, because the
    tests draw specks and holes into it.
    """

    def coverage(distance):
        """Alpha at a signed distance in px from the edge (negative inside)."""
        return np.clip(0.5 - distance / 3.0, 0.0, 1.0)

    def ellipse(xx, yy, cx, cy, rx, ry):
        # first-order distance to the edge: how far past 1 the radius reaches, over its slope
        u, v = (xx - cx) / rx, (yy - cy) / ry
        radius = np.hypot(u, v)
        slope = np.hypot(u / rx, v / ry) / np.maximum(radius, 1e-6)
        return (radius - 1) / np.maximum(slope, 1e-6)

    def rounded_box(xx, yy, cx, cy, half_w, half_h, r):
        qx = np.abs(xx - cx) - (half_w - r)
        qy = np.abs(yy - cy) - (half_h - r)
        outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        return outside + np.minimum(np.maximum(qx, qy), 0) - r

    def make(w, h):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5  # pixel centers
        head = ellipse(xx, yy, 0.5 * w, 0.34 * h, 0.10 * w, 0.26 * h)
        neck = rounded_box(xx, yy, 0.5 * w, 0.58 * h, 0.04 * w, 0.12 * h, 0.02 * w)
        top, radius = 0.62 * h, 0.12 * w
        bottom = h + 2 * radius  # well below the frame
        middle, half_h = (top + bottom) / 2, (bottom - top) / 2
        body = rounded_box(xx, yy, 0.5 * w, middle, 0.31 * w, half_h, radius)
        return coverage(np.minimum(np.minimum(head, neck), body)).astype(np.float32)

    return make


class FakeProject(NamedTuple):
    """What fake_project gives a test: the project folder and the project dict, in that order
    (so `folder, project = fake_project` works, and so does `fake_project.project`)."""

    dir: Path
    project: dict


@pytest.fixture(scope="session")
def fake_frames(synthetic_clip, tmp_path_factory):
    """Every frame of the synthetic clip as the matte reads them: 1080x608 JPEGs written by
    mclib.media.extract_frames, the one other place (with frames_dir_20) where a fixture leans on
    media. Tests never touch this folder: fake_project copies the frames it needs."""
    from mclib import media

    folder = tmp_path_factory.mktemp("fake_frames")
    media.extract_frames(synthetic_clip, folder, (1080, 608))
    return folder


@pytest.fixture(scope="session")
def fake_frames_24(synthetic_clip_24, tmp_path_factory):
    """Every frame of the 24 fps synthetic clip (96 of them) as the matte reads them: 1080x608
    JPEGs written by mclib.media.extract_frames at 24 fps, like fake_frames."""
    from mclib import media

    folder = tmp_path_factory.mktemp("fake_frames_24")
    media.extract_frames(synthetic_clip_24, folder, (1080, 608), 24)
    return folder


class FakeMatte:
    """A matte runner with the signature of matte.run_rvm. For each frame in range it writes a PNG
    of a person: conftest's synthetic one, or a band of the given share of the picture, or nobody.
    calls is a list of (ratio rounded to 3 places, start, end) with end counted, never None."""

    def __init__(self, person_alpha, area=None, nobody=False):
        self._person_alpha, self._area, self._nobody = person_alpha, area, nobody
        self.calls = []

    def _png(self, width, height):
        if self._nobody:
            alpha = np.zeros((height, width), np.float32)
        elif self._area is not None:
            alpha = np.zeros((height, width), np.float32)
            alpha[height - max(1, round(self._area * height)):] = 1.0
        else:
            alpha = self._person_alpha(width, height)
        buffer = io.BytesIO()
        Image.fromarray(np.rint(alpha * 255).astype(np.uint8)).save(buffer, "PNG")
        return buffer.getvalue()

    def __call__(self, frames_dir, out_dir, ratio, start=0, end=None, warmup=12, progress=None):
        frames = sorted(Path(frames_dir).glob("*.jpg"))
        end = len(frames) if end is None else min(end, len(frames))
        with Image.open(frames[0]) as first:
            png = self._png(*first.size)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for number in range(start, end):
            (Path(out_dir) / f"{number:06d}.png").write_bytes(png)
            if progress:
                progress(number - start + 1, end - start)
        self.calls.append((round(ratio, 3), start, end))
        return end - start


@pytest.fixture(scope="session")
def fake_matte(person_alpha):
    """fake_matte() is a FakeMatte, a matte runner that stands in for RVM. fake_matte(area=0.1) cuts
    out a band that fills that share of the picture, and fake_matte(nobody=True) cuts out nobody."""

    def make(area=None, nobody=False):
        return FakeMatte(person_alpha, area, nobody)

    return make


@pytest.fixture(scope="session")
def project_maker(synthetic_clip, fake_frames, person_alpha):
    """project_maker(folder) builds the project fake_project describes in folder, and returns it as
    a FakeProject. A test that needs the project for longer than one test (a render that several
    tests look at) builds it in a folder of its own with this.

    To build one for another clip: project_maker(folder, clip=path, frames=folder_of_jpegs,
    fps=24.0, count=96, freeze=48, stem="synthetic24"). The folder of frames is only read.
    """
    from mclib import project as project_file
    from mclib.styles.recipes import BUILTINS

    def make(folder, clip=synthetic_clip, frames=fake_frames, fps=30.0, count=120, freeze=60,
             stem="synthetic"):
        folder = Path(folder)
        (folder / "frames").mkdir()
        for frame in (0, freeze):
            shutil.copy(frames / f"{frame:06d}.jpg", folder / "frames")
        (folder / "matte").mkdir()
        matte = np.rint(person_alpha(1080, 608) * 255).astype(np.uint8)
        Image.fromarray(matte).save(folder / "matte" / f"{freeze:06d}.png")

        info = {
            "path": str(clip),
            "stem": stem,
            "width": 1920,
            "height": 1080,
            "fps": fps,
            "duration": 4.0,
            "frames": count,
            "orientation": "landscape",
        }
        project = project_file.default_project(info, "landscape", (1080, 608), BUILTINS["bedroom-pop"])
        project["freeze"]["frame"] = freeze
        project["copy"] = deepcopy(SAMPLE_COPY)
        project["versions"]["C"] = deepcopy(SAMPLE_COPY)
        project["picked"] = "C"
        project_file.validate(project)  # raises if this is not a real project (its warnings are soft limits)
        project_file.save(folder, project)
        return FakeProject(folder, project)

    return make


PLATE_FRAME = Path(__file__).parent / "fixtures" / "plate_frame.jpg"  # a drawn room, nobody's photo
PLATE_MATTE = Path(__file__).parent / "fixtures" / "close_alpha.png"  # a real close-up, as a silhouette
PLATE_TALL_MATTE = Path(__file__).parent / "fixtures" / "vertical_alpha.png"  # a vertical photo, 1080x1920


@pytest.fixture(scope="session")
def plate_maker():
    """plate_maker(folder, recipe) builds the "plate" project in folder and returns it as a
    FakeProject: a real close-up matte (close_alpha.png, which fills the frame and has a hood)
    over a drawn room (plate_frame.jpg, a 1080x608 JPEG), the sample copy, and the recipe asked for (a
    BUILTINS entry) frozen on frame 60 of a 1920x1080 clip. Nothing is cut from a clip, so it needs
    neither ffmpeg nor RVM. The approved_landscape*.png pictures in tests/fixtures are the card of
    this project in each of the four built-in styles, drawn with the sample copy by code that matched
    the approved look byte for byte: what the approved look is (test_scale).

    plate_maker(folder, recipe, portrait=True) is the same on a 9:16 plan (1080x1920, from a clip of
    1080x1920): a vertical photo's matte (vertical_alpha.png) over the same room stretched to it."""
    from mclib import project as project_file

    def make(folder, recipe, portrait=False):
        folder = Path(folder)
        (folder / "frames").mkdir(parents=True)
        (folder / "matte").mkdir()
        if portrait:
            with Image.open(PLATE_FRAME) as room:
                room.convert("RGB").resize((1080, 1920)).save(folder / "frames" / "000060.jpg", quality=95)
            shutil.copy(PLATE_TALL_MATTE, folder / "matte" / "000060.png")
            info = {
                "path": "plate.mp4", "stem": "plate", "width": 1080, "height": 1920, "fps": 30.0,
                "duration": 4.0, "frames": 120, "orientation": "portrait",
            }
            project = project_file.default_project(info, "portrait", (1080, 1920), recipe)
        else:
            shutil.copy(PLATE_FRAME, folder / "frames" / "000060.jpg")
            shutil.copy(PLATE_MATTE, folder / "matte" / "000060.png")
            info = {
                "path": "plate.mp4", "stem": "plate", "width": 1920, "height": 1080, "fps": 30.0,
                "duration": 4.0, "frames": 120, "orientation": "landscape",
            }
            project = project_file.default_project(info, "landscape", (1080, 608), recipe)
        project["freeze"]["frame"] = 60
        project["copy"] = deepcopy(SAMPLE_COPY)
        project["versions"]["C"] = deepcopy(SAMPLE_COPY)
        project["picked"] = "C"
        project_file.validate(project)
        project_file.save(folder, project)
        return FakeProject(folder, project)

    return make


@pytest.fixture
def fake_project(tmp_path, project_maker):
    """A landscape project in tmp_path that renders without RVM or a render: the sample copy, the
    Bedroom Pop recipe and the freeze on frame 60 of the synthetic clip.

    The folder holds frames/000000.jpg (the fill frame) and frames/000060.jpg (the freeze), the
    matte of the freeze as matte/000060.png (conftest's synthetic person, not RVM) and
    project.json. Frame 60 at 30 fps is 2.0 s into the 4 s clip.
    """
    return project_maker(tmp_path)


# --- the studio server ------------------------------------------------------------------------


class Reply(NamedTuple):
    """An answer from the studio server: its status, its headers (the names in lower case) and its
    body. json() reads the body as JSON."""

    status: int
    headers: dict
    body: bytes

    def json(self):
        return json.loads(self.body.decode("utf-8"))


class StudioClient:
    """A running studio server, for a test to talk to the way the page does. get and post return a
    Reply and never raise for a 4xx, so a test can look at the error. project() reads project.json
    from the folder, which is the file the server keeps and nothing else."""

    def __init__(self, httpd, folder: Path, runner):
        self.httpd, self.dir, self.runner = httpd, Path(folder), runner
        self.port = httpd.server_address[1]

    def request(self, method, path, body=None, headers=None) -> Reply:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=180)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            answer = connection.getresponse()
            shown = {name.lower(): value for name, value in answer.getheaders()}
            return Reply(answer.status, shown, answer.read())
        finally:
            connection.close()

    def get(self, path, headers=None) -> Reply:
        return self.request("GET", path, headers=headers)

    def post(self, path, data=None, raw=None, headers=None) -> Reply:
        """POST data as JSON, or raw bytes as they are. With neither, the body is empty."""
        sent = dict(headers or {})
        if raw is None:
            raw = b"" if data is None else json.dumps(data).encode("utf-8")
            sent.setdefault("Content-Type", "application/json")
        return self.request("POST", path, body=raw, headers=sent)

    def project(self) -> dict:
        from mclib import project as project_file

        return project_file.load(self.dir)


@pytest.fixture
def server(fake_project, fake_matte, tmp_path_factory, monkeypatch):
    """A studio server on the fake project, on a free port of 127.0.0.1, in a thread, as a
    StudioClient. The matte runner is a FakeMatte (server.runner), the saved styles are in a
    folder of their own (MC_HOME), and the server is stopped when the test is over."""
    from mclib import server as studio_server

    monkeypatch.setenv("MC_HOME", str(tmp_path_factory.mktemp("mc-home")))
    runner = fake_matte()
    httpd = studio_server.make_server(fake_project.dir, 0, matte_runner=runner)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield StudioClient(httpd, fake_project.dir, runner)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=10)
