"""The cutout: Robust Video Matting (RVM) on onnxruntime, and the cleanup and edge controls that
every matte goes through.

run_rvm turns a folder of frames into a folder of alpha PNGs, one for each frame, named by its
number: matte/000123.png is the alpha of frame 123. load_alpha reads one back for the renderers.
An alpha is a float32 array, height x width, 0-1.

The model is not bundled (RVM is GPL-3.0, and the file is about 15 MB). ensure_model says whether
it is on this machine, and download_model fetches it, which the caller does only after the person
agrees. onnxruntime is imported when a run starts, so the code that only loads mattes never pays
for it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from .config import MODEL_PATH, MODEL_SHA256, MODEL_URL
from .fileio import retry_locked
from .raster import _gaussian, blur, dilate, from_image

_FRAME_FILES = "[0-9]" * 6 + ".jpg"  # the names extract_frames gives the frames
_MIN_MODEL_BYTES = 10_000_000  # anything smaller than this is not the model
_CURL_SECONDS = 900  # how long curl may take to bring the model
_CERTIFICATE = (
    "Python couldn't check the download's certificate. On a Mac with Python from python.org, run "
    '"Install Certificates.command" in the Python folder, then try again.'
)

# downsample_for. The model is shown a picture 576 px on its long side; the detail settings
# multiply that. A person who fills the frame gets at least half size.
_SEEN_LONG_SIDE = 576
_DETAIL = {"auto": 1.0, "high": 1.6, "max": 2.4}
_FILLS_FRAME = 0.45  # a person covering more than this share of the frame
_FILLS_FRAME_RATIO = 0.5  # gets at least this ratio

# The alpha floor, as in the reference rvm_matte.py: alpha under LO is cut and between LO and HI it
# ramps back up to the raw alpha, so no faint veil of the background lands on what is under the
# cutout.
_FLOOR_LO, _FLOOR_HI = 0.03, 0.10

# clean_alpha
_CELL = 4  # it works on cells of 4 x 4 pixels, a quarter of the picture on each side
_PERSON = 0.5  # alpha at or above this counts as person
_MIN_REGION = 0.01  # a region smaller than this share of the largest one is dropped
_FADE_SIGMA = 2.0  # px; the Gaussian that softens where the cleanup cuts
_MAX_HOLE = 0.01  # an enclosed gap over this share of the kept area is left open


# --- the model --------------------------------------------------------------------------------


class ModelMissing(RuntimeError):
    """The RVM model is not on this machine, or it could not be downloaded, or the download came back wrong."""


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """The path of the model, or ModelMissing saying where it should be, how big it is and where
    it comes from. It never downloads anything."""
    path = Path(path)
    if not path.is_file():
        raise ModelMissing(
            f"The matte model is missing: it should be at {path}. "
            f"It is a download of about 15 MB from {MODEL_URL}."
        )
    return path


def download_model(dest: Path = MODEL_PATH, url: str = MODEL_URL, sha256: str = MODEL_SHA256) -> Path:
    """Download the model to dest and return dest. Call it only after the person has agreed.

    The bytes go to a .part file beside dest and are renamed when they are all there, so an
    interrupted download never leaves a broken model where ensure_model would find it. Anything
    under 10 MB is refused with ModelMissing and nothing is kept: an error page is a few KB. What
    is left is checked against sha256 (MODEL_SHA256: the official release's), and a file that is
    not that one is refused the same way. Every failure to download is a ModelMissing with a
    sentence for the person (see _fetch). On Windows a virus scan can hold the new file for a moment
    and refuse the rename, which is waited out (fileio.retry_locked).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        _fetch(url, part)
        size = part.stat().st_size
        if size < _MIN_MODEL_BYTES:
            raise ModelMissing(
                f"The download from {url} was only {size:,} bytes, too small for the matte "
                "model (about 15 MB), so it was not kept."
            )
        found = _sha256(part)
        if found != sha256:
            raise ModelMissing(
                f"The file from {url} is not the matte model I expected (its SHA-256 starts "
                f"{found[:12]}, not {sha256[:12]}), so it was not kept. Try the download again."
            )
        retry_locked(lambda: os.replace(part, dest), dest)
    finally:
        _discard(part)  # gone already after a good download
    return dest


def _fetch(url: str, part: Path) -> None:
    """Write what url holds to part. Python's own download comes first. python.org's Python on a Mac
    has no certificates until its "Install Certificates.command" is run, and until then it fails with
    CERTIFICATE_VERIFY_FAILED: that goes on to curl, which uses the system's certificates. Any other
    failure is a ModelMissing that says what happened, in a sentence."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except ssl.SSLCertVerificationError:
        _fetch_with_curl(url, part)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, ssl.SSLCertVerificationError):
            _fetch_with_curl(url, part)
        else:
            raise ModelMissing(
                f"Couldn't download the matte model from {url}: {reason}. "
                "Check the internet connection and try again."
            ) from error


def _fetch_with_curl(url: str, part: Path) -> None:
    """Write what url holds to part with curl (-L follows the redirect GitHub gives, --fail makes an
    error page a failure). Without curl the person is told how to give Python its certificates."""
    curl = shutil.which("curl")
    if curl is None:
        raise ModelMissing(_CERTIFICATE)
    try:
        done = subprocess.run(
            [curl, "-L", "--fail", "-o", str(part), url], stdin=subprocess.DEVNULL, capture_output=True,
            encoding="utf-8", errors="replace", timeout=_CURL_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ModelMissing(f"curl couldn't download the matte model from {url}: {error}.") from error
    if done.returncode != 0:
        # curl's progress meter fills its stderr; its error is the last line
        lines = [line.strip() for line in done.stderr.splitlines() if line.strip()]
        raise ModelMissing(
            f"curl couldn't download the matte model from {url} ({lines[-1] if lines else 'it failed'}). "
            "Check the internet connection and try again."
        )


def _sha256(path: Path) -> str:
    """The SHA-256 of a file, as lowercase hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _discard(part: Path) -> None:
    """Delete the .part file if it is there. A refusal that does not clear is left: the next download
    writes over it, and an error here must not hide the one that is being raised."""
    try:
        retry_locked(lambda: part.unlink(missing_ok=True), part)
    except PermissionError:
        pass


def downsample_for(
    work_w: int, work_h: int, detail: str = "auto", person_area: float | None = None
) -> float:
    """The downsample ratio RVM should run at for frames of work_w x work_h.

    auto is 576 divided by the long side, at most 1, so the model always sees about the same
    number of pixels. When person_area (the share of the frame the person covers, 0-1) is over
    0.45 it is raised to at least 0.5: a person who fills the frame needs a closer look, or the
    model loses arms and hands. high and max ask for more detail, 1.6 and 2.4 times that, again
    at most 1.
    """
    if detail not in _DETAIL:
        raise ValueError(f'detail must be "auto", "high" or "max", got {detail!r}')
    auto = min(1.0, _SEEN_LONG_SIDE / max(work_w, work_h))
    if person_area is not None and person_area > _FILLS_FRAME:
        auto = max(auto, _FILLS_FRAME_RATIO)
    return min(1.0, auto * _DETAIL[detail])


# --- running RVM ------------------------------------------------------------------------------


def _open_session():
    """An onnxruntime session on the model: CPU only, with a thread for every core."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = os.cpu_count() or 4
    return onnxruntime.InferenceSession(
        str(ensure_model()), options, providers=["CPUExecutionProvider"]
    )


def _read_frame(frames_dir: Path, number: int) -> np.ndarray:
    """Frame number as RGB, height x width x 3, uint8."""
    with Image.open(frames_dir / f"{number:06d}.jpg") as image:
        return np.asarray(image.convert("RGB"))


def _step(session, frame: np.ndarray, state: list[np.ndarray], ratio: np.ndarray):
    """Run the model on one frame, given the state the previous frame left. Returns the raw alpha
    (height x width, float32) and the new state. The foreground color the model also makes is not
    asked for."""
    src = np.ascontiguousarray((frame.astype(np.float32) / 255).transpose(2, 0, 1)[None])
    feeds = {
        "src": src, "r1i": state[0], "r2i": state[1], "r3i": state[2], "r4i": state[3],
        "downsample_ratio": ratio,
    }
    alpha, *state = session.run(["pha", "r1o", "r2o", "r3o", "r4o"], feeds)
    return alpha[0, 0], state


def _save_alpha(raw: np.ndarray, path: Path) -> None:
    """Apply the alpha floor to a raw alpha and write it as an 8-bit gray PNG."""
    alpha = np.clip(raw, 0.0, 1.0)
    ramp = (alpha - _FLOOR_LO) / (_FLOOR_HI - _FLOOR_LO) * alpha
    alpha = np.where(alpha < _FLOOR_LO, 0.0, np.where(alpha < _FLOOR_HI, ramp, alpha))
    Image.fromarray((alpha * 255 + 0.5).astype(np.uint8)).save(path)


def run_rvm(
    frames_dir: Path,
    out_dir: Path,
    ratio: float,
    start: int = 0,
    end: int | None = None,
    warmup: int = 12,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Cut the person out of frames start to end - 1 (all the way to the last frame when end is
    None) of the 000000.jpg, 000001.jpg... frames in frames_dir. Each alpha is written as an
    8-bit gray PNG named for its frame, such as out_dir/000123.png, and the number of PNGs written
    is returned.

    RVM remembers the frames before the one it sees, and a cold start is poor. So it first runs
    over warmup frames without writing anything: the frames just before start, or, when start is
    0, the first frames of the clip, after which the run starts over at frame 0 with the warmed-up
    state. ratio is the downsample ratio (see downsample_for). progress, if given, is called as
    progress(done, total) after each frame is written. Raises ModelMissing when the model is not
    on this machine.
    """
    frames_dir, out_dir = Path(frames_dir), Path(out_dir)
    available = sum(1 for _ in frames_dir.glob(_FRAME_FILES))
    if available == 0:
        raise FileNotFoundError(f"There are no frames (000000.jpg, ...) in {frames_dir}.")
    if start < 0:
        raise ValueError(f"start must be a frame number, 0 or more, got {start}")
    end = available if end is None else min(end, available)
    if start >= end:
        return 0

    session = _open_session()
    ratio_input = np.array([ratio], np.float32)
    state = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]

    warming = range(max(0, start - warmup), start) if start > 0 else range(min(warmup, available))
    for number in warming:
        _, state = _step(session, _read_frame(frames_dir, number), state, ratio_input)

    out_dir.mkdir(parents=True, exist_ok=True)
    total = end - start
    for done, number in enumerate(range(start, end), 1):
        raw, state = _step(session, _read_frame(frames_dir, number), state, ratio_input)
        _save_alpha(raw, out_dir / f"{number:06d}.png")
        if progress:
            progress(done, total)
    return total


# --- cleaning a matte -------------------------------------------------------------------------


def _pool(mask: np.ndarray, combine: np.ufunc, cell: int = _CELL) -> np.ndarray:
    """One value for each 4 x 4 cell of a boolean picture (cell x cell, for a picture bigger than
    the plan): combine (np.logical_or or np.logical_and) of the cell's pixels. A cell cut short by
    the edge combines what it has."""
    rows = combine.reduceat(mask, np.arange(0, mask.shape[0], cell), axis=0)
    return combine.reduceat(rows, np.arange(0, mask.shape[1], cell), axis=1)


def _enlarge(cells: np.ndarray, shape: tuple[int, int], cell: int = _CELL) -> np.ndarray:
    """Cells back to pixels, nearest neighbor: each cell becomes a 4 x 4 block, cut to shape."""
    return np.repeat(np.repeat(cells, cell, axis=0), cell, axis=1)[: shape[0], : shape[1]]


def _grow(cells: np.ndarray, square: bool = False) -> np.ndarray:
    """The cells and their neighbors: the four that share a side, or with square=True all eight
    around each cell. A region one cell wider on every side."""

    def wider(mask):  # each cell and its left and right neighbors
        out = mask.copy()
        out[:, 1:] |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        return out

    def taller(mask):  # each cell and its upper and lower neighbors
        out = mask.copy()
        out[1:] |= mask[:-1]
        out[:-1] |= mask[1:]
        return out

    return taller(wider(cells)) if square else wider(cells) | taller(cells)


def _fade(
    cells: np.ndarray, shape: tuple[int, int], cell: int = _CELL, sigma: float = _FADE_SIGMA
) -> np.ndarray:
    """Cells as a smooth float picture of the given shape: 1 inside them and 0 outside.

    Each cell is enlarged to 4 x 4 pixels bilinearly, so its value sits at its center and a step
    from one cell to the next becomes a ramp 4 px long. A Gaussian of 2 px then rounds that off.
    (For a picture bigger than the plan, the cell and the Gaussian are bigger by the same k.) It is
    done in float, as raster.dilate does it, because the 8-bit blur would move in steps of 1/255.
    Rounding to 5 decimals makes the 0.9999998 the Gaussian leaves in the middle of the cells
    exactly 1, and keeps the zero far from them exactly 0.
    """
    rows, cols = cells.shape
    enlarged = Image.fromarray(cells.astype(np.float32)).resize(
        (cols * cell, rows * cell), Image.Resampling.BILINEAR
    )
    soft = _gaussian(np.asarray(enlarged)[: shape[0], : shape[1]], sigma)
    return np.round(soft, 5)


def _grid(open_cells: np.ndarray, outside: int) -> tuple[bytearray, tuple[int, int]]:
    """Open cells (a boolean picture) as a flat bytearray, for a search written in plain Python.

    Two rings go around the cells: one of `outside` (1 for the open surroundings of the picture,
    0 for none) and, around that, one of closed cells. A search can then step to the four
    neighbors of any cell without leaving the array. Returns the bytearray and its 2-D shape; the
    cell at (row, col) is at flat index (row + 2) * (columns + 4) + col + 2.
    """
    rows, cols = open_cells.shape
    padded = np.zeros((rows + 4, cols + 4), np.uint8)
    padded[1:-1, 1:-1] = outside
    padded[2:-2, 2:-2] = open_cells
    return bytearray(padded.tobytes()), padded.shape


def _cells(grid: bytearray, shape: tuple[int, int]) -> np.ndarray:
    """The cells of a _grid, as a boolean picture: which of them are open."""
    return np.frombuffer(grid, np.uint8).reshape(shape)[2:-2, 2:-2].astype(bool)


def _search(grid: bytearray, stride: int, seed: int) -> list[int]:
    """Breadth-first search from seed through the open cells that touch along a side. Closes the
    cells it reaches and returns their flat indexes (the queue is a list the loop keeps
    extending)."""
    grid[seed] = 0
    queue = [seed]
    for cell in queue:
        for near in (cell - 1, cell + 1, cell - stride, cell + stride):
            if grid[near]:
                grid[near] = 0
                queue.append(near)
    return queue


def _regions(cells: np.ndarray) -> tuple[list[list[int]], tuple[int, int]]:
    """The 4-connected regions of the True cells, found by breadth-first search. Returns the
    regions, each as a list of flat indexes, and the padded shape those indexes refer to (see
    _grid)."""
    grid, shape = _grid(cells, outside=0)
    regions = []
    for start in np.flatnonzero(np.frombuffer(grid, np.uint8)).tolist():
        if grid[start]:  # not reached by an earlier search
            regions.append(_search(grid, shape[1], start))
    return regions, shape


def _picture(regions: list[list[int]], shape: tuple[int, int]) -> np.ndarray:
    """The cells of some of the regions that _regions found, as a boolean picture."""
    flat = np.zeros(shape[0] * shape[1], bool)
    for region in regions:
        flat[region] = True
    return flat.reshape(shape)[2:-2, 2:-2]


def _big_regions(cells: np.ndarray) -> np.ndarray:
    """The cells that belong to a 4-connected region at least 1% the size of the largest."""
    regions, shape = _regions(cells)
    smallest = _MIN_REGION * max((len(region) for region in regions), default=0)
    return _picture([region for region in regions if len(region) >= smallest], shape)


def _small_regions(cells: np.ndarray, largest: float) -> np.ndarray:
    """The cells that belong to a 4-connected region of at most `largest` cells."""
    regions, shape = _regions(cells)
    return _picture([region for region in regions if len(region) <= largest], shape)


def _enclosed(background: np.ndarray) -> np.ndarray:
    """The background cells the border of the picture cannot reach through other background
    cells: the holes."""
    grid, shape = _grid(background, outside=1)
    _search(grid, shape[1], shape[1] + 1)  # from the ring around the picture
    return _cells(grid, shape)  # what the search left open


def clean_alpha(alpha: np.ndarray, k: float = 1.0) -> np.ndarray:
    """A new alpha with the stray bits of the cutout removed and the holes in the person filled.

    The work is done on cells of 4 x 4 pixels (270 x 152 for a 1080 x 608 frame), small enough
    for a search in plain Python. A cell is touched when any of its pixels is person (alpha of
    0.5 or more) and solid when all of them are. For an alpha bigger than the plan by k (a final
    render's), the cells are max(4, round(4 k)) pixels across and the fade's Gaussian is 2 k, so
    that it is as many cells, the same cleanup, and a person's edge grown and rounded by as much of
    the picture as at the plan.

    1. Touched cells that meet along a side form regions. Those under 1% of the largest region
       are dropped: specks, and background objects the model took for part of the person. The
       rest are grown by one cell and turned into a smooth 0-1 picture (see _fade), and the
       alpha is multiplied by it. The soft edge, hair and fingertips just past the 0.5 contour
       keep most of their alpha, and a fainter haze that reaches further out fades away over a
       few pixels instead of ending in a step. Far from the person the picture is exactly 0, so
       a speck ends at exactly 0.
    2. Everything that is not a solid cell of a kept region is background. Background that the
       border of the picture cannot reach is enclosed. An enclosed region of at most 1% of the
       kept cells is a hole in the person: it is grown by one cell, corners included, and its
       alpha is set to 1, so it is filled together with its soft rim. Only solid cells are
       added: a background cell that touches a hole at a corner, where the person is pinched,
       must not become part of the silhouette. A bigger enclosed region is a gap the model got
       right, such as the loop between an arm and the torso with a hand on the hip. It is left
       open, with no fill and no growth.
    """
    alpha = np.asarray(alpha, np.float32)
    cell, sigma = max(_CELL, round(_CELL * k)), _FADE_SIGMA * k
    person = alpha >= _PERSON
    kept = _big_regions(_pool(person, np.logical_or, cell))
    solid = _pool(person, np.logical_and, cell) & kept
    holes = _small_regions(_enclosed(~solid), _MAX_HOLE * kept.sum())

    result = alpha * _fade(_grow(kept), alpha.shape, cell, sigma)
    result[_enlarge(holes | (_grow(holes, square=True) & solid), alpha.shape, cell)] = 1.0
    return result


def person_regions(alpha: np.ndarray) -> list[float]:
    """The separate pieces of the person in an alpha (0.5 or more is person), as the share of the
    person's area that each has, the biggest first. One person is one piece, two people side by side
    are two big ones, and a hand that the cutout kept apart from the body is a small one. The pieces
    are the 4-connected regions of the 4 x 4 pixel cells clean_alpha works on, and the areas are
    counted in cells. Nobody in the alpha is an empty list."""
    cells = _pool(np.asarray(alpha) >= _PERSON, np.logical_or)
    regions, _ = _regions(cells)
    sizes = sorted((len(region) for region in regions), reverse=True)
    total = sum(sizes)
    return [size / total for size in sizes]


def adjust_edge(alpha: np.ndarray, px: float) -> np.ndarray:
    """Shrink or soften the border of the cutout by px pixels (the studio allows -4 to 4).

    A negative px shrinks the person: it grows the background by |px| with dilate and takes that
    away from 1, which leaves an edge 1 px wide. A positive px softens it with a Gaussian blur of
    sigma px / 2. 0 returns alpha itself.
    """
    if px < 0:
        return 1 - dilate(1 - alpha, -px)
    if px > 0:
        return blur(alpha, px / 2)
    return alpha


# --- the stored matte -------------------------------------------------------------------------


def matte_path(project_dir: Path, frame: int) -> Path:
    """Where the alpha of a frame is kept: <project>/matte/000123.png for frame 123."""
    return Path(project_dir) / "matte" / f"{frame:06d}.png"


def load_alpha(
    project_dir: Path,
    project: dict,
    frame: int,
    size: tuple[int, int] | None = None,
    path: Path | None = None,
) -> np.ndarray:
    """The alpha of a frame as the renderers use it: read from the project's matte folder,
    cleaned, then its edge adjusted by the project's matte.edge (0 when it is not set).

    size is the (width, height) wanted, for a card drawn bigger than the plan (project["work"]):
    the matte is read from `path`, a cutout made at that size such as a final render's, or else the
    project's own, which is enlarged to size when it is smaller (bilinear). The cleanup and the edge
    are then those of k = size width / plan width: see clean_alpha, and matte.edge is round(edge * k)
    px, a whole number of them.

    A cutout from `path` has to be exactly `size`: a matte of another size does not belong to the
    frame it would be drawn with, and enlarging it would hide that, so it is a ValueError, as a frame
    of the wrong size is (card.render_card).
    """
    with Image.open(matte_path(project_dir, frame) if path is None else path) as image:
        if size is not None and image.size != tuple(size):
            if path is not None:
                raise ValueError(
                    f"The matte is {image.size[0]}x{image.size[1]}, but the card is {size[0]}x{size[1]}."
                )
            image = image.convert("L").resize(tuple(size), Image.Resampling.BILINEAR)
        alpha = from_image(image)
    edge = project["matte"].get("edge", 0)
    if size is None:
        return adjust_edge(clean_alpha(alpha), edge)
    k = size[0] / project["work"]["width"]
    return adjust_edge(clean_alpha(alpha, k), round(edge * k))
