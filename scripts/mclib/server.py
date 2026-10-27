"""The studio server: the page's one way to the project folder.

`python mc.py serve PROJECT_DIR` runs it. It listens on 127.0.0.1 only, serves the studio page
(studio/), the fonts (fonts/) and the finished videos (the project's renders/), and answers the
JSON routes the page calls (_ROUTES). It uses Python's standard library, and the skill's own
modules for everything that is not plain HTTP.

What it keeps, and what it does not:

- No copy of the project. project.json is read from the disk on every request and written back
  through a merge patch, validate and save, so what Claude writes into the file while the studio is
  open (the answer to "Ask Claude") is not overwritten by the page's next edit. The writers of the
  file take turns, from the read to the save (project.UPDATE_LOCK, which the render's record of its
  status takes too).
- The cards it drew lately, as PNG bytes, under a hash of everything a card is drawn from: the
  project's content that is on the card, the style asked for, the width, the size of the plan it is
  drawn at and the stamps of the frame and the matte. A card that is asked for again, or after an
  edit that is not on the card (the timing), is not drawn again, and an edit that is (one note)
  draws only what changed, because the card module keeps the stage, the matte and the layout it can.
- One lock for the jobs that rewrite the project's files: a render, a preview and a new cutout.
  render owns renders/_work and is not re-entrant, and a new cutout rewrites frames/ and matte/
  while a render reads them. A second job gets a 409 at once. The lock is let go whatever the job
  does (try/finally).

Errors are {"error": "<sentence>"} with a status of 400 (the request, or something the person can
fix), 403 (a request from another web page), 404 (nothing there), 409 (busy) or 415 (a JSON route
that was not sent as JSON). A ValueError, RuntimeError or OSError carries the sentence, and its
traceback is printed on stderr as well, so that a bug behind a sentence is not silent. Anything else
is a 500 and a traceback on stderr.

Windows: a file that a video player has open cannot be replaced. This server never keeps a file open
between two chunks of an answer (see _stream), and replaces the files it writes through
fileio.retry_locked.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import traceback
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

from PIL import Image

from . import card, config, matte, onsets, palette, pipeline, render
from . import project as project_file
from .config import FONTS_DIR, STUDIO_DIR
from .fileio import retry_locked
from .styles import recipes

HOST = "127.0.0.1"
FIRST_PORT = 8340  # serve takes the first free port from here
PORT_TRIES = 100
CHUNK = 256 * 1024  # bytes of a file sent, or of an upload read, at a time
MAX_JSON = 4 * 1024 * 1024
MAX_UPLOAD = 1024 * 1024 * 1024  # a music track, or a picture from a camera

BUSY = "A render is already running."
REBUILDING = "The cutout is being rebuilt. Try again in a moment."
FORBIDDEN_HOST = "This studio only answers to the computer it runs on."
FORBIDDEN_ORIGIN = "That request came from another web page, so the studio refused it."

_SERVER_KEYS = ("clip", "status")  # written by the server alone; a patch from the page never sets them
_DRAGGABLE = (*project_file.NOTE_KEYS, "burst")  # the elements a note can be pinned for (positions)
# What a card does not depend on, so that a change of any of it is not a new card
_NOT_ON_THE_CARD = (
    "status", "style_request", "picked", "versions", "audio", "timing", "style_source", "clip", "notice",
)
_LOCAL_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",  # (not the system's table: a Windows registry can say text/plain)
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".mp4": "video/mp4",
    ".ttf": "font/ttf",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
_RESERVED = re.compile(r"(con|prn|aux|nul|com\d|lpt\d)(\..*)?", re.IGNORECASE)  # names Windows keeps for devices


def now_stamp() -> str:
    """The time now as the waiter reads it: UTC, to the millisecond, ending in +00:00. This is the
    one place that stamps status.done_at and style_request.at, so a stamp is never in local time (the
    waiter would take it for UTC) and never cut to the second (it could come before the moment a
    waiter started)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class HttpError(Exception):
    """An answer that is an error: a status and one sentence for the person."""

    def __init__(self, status: int, message: str, headers: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.headers = status, message, headers or {}


def _bad(message: str) -> HttpError:
    return HttpError(400, message)


def _missing(message: str) -> HttpError:
    return HttpError(404, message)


class _ByteCache:
    """The cards drawn lately, as bytes under a key, least recently used out first, kept under a
    number of bytes as well as a number of cards. It is safe to use from the server's threads."""

    def __init__(self, max_bytes: int = 192 * 1024 * 1024, max_items: int = 256):
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._size, self._max_bytes, self._max_items = 0, max_bytes, max_items
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        return None

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._size += len(value) - len(self._items.get(key, b""))
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > 1 and (len(self._items) > self._max_items or self._size > self._max_bytes):
                _, dropped = self._items.popitem(last=False)
                self._size -= len(dropped)


class Studio:
    """What one server knows and does for its project folder, apart from HTTP: reading and writing
    the project, the jobs, the card and the layout, the styles, the palettes and the uploads."""

    def __init__(self, project_dir: Path, matte_runner: Callable | None = None):
        self.dir = Path(project_dir).resolve()
        self.matte_runner = matte_runner  # a stand-in for matte.run_rvm, for tests; None is the real one
        self.rebuilding = False  # a job that rewrites frames/ and matte/ is running
        self.cards = _ByteCache()
        self._job = threading.Lock()  # one job that rewrites the project's files at a time

    # --- the project file ---------------------------------------------------------------------

    def load(self) -> dict:
        """The project, read from the disk now. (A read waits out a moment in which Windows has the
        file locked for a replace.)"""
        path = self.dir / project_file.PROJECT_FILE
        try:
            return retry_locked(lambda: project_file.load(self.dir), path)
        except FileNotFoundError:
            raise _missing(f"There is no project in {self.dir}.") from None
        except ValueError as error:
            raise _bad(f"{path} cannot be read ({error}). Fix it, or move it away to start a new project.") from error

    def update(
        self, patch: dict | None = None, mutate: Callable[[dict], None] | None = None
    ) -> tuple[dict, list[str]]:
        """Change the project on the disk and return it, with the warnings of validate.

        The file is read again, then patch (a JSON merge patch from the page) is applied, then mutate
        (a function that edits the project in place: for what the server itself writes, and for values
        that a merge patch cannot say, such as a null that stays) and the result is validated and
        saved. A patch never sets the keys that only the server writes (_SERVER_KEYS), and a patch of
        the freeze frame has to stay inside the clip. ValueError, with a sentence, when the result is
        not a valid project; nothing is saved then. The file is not written when nothing changed.
        """
        if patch is not None:
            patch = {key: value for key, value in patch.items() if key not in _SERVER_KEYS}
        with project_file.UPDATE_LOCK:
            return self._apply(patch, mutate)

    def _apply(self, patch: dict | None, mutate: Callable[[dict], None] | None) -> tuple[dict, list[str]]:
        current = self.load()
        merged = project_file.merge_patch(current, patch or {})
        if mutate is not None:
            mutate(merged)
        audio = merged.get("audio")
        if isinstance(audio, dict):
            audio.setdefault("track", None)  # a null in a patch deleted it, and render reads audio["track"]
            audio.setdefault("original", True)  # a null here puts the default back: the clip keeps its sound
        warnings = project_file.validate(merged)
        if patch and isinstance(patch.get("freeze"), dict) and "frame" in patch["freeze"]:
            self._check_freeze(merged)
        if merged != current:
            project_file.save(self.dir, merged)
        return merged, warnings

    @staticmethod
    def _check_freeze(project: dict) -> None:
        frames, frame = project["clip"].get("frames"), project["freeze"]["frame"]
        if isinstance(frames, int) and not isinstance(frames, bool) and frame >= frames:
            raise _bad(f"The freeze frame must be between 0 and {frames - 1}: the clip has {frames} frames.")

    # --- jobs ---------------------------------------------------------------------------------

    @contextmanager
    def job(self, rebuilds_cutout: bool = False):
        """Hold the one lock of the jobs that rewrite the project's files, or raise the 409 at once.
        rebuilds_cutout says that frames/ and matte/ are being rewritten, so that they are not read
        meanwhile (see check_cutout_ready). The lock is let go whatever happens."""
        if not self._job.acquire(blocking=False):
            raise HttpError(409, BUSY)
        self.rebuilding = rebuilds_cutout
        try:
            yield
        finally:
            self.rebuilding = False
            self._job.release()

    def check_cutout_ready(self) -> None:
        if self.rebuilding:
            raise HttpError(409, REBUILDING)

    def preview(self) -> dict:
        with self.job():
            result = render.render(self.dir, preview=True, matte_runner=self.matte_runner)
        return {"url": "/renders/" + result.video.name, "warning": result.warning, "notes": result.notes}

    def render_final(self) -> dict:
        """Render, then stamp status.done_at, which is what wakes the waiter with DONE. The stamp is
        written after the render has saved its own status, and to the file as it is then."""
        with self.job():
            result = render.render(self.dir, matte_runner=self.matte_runner)
            made = (result.video, result.with_music, result.cover)
            outputs = [f"renders/{path.name}" for path in made if path is not None]
            done = now_stamp()
            self.update(mutate=lambda project: project["status"].update(done_at=done))
        return {"outputs": outputs, "warning": result.warning, "notes": result.notes, "done_at": done}

    def refine(self, detail: str) -> dict:
        """Cut the person out again around the freeze frame at a finer detail: the frames from 12
        before it to the one after, by the same runner as the first cutout (pipeline.refine_matte, which
        `mc.py refine` runs too). matte.detail is saved when the cutout is made, so a cutout that fails
        changes nothing."""
        if detail not in pipeline.DETAILS:
            raise _bad(pipeline.BAD_DETAIL)
        with self.job(rebuilds_cutout=True):
            pipeline.refine_matte(self.dir, self.load(), detail, self.matte_runner)
            project, warnings = self.update()  # the project as it is now, with the new detail
        return {"project": project, "warnings": warnings}

    # --- the card, the frames and the layout --------------------------------------------------

    def _need_picture(self, project: dict) -> None:
        """A 404, with a sentence, unless the freeze frame's picture and cutout are in the folder."""
        frame = project["freeze"]["frame"]
        needed = (("picture", self.dir / "frames" / f"{frame:06d}.jpg"), ("cutout", matte.matte_path(self.dir, frame)))
        for what, path in needed:
            if not path.is_file():
                raise _missing(
                    f"Frame {frame} has no {what} ({path.name} is missing). "
                    "Pick another frame, or run init on the clip again."
                )

    def card_png(self, style_id: str | None, width: int | None) -> bytes:
        """The card as a PNG at the plan's size, all its elements visible. style_id draws a built-in or
        a saved style instead of the project's; width (never more than the card's own) shrinks it."""
        self.check_cutout_ready()
        project = self.load()
        recipe = None
        if style_id:
            try:
                recipe = recipes.get_style(style_id)["recipe"]
            except ValueError as error:
                raise _missing(str(error)) from None
        self._need_picture(project)
        key = self._card_key(project, recipe, width)
        cached = self.cards.get(key)
        if cached is not None:
            return cached
        image = card.render_card(self.dir, project, style=recipe)
        if width is not None and width < image.width:
            image = image.resize((width, max(1, round(image.height * width / image.width))), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "PNG", compress_level=1)  # a card is a photograph: more effort saves little
        self.cards.put(key, buffer.getvalue())
        return buffer.getvalue()

    def _card_key(self, project: dict, recipe: dict | None, width: int | None) -> str:
        """A hash of what the card is drawn from. A style asked for replaces the project's, so the
        project's own is not part of that key (a thumbnail does not change when the person's style
        does)."""
        frame = project["freeze"]["frame"]
        shown = {
            key: value for key, value in project.items()
            if key not in _NOT_ON_THE_CARD and not (recipe is not None and key == "style")
        }
        files = [_stamp(self.dir / "frames" / f"{frame:06d}.jpg"), _stamp(matte.matte_path(self.dir, frame))]
        plan = [project["work"]["width"], project["work"]["height"]]  # (it is among `shown` as well)
        text = json.dumps([shown, recipe, width, plan, files], sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def layout(self) -> dict:
        """Where the stage is and where the notes can be dragged, in px of the plan (the size the card
        is drawn at, and the frames are cut at): {stage: [0, 0, w, h], elements: {key: [x, y, w, h]},
        flags: {key: "cut" | "clash"}}. Only the elements a note can be pinned for are in it (the HUD's
        labels are not), and flags has only the ones that do not fit: a note whose text was cut (too long
        for its box, or with a character its font can't draw), or one that is in the way of the face, the
        title room or the safe area, or another note."""
        self.check_cutout_ready()
        project = self.load()
        self._need_picture(project)
        boxes = card.elements(self.dir, project)
        width, height = project["work"]["width"], project["work"]["height"]
        drag = {key: box for key, box in boxes.items() if key in _DRAGGABLE}
        return {
            "stage": [0, 0, width, height],
            "elements": {key: [box.x, box.y, box.w, box.h] for key, box in drag.items()},
            "flags": {key: "cut" if box.cut else "clash" for key, box in drag.items() if box.clash or box.cut},
        }

    def frames(self) -> dict:
        """The frames the project can freeze on, as {"ranges": [[first, last], ...]}: the ones that
        have a picture (frames/) and a cutout (matte/) in the folder. A project made by init has all of
        them, and the page asks, so that it never offers a frame that cannot be drawn."""
        self.check_cutout_ready()

        def numbers(folder: str, suffix: str) -> set[int]:
            try:
                names = [entry.name for entry in os.scandir(self.dir / folder)]
            except OSError:
                return set()
            return {int(name[:6]) for name in names if re.fullmatch(r"\d{6}" + re.escape(suffix), name)}

        ranges: list[list[int]] = []
        for number in sorted(numbers("frames", ".jpg") & numbers("matte", ".png")):
            if ranges and ranges[-1][1] == number - 1:
                ranges[-1][1] = number
            else:
                ranges.append([number, number])
        return {"ranges": ranges}

    def frame_jpeg(self, number: int, width: int | None) -> tuple[bytes, str]:
        """The JPEG of a frame, whole or shrunk to `width` (never larger than it is), and a validator
        for it."""
        self.check_cutout_ready()
        path = self.dir / "frames" / f"{number:06d}.jpg"
        try:
            stamp = _stamp(path)
        except OSError:
            raise _missing(f"There is no frame {number} in this project.") from None
        etag = f'"{stamp[0]:x}-{stamp[1]:x}-{width or 0}"'
        if width is None:
            return path.read_bytes(), etag
        with Image.open(path) as image:
            if width >= image.width:
                return path.read_bytes(), etag
            height = max(1, round(image.height * width / image.width))
            image.draft("RGB", (width, height))  # a JPEG is decoded at a fraction of its size
            small = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        small.save(buffer, "JPEG", quality=80)
        return buffer.getvalue(), etag

    # --- styles and colors --------------------------------------------------------------------

    @staticmethod
    def options() -> dict:
        """The tables the page builds its controls from, so that they are the code's own."""
        return {
            "options": recipes.OPTIONS,
            "labels": recipes.LABELS,
            "palettes": recipes.PALETTES,
            "roles": recipes.ROLES,
            "soft_limits": project_file.SOFT_LIMITS,
            "timing": {"defaults": config.TIMING_DEFAULTS, "ranges": config.TIMING_RANGES},
        }

    def save_style(self, name) -> dict:
        if not isinstance(name, str) or not name.strip():
            raise _bad("Name your style first.")
        return recipes.save_user_style(self.load()["style"], name.strip())

    def colors_of(self, upload) -> dict:
        """The five colors of an image (an open file) and the contrast the project's lettering would
        have with them. Nothing is saved."""
        try:
            colors = palette.palette_from_image(upload)
        except palette.NoVisiblePixels:
            raise _bad("That image has no visible pixels.") from None
        except Image.DecompressionBombError:
            raise _bad("That image is too large to read. Use a smaller one.") from None
        except (OSError, ValueError):  # Pillow's UnidentifiedImageError is an OSError
            raise _bad("That file is not an image I can read.") from None
        style = {**self.load()["style"], "palette": colors}
        return {"palette": colors, "contrast": palette.text_contrast(style)}

    def ask_for_a_style(self, text, image) -> dict:
        """Record what the person asked Claude for: style_request = {text, image, at, answered_at}.
        The waiter wakes on at. Claude answers by writing the style and answered_at. It is set by a
        function and not by a patch, because the nulls of the request (no image, not answered) are
        values, and a merge patch would delete them."""
        if not isinstance(text, str) or not text.strip():
            raise _bad("Write a few words about the style you want first.")
        if image is not None and not isinstance(image, str):
            raise _bad("The image has to be the path of a picture, or null.")
        request = {"text": text.strip(), "image": image or None, "at": now_stamp(), "answered_at": None}
        project, _ = self.update(mutate=lambda p: p.update(style_request=request))
        return project

    # --- uploads ------------------------------------------------------------------------------

    def keep_track(self, saved: Path, hits: list[float]) -> dict:
        """Put an uploaded music track into the project: the track (a path inside the project folder)
        and the times of its hits."""
        track = saved.relative_to(self.dir).as_posix()
        project, warnings = self.update(mutate=lambda p: p["audio"].update(track=track, onsets=hits))
        return {"project": project, "warnings": warnings, "onsets": hits}


def _stamp(path: Path) -> tuple[int, int]:
    """When a file was last written and its size: a change to either is a new picture."""
    info = path.stat()
    return info.st_mtime_ns, info.st_size


def _byte_range(spec: str, size: int) -> tuple[int, int] | None:
    """The first and the last byte that a Range header asks for in a file of `size` bytes, or None
    when it is to be ignored and the whole file sent: more than one range, a unit that is not bytes,
    a range that ends before it starts. A range that starts past the end is a 416, and so is a
    suffix of no bytes."""
    found = _RANGE.fullmatch(spec.strip())
    if not found or not size:
        return None
    first, last = found.groups()
    if first:
        start = int(first)
        if last and int(last) < start:
            return None
        if start >= size:
            raise HttpError(416, "That part of the file does not exist.", {"Content-Range": f"bytes */{size}"})
        return start, min(int(last), size - 1) if last else size - 1
    if last:  # the last n bytes
        if int(last) == 0:
            raise HttpError(416, "That part of the file does not exist.", {"Content-Range": f"bytes */{size}"})
        return max(size - int(last), 0), size - 1
    return None


def _safe_name(raw: str, default: str) -> str:
    """The file name an upload is saved under, from the name the page sent (percent-encoded in the
    header): only its last part, with nothing in it that a Windows name cannot have, no dots or
    spaces at the ends and not a device name (con, nul, ...), cut to 120 characters. `default` when
    nothing is left."""
    name = re.split(r"[\\/]", unquote(raw, errors="replace"))[-1]
    name = re.sub(r'[\x00-\x1f<>:"|?*]', "_", name).strip(" .")
    if len(name) > 120:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 120 - len(ext) - 1] + dot + ext) if dot and len(ext) < 20 else name[:120]
    if _RESERVED.fullmatch(name):
        name = "_" + name
    return name or default


# --- HTTP -----------------------------------------------------------------------------------

_ROUTES = {
    ("GET", "/"): "_get_index",
    ("GET", "/api/project"): "_get_project",
    ("POST", "/api/project"): "_post_project",
    ("GET", "/api/options"): "_get_options",
    ("GET", "/api/card.png"): "_get_card",
    ("GET", "/api/layout"): "_get_layout",
    ("GET", "/api/frames"): "_get_frames",
    ("GET", "/api/styles"): "_get_styles",
    ("POST", "/api/styles"): "_post_styles",
    ("POST", "/api/palette"): "_post_palette",
    ("POST", "/api/style/reference"): "_post_reference",
    ("POST", "/api/style/ask"): "_post_ask",
    ("POST", "/api/matte/refine"): "_post_refine",
    ("POST", "/api/audio"): "_post_audio",
    ("POST", "/api/preview"): "_post_preview",
    ("POST", "/api/render"): "_post_render",
}
_PREFIXES = {
    "/studio/": "_get_studio",
    "/fonts/": "_get_font",
    "/renders/": "_get_render",
    "/api/frame/": "_get_frame",
}


class _Handler(BaseHTTPRequestHandler):
    server_version = "MainCharacterStudio"
    protocol_version = "HTTP/1.0"  # one request to a connection: nothing to keep in step

    @property
    def studio(self) -> Studio:
        return self.server.studio

    def log_message(self, format, *args):  # the studio does not log every request
        pass

    def send_error(self, code, message=None, explain=None):
        """Errors the base class finds in the request (a line it cannot read, a method it does not
        know) are answered as every other error is."""
        self._send_json(code, {"error": message or f"{HTTPStatus(code).phrase}."})

    do_GET = do_HEAD = do_POST = lambda self: self._serve()

    # --- the one entrance ---------------------------------------------------------------------

    def _serve(self) -> None:
        self._consumed = False
        try:
            self._check_local()
            split = urlsplit(self.path)
            method = "GET" if self.command == "HEAD" else self.command
            name = _ROUTES.get((method, split.path))
            if name is None:
                for prefix, candidate in _PREFIXES.items():
                    if method == "GET" and split.path.startswith(prefix):
                        name = candidate
                        split = split._replace(path=split.path[len(prefix):])
                        break
            if name is None:
                raise _missing("There is nothing at that address.")
            getattr(self, name)(split)
        except HttpError as error:
            self._fail(error.status, error.message, error.headers)
        except (ConnectionError, TimeoutError):
            self.close_connection = True  # the browser went away
        except (ValueError, RuntimeError, OSError) as error:  # each of these carries a sentence for the person
            traceback.print_exc()  # who runs the studio can still find the bug behind a sentence
            self._fail(400, str(error) or error.__class__.__name__)
        except Exception:
            traceback.print_exc()
            self._fail(500, "Something went wrong inside the studio. The reason is in the window it runs in.")

    def _fail(self, status: int, message: str, headers: dict | None = None) -> None:
        """Answer with an error. A request body that nobody read is read first (a small one), because
        a server that closes the connection with it unread can make the browser lose the answer."""
        if not self._consumed:
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if 0 < length <= MAX_JSON:
                    self.rfile.read(length)
            except (ValueError, OSError):
                pass
        self._send_json(status, {"error": message}, headers)

    def _check_local(self) -> None:
        """Refuse a request that is not from this server's own page. A web page on another site can make
        the browser send requests to localhost: through a name of its own that points here (the Host header
        says which, and it has to be this computer's), or by posting a form (the Origin header says where
        from, and for a POST it has to be this very server: the same name and the same port as the Host the
        request was sent to, so that a page on another port of this computer, such as a dev server, is refused
        too)."""
        asked = self.headers.get("Host") or ""
        if urlsplit("//" + (asked or "localhost")).hostname not in _LOCAL_NAMES:
            raise HttpError(403, FORBIDDEN_HOST)
        origin = self.headers.get("Origin")
        if self.command == "POST" and origin is not None:
            where = urlsplit(origin)
            own = where.netloc.lower() == asked.lower() if asked else where.hostname in _LOCAL_NAMES
            if where.scheme != "http" or not own:
                raise HttpError(403, FORBIDDEN_ORIGIN)

    # --- reading what the browser sent --------------------------------------------------------

    def _content_length(self, limit: int) -> int:
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            raise _bad("Send the file with a length, not in chunks.")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise _bad("The request has a length that is not a number.") from None
        if length < 0 or length > limit:
            raise _bad(f"That is too big: the limit is {limit // (1024 * 1024)} MB.")
        return length

    def _json_body(self) -> dict:
        """The JSON object that a JSON route was sent ({} for no body). These routes take only
        application/json: a web page on another site can post text/plain to localhost without asking first,
        but JSON makes the browser ask (an OPTIONS request, which this server answers with a 501), so it
        cannot. The routes that take a picture or a sound (_copy_body) keep their own content types."""
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            raise HttpError(415, "Send the request as application/json.")
        length = self._content_length(MAX_JSON)
        raw = self.rfile.read(length) if length else b""
        self._consumed = True
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw.decode("utf-8-sig"))
        except ValueError:  # (UnicodeDecodeError is one)
            raise _bad("The request is not valid JSON.") from None
        if not isinstance(data, dict):
            raise _bad("The request has to be a JSON object.")
        return data

    def _copy_body(self, out) -> int:
        """Copy the request's body into the open file `out`, a chunk at a time, and return its size."""
        length = left = self._content_length(MAX_UPLOAD)
        while left:
            chunk = self.rfile.read(min(left, CHUNK))
            if not chunk:
                raise _bad("The upload stopped before it was finished.")
            out.write(chunk)
            left -= len(chunk)
        self._consumed = True
        return length

    def _receive(self, folder: str, default: str) -> tuple[Path, Path]:
        """Read an upload into a temporary file in <project>/<folder> and return it with the path
        it is to get (from the X-Filename header; see _safe_name). The caller moves the one to the
        other with _keep, or deletes the temporary file."""
        where = self.studio.dir / folder
        where.mkdir(parents=True, exist_ok=True)
        dest = where / _safe_name(self.headers.get("X-Filename") or "", default)
        temp = where / f".upload-{uuid4().hex[:8]}{dest.suffix}"
        try:
            with open(temp, "wb") as out:
                self._copy_body(out)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        return temp, dest

    @staticmethod
    def _keep(temp: Path, dest: Path) -> None:
        """Put an upload in its place, over a file of the same name. A file that a program has open
        is waited for (fileio.retry_locked)."""
        try:
            retry_locked(lambda: os.replace(temp, dest), dest)
        finally:
            temp.unlink(missing_ok=True)

    # --- sending ------------------------------------------------------------------------------

    def _write_body(self, chunk: bytes) -> None:
        self.wfile.write(chunk)

    def _send_bytes(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD" and body:
            self._write_body(body)

    def _send_json(self, status: int, data, headers: dict | None = None) -> None:
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8", {"Cache-Control": "no-store", **(headers or {})})

    def _not_modified(self, etag: str) -> bool:
        """Whether the browser already has this version (If-None-Match), and was told so."""
        sent = [tag.strip().removeprefix("W/") for tag in (self.headers.get("If-None-Match") or "").split(",")]
        if etag not in sent:
            return False
        self.send_response(304)
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        return True

    def _send_file(self, path: Path) -> None:
        """Send a file, with its validator, and with the part of it that a Range header asks for (a
        video element asks for the middle of a video, and needs a 206 to seek)."""
        content_type = _TYPES.get(path.suffix.lower(), "application/octet-stream")
        try:
            stamp = _stamp(path)
        except OSError:
            raise _missing(f"There is no file {path.name} here.") from None
        size = stamp[1]
        etag = f'"{stamp[0]:x}-{size:x}"'
        if self._not_modified(etag):
            return
        start, end, partial = 0, size - 1, False
        if self.headers.get("If-Range", etag) == etag:  # (a resumed download of an older file starts again)
            wanted = _byte_range(self.headers.get("Range") or "", size)
            if wanted is not None:
                (start, end), partial = wanted, True
        length = max(end - start + 1, 0)
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command != "HEAD":
            self._stream(path, start, length, stamp)

    def _stream(self, path: Path, start: int, length: int, stamp: tuple[int, int]) -> None:
        """Send `length` bytes of a file from `start`, with the file open for one chunk at a time. A
        client that reads slowly, such as a video element that is paused, then never keeps a file
        open, and Windows can replace it when the next render is done. If the file changes meanwhile
        the answer stops short (the client sees it was cut and asks again) and never mixes two files."""
        sent = 0
        while sent < length:
            with open(path, "rb") as source:
                now = os.fstat(source.fileno())
                if (now.st_mtime_ns, now.st_size) != stamp:
                    return
                source.seek(start + sent)
                chunk = source.read(min(CHUNK, length - sent))
            if not chunk:
                return
            self._write_body(chunk)
            sent += len(chunk)

    # --- routes: files ------------------------------------------------------------------------

    @staticmethod
    def _file_in(folder: Path, raw: str) -> Path:
        """The file a URL names in a folder. The name has to be the name of a file that is there, as
        the folder lists it: nothing with a slash, a backslash or a colon in it, so no way out of the
        folder (not through ..%2F, not with a drive letter), and no device name, and a name that
        differs in case or in a dot at the end is another name."""
        name = unquote(raw)
        if not name or name in (".", "..") or any(mark in name for mark in "/\\:\0"):
            raise _missing("There is no such file.")
        try:
            names = {entry.name for entry in os.scandir(folder) if entry.is_file()}
        except OSError:
            names = set()
        if name not in names:
            raise _missing("There is no such file.")
        return folder / name

    def _get_index(self, split) -> None:
        self._send_file(self._file_in(STUDIO_DIR, "index.html"))

    def _get_studio(self, split) -> None:
        self._send_file(self._file_in(STUDIO_DIR, split.path))

    def _get_font(self, split) -> None:
        self._send_file(self._file_in(FONTS_DIR, split.path))

    def _get_render(self, split) -> None:
        self._send_file(self._file_in(self.studio.dir / "renders", split.path))

    # --- routes: the project ------------------------------------------------------------------

    def _get_project(self, split) -> None:
        self._send_json(200, self.studio.load())

    def _post_project(self, split) -> None:
        project, warnings = self.studio.update(self._json_body())
        contrast = palette.text_contrast(project["style"])
        self._send_json(200, {"project": project, "warnings": warnings, "contrast": contrast})

    def _get_options(self, split) -> None:
        self._send_json(200, self.studio.options())

    # --- routes: pictures ---------------------------------------------------------------------

    @staticmethod
    def _width(query: dict) -> int | None:
        """The w of a query: a whole number of pixels, 1 or more, or None when there is none."""
        found = query.get("w")
        if not found:
            return None
        if not found[0].isdigit() or int(found[0]) < 1:
            raise _bad("The width (w) has to be a whole number of pixels.")
        return int(found[0])

    def _get_card(self, split) -> None:
        query = parse_qs(split.query)
        body = self.studio.card_png((query.get("style") or [None])[0], self._width(query))
        self._send_bytes(200, body, "image/png", {"Cache-Control": "no-store"})

    def _get_layout(self, split) -> None:
        self._send_json(200, self.studio.layout())

    def _get_frames(self, split) -> None:
        self._send_json(200, self.studio.frames())

    def _get_frame(self, split) -> None:
        if not re.fullmatch(r"\d{1,9}", split.path):
            raise _missing(f"There is no frame {unquote(split.path)}.")
        body, etag = self.studio.frame_jpeg(int(split.path), self._width(parse_qs(split.query)))
        if not self._not_modified(etag):
            self._send_bytes(200, body, "image/jpeg", {"ETag": etag, "Cache-Control": "no-cache"})

    # --- routes: styles and colors ------------------------------------------------------------

    def _get_styles(self, split) -> None:
        self._send_json(200, recipes.list_styles())

    def _post_styles(self, split) -> None:
        self._send_json(200, self.studio.save_style(self._json_body().get("name")))

    def _post_palette(self, split) -> None:
        with tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024) as upload:
            if self._copy_body(upload) == 0:
                raise _bad("Send the image in the body of the request.")
            upload.seek(0)
            answer = self.studio.colors_of(upload)
        self._send_json(200, answer)

    def _post_reference(self, split) -> None:
        temp, dest = self._receive("refs", "reference")
        self._keep(temp, dest)
        self._send_json(200, {"path": str(dest)})

    def _post_ask(self, split) -> None:
        body = self._json_body()
        self._send_json(200, self.studio.ask_for_a_style(body.get("text"), body.get("image")))

    # --- routes: the cutout, the music --------------------------------------------------------

    def _post_refine(self, split) -> None:
        self._send_json(200, self.studio.refine(self._json_body().get("detail")))

    def _post_audio(self, split) -> None:
        temp, dest = self._receive("audio", "track")
        try:
            # (MediaError, a sentence, when it is not sound.) Read before it replaces the old track
            hits = onsets.detect_onsets(onsets.load_audio_mono(temp))
            self._keep(temp, dest)
        finally:
            temp.unlink(missing_ok=True)
        self._send_json(200, self.studio.keep_track(dest, hits))

    # --- routes: the renders ------------------------------------------------------------------

    def _post_preview(self, split) -> None:
        self._json_body()
        self._send_json(200, self.studio.preview())

    def _post_render(self, split) -> None:
        self._json_body()
        self._send_json(200, self.studio.render_final())


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64  # a page asks for many thumbnails at once; the default of 5 can refuse some on Windows
    # On Windows a socket that sets SO_REUSEADDR can bind a port that another program is listening
    # on, and then "the first free port" would be any port. Elsewhere it only lets a restart bind a
    # port that is waiting out its last connections.
    allow_reuse_address = os.name != "nt"

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return  # the browser went away in the middle of an answer
        super().handle_error(request, client_address)


def make_server(project_dir: Path, port: int = 0, matte_runner: Callable | None = None) -> ThreadingHTTPServer:
    """The studio server for a project folder, bound to 127.0.0.1 on `port` (0 for any free one; the
    one it got is httpd.server_address[1]) but not running: call serve_forever on it. OSError when
    the port is in use, FileNotFoundError when there is no project in the folder. matte_runner stands
    in for matte.run_rvm, for the new cutouts of the Detail control and of a final render."""
    folder = Path(project_dir)
    if not (folder / project_file.PROJECT_FILE).is_file():
        raise FileNotFoundError(f"There is no project in {folder.resolve()}. Run init first.")
    httpd = _Server((HOST, port), _Handler)
    httpd.studio = Studio(folder, matte_runner)
    return httpd


def bind(project_dir: Path, port: int | None = None, matte_runner: Callable | None = None) -> ThreadingHTTPServer:
    """make_server on `port`, or on the first free port from FIRST_PORT when there is none asked for.
    A port that was asked for and is in use is an error that says so, not a reason to take another."""
    if port is not None:
        try:
            return make_server(project_dir, port, matte_runner)
        except FileNotFoundError:
            raise
        except OSError as error:
            raise OSError(f"Port {port} is in use. Close what is using it, or leave out --port to take a free one.") from error
    for candidate in range(FIRST_PORT, FIRST_PORT + PORT_TRIES):
        try:
            return make_server(project_dir, candidate, matte_runner)
        except FileNotFoundError:
            raise
        except OSError:
            continue  # in use: try the next
    raise OSError(f"No port from {FIRST_PORT} to {FIRST_PORT + PORT_TRIES - 1} is free. Close a studio you are not using.")


def serve(project_dir: Path, port: int | None = None) -> None:
    """Run the studio for a project folder until it is stopped: on `port`, or on the first free port
    from 8340. Prints "URL http://127.0.0.1:<port>/" as soon as it listens, which is what Claude
    gives the person."""
    httpd = bind(project_dir, port)
    print(f"URL http://{HOST}:{httpd.server_address[1]}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
