"""Tests for the studio server (scripts/mclib/server.py): the routes the page uses, how it keeps
the project file, and how it protects what is on disk.

The server runs in a thread on a free port (the `server` fixture in conftest, on the fake project),
and each test talks to it over HTTP the way the page does. The matte runner is a fake, and a few
tests replace render.render with one that waits, so that no test needs ffmpeg for the lock.
"""

import http.client
import io
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mclib import analyze, matte, render, wait
from mclib import project as project_file
from mclib import server as studio_server
from mclib.palette import text_contrast
from mclib.raster import from_image
from mclib.render import RenderResult
from mclib.styles import engine

BLOCKS = ("#101010", "#f0f0f0", "#e8261c", "#1d4ed8", "#808080")  # dark, light, red, blue, gray
BUSY = {"error": "A render is already running."}
STAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3,}(Z|\+00:00)")  # UTC, to the millisecond


def picture(reply) -> Image.Image:
    """The image a reply holds."""
    return Image.open(io.BytesIO(reply.body))


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def blocks_png(colors=BLOCKS, size=(64, 64)) -> bytes:
    """A PNG of the colors as bands side by side (at 64 x 64 the palette reads it as it is, with no blend)."""
    pixels = np.zeros((size[1], size[0], 3), np.uint8)
    for color, columns in zip(colors, np.array_split(np.arange(size[0]), len(colors))):
        pixels[:, columns] = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return png_bytes(Image.fromarray(pixels))


def make_result(folder, preview=False) -> RenderResult:
    """What render.render returns, for a fake that wrote nothing."""
    renders = Path(folder) / "renders"
    if preview:
        return RenderResult(renders / "preview.mp4", None, 10, 2.0, 5.0, [])
    return RenderResult(
        renders / "synthetic-main-character.mp4", renders / "synthetic-cover.png", 10, 2.0, 5.0,
        ["Rendered at 3840x2026; your clip is 4096x2160, bigger than 4K."],
    )


class Gate:
    """A stand-in for render.render or a matte runner that waits until the test lets it go, so that
    a request can be held while another one is tried. calls lists what it was called with."""

    def __init__(self, folder, inner=None):
        self.folder, self.inner = folder, inner
        self.started, self.release, self.calls = threading.Event(), threading.Event(), []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.started.set()
        assert self.release.wait(60), "the test never let the held call go"
        if self.inner is not None:
            return self.inner(*args, **kwargs)
        return make_result(self.folder, preview=bool(kwargs.get("preview")))


def hold(server, path, data=None):
    """POST in a thread, and return the thread and a list that gets the reply."""
    got = []
    thread = threading.Thread(target=lambda: got.append(server.post(path, data)))
    thread.start()
    return thread, got


# --- the project -------------------------------------------------------------------------------


def test_get_project(server):
    reply = server.get("/api/project")

    assert reply.status == 200
    assert reply.headers["content-type"] == "application/json; charset=utf-8"
    project = reply.json()
    assert project == server.project() and project["schema"] == 2
    assert project["copy"]["notes"]["tr"] == "@your_handle"


def test_patch_project_validates(server):
    bad = server.post("/api/project", {"style": {"bg": "glitter"}})
    assert bad.status == 400
    assert "bg" in bad.json()["error"] and "glitter" in bad.json()["error"]
    assert server.project()["style"]["bg"] == "dots"  # nothing was saved

    good = server.post("/api/project", {"timing": {"hold": 4}})
    assert good.status == 200
    answer = good.json()
    assert set(answer) == {"project", "warnings", "contrast"}
    assert answer["project"]["timing"]["hold"] == 4 and server.project()["timing"]["hold"] == 4
    assert answer["contrast"] == pytest.approx(text_contrast(server.project()["style"]))
    assert isinstance(answer["warnings"], list)

    # A patch is a merge patch: it keeps what it does not name, and the warnings are soft limits
    long_note = server.post("/api/project", {"copy": {"notes": {"tl": "x" * 31}}})
    assert long_note.status == 200 and long_note.json()["project"]["copy"]["notes"]["bl"]
    assert any("copy.notes.tl is 31 characters" in line for line in long_note.json()["warnings"])


def test_bad_requests_are_plain_sentences(server):
    cases = [
        ("POST", "/api/project", b"{nope", 400),  # not JSON
        ("POST", "/api/project", b"[1, 2]", 400),  # JSON, but not an object
        ("GET", "/api/nothing-here", None, 404),
        ("POST", "/api/nothing-here", b"{}", 404),
        ("GET", "/api/card.png?w=big", None, 400),
        ("GET", "/api/card.png?style=no-such-style", None, 404),
        ("GET", "/api/frame/abc", None, 404),
        ("GET", "/api/frame/9999", None, 404),
        ("POST", "/api/styles", b'{"name": ""}', 400),
        ("POST", "/api/style/ask", b'{"text": "   "}', 400),
        ("POST", "/api/matte/refine", b'{"detail": "huge"}', 400),
        ("POST", "/api/layout-mode", b'{"layout": "banana"}', 404),  # (the format switch is gone)
    ]
    for method, path, body, status in cases:
        sent_as = {"Content-Type": "application/json"} if method == "POST" else None  # (as the page sends it)
        reply = server.request(method, path, body=body, headers=sent_as)
        assert reply.status == status, (method, path, reply.body)
        error = reply.json()["error"]
        assert error and error[0].isupper() and error.endswith("."), (path, error)


def test_a_sentence_error_still_leaves_its_traceback_in_the_servers_window(server, monkeypatch, capsys):
    # The person gets the plain sentence (a 400). Whoever runs the studio can still find the bug behind it
    def broken(folder, **kwargs):
        raise ValueError("The clip is broken.")

    monkeypatch.setattr(render, "render", broken)
    capsys.readouterr()

    reply = server.post("/api/render")

    assert reply.status == 400 and reply.json() == {"error": "The clip is broken."}
    printed = capsys.readouterr().err
    assert "Traceback" in printed and "ValueError: The clip is broken." in printed

    # A request that is wrong in a way the server expects (a refusal, nothing there) is not a bug: it prints nothing
    assert server.get("/api/frame/9999").status == 404 and server.post("/api/styles", {"name": ""}).status == 400
    assert capsys.readouterr().err == ""


def test_patch_drops_the_keys_only_the_server_writes(server):
    before = server.project()

    reply = server.post("/api/project", {
        "clip": {"frames": 5, "path": "elsewhere.mp4"},
        "status": {"done_at": "2020-01-01T00:00:00.000+00:00", "outputs": ["renders/fake.mp4"]},
        "timing": {"hold": 4},
    })

    assert reply.status == 200
    saved = server.project()
    assert saved["clip"] == before["clip"] and saved["status"] == before["status"]
    assert saved["timing"]["hold"] == 4  # the rest of the patch went through


def test_patch_keeps_the_freeze_inside_the_clip(server):
    frames = server.project()["clip"]["frames"]
    assert frames == 120

    for bad in (-1, frames, frames + 5, 99999):
        reply = server.post("/api/project", {"freeze": {"frame": bad}})
        assert reply.status == 400, bad
        assert "freeze" in reply.json()["error"].lower()
    assert reply.json()["error"].endswith(".")
    assert server.project()["freeze"]["frame"] == 60  # unchanged

    for good in (0, frames - 1):
        assert server.post("/api/project", {"freeze": {"frame": good}}).status == 200
        assert server.project()["freeze"]["frame"] == good


def test_nullable_keys_can_be_deleted_by_a_patch(server):
    # A merge patch with null deletes the key, so none of these may be needed to be there
    assert server.post("/api/project", {"picked": None}).status == 200
    assert "picked" not in server.project()

    ask = server.post("/api/style/ask", {"text": "vaporwave"})
    assert ask.status == 200
    assert server.post("/api/project", {"style_request": None}).status == 200
    assert "style_request" not in server.project()
    assert server.post("/api/project", {}).status == 200  # and the page still loads

    # A request the page sees before Claude has answered, with no answered_at in the file at all
    request = {"text": "x", "at": "2026-10-26T05:00:00.000+00:00"}
    outside = server.project()
    outside["style_request"] = request
    project_file.save(server.dir, outside)
    assert server.get("/api/project").json()["style_request"] == request

    # There is no music: a patch that sets a track, its beats or the trend's freeze time is refused
    for patch in ({"audio": {"track": "song.mp3"}}, {"audio": {"onsets": [1.0]}}, {"timing": {"freeze_at": 2}}):
        assert server.post("/api/project", patch).status == 400, patch
    assert server.project()["audio"] == {"original": True} and "freeze_at" not in server.project()["timing"]

    # The Sound switch edits audio.original; anything but a bool is refused
    assert server.project()["audio"]["original"] is True
    assert server.post("/api/project", {"audio": {"original": False}}).status == 200
    assert server.project()["audio"]["original"] is False
    assert server.post("/api/project", {"audio": {"original": "off"}}).status == 400
    assert server.project()["audio"]["original"] is False


def test_a_style_request_can_be_cancelled_and_asked_again(server):
    # Claude's session may not be waiting. The page's Cancel is a patch with a null, and "Ask again" a new request
    assert server.post("/api/style/ask", {"text": "vaporwave"}).status == 200
    assert server.get("/api/project").json()["style_request"]["text"] == "vaporwave"

    cancelled = server.post("/api/project", {"style_request": None})

    assert cancelled.status == 200 and "style_request" not in cancelled.json()["project"]
    assert "style_request" not in server.get("/api/project").json()
    assert "style_request" not in server.project()

    first = server.post("/api/style/ask", {"text": "chrome letters"}).json()["style_request"]
    time.sleep(0.05)
    second = server.post("/api/style/ask", {"text": "neon jungle"}).json()["style_request"]  # replaces a pending one
    assert second["text"] == "neon jungle" and second["answered_at"] is None and second["at"] > first["at"]
    assert server.get("/api/project").json()["style_request"] == second
    assert wait.wait_event(server.dir, since=first["at"], poll=0.05, timeout=5) == ("STYLE", second["at"])


def test_the_server_keeps_no_copy_of_the_project(server):
    # Claude edits project.json while the studio is open, for example to answer "Ask Claude"
    before_card = server.get("/api/card.png?w=300").body
    outside = server.project()
    outside["style"]["bg"] = "night"
    outside["style_source"] = "custom"
    outside["style_request"] = {
        "text": "x", "image": None, "at": "2026-10-26T05:00:00.000+00:00",
        "answered_at": "2026-10-26T05:00:09.000+00:00",
    }
    project_file.save(server.dir, outside)

    assert server.get("/api/project").json()["style"]["bg"] == "night"
    assert server.get("/api/card.png?w=300").body != before_card  # the cache key has the content

    # A patch from the page keeps the outside change it did not touch
    reply = server.post("/api/project", {"copy": {"name": "Pauli"}})
    assert reply.status == 200
    saved = server.project()
    assert saved["copy"]["name"] == "Pauli"
    assert saved["style"]["bg"] == "night" and saved["style_source"] == "custom"
    assert saved["style_request"]["answered_at"] == "2026-10-26T05:00:09.000+00:00"
    assert reply.json()["project"]["style"]["bg"] == "night"

    # and so does the second one, after another outside change
    outside = server.project()
    outside["timing"]["tail"] = 2.0
    project_file.save(server.dir, outside)
    server.post("/api/project", {"copy": {"name": "Robin"}})
    assert server.project()["timing"]["tail"] == 2.0 and server.project()["copy"]["name"] == "Robin"


def test_patches_from_many_requests_do_not_lose_each_other(server):
    def patch_many(key, values):
        for value in values:
            assert server.post("/api/project", {"copy": {"notes": {key: value}}}).status == 200

    threads = [
        threading.Thread(target=patch_many, args=("tl", [f"top {n}" for n in range(12)])),
        threading.Thread(target=patch_many, args=("bl", [f"But {n}" for n in range(12)])),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    notes = server.project()["copy"]["notes"]
    assert notes["tl"] == "top 11" and notes["bl"] == "But 11"


def test_a_patch_and_a_renders_record_take_turns(server):
    # One lock (project.UPDATE_LOCK) is held by the server for a patch, from the read to the save, and by
    # render for the record of its status, so neither can overwrite the other
    recording = threading.Thread(
        target=render._record, args=(server.dir, [server.dir / "renders" / "a.mp4"], 2.0, 5.0)
    )
    patching = threading.Thread(target=lambda: server.post("/api/project", {"timing": {"hold": 4}}))
    with project_file.UPDATE_LOCK:  # someone is saving project.json
        recording.start()
        patching.start()
        time.sleep(0.4)
        assert recording.is_alive() and patching.is_alive()  # both wait for their turn
    recording.join(30)
    patching.join(30)
    saved = server.project()
    assert saved["status"]["outputs"] == ["renders/a.mp4"] and saved["timing"]["hold"] == 4  # and neither is lost

    def patches():
        for number in range(30):
            assert server.post("/api/project", {"copy": {"notes": {"tl": f"top {number}"}}}).status == 200

    def records():
        for number in range(30):
            render._record(server.dir, [server.dir / "renders" / f"clip-{number}.mp4"], 2.0, 5.0)

    threads = [threading.Thread(target=patches), threading.Thread(target=records)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    saved = server.project()
    assert saved["copy"]["notes"]["tl"] == "top 29"
    assert saved["status"]["outputs"] == ["renders/clip-29.mp4"] and saved["status"]["rendered_at"]


def test_options_have_what_the_page_builds_its_controls_from(server):
    from mclib import config
    from mclib.styles import recipes

    options = server.get("/api/options").json()

    assert options["options"]["bg"] == list(recipes.OPTIONS["bg"])
    assert options["labels"]["notes"]["comic"] == "Comic boxes"
    assert options["palettes"]["Magenta"] == recipes.PALETTES["Magenta"] and len(options["palettes"]) == 7
    assert options["roles"] == list(recipes.ROLES)
    assert options["soft_limits"] == project_file.SOFT_LIMITS
    assert options["soft_limits"] == {"tl": 30, "tr": 24, "bl": 34, "br": 30, "name": 16, "bubble": 14}  # no limit for a title
    assert options["timing"]["defaults"] == config.TIMING_DEFAULTS and "freeze_at" not in options["timing"]["defaults"]
    assert "freeze_at" not in options["timing"]["ranges"]
    assert options["timing"]["ranges"]["hold"] == [1.0, 6.0]


# --- the card, the frames and the layout -------------------------------------------------------


def test_card_png(server):
    reply = server.get("/api/card.png")
    assert reply.status == 200 and reply.headers["content-type"] == "image/png"
    assert picture(reply).size == (1080, 608)  # the card, at the plan: the clip's own frame, nothing around it

    small = server.get("/api/card.png?style=comic-pop&w=270")
    assert small.status == 200 and picture(small).size == (270, 152)
    assert small.body != server.get("/api/card.png?w=270").body  # another style is another card

    assert picture(server.get("/api/card.png?w=5000")).size == (1080, 608)  # never blown up


def test_a_saved_style_is_a_thumbnail_too(server):
    saved = server.post("/api/styles", {"name": "Night Owl"}).json()

    thumb = server.get(f"/api/card.png?style={saved['id']}&w=200")

    assert thumb.status == 200 and picture(thumb).size == (200, 113)


def test_card_is_cached_and_only_what_changed_is_drawn_again(server, monkeypatch):
    from mclib import card

    drawn = []
    real_render_card = card.render_card
    stages = []
    real_compose = engine.compose_stage
    contexts = []
    real_context = engine.matte_context

    monkeypatch.setattr(card, "render_card", lambda *a, **k: drawn.append(1) or real_render_card(*a, **k))
    monkeypatch.setattr(engine, "compose_stage", lambda *a, **k: stages.append(1) or real_compose(*a, **k))
    monkeypatch.setattr(engine, "matte_context", lambda *a, **k: contexts.append(1) or real_context(*a, **k))

    first = server.get("/api/card.png?w=400")
    again = server.get("/api/card.png?w=400")
    assert first.body == again.body and len(drawn) == 1  # the same inputs, the same bytes, no drawing

    # An edit of one note draws the card again, but not the stage and not what the matte alone gives
    assert server.post("/api/project", {"copy": {"notes": {"bl": "But very sleepy"}}}).status == 200
    edited = server.get("/api/card.png?w=400")
    assert edited.body != first.body and len(drawn) == 2
    assert len(stages) == 1 and len(contexts) == 1

    # A change that is not on the card (the timing) is not drawn again
    assert server.post("/api/project", {"timing": {"hold": 4}}).status == 200
    assert server.get("/api/card.png?w=400").body == edited.body and len(drawn) == 2


def test_a_title_left_in_an_older_file_is_not_sent_and_is_not_on_the_card(server, monkeypatch):
    # The edit has no title. An older project.json that still has one (copy.hook, and the hook of each version)
    # is read without it: the page is never sent it, and the card, which is keyed on the copy, is the same card
    from mclib import card

    drawn = []
    real_render_card = card.render_card
    monkeypatch.setattr(card, "render_card", lambda *a, **k: drawn.append(1) or real_render_card(*a, **k))
    first = server.get("/api/card.png?w=400")
    assert len(drawn) == 1

    path = server.dir / "project.json"
    older = json.loads(path.read_text(encoding="utf-8"))
    for block in (older["copy"], *older["versions"].values()):
        block["hook"] = ["this is your sign to", "post it"]
    path.write_text(json.dumps(older), encoding="utf-8")

    project = server.get("/api/project").json()
    assert "hook" not in project["copy"] and all("hook" not in version for version in project["versions"].values())
    assert server.get("/api/card.png?w=400").body == first.body and len(drawn) == 1  # no new drawing

    # and an edit through the page saves the file without it
    assert server.post("/api/project", {"copy": {"name": "Pauli"}}).status == 200
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "hook" not in saved["copy"] and all("hook" not in version for version in saved["versions"].values())


def test_a_notice_for_the_page_comes_with_the_project_and_is_not_on_the_card(server, monkeypatch):
    # init leaves one in the file when it resets the pins (pipeline.POSITIONS_RESET): the page shows it as a toast
    # and takes it away with a null, as it does any key it is finished with. It is not on the card
    from mclib import card

    sentence = "Your note positions were reset because the clip's shape changed."
    drawn = []
    real_render_card = card.render_card
    monkeypatch.setattr(card, "render_card", lambda *a, **k: drawn.append(1) or real_render_card(*a, **k))
    first = server.get("/api/card.png?w=400")

    project_file.update(server.dir, {"notice": sentence})  # (as init writes it)
    assert server.get("/api/project").json()["notice"] == sentence
    assert server.get("/api/card.png?w=400").body == first.body and len(drawn) == 1

    answer = server.post("/api/project", {"notice": None})
    assert answer.status == 200 and "notice" not in answer.json()["project"]
    assert "notice" not in server.project()
    assert server.post("/api/project", {"notice": 7}).status == 400  # text or nothing
    assert server.get("/api/card.png?w=400").body == first.body and len(drawn) == 1


def test_frame_jpeg(server):
    whole = server.get("/api/frame/60")
    assert whole.status == 200 and whole.headers["content-type"] == "image/jpeg"
    assert whole.body == (server.dir / "frames" / "000060.jpg").read_bytes()

    thumb = server.get("/api/frame/60?w=120")
    assert thumb.status == 200 and picture(thumb).size == (120, 68)
    assert picture(server.get("/api/frame/0?w=5000")).size == (1080, 608)  # never blown up

    assert server.get("/api/frame/61").status == 404  # the fake project has two frames only
    again = server.get("/api/frame/60?w=120", headers={"If-None-Match": thumb.headers["etag"]})
    assert again.status == 304 and again.body == b""


def test_frames_lists_what_can_be_frozen_on(server, fake_frames):
    # The fake project has the picture of frames 0 and 60 and the cutout of frame 60 only
    assert server.get("/api/frames").json() == {"ranges": [[60, 60]]}

    for frame in fake_frames.glob("*.jpg"):  # every picture, but still one cutout
        (server.dir / "frames" / frame.name).write_bytes(frame.read_bytes())
    assert server.get("/api/frames").json() == {"ranges": [[60, 60]]}

    cutout = (server.dir / "matte" / "000060.png").read_bytes()
    for frame in (10, 11, 12, 13, 61, 62, 200):  # (a cutout without a picture is not a frame either)
        (server.dir / "matte" / f"{frame:06d}.png").write_bytes(cutout)
    (server.dir / "matte" / "notes.txt").write_text("not a frame", encoding="utf-8")
    assert server.get("/api/frames").json() == {"ranges": [[10, 13], [60, 62]]}


def test_layout(server):
    reply = server.get("/api/layout")

    assert reply.status == 200
    answer = reply.json()
    assert answer["stage"] == [0, 0, 1080, 608]  # in px of the plan, which is the card's own size
    assert {"tl", "tr", "bl", "br"} <= set(answer["elements"])
    for key, (x, y, w, h) in answer["elements"].items():  # in the same px: inside the stage
        assert 0 <= x and x + w <= 1080 and 0 <= y and y + h <= 608, key
    assert answer["flags"] == {}  # The sample copy fits the fake matte's person cleanly


def test_layout_flags_a_note_its_font_cannot_draw_and_the_patch_says_why(server):
    # The studio draws its amber outline from these flags. A Cyrillic letter in a Marker note is cut (the
    # font draws a gap for it), and the answer to the edit says which characters, and in which font
    ya = chr(0x42F)
    reply = server.post("/api/project", {"copy": {"notes": {"tl": f"{ya} educator"}}})

    assert reply.status == 200
    assert f"copy.notes.tl has characters the Marker font can't draw: {ya}." in reply.json()["warnings"]
    assert server.get("/api/layout").json()["flags"].get("tl") == "cut"

    # a note in a font that has it is not flagged (Caveat, the handwritten strips, draws Cyrillic)
    hand = dict(server.project()["style"], notes="hand")
    reply = server.post("/api/project", {"style": hand})
    assert reply.status == 200 and not [line for line in reply.json()["warnings"] if "can't draw" in line]
    assert "tl" not in server.get("/api/layout").json()["flags"]


def test_dragging_pins_a_note_in_stage_pixels(server):
    # positions are the top-left corner in px of the plan, and the layout answers in the same px
    assert server.post("/api/project", {"positions": {"bl": [300, 250]}}).status == 200

    x, y, w, h = server.get("/api/layout").json()["elements"]["bl"]

    assert (x, y) == (300, 250)

    # and a reset (a null for each) puts it back to where the layout wants it
    nulls = {key: None for key in ("tl", "tr", "bl", "br", "burst")}
    assert server.post("/api/project", {"positions": nulls}).status == 200
    assert server.get("/api/layout").json()["elements"]["bl"][:2] != [300, 250]
    assert server.project()["positions"] == {}


def test_layout_flags_what_does_not_fit_and_leaves_out_the_hud(server):
    # A note too long for three lines at the smallest size is cut, and flagged as such
    assert server.post("/api/project", {"copy": {"notes": {"tr": "word " * 60}}}).status == 200
    flags = server.get("/api/layout").json()["flags"]
    assert flags.get("tr") == "cut"

    # Two notes pinned on top of each other clash
    assert server.post("/api/project", {
        "copy": {"notes": {"tr": "@your_handle"}},
        "positions": {"tl": [300, 100], "bl": [310, 110]},
    }).status == 200
    flags = server.get("/api/layout").json()["flags"]
    assert flags.get("tl") == "clash" and flags.get("bl") == "clash"
    assert set(flags.values()) <= {"cut", "clash"}

    # The HUD's two labels are not draggable, so they get no handle
    hud = dict(server.project()["style"], hero="hud")
    assert server.post("/api/project", {"style": hud, "positions": {"tl": None, "bl": None}}).status == 200
    elements = server.get("/api/layout").json()["elements"]
    assert not {"hud_top", "hud_bottom"} & set(elements) and {"tl", "bl", "br"} <= set(elements)

    # The stats panel takes the key of the tr note, and the burst has a key of its own
    stats = dict(hud, hero="stats")
    assert server.post("/api/project", {"style": stats}).status == 200
    assert "tr" in server.get("/api/layout").json()["elements"]
    burst = dict(hud, hero="burst")
    assert server.post("/api/project", {"style": burst}).status == 200
    assert "burst" in server.get("/api/layout").json()["elements"]


# --- styles and palettes -----------------------------------------------------------------------


def test_styles_list_and_save(server):
    listed = server.get("/api/styles").json()
    assert [style["id"] for style in listed] == ["bedroom-pop", "comic-pop", "scrapbook", "player-select"]
    assert all(style["builtin"] for style in listed)

    saved = server.post("/api/styles", {"name": "Night Owl"})
    assert saved.status == 200
    entry = saved.json()
    assert entry["id"] == "u-night-owl" and entry["name"] == "Night Owl" and entry["builtin"] is False
    assert entry["recipe"]["bg"] == server.project()["style"]["bg"]

    after = server.get("/api/styles").json()
    assert [style["id"] for style in after][-1] == "u-night-owl" and len(after) == 5


def test_palette_from_upload(server):
    before = server.project()

    reply = server.post("/api/palette", raw=blocks_png(), headers={"Content-Type": "image/png"})

    assert reply.status == 200
    answer = reply.json()
    assert set(answer["palette"]) == {"dark", "light", "tex", "edge", "ink"}
    assert all(re.fullmatch(r"#[0-9a-f]{6}", color) for color in answer["palette"].values())
    assert answer["palette"]["dark"] == "#101010" and answer["palette"]["light"] == "#f0f0f0"
    expected = text_contrast({**before["style"], "palette": answer["palette"]})
    assert answer["contrast"] == pytest.approx(expected)
    assert server.project() == before  # it does not save


def test_palette_from_an_image_with_nothing_to_see_is_a_plain_400(server):
    clear = png_bytes(Image.new("RGBA", (40, 40), (255, 0, 0, 0)))

    reply = server.post("/api/palette", raw=clear)

    assert reply.status == 400 and reply.json() == {"error": "That image has no visible pixels."}

    notimage = server.post("/api/palette", raw=b"this is not a picture")
    assert notimage.status == 400 and "image" in notimage.json()["error"]
    assert server.post("/api/palette", raw=b"").status == 400


def test_style_ask_sets_request(server):
    started = datetime.now(timezone.utc)

    reply = server.post("/api/style/ask", {"text": "vaporwave", "image": None})

    assert reply.status == 200
    request = reply.json()["style_request"]
    assert request["text"] == "vaporwave" and request["image"] is None
    assert request["answered_at"] is None and "answered_at" in server.project()["style_request"]
    assert STAMP.fullmatch(request["at"]) and wait.parse_time(request["at"]) >= started
    assert server.project()["style_request"] == request

    # With a reference image the path comes back as it was given, and a second request replaces it
    again = server.post("/api/style/ask", {"text": "chrome letters", "image": "refs/look.png"}).json()
    assert again["style_request"]["image"] == "refs/look.png"
    assert server.project()["style_request"]["text"] == "chrome letters"


def test_the_stamps_are_utc_to_the_millisecond_and_wake_the_waiter(server, monkeypatch):
    before = datetime.now(timezone.utc)
    time.sleep(0.01)

    # A waiter that is already waiting is woken by the style request, at once
    woke = []
    waiter = threading.Thread(
        target=lambda: woke.append(wait.wait_event(server.dir, poll=0.05, timeout=20))
    )
    waiter.start()
    time.sleep(0.3)
    stamp = server.post("/api/style/ask", {"text": "vaporwave"}).json()["style_request"]["at"]
    waiter.join(20)
    assert woke == [("STYLE", stamp)]
    assert STAMP.fullmatch(stamp) and wait.parse_time(stamp) > before

    # and so is one that starts a moment later and counts from before the click
    assert wait.wait_event(server.dir, since=before.isoformat(), poll=0.05, timeout=5) == ("STYLE", stamp)

    # A render's Done is stamped the same way
    monkeypatch.setattr(render, "render", lambda folder, **kwargs: make_result(folder))
    started = datetime.now(timezone.utc)
    time.sleep(0.01)
    assert server.post("/api/render").status == 200
    done = server.project()["status"]["done_at"]
    assert STAMP.fullmatch(done) and wait.parse_time(done) > started
    assert wait.wait_event(server.dir, since=stamp, poll=0.05, timeout=5) == ("DONE", done)


def test_posted_names_and_bytes_land_in_the_project_folder(server):
    sent = server.post(
        "/api/style/reference", raw=b"pretend this is a picture",
        headers={"X-Filename": "..%2F..%2Fmy%20look%C3%A9.png"},  # ../../my look + e acute
    )

    assert sent.status == 200
    path = Path(sent.json()["path"])
    assert path.parent == (server.dir / "refs") and path.name == "my look" + chr(0xE9) + ".png"
    assert path.read_bytes() == b"pretend this is a picture"  # whatever the name tried
    assert not any(p.suffix == ".part" for p in (server.dir / "refs").iterdir())

    for name in ("C:%5CWindows%5Cevil.png", "..", "", "con", "a%3A%2A%3F.png"):
        reply = server.post("/api/style/reference", raw=b"x", headers={"X-Filename": name})
        assert reply.status == 200, name
        landed = Path(reply.json()["path"])
        assert landed.parent == server.dir / "refs" and landed.is_file(), name
        assert landed.name.upper() not in {"CON", "..", ""} and not re.search(r'[:*?"<>|]', landed.name)


def test_a_file_over_the_limit_is_refused_and_nothing_is_kept(server, monkeypatch):
    monkeypatch.setattr(studio_server, "MAX_UPLOAD", 2 * 1024 * 1024)

    refused = server.post("/api/style/reference", raw=b"x" * (3 * 1024 * 1024), headers={"X-Filename": "huge.png"})

    assert refused.status == 400 and refused.json() == {"error": "That is too big: the limit is 2 MB."}
    refs = server.dir / "refs"
    assert not refs.exists() or not any(refs.iterdir())  # neither the file nor a temporary one
    chunked = server.request("POST", "/api/style/reference", body=b"x", headers={"Transfer-Encoding": "chunked", "X-Filename": "a.png"})
    assert chunked.status == 400 and "length" in chunked.json()["error"]


def test_uploads_wait_out_a_file_lock(server, monkeypatch):
    real, refusals, pauses = os.replace, [], []

    def held_for_a_moment(source, destination):
        if len(refusals) < 2:
            refusals.append(destination)
            raise PermissionError(5, "Access is denied")
        return real(source, destination)

    monkeypatch.setattr(os, "replace", held_for_a_moment)
    monkeypatch.setattr(time, "sleep", pauses.append)

    reply = server.post("/api/style/reference", raw=b"look", headers={"X-Filename": "look.png"})

    assert reply.status == 200 and len(refusals) == 2 and pauses[:2] == [0.1, 0.1]
    assert (server.dir / "refs" / "look.png").read_bytes() == b"look"


def test_there_is_no_route_to_upload_music(server):
    reply = server.post("/api/audio", raw=b"RIFF", headers={"X-Filename": "beat.wav", "Content-Type": "audio/wav"})

    assert reply.status in (404, 405)
    assert not (server.dir / "audio").exists()
    assert "track" not in server.project()["audio"]


# --- the cutout, the format and the renders -----------------------------------------------------


def test_matte_refine_reruns_the_frames_around_the_freeze(server, fake_frames):
    for frame in fake_frames.glob("*.jpg"):  # the fake project has two frames; give it them all
        (server.dir / "frames" / frame.name).write_bytes(frame.read_bytes())
    with Image.open(server.dir / "matte" / "000060.png") as image:
        area = analyze.person_boxes(from_image(image.convert("L")))["area"]

    reply = server.post("/api/matte/refine", {"detail": "high"})

    assert reply.status == 200
    assert reply.json()["project"]["matte"]["detail"] == "high"
    assert server.project()["matte"]["detail"] == "high"
    (ratio, start, end), = server.runner.calls
    assert (start, end) == (48, 61)  # [freeze - 12, freeze + 1)
    assert ratio == pytest.approx(matte.downsample_for(1080, 608, "high", area), abs=1e-3)
    assert (server.dir / "matte" / "000048.png").is_file()

    near = server.project()
    near["freeze"]["frame"] = 3  # near the start, the run begins at frame 0
    project_file.save(server.dir, near)
    assert server.post("/api/matte/refine", {"detail": "max"}).status == 200
    assert server.runner.calls[-1][1:] == (0, 4)


def test_matte_refine_without_the_model_says_so(server, monkeypatch):
    server.httpd.studio.matte_runner = None  # the real runner, which needs the model

    def missing(*args, **kwargs):
        raise matte.ModelMissing("The matte model is missing: put it here.")

    monkeypatch.setattr(matte, "ensure_model", missing)

    reply = server.post("/api/matte/refine", {"detail": "high"})

    assert reply.status == 400 and reply.json() == {"error": "The matte model is missing: put it here."}
    assert server.project()["matte"]["detail"] == "auto"  # nothing changed
    # and the lock is let go: the next call is not told that a render is running
    assert server.post("/api/matte/refine", {"detail": "bad"}).status == 400


def test_preview_and_render(server):
    first = server.project()["status"]

    preview = server.post("/api/preview")

    assert preview.status == 200
    answer = preview.json()
    assert answer["url"] == "/renders/preview.mp4" and "warning" not in answer
    video = server.get(answer["url"])
    assert video.status == 200 and video.headers["content-type"] == "video/mp4" and len(video.body) > 1000
    assert server.project()["status"] == first  # a preview leaves the project alone

    started = datetime.now(timezone.utc)
    time.sleep(0.01)
    final = server.post("/api/render")

    assert final.status == 200
    body = final.json()
    names = [Path(output).name for output in body["outputs"]]
    assert names == ["synthetic-main-character.mp4", "synthetic-cover.png"] and "warning" not in body
    saved = server.project()["status"]
    assert saved["outputs"] == body["outputs"] and saved["rendered_at"]
    assert STAMP.fullmatch(saved["done_at"]) and wait.parse_time(saved["done_at"]) > started
    cover = server.get("/" + body["outputs"][1])
    assert cover.status == 200 and picture(cover).size == (1920, 1080)  # the card at the size of the video


def test_render_conflict(server, monkeypatch):
    slow = Gate(server.dir)
    monkeypatch.setattr(render, "render", slow)
    first, got = hold(server, "/api/render")
    assert slow.started.wait(30)

    # Everything that rewrites the project's files waits its turn, with the same sentence
    for path, data in (("/api/render", None), ("/api/preview", None), ("/api/matte/refine", {"detail": "high"})):
        refused = server.post(path, data)
        assert refused.status == 409 and refused.json() == BUSY, path
    assert server.runner.calls == []  # the refused calls did nothing

    # Reading the project and drawing the card go on while a render runs
    assert server.get("/api/project").status == 200 and server.get("/api/card.png?w=100").status == 200
    assert server.post("/api/project", {"timing": {"hold": 4}}).status == 200

    slow.release.set()
    first.join(30)
    assert got[0].status == 200 and len(slow.calls) == 1

    # The lock is let go afterwards, and also after a render that failed
    monkeypatch.setattr(render, "render", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Boom.")))
    failed = server.post("/api/render")
    assert failed.status == 400 and failed.json() == {"error": "Boom."}
    monkeypatch.setattr(render, "render", lambda folder, **k: make_result(folder, preview=k.get("preview", False)))
    assert server.post("/api/preview").status == 200


def test_a_cutout_job_holds_the_same_lock_and_the_reads_that_it_would_break(server, fake_frames):
    for frame in fake_frames.glob("*.jpg"):
        (server.dir / "frames" / frame.name).write_bytes(frame.read_bytes())
    gate = Gate(server.dir, inner=server.runner)
    server.httpd.studio.matte_runner = gate
    first, got = hold(server, "/api/matte/refine", {"detail": "high"})
    assert gate.started.wait(30)

    for path, data in (("/api/render", None), ("/api/preview", None), ("/api/matte/refine", {"detail": "max"})):
        refused = server.post(path, data)
        assert refused.status == 409 and refused.json() == BUSY, path

    # The frames and the mattes are being rewritten, so they are not read now
    for path in ("/api/card.png", "/api/layout", "/api/frame/60", "/api/frames"):
        reply = server.get(path)
        assert reply.status == 409 and "being rebuilt" in reply.json()["error"], path
    assert server.get("/api/project").status == 200

    gate.release.set()
    first.join(30)
    assert got[0].status == 200
    assert server.get("/api/card.png?w=100").status == 200


# --- files: the page, the fonts and the renders -------------------------------------------------


@pytest.fixture
def studio_dir(tmp_path_factory, monkeypatch):
    """A studio folder of its own (the page does not exist yet in the commit that adds the server,
    and these tests are about the serving), with a secret one folder up."""
    root = tmp_path_factory.mktemp("studio-root")
    folder = root / "studio"
    folder.mkdir()
    (folder / "index.html").write_text("<!doctype html><title>Studio</title>", encoding="utf-8")
    (folder / "app.js").write_text("console.log('hi');", encoding="utf-8")
    (folder / "style.css").write_text("body { margin: 0 }", encoding="utf-8")
    (root / "project.json").write_text("SECRET", encoding="utf-8")
    monkeypatch.setattr(studio_server, "STUDIO_DIR", folder)
    return folder


def test_the_page_and_its_files_are_served(server, studio_dir):
    page = server.get("/")
    assert page.status == 200 and page.headers["content-type"] == "text/html; charset=utf-8"
    assert b"<title>Studio</title>" in page.body

    assert server.get("/studio/app.js").headers["content-type"] == "text/javascript; charset=utf-8"
    assert server.get("/studio/style.css").headers["content-type"] == "text/css; charset=utf-8"
    assert server.get("/studio/nothing.js").status == 404

    # A browser that has the file asks only if it changed
    tag = page.headers["etag"]
    assert server.get("/", headers={"If-None-Match": tag}).status == 304

    # The fonts are the skill's own, including the ones with brackets in their names
    for name in ("Caveat%5Bwght%5D.ttf", "Caveat[wght].ttf", "Bangers-Regular.ttf"):
        font = server.get(f"/fonts/{name}")
        assert font.status == 200 and font.headers["content-type"] == "font/ttf", name
        assert len(font.body) > 10_000
    assert server.get("/fonts/Nunito%5Bwght%5D.ttf").status == 404  # the title's font went with the title


def test_static_traversal_blocked(server, studio_dir):
    (server.dir / "renders" / "_work").mkdir(parents=True)  # a render's own folder: not a file
    (server.dir / "renders" / "clip.mp4").write_bytes(b"video")
    paths = [
        "/studio/..%2Fproject.json", "/studio/%2e%2e%2fproject.json", "/studio/..%5Cproject.json",
        "/studio/../project.json", "/studio/%2e%2e/project.json", "/studio/..", "/studio/",
        "/studio/index.html%00.png", "/studio/index.html::$DATA", "/studio/index.html.", "/studio/INDEX.HTML",
        "/studio/nul", "/studio/sub/../index.html",
        "/fonts/..%2Fproject.json", "/fonts/C:%5CWindows%5Cwin.ini", "/fonts/%252e%252e%252fproject.json",
        "/fonts/", "/fonts/C:/Windows/win.ini", "/fonts/Caveat%5Bwght%5D.ttf%00",
        "/renders/..%2Fproject.json", "/renders/..%5Cproject.json", "/renders/_work", "/renders/%2e%2e%2f",
        "/studio%2Findex.html", "//studio/index.html/", "/index.html",
    ]
    for path in paths:
        reply = server.get(path)
        assert reply.status == 404, path
        assert b"SECRET" not in reply.body and reply.headers["content-type"].startswith("application/json")
    assert server.get("/studio/..%2Fproject.json").status == 404  # the brief's own case


def test_renders_are_served_with_ranges(server):
    renders = server.dir / "renders"
    renders.mkdir()
    data = bytes(range(256)) * 4  # 1024 bytes
    (renders / "clip.mp4").write_bytes(data)
    (renders / "sp ace.mp4").write_bytes(b"spaced")

    whole = server.get("/renders/clip.mp4")
    assert whole.status == 200 and whole.body == data
    assert whole.headers["accept-ranges"] == "bytes" and whole.headers["content-length"] == "1024"
    assert whole.headers["content-type"] == "video/mp4"
    assert server.get("/renders/sp%20ace.mp4").body == b"spaced"

    def ranged(spec, **headers):
        return server.get("/renders/clip.mp4", headers={"Range": spec, **headers})

    first = ranged("bytes=0-9")
    assert first.status == 206 and first.body == data[:10]
    assert first.headers["content-range"] == "bytes 0-9/1024" and first.headers["content-length"] == "10"
    assert ranged("bytes=1000-").body == data[1000:] and ranged("bytes=1000-").status == 206
    assert ranged("bytes=-5").body == data[-5:]
    assert ranged("bytes=1000-9999").headers["content-range"] == "bytes 1000-1023/1024"
    assert ranged("bytes=0-0").body == data[:1]

    assert ranged("bytes=2000-").status == 416
    assert ranged("bytes=2000-").headers["content-range"] == "bytes */1024"
    assert ranged("bytes=-0").status == 416
    for ignored in ("bytes=5-2", "bytes=0-1,5-9", "chunks=1-2", "bytes=", "bytes=a-b"):
        assert ranged(ignored).status == 200 and ranged(ignored).body == data, ignored

    # A resumed download whose validator is out of date starts again from the top
    assert ranged("bytes=0-9", **{"If-Range": '"stale"'}).status == 200
    assert ranged("bytes=0-9", **{"If-Range": whole.headers["etag"]}).status == 206
    assert server.get("/renders/clip.mp4", headers={"If-None-Match": whole.headers["etag"]}).status == 304

    head = server.request("HEAD", "/renders/clip.mp4")
    assert head.status == 200 and head.body == b"" and head.headers["content-length"] == "1024"
    assert server.request("HEAD", "/renders/missing.mp4").status == 404


def test_a_file_that_changes_while_it_is_sent_is_not_sent_mixed(server, monkeypatch):
    renders = server.dir / "renders"
    renders.mkdir()
    path = renders / "clip.mp4"
    path.write_bytes(b"A" * 600_000)
    monkeypatch.setattr(studio_server, "CHUNK", 100_000)
    real_write, writes = studio_server._Handler._write_body, []

    def rewrite_after_the_first_chunk(self, chunk):
        writes.append(len(chunk))
        if len(writes) == 1:
            path.write_bytes(b"B" * 700_000)  # (the server has the file open for one chunk at a time)
        return real_write(self, chunk)

    monkeypatch.setattr(studio_server._Handler, "_write_body", rewrite_after_the_first_chunk)

    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=60)
    try:
        connection.request("GET", "/renders/clip.mp4")
        answer = connection.getresponse()
        assert answer.status == 200 and answer.getheader("Content-Length") == "600000"
        with pytest.raises(http.client.IncompleteRead) as cut_short:
            answer.read()
    finally:
        connection.close()

    assert b"B" not in cut_short.value.partial and len(cut_short.value.partial) < 600_000


def test_only_this_computer_may_talk_to_the_server(server):
    # A page on another site can make the browser send requests to localhost: those are refused
    foreign_host = server.get("/api/project", headers={"Host": "evil.example:80"})
    assert foreign_host.status == 403

    here, named = f"127.0.0.1:{server.port}", f"localhost:{server.port}"
    assert server.request("GET", "/api/project", headers={"Host": named}).status == 200  # the name of this computer

    def change(hold, host, origin):
        return server.post("/api/project", {"timing": {"hold": hold}}, headers={"Host": host, "Origin": origin})

    for origin in (
        "https://evil.example", "http://evil.example:8340", "null",
        "http://localhost:3000", "http://127.0.0.1:9999",  # a dev server on another port of this computer
        f"https://{here}",  # not the scheme of this server
        f"http://{named}",  # the other name of this computer is another origin, for a browser
    ):
        assert change(5, here, origin).status == 403, origin
    assert server.project()["timing"]["hold"] == 3.0  # not changed

    assert change(5, here, f"http://{here}").status == 200  # its own page
    assert change(4, named, f"http://{named}").status == 200  # and by the other name
    assert server.post("/api/project", {"timing": {"hold": 3}}).status == 200  # no Origin: not a browser


def test_a_cross_site_preflight_is_refused(server):
    # A page that sends JSON to another site has to ask first (OPTIONS), and this server answers no
    reply = server.request("OPTIONS", "/api/project", headers={
        "Origin": "http://evil.example", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })

    assert reply.status == 501 and "access-control-allow-origin" not in reply.headers


def test_json_routes_need_json_and_upload_routes_keep_their_own_types(server):
    # Without it a web page could post "text/plain" to localhost with no preflight at all
    for path in ("/api/project", "/api/styles", "/api/style/ask", "/api/matte/refine",
                 "/api/preview", "/api/render"):
        for kind in (None, "text/plain", "text/plain;charset=UTF-8", "application/x-www-form-urlencoded",
                     "multipart/form-data; boundary=x", "application/jsonp"):
            reply = server.request("POST", path, body=b"{}", headers={"Content-Type": kind} if kind else {})
            assert reply.status == 415, (path, kind)
            assert reply.json() == {"error": "Send the request as application/json."}
    assert server.runner.calls == [] and server.project().get("style_request") is None  # none of them did anything

    for kind in ("application/json", "Application/JSON", "application/json; charset=utf-8"):
        assert server.request("POST", "/api/project", body=b"{}", headers={"Content-Type": kind}).status == 200

    # The raw routes take pictures, with their own content types (or none)
    for kind in ("image/png", "application/octet-stream", None):
        sent = server.post("/api/palette", raw=blocks_png(), headers={"Content-Type": kind} if kind else {})
        assert sent.status == 200, kind
    look = server.post("/api/style/reference", raw=b"look", headers={"Content-Type": "image/png", "X-Filename": "a.png"})
    assert look.status == 200


# --- starting the server ----------------------------------------------------------------------


def test_serve_takes_the_first_free_port(fake_project, monkeypatch):
    import socket

    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    first = taken.getsockname()[1]
    monkeypatch.setattr(studio_server, "FIRST_PORT", first)
    try:
        httpd = studio_server.bind(fake_project.dir)
        try:
            assert httpd.server_address[0] == "127.0.0.1" and httpd.server_address[1] > first
        finally:
            httpd.server_close()
        with pytest.raises(OSError):  # a port that was asked for by name is not looked past
            studio_server.bind(fake_project.dir, port=first)
    finally:
        taken.close()


def test_make_server_needs_a_project(tmp_path):
    with pytest.raises(FileNotFoundError, match="There is no project in"):
        studio_server.make_server(tmp_path)


def test_server_is_bound_to_this_computer_only(server):
    assert server.httpd.server_address[0] == "127.0.0.1"
