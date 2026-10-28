"""Tests for the studio page (studio/index.html, app.js and style.css).

A page is checked in a browser, and the Browser pane check of the plan does that. What a test can
pin is what the page cannot do without: the ids the plan fixes, the copy in US English, a script
that parses, files that the server really serves, no address that is not this computer's, and no
call to a route that is not there.
"""

import json
import re
import shutil
import subprocess
from urllib.parse import unquote

import pytest

from mclib import server as studio_server
from mclib.config import FONTS_DIR, STUDIO_DIR

# The ids of Task 16, which the plan and the manual check refer to
FIXED_IDS = [
    "btn-preview", "btn-render", "preview", "btn-reset", "mode", "preview-video", "filmstrip",
    "tab-lines", "tab-style", "tab-timing", "tab-shoot", "tab-results",
    "styles-gallery", "swatches", "img-palette", "contrast", "describe", "btn-ask", "style-name",
    "btn-save-style",
]
FILES = ("index.html", "app.js", "style.css")


def read(name: str) -> str:
    return (STUDIO_DIR / name).read_text(encoding="utf-8")


def block(name: str) -> str:
    """The text of app.js between `/* name:begin ... */` and `/* name:end */`: the pieces of the page that can
    run without a page (the merge patches, the calls to the server, the Ask Claude button)."""
    text = read("app.js")
    return text[text.index(f"/* {name}:begin"):text.index(f"/* {name}:end */")]


def run_node(script: str, data) -> object:
    """Run a script in node, with `data` as JSON on its standard input, and return the JSON it prints."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    prelude = "const data = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
    done = subprocess.run(
        [node, "-e", prelude + script], input=json.dumps(data), capture_output=True, text=True, encoding="utf-8"
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_page_has_fixed_ids(server):
    page = server.get("/")

    assert page.status == 200 and page.headers["content-type"] == "text/html; charset=utf-8"
    html = page.body.decode("utf-8")
    for fixed in FIXED_IDS:
        assert f'id="{fixed}"' in html, fixed
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})  # an id is one element's


def test_page_copy_is_us_english():
    for name in FILES:
        text = read(name)
        assert not re.search("colour", text, re.IGNORECASE), name  # "color", never "colour"
    assert "color" in read("app.js").lower()  # (and the test is looking at real files)


def test_app_js_syntax():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    checked = subprocess.run([node, "--check", str(STUDIO_DIR / "app.js")], capture_output=True, text=True)

    assert checked.returncode == 0, checked.stderr


def test_the_files_of_the_page_are_served(server):
    assert server.get("/studio/app.js").headers["content-type"] == "text/javascript; charset=utf-8"
    assert server.get("/studio/style.css").headers["content-type"] == "text/css; charset=utf-8"
    html = read("index.html")
    assert 'href="/studio/style.css"' in html and 'src="/studio/app.js"' in html
    for path in re.findall(r'(?:href|src)="(/studio/[^"]+)"', html):
        assert server.get(path).status == 200, path


def test_the_page_needs_no_network(server):
    # Nothing but this computer: every address in the files is on the server, and the one that is
    # not (the namespace of the page's inline icon) is not fetched
    for name in FILES:
        for address in re.findall(r"https?://[^\s\"'<>)]+", read(name)):
            assert address.startswith("http://www.w3.org/2000/svg"), (name, address)
    assert "@import" not in read("style.css")

    # The fonts of the lettering come from /fonts/, and every one of them is there. They are the notes' five (the
    # sixth, Press Start 2P, is the HUD's and is only drawn into the video); the title had one too, and went
    wanted = re.findall(r'url\("(/fonts/[^"]+)"\)', read("style.css"))
    assert len(wanted) == 5 and len(set(wanted)) == 5
    for url in wanted:
        assert (FONTS_DIR / unquote(url.removeprefix("/fonts/"))).is_file(), url
        reply = server.get(url)
        assert reply.status == 200 and reply.headers["content-type"] == "font/ttf", url
    assert "Nunito" not in read("style.css")
    assert "fonts.googleapis" not in read("index.html") + read("style.css") + read("app.js")


def test_the_page_calls_only_routes_that_exist():
    routes = {path for _, path in studio_server._ROUTES}
    prefixes = {prefix.rstrip("/") for prefix in studio_server._PREFIXES}
    called = {match.rstrip("/") for match in re.findall(r"/api/[A-Za-z0-9_./-]*", read("app.js"))}

    assert called  # it does call the server
    for path in called:
        assert path in routes or path in prefixes, path
    # and it uses the main ones. (There is no /api/layout-mode: the clip has the format it has.)
    assert {"/api/project", "/api/card.png", "/api/layout", "/api/styles", "/api/palette", "/api/style/ask",
            "/api/style/reference", "/api/matte/refine",
            "/api/preview", "/api/render", "/api/frame", "/api/frames", "/api/options"} <= called
    assert "/api/layout-mode" not in called
    assert "/api/audio" not in called and "/api/audio" not in routes  # there is no music to upload


def test_the_script_finds_the_elements_it_asks_for():
    html, script = read("index.html"), read("app.js")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    asked = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", script)) | set(re.findall(r"\$\('\[data-count=\"([^\"]+)\"\]'\)", script))
    asked |= set(re.findall(r"'(f-[a-z0-9-]+|t-[a-z_]+|sw-[a-z]+)'", script)) - {"t-", "sw-"}
    dynamic = {"t-hold", "t-head_speed", "t-tail", "t-release_from"}  # made from the timing table
    missing = {name for name in asked if name not in ids and name not in dynamic}
    assert not missing, missing


def test_the_copy_of_the_plan_is_on_the_page():
    text = read("index.html") + read("app.js")
    for words in (
        "Main Character Studio", "Preview video", "Render", "Rendering\u2026",
        "What you do", "Handle", "Oddly specific detail", "Name on screen",
        "Burst word", "Emphasis word", "Still", "Play",
        "Background dark", "Background light", "Texture", "Edge", "Text",
        "Take the colors from an image", "Describe it to Claude", "Ask Claude", "Claude is on it",
        "Claude updated your style", "Save as my style", "Name your style",
        "Hold", "Head speed", "Tail", "Release from", "Staggered", "All at once", "Boil",
        "Shutter click", "Original sound", "Freeze lands at", "Shooting guide",
        "This note doesn't fit here. Shorten it or drag it somewhere else.",
        "easy to read on a phone", "hard to read on a phone. Try a darker or lighter text color.",
        "Little extras", "One hero element", "Cutout edge", "Notes", "Arrows", "Background",
    ):
        assert words in text, words
    assert "Your 'but&hellip;'" in read("index.html")  # the label of the confession line


def test_the_page_says_where_the_title_goes():
    html = read("index.html")

    # One quiet line under the lines: the title is not part of the edit, the person adds it in the app
    assert 'The title ("this is your sign to&hellip;") goes on in Instagram or TikTok, like the sound. ' \
           "Claude suggests one when you render." in html
    assert html.index("Edit the lines") < html.index("goes on in Instagram") < html.index('id="f-tl"')
    assert re.search(r'<p class="[^"]*\bhint\b[^"]*">The title \(', html)  # a quiet one: the page's own muted hint


def test_the_page_does_not_show_the_versions():
    # The interview still writes three versions, but the person picks in the chat (Claude patches the copy), so
    # the studio has no "Pick a version" block, no cards for them and no styles for the cards. The lines stay
    html, script, css = read("index.html"), read("app.js"), read("style.css")
    assert "Pick a version" not in html and "Nothing written yet" not in script
    assert 'id="versions"' not in html and "paintVersions" not in script and "pickVersion" not in script
    assert not re.search(r"\.ver\b|\.ver-|\.versions", css)
    assert "Edit the lines" in html and 'id="f-bl"' in html
    # and editing a line changes the copy and nothing else: the versions are as the interview wrote them
    assert re.search(r"function editCopy\(change\) \{\s*edit\(\{ copy: change \}\);\s*\}", script)


def test_the_original_sound_has_a_switch_and_the_preview_is_not_muted():
    html, script = read("index.html"), read("app.js")

    # The Sound switch is in the Timing tab with the others, off the same markup, and it edits audio.original
    timing = html[html.index('id="tab-timing"'):html.index('id="tab-shoot"')]
    assert re.search(r'<button type="button" class="tog" id="tog-sound" aria-pressed="true" aria-label="Original sound">On</button>', timing)
    assert "The clip's own sound. Off mutes it." in timing
    assert timing.index('id="tog-shutter"') < timing.index('id="tog-sound"')
    assert "edit({ audio: { original: !(state.project.audio.original !== false) } }, { now: true })" in script
    assert "['tog-sound', state.project.audio.original !== false]" in script
    # the preview plays with its sound
    video = re.search(r'<video id="preview-video"[^>]*>', html).group(0)
    assert "muted" not in video


def test_the_page_has_no_music_controls_and_no_trend_timing():
    html, script, css = read("index.html"), read("app.js"), read("style.css")

    # The skill only makes the freeze-frame effect: the person adds the music in their own app
    assert "music" not in (html + script + css).lower()
    assert not re.findall(r'id="music-[^"]*"', html) and "type=\"file\" accept=\"audio" not in html
    assert "audio.track" not in script and "onsets" not in script and "with-music" not in script
    assert "freeze_at" not in script and "trending sound" not in html + script
    assert "t-freeze_at" not in html


def test_the_timing_tab_says_where_the_freeze_lands():
    html, script = read("index.html"), read("app.js")

    # A read-only line in the Timing tab, above the sliders
    timing = html[html.index('id="tab-timing"'):html.index('id="tab-shoot"')]
    assert re.search(r'<p id="freeze-lands" class="freeze-lands" role="status"></p>', timing)
    assert timing.index('id="freeze-lands"') < timing.index('id="timing-sliders"')
    # It is painted whenever the timing is, which is whenever the head speed or the freeze frame changes
    assert re.search(r"function paintTiming\(\) \{.*?paintFreezeLands\(\);\s*\}", script, re.S)

    # and it is computed as the timeline does: freeze frame / fps / head speed, at 2 decimals
    start = script.index("function paintFreezeLands() {")
    end = script.index("\n}\n", start)
    function = script[start:end + 2]
    code = """
    const line = { textContent: '' };
    const $ = () => line;
    const run = new Function('state', '$', FUNCTION + '; paintFreezeLands(); return null;');
    const out = [];
    for (const [frame, fps, speed] of data) {
      const state = { project: { clip: { fps }, freeze: { frame }, timing: { head_speed: speed } } };
      run(state, $);
      out.push(line.textContent);
    }
    console.log(JSON.stringify(out));
    """.replace("FUNCTION", json.dumps(function))
    shown = run_node(code, [[74, 24, 1.0], [74, 24, 0.75], [90, 30, 1.0], [0, 30, 1.5]])
    assert shown == ["Freeze lands at 3.08 s", "Freeze lands at 4.11 s", "Freeze lands at 3.00 s", "Freeze lands at 0.00 s"]


def test_the_shooting_guide_is_two_steps():
    html = read("index.html")
    guide = html[html.index('id="tab-shoot"'):html.index('id="tab-results"')]

    assert "Five minutes" not in guide
    assert re.findall(r"<li>(.*?)</li>", guide) == [
        "<b>Record one shot.</b>",
        "<b>Hold a deadpan stare</b> for two seconds, then carry on naturally.",
    ]


def test_the_page_has_no_title_fields_and_no_format_switch():
    # The edit has no title and takes the clip's own format (Amendment 1), so the page has neither the title's two
    # fields (with their counters and their names in the warnings) nor a Format block nor the call that rebuilt the
    # cutout for one, and none of the title's lettering
    for name in FILES:
        text = read(name)
        assert not re.search("hook", text, re.IGNORECASE), name
        assert "Nunito" not in text and "layout-mode" not in text, name
        assert not re.search(r"Letterbox|Full screen|letterbox|fullscreen", text), name
        for gone in ("switchLayout", "format-", "dataset.layout", "data-layout", "CANVAS", "1920", ".phone"):
            assert gone not in text, (name, gone)


def test_the_preview_takes_the_shape_of_the_clip_by_its_css_and_its_script():
    css, script = read("style.css"), read("app.js")

    # One number, the clip's width over its height, is the shape of everything that shows the card or the video:
    # the preview, the filmstrip's frames, the gallery's thumbnails and the Results video. The page sets it from the
    # plan as soon as it has the project (the css has a shape for the moment before)
    assert re.search(r":root\s*\{[^}]*--aspect:\s*[\d.]+;", css)
    assert re.search(r"setProperty\('--aspect',\s*String\(work\.width / work\.height\)\)", script)
    assert "9 / 16" not in css and "9/16" not in css and "1080 / 764" not in css  # no shape of its own
    # and the preview waits for it, so that its box does not change shape in front of the person at load
    assert re.search(r"html:not\(\.shaped\)\s+\.clip-box\s*\{[^}]*visibility:\s*hidden", css)
    assert "classList.add('shaped')" in script
    for rule in (r"\.clip-box", r"\.frame", r"\.gthumb", r"\.result-video"):
        found = re.search(rule + r"\s*\{([^}]*)\}", css)
        assert found and re.search(r"aspect-ratio:\s*var\(--aspect", found.group(1)), rule
    # the preview fits its area whole, with no page scroll: the biggest box of that shape that fits. A tall clip
    # is limited by the height of the area and a wide one by its width, in the one formula
    assert "container-type: size" in css
    assert re.search(r"width:\s*min\(100cqw,\s*calc\(100cqh \* var\(--aspect\)\)\)", css)
    box = re.search(r"\.clip-box\s*\{([^}]*)\}", css).group(1)
    assert re.search(r"width:\s*min\(100%,\s*calc\(\(100vh - \d+px\) \* var\(--aspect\)\)\)", box)  # (older browsers)
    # the card and the preview video fill the box: it has their shape
    fill = re.search(r"\.card-img,\s*\.preview-video\s*\{([^}]*)\}", css).group(1)
    assert "inset: 0" in fill and "width: 100%" in fill and "height: 100%" in fill
    assert "object-fit: contain" in fill  # (the box and the card are the same shape, so contain fills it)


def test_the_page_fits_the_screen_by_its_css():
    css = read("style.css")
    # the controls column scrolls on its own, and the columns stack under 900 px
    assert re.search(r"\.panels\s*\{[^}]*overflow-y:\s*auto", css)
    assert "@media (max-width: 899px)" in css
    # the interface's own font is the system's
    assert 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif' in css
    # the tokens of the approved prototype
    for token in ("--bg: #0e0e11", "--card: #17171c", "--line: #2a2a31", "--text: #ecebe6", "--muted: #a09ea8",
                  "--accent: #f5df98"):
        assert token in css, token


def test_every_request_the_page_posts_says_it_is_json():
    # The server refuses a JSON route that is not sent as application/json, so a web page on another site
    # cannot post to it without the preflight that it refuses. A call with no body sends an empty object
    script = block("api") + """
    const calls = [];
    globalThis.fetch = async (path, init) => {
      calls.push({ path, method: init.method, headers: init.headers, body: init.body instanceof Blob ? 'a blob' : init.body ?? null });
      return { ok: true, json: async () => ({}) };
    };
    (async () => {
      await api('POST', '/api/preview');
      await api('POST', '/api/project', { copy: { name: 'Robin' } });
      await api('GET', '/api/project');
      await api('POST', '/api/palette', new Blob(['x']), { 'Content-Type': 'image/png' });
      await api('POST', '/api/style/reference', new Blob(['x']), { 'X-Filename': 'a.png' });
      console.log(JSON.stringify(calls));
    })();
    """

    preview, patch, read_only, palette, reference = run_node(script, None)

    assert preview["headers"] == {"Content-Type": "application/json"} and preview["body"] == "{}"
    assert patch["headers"] == {"Content-Type": "application/json"} and patch["body"] == '{"copy":{"name":"Robin"}}'
    assert read_only["body"] is None and "Content-Type" not in read_only["headers"]
    assert palette["headers"] == {"Content-Type": "image/png"} and palette["body"] == "a blob"  # its own type
    assert reference["headers"] == {"X-Filename": "a.png"} and reference["body"] == "a blob"
    # and the one call that does not go through api(), the last edit sent when the page is closed
    for line in read("app.js").splitlines():
        if "fetch(" in line and "method: 'POST'" in line:
            assert "'Content-Type': 'application/json'" in line, line


def test_ask_claude_never_dead_ends():
    # A request that Claude's session never answers must not leave the button dead: while one is pending the
    # button reads "Ask again" and posts a new one, and Cancel takes the request away
    html, script = read("index.html"), read("app.js")
    assert 'id="btn-ask-cancel"' in html and ">Cancel</button>" in html
    assert re.search(r'<button[^>]*id="btn-ask"[^>]*>Ask Claude</button>', html)  # the plan's label, to begin with
    assert "style_request: null" in script  # Cancel is a patch with a null

    paint = block("ask-button") + """
    const states = [];
    const seen = (label) => states.push({
      label, button: button.textContent, disabled: button.disabled, cancelHidden: cancel.hidden, status: status.textContent,
    });
    const sending = (on) => { ask.sending = on; };
    state.project = { style_request: null };
    paintAsk('');
    seen('nothing asked');
    state.project = { style_request: { text: 'x', at: 'a', answered_at: null } };
    paintAsk(data.on);
    seen('pending');
    sending(true);
    paintAsk(data.on);
    seen('pending, sending again');
    sending(false);
    state.project = { style_request: { text: 'x', at: 'a', answered_at: 'b' } };
    paintAsk('Claude updated your style', 'done');
    seen('answered');
    state.project = {};  // a project whose key was deleted by a patch with a null
    paintAsk('');
    seen('cancelled');
    console.log(JSON.stringify(states));
    """
    stubs = """
    const button = { textContent: 'Ask Claude', disabled: false };
    const cancel = { hidden: true };
    const status = { textContent: '', className: '' };
    const $ = (selector) => ({ '#btn-ask': button, '#btn-ask-cancel': cancel, '#ask-status': status })[selector];
    const state = { project: {} };
    """
    # the stubs come first: the block uses them the way the page's own $ and state do
    ON_IT = "Claude is on it" + chr(0x2026)  # (built, not typed, so that no tool can change it)
    results = {entry["label"]: entry for entry in run_node(stubs + paint, {"on": ON_IT})}

    assert results["nothing asked"]["button"] == "Ask Claude" and not results["nothing asked"]["disabled"]
    assert results["nothing asked"]["cancelHidden"] is True
    pending = results["pending"]
    assert pending["button"] == "Ask again" and pending["disabled"] is False and pending["cancelHidden"] is False
    assert pending["status"] == ON_IT and ON_IT in script  # the page says it as the plan does
    assert results["pending, sending again"]["disabled"] is True  # only while a request is on its way
    assert results["answered"]["button"] == "Ask Claude" and results["answered"]["cancelHidden"] is True
    assert results["cancelled"]["button"] == "Ask Claude" and results["cancelled"]["cancelHidden"] is True


def test_the_gallery_shows_the_whole_card_in_the_clips_shape():
    # The card is the clip's own frame (nothing around it, no bars to crop away), so a thumbnail is the card whole
    # at the clip's aspect: no crop, no layout to tell the css about
    css, script = read("style.css"), read("app.js")
    thumb = re.search(r"\.gthumb\s*\{([^}]*)\}", css).group(1)
    assert "aspect-ratio: var(--aspect)" in thumb
    image = re.search(r"\.gthumb img\s*\{([^}]*)\}", css).group(1)
    assert "object-fit: cover" in image and "object-position" not in image  # the same shape: cover fills, crops nothing
    assert "[data-layout" not in css and "dataset.layout" not in script


def test_a_handle_is_a_share_of_the_card_and_a_drag_counts_plan_pixels():
    # The notes sit on the whole card, [0, 0, w, h] in px of the plan, whatever shape the clip is: a handle is
    # placed by its own box as a share of the plan (not of a canvas, and not past the offset of a stage on it),
    # and a move of the mouse is as many plan px as the plan is wider than the preview that is shown
    cases = {
        "landscape": {"stage": [0, 0, 1080, 608], "box": [270, 152, 324, 121.6], "shown": 540},
        "portrait": {"stage": [0, 0, 1080, 1920], "box": [108, 192, 540, 384], "shown": 270},
    }
    script = block("geometry") + """
    const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
    const answer = {};
    for (const [name, c] of Object.entries(data)) {
      const scale = planPerShown(c.stage, c.shown);
      answer[name] = {
        style: boxStyle(c.box, c.stage), scale,
        dragged: movedBox(c.box, 50 * scale, 25 * scale, c.stage),
        bottom_right: movedBox(c.box, 5000, 5000, c.stage),
        top_left: movedBox(c.box, -5000, -5000, c.stage),
        pin: pinPatch('bl', 123.4, 56.6),
      };
    }
    console.log(JSON.stringify(answer));
    """

    answer = run_node(script, cases)

    def share(styles):
        return {key: float(text.removesuffix("%")) for key, text in styles.items()}

    land, tall = answer["landscape"], answer["portrait"]
    assert share(land["style"]) == pytest.approx({"left": 25, "top": 25, "width": 30, "height": 20})
    assert share(tall["style"]) == pytest.approx({"left": 10, "top": 10, "width": 50, "height": 20})
    assert (land["scale"], tall["scale"]) == (2, 4)  # 1080 plan px over the 540 and the 270 that are shown
    assert land["dragged"] == pytest.approx([270 + 100, 152 + 50])
    assert tall["dragged"] == pytest.approx([108 + 200, 192 + 100])
    # a box stays on the card, up to its own edge: the card is [0, 0, w, h], so the corner is the card's
    assert land["bottom_right"] == pytest.approx([1080 - 324, 608 - 121.6]) and land["top_left"] == [0, 0]
    assert tall["bottom_right"] == pytest.approx([1080 - 540, 1920 - 384]) and tall["top_left"] == [0, 0]
    # a note is pinned by the corner of its box in whole px of the plan: nothing to take away from it
    assert land["pin"] == tall["pin"] == {"positions": {"bl": [123, 57]}}


def test_the_page_shows_the_notice_init_left_in_the_project_and_takes_it_away():
    script = read("app.js")

    # init leaves a sentence in project.notice when it resets the pinned notes (the page shows it as a toast, once)
    assert re.search(r"toast\(state\.project\.notice\)", script)
    assert "edit({ notice: null }, { now: true })" in script  # a null in a patch deletes the key
    assert re.search(r"refreshCard\(\);\s*showNotice\(\);", script)  # once, when the page opens


def test_the_arrow_keys_step_the_freeze_only_in_still_mode():
    # Scrubbing the preview with the arrows moved the freeze by accident: in Play mode they do nothing to it
    script = read("app.js")
    start = script.index("document.addEventListener('keydown', (event) => {")
    end = script.index("\n});", start)
    handler = script[start + len("document.addEventListener('keydown', "):end + 2]
    assert "ArrowRight" in handler and "setFreeze" in handler
    code = """
    class Element {}
    const results = {};
    for (const mode of ['play', 'still']) {
      const state = { mode, project: { freeze: { frame: 71 } } };
      const nextFrame = (frame, step) => frame + step;
      const setFreeze = (frame) => { state.project.freeze.frame = frame; };
      const run = new Function('state', 'nextFrame', 'setFreeze', 'Element', 'return ' + HANDLER);
      run(state, nextFrame, setFreeze, Element)({ key: 'ArrowRight', target: null, preventDefault() {} });
      results[mode] = state.project.freeze.frame;
    }
    console.log(JSON.stringify(results));
    """.replace("HANDLER", json.dumps(handler))
    assert run_node(code, None) == {"play": 71, "still": 72}
    assert "In Still mode, the left and right arrow keys step one frame." in script


def test_the_style_tab_builds_one_toggle_for_each_of_the_fourteen_extras(server):
    from mclib.styles import recipes

    options = server.get("/api/options").json()

    # the page makes a toggle for each id the server lists, in this order, with the server's labels
    assert options["options"]["extras"] == [
        "tape", "stars", "grain", "hearts", "circle", "crown", "sparkles", "motion", "confetti", "rec", "vhs",
        "leak", "nametag", "bubble",
    ]
    assert [options["labels"]["extras"][key] for key in options["options"]["extras"]] == [
        "Tape", "Stars", "Film grain", "Hearts", "Scribble circle", "Doodle crown", "Sparkles", "Motion lines",
        "Confetti", "REC timestamp", "VHS lines", "Light leak", "Name tag", "Speech bubble",
    ]
    assert set(options["options"]["extras"]) == set(recipes.EXTRAS)
    script = read("app.js")
    assert "for (const id of options.extras)" in script and "labels.extras[id]" in script
    assert 'id="extras"' in read("index.html")
    # the toggles wrap, so that fourteen of them fit a 1280 x 720 window without a sideways scroll
    assert re.search(r"\.toggles\s*\{[^}]*flex-wrap:\s*wrap", read("style.css")) or re.search(
        r"\.opts,\s*\.chips,\s*\.toggles\s*\{[^}]*flex-wrap:\s*wrap", read("style.css")
    )


def test_the_bubble_field_is_in_the_lines_tab_and_shows_only_with_the_bubble_extra():
    html, script = read("index.html"), read("app.js")

    field = re.search(r'<label class="field" for="f-bubble" id="field-bubble">.*?</label>', html, re.S).group(0)
    assert 'id="f-bubble"' in field and 'data-count="f-bubble"' in field and "Bubble text" in field
    assert html.index('id="extras-fields"') < html.index('id="f-bubble"') < html.index('id="field-stats"')
    # it counts against the bubble's soft limit like the notes, writes copy.extras.bubble (null when empty),
    # and is hidden unless the style has the bubble extra on
    assert "'f-bubble': { read: (c) => (c.extras || {}).bubble" in script and "limit: 'bubble'" in script
    assert "{ extras: { bubble: v.trim() === '' ? null : v } }" in script
    assert "$('#field-bubble').hidden = !bubble;" in script
    assert "const bubble = Boolean(style.extras && style.extras.bubble);" in script
    assert "'copy.extras.bubble': 'Bubble text'" in script


def test_the_server_gives_the_page_the_bubble_limit(server):
    assert server.get("/api/options").json()["soft_limits"]["bubble"] == 14
