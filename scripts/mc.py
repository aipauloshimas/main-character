"""mc.py: the command line of /main-character.

    python mc.py doctor [--json] [--download-model]
    python mc.py init CLIP [--project-dir DIR]
    python mc.py card PROJECT_DIR [--out PNG] [--style ID]
    python mc.py patch PROJECT_DIR --file PATCH.json
    python mc.py style PROJECT_DIR --file RECIPE.json [--answers STAMP]
    python mc.py refine PROJECT_DIR --detail auto|high|max
    python mc.py render PROJECT_DIR [--preview]
    python mc.py serve PROJECT_DIR [--port N]
    python mc.py wait PROJECT_DIR [--since ISO]

Each command prints lines that start with a word (PROJECT, WARNING, FREEZE, FLAG, NOTE, CONTRAST, STYLE,
REFINED, VIDEO, DONE, ...) so that Claude can read them. A failure prints "ERROR" and the reason on
stderr and exits with 1, and so does doctor when something is missing. Progress goes to stderr.
Everything the commands need is imported when a command runs, so that doctor still works on a
machine that has none of the packages.

Claude changes project.json only through patch (a JSON merge patch) and style (a whole style recipe,
which also answers "Ask Claude"), and through refine (a finer cutout, which saves matte.detail).
They read and write the file the way the studio does, so patch and style are safe while the studio is
open. refine rewrites the cutout's pictures, so it is run only while no studio render runs: nothing but
that rule keeps the two apart yet (a lock across programs is still to come).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_HUD_KEYS = ("hud_top", "hud_bottom")  # the Game HUD's labels: boxes of the layout that nothing can pin
_HUD_NOTE = "NOTE The Game HUD has no room here; pick another hero if it covers something."
_REQUEST_CHANGED = "NOTE The request changed; the page shows the newest one."


def main(argv: list[str] | None = None) -> int:
    """Run the command in argv (the command line, without the program name) and return the exit
    code: 0 when it worked, 1 when it failed, and 2 for a command line that makes no sense."""
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError, RuntimeError) as error:
        # A missing file or a timeout (OSErrors), a project or a setting that is not right
        # (ValueErrors) and the media and matte errors (RuntimeErrors) all carry a sentence written
        # for the person
        print(f"ERROR {error}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mc.py", description="Edit a freeze-frame character intro.")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check that everything the skill needs is installed")
    doctor.add_argument("--json", action="store_true", help="print the result as JSON")
    doctor.add_argument(
        "--download-model", action="store_true",
        help="download the matte model (about 15 MB); only after the person agrees",
    )
    doctor.set_defaults(run=_doctor)

    init = commands.add_parser("init", help="make a project from a clip")
    init.add_argument("clip", help="the video file, 2 to 30 seconds long")
    init.add_argument("--project-dir", help="the project folder (default: ./main-character/<clip name>)")
    init.set_defaults(run=_init)

    card = commands.add_parser("card", help="draw the card of the freeze frame as a PNG")
    card.add_argument("project_dir")
    card.add_argument("--out", help="where to write the PNG (default: card.png in the project folder)")
    card.add_argument("--style", help="the id of a style to draw instead of the project's")
    card.set_defaults(run=_card)

    patch = commands.add_parser("patch", help="change project.json with a JSON merge patch (a null deletes a key)")
    patch.add_argument("project_dir")
    patch.add_argument("--file", required=True, help="the JSON file that holds the patch")
    patch.set_defaults(run=_patch)

    style = commands.add_parser("style", help="save a style recipe as the project's own, and answer Ask Claude")
    style.add_argument("project_dir")
    style.add_argument("--file", required=True, help="the JSON file that holds the whole recipe")
    style.add_argument(
        "--answers", metavar="STAMP",
        help="the STYLE stamp the recipe is for: the request is marked answered only if it is still that one",
    )
    style.set_defaults(run=_style)

    refine = commands.add_parser("refine", help="cut the person out again around the freeze frame, in more detail")
    refine.add_argument("project_dir")
    refine.add_argument("--detail", required=True, help='how closely to cut: "auto", "high" or "max"')
    refine.set_defaults(run=_refine)

    render = commands.add_parser("render", help="render the videos")
    render.add_argument("project_dir")
    render.add_argument("--preview", action="store_true", help="a small, fast video, with the clip's own sound")
    render.set_defaults(run=_render)

    serve = commands.add_parser("serve", help="run the studio page for a project, on this computer only")
    serve.add_argument("project_dir")
    serve.add_argument("--port", type=_port, help="the port to listen on (default: the first free one from 8340)")
    serve.set_defaults(run=_serve)

    wait = commands.add_parser("wait", help="wait for Done or a style request in the studio")
    wait.add_argument("project_dir")
    wait.add_argument("--since", help="only an event after this ISO time counts")
    wait.set_defaults(run=_wait)
    return parser


# --- the commands ---------------------------------------------------------------------------


def _doctor(args: argparse.Namespace) -> int:
    from mclib import doctor

    if args.download_model:
        _download_model(doctor)
    result = doctor.check()
    print(json.dumps(result, indent=2) if args.json else doctor.report(result))
    return 0 if result["ok"] else 1


def _download_model(doctor) -> None:
    """Download the matte model, unless it is there already. A failure is an error, as the person
    needs to know that it did not work."""
    if doctor.MODEL_PATH.is_file():
        print(f"The matte model is already at {doctor.MODEL_PATH}.", file=sys.stderr)
        return
    try:
        from mclib import matte
    except ImportError as error:  # numpy or Pillow is missing
        raise RuntimeError(
            f"The packages are not installed yet ({error}). Run {doctor.PIP_FIX} first."
        ) from error
    print("Downloading the matte model (about 15 MB)...", file=sys.stderr)
    print(f"Downloaded the matte model to {matte.download_model(doctor.MODEL_PATH)}.", file=sys.stderr)


def _init(args: argparse.Namespace) -> int:
    from mclib import pipeline
    from mclib import project as project_file

    project_dir, warnings = pipeline.init_project(
        Path(args.clip),
        Path(args.project_dir) if args.project_dir else None,
        progress=_show_progress,
    )
    freeze = project_file.load(project_dir)["freeze"]
    print(f"PROJECT {project_dir}")
    for warning in warnings:
        print(f"WARNING {warning}")
    print(f"FREEZE {freeze['frame']} SUGGESTED {','.join(str(frame) for frame in freeze['suggested'])}")
    return 0


def _card(args: argparse.Namespace) -> int:
    from mclib import card
    from mclib import project as project_file
    from mclib.styles import recipes

    project_dir = _project_dir(args.project_dir)
    project = project_file.load(project_dir)
    recipe = recipes.get_style(args.style)["recipe"] if args.style else None
    image = card.render_card(project_dir, project, style=recipe)
    out = Path(args.out) if args.out else project_dir / "card.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(out.resolve())
    # What does not fit, in the style that was drawn: a note whose text was cut (too long for its box, or
    # with a character its font can't draw), or that touches the face, overlaps another note or sits too
    # close to one (notes keep a gap between them), or is on what the app covers (the title room, the
    # caption, the buttons). (A note that is both is cut.) The Game HUD's two labels get no FLAG: a FLAG
    # names a key that positions can move, and they have none. One NOTE says there is no room for them.
    drawn = project if recipe is None else {**project, "style": recipe}
    hud = False
    for key, box in card.elements(project_dir, drawn).items():
        if key in _HUD_KEYS:
            hud = hud or box.cut or box.clash
        elif box.cut or box.clash:
            print(f"FLAG {key} {'cut' if box.cut else 'clash'}")
    if hud:
        print(_HUD_NOTE)
    return 0


def _patch(args: argparse.Namespace) -> int:
    from mclib import palette
    from mclib import project as project_file

    project_dir = _project_dir(args.project_dir)
    patch = _read_json_object(args.file, "patch")
    if "clip" in patch or "status" in patch:  # (the studio's patches never set them either)
        raise ValueError(
            "A patch can't touch clip or status: init and render write those. Leave them out of the patch."
        )
    project, warnings = project_file.update(project_dir, patch)
    for warning in warnings:
        print(f"WARNING {warning}")
    print(f"CONTRAST {palette.floor_contrast(palette.text_contrast(project['style']))}")
    return 0


def _style(args: argparse.Namespace) -> int:
    from mclib import config, palette, wait
    from mclib import project as project_file
    from mclib.server import now_stamp
    from mclib.styles import recipes

    project_dir = _project_dir(args.project_dir)
    answers = None
    if args.answers is not None:  # an error now, and not a style that answers the wrong request
        try:
            answers = wait.parse_time(args.answers)
        except ValueError:
            raise ValueError("--answers must be a time like 2026-10-26T05:00:00Z.") from None
    recipe = _read_json_object(args.file, "style")
    recipes.validate_recipe(recipe)
    recipe = recipes.complete(recipe)  # (a style written before the newer extras has them off)
    contrast = palette.text_contrast(recipe)
    if contrast < config.CONTRAST_MIN:
        raise ValueError(
            f"Text contrast {palette.floor_contrast(contrast)}:1 is hard to read on a phone. "
            "Change the text or background colors."
        )

    changed = []

    def keep(project: dict) -> None:
        project["style"], project["style_source"] = recipe, "custom"
        request = project.get("style_request")
        asked = request.get("at") if isinstance(request, dict) else None
        if answers is not None and not _same_moment(asked, answers):
            changed.append(True)  # asked again or cancelled meanwhile: this style answers no request that is there
        elif isinstance(request, dict) and asked and not request.get("answered_at"):
            request["answered_at"] = now_stamp()  # the page is waiting for it, and shows the new style then

    project_file.update(project_dir, mutate=keep)
    print(f"CONTRAST {palette.floor_contrast(contrast)}")
    print(f"STYLE {' '.join(recipe['name'].split())}")  # (one line, whatever the name holds: Claude reads lines)
    if changed:
        print(_REQUEST_CHANGED)
    return 0


def _refine(args: argparse.Namespace) -> int:
    from mclib import pipeline
    from mclib import project as project_file

    project_dir = _project_dir(args.project_dir)
    frames = pipeline.refine_matte(project_dir, project_file.load(project_dir), args.detail)
    print(f"REFINED {frames} frames")
    return 0


def _render(args: argparse.Namespace) -> int:
    from mclib import render

    result = render.render(_project_dir(args.project_dir), preview=args.preview, progress=_show_progress)
    print(f"VIDEO {result.video}")
    if result.with_music:
        print(f"WITH_MUSIC {result.with_music}")
    if result.cover:
        print(f"COVER {result.cover}")
    print(f"FRAMES {result.frames} FREEZE {result.freeze_out:.2f} RELEASE {result.release_out:.2f}")
    for warning in ([result.warning] if result.warning else []) + result.notes:
        print(f"WARNING {warning}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from mclib import server

    server.serve(_project_dir(args.project_dir), args.port)  # prints the URL; runs until it is stopped
    return 0


def _wait(args: argparse.Namespace) -> int:
    from mclib import wait

    if args.since:
        try:
            wait.parse_time(args.since)
        except ValueError:  # an error now, and not a wait that never ends
            raise ValueError("--since must be a time like 2026-10-26T05:00:00Z.") from None
    event, stamp = wait.wait_event(_project_dir(args.project_dir), since=args.since)
    print(f"{event} {stamp}")
    return 0


# --- helpers --------------------------------------------------------------------------------


def _project_dir(name: str) -> Path:
    """The project folder a command was given, checked to have a project in it."""
    from mclib.project import PROJECT_FILE

    folder = Path(name)
    if not (folder / PROJECT_FILE).is_file():
        raise FileNotFoundError(f"There is no project in {folder.resolve()}. Run init first.")
    return folder.resolve()


def _read_json_object(name: str, what: str) -> dict:
    """The JSON object in a file that a command was given (what is "patch" or "style", for the
    sentences). A byte order mark at the start is let through."""
    path = Path(name)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise FileNotFoundError(f"There is no {what} file at {path}.") from None
    except ValueError as error:  # not UTF-8 text, or not JSON: both are ValueErrors
        raise ValueError(f"The {what} file {path} is not valid JSON ({error}).") from None
    if not isinstance(data, dict):
        raise ValueError(f"The {what} file {path} must hold one JSON object.")
    return data


def _same_moment(stamp, when) -> bool:
    """Whether a stamp from project.json (an ISO time, or None) is the datetime `when`, however the
    time is written: the waiter prints a stamp as the file has it, and Claude hands it back."""
    from mclib import wait

    try:
        return wait.parse_time(stamp) == when
    except (AttributeError, TypeError, ValueError):  # no stamp, or one that is not a time
        return False


def _port(text: str) -> int:
    """A port number for argparse: a whole number from 1 to 65535."""
    if not text.isdigit() or not 1 <= int(text) <= 65535:
        raise argparse.ArgumentTypeError("must be a number from 1 to 65535")
    return int(text)


def _show_progress(stage: str, done: int, total: int) -> None:
    """Print progress to stderr: the first piece of a stage, the last, and about every tenth."""
    if done == total or done == 1 or done * 10 // total != (done - 1) * 10 // total:
        print(f"{stage} {done}/{total}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    # Say it in UTF-8 whatever the console is set to: paths hold accents, and a console set to a
    # legacy code page would fail on a letter it cannot show, in the middle of a command
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
