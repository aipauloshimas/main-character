"""Tests for SKILL.md, the steps that Claude follows when the skill runs.

The file is instructions, so the tests hold it to what the code really has: the frontmatter that
makes Claude Code load the skill, every command it tells Claude to run, the interview it asks word
for word, and the two events it waits for.
"""

import argparse
import json
import re
from pathlib import Path

import mc
from mclib.project import SOFT_LIMITS

SKILL = Path(__file__).resolve().parents[1] / "SKILL.md"

# What the description has to trigger on, as the task brief words it
TRIGGERS = (
    "/main-character",
    "freeze-frame intro",
    "this is your sign to make an introduction video",
    "main character intro",
    "character intro edit",
    "edita minha intro estilo main character",
    "faz aquela intro que congela",
)

# The five questions of the interview (design, section 9), word for word. The ellipsis of "but..."
# is built from its code point, so that no editor or tool can change it.
QUESTIONS = (
    "What do you do? One line, the way you'd say it to someone at a party.",
    'What\'s your "but' + chr(0x2026) + '"? An honest confession, a flaw, or a funny contradiction. '
    "This is the line that gets the laugh.",
    'One oddly specific detail that\'s only yours: a number, a habit, an object. "4 coffees before 9am" '
    'beats "loves coffee".',
    "Your @ handle, and the name you'd like on screen.",
    "The tone: laughing at yourself, confident, chaotic, mysterious, or something else?",
)
CLOSING = "Stuck on any of these? Say 'give me options'."


def skill_text():
    return SKILL.read_text(encoding="utf-8")


def test_skill_md_frontmatter():
    text = skill_text()

    assert text.startswith("---\n")
    head = text[len("---\n"):].split("\n---\n", 1)[0].splitlines()
    assert "name: main-character" in head
    (description,) = [line for line in head if line.startswith("description:")]
    for trigger in TRIGGERS:
        assert trigger in description, trigger
    # It is one single-quoted YAML text: the description holds ": " and quotes, which a plain text
    # could not, and an apostrophe inside would end it (it would have to be doubled)
    value = description.removeprefix("description: ")
    assert len(value) > 2 and value.startswith("'") and value.endswith("'")
    assert "'" not in value[1:-1].replace("''", "")


def test_skill_md_commands_exist():
    parser = mc._parser()
    (subcommands,) = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    commands = set(subcommands.choices)
    used = set(re.findall(r'mc\.py"?\s+(\w+)', skill_text()))

    assert used <= commands, sorted(used - commands)  # every command it runs is a real one
    assert {"patch", "style", "refine"} <= commands  # including the three the skill added
    assert {"doctor", "init", "card", "patch", "style", "refine", "render", "serve", "wait"} <= used


def test_skill_md_has_interview_questions():
    lines = skill_text().splitlines()

    # Each question is one whole line (a wrapped one would be cut in two), and the five are one block
    found = [[number for number, line in enumerate(lines) if question in line] for question in QUESTIONS]
    assert all(len(where) == 1 for where in found), found
    first = found[0][0]
    assert [where[0] for where in found] == list(range(first, first + len(QUESTIONS)))
    assert sum(CLOSING in line for line in lines) == 1  # and the offer of options closes it


def test_skill_md_handles_both_events():
    text = skill_text()

    # The waiter prints STYLE <stamp> or DONE <stamp>, and each has its own step
    assert "STYLE <stamp>" in text and "DONE <stamp>" in text
    assert re.search(r"^## .*On STYLE", text, re.MULTILINE)
    assert re.search(r"^## .*On DONE", text, re.MULTILINE)


# --- Amendment 1: the clip's own format, and no title in the edit -------------------------------


def test_skill_md_renders_the_clips_own_format():
    text = skill_text()
    head = text.split("\n---\n", 1)[0]

    # the description says what the skill makes, and it is not a 9:16 MP4 any more
    assert "renders the clip in its own format (16:9 stays 16:9, 9:16 stays 9:16)" in head
    assert "9:16 MP4" not in head
    # init has no --layout (the command has none), and no letterbox is mentioned: the video is the clip's frame
    assert "--layout" not in text and "letterbox" not in text.lower() and "full screen" not in text.lower()
    assert "keeps the clip's own format and resolution" in text
    assert "A clip under 1080 wide is scaled up to 1080" in text
    # the card is the plan size, and the crop numbers follow from its height
    assert "1080x608 for 16:9, 1080x1920 for 9:16" in text
    assert "x goes up to 600 and y up to H" + chr(0x2212) + "480, where H is the card's height" in text
    assert 'patch `{"positions": {"<key>": [x, y]}}`' in text.replace("\n  ", " ")  # how a note that clashes is moved
    assert "Never put a note on the face." in text


def test_skill_md_has_no_title_in_the_copy_and_suggests_one_in_the_caption():
    text = skill_text()

    # a version is the four notes, the name and the extras: no hook anywhere, in the rules, the limits or the example
    assert "hook" not in text.lower()
    example = re.search(r"One version looks like this:\n\n```json\n(.*?)\n```", text, re.DOTALL).group(1)
    version = json.loads(example)
    assert set(version) == {"notes", "name", "extras"} and set(version["notes"]) == {"tl", "tr", "bl", "br"}
    listed = re.search(r"within the soft limits: (.*?)\.\n", text, re.DOTALL).group(1)
    limits = dict(re.findall(r"`(tl|tr|bl|br|name)` (\d+)", listed))
    # the code's own, and none for a title (the speech bubble's limit is told with the extras, not with the notes')
    assert {key: int(value) for key, value in limits.items()} == {k: v for k, v in SOFT_LIMITS.items() if k != "bubble"}
    assert "`extras.bubble`" in text and f"{SOFT_LIMITS['bubble']} characters at" in " ".join(text.split())

    # the title is suggested in caption.txt, before the hashtags, and the rules of the old hook are its rules
    step = text[text.index("## 9. On DONE"):text.index("## Style recipes")]
    assert step.index("a short caption") < step.index('"Title to add in Instagram:"') < step.index("one line of 5 hashtags")
    assert 'Line one is "this is your sign to".' in step
    assert 'Line two is the trend\'s own ("make an introduction video of yourself") or a twist' in step
    assert "lowercase" in step
    # the stills are at the clip's resolution, so the crops are sized by the still
    assert "own resolution" in step and "x up to W" + chr(0x2212) + "480 and y up to H" + chr(0x2212) + "480" in step


def test_readme_says_the_reel_keeps_the_clips_own_format():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "your clip's own format and resolution" in readme
    assert "1080x1920" not in readme and "letterbox" not in readme.lower() and "hook" not in readme.lower()
    assert "a two-line title to add in Instagram or TikTok" in readme
    assert "1. Record one shot.\n2. Hold a deadpan stare for two seconds, then carry on naturally.\n" in readme  # the guide
    assert "Five minutes" not in readme and "Shoot horizontally" not in readme


def test_skill_md_setup_step_names_what_the_doctor_checks_of_ffmpeg():
    text = skill_text()
    step = text[text.index("## 1. Check the setup"):text.index("## 2.")]

    # An iPhone shoots HDR by default: the doctor checks that ffmpeg can convert it, and has libx264
    assert "`HDR clips`" in step and "`libx264`" in step
    assert "ffmpeg 8 or newer" in step and "`brew install ffmpeg-full`" in step and "PATH" in step
    # and a photo given as the clip is one of the errors init can give
    assert "a photo instead of a video" in text[text.index("## 4. Make the project"):text.index("## 5.")]


def test_skill_md_asks_for_the_latin_alphabet_and_explains_a_cut_that_is_a_missing_character():
    text = skill_text()
    step5 = text[text.index("## 5. Write the copy"):text.index("## 6.")]
    step6 = text[text.index("## 6. Check the card"):text.index("## 7.")]

    # the checklist item, in the words of the review: Latin alphabet, accents where the font has them, FLAG tells
    item = " ".join(step5[step5.index("- [ ] The on-screen copy"):].split("\n- ")[0].split())
    assert "uses the Latin alphabet" in item and "Accents are fine where the font has them" in item
    assert "`card`'s `FLAG` lines tell you" in item and "can't draw" in item
    # a cut is also a character the font lacks, and what to do about it
    assert "or has a character its font can't draw" in step6
    assert "write the line again with letters the font has" in " ".join(step6.split())
    assert "never removed" in text[text.index("## Rules"):]  # (emoji are the one thing patch takes out)


def test_skill_md_tells_what_to_do_when_python_cannot_check_the_models_certificate():
    text = " ".join(skill_text().split())
    step = text[text.index("## 1. Check the setup"):text.index("## 2.")]

    # python.org's Python on a Mac fails the download with a certificate error: the fix is a command in its folder
    assert "checked against the official file's SHA-256" in step
    assert "Python couldn't check the certificate (Python from python.org on a Mac)" in step
    assert '"Install Certificates.command" in the Python folder' in step
    readme = (SKILL.parent / "README.md").read_text(encoding="utf-8")
    assert "Install Certificates.command" in readme


def test_skill_md_says_a_clash_is_also_a_note_too_close_to_another():
    text = skill_text()
    step6 = " ".join(text[text.index("## 6. Check the card"):text.index("## 7.")].split())

    # since the notes keep a gap between them, a clash is not only a touch (the same words as mc.py's)
    assert "**clash:** it touches the face, overlaps another note or sits too close to one" in step6
    assert "notes keep a gap" in step6
    assert "touches the face or another note" not in text


def test_skill_md_step_6_says_how_the_game_hud_and_the_keys_of_positions_work():
    text = skill_text()
    step6 = " ".join(text[text.index("## 6. Check the card"):text.index("## 7.")].split())

    # a FLAG line names a key to patch into positions, and the HUD's labels have none: one NOTE instead
    assert "`NOTE The Game HUD has no room here; pick another hero if it covers something.`" in step6
    assert "no `FLAG` line" in step6
    assert "`tl`, `tr`, `bl`, `br` or `burst`" in step6 and "any other key is refused" in step6


def test_skill_md_step_8_passes_the_style_stamp_and_skips_a_request_that_is_gone():
    text = skill_text()
    step8 = " ".join(text[text.index("## 8. On STYLE"):text.index("## 9.")].split())

    assert '--answers "<the STYLE stamp>"' in step8
    assert "NOTE The request changed; the page shows the newest one." in step8
    # the person can cancel or ask again before Claude reads the request
    assert "If `style_request` is gone or already answered when you read it, skip it and run `wait` again" in step8
    # and the command it names takes the flag
    parsed = mc._parser().parse_args(["style", "p", "--file", "f", "--answers", "2026-10-26T05:00:00Z"])
    assert parsed.answers == "2026-10-26T05:00:00Z"


def test_skill_md_runs_refine_only_while_no_studio_render_runs():
    text = " ".join(skill_text().split())

    assert "Run `refine` only while no studio render is running" in text
    assert "no lock across programs yet" in text


def test_the_shooting_guide_is_two_steps_in_the_skill_and_the_readme():
    guide = "1. Record one shot.\n2. Hold a deadpan stare for two seconds, then carry on naturally.\n"
    text = skill_text()
    step = text[text.index("## 2. No clip?"):text.index("## 3. The interview")]

    assert guide in step and "Walk in close" not in text and "Five minutes" not in text
    assert guide in (SKILL.parent / "README.md").read_text(encoding="utf-8")


def test_skill_md_keeps_the_clips_own_sound_and_picks_the_version_in_the_chat():
    text = " ".join(skill_text().split())
    readme = (SKILL.parent / "README.md").read_text(encoding="utf-8")

    # the video keeps the clip's sound unless the person mutes it; nothing says it is silent any more
    assert "the video is silent" not in text and "silent" not in readme.replace("is silent while the card is up", "")
    assert "`audio.original`: true (the default) keeps the clip's own sound" in text
    assert "Original sound" in text and "Original sound" in readme
    # the studio does not show the versions: the person picks in the chat and Claude patches the copy
    step7 = text[text.index("## 7. Open the studio"):text.index("## 8. On STYLE")]
    assert "which one they want" in step7 and "`copy` is the chosen version" in step7
    assert "pick a version and edit" not in text
