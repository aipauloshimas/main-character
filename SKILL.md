---
name: main-character
description: 'Use when someone wants the "this is your sign to make an introduction video" freeze-frame intro made from their own clip: the video freezes on them, the room turns into a styled card, and hand-drawn notes with arrows describe them. Triggers on /main-character, "freeze-frame intro", "this is your sign to make an introduction video", "main character intro", "character intro edit", or a dropped clip of the person together with a request for that edit. PT examples for reliability: "edita minha intro estilo main character", "faz aquela intro que congela". Editing only: it interviews the person, writes the notes, opens a local studio and renders the clip in its own format (16:9 stays 16:9, 9:16 stays 9:16). NOT for other video edits (captions, cuts, color grades, multicam, b-roll), for making footage, or for posting.'
---

# /main-character: the freeze-frame intro

The person brings one clip of themselves walking up to the camera. You check the setup, interview them, write
the notes, check the card, and open a local studio where they edit, pick a style and render. In the reel, the
clip freezes on them, the room turns into a styled card, hand-drawn notes with arrows describe them, and the
video lets go again. The video keeps the clip's own sound, which is silent while the card is up. The skill only
edits: it never makes footage, ships music or posts.

## How to work

- `<skill dir>` is the folder that holds this SKILL.md (Claude Code names it as the skill's base directory).
  Commands look like `python "<skill dir>/scripts/mc.py" card "<project>"`. Where `python` is missing (common on
  macOS), use `python3`.
- `<project>` is the folder `init` prints on its `PROJECT` line: `./main-character/<clip name>` in the working
  folder by default. Quote every path; paths can hold spaces and accents.
- Output lines start with a word, such as `PROJECT`, `WARNING` or `URL`. A failure prints `ERROR <sentence>` and
  exits non-zero: say the sentence in plain words, fix what it names, and run the command again.
- `init`, `serve` and `wait` run in the background (Bash `run_in_background`). When one exits, its output wakes
  you. Don't poll.
- **Every change to `project.json` goes through `patch`.** Never edit the file by hand, and never patch `clip`
  or `status`.
  1. Write a JSON merge patch with your Write tool to `<project>/patch.json`, so nothing goes through shell
     quoting. Objects merge, a key set to `null` is deleted, and any other value replaces what was there.
  2. Run `python "<skill dir>/scripts/mc.py" patch "<project>" --file "<project>/patch.json"`. It validates and
     saves, then prints a `WARNING <sentence>` line per warning and `CONTRAST <x.x>`. On `ERROR`, nothing was
     saved: fix the patch and run it again. It is safe while the studio is open: the server reads the file
     fresh on every request.
- Talk to the person in their language. The on-screen copy is in English unless they ask for another language.
- If the person's own instructions (their CLAUDE.md) add review gates, such as a storyboard before any edit or a
  go-ahead word before the final render, follow them on top of these steps.

## 1. Check the setup

Run `python "<skill dir>/scripts/mc.py" doctor`. It prints one line per item and, when something is missing, the
commands that fix it under "To fix:" (exit code 1). Ask before each install or download, and run it only after a
yes:
- **Packages:** `python -m pip install numpy pillow onnxruntime`, about 30 to 60 MB from PyPI (Pillow must be
  9.1 or newer). If pip refuses with "externally-managed-environment" (Homebrew Python on macOS), offer a
  virtual environment: `python3 -m venv ~/.main-character/venv`, then use its `bin/python` in every command.
- **ffmpeg and ffprobe:** the install line the doctor prints (winget's `Gyan.FFmpeg` on Windows, Homebrew's
  `ffmpeg` on macOS), often 100 MB or more. Claude Code may need a restart to find them on the PATH.
- **What ffmpeg can do:** an iPhone shoots HDR by default, so the doctor also checks `HDR clips` (zscale, or
  ffmpeg 8 or newer) and `libx264`. When one says it can't, its "To fix:" line names the build to install: the
  full build on Windows; on a Mac, ffmpeg 8 or newer, or `brew install ffmpeg-full` with its `bin` folder first
  on the PATH.
- **The matte model:** ask "The cutout needs the Robust Video Matting model: about 15 MB, from the official RVM
  release on GitHub (PeterL1n/RobustVideoMatting, v1.0.0), saved to ~/.cache/rvm. OK to download it?" After a
  yes: `python "<skill dir>/scripts/mc.py" doctor --download-model`. The download is checked against the official
  file's SHA-256. If it says Python couldn't check the certificate (Python from python.org on a Mac), have them run
  "Install Certificates.command" in the Python folder, then run the command again. After a no, stop: nothing can be
  cut out without it.
- **Fonts missing:** the folder was copied incompletely. Ask them to copy the whole skill folder again.
- **No Python at all:** the doctor can't run. Ask them to install Python 3.10 or newer from python.org (on
  Windows, with "Add python.exe to PATH" ticked).

Run the doctor again until it ends with "Everything is ready."

## 2. No clip? Show the shooting guide and stop

A clip is one video file, 2 to 30 seconds long, with one person in it: a path in the message or a file they
dropped. Without one, send this guide and stop:

1. Record one shot.
2. Hold a deadpan stare for two seconds, then carry on naturally.

End with: "Run /main-character again with your best take."

## 3. The interview

If `main-character/<clip name>/project.json` already exists in the working folder, this clip was started before.
Ask whether to reopen it (run step 4, which keeps their lines, then go to step 7) or to write new lines.

Send the five questions in one message, word for word, with the closing line:

> 1. What do you do? One line, the way you'd say it to someone at a party.
> 2. What's your "but…"? An honest confession, a flaw, or a funny contradiction. This is the line that gets the laugh.
> 3. One oddly specific detail that's only yours: a number, a habit, an object. "4 coffees before 9am" beats "loves coffee".
> 4. Your @ handle, and the name you'd like on screen.
> 5. The tone: laughing at yourself, confident, chaotic, mysterious, or something else?
>
> Stuck on any of these? Say 'give me options'.

- Someone writing in another language gets the same five questions, faithfully translated.
- Start step 4 in the background in the same turn, so the cutout runs while they type.
- On "give me options": offer three or four options for that question, built from what you know about them.
  They stay options: nothing is true until the person picks or confirms it.
- An answer missing: ask once more for just that one, or offer options.

## 4. Make the project

Run `python "<skill dir>/scripts/mc.py" init "<clip>"`. It cuts the frames, cuts the person out of every frame
with RVM (a minute or more), and suggests freeze frames. It prints:
- `PROJECT <folder>`: this is `<project>`.
- `WARNING <sentence>`: tell the person in plain words ("I couldn't find you in most frames. Right clip?").
- `FREEZE <frame> SUGGESTED <frames>`: the freeze frame, then the suggested frames, best first (for step 6).

The video keeps the clip's own format and resolution: 16:9 stays 16:9 and 9:16 stays 9:16, full frame, with no
bars and no title. A clip under 1080 wide is scaled up to 1080, and the render says so in a note. Common errors: a
clip outside 2 to 30 seconds (ask them to trim it), a photo instead of a video (ask for a clip), a missing model
or a missing ffmpeg (back to step 1), an HDR clip that this ffmpeg can't convert (the sentence says what to do).

## 5. Write the copy

Write versions A, B and C: three different jokes in the person's tone, not three rewordings. A version is the four
notes, the name and the extras. The video has no title: the person adds it in Instagram or TikTok, and you suggest
one in `caption.txt` (step 9). Run this checklist on each version before anyone sees it:

- [ ] Top left (`tl`) says what they do, in their own words.
- [ ] Bottom left (`bl`) starts with "But" and is the joke: true, specific, about 34 characters at most.
- [ ] Bottom right (`br`) is the oddly specific detail, about 30 characters at most.
- [ ] Top right (`tr`) is the handle.
- [ ] The phrasing is subjectless, so no pronouns: "But overthinks every 6-second reel".
- [ ] Nothing is stated as true that the person didn't say or confirm. Anything you added is offered as an option.
- [ ] Nothing generic that could belong to anyone ("passionate creator who loves making content").
- [ ] The extras, in every version: the burst word (the name on screen by default; keep it short), an emphasis
      word taken from the "but…" line, and three stats in game language as `[label, 0 to 10]` with short labels.
      The confession becomes the funny stat (`["Overthinking", 10]`). `extras.bubble` (a few words, 14 characters at
      most) is for the Speech bubble extra only; leave it `null` unless the person turned that extra on.
- [ ] Natural case (each notes style applies its own capitals), no emojis, and within the soft limits: `tl` 30,
      `tr` 24, `bl` 34, `br` 30 and `name` 16.
- [ ] The on-screen copy uses the Latin alphabet. Accents are fine where the font has them, and `card`'s `FLAG`
      lines tell you where it doesn't: Arabic, Chinese and Hebrew are in no font, Cyrillic and Greek in only a
      few, and symbols such as stars are mostly missing. `patch` warns with the characters and the font
      ("has characters the Marker font can't draw"), and `card` flags the note `cut`.

Pick the built-in style that fits the person (see "Style recipes"). Bedroom Pop, the trend's own look, is the
fallback. Comic Pop suits loud, confident or chaotic; Scrapbook, crafty and warm; Player Select, gamers and tech.

Write it all in one patch: `versions` (A, B and C), `copy` (the same as A), `picked` "A", `style` (the whole
recipe) and `style_source` (its id). The studio does not show the versions: the person picks one in the chat
(step 7), and you put it in `copy` with `patch`. One version looks like this:

```json
{"notes": {"tl": "<what they do>", "tr": "<@handle>", "bl": "But <the confession>", "br": "<the detail>"},
 "name": "<name on screen>",
 "extras": {"burst": "<burst word>", "emphasis": "<a word of the bl line>",
            "stats": [["<label>", 9], ["<label>", 10], ["<label>", 2]]}}
```

Shorten any line that a soft-limit `WARNING` names. Then save the five answers, in the person's words, to
`<project>/interview.md` with Write.

## 6. Check the card before the person sees it

Run `python "<skill dir>/scripts/mc.py" card "<project>"`.
- It draws the card on the freeze frame and prints the path of `<project>/card.png`. The card is the plan size,
  1080 wide and as tall as the clip's shape makes it: 1080x608 for 16:9, 1080x1920 for 9:16.
- It prints a `FLAG <key> cut` or `FLAG <key> clash` line for each note that doesn't fit:
  - **cut:** its text didn't fit at the 36 px minimum, or has a character its font can't draw;
  - **clash:** it touches the face, overlaps another note or sits too close to one (notes keep a gap between
    them). On a 9:16 card it also means what Instagram covers: the top, where the person adds the title, the
    caption at the bottom and the buttons at the right.

Look at the card whole, then at 1:1 crops of the hair, the edge of the face and the hands. Those are 480x480 crops,
so x goes up to 600 and y up to H−480, where H is the card's height:
`ffmpeg -v error -y -i "<project>/card.png" -vf "crop=480:480:<x>:<y>" "<project>/check-1.png"`

- **A `FLAG` line, or a note on the face.** Shorten that note: patch it, then run `card` again. Never use a smaller
  font; notes are never drawn under 36 px. In the studio the person can also drag it. A `cut` that shortening
  doesn't fix is a character the font can't draw: write the line again with letters the font has.
- **A `FLAG <key> clash` that shorter copy can't fix**, because the note has no room in its corner, is fixed by
  moving the note. Look at the card, pick a free spot, patch `{"positions": {"<key>": [x, y]}}` (the box's top-left
  in card pixels), and run `card` again. Never put a note on the face. `<key>` is `tl`, `tr`, `bl`, `br` or
  `burst`: any other key is refused.
- **`NOTE The Game HUD has no room here; pick another hero if it covers something.`** The Game HUD's two labels
  have no `FLAG` line, because nothing can pin them. Look at the card: if a label covers the face or a note, tell
  the person and offer another hero (patch the style's `hero` to `none`, `burst` or `stats`).
- **The cutout edge is wrong at 1:1** (a halo, chewed hair or fingers, holes, background pulled in).
  1. A thin light rim alone: patch `{"matte": {"edge": -1}}` first, going down to -2.
  2. Otherwise re-cut the frames around the freeze at a higher detail:
     `python "<skill dir>/scripts/mc.py" refine "<project>" --detail high`. Run `card` and look again. Run
     `refine` only while no studio render is running: there is no lock across programs yet.
  3. Still wrong: the same with `--detail max`.
  4. Still wrong: patch `{"freeze": {"frame": <the next SUGGESTED frame>}}`, run `card`, and tell the person why in
     one line.

## 7. Open the studio, start the waiter, and stop

1. Run `python "<skill dir>/scripts/mc.py" serve "<project>"` in the background, unless it already runs. It
   prints `URL http://127.0.0.1:<port>/` and serves until stopped.
2. Run `python "<skill dir>/scripts/mc.py" wait "<project>"` in the background, before the person has the URL,
   so no click is missed. It exits on the next event: `STYLE <stamp>` (step 8) or `DONE <stamp>` (step 9).
3. Show the person the three jokes, as versions A, B and C with all four notes of each, and ask which one they
   want: A, B or C, or a mix of lines. Then stop: end your turn. Don't poll.
4. When they answer, put their pick in the live copy with `patch`: `copy` is the chosen version (all of it, with
   a `null` for each extra the version doesn't have, so nothing of the earlier copy stays) and `picked` is its
   letter. A mix of lines is a patch of just those lines. Run `card` again if the lines changed.
5. Give the person the URL and how the studio works: edit any line; pick a style or build one, or describe one
   and press "Ask Claude"; drag a note to move it; the Timing tab has the Original sound switch, which mutes the
   clip's own sound; press Render when it looks right. Then stop again.

A change the person asks for in the chat goes through `patch`; then ask them to reload the page.

## 8. On STYLE: answer "Ask Claude"

1. Read `style_request.text` in `<project>/project.json`. If `style_request` is gone or already answered when you
   read it, skip it and run `wait` again with `--since "<the STYLE stamp>"` (the person cancelled, or the event is
   old). If `style_request.image` is set, look at that image (a file in `<project>/refs/`).
2. Write the whole recipe to `<project>/style.json` with your Write tool, then run
   `python "<skill dir>/scripts/mc.py" style "<project>" --file "<project>/style.json" --answers "<the STYLE stamp>"`.
   For example:
   ```json
   {"id": "custom", "name": "Sunset Arcade", "bg": "night", "edge": "neon", "notes": "pixel",
    "arrows": "none", "hero": "hud", "extras": {"tape": false, "stars": true, "grain": false},
    "palette": {"dark": "#1a0b2e", "light": "#ff6ad5", "tex": "#26c6da", "edge": "#ffd1f7", "ink": "#fff3a8"}}
   ```
   - Use only the ids of "Style recipes". `hero` holds one id, so there is never more than one hero.
   - `style` validates the recipe and prints `CONTRAST <x.x>`.
   - It refuses a recipe under 3.0 with `ERROR`. Change the colors your notes style is measured on and run it
     again.
   - On success it saves the recipe as the custom style and marks the request answered. The waiting page
     refreshes by itself.
   - `--answers` is the stamp `wait` printed on its `STYLE` line. The request is marked answered only if it is
     still that one: if the person asked again or cancelled while you worked, the style is saved all the same and
     it prints `NOTE The request changed; the page shows the newest one.` The next `wait` brings the new request.
   - For a look the layers can't make (a cinematic grade, 3D lettering), get as close as the options allow and say
     what differs. Every style keeps the sticker cutout and the playful notes.
3. Run `card` and look. If the new style flags a note, tell the person and offer a shorter line. Don't rewrite
   their lines unasked.
4. Explain the picks in the chat: one line per layer (background, edge, notes, arrows, hero, extras), and one for
   the colors with their contrast.
5. Run `python "<skill dir>/scripts/mc.py" wait "<project>" --since "<the STYLE stamp>"` in the background, then
   stop.

## 9. On DONE: check the render

1. `status.outputs` in project.json lists the files inside `<project>`: `renders/<clip>-main-character.mp4`,
   `renders/<clip>-cover.png`, and `renders/<clip>-with-music.mp4` when they added their own track.
2. Look at three stills of the video. The render records its own moments, in seconds, as `status.freeze_out` and
   `status.release_out`. Take:
   - the freeze, at `freeze_out` + 0.3;
   - the middle of the hold;
   - the release, at `release_out` + 0.3.

   `ffmpeg -v error -y -ss <seconds> -i "<project>/renders/<clip>-main-character.mp4" -frames:v 1 "<project>/check-freeze.png"`

   Check:
   - the face is clear;
   - nothing is cut;
   - the edge is clean at 1:1. The stills are at the video's own resolution, the clip's, not the card's 1080, so
     crop as in step 6 with x up to W−480 and y up to H−480, where W and H are the still's width and height
     (`ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "<video>"` prints
     them);
   - the release catches the smile starting.
3. Offer at most two tweaks, each with its setting and value (see "Settings"). Patch only after a yes, then ask
   them to reload and press Render again. If the studio is closed, run
   `python "<skill dir>/scripts/mc.py" render "<project>"` yourself, never while a studio render runs.
4. Write `<project>/renders/caption.txt` with Write, in this order:
   - a short caption in the person's voice, built only from their answers;
   - a line "Title to add in Instagram:", then the title as two lines;
   - one line of 5 hashtags (two for the trend, three for their niche).

   On a later DONE, rewrite it only if the lines changed. The title is not in the video: the person adds it in
   Instagram or TikTok, like the sound. Its rules:
   - Line one is "this is your sign to".
   - Line two is the trend's own ("make an introduction video of yourself") or a twist that answers the
     confession.
   - Keep both lines lowercase and about 40 characters at most each. The person types them as written.
5. Tell them where the files are, and that the video keeps the clip's own sound (silent over the card, back in
   step when the video lets go; Original sound in the Timing tab mutes it) and has no title: they add the
   trend's sound and the title from `caption.txt` in Instagram or TikTok. With the default timing, the freeze lands at 2.07 s, on
   the sound's hit.
6. If the studio is still open, run `wait` again with `--since "<the DONE stamp>"` in the background. When they
   are finished, stop the `serve` and `wait` background tasks.

## Style recipes

A recipe has an `id` and a `name` (text), one id per layer, the little extras and a palette. Every key is
required except the eleven newer extras (see below), and the ids are exact:

| Layer | Ids, with the studio's labels |
|---|---|
| `bg` | `dots` Duotone + dots, `comic` Comic print, `riso` Riso on paper, `night` Night scanlines, `flat` Flat color, `mono` Black & white |
| `edge` | `handcut` Hand-cut sticker, `comic` Comic double line, `paper` Paper + shadow, `neon` Neon glow, `thin` Thin line, `none` No edge |
| `notes` | `marker` Marker, `comic` Comic boxes, `hand` Handwritten strips, `pixel` Pixel, `label` Bold labels |
| `arrows` | `curly` Curly, `doodle` Doodle, `straight` Line + dot, `none` None |
| `hero` | `none`, `burst` Name burst, `stats` Stats panel (it takes the handle note's place), `hud` Game HUD |
| `extras` | `{"tape": …, "stars": …, "grain": …}`, each true or false, and any of the eleven newer ones below |

The little extras are toggles under "Little extras" in the studio's Style tab. The eleven newer ones are off in the
four built-ins and in a new project (the built-ins turn on only the ones in their table below). A style
saved before the eleven newer ones, or a recipe that leaves them out, still works: they are off.

| Extra | What it draws |
|---|---|
| `tape` Tape, `stars` Stars | tape across the notes' top corners, and outline stars beside the face |
| `grain` Film grain | soft-light grain over the whole stage |
| `hearts` Hearts | two or three hand-drawn outline hearts beside the head (with the arrows) |
| `circle` Scribble circle | a loop drawn twice around the emphasis word (`copy.extras.emphasis`) where it is in a note; nothing without one (with that note) |
| `crown` Doodle crown | an outline crown, tilted, just above the head (with the arrows); left out when the head touches the top of the picture |
| `sparkles` Sparkles | three to five four-point sparkles near the person's outline (with the arrows) |
| `motion` Motion lines | three short manga strokes beside the top of the head (with the arrows) |
| `confetti` Confetti | 20 to 40 small paper rectangles and dots in the palette's colors, on the background only, from the first frame; it re-scatters a little with each boil picture |
| `rec` REC timestamp | a red dot and "REC" in one top corner and "00:00:02" in the other, in the pixel font with a dark outline, the whole card; under the title room on a 9:16 card |
| `vhs` VHS lines | thin scanlines over the whole stage and a 1 to 2 px red and cyan offset |
| `leak` Light leak | a soft orange and red glow bleeding in from the side away from the head |
| `nametag` Name tag | a "HELLO my name is" sticker with `copy.name` on the chest, just under the face; nothing without a name (with the notes) |
| `bubble` Speech bubble | a speech bubble beside the mouth with `copy.extras.bubble` (14 characters at most, a soft limit; nothing when empty; with the arrows) |

The drawn ones keep off the face, the notes, the arrows, the hero's boxes, the 9:16 title room, bottom band and
rail, the tape, the stars and each other. One that finds no room is left out, never drawn over something.

`palette` holds exactly five `#rrggbb` colors: `dark` (background dark), `light` (background light), `tex`
(texture), `edge` (cutout edge) and `ink` (text). The seven presets:

| Preset | dark | light | tex | edge | ink |
|---|---|---|---|---|---|
| Magenta | `#7d2a72` | `#f2a9de` | `#4c0f45` | `#f5df98` | `#f7e8b0` |
| Lagoon | `#1d4f5c` | `#a3e6dc` | `#0b2e36` | `#ffe9a8` | `#fff1c4` |
| Tangerine | `#8a3a12` | `#ffb98a` | `#4a1a06` | `#fff0c8` | `#fff6dc` |
| Comic | `#111111` | `#ffd21f` | `#e8261c` | `#ffffff` | `#111111` |
| Riso | `#2f4fa8` | `#efe4cb` | `#2f4fa8` | `#fbf8f0` | `#1b1b1b` |
| Neon | `#07061c` | `#5a24c9` | `#3ef0ff` | `#9ff8ff` | `#ffe14d` |
| Mono | `#141414` | `#e9e6df` | `#141414` | `#ffe600` | `#ffe600` |

The four built-ins (extras not listed are false):

| id | name | bg | edge | notes | arrows | hero | extras on | palette |
|---|---|---|---|---|---|---|---|---|
| `bedroom-pop` | Bedroom Pop | dots | handcut | marker | curly | none | grain | Magenta |
| `comic-pop` | Comic Pop | comic | comic | comic | curly | burst | none | Comic |
| `scrapbook` | Scrapbook | riso | paper | hand | doodle | none | tape, stars | Riso |
| `player-select` | Player Select | night | neon | pixel | none | stats | none | Neon |

The contrast guard: at least 3.0 (WCAG), measured where the lettering sits.

| notes | Measured between | What passes |
|---|---|---|
| `marker` | `ink` and `dark` (its glow) | a light ink on a deep dark |
| `comic` | `#111111` and `light` mixed 45% toward white | a light `light` |
| `hand` | `ink` and the paper strip, `#fffdf5` | a dark ink |
| `label` | `dark` and `edge` (the pill) | a deep dark on a light edge |
| `pixel` | `ink` and the background's average color | a bright ink on `night` or `mono`; a dark ink on `dots`, `comic`, `riso` or `flat` |

## Settings

- `timing`: `freeze_at` 0.5 to 5.0 s (default 2.07), `hold` 1.0 to 6.0 (3.0), `head_speed` 0.75 to 1.5 (1.0),
  `tail` 0.0 to 30.0 (30.0: the rest of the clip plays to its last frame; set it lower to end sooner), `release_from` 0.0 to 1.0 (0.0; skips dead time after the freeze, to reach the
  smile), `reveal` "staggered" or "all", `boil` and `shutter` true or false.
- `audio.original`: true (the default) keeps the clip's own sound in the video, silent over the card and back in
  step with the picture when the video lets go; false mutes it. `audio.track` is the person's music, which
  only the `-with-music` file has.
- `freeze.frame`: a frame of the clip, from 0. `matte.edge`: a whole number from -4 to 4 (below 0 shrinks the
  cutout, above 0 softens it).

## Rules

- **People are cut out with RVM only**, the model this skill runs. Never u2net, rembg, a remove-background tool
  or difference keying, even when another skill suggests one.
- **Nothing is stated as fact that the person didn't say.** Your ideas are options until they confirm them.
- **Ask before any download or install**, naming what it is, its size and its source.
- **No trending audio.** Never download or add one: the person adds the sound in the app. A track they upload
  in the studio themselves is fine.
- **Keep the person's own words.** Trim to fit; never change what they mean. Emojis are stripped: the video's
  fonts can't draw them, and `patch` removes them with a warning. Any other character a font can't draw is
  warned about and flagged, never removed: ask the person what to write instead.
