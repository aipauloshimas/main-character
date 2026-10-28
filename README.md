# Main Character

Turn one short clip of yourself into the "this is your sign to make an introduction video" freeze-frame intro,
right inside Claude Code.

The clip freezes on you, the room turns into a styled card, and hand-drawn notes with arrows say who you are.
Three seconds later it lets go, right as you crack a smile. Claude interviews you, writes the notes, and opens a
studio page on your computer where you edit every line, pick or build a style, and render.

By Paulo (@pauloshimas). Everything runs on your computer: no paid service, no API key.

## What you get

Three files, in `main-character/<your clip>/renders/`:

| File | What it is |
|---|---|
| `<clip>-main-character.mp4` | The reel, in your clip's own format and resolution (16:9 stays 16:9, 9:16 stays 9:16), H.264 at your clip's frame rate, with your clip's own sound |
| `<clip>-cover.png` | The full card, ready to use as a thumbnail |
| `caption.txt` | A caption, a two-line title to add in Instagram or TikTok, and 5 hashtags, written from your answers |

The video keeps your clip's own sound (mute it in the studio with Original sound) and has no title, so you can
add the trending sound and your title in Instagram or TikTok. The sound is silent while the card is up, and comes
back in step with you when the video lets go. The studio's Timing tab tells you the second the freeze lands, so
you can line your sound up with it.

## Requirements

- Claude Code, on Windows or macOS.
- Python 3.10 or newer.
- ffmpeg, with ffprobe, on your PATH, built with libx264. An iPhone shoots HDR by default, so it also has to
  convert HDR: the full build on Windows, or ffmpeg 8 or newer on a Mac. The doctor checks both.
- Three Python packages: `python -m pip install numpy pillow onnxruntime` (Pillow 9.1 or newer).
- The Robust Video Matting model, about 15 MB. Claude downloads it on the first run, after you say yes.

## Install

1. Download this repository (Code > Download ZIP, or `git clone`) and put the whole folder, named
   `main-character`, into your Claude Code skills folder, so that `SKILL.md` ends up at:
   - macOS: `~/.claude/skills/main-character/SKILL.md`
   - Windows: `C:\Users\<you>\.claude\skills\main-character\SKILL.md`
2. Start Claude Code, or restart it if it was open.

## First run

1. Open Claude Code in the folder that holds your clip.
2. Type `/main-character` and give it your clip: drop the file or paste its path. No clip yet? Claude shows you
   how to shoot one.
3. Claude runs the setup check first: the doctor. It lists what is installed, what is missing, and the command
   that fixes each thing. Claude asks before it installs or downloads anything. To run it yourself:
   - macOS: `python3 ~/.claude/skills/main-character/scripts/mc.py doctor`
   - Windows (PowerShell): `python "$HOME\.claude\skills\main-character\scripts\mc.py" doctor`
4. Answer five quick questions. Claude writes three versions of your notes, you pick one in the chat, and Claude
   gives you a link to the studio.
5. In the studio, edit any line, pick or build a style, and press Render. Claude checks the video and writes
   your caption.

## Shoot your take

1. Record one shot.
2. Hold a deadpan stare for two seconds, then carry on naturally.

The clip must be 2 to 30 seconds long, with one person in it. The reel keeps your clip's own shape: a horizontal
clip stays horizontal, and a vertical one stays vertical.

## Troubleshooting

What the doctor says, and how to fix it:

| The doctor says | Fix |
|---|---|
| `Python ...: 3.10 or newer is needed` | Install Python from https://www.python.org/downloads/ (on Windows, tick "Add python.exe to PATH") |
| `numpy: not installed` (or Pillow, or onnxruntime) | `python -m pip install numpy pillow onnxruntime` |
| `Pillow ...: 9.1 or newer is needed` | `python -m pip install --upgrade pillow` |
| `ffmpeg: not found on the PATH` (or ffprobe) | Windows: `winget install Gyan.FFmpeg`. macOS: `brew install ffmpeg`. Then open a new terminal and restart Claude Code |
| `HDR clips: this ffmpeg can't convert them` | An iPhone shoots HDR by default. Windows: `winget install Gyan.FFmpeg` (the full build). macOS: ffmpeg 8 or newer (`brew install ffmpeg`), or `brew install ffmpeg-full` and put its `bin` folder first on your PATH (`brew --prefix ffmpeg-full` shows where it is) |
| `libx264: missing` | Your ffmpeg build can't write H.264 video. Install one that can: Windows `winget install Gyan.FFmpeg`, macOS `brew install ffmpeg` |
| `fonts: missing ...` | Copy the whole `main-character` folder again: its `fonts` folder is incomplete |
| `matte model: missing` | Run the doctor command above with `--download-model` at the end (about 15 MB, from the official RVM release on GitHub) |

Other messages:
- **`python` is not found** (macOS): use `python3`.
- **pip says "externally-managed-environment"** (Homebrew Python on macOS): ask Claude to set up a virtual
  environment for the skill, or install Python from python.org.
- **"The clip is ... seconds long"**: trim it to 2 to 30 seconds.
- **"Python couldn't check the download's certificate"** (macOS, Python from python.org): open the Python folder
  in Applications and run `Install Certificates.command`, then run the doctor command above with
  `--download-model` again.
- **"That's a picture, not a video"**: give it a video clip of 2 to 30 seconds, not a photo.
- **"ffmpeg isn't installed, or isn't on the PATH"**: run the doctor, which says what to install.
- **"No person found in most frames"**: check that the clip shows you, close enough to the camera.
- **"Two people are in the frame; the notes avoid only one face"**: the notes are laid out around one person, so
  one may land on the other's face. Use a take with only you in it, or drag the notes clear in the studio.
- **"This clip is HDR, and this ffmpeg build can't convert HDR"**: turn off HDR video in your camera settings,
  or install an ffmpeg that can convert it (see `HDR clips` in the table above).
- **"... is in use by another program"** (Windows): close the video player or window that has the file open,
  and try again.
- **"Text contrast ... is hard to read on a phone"**: in the studio's Style tab, pick a darker or lighter text
  color until the contrast is 3:1 or more.
- **"... has characters the Marker font can't draw"**: the video's fonts draw Latin letters with accents. Another
  alphabet, or a symbol such as a star, comes out as a gap, so the note is flagged in the studio. Write the
  line again with letters the font has.

## Licenses

The skill's own code is under the MIT License (`LICENSE`). The cutout model, Robust Video Matting, is GPL-3.0,
and it is downloaded, not bundled. The fonts are under the
SIL Open Font License, and the shutter sound is synthesized. The details are in `LICENSE-NOTES.md`.
