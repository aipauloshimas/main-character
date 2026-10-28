# License notes

What the main-character skill bundles, what it downloads, and what to check before you share or sell it.

## The skill's own code: MIT

The code, the studio page and the docs in this folder are under the MIT License, in `LICENSE`. The parts below
keep their own licenses.

## Robust Video Matting: GPL-3.0, downloaded, not bundled

- People are cut out with Robust Video Matting (RVM), https://github.com/PeterL1n/RobustVideoMatting, which is
  released under the GNU General Public License v3.0.
- The weights are not in this folder. On the first run, and only after the person agrees, the skill downloads
  `rvm_mobilenetv3_fp32.onnx` (about 15 MB) from the official v1.0.0 release to `~/.cache/rvm/`, keeps it only
  if its SHA-256 is the official file's, and runs it with onnxruntime on the CPU.
- Check the GPL-3.0 terms before you sell this skill or make it closed-source.

## Fonts: SIL Open Font License 1.1

Six fonts are bundled, unmodified, in `fonts/`. Each one's own license text sits next to it:

| Font | File | License text | Used for |
|---|---|---|---|
| Archivo Black | `ArchivoBlack-Regular.ttf` | `OFL-archivoblack.txt` | Bold labels notes |
| Bangers | `Bangers-Regular.ttf` | `OFL-bangers.txt` | Comic boxes notes and the name burst |
| Caveat | `Caveat[wght].ttf` | `OFL-caveat.txt` | Handwritten strips notes |
| Patrick Hand SC | `PatrickHandSC-Regular.ttf` | `OFL-patrickhandsc.txt` | Marker notes |
| Press Start 2P | `PressStart2P-Regular.ttf` | `OFL-pressstart2p.txt` | The Game HUD labels |
| VT323 | `VT323-Regular.ttf` | `OFL-vt323.txt` | Pixel notes and the stats panel |

`scripts/mclib/glyphs.json` lists which characters each of those fonts can draw. It holds only code point
numbers, which `scripts/make_glyphs.py` reads from the fonts' own tables: no font data is copied.

The OFL lets you bundle and share the fonts with software. They may not be sold on their own, and each copy
keeps its copyright notice and license text. Press Start 2P has a Reserved Font Name, so a modified copy of it
must take another name.

## Sound

- The shutter click is synthesized by the skill's own code (`scripts/mclib/sfx.py`). There is no sound file.
- No music is bundled, trending or otherwise, and the skill never takes any. People add the sound in their own
  app, and its rights are theirs to check.

## Installed separately, not bundled

| Part | License |
|---|---|
| numpy | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |
| onnxruntime | MIT |
| ffmpeg and ffprobe | LGPL-2.1 or later, or GPL for builds with GPL parts; the skill calls them as separate programs |
