"""Download the bundled fonts from the google/fonts repository. For maintainers only.

The skill never runs this. Its fonts/ folder is committed together with the OFL license text of
every family, so a copy of the skill works offline and renders the same everywhere. Run it again
only to refresh the fonts:

    python scripts/fetch_fonts.py

It downloads exactly the font files named in mclib.config.FONT_FILES, plus each family's OFL.txt
(saved as fonts/OFL-<folder>.txt), from https://github.com/google/fonts, and nothing else.
"""

from __future__ import annotations

import sys
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import ImageFont

from mclib.config import FONT_FILES, FONTS_DIR

BASE_URL = "https://github.com/google/fonts/raw/main/ofl"

# The folder of each family under ofl/ in google/fonts. The file names come from FONT_FILES, so
# the two cannot drift apart.
FAMILY_DIRS = {
    "marker": "patrickhandsc",
    "comic": "bangers",
    "hand": "caveat",
    "pixel": "pressstart2p",
    "terminal": "vt323",
    "label": "archivoblack",
}

MIN_FONT_BYTES = 10_000  # a smaller file is an error page or a truncated download
FONT_MAGIC = (b"\x00\x01\x00\x00", b"true", b"OTTO")  # TrueType, Apple TrueType, OpenType


def font_url(folder: str, file_name: str) -> str:
    """The raw URL of a file in ofl/<folder>, with the brackets of variable-font names encoded."""
    return f"{BASE_URL}/{folder}/{quote(file_name, safe='')}"


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "main-character-fetch-fonts"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def fetch_font(role: str) -> None:
    file_name, axes = FONT_FILES[role]
    url = font_url(FAMILY_DIRS[role], file_name)
    data = download(url)
    if len(data) < MIN_FONT_BYTES or data[:4] not in FONT_MAGIC:
        raise ValueError(f"{url} is not a font file ({len(data)} bytes)")

    target = FONTS_DIR / file_name
    target.write_bytes(data)

    # Prove the file loads, and that a variable font accepts the axes the renderer will set.
    loaded = ImageFont.truetype(str(target), 40)
    if axes:
        loaded.set_variation_by_axes(axes)
    print(f"ok  {target.name}  {len(data) / 1024:.0f} KB")


def fetch_license(role: str) -> None:
    folder = FAMILY_DIRS[role]
    url = f"{BASE_URL}/{folder}/OFL.txt"
    data = download(url)
    if b"open font license" not in data.lower():
        raise ValueError(f"{url} is not an OFL license text")

    target = FONTS_DIR / f"OFL-{folder}.txt"
    target.write_bytes(data)
    print(f"ok  {target.name}")


def main() -> int:
    if set(FAMILY_DIRS) != set(FONT_FILES):
        print("FAMILY_DIRS and mclib.config.FONT_FILES list different roles.", file=sys.stderr)
        return 1

    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for role in FONT_FILES:
            fetch_font(role)
            fetch_license(role)
    except (URLError, ValueError, OSError) as error:
        print(f"Could not fetch the fonts: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
