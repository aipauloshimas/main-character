"""The shutter click, made here so the skill ships no sound file.

shutter_click writes it as a small WAV for render.py to place on the freeze. Nothing in it is
random from one run to the next: the same call always writes the same samples.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np

CLICK_SECONDS = 0.120  # the whole file
_TICK = 0.002  # the click itself, a sharp tick
_BURST = 0.060  # the noise that follows it, which dies away well before the end
_PEAK = 0.8  # the loudest sample, as a share of full scale
_SEED = 7


def shutter_click(dest: Path, sr: int = 44100) -> Path:
    """Write the shutter click to dest and return dest. It is written once: if dest is there
    already, it is left as it is.

    The sound is 120 ms of mono, 16-bit WAV at sr Hz. It opens with a 2 ms click, a bright tick that
    fades out over those 2 ms, on top of a 60 ms burst of noise that decays smoothly (to under 1%
    of where it began), then silence to the end. The loudest sample is 0.8 of full scale. The file
    is written to a .part file first and renamed, so a reader never finds half of it.
    """
    dest = Path(dest)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    total = round(CLICK_SECONDS * sr)
    sound = np.zeros(total)

    tick = round(_TICK * sr)
    seconds = np.arange(tick) / sr
    sound[:tick] += np.sin(2 * np.pi * 1800 * seconds) * np.exp(-seconds / 0.0006)

    burst = round(_BURST * sr)
    seconds = np.arange(burst) / sr
    hiss = np.diff(np.random.default_rng(_SEED).standard_normal(burst + 1))  # differenced: crisp
    sound[:burst] += 0.35 * hiss / np.abs(hiss).max() * np.exp(-seconds / 0.012)

    pcm = np.rint(sound / np.abs(sound).max() * _PEAK * 32767).astype("<i2")
    part = dest.with_name(dest.name + ".part")
    try:
        with wave.open(str(part), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sr)
            handle.writeframes(pcm.tobytes())
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)  # gone already after a good write
    return dest
