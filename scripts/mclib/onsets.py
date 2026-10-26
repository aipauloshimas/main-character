"""Music hits: where a track gets suddenly louder, so the freeze and release can land on a beat.

load_audio_mono decodes a track to mono samples, and detect_onsets finds the hits in them, as a
list of times in seconds. The project keeps that list (audio.onsets), and timeline.snap_to_onsets
moves the freeze and the release onto the nearest hit within config.SNAP_WINDOW.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .media import MediaError, missing_tool

SAMPLE_RATE = 22050  # Hz; tracks are loaded at this rate
HOP = 512  # samples in a step of the analysis, 23 ms at 22050 Hz
PEAK_K = 2.0  # a hit has to rise this many standard deviations over the mean rise
RMS_FLOOR = 1e-4  # -80 dBFS: a step quieter than this counts as silence (see detect_onsets)


def load_audio_mono(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """The first audio track of any file ffmpeg can read, as mono float32 samples at sr Hz. ffmpeg
    mixes the channels down by adding them at -3 dB each, so a loud stereo track can go a little
    past 1; the level does not matter to detect_onsets. Raises MediaError when ffmpeg cannot read
    audio from the file, or is not on this computer."""
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    # Not media.run: the answer is binary, and that helper reads output as text
    try:
        done = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True)
    except FileNotFoundError:
        raise missing_tool("ffmpeg") from None
    if done.returncode != 0:
        message = done.stderr.decode("utf-8", errors="replace").strip()
        raise MediaError(f"ffmpeg could not read the audio of {Path(path).name}: {message}")
    return np.frombuffer(done.stdout, dtype="<f4").astype(np.float32)


def detect_onsets(
    x: np.ndarray, sr: int = SAMPLE_RATE, hop: int = HOP, k: float = PEAK_K
) -> list[float]:
    """The times in seconds, in order, where the samples x get suddenly louder. x is mono at sr Hz,
    scaled as load_audio_mono gives it (full scale is about 1).

    The samples are cut into steps of hop samples. Each step's loudness is the log of its RMS, but
    not under RMS_FLOOR, and the flux is how far that rose from one step to the next, with falls
    counted as 0. The floor makes digital silence and a very faint hiss the same level. Without
    it a track that starts with some silence would open with one huge rise, from exact zeros into
    the music (18 or so in log units), which would lift the mean and the standard deviation of the
    whole track and hide its soft hits. A hit is a local maximum of the flux that is over its mean
    plus k times its standard deviation. Hit i is between steps i and i + 1, so it is given the
    time the louder step begins: (i + 1) * hop / sr. A track shorter than two steps has none.
    """
    steps = len(x) // hop
    if steps < 2:
        return []
    blocks = np.asarray(x, np.float32)[: steps * hop].reshape(steps, hop)
    energy = np.einsum("ij,ij->i", blocks, blocks, dtype=np.float64)  # no float64 copy of x
    loudness = np.log(np.maximum(np.sqrt(energy / hop), RMS_FLOOR))
    flux = np.maximum(np.diff(loudness), 0.0)

    # A local maximum is over the step before it and at least as high as the one after, so a run
    # of equal steps counts once, at its start. The ends have one neighbor, and the missing one
    # counts as minus infinity.
    before = np.concatenate(([-np.inf], flux[:-1]))
    after = np.concatenate((flux[1:], [-np.inf]))
    threshold = flux.mean() + k * flux.std()
    peaks = np.flatnonzero((flux > threshold) & (flux > before) & (flux >= after))
    return [float((i + 1) * hop / sr) for i in peaks]
