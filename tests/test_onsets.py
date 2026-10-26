"""Tests for the music hits: loading a track through ffmpeg, and finding where it gets suddenly
louder.

The track is made here: three clicks of a few milliseconds over a faint hiss, written as a stereo
WAV with the wave module, so loading it runs the real ffmpeg (a stereo file also checks that it
is mixed down to mono). The rule itself is also run on signals made by hand: a loudness curve, a
beat that follows some digital silence, and a very quiet staircase.
"""

import wave

import numpy as np
import pytest

from mclib.media import MediaError
from mclib.onsets import detect_onsets, load_audio_mono

SR = 22050
HOP = 512
CLICKS = [0.5, 1.25, 2.07]  # seconds; the second one falls across a hop boundary


def write_clicks(path, times, seconds):
    """Write a stereo 16-bit WAV, `seconds` long at SR: a hiss at about -54 dBFS, and at each of
    `times` a click, 5 ms of noise that dies away in 1 ms, peaking at 0.8."""
    rng = np.random.default_rng(7)
    track = rng.standard_normal(round(seconds * SR)) * 0.002
    length = round(0.005 * SR)
    click = rng.standard_normal(length) * np.exp(-np.arange(length) / (0.001 * SR))
    click *= 0.8 / np.abs(click).max()
    for time in times:
        start = round(time * SR)
        track[start : start + length] += click
    pcm = np.rint(track * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())


def loudness_curve():
    """80 blocks of HOP samples, each of one constant value, so a block's RMS is that value and the
    flux is the curve's own steps (the change in the log of the level from block to block). Hits
    are the local maxima of the flux over mean + 2 std, which is 0.66 here. Returns the samples
    and the flux positions of the hits."""
    steps = np.zeros(79)
    # Hits: the first step, a lone rise, the first step of a two-step rise, the second step of
    # another one, and the last step
    hits = [0, 10, 30, 41, 78]
    steps[hits] = 1.0
    # The other step of each two-step rise is over the threshold too, but it is not a local
    # maximum, and one click must not be two hits
    steps[[31, 40]] = 0.97
    # A soft rise, under the threshold
    steps[20] = 0.4
    # Falls are no hits, and must not lift the threshold; one is a sound that stops dead
    steps[[3, 12, 22, 33, 43, 50]] = [-1.0, -1.0, -0.4, -1.97, -1.97, -6.0]
    # The curve starts at -2.5, so it never goes under -8.5, above the detector's floor (-9.2)
    levels = -2.5 + np.concatenate(([0.0], np.cumsum(steps)))
    return np.repeat(np.exp(levels), HOP), hits


def beat():
    """A beat made like the curve: 17 periods of 22 blocks, 8.7 s, each block one constant value.
    A bed at 0.01 has a hit at the start of every period but the first, one block long: a kick
    (0.25) on every second one and a softer snare (0.06) on the others, which rise 3.2 and 1.8 in
    log units. Returns the samples and the 16 times of the hits."""
    blocks = np.full(22 * 17, 0.01)
    blocks[22 : 22 * 17 : 22] = [0.25 if hit % 2 == 0 else 0.06 for hit in range(1, 17)]
    return np.repeat(blocks, HOP), [22 * hit * HOP / SR for hit in range(1, 17)]


@pytest.mark.filterwarnings("error")  # a mean of nothing, for one, must not warn
def test_detect_onsets_finds_clicks(tmp_path):
    # A file name with an accent and a space, as a person's songs have.
    wav = tmp_path / ("m" + chr(0xFA) + "sica t" + chr(0xE9) + "ste.wav")
    write_clicks(wav, CLICKS, seconds=3.0)

    # Loading gives mono float samples at 22050 Hz: one sample for each stereo pair in the file,
    # not two. ffmpeg's mix-down adds the channels at -3 dB each, so identical ones come out about
    # 1.4 times louder, and the clicks (0.8) are well over 0.5.
    samples = load_audio_mono(wav)
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert len(samples) == pytest.approx(3.0 * SR, abs=64)
    assert 0.5 < samples.max() < 1.5

    # One hit for each click and nothing else, each within 0.03 s of its click. The time of a hit
    # is the start of the hop where the sound first rises, so it is exact to the hop: the click at
    # sample s lands in hop s // 512, whose start is s // 512 * 512 / SR seconds.
    found = detect_onsets(samples)
    assert len(found) == len(CLICKS)
    for time, got in zip(CLICKS, found):
        assert abs(got - time) <= 0.03
        assert got == pytest.approx(round(time * SR) // HOP * HOP / SR)

    # k scales the threshold: a huge one keeps nothing, and 0 (just the mean) lets the hiss's
    # ripple through too.
    assert detect_onsets(samples, k=1000) == []
    assert len(detect_onsets(samples, k=0)) > len(CLICKS)

    # hop sets the grid the times fall on.
    fine = detect_onsets(samples, hop=256)
    assert len(fine) == len(CLICKS)
    for time, got in zip(CLICKS, fine):
        assert got == pytest.approx(round(time * SR) // 256 * 256 / SR)

    # The rule on the curve made by hand: hit i is at (i + 1) hops, and a lower bar (k=1) lets the
    # soft rise in.
    curve, hits = loudness_curve()
    assert detect_onsets(curve) == pytest.approx([(i + 1) * HOP / SR for i in hits])
    assert len(detect_onsets(curve, k=1)) == len(hits) + 1

    # Digital silence before the music (many tracks start with some) must not change what the beat
    # gives. The step from exact zeros into the music is a hit itself, but with the floor on the
    # RMS it is a modest one (4.6 in log units, where it was 18), so it does not lift the threshold
    # over the soft snares. The lead is a whole number of hops, about half a second, so the blocks
    # line up and every hit moves by exactly the lead.
    music, times = beat()
    lead = 22 * HOP / SR
    assert detect_onsets(music) == pytest.approx(times)
    after_silence = detect_onsets(np.concatenate([np.zeros(22 * HOP), music]))
    assert after_silence == pytest.approx([lead] + [time + lead for time in times])

    # The floor is 1e-4 (-80 dBFS), and anything quieter is silence to the detector. In this
    # staircase, 3e-5 to 9e-5 is a rise of 1.1 that is no hit, and the steps up to 2e-4 and 8e-4,
    # at 80 and 120 hops, are.
    quiet = np.repeat([3e-5, 9e-5, 2e-4, 8e-4], 40 * HOP)
    assert detect_onsets(quiet) == pytest.approx([80 * HOP / SR, 120 * HOP / SR])

    # Too short to have a step to measure: nothing.
    assert detect_onsets(np.zeros(0)) == []
    assert detect_onsets(np.zeros(HOP + 100)) == []
    assert detect_onsets(np.zeros(2 * HOP)) == []

    # A file ffmpeg cannot read is a MediaError that names the file, and so is one that is not
    # audio at all.
    with pytest.raises(MediaError, match="missing.wav"):
        load_audio_mono(tmp_path / "missing.wav")
    notes = tmp_path / "notes.txt"
    notes.write_text("not a song\n")
    with pytest.raises(MediaError, match="notes.txt"):
        load_audio_mono(notes)


def test_a_missing_ffmpeg_is_a_plain_error_here_too(tmp_path, monkeypatch):
    # The track is decoded by ffmpeg directly (the answer is binary), so the missing-tool sentence
    # has to be here as well: ffmpeg is named, and the doctor is where to look
    wav = tmp_path / "clicks.wav"
    write_clicks(wav, CLICKS, seconds=3.0)
    monkeypatch.setenv("PATH", str(tmp_path))  # a folder with no ffmpeg in it

    with pytest.raises(MediaError) as error:
        load_audio_mono(wav)

    assert str(error.value).startswith("ffmpeg isn't installed, or isn't on the PATH.")
    assert "python mc.py doctor" in str(error.value)
    assert "WinError" not in str(error.value) and "Errno" not in str(error.value)
