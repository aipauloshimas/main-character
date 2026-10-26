"""Tests for the shutter click: the sound's shape, and that it is written once."""

import wave

import numpy as np
import pytest
from conftest import rms

from mclib import sfx


def read_wav(path):
    """A WAV file as (parameters, mono float samples in -1..1)."""
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        data = handle.readframes(params.nframes)
    return params, np.frombuffer(data, dtype="<i2") / 32768.0


def test_shutter_click_is_a_short_16_bit_wav(tmp_path):
    dest = tmp_path / "sounds" / "click.wav"

    returned = sfx.shutter_click(dest)

    assert returned == dest
    params, samples = read_wav(dest)
    assert (params.nchannels, params.sampwidth, params.framerate) == (1, 2, 44100)
    assert params.nframes == 5292  # 120 ms
    assert not (tmp_path / "sounds" / "click.wav.part").exists()

    # The loudest sample is 0.8 of full scale, in the first few milliseconds: the click
    assert np.abs(samples).max() == pytest.approx(0.8, abs=0.001)
    assert np.abs(samples[:int(0.003 * 44100)]).max() > 0.7

    # The noise after it dies away: each stretch is quieter than the one before, it is under 1% of
    # the click's level by its end at 62 ms, and the rest of the 120 ms is silence
    levels = [rms(samples, start, start + 0.01) for start in (0.004, 0.014, 0.024, 0.034, 0.044)]
    assert levels == sorted(levels, reverse=True)
    assert levels[0] > 0.02  # there is a body to the noise, not just the click
    assert np.abs(samples[int(0.058 * 44100):int(0.062 * 44100)]).max() < 0.01
    assert not samples[int(0.062 * 44100):].any()


def test_shutter_click_is_the_same_every_time_and_follows_the_rate(tmp_path):
    first, second = sfx.shutter_click(tmp_path / "a.wav"), sfx.shutter_click(tmp_path / "b.wav")
    assert first.read_bytes() == second.read_bytes()

    sfx.shutter_click(tmp_path / "c.wav", sr=48000)
    params, samples = read_wav(tmp_path / "c.wav")
    assert (params.framerate, params.nframes) == (48000, 5760)  # still 120 ms
    assert np.abs(samples).max() == pytest.approx(0.8, abs=0.001)


def test_shutter_click_is_written_once(tmp_path):
    dest = tmp_path / "click.wav"
    sfx.shutter_click(dest)
    dest.write_bytes(b"already here")

    assert sfx.shutter_click(dest) == dest
    assert dest.read_bytes() == b"already here"  # left as it was
