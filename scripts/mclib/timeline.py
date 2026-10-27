"""Timeline math: where each part of the clip lands in the output, and how the card animates.

Everything is counted in output frames, so the encoder can follow it frame by frame. Nothing in
this module reads a file or calls ffmpeg.

The output has three parts, in order:

1. The head: the stretch of clip that ends on the freeze frame (a "clip" segment). When the clip
   has too little footage before the freeze, its first frame is held first, for at most
   max_fill seconds (a "fill" segment). When even that is not enough, the head is whatever
   exists, the freeze lands earlier than asked, and the timeline carries a warning.
2. The card: the frozen frame under the illustrated card, for hold seconds (a "card" segment).
3. The tail: the clip playing on from the freeze, for up to tail seconds or as much as the clip
   has left (a "clip" segment).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import groupby

from .config import FPS, MAX_FILL, SNAP_WINDOW, STAGGER

_REVEAL_MODES = ("staggered", "all")

# Added before a frame count is rounded down. A length that is a whole number of frames on paper
# can come out a hair under it as a float, for example (3.3 - 3.0) * 30 is 8.999999999999995, and
# without this it would lose a frame.
_EPSILON = 1e-6


@dataclass(frozen=True)
class Segment:
    """One run of output frames.

    kind is "fill" (the clip's first frame, held), "clip" (the clip playing) or "card" (the
    frozen frame under the card). For a clip, output frame i shows the clip at
    src_start + i * speed / fps seconds. A fill or a card has no source time to follow.
    """

    kind: str
    frames: int
    src_start: float = 0.0
    speed: float = 1.0


@dataclass(frozen=True)
class Timeline:
    """The whole output: its segments in order, and when the card starts and ends.

    freeze_out and release_out are output times in seconds, and total_frames is the sum of every
    segment's frames. warning is a sentence for the person when the freeze could not land where
    they asked, otherwise None.
    """

    segments: tuple[Segment, ...]
    freeze_out: float
    release_out: float
    total_frames: int
    warning: str | None = None


@dataclass(frozen=True)
class CardState:
    """A run of card frames that all look the same.

    start is a card frame number, counted from the first frame of the card. visible is the set
    of note layers that are on, from "bottom", "top" and "arrows". variant is the boil picture
    (0, 1 or 2), and flash is the level of white laid over the frame (0.0 for none).
    """

    start: int
    frames: int
    visible: frozenset[str]
    variant: int
    flash: float = 0.0


def build_timeline(
    clip_duration: float,
    freeze_time: float,
    timing: dict,
    fps: float = FPS,
    max_fill: float = MAX_FILL,
) -> Timeline:
    """Lay the clip out as head, card and tail, counted in frames of fps: the clip's own rate, as
    media.output_fps gives it (a Fraction is fine).

    clip_duration is the clip's length and freeze_time the time of the frozen frame in the clip,
    both in seconds. timing needs freeze_at (where the freeze should land in the output), hold,
    head_speed, tail and release_from.
    """
    fps = float(fps)  # a Fraction would make the times below Fractions too
    speed = timing["head_speed"]
    wanted = round(timing["freeze_at"] * fps)  # head frames wanted before the freeze
    avail = math.floor(freeze_time * fps / speed + _EPSILON)  # head frames the clip can give
    warning = None

    if avail >= wanted:
        # Enough clip: play the stretch that ends on the freeze frame. The clamp only absorbs
        # float noise, so a clip starting a hair before 0 is a clip starting at 0.
        start = max(0.0, freeze_time - wanted * speed / fps)
        parts = [Segment("clip", wanted, start, speed)]
    elif wanted - avail <= round(max_fill * fps):
        # A little short: hold the first frame for the missing frames, then play all there is.
        parts = [Segment("fill", wanted - avail), Segment("clip", avail, 0.0, speed)]
    else:
        # Too short to fill: play what there is, and the freeze lands early.
        parts = [Segment("clip", avail, 0.0, speed)]
        warning = f"The freeze lands at {avail / fps:.2f} s, so the trend's sound will not line up."
    head_frames = sum(part.frames for part in parts)

    card_frames = round(timing["hold"] * fps)
    parts.append(Segment("card", card_frames))

    tail_start = min(freeze_time + timing["release_from"], clip_duration)
    # The tail asked for, but no more frames than the clip has left: every frame that starts before
    # the clip ends, so a clip whose length is not a whole number of frames keeps its last one.
    frames_left = math.ceil((clip_duration - tail_start) * fps - _EPSILON)
    tail_frames = min(math.floor(timing["tail"] * fps + _EPSILON), max(frames_left, 0))
    parts.append(Segment("clip", tail_frames, tail_start))

    # A freeze on the first frame, for one, leaves a head of 0 frames. An empty segment would
    # only give the encoder something to trip on, so those are dropped.
    segments = tuple(part for part in parts if part.frames > 0)
    freeze_out = head_frames / fps
    return Timeline(
        segments=segments,
        freeze_out=freeze_out,
        release_out=freeze_out + card_frames / fps,
        total_frames=sum(segment.frames for segment in segments),
        warning=warning,
    )


def card_states(
    hold_frames: int, reveal: str, boil: bool, shutter: bool = False, fps: float = FPS
) -> list[CardState]:
    """How the card looks on each of its hold_frames frames, as runs of identical frames.

    The bottom notes are on from the first frame. With reveal "staggered" the top notes join at
    card frame 2 and the arrows at frame 5, and with "all" every layer is on from the start.
    Whatever the reveal, everything is off for the last 6 frames. With boil on, the picture
    cycles through three variants, one every 7 frames, and it keeps cycling while the card goes
    out. With shutter on, a white flash (0.85, then 0.40) rides on the first two frames without
    changing what is visible. Consecutive frames merge into one state only when their visible
    layers, variant and flash are all equal.

    Those frame counts are for 30 fps. At another fps they keep their times (see _stagger_at), so
    the card looks the same on a 24 fps clip as on a 30 fps one.
    """
    if reveal not in _REVEAL_MODES:
        raise ValueError(f'reveal must be "staggered" or "all", got {reveal!r}')
    stagger = _stagger_at(fps)
    out_from = hold_frames - stagger["out_before"]
    looks = [
        _card_look(frame, out_from, reveal, boil, shutter, stagger) for frame in range(hold_frames)
    ]

    states = []
    start = 0
    for look, run in groupby(looks):
        frames = len(list(run))
        states.append(CardState(start, frames, *look))
        start += frames
    return states


def _stagger_at(fps: float) -> dict:
    """config.STAGGER for a clip at fps frames per second.

    STAGGER counts card frames at 30 fps (config.FPS). Every count n becomes round(n * fps / 30),
    which is the same time in frames of this rate, so the notes arrive, the picture boils and the
    notes leave at the same moments whatever the rate. Each level of the white flash lasts as many
    frames as one 30 fps frame comes to, and at least 1. At 30 fps the result is STAGGER itself.
    """
    scale = fps / FPS

    def frames(count: int) -> int:
        return round(count * scale)

    return {
        "top_after": frames(STAGGER["top_after"]),
        "arrows_after": frames(STAGGER["arrows_after"]),
        "out_before": frames(STAGGER["out_before"]),
        "boil_every": max(1, frames(STAGGER["boil_every"])),
        "boil_variants": STAGGER["boil_variants"],
        "flash": STAGGER["flash"],
        "flash_frames": max(1, round(scale)),
    }


def _card_look(
    frame: int, out_from: int, reveal: str, boil: bool, shutter: bool, stagger: dict
) -> tuple[frozenset[str], int, float]:
    """What one card frame looks like, as (visible, variant, flash). stagger is _stagger_at's."""
    if frame >= out_from:
        visible = frozenset()
    else:
        layers = {"bottom"}
        if reveal == "all" or frame >= stagger["top_after"]:
            layers.add("top")
        if reveal == "all" or frame >= stagger["arrows_after"]:
            layers.add("arrows")
        visible = frozenset(layers)
    variant = (frame // stagger["boil_every"]) % stagger["boil_variants"] if boil else 0
    level = frame // stagger["flash_frames"]  # which of the flash's levels this frame is on
    flash = stagger["flash"][level] if shutter and level < len(stagger["flash"]) else 0.0
    return visible, variant, flash


def snap_to_onsets(
    freeze_at: float, hold: float, onsets: list[float], window: float = SNAP_WINDOW
) -> tuple[float, float]:
    """Move the freeze and the release onto music onsets, where one is close enough.

    The freeze snaps to the onset nearest freeze_at, if one is within window seconds. The
    release, which is the snapped freeze plus hold, then snaps the same way. An end with no onset
    in reach stays where it was. Returns (freeze, hold) to use: the freeze, and the distance from
    it to the release.
    """
    freeze = _nearest(freeze_at, onsets, window)
    if freeze is None:
        freeze = freeze_at
    release = _nearest(freeze + hold, onsets, window)
    if release is None:
        return freeze, hold  # the release did not move, so the hold is exactly what came in
    return freeze, release - freeze


def _nearest(target: float, onsets: list[float], window: float) -> float | None:
    """The onset closest to target, or None if none is within window seconds of it."""
    reachable = [onset for onset in onsets if abs(onset - target) <= window]
    return min(reachable, key=lambda onset: abs(onset - target), default=None)
