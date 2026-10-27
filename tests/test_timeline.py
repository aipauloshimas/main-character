"""Tests for the timeline: clip-to-output mapping, card frame states and music snapping."""

from fractions import Fraction

import pytest

from mclib.config import TIMING_DEFAULTS
from mclib.timeline import build_timeline, card_states, snap_to_onsets

ALL_LAYERS = frozenset({"bottom", "top", "arrows"})
NO_LAYERS = frozenset()


def timing_with(**changes):
    """The default timing with a few values changed."""
    return {**TIMING_DEFAULTS, **changes}


def kinds(timeline):
    return [segment.kind for segment in timeline.segments]


def frame_by_frame(states):
    """One state per card frame, after checking that the states tile the card with no gaps."""
    frames = []
    for state in states:
        assert state.start == len(frames)
        assert state.frames > 0
        frames.extend([state] * state.frames)
    return frames


# --- build_timeline: the head ---------------------------------------------------------------


def test_head_trims_long_clip():
    # A 6 s clip frozen at 3.0 s: the head is the 62 frames (2.07 s) that end on the freeze, and the
    # tail the rest of the clip, 3.0 s.
    tl = build_timeline(6.0, 3.0, timing_with())

    assert kinds(tl) == ["clip", "card", "clip"]
    head, card, tail = tl.segments
    assert head.frames == 62
    assert head.src_start == pytest.approx(3.0 - 62 / 30)  # 0.9333
    assert head.speed == 1.0
    assert card.frames == 90
    assert tail.frames == 90
    assert tail.src_start == pytest.approx(3.0)
    assert tail.speed == 1.0
    assert tl.total_frames == 242
    assert tl.freeze_out == pytest.approx(62 / 30)
    assert tl.release_out == pytest.approx(62 / 30 + 90 / 30)
    assert tl.warning is None

    # A freeze a hair short of 62 frames still fills the head, and the clip never starts before 0.
    edge = build_timeline(6.0, 62 / 30 - 1e-9, timing_with())
    assert kinds(edge) == ["clip", "card", "clip"]
    assert edge.segments[0].frames == 62
    assert edge.segments[0].src_start == 0.0

    # Every duration follows fps, not a fixed 30.
    at60 = build_timeline(6.0, 3.0, timing_with(), fps=60)
    assert [segment.frames for segment in at60.segments] == [124, 180, 180]
    assert at60.total_frames == 484
    assert at60.freeze_out == pytest.approx(124 / 60)
    assert at60.release_out == pytest.approx(304 / 60)


def test_head_speed_fits_three_second_clip():
    # A 3 s clip frozen on its last frame has just 2.07 s of head, unless the head plays at 1.45x.
    tl = build_timeline(3.0, 3.0, timing_with(head_speed=1.45, tail=0.0))

    assert kinds(tl) == ["clip", "card"]
    clip = tl.segments[0]
    assert clip.frames == 62
    assert clip.src_start == pytest.approx(0.0033, abs=1e-3)
    assert clip.speed == 1.45
    # The 62 frames take 62 * 1.45 / 30 s of source and end on the freeze frame.
    assert clip.src_start + clip.frames * clip.speed / 30 == pytest.approx(3.0)
    assert tl.warning is None

    # Only the head is sped up: the tail still plays at normal speed.
    with_tail = build_timeline(6.0, 3.0, timing_with(head_speed=1.45))
    assert [s.speed for s in with_tail.segments if s.kind == "clip"] == [1.45, 1.0]


def test_short_head_holds_first_frame():
    # Only 1.5 s of clip before the freeze: hold the first frame for the 17 frames that are missing.
    tl = build_timeline(4.0, 1.5, timing_with())

    assert kinds(tl) == ["fill", "clip", "card", "clip"]
    fill, clip = tl.segments[:2]
    assert fill.frames == 17
    assert clip.frames == 45
    assert clip.src_start == 0.0
    assert clip.speed == 1.0
    assert tl.freeze_out == pytest.approx(62 / 30)
    assert tl.warning is None

    # Exactly 1.0 s of fill (30 frames) is still allowed.
    limit = build_timeline(4.0, 32 / 30, timing_with())
    assert kinds(limit) == ["fill", "clip", "card", "clip"]
    assert limit.segments[0].frames == 30
    assert limit.segments[1].frames == 32
    assert limit.warning is None


def test_too_short_head_moves_freeze():
    # 0.8 s of clip is 24 frames, 38 short of the 62 wanted: more fill than allowed, so the
    # freeze lands early instead.
    tl = build_timeline(4.0, 0.8, timing_with())

    assert kinds(tl) == ["clip", "card", "clip"]
    head = tl.segments[0]
    assert head.frames == 24
    assert head.src_start == 0.0
    assert tl.freeze_out == pytest.approx(0.8)
    assert tl.release_out == pytest.approx(0.8 + 3.0)
    assert "0.80 s" in tl.warning

    # One frame past the fill limit is already too short.
    over = build_timeline(4.0, 31 / 30, timing_with())
    assert kinds(over) == ["clip", "card", "clip"]
    assert over.segments[0].frames == 31
    assert "1.03 s" in over.warning

    # max_fill is honored: with no fill allowed, 17 missing frames are too many.
    no_fill = build_timeline(4.0, 1.5, timing_with(), max_fill=0.0)
    assert kinds(no_fill) == ["clip", "card", "clip"]
    assert "1.50 s" in no_fill.warning

    # A freeze on the very first frame leaves no head at all, and no empty segment is kept.
    first = build_timeline(4.0, 0.0, timing_with())
    assert kinds(first) == ["card", "clip"]
    assert first.freeze_out == 0.0
    assert "0.00 s" in first.warning


# --- build_timeline: the tail ---------------------------------------------------------------


def test_tail_cut_by_clip_end():
    # Only 0.5 s of clip is left after the freeze, so the 1.2 s tail shrinks to 15 frames.
    tl = build_timeline(6.0, 5.5, timing_with())

    tail = tl.segments[-1]
    assert tail.kind == "clip"
    assert tail.frames == 15
    assert tail.src_start == pytest.approx(5.5)
    assert tail.speed == 1.0
    assert tl.total_frames == 62 + 90 + 15

    # Float noise must not cost a frame: 3.3 - 3.0 is 0.2999999999999998 s, which is 9 frames.
    noisy = build_timeline(3.3, 3.0, timing_with())
    assert noisy.segments[-1].frames == 9

    # A freeze on the last frame leaves nothing to play, so there is no tail segment.
    last = build_timeline(6.0, 6.0, timing_with())
    assert kinds(last) == ["clip", "card"]


def test_release_from_skips_ahead():
    # The clip resumes 0.5 s past the freeze frame instead of on it, and plays its last 2.5 s.
    tl = build_timeline(6.0, 3.0, timing_with(release_from=0.5))

    tail = tl.segments[-1]
    assert tail.kind == "clip"
    assert tail.src_start == pytest.approx(3.5)
    assert tail.frames == 75
    assert tail.speed == 1.0
    assert tl.total_frames == 227  # the head and the card do not change


# --- card_states ----------------------------------------------------------------------------


# --- build_timeline at the timings the studio's sliders can set -----------------------------------
#
# The sliders have ranges (config.TIMING_RANGES) and the default timing sits in the middle of them, so
# the tests above mostly run at the default. These four are worked out by hand, each at a different end
# of the ranges: (clip length, the freeze frame's time, the timing changed, the rate) and the segments
# that follow, as (kind, frames, where in the clip it starts, speed).

SLIDER_CASES = [
    pytest.param(
        # head_speed 0.75 (the slowest), the freeze at 1.0 s and a hold of 4 s. The head wants 30 frames
        # (1.0 x 30) and the clip has 3.0 x 30 / 0.75 = 120, so it takes the 30 that end on the freeze:
        # 30 x 0.75 / 30 = 0.75 s of clip, from 3.0 - 0.75 = 2.25. The card is 4 x 30 = 120 frames and
        # the tail the rest of the clip, 3.0 s = 90 frames from the freeze. 240 in all.
        6.0, 3.0, dict(head_speed=0.75, freeze_at=1.0, hold=4.0), 30,
        [("clip", 30, 2.25, 0.75), ("card", 120, 0.0, 1.0), ("clip", 90, 3.0, 1.0)], 1.0, 5.0, None,
        id="slow head, long hold",
    ),
    pytest.param(
        # head_speed 1.5 (the fastest) and the freeze at 5.0 s (the latest): 150 frames wanted, and the
        # 6.0 s of clip before the freeze give 6.0 x 30 / 1.5 = 120. The 30 that are missing are exactly
        # the 1.0 s the first frame may be held, so the head is 30 held frames and then all 120. The
        # card is the default 90 and the tail the 2.0 s of clip left, 60 frames. 300 in all.
        8.0, 6.0, dict(head_speed=1.5, freeze_at=5.0), 30,
        [("fill", 30, 0.0, 1.0), ("clip", 120, 0.0, 1.5), ("card", 90, 0.0, 1.0), ("clip", 60, 6.0, 1.0)],
        5.0, 8.0, None,
        id="fast head that needs the first frame held",
    ),
    pytest.param(
        # release_from 1.0 (the most) and a tail of 3.0 (the most): the head is the default 62 frames
        # (the clip has 2.5 x 30 = 75), from 2.5 - 62 / 30 = 0.4333. The card is the shortest hold, 30
        # frames. The tail starts 1.0 s past the freeze, at 3.5 s, and asks for 3.0 s of a clip that has
        # 6.0 - 3.5 = 2.5 s left: 75 frames, cut by the end of the clip. 62 + 30 + 75 = 167.
        6.0, 2.5, dict(release_from=1.0, tail=3.0, hold=1.0), 30,
        [("clip", 62, 2.5 - 62 / 30, 1.0), ("card", 30, 0.0, 1.0), ("clip", 75, 3.5, 1.0)],
        62 / 30, 62 / 30 + 1.0, None,
        id="release skipped ahead, tail cut by the clip's end",
    ),
    pytest.param(
        # at 24 fps, with no tail at all (0.0), the longest hold (6.0 s) and the earliest freeze (0.5 s):
        # 12 head frames from 1.0 - 12 / 24 = 0.5, a card of 6 x 24 = 144 frames, and a tail of no
        # frames, which is not a segment. 156 in all, and the card ends 6.5 s in.
        3.0, 1.0, dict(tail=0.0, hold=6.0, freeze_at=0.5), 24,
        [("clip", 12, 0.5, 1.0), ("card", 144, 0.0, 1.0)], 0.5, 6.5, None,
        id="no tail, longest hold, earliest freeze, 24 fps",
    ),
]


@pytest.mark.parametrize("duration, freeze, changes, fps, segments, freeze_out, release_out, warning", SLIDER_CASES)
def test_build_timeline_at_timings_the_sliders_can_set(
    duration, freeze, changes, fps, segments, freeze_out, release_out, warning
):
    tl = build_timeline(duration, freeze, timing_with(**changes), fps=fps)

    assert [(s.kind, s.frames) for s in tl.segments] == [(kind, frames) for kind, frames, _, _ in segments]
    for segment, (_, _, start, speed) in zip(tl.segments, segments):
        assert segment.src_start == pytest.approx(start, abs=1e-9)
        assert segment.speed == speed
    assert tl.total_frames == sum(frames for _, frames, _, _ in segments)
    assert tl.freeze_out == pytest.approx(freeze_out)
    assert tl.release_out == pytest.approx(release_out)
    assert tl.warning == warning


def test_card_states_staggered():
    states = card_states(90, "staggered", True)
    frames = frame_by_frame(states)

    assert len(frames) == 90
    assert sum(state.frames for state in states) == 90
    assert frames[0].visible == frozenset({"bottom"})
    assert frames[1].visible == frozenset({"bottom"})
    assert frames[2].visible == frozenset({"bottom", "top"})  # the top notes join at frame 2
    assert frames[4].visible == frozenset({"bottom", "top"})
    assert frames[5].visible == ALL_LAYERS  # the arrows join at frame 5
    assert frames[83].visible == ALL_LAYERS
    for number in range(84, 90):  # everything goes off for the last 6 frames
        assert frames[number].visible == NO_LAYERS

    # Neighbors that look the same are one state, so only the 15 real changes are kept: the
    # reveal at frames 2 and 5, the boil every 7 frames, and the exit at frame 84.
    assert len(states) == 15

    def look(state):
        return state.visible, state.variant, state.flash

    for before, after in zip(states, states[1:]):
        assert look(before) != look(after)


def test_card_states_boil_cycles():
    frames = frame_by_frame(card_states(90, "staggered", True))
    assert [frames[number].variant for number in (0, 7, 14, 21)] == [0, 1, 2, 0]
    assert {frames[number].variant for number in range(0, 7)} == {0}
    assert {frames[number].variant for number in range(7, 14)} == {1}

    still = frame_by_frame(card_states(90, "staggered", False))
    assert {state.variant for state in still} == {0}

    # The boil keeps cycling while the card goes out: with 93 frames, 87-92 are empty.
    leaving = frame_by_frame(card_states(93, "staggered", True))[87:]
    assert {state.visible for state in leaving} == {NO_LAYERS}
    assert [state.variant for state in leaving] == [0, 0, 0, 0, 1, 1]


def test_card_states_all_reveal():
    frames = frame_by_frame(card_states(90, "all", True))

    assert frames[0].visible == ALL_LAYERS
    assert frames[83].visible == ALL_LAYERS
    assert frames[84].visible == NO_LAYERS  # "all" still goes out for the last 6 frames

    with pytest.raises(ValueError):
        card_states(90, "sideways", True)


def test_card_states_shutter_flash():
    frames = frame_by_frame(card_states(90, "staggered", True, shutter=True))

    assert frames[0].flash == 0.85
    assert frames[1].flash == 0.40
    assert frames[2].flash == 0.0
    assert all(state.flash == 0.0 for state in frames[2:])

    # The flash rides on top: it changes neither what is visible nor the boil variant, and it
    # is off unless asked for.
    plain = frame_by_frame(card_states(90, "staggered", True))
    assert all(state.flash == 0.0 for state in plain)
    for flashed, unflashed in zip(frames, plain):
        assert (flashed.visible, flashed.variant) == (unflashed.visible, unflashed.variant)


# --- snap_to_onsets -------------------------------------------------------------------------


def test_snap_to_onsets():
    # The freeze snaps to 2.05 and the release (2.05 + 3.0 = 5.05) to 5.06.
    assert snap_to_onsets(2.07, 3.0, [0.63, 2.05, 3.16, 5.06]) == pytest.approx((2.05, 3.01))

    # Of the onsets in reach, the nearest wins: not the first, not the last.
    assert snap_to_onsets(2.07, 3.0, [1.80, 2.10, 2.30, 5.0, 5.3]) == pytest.approx((2.10, 2.90))

    # The release looks around the snapped freeze, not the original one: 1.60 + 3.0 = 4.60 has
    # no onset in reach, even though 5.55 is within 0.5 s of 2.07 + 3.0.
    assert snap_to_onsets(2.07, 3.0, [1.60, 5.55]) == pytest.approx((1.60, 3.0))

    # An onset exactly at the edge of the window counts, and the window can be changed.
    assert snap_to_onsets(2.0, 3.0, [2.5]) == (2.5, 3.0)
    assert snap_to_onsets(2.0, 3.0, [2.5], window=0.25) == (2.0, 3.0)


def test_snap_without_near_onset_keeps_values():
    assert snap_to_onsets(2.07, 3.0, [0.5]) == pytest.approx((2.07, 3.0))
    # With nothing to snap to, the values come back exactly, with no float noise in the hold.
    assert snap_to_onsets(2.07, 3.0, []) == (2.07, 3.0)


# --- other frame rates ----------------------------------------------------------------------


def test_timeline_is_counted_in_frames_of_the_clips_rate():
    # A 4 s clip at 24 fps frozen on frame 48 (2.0 s): 50 head frames are wanted (2.07 s) and the
    # clip has 48, so its first frame is held for 2. The card is 3 s, the tail the 2 s of clip left.
    tl = build_timeline(4.0, 48 / 24, timing_with(), fps=24)

    assert kinds(tl) == ["fill", "clip", "card", "clip"]
    assert [segment.frames for segment in tl.segments] == [2, 48, 72, 48]
    assert tl.total_frames == 170
    assert tl.freeze_out == pytest.approx(50 / 24)
    assert tl.release_out == pytest.approx(122 / 24)
    assert tl.warning is None
    assert tl.segments[-1].src_start == pytest.approx(2.0)

    # An exact rate (a Fraction, as media.output_fps gives it) is the same as its float, and the
    # times come out as plain floats
    ntsc = build_timeline(4.0, 48 * 1001 / 24000, timing_with(), fps=Fraction(24000, 1001))
    same = build_timeline(4.0, 48 * 1001 / 24000, timing_with(), fps=24000 / 1001)
    assert ntsc == same
    assert isinstance(ntsc.freeze_out, float) and isinstance(ntsc.release_out, float)
    assert [segment.frames for segment in ntsc.segments] == [2, 48, 72, 48]  # the clip's last frame, at 3.96 s, too


def test_card_states_keep_their_times_at_other_rates():
    # At 30 fps it is exactly what it was, however the rate is written
    base = card_states(90, "staggered", True, shutter=True)
    for rate in (30, 30.0, Fraction(30), Fraction(30000, 1001)):  # 29.97 fps counts the same
        assert card_states(90, "staggered", True, shutter=True, fps=rate) == base

    # At 24 fps the counts are the same times: 2 -> 2, 5 -> 4, the last 6 -> 5, a boil of 7 -> 6
    states = card_states(72, "staggered", True, fps=24)
    frames = frame_by_frame(states)
    assert len(frames) == 72
    assert frames[1].visible == frozenset({"bottom"})
    assert frames[2].visible == frozenset({"bottom", "top"})
    assert frames[3].visible == frozenset({"bottom", "top"})
    assert frames[4].visible == ALL_LAYERS
    assert frames[66].visible == ALL_LAYERS
    assert all(frames[number].visible == NO_LAYERS for number in range(67, 72))
    assert [frames[number].variant for number in (0, 5, 6, 11, 12, 18)] == [0, 0, 1, 1, 2, 0]
    # neighbors that look the same are still one state
    for before, after in zip(states, states[1:]):
        assert (before.visible, before.variant, before.flash) != (
            after.visible, after.variant, after.flash)
    # and 23.976 fps counts the same as 24
    assert card_states(72, "staggered", True, fps=Fraction(24000, 1001)) == states

    # At 60 fps, twice as many frames: 4, 10, the last 12, a boil of 14
    frames = frame_by_frame(card_states(180, "staggered", True, fps=60))
    assert frames[3].visible == frozenset({"bottom"})
    assert frames[4].visible == frozenset({"bottom", "top"})
    assert frames[9].visible == frozenset({"bottom", "top"})
    assert frames[10].visible == ALL_LAYERS
    assert frames[167].visible == ALL_LAYERS
    assert all(frames[number].visible == NO_LAYERS for number in range(168, 180))
    assert [frames[number].variant for number in (0, 13, 14, 27, 28, 42)] == [0, 0, 1, 1, 2, 0]


def test_flash_keeps_its_two_levels_at_other_rates():
    def flashes(hold, fps):
        return [state.flash for state in frame_by_frame(card_states(hold, "staggered", True, True, fps))]

    # Each level lasts one 30 fps frame's worth of frames: 1 at 24 fps, 2 at 60 fps
    assert flashes(72, 24)[:4] == [0.85, 0.40, 0.0, 0.0]
    assert flashes(180, 60)[:6] == [0.85, 0.85, 0.40, 0.40, 0.0, 0.0]
    assert flashes(90, 30)[:3] == [0.85, 0.40, 0.0]
    assert set(flashes(180, 60)[4:]) == {0.0}

    # and it changes neither what is visible nor the boil
    flashed = frame_by_frame(card_states(180, "staggered", True, True, 60))
    plain = frame_by_frame(card_states(180, "staggered", True, False, 60))
    assert [(s.visible, s.variant) for s in flashed] == [(s.visible, s.variant) for s in plain]


def test_the_default_tail_plays_the_rest_of_the_clip():
    # After the card the clip goes on to its last frame unless the tail is set shorter, whatever the
    # clip's length: a person's wink a second after the freeze must not be cut. A 4.04 s
    # clip at 24 fps (97 frames) frozen on frame 71 has frames 71 to 96 left: 26 frames.
    tl = build_timeline(97 / 24, 71 / 24, timing_with(), 24)
    tail = tl.segments[-1]
    assert (tail.kind, tail.frames, tail.src_start) == ("clip", 26, 71 / 24)

    # The longest clip the skill takes, 30 s, frozen at 2 s: the 28 s left all play.
    assert build_timeline(30.0, 2.0, timing_with()).segments[-1].frames == 28 * 30
