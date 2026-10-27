"""Tests for waiting out a file that another program has open.

On Windows a file cannot be replaced or deleted while a video player, a virus scan or the waiter
has it open, and the refusal is a PermissionError that lasts a moment. fileio.retry_locked tries
again, ten times 0.1 s apart, so no test here really waits: time.sleep is replaced by a list of
the pauses that were asked for.
"""

import errno
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from mclib import fileio


@pytest.fixture
def pauses(monkeypatch):
    """Every pause retry_locked takes, as a list of seconds, with no real waiting."""
    taken = []
    monkeypatch.setattr(time, "sleep", taken.append)
    return taken


def refusing(times, then="done"):
    """An action that raises PermissionError the first `times` times it is called and then returns
    `then`. Its calls attribute lists the calls."""
    calls = []

    def action():
        calls.append(1)
        if len(calls) <= times:
            raise PermissionError(errno.EACCES, "Access is denied")
        return then

    action.calls = calls
    return action


def test_retry_locked_returns_what_the_action_returns_without_waiting(pauses):
    action = refusing(0, then=42)

    assert fileio.retry_locked(action, Path("a.txt")) == 42

    assert len(action.calls) == 1 and pauses == []


def test_retry_locked_waits_out_a_lock_that_clears(pauses):
    action = refusing(3)

    assert fileio.retry_locked(action, Path("a.txt")) == "done"

    assert len(action.calls) == 4  # three refusals and the try that went through
    assert pauses == [0.1, 0.1, 0.1]  # a pause after each refusal


def test_retry_locked_gives_up_after_ten_tries_and_names_the_file(pauses, tmp_path):
    video = tmp_path / "my video.mp4"
    action = refusing(99)

    with pytest.raises(PermissionError) as stopped:
        fileio.retry_locked(action, video)

    assert len(action.calls) == 10 and pauses == [0.1] * 9  # no pause after the last try
    message = str(stopped.value)
    assert message.startswith(f"{video} is in use by another program.")
    assert "Close whatever has it open" in message and "try again" in message
    assert message.endswith("The system says: Access is denied.")
    assert isinstance(stopped.value.__cause__, PermissionError)  # the system's own error is kept


def test_retry_locked_does_not_retry_what_is_not_a_lock(pauses):
    calls = []

    def disk_failed():
        calls.append(1)
        raise OSError("disk went away")

    with pytest.raises(OSError, match="disk went away"):
        fileio.retry_locked(disk_failed, Path("a.txt"))

    def gone():
        calls.append(1)
        raise FileNotFoundError(errno.ENOENT, "No such file")

    with pytest.raises(FileNotFoundError):
        fileio.retry_locked(gone, Path("a.txt"))
    assert len(calls) == 2 and pauses == []  # each was tried once


def test_retry_locked_can_be_told_which_errors_to_wait_out(pauses, tmp_path):
    # A folder whose last file is still being let go of: Windows says the directory is not empty
    calls = []

    def not_empty_yet():
        calls.append(1)
        if len(calls) < 3:
            raise OSError(errno.ENOTEMPTY, "The directory is not empty")
        return "removed"

    assert fileio.retry_locked(not_empty_yet, tmp_path, errors=OSError) == "removed"
    assert len(calls) == 3 and pauses == [0.1, 0.1]

    # which is not waited out unless asked for
    calls.clear()
    pauses.clear()
    with pytest.raises(OSError, match="not empty"):
        fileio.retry_locked(not_empty_yet, tmp_path)
    assert len(calls) == 1 and pauses == []


# --- a lock made by the system itself ---------------------------------------------------------

only_windows = pytest.mark.skipif(
    sys.platform != "win32", reason="only Windows refuses to replace a file that is open")


@only_windows
def test_a_file_that_is_really_open_blocks_the_replace_until_it_is_closed(tmp_path):
    video, new = tmp_path / "video.mp4", tmp_path / "new.mp4"
    video.write_bytes(b"old")
    new.write_bytes(b"new")

    with open(video, "rb") as player:  # a video player has it open
        with pytest.raises(PermissionError):  # which is what Windows answers: "Access is denied"
            os.replace(new, video)
        threading.Timer(0.3, player.close).start()  # and lets go of it in a moment

        fileio.retry_locked(lambda: os.replace(new, video), video)

    assert video.read_bytes() == b"new"


@only_windows
def test_a_file_that_stays_open_is_a_plain_error_and_is_left_as_it_was(tmp_path):
    video, new = tmp_path / "video.mp4", tmp_path / "new.mp4"
    video.write_bytes(b"old")
    new.write_bytes(b"new")

    with open(video, "rb"):  # the player is still playing it ten tries later
        with pytest.raises(PermissionError, match=r"video\.mp4 is in use by another program"):
            fileio.retry_locked(lambda: os.replace(new, video), video)

    assert video.read_bytes() == b"old" and new.read_bytes() == b"new"


# --- held_alone: one program at a time ------------------------------------------------------------


def test_a_second_holder_is_turned_away_at_once(tmp_path):
    lock = tmp_path / "renders" / ".render-lock"
    with fileio.held_alone(lock, "busy now"):
        with pytest.raises(fileio.Busy, match="busy now"):
            with fileio.held_alone(lock, "busy now"):
                pass
    assert lock.exists()  # the folder and the file are made, and the file is left


def test_the_lock_is_let_go_after_the_block_even_when_it_fails(tmp_path):
    lock = tmp_path / ".render-lock"
    with pytest.raises(ZeroDivisionError):
        with fileio.held_alone(lock, "busy"):
            1 / 0
    with fileio.held_alone(lock, "busy"):
        pass


def test_another_program_holding_the_lock_turns_this_one_away(tmp_path):
    import subprocess

    lock = tmp_path / ".render-lock"
    root = Path(fileio.__file__).resolve().parent.parent
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; sys.path.insert(0, sys.argv[1]); from mclib import fileio\n"
         "with fileio.held_alone(sys.argv[2], 'busy'):\n"
         "    print('held', flush=True); sys.stdin.readline()",
         str(root), str(lock)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(fileio.Busy):
            with fileio.held_alone(lock, "busy"):
                pass
    finally:
        holder.communicate("\n", timeout=30)
    with fileio.held_alone(lock, "busy"):  # and once that program is done, it is free again
        pass
