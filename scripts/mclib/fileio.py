"""Waiting out a file that another program has open.

On Windows a file cannot be replaced or deleted while another program has it open: the studio's
video player, a virus scan, the indexer, the waiter reading project.json. The refusal is a
PermissionError ("Access is denied") that lasts a moment, so the work is tried again a few times
before it counts as a failure. retry_locked is the one place that does it, for everything that
replaces or deletes a file the person's computer might be looking at: project.save, and the render
when it publishes its videos and clears its work folder. On macOS and Linux a file can be replaced
while it is open, and nothing here ever waits.

held_alone is the other side of it: a lock across programs, so that the studio and a render run
from the command line never work on one project's files at the same time.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

TRIES = 10  # how many times an action is tried in all
PAUSE = 0.1  # seconds between one try and the next


def retry_locked(
    action: Callable[[], T],
    path: Path,
    errors: type[OSError] | tuple[type[OSError], ...] = PermissionError,
) -> T:
    """Run action() and return what it returns. When it raises one of `errors` (PermissionError,
    unless the caller says another kind of OSError is a lock too), wait PAUSE seconds and try
    again, up to TRIES tries in all. Any other error is raised at once, as it is: a file that is
    missing or a disk that failed is not a lock.

    If the file is still locked after the last try, a PermissionError is raised with a sentence the
    person can act on, naming `path`, the file that has to be let go of, and ending with what the
    system said (on a machine where a file can be replaced while it is open, a refusal is not a
    lock at all, and that tells which); the system's own error is kept as its cause.
    """
    for attempt in range(1, TRIES + 1):
        try:
            return action()
        except errors as error:
            if attempt == TRIES:
                reason = f" The system says: {error.strerror}." if error.strerror else ""
                raise PermissionError(
                    f"{path} is in use by another program. Close whatever has it open, such as a "
                    f"video player, and try again.{reason}"
                ) from error
            time.sleep(PAUSE)


class Busy(RuntimeError):
    """Another program holds the lock that held_alone asked for."""


@contextmanager
def held_alone(lock_file: Path, busy: str) -> Iterator[None]:
    """Hold an exclusive lock on lock_file for the length of the with block, or raise Busy(busy) at
    once if another program (or another thread) holds it. The lock is the system's own, on the open
    file, so it goes with the program that held it: one killed mid-render leaves the file behind
    but not the lock. The file is created if it is missing, and left in place."""
    lock_file = Path(lock_file)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_file, "a+b")
    try:
        try:
            _lock(handle)
        except OSError:
            raise Busy(busy) from None
        try:
            yield
        finally:
            _unlock(handle)
    finally:
        handle.close()


if sys.platform == "win32":
    import msvcrt

    def _lock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # the first byte, even past the end

    def _unlock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
