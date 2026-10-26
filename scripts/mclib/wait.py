"""The waiter: wakes Claude when the person does something in the studio.

The studio stamps project.json when the person presses Done (status.done_at) and when they ask
Claude for a style (style_request.at). wait_event polls the file until one of those stamps is
newer than a moment, and says which. Claude runs it in the background, and what it prints is what
wakes the session up.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .project import PROJECT_FILE


def wait_event(
    project_dir: Path, since: str | None = None, poll: float = 1.0, timeout: float | None = None
) -> tuple[str, str]:
    """Wait for the next event of a project and return it as ("DONE", status.done_at) or ("STYLE",
    style_request.at), with the stamp as the file has it.

    An event counts when its stamp is later than `since`, an ISO time (the stamp of the event that
    woke Claude last, so that it is not told twice), or later than the moment of the call when there
    is no since. Of two events that both count, the one that came first is returned, and the other
    is there for the next call. The file is read every `poll` seconds. With a timeout in seconds,
    TimeoutError is raised when none came in that time.
    """
    after = parse_time(since) if since else datetime.now(timezone.utc)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        found = _events_after(Path(project_dir), after)
        if found:
            _, name, stamp = min(found)
            return name, stamp
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Nothing happened in the studio in {timeout:g} seconds.")
        time.sleep(poll)


def parse_time(stamp: str) -> datetime:
    """An ISO 8601 time as an aware datetime. A time with no zone is taken to be UTC, and a "Z" is
    UTC too (which fromisoformat does not read before Python 3.11). ValueError for anything else."""
    when = datetime.fromisoformat(stamp.strip().replace("Z", "+00:00"))
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _events_after(project_dir: Path, after: datetime) -> list[tuple[datetime, str, str]]:
    """The events of the project that came after a moment, as (time, name, stamp) with DONE before
    STYLE when they came together. A file that cannot be read this time (the studio is writing it,
    on Windows) or a stamp that is not a time counts as no event; the next poll tries again."""
    try:
        text = (project_dir / PROJECT_FILE).read_text(encoding="utf-8-sig")
        project = json.loads(text)
    except (OSError, ValueError):
        return []
    stamps = [
        ("DONE", (project.get("status") or {}).get("done_at")),
        ("STYLE", (project.get("style_request") or {}).get("at")),
    ]
    found = []
    for name, stamp in stamps:
        try:
            when = parse_time(stamp)
        except (AttributeError, ValueError):  # no stamp, or one that is not a time
            continue
        if when > after:
            found.append((when, name, stamp))
    return found
