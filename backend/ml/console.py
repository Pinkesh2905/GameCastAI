"""
Console helpers.

Windows terminals often run under cp1252, where printing a stray arrow or
check mark raises UnicodeEncodeError and kills a long pipeline run. Every CLI
in this package routes its output through here so that never happens.
"""

from __future__ import annotations

import sys

# Try to give the stream a UTF-8 encoder; fall back to replacing what it
# cannot represent so a print can never abort the process.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def say(message: str = "") -> None:
    """print() that degrades instead of raising on an unencodable character."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(message.encode(encoding, "replace").decode(encoding), flush=True)


def step(message: str) -> None:
    say(f"\n== {message}")


def bullet(message: str) -> None:
    say(f"   - {message}")


def progress(done: int, total: int, label: str = "") -> None:
    """Single-line progress that stays quiet when the total is unknown."""
    if not total:
        return
    # A carriage-return bar turns into thousands of lines when the output is
    # piped to a file or another process, so only draw it for a real terminal.
    if not sys.stdout.isatty():
        return
    pct = done / total * 100
    bar_width = 24
    filled = int(bar_width * done / total)
    bar = "#" * filled + "." * (bar_width - filled)
    sys.stdout.write(f"\r   [{bar}] {pct:5.1f}%  {label}")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()
