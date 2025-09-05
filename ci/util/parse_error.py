#!/usr/bin/env python3
"""Extract errors from build and test logs.

Reads a log file (or stdin) and reports the first N lines matching known
error patterns from configure, build, and test phases. Matches are written in
Markdown, GitHub-flavored Markdown, or JSON formats.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Dict, Iterable, Iterator, List, Optional

# -----------------------------------------------------------------------------
# Regex patterns
# -----------------------------------------------------------------------------

# --- configure ---------------------------------------------------------------
CONFIGURE_PATTERNS: List[re.Pattern[str]] = [
    # CMake configure errors
    # example: "CMake Error at CMakeLists.txt:5 (message):"
    re.compile(
        r"^CMake (Error|Fatal) at (?P<file>[^:\n]+):(?P<line>\d+) \((?P<msg>[^)]+)\):",
        re.IGNORECASE,
    ),
]

# --- build ------------------------------------------------------------------
BUILD_PATTERNS: List[re.Pattern[str]] = [
    # C/C++ compiler diagnostics (clang, GCC)
    # example: "foo.cpp:3:5: error: expected ';' after expression"
    re.compile(
        r"^(?P<file>[^:\n]+):(?P<line>\d+):(?:\d+:)?\s*(?P<msg>.*\b(error|fatal)\b.*)$",
        re.IGNORECASE,
    ),
    # NVCC diagnostics
    # example: "foo.cu(10): error: identifier 'bar' is undefined"
    re.compile(
        r"^(?P<file>[^:(\n]+)\((?P<line>\d+)\):\s*(?P<msg>.*\b(error|fatal)\b.*)$",
        re.IGNORECASE,
    ),
]

# --- test -------------------------------------------------------------------
TEST_PATTERNS: List[re.Pattern[str]] = [
    # CTest summary lines
    # example: "1 - fail (Failed)"
    re.compile(
        r"^\s*(?P<line>\d+) - (?P<file>[^()]+) \((?P<msg>[^)]+)\)$",
        re.IGNORECASE,
    ),
    # lit result lines (unexpected failures/passes)
    # example: "FAIL: suite :: test (1 of 2)"
    re.compile(
        r"^(?P<msg>FAIL|XPASS|ERROR):\s+(?P<file>.+?) \(\d+ of \d+\)$",
        re.IGNORECASE,
    ),
    # lit diagnostics
    # example: "lit: /path/format.py:130: fatal: Unsupported RUN line"
    re.compile(
        r"^lit:\s+(?P<file>[^:\n]+):(?P<line>\d+):\s*(?P<msg>.*\b(error|fatal)\b.*)$",
        re.IGNORECASE,
    ),
]

# Combined list of all patterns in evaluation order
ERROR_PATTERNS: List[re.Pattern[str]] = (
    CONFIGURE_PATTERNS + BUILD_PATTERNS + TEST_PATTERNS
)


# -----------------------------------------------------------------------------
# Matching utilities
# -----------------------------------------------------------------------------


def find_match(line: str) -> Optional[re.Match[str]]:
    """Return the first regex match for *line*, or ``None``."""

    for regex in ERROR_PATTERNS:
        match = regex.match(line)
        if match:
            return match
    return None


def iter_matches(
    lines: Iterable[str], limit: Optional[int] = 1
) -> Iterator[Dict[str, str]]:
    """Yield dicts of capture groups for each matching line.

    Parameters
    ----------
    lines:
        Iterable of log lines.
    limit:
        Maximum number of matches to yield. ``None`` yields all.
    """

    count = 0
    for raw in lines:
        line = raw.rstrip("\n")
        match = find_match(line)
        if match:
            result = match.groupdict()
            result["full"] = line
            yield result
            count += 1
            if limit is not None and count >= limit:
                break


# -----------------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------------


def format_md(matches: List[Dict[str, str]]) -> str:
    """Return matches formatted as plain Markdown."""

    lines = []
    for m in matches:
        loc = f"{m.get('file', '')}:{m.get('line', '')}".strip(":")
        msg = m.get("msg", "").strip()
        lines.append(f"- `{loc}`: `{msg}`")
    return "\n".join(lines) + ("\n" if lines else "")


def format_gh(matches: List[Dict[str, str]]) -> str:
    """Return matches formatted for GitHub step summaries."""

    SUMMARY_LIMIT = 80
    sections = []
    for m in matches:
        loc = f"{m.get('file', '')}:{m.get('line', '')}".strip(":")
        msg = m.get("msg", "").strip().replace("`", "'")
        prefix = f"`{loc}`: `" if loc else "`"
        max_msg = SUMMARY_LIMIT - len(prefix) - 1
        msg_display = msg
        if len(msg_display) > max_msg:
            msg_display = msg_display[: max_msg - 3] + "..."
        summary = f"{prefix}{msg_display}`"
        body = m.get("full", "").replace("`", "'")
        sections.append(
            f"<details><summary>{summary}</summary>\n\n<pre>\n{body}\n</pre>\n</details>"
        )
    return "\n\n".join(sections) + ("\n" if sections else "")


def format_json(matches: List[Dict[str, str]]) -> str:
    """Return matches encoded as JSON."""

    return json.dumps(matches, indent=2) + "\n"


FORMATTERS = {
    "md": format_md,
    "gh": format_gh,
    "json": format_json,
}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract error lines from logs",
        epilog="Reads from <filenames> or stdin and prints the first N matches",
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Log file(s) to parse; reads stdin if omitted",
    )
    parser.add_argument(
        "-n", type=int, default=1, help="Number of matches to output (0 for all)"
    )
    parser.add_argument(
        "-o", "--output", help="Write results to FILE instead of stdout"
    )
    parser.add_argument(
        "--format", choices=FORMATTERS.keys(), default="md", help="Output format"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.filenames:
        lines: List[str] = []
        for name in args.filenames:
            with open(name, "r", errors="ignore") as f:
                lines.extend(f)
    else:
        lines = list(sys.stdin)

    limit = None if args.n == 0 else args.n
    matches = list(iter_matches(lines, limit=limit))
    formatter = FORMATTERS[args.format]
    output = formatter(matches)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
