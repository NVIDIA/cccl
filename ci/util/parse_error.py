#!/usr/bin/env python3
"""Extract errors from configure/build/test logs.

Reads one or more logs (or stdin) and reports the first N lines matching
known error patterns. Output can be JSON or Markdown; JSON is used as the
canonical representation and the Markdown is derived from it to keep both
formats in sync.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Regex patterns
# -----------------------------------------------------------------------------

# --- configure ---------------------------------------------------------------
@dataclass
class PatternSpec:
    error: re.Pattern[str]
    # Optional context anchors: search backwards to context_begin (inclusive),
    # then forwards to context_end (inclusive), surrounding the error line.
    context_begin: Optional[re.Pattern[str]] = None
    context_end: Optional[re.Pattern[str]] = None


CONFIGURE_SPECS: List[PatternSpec] = [
    # CMake configure errors
    # example: "CMake Error at CMakeLists.txt:5 (message):"
    PatternSpec(
        error=re.compile(
            r"^\s*CMake (Error|Fatal) at (?P<file>[^:\n]+):(?P<line>\d+) \((?P<msg>[^)]+)\):",
            re.IGNORECASE,
        ),
        context_end=re.compile(
            r"(Configuring incomplete, errors occurred!|See also )",
            re.IGNORECASE,
        ),
    ),
]

# --- build ------------------------------------------------------------------
BUILD_SPECS: List[PatternSpec] = [
    # C/C++ compiler diagnostics (clang, GCC)
    # example: "foo.cpp:3:5: error: expected ';' after expression"
    PatternSpec(
        error=re.compile(
            r"^(?P<file>[^:\n]+):(?P<line>\d+):(?:\d+:)?\s*(?P<msg>.*\b(error|fatal)\b.*)$",
            re.IGNORECASE,
        ),
    ),
    # NVCC diagnostics
    # example: "foo.cu(10): error: identifier 'bar' is undefined"
    PatternSpec(
        error=re.compile(
            r"^(?P<file>[^:(\n]+)\((?P<line>\d+)\):\s*(?P<msg>.*\b(error|fatal)\b.*)$",
            re.IGNORECASE,
        ),
        # Context capture for nvcc in Ninja output
        context_begin=re.compile(r"^\s*FAILED:\s+"),
        context_end=re.compile(r"^\s*\d+\s+error detected in the compilation of "),
    ),
]

# --- LIT --------------------------------------------------------------------
LIT_SPECS: List[PatternSpec] = [
    # lit result lines (unexpected failures/passes)
    # example: "FAIL: suite :: test (1 of 2)"
    PatternSpec(
        error=re.compile(
            r"^\s*(?P<msg>FAIL|XPASS|ERROR):\s+(?:.+?\s+::\s+)?(?P<file>.+?) \(\d+ of \d+\)$",
            re.IGNORECASE,
        ),
        # lit context: between a FAILED banner and the closing banner
        # Example begin: ******************** TEST 'suite :: test' FAILED ********************
        # Example end:   ********************
        context_begin=re.compile(r"^\s*\*{6,}\s*TEST '.*' FAILED\s*\*{6,}\s*$"),
        context_end=re.compile(r"^\s*\*{6,}\s*$"),
    ),
    # lit diagnostics
    # example: "lit: /path/format.py:130: fatal: Unsupported RUN line"
    PatternSpec(
        error=re.compile(
            r"^lit:\s+(?P<file>[^:\n]+):(?P<line>\d+):\s*(?P<msg>.*\b(error|fatal)\b.*)$",
            re.IGNORECASE,
        ),
    ),
]

# --- test -------------------------------------------------------------------
TEST_SPECS: List[PatternSpec] = [
    # CTest summary lines
    # example: "1 - fail (Failed)"
    PatternSpec(
        error=re.compile(
            r"^\s*(?P<line>\d+) - (?P<file>[^()]+) \((?P<msg>[^)]+)\)$",
            re.IGNORECASE,
        ),
    ),
]

# Combined list of all patterns in evaluation order
# Prioritize lit/test matches over raw compiler diagnostics,
# as some tests may produce build errors in their output.
ERROR_SPECS: List[PatternSpec] = CONFIGURE_SPECS + LIT_SPECS + TEST_SPECS + BUILD_SPECS


# -----------------------------------------------------------------------------
# Matching utilities
# -----------------------------------------------------------------------------

_BUILD_PREFIX_RE = re.compile(r"^.*?/cccl/build/[^/]+/[^/]+/")
_SRC_PREFIX_RE = re.compile(r"^.*?/cccl/")


def _normalize_file(path: str) -> str:
    """Return a path relative to the repo root when possible."""
    p = _BUILD_PREFIX_RE.sub("", path)
    p = _SRC_PREFIX_RE.sub("", p)
    return p


def find_spec_and_match(line: str) -> Optional[tuple[PatternSpec, re.Match[str]]]:
    """Return the first (spec, match) for the line, or None."""

    for spec in ERROR_SPECS:
        match = spec.error.match(line)
        if match:
            return spec, match
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
    raw_lines = list(lines)
    for i, raw in enumerate(raw_lines):
        line = raw.rstrip("\n")
        found = find_spec_and_match(line)
        if found:
            spec, match = found
            result = match.groupdict()
            # Preserve absolute path as captured, and provide normalized + basename variants.
            abs_filepath = (result.get("file", "") or "").strip()
            rel_filepath = _normalize_file(abs_filepath).strip() if abs_filepath else ""
            filename = (
                os.path.basename(rel_filepath or abs_filepath)
                if (rel_filepath or abs_filepath)
                else ""
            )

            # For legacy consumers, keep 'file' as the normalized relative path if available.
            if abs_filepath:
                result["file"] = (rel_filepath or abs_filepath).strip()
            result["abs_filepath"] = abs_filepath
            result["rel_filepath"] = rel_filepath
            result["filename"] = filename

            result["full"] = line

            # Build contextual body: preamble (inclusive) .. error line .. epilogue (inclusive)
            preamble_lines: List[str] = []
            start_idx: Optional[int] = None
            forward_begin_idx: Optional[int] = None
            if spec.context_begin is not None:
                for j in range(i - 1, -1, -1):
                    if spec.context_begin.search(raw_lines[j].rstrip("\n")):
                        start_idx = j
                        break
                if start_idx is None:
                    for j in range(i + 1, len(raw_lines)):
                        if spec.context_begin.search(raw_lines[j].rstrip("\n")):
                            forward_begin_idx = j
                            break
                if start_idx is not None:
                    preamble_lines = [s.rstrip("\n") for s in raw_lines[start_idx:i]]

            end_idx: Optional[int] = None
            epilogue_lines: List[str] = []
            if spec.context_end is not None:
                for k in range(i + 1, len(raw_lines)):
                    if spec.context_end.search(raw_lines[k].rstrip("\n")):
                        end_idx = k
                        break
                if end_idx is not None and (start_idx is not None or forward_begin_idx is None):
                    epilogue_lines = [s.rstrip("\n") for s in raw_lines[i + 1 : end_idx + 1]]

            # Build context: if a forward-begin block exists (e.g., LIT FAILED section),
            # capture exactly from that banner through the closing marker.
            if forward_begin_idx is not None and end_idx is not None:
                context_parts = [s.rstrip("\n") for s in raw_lines[forward_begin_idx : end_idx + 1]]
            else:
                # Insert an alert emoji line just before the error line in context.
                # Indent the emoji to match the error line's leading whitespace
                leading_ws = re.match(r"^(\s*)", line).group(1) if line else ""
                alert_line = f"{leading_ws}⚠️"
                context_parts = preamble_lines + [alert_line] + [line] + epilogue_lines
            result["context"] = "\n".join(context_parts)
            yield result
            count += 1
            if limit is not None and count >= limit:
                break


# -----------------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------------


SUMMARY_LIMIT = 80


def build_summary(m: Dict[str, str], limit: int = SUMMARY_LIMIT) -> str:
    """Build a compact, consistent summary used across all formats.

    Uses the basename filename (when available) for location to keep summaries
    short and stable. The summary is formatted as:

        `file:line`: `truncated message`

    The total summary length is capped by ``limit``.
    """

    loc_file = m.get("filename", "") or m.get("file", "")
    loc = f"{loc_file}:{m.get('line', '')}".strip(":")
    msg = m.get("msg", "").strip().replace("`", "'")
    prefix = f"`{loc}`: `" if loc else "`"
    max_msg = max(8, limit - len(prefix) - 1)
    msg_display = msg
    if len(msg_display) > max_msg:
        msg_display = msg_display[: max_msg - 3] + "..."
    return f"{prefix}{msg_display}`"


def format_json(matches: List[Dict[str, str]]) -> str:
    """Return matches encoded as JSON, with extras for consumers.

    Adds:
      - "location": "file:line" convenience field (normalized "file")
      - "summary":  shared truncated summary identical to MD/GH
    """

    out: List[Dict[str, str]] = []
    for m in matches:
        # Use normalized file for location; keep consistent with previous behavior.
        loc = f"{m.get('file', '')}:{m.get('line', '')}".strip(":")
        summary = build_summary(m)
        enriched = dict(m)
        # Rename convenience field to 'location' (was 'loc').
        enriched["location"] = loc
        enriched["summary"] = summary
        out.append(enriched)
    return json.dumps(out, indent=2) + "\n"


def format_md(matches: List[Dict[str, str]]) -> str:
    """Return matches in compact Markdown form.

    Sections per match:
      - Shared summary (same as JSON "summary")
      - Location line using normalized "rel_filepath:line"
      - Full Error pre block with full context
    """

    # Build from JSON to ensure a single source of truth
    records: List[Dict[str, str]] = json.loads(format_json(matches))
    sections: List[str] = []
    for m in records:
        summary = m.get("summary", build_summary(m))
        body = (m.get("context") or m.get("full", "") or "").replace("`", "'")
        rel = m.get("rel_filepath", "") or m.get("file", "")
        line = m.get("line", "")
        loc = f"{rel}:{line}".strip(":")
        sections.append(
            f"📝 {summary}\n\n"
            f"📍 Location: `{loc}`\n\n"
            f"🔍 Full Error:\n\n<pre>\n{body}\n</pre>"
        )
    return "\n\n".join(sections) + ("\n" if sections else "")


FORMATTERS = {
    "json": format_json,
    "md": format_md,
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
        "--format", choices=FORMATTERS.keys(), default="json", help="Output format"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    formatter = FORMATTERS[args.format]
    limit = None if args.n == 0 else args.n

    if args.filenames:
        # Preserve per-file boundaries: format each file independently and
        # concatenate, matching the behavior used by reference outputs.
        chunks: List[str] = []
        for name in args.filenames:
            with open(name, "r", errors="ignore") as f:
                file_lines = list(f)
            file_matches = list(iter_matches(file_lines, limit=limit))
            chunks.append(formatter(file_matches))
        output = "".join(chunks)
    else:
        lines = list(sys.stdin)
        matches = list(iter_matches(lines, limit=limit))
        output = formatter(matches)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
