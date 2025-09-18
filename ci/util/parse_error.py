#!/usr/bin/env python3
"""Parse configure/build/test logs and extract concise error diagnostics.

- Reads one or more logs (or stdin) and reports the first N matching errors.
- JSON is the canonical output; Markdown is derived from the same JSON so both
  formats remain synchronized. The parse_error tests validate both.

Key behaviors and special cases:
- Patterns (PatternSpec) can capture multi-line context using begin/end
  anchors (e.g., Ninja FAILED banner to clang epilogue) and can refine an
  initial match by re-parsing the captured context with other pattern types
  (e.g., build errors inside LIT/CTest sections).
- Paths are canonicalized to repo-relative POSIX style with ".." collapsed.
- Results include origin metadata (pattern_type/name) and optional target_name.
- Deduplication prefers targeted entries when multiple matches share a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import sys
from typing import Dict, Iterable, Iterator, List, Optional
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Regex patterns
# -----------------------------------------------------------------------------

@dataclass
class PatternSpec:
    """Specification for a single error pattern.

    name: Human-readable identifier (e.g., "cmake error").
    type: Category string: 'config' | 'build' | 'test' | 'lit'.
    error: Compiled regex matching the error line; should expose 'file',
      'line', and 'msg' when available.
    context_begin/context_end: Optional anchors delimiting a multi-line
      context block around the error. See iter_matches for capture rules.
    refine_error_by_type: Optional list of types to search within the captured
      context; when a refinement matches, replace file/line/msg/full while
      preserving the captured context.
    """
    name: str
    type: str
    error: re.Pattern[str]
    context_begin: Optional[re.Pattern[str]] = None
    context_end: Optional[re.Pattern[str]] = None
    context_begin_inclusive: bool = True
    context_end_inclusive: bool = True
    refine_error_by_type: Optional[List[str]] = None
    # Optional regex to extract a target/test name. Should contain a named
    # group 'target' or a first positional group with the desired name.
    target_name: Optional[re.Pattern[str]] = None
    # Optional repo-relative prefixes to try when resolving files on disk.
    # These are hints for downstream consumers (e.g., failure renderers) to
    # locate files that are referenced by relative paths in logs.
    file_prefixes: Optional[List[str]] = None


CONFIGURE_SPECS: List[PatternSpec] = [
    # CMake configure errors
    # example: "CMake Error at CMakeLists.txt:5 (message):"
    PatternSpec(
        name="cmake error",
        type="config",
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
        name="gcc/clang error",
        type="build",
        error=re.compile(
            r"^(?P<file>[^:\n]+):(?P<line>\d+):(?:\d+:)?\s*(?P<msg>.*\b(error|fatal)\b.*)$",
            re.IGNORECASE,
        ),
        # Capture context from the preceding Ninja failure banner to the
        # final clang summary. For CUDA compilation via clang, the tool emits
        # lines like "8 errors generated when compiling for sm_80."; include
        # those as the end of context when present.
        context_begin=re.compile(r"^\s*FAILED:\s+"),
        context_end=re.compile(
            r"^\s*\d+\s+errors?\s+generated\s+when\s+compiling\s+for\s+sm_\d+\.?\s*$",
            re.IGNORECASE,
        ),
        target_name=re.compile(r"CMakeFiles/(?P<target>[^/\s]+)\.dir/"),
    ),
    # NVCC diagnostics
    # example: "foo.cu(10): error: identifier 'bar' is undefined"
    PatternSpec(
        name="nvcc error",
        type="build",
        error=re.compile(
            r"^(?P<file>[^:(\n]+)\((?P<line>\d+)\):\s*(?P<msg>.*\b(error|fatal)\b.*)$",
            re.IGNORECASE,
        ),
        # Context capture for nvcc in Ninja output
        context_begin=re.compile(r"^\s*FAILED:\s+"),
        context_end=re.compile(r"^\s*\d+\s+error detected in the compilation of "),
        target_name=re.compile(r"CMakeFiles/(?P<target>[^/\s]+)\.dir/"),
    ),
]

# --- LIT --------------------------------------------------------------------
LIT_SPECS: List[PatternSpec] = [
    # lit result lines (unexpected failures/passes)
    # example: "FAIL: suite :: test (1 of 2)"
    PatternSpec(
        name="lit error",
        type="lit",
        error=re.compile(
            r"^\s*(?P<msg>FAIL|XPASS|ERROR):\s+(?:.+?\s+::\s+)?(?P<file>.+?) \(\d+ of \d+\)$",
            re.IGNORECASE,
        ),
        # lit context: between a FAILED banner and the closing banner
        # Example begin: ******************** TEST 'suite :: test' FAILED ********************
        # Example end:   ********************
        context_begin=re.compile(r"^\s*\*{6,}\s*TEST '.*' FAILED\s*\*{6,}\s*$"),
        context_end=re.compile(r"^\s*\*{6,}\s*$"),
        # Update error info with any build errors within the full context match.
        refine_error_by_type=["build"],
        target_name=re.compile(r"TEST '.*?::\s*(?P<target>.*?)'\s*FAILED", re.IGNORECASE),
        file_prefixes=["libcudacxx/test/libcudacxx"],
    ),
    # lit diagnostics
    # example: "lit: /path/format.py:130: fatal: Unsupported RUN line"
    PatternSpec(
        name="lit diagnostic",
        type="lit",
        error=re.compile(
            r"^lit:\s+(?P<file>[^:\n]+):(?P<line>\d+):\s*(?P<msg>.*\b(error|fatal)\b.*)$",
            re.IGNORECASE,
        ),
        file_prefixes=["libcudacxx/test/libcudacxx"],
    ),
]

# --- test -------------------------------------------------------------------
TEST_SPECS: List[PatternSpec] = [
    # CTest summary lines
    # example: "1 - fail (Failed)"
    PatternSpec(
        name="ctest failure",
        type="test",
        error=re.compile(
            r"^\s*\d+/\d+\s+Test\s+#(?P<line>\d+):\s+[^\s]+\s+\.\.+\*\*\*(?P<msg>Failed|Timeout|Not Run|Skipped|Passed)\b.*$",
            re.IGNORECASE,
        ),
        # Capture epilogue from the error header line down to the next test start or final summary
        # End examples:  "Start 2: fail2" OR "0% tests passed, 3 tests failed out of 3"
        context_end=re.compile(
            r"^(\s*Start\s+\d+:|\s*\d+%\s+tests\s+passed,\s+\d+\s+tests\s+failed\s+out\s+of\s+\d+)", re.IGNORECASE),
        context_end_inclusive=False,
        # Sometimes build errors end up in these:
        refine_error_by_type=["build"],
        # Extract test name from the header line itself
        target_name=re.compile(r"^\s*\d+/\d+\s+Test\s+#\d+:\s+(?P<target>[^\s]+)\s+\.+", re.IGNORECASE),
    ),
]

# Combined list of all patterns in evaluation order
# Order matters – tuned for useful diagnostics first.
ERROR_SPECS: List[PatternSpec] = CONFIGURE_SPECS + LIT_SPECS + TEST_SPECS + BUILD_SPECS

# Queryable registries
SPECS_BY_NAME: Dict[str, PatternSpec] = {}
SPECS_BY_TYPE: Dict[str, List[PatternSpec]] = {"config": [], "build": [], "test": [], "lit": []}
for spec in (CONFIGURE_SPECS + BUILD_SPECS + LIT_SPECS + TEST_SPECS):
    SPECS_BY_NAME[spec.name] = spec
    SPECS_BY_TYPE.setdefault(spec.type, []).append(spec)


def get_specs(query: Optional[Iterable[str] | str] = None, *, type: Optional[str] = None) -> List[PatternSpec]:
    """Return pattern specs by name(s) or by type.

    If ``query`` is None, returns all specs (unordered). If a string, returns
    the matching spec (if present). If an iterable of strings, returns the
    matching specs in the given order. If ``ptype`` is provided, returns all
    specs of that category.
    """
    if type is not None:
        return list(SPECS_BY_TYPE.get(type, []))
    if query is None:
        return list(SPECS_BY_NAME.values())
    if isinstance(query, str):
        return [SPECS_BY_NAME[query]] if query in SPECS_BY_NAME else []
    out: List[PatternSpec] = []
    for name in query:
        if name in SPECS_BY_NAME:
            out.append(SPECS_BY_NAME[name])
    return out


# -----------------------------------------------------------------------------
# Matching utilities
# -----------------------------------------------------------------------------

def _detect_repo_name() -> str:
    """Detect the repo directory name for prefix stripping.

    Honors PARSE_ERROR_REPO_NAME if set; otherwise uses cwd's basename.
    Tests in CCCL should not set the override so defaults remain "cccl".
    """
    env = os.environ.get("PARSE_ERROR_REPO_NAME")
    if env:
        return env.strip("/")
    try:
        return os.path.basename(os.getcwd())
    except Exception:
        return "cccl"


def _prefix_patterns(repo: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    build = re.compile(rf"^.*/{re.escape(repo)}/build/[^/]+/[^/]+/")
    src = re.compile(rf"^.*/{re.escape(repo)}/")
    return build, src


_REPO_NAME = _detect_repo_name()
_BUILD_PREFIX_RE, _SRC_PREFIX_RE = _prefix_patterns(_REPO_NAME)


def _normalize_file(path: str) -> str:
    """Return a canonical, repo-relative POSIX path when possible.

    - Strips build and repo absolute prefixes.
    - Normalizes and collapses any "../" segments (e.g., lib/.../../../../foo -> foo).
    - Uses POSIX separators regardless of host OS to keep outputs stable.
    """
    if not path:
        return ""
    # Work with POSIX-style separators
    p = path.replace("\\", "/")
    # Drop leading build and repo prefixes
    p = _BUILD_PREFIX_RE.sub("", p)
    p = _SRC_PREFIX_RE.sub("", p)
    # Collapse .. and . segments
    p = posixpath.normpath(p)
    # Remove any accidental leading ./
    if p.startswith("./"):
        p = p[2:]
    return p


def _resolve_with_prefixes(rel_path: str, prefixes: Optional[List[str]]) -> Optional[str]:
    """Return a repo-relative path using optional prefixes.

    Preference order:
    1) rel_path itself if it exists under the repo root.
    2) The first prefix+rel_path that exists.
    3) Fallback to the input string even if it doesn't exist.
    """
    if not rel_path or not prefixes:
        return None
    try:
        repo_root = os.getcwd()
    except Exception:
        return None
    # 1) Plain relative path
    cand = os.path.join(repo_root, rel_path)
    if os.path.isfile(cand):
        return os.path.relpath(cand, repo_root).replace(os.sep, "/")
    # 2) Existing prefixed path
    for pref in prefixes or []:
        cand = os.path.join(repo_root, pref, rel_path)
        if os.path.isfile(cand):
            return os.path.relpath(cand, repo_root).replace(os.sep, "/")
    # 3) Fallback to input
    if prefixes:
        return rel_path
    return None


@dataclass
class MatchRecord:
    """Structured error record to keep fields consistent and documented."""

    line: str = ""
    msg: str = ""
    pattern_type: str = ""
    pattern_name: str = ""
    rel_filepath: str = ""
    filename: str = ""
    full: str = ""
    context: str = ""
    target_name: str = ""
    location: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, str]:
        d = {
            "error_short": self.msg,
            "pattern_type": self.pattern_type,
            "pattern_name": self.pattern_name,
            "rel_filepath": self.rel_filepath,
            "filename": self.filename,
            "error_line": self.full,
            "error_context": self.context,
        }
        if self.line:
            d["line"] = self.line
        if self.target_name:
            d["target_name"] = self.target_name
        if self.location:
            d["location"] = self.location
        if self.summary:
            d["summary"] = self.summary
        return d


def _capture_context(raw_lines: List[str], i: int, spec: PatternSpec, line: str) -> str:
    """Capture a context block around match at index i according to spec."""
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
            stop = end_idx + (1 if spec.context_end_inclusive else 0)
            epilogue_lines = [s.rstrip("\n") for s in raw_lines[i + 1 : stop]]

    if forward_begin_idx is not None and end_idx is not None:
        stop = end_idx + (1 if spec.context_end_inclusive else 0)
        return "\n".join(s.rstrip("\n") for s in raw_lines[forward_begin_idx:stop])

    if start_idx is not None and spec.context_begin is not None and not spec.context_begin_inclusive:
        preamble_lines = preamble_lines[1:] if preamble_lines else []
    return "\n".join(preamble_lines + [line] + epilogue_lines)


def _extract_target(spec: PatternSpec, context: str, line: str) -> str:
    tn = getattr(spec, "target_name", None)
    if tn is None:
        return ""
    for s in (context, line):
        if not s:
            continue
        m = tn.search(s)
        if m:
            return (m.groupdict().get("target") or m.group(1)).strip()
    return ""


def _refine_from_context(spec: PatternSpec, context: str, base: Dict[str, str]) -> Dict[str, str]:
    if not spec.refine_error_by_type:
        return base
    for rtype in spec.refine_error_by_type:
        for rspec in SPECS_BY_TYPE.get(rtype, []):
            for cl in context.splitlines():
                rmatch = rspec.error.match(cl.strip("\n"))
                if not rmatch:
                    continue
                r = rmatch.groupdict()
                abs_fp = (r.get("file", "") or "").strip()
                rel_fp = _normalize_file(abs_fp).strip() if abs_fp else ""
                fname = os.path.basename(rel_fp or abs_fp) if (rel_fp or abs_fp) else ""
                updated = {
                    **base,
                    **{k: v for k, v in r.items() if k in {"file", "line", "msg"}},
                }
                if abs_fp:
                    updated.update(
                        {
                            "file": rel_fp or abs_fp,
                            "abs_filepath": abs_fp,
                            "rel_filepath": rel_fp,
                            "filename": fname,
                            "full": rmatch.group(0),
                        }
                    )
                return updated
    return base


def find_spec_and_match(line: str) -> Optional[tuple[PatternSpec, re.Match[str]]]:
    """Return the first (spec, match) for the line, or None."""

    for spec in ERROR_SPECS:
        match = spec.error.match(line)
        if match:
            return spec, match
    return None


def iter_matches(lines: Iterable[str], limit: Optional[int] = 1) -> Iterator[Dict[str, str]]:
    """Yield structured match dicts for each error line found."""

    count = 0
    raw_lines = list(lines)
    for i, raw in enumerate(raw_lines):
        line = raw.rstrip("\n")
        found = find_spec_and_match(line)
        if not found:
            continue
        spec, match = found
        gd = match.groupdict()
        rec = MatchRecord(
            msg=(gd.get("msg") or "").strip(),
            line=(gd.get("line") or "").strip(),
            pattern_type=spec.type,
            pattern_name=spec.name,
        )
        abs_fp = (gd.get("file", "") or "").strip()
        if abs_fp:
            rel_fp = _normalize_file(abs_fp).strip()
            rec.rel_filepath = rel_fp
            rec.filename = os.path.basename(rel_fp or abs_fp)
        rec.full = line
        context_str = _capture_context(raw_lines, i, spec, line)
        rec.context = context_str
        tname = _extract_target(spec, context_str, line)
        if tname:
            rec.target_name = tname
        # Refinement
        refined = _refine_from_context(spec, context_str, rec.to_dict())
        rec.line = refined.get("line", rec.line)
        rec.msg = refined.get("msg", rec.msg)
        rec.full = refined.get("full", rec.full)
        rec.rel_filepath = refined.get("rel_filepath", rec.rel_filepath)
        rec.filename = refined.get("filename", rec.filename)

        # Attempt to resolve relative files using spec-provided prefixes
        # Use prefixes to resolve rel filepath when present
        if rec.rel_filepath and getattr(spec, 'file_prefixes', None):
            rel_res = _resolve_with_prefixes(rec.rel_filepath, spec.file_prefixes)
            if rel_res:
                rec.rel_filepath = rel_res
                rec.filename = os.path.basename(rel_res)

        base_loc = rec.rel_filepath or rec.filename or rec.target_name
        if base_loc and rec.line:
            rec.location = f"{base_loc}:{rec.line}"
        rec.summary = build_summary(rec.to_dict())

        yield rec.to_dict()
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

    loc_file = m.get("filename", "") or m.get("rel_filepath", "") or m.get("target_name", "")
    loc = f"{loc_file}:{m.get('line', '')}".strip(":")
    msg = m.get("error_short", "").strip().replace("`", "'")
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
        # Build location from rel path or filename; fall back to target_name
        loc_base = m.get('rel_filepath', '') or m.get('filename', '') or m.get('target_name', '')
        loc = f"{loc_base}:{m.get('line', '')}".strip(":")
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
        body = (m.get("error_context") or m.get("error_line", "") or "").replace("`", "'")
        rel = m.get("rel_filepath", "") or m.get("filename", "") or m.get("target_name", "")
        line = m.get("line", "")
        loc = f"{rel}:{line}".strip(":")
        tgt = m.get("target_name")
        target_line = f"🎯 Target Name: {tgt}\n\n" if tgt else ""
        sections.append(
            f"📝 {summary}\n\n"
            f"📍 Location: `{loc}`\n\n"
            f"{target_line}"
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

    def _dedupe(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
        order: List[tuple[str, str]] = []  # (summary, tkey)
        buckets: Dict[str, Dict[str, Dict[str, str]]] = {}
        for m in records:
            # Build a stable summary key identical to JSON summary
            s = build_summary(m)
            t = (m.get("target_name") or "").strip()
            if s not in buckets:
                buckets[s] = {}
            # If a targetless entry arrives but a targeted one already exists, skip it
            if not t and any(k for k in buckets[s].keys() if k):
                continue
            if t in buckets[s]:
                # already have an entry for (summary, target); keep first
                continue
            if t and "" in buckets[s]:
                # Prefer the one with target over the prior targetless one
                buckets[s][t] = m
                # Replace in order list at the position of the targetless one
                for idx, (os, ot) in enumerate(order):
                    if os == s and ot == "":
                        order[idx] = (s, t)
                        break
                del buckets[s][""]
            else:
                buckets[s][t] = m
                order.append((s, t))
        # Flatten preserving order
        result: List[Dict[str, str]] = []
        for s, t in order:
            result.append(buckets[s][t])
        return result

    if args.filenames:
        # Preserve per-file boundaries: format each file independently and
        # concatenate, matching the behavior used by reference outputs.
        chunks: List[str] = []
        for name in args.filenames:
            with open(name, "r", errors="ignore") as f:
                file_lines = list(f)
            file_matches = list(iter_matches(file_lines, limit=None))
            file_matches = _dedupe(file_matches)
            if limit is not None:
                file_matches = file_matches[:limit]
            chunks.append(formatter(file_matches))
        output = "".join(chunks)
    else:
        lines = list(sys.stdin)
        matches = list(iter_matches(lines, limit=None))
        matches = _dedupe(matches)
        if limit is not None:
            matches = matches[:limit]
        output = formatter(matches)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
