#!/usr/bin/env python3
"""Test harness for ci/inspect_changes.py."""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inspect_changes.py and compare output"
    )
    parser.add_argument(
        "--script", required=True, type=Path, help="Path to inspect_changes.py"
    )
    parser.add_argument(
        "--dirty", required=True, type=Path, help="File listing dirty files"
    )
    parser.add_argument(
        "--expected", required=True, type=Path, help="Expected stdout contents"
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Python interpreter to use"
    )
    return parser.parse_args()


def build_command(python: str, script: Path, dirty_file: Path) -> list[str]:
    cmd = [python, str(script), "--file", str(dirty_file)]
    return cmd


def main() -> int:
    args = parse_args()
    cmd = build_command(args.python, args.script, args.dirty)

    sys.stdout.write(f"COMMAND: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write("OUTPUT:\n")
    sys.stdout.write(result.stdout)
    if result.stdout and not result.stdout.endswith("\n"):
        sys.stdout.write("\n")
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.stderr.write("\nCommand failed: {}\n".format(" ".join(cmd)))
        return result.returncode

    actual_lines = result.stdout.splitlines()
    expected_lines = [
        line.strip()
        for line in args.expected.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    missing = [line for line in expected_lines if line not in actual_lines]
    if missing:
        diff = "\n".join(
            difflib.unified_diff(
                expected_lines,
                actual_lines,
                fromfile="expected(subset)",
                tofile="actual",
                lineterm="",
            )
        )
        if diff:
            sys.stderr.write(diff + "\n\n")
        sys.stderr.write("\nExpected lines missing from output:\n")
        for line in missing:
            sys.stderr.write(f"  {line}\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
