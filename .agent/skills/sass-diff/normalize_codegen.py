#!/usr/bin/env python3
"""Normalize CUDA PTX/SASS dumps for lightweight textual diffs.

This intentionally removes common non-semantic noise but does not try to prove
semantic equivalence. Always inspect meaningful opcode, operand, and control-flow
changes manually after diffing normalized output.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def normalize_common(line: str) -> str:
    line = line.rstrip()
    line = re.sub(r"/tmp/[^\s,)]+", "/tmp/PATH", line)
    line = re.sub(r"/home/[^\s,)]+", "/home/PATH", line)
    line = re.sub(r"0x[0-9a-fA-F]+", "0xHEX", line)
    line = re.sub(r"\b[0-9a-fA-F]{16,}\b", "HEX", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def normalize_sass(line: str) -> str:
    line = re.sub(r"^\s*/\*[0-9a-fA-F]+\*/\s*", "", line)
    line = re.sub(r"^\s*/\*\s*[0-9]+\s*\*/\s*", "", line)
    line = re.sub(r"^\s*//.*$", "", line)
    line = normalize_common(line)
    line = re.sub(r"`\(.*\)$", "`(target)", line)
    return line


def normalize_ptx(line: str) -> str:
    line = re.sub(r"^\s*//.*$", "", line)
    line = re.sub(r"^\s*\.file\s+\d+\s+.*$", ".file N PATH", line)
    line = re.sub(r"^\s*\.loc\s+\d+\s+\d+.*$", ".loc N N", line)
    line = re.sub(r"\$L__[A-Za-z0-9_]+", "$L__LABEL", line)
    line = normalize_common(line)
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="Input file; defaults to stdin")
    parser.add_argument("--kind", choices=("sass", "ptx"), required=True)
    args = parser.parse_args()

    if args.path:
        lines = Path(args.path).read_text(errors="replace").splitlines()
    else:
        lines = sys.stdin.read().splitlines()

    normalizer = normalize_sass if args.kind == "sass" else normalize_ptx
    previous_blank = False
    for raw in lines:
        line = normalizer(raw)
        if not line:
            if not previous_blank:
                print()
            previous_blank = True
            continue
        print(line)
        previous_blank = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
