#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# strip_unprintable.py - Remove invisible / unprintable characters from text files.
import re
import sys
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

# Canonical definition of what gets removed. One row per range group:
#   (regex character-class fragment, human description)
# The compiled character class and the --help listing are both derived from this
# table, so adding or removing a range only needs to happen here. TAB (U+0009),
# LF (U+000A), and CR (U+000D) are deliberately excluded from the C0 range. The
# fragments use \x/\u escapes so the source file itself stays free of the very
# characters this script removes.
RANGES = (
    (
        r"\x00-\x08\x0b\x0c\x0e-\x1f\x7f",
        "C0 controls / DEL (TAB, LF, CR preserved)",
    ),
    (r"\x80-\x9f", "C1 controls"),
    (r"\xa0", "no-break space"),
    (r"\u200b-\u200f", "zero-width space/joiners, bidi marks"),
    (r"\u202a-\u202e", "bidi embedding/override"),
    (r"\u2060-\u2064", "word joiner, invisible operators"),
    (r"\ufeff", "BOM / zero-width no-break space"),
)

# Character class assembled from column 1 of the ranges table.
BAD_RE = re.compile("[" + "".join(frag for frag, _ in RANGES) + "]")


def parse_args() -> Namespace:
    removed = "\n".join(f"  {frag}\t{desc}" for frag, desc in RANGES)
    parser = ArgumentParser(
        description=(
            "Remove invisible / unprintable characters from text files, in place, "
            "while preserving ordinary whitespace (TAB U+0009, LF U+000A, "
            "CR U+000D)."
        ),
        epilog=f"Removed characters:\n{removed}",
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Report offending files and their line/column locations; make no "
            "edits. Exits non-zero if any unprintable characters are found."
        ),
    )
    parser.add_argument("files", nargs="+", metavar="FILE")
    return parser.parse_args()


def check(files: list[str]) -> int:
    ret = 0
    for f in files:
        with open(f, encoding="utf-8", errors="surrogateescape") as fd:
            for lineno, line in enumerate(fd, start=1):
                for m in BAD_RE.finditer(line):
                    print(f"{f}:{lineno}:{m.start() + 1}: U+{ord(m.group()):04X}")
                    ret = 1

    return ret


def strip(files: list[str]) -> int:
    ret = 0
    for f in files:
        with open(f, encoding="utf-8", errors="surrogateescape") as fd:
            original = fd.read()
        stripped = BAD_RE.sub("", original)
        if stripped != original:
            with open(
                f, "w", encoding="utf-8", errors="surrogateescape", newline=""
            ) as fd:
                fd.write(stripped)
            ret = 1

    return ret


def main() -> int:
    args = parse_args()
    if args.check:
        return check(args.files)
    return strip(args.files)


if __name__ == "__main__":
    sys.exit(main())
