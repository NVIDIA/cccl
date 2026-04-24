# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path
    from re import Match


def is_in_comment(re_match: Match) -> bool:
    string = re_match.string
    prev_end = re_match.start()
    try:
        line_begin = string.rindex("\n", 0, prev_end) + 1
    except ValueError:
        # ValueError: substring not found, means we are on the first line of
        # the file
        line_begin = 0
    line_prefix = string[line_begin:prev_end]
    return line_prefix.lstrip().startswith(("//", "/*", "*"))


HEADER_SUFFIXES: Final = {
    ".h",
    ".inl",
    ".cuh",
    ".cuinl",
    ".hpp",
    ".inc",
    ".HH",
}


def is_header(path: Path) -> bool:
    return path.suffix in HEADER_SUFFIXES


def is_publically_accessible(path: Path) -> bool:
    return "detail" not in path.parts


def is_in_test_dir(path: Path) -> bool:
    return "tests" in path.parts
