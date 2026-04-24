#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

import sys
from datetime import datetime
from re import IGNORECASE
from typing import TYPE_CHECKING, Final

from util.re_replacement import RegexReplacement, Replacement

if TYPE_CHECKING:
    from pathlib import Path
    from re import Match


CUR_YEAR: Final = datetime.now().year


def replacement(path: Path, re_match: Match) -> str:
    if re_match["single"] is not None:
        year = CUR_YEAR
    elif (range_begin := re_match["range_begin"]) is not None:
        year = f"{range_begin}-{CUR_YEAR}"
    else:
        m = (
            "Neither single-year nor multi-year regex "
            f"matched for {re_match[0]} ({path})"
        )
        raise ValueError(m)

    return f"(c) {year}, NVIDIA CORPORATION"


def main() -> int:
    single_year_re = r"(?P<single>[\d]+)"
    range_year_re = r"(?P<range_begin>\d+)\s*\-\s*\d+"
    re_str = (
        rf"\(c\)\s*({single_year_re}|{range_year_re})\s*,?\s*(nvidia\s+corporation)"
    )
    repl = Replacement(
        pattern=re_str,
        repl=replacement,
        pragma_keyword="copyright",
        flags=IGNORECASE,
    )
    return RegexReplacement(
        description="Find and fix the date in copyright notices",
        replacements=[repl],
        allowed_suffixes=RegexReplacement.AllSuffixes,
    ).main()


if __name__ == "__main__":
    sys.exit(main())
