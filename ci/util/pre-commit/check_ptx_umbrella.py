#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Require <cuda/ptx> to export every PTX instruction header.

Every generated instruction must be included by a wrapper in cuda/__ptx/{instructions,pragmas}/, every wrapper must have
a public cuda/ptxs/ header, and <cuda/ptx> must include exactly the cuda/ptxs/ headers.
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CUDA = Path("libcudacxx/include/cuda")
INCLUDE_RE = re.compile(r"^\s*#\s*include\s*<cuda/ptxs/([^>]+)>", re.MULTILINE)

# Wrappers without a cuda/ptxs/ header because another wrapper already includes their generated body.
INCLUDED_BY_OTHER = {"cp_async_mbarrier_arrive_noinc.h"}


def names(path: Path) -> set[str]:
    return {p.name for p in path.glob("*.h")}


def main() -> int:
    os.chdir(ROOT)
    errors = []
    umbrella = set(INCLUDE_RE.findall((CUDA / "ptx").read_text()))
    ptxs = names(CUDA / "ptxs")
    errors += [
        f"{CUDA / 'ptx'}: missing #include <cuda/ptxs/{h}>"
        for h in sorted(ptxs - umbrella)
    ]
    errors += [
        f"{CUDA / 'ptx'}: includes <cuda/ptxs/{h}>, which does not exist"
        for h in sorted(umbrella - ptxs)
    ]

    wrappers = names(CUDA / "__ptx/instructions") | names(CUDA / "__ptx/pragmas")
    for h in sorted(wrappers - ptxs - INCLUDED_BY_OTHER):
        errors.append(
            f"{CUDA / 'ptxs' / h}: missing public header for cuda/__ptx wrapper {h}"
        )

    wrapper_text = "".join(
        p.read_text() for p in (CUDA / "__ptx/instructions").glob("*.h")
    )
    for h in sorted(names(CUDA / "__ptx/instructions/generated")):
        if f"<cuda/__ptx/instructions/generated/{h}>" not in wrapper_text:
            errors.append(
                f"{CUDA / '__ptx/instructions/generated' / h}: not included by any cuda/__ptx/instructions/ wrapper"
            )

    for e in errors:
        print(e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
