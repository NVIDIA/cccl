#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import re
import sys


def main() -> int:
    # We pre-compile this regular expression as a micro-optimization for speed. The
    # assumption is that the shebang is correct, so everything in the happy path should be
    # as fast as possible.
    FIRST_LINE_RE = re.compile(r"#!\s*/usr/bin/env.*")
    ret = 0
    for f in sys.argv[1:]:
        with open(f) as fd:
            first_line = fd.readline()

        if not first_line.startswith("#!"):
            # Not a shebang
            continue

        if FIRST_LINE_RE.match(first_line):
            # Already correct
            continue

        ret = 1

        if not (
            m := re.match(r"#!\s*(?:/bin/(\w+)|/usr/bin/(\w+))\s*(.*)", first_line)
        ):
            # Not assert, pre-commit may compile with -O
            raise AssertionError(f"Failed to match shebang for {first_line}")

        fixed = f"#!/usr/bin/env {m[1] or m[2]}".rstrip()
        if rest := m[3].strip():
            fixed += f" {rest}"
        fixed += "\n"

        with open(f) as fd:
            # Read the remaining lines, we need them in order to overwrite
            lines = fd.readlines()

        lines[0] = fixed

        with open(f, "w") as fd:
            fd.writelines(lines)

    return ret


if __name__ == "__main__":
    sys.exit(main())
