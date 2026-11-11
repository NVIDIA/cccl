# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Read executable RST blocks directly from documentation source."""

from __future__ import annotations

import re
import textwrap


def code_blocks(path, language="cpp"):
    """Read literal code directives and retain their source locations.

    The runtime tests execute these exact snippets, rather than maintaining
    copies of the examples. Each result contains the dedented code, absolute
    page path, and first source line. Directive options stay out of the code.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    languages = (
        r"(?:cpp|c\+\+|cuda)" if language == "cpp" else re.escape(language)
    )
    found = []
    for index, line in enumerate(lines):
        match = re.fullmatch(rf"( *)\.\. code-block:: {languages}\s*", line)
        if not match:
            continue
        depth = len(match[1])
        body = []
        for following in lines[index + 1 :]:
            if (
                following.strip()
                and len(following) - len(following.lstrip()) <= depth
            ):
                break
            body.append(following)
        # Directive options precede the blank line introducing the code body.
        # This includes :force:, which permits preview C++ attributes.
        start = 0
        while start < len(body) and (
            not body[start].strip()
            or re.match(r":[\w-]+:(?:\s|$)", body[start].lstrip())
        ):
            start += 1
        code = textwrap.dedent("\n".join(body[start:])).rstrip() + "\n"
        found.append(
            {
                "path": str(path.resolve()),
                "line": index + 2 + start,
                "code": code,
            }
        )
    return found
