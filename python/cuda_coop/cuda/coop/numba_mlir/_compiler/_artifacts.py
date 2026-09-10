# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Temporary artifact and compiler-output helpers.

These functions own short-lived binary files and PTX metadata extraction used
by NVRTC/LTO assembly.  They do not normalize primitive parameters or define a
persistent cache schema.
"""

import os
import re
import tempfile
from collections import namedtuple
from typing import BinaryIO

version = namedtuple("version", ("major", "minor"))


def make_binary_tempfile(content: bytes, suffix: str) -> BinaryIO:
    """Write content to a closed, unbuffered temporary binary file."""

    tmp = tempfile.NamedTemporaryFile(
        mode="w+b", suffix=suffix, buffering=0, delete=False
    )
    try:
        tmp.write(content)
    except Exception:
        name = tmp.name
        tmp.close()
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise
    tmp.close()
    return tmp


def check_in(name, arg, values):
    """Validate a small closed-set compiler option."""

    if arg not in values:
        raise ValueError(f"{name} must be in {values} ; got {name} = {arg}")


def find_unsigned(name, txt):
    """Read one optional-initialized unsigned global from PTX text."""

    escaped_name = re.escape(name)
    regex = re.compile(
        f".global .align 4 .u32 {escaped_name} = ([0-9]*);", re.MULTILINE
    )
    found = regex.search(txt)
    if found is not None:
        return int(found.group(1))

    declaration = re.compile(f".global .align 4 .u32 {escaped_name};", re.MULTILINE)
    if declaration.search(txt) is not None:
        return 0
    raise ValueError(f"{name} not found in text")


__all__ = ["check_in", "find_unsigned", "make_binary_tempfile", "version"]
