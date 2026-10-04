# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Prepare compiled provider files for linking.

The linker consumes binary inputs by filename after provider construction.
Temporary-file helpers therefore keep those paths alive for the owner to clean
up later.
"""

import os
import tempfile
from collections import namedtuple
from typing import IO

version = namedtuple("version", ("major", "minor"))


def make_binary_tempfile(content: bytes, suffix: str) -> IO[bytes]:
    """Write a persistent link-input file and return its closed handle.

    Compiler link inputs are consumed by filename after provider construction,
    so flush and close the file before returning while leaving its path
    intact. The caller must arrange successful-file cleanup, normally through
    an ``Invocable`` or shared-bundle owner. If writing raises, remove the
    partially written file before propagating the exception.

    Parameters
    ----------
    content : bytes
        Binary image to write without buffering.
    suffix : str
        Filename suffix used by the linker to recognize the image format.

    Returns
    -------
    IO[bytes]
        Closed ``NamedTemporaryFile`` wrapper whose ``name`` identifies the
        persistent file. It is not an open stream for subsequent writes.
    """

    with tempfile.NamedTemporaryFile(
        mode="w+b", suffix=suffix, buffering=0, delete=False
    ) as tmp:
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
    return tmp


def check_in(name, arg, values):
    """Validate a small closed-set compiler option."""

    if arg not in values:
        raise ValueError(f"{name} must be in {values} ; got {name} = {arg}")


__all__ = ["check_in", "make_binary_tempfile", "version"]
