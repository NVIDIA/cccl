# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import tempfile
from collections import namedtuple

version = namedtuple("version", ("major", "minor"))


def make_binary_tempfile(content: bytes, suffix: str):
    """Write a persistent link-input file and return its closed handle.

    Compiler link inputs are consumed by filename after provider construction,
    so flush and close the file before returning while leaving its path intact.
    The caller must arrange successful-file cleanup, normally through an
    ``Invocable`` or shared-bundle owner. If writing raises, remove the
    partially written file before propagating the exception.

    Parameters
    ----------
    content : bytes
        Binary image to write without buffering.
    suffix : str
        Filename suffix used by the linker to recognize the image format.

    Returns
    -------
    file object
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


def find_unsigned(name, txt):
    """Read an emitted unsigned storage-metadata global from PTX text.

    Provider compilation emits C++ ``sizeof`` and ``alignof`` constants, then
    links to PTX to inspect their values without executing a GPU kernel. Match
    the compiler's aligned 32-bit unsigned global declaration for the requested
    symbol. A declaration without an initializer denotes zero. This is a narrow
    metadata extractor, not a general PTX parser.

    Parameters
    ----------
    name : str
        Exact global symbol name, escaped before constructing the search
        pattern.
    txt : str
        PTX containing the provider's metadata globals.

    Returns
    -------
    int
        Decimal initializer value, or zero for an uninitialized declaration.

    Raises
    ------
    ValueError
        No recognized declaration for ``name`` is present.
    """

    escaped_name = re.escape(name)
    regex = re.compile(
        f".global .align 4 .u32 {escaped_name} = ([0-9]*);", re.MULTILINE
    )
    found = regex.search(txt)
    if found is not None:
        return int(found.group(1))

    declaration = re.compile(
        f".global .align 4 .u32 {escaped_name};", re.MULTILINE
    )
    if declaration.search(txt) is not None:
        return 0
    raise ValueError(f"{name} not found in text")


__all__ = ["check_in", "find_unsigned", "make_binary_tempfile", "version"]
