# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB entry point for CUDA C++ Core Libraries pretty printers.

Requires Python 3.12 or newer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gdb

_SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
if _SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, _SCRIPT_DIRECTORY)

import atomic  # noqa: E402
import buffer  # noqa: E402
import complex  # noqa: E402
import event  # noqa: E402
import hierarchy  # noqa: E402
import inplace_vector  # noqa: E402
import mdspan  # noqa: E402
import memory_pool  # noqa: E402
import memory_resource  # noqa: E402
import span  # noqa: E402
import std_array  # noqa: E402
import stream  # noqa: E402
import tuple  # noqa: E402

_PRINTERS = (
    memory_resource,
    atomic,
    buffer,
    std_array,
    complex,
    stream,
    tuple,
    inplace_vector,
    event,
    hierarchy,
    mdspan,
    memory_pool,
    span,
)


def register() -> None:
    """Register every CCCL GDB pretty printer."""
    for printer in _PRINTERS:
        printer.register(gdb)


register()
