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

import atomic
import buffer
import complex
import event
import hierarchy
import inplace_vector
import mdspan
import memory_pool
import memory_resource
import optional
import shared_resource
import span
import std_array
import stream
import tuple

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
    optional,
    shared_resource,
)


def register() -> None:
    """Register every CCCL GDB pretty printer."""
    for printer in _PRINTERS:
        printer.register(gdb)


register()
