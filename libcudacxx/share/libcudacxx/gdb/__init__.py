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

import buffer  # noqa: E402
import memory_resource  # noqa: E402

_PRINTERS = (memory_resource, buffer)


def register() -> None:
    """Register every CCCL GDB pretty printer."""
    for printer in _PRINTERS:
        printer.register(gdb)


register()
