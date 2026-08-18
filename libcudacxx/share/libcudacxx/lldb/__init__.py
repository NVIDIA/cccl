# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB entry point for CUDA C++ Core Libraries pretty printers.

Requires Python 3.12 or newer.
"""

from __future__ import annotations

import atomic
import buffer
import complex
import event
import hierarchy
import inplace_vector
import mdspan
import memory_pool
import memory_resource
import span
import std_array
import stream
import tuple

import lldb

_CATEGORY = "cccl"
_FORMATTERS = (
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
InternalDict = dict[str, object]


def __lldb_init_module(debugger: lldb.SBDebugger, _internal_dict: InternalDict) -> None:
    debugger.HandleCommand(f"type category define {_CATEGORY}")
    for formatter in _FORMATTERS:
        module = f"{__name__}.{formatter.__name__.rsplit('.', 1)[-1]}"
        formatter.register(debugger, _CATEGORY, module)
    debugger.HandleCommand(f"type category enable {_CATEGORY}")
