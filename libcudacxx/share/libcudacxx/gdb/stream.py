# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printers for cuda::stream and cuda::stream_ref."""

from __future__ import annotations

from types import ModuleType

import memory_resource

import gdb
import gdb.printing

_STREAM_TYPES = frozenset({"cuda::stream", "cuda::stream_ref"})
# These are the public values of cudaStreamLegacy and cudaStreamPerThread.
# The debugger expression parser does not necessarily expose the macros.
_CUDA_STREAM_LEGACY_HANDLE = 1
_CUDA_STREAM_PER_THREAD_HANDLE = 2


def _stream_type_name(value_type: gdb.Type) -> str | None:
    value_type = value_type.strip_typedefs().unqualified()
    type_name = memory_resource.public_type_name(value_type)
    if type_name in _STREAM_TYPES:
        return type_name
    return None


def _handle_description(handle: int, byte_size: int) -> str:
    if handle == 0:
        return "default"
    if handle == _CUDA_STREAM_LEGACY_HANDLE:
        return "legacy"
    if handle == _CUDA_STREAM_PER_THREAD_HANDLE:
        return "per-thread"
    if handle == (1 << (byte_size * 8)) - 1:
        return "invalid"
    return f"{handle:#x}"


class StreamPrinter:
    """Summarize a CUDA stream wrapper without calling into the inferior."""

    def __init__(self, value: gdb.Value, type_name: str) -> None:
        self.value = value
        self.type_name = type_name

    def to_string(self) -> str:
        try:
            handle = self.value["__stream"]
            description = _handle_description(int(handle), handle.type.sizeof)
        except (gdb.error, TypeError, ValueError):
            return self.type_name
        return f"{self.type_name} handle={description}"


class StreamPrinterLookup(gdb.printing.PrettyPrinter):
    """Select printers for cuda::stream and cuda::stream_ref."""

    def __init__(self) -> None:
        super().__init__("cuda::stream")

    def __call__(self, value: gdb.Value) -> StreamPrinter | None:
        type_name = _stream_type_name(value.type)
        if type_name is None:
            return None
        return StreamPrinter(value, type_name)


def register(objfile: ModuleType) -> None:
    """Register CUDA stream formatters with GDB."""
    gdb.printing.register_pretty_printer(objfile, StreamPrinterLookup(), replace=True)
