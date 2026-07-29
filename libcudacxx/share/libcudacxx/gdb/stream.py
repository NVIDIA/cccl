# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printers for cuda::stream and cuda::stream_ref."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import NamedTuple

import memory_resource

import gdb
import gdb.printing

_STREAM_TYPES = frozenset({"cuda::stream", "cuda::stream_ref"})
# These are the public values of cudaStreamLegacy and cudaStreamPerThread.
# The debugger expression parser does not necessarily expose the macros.
_CUDA_STREAM_LEGACY_HANDLE = 1
_CUDA_STREAM_PER_THREAD_HANDLE = 2
_CUDA_STREAM_CAPTURE_STATUS_NONE = 0
_CUDA_STREAM_CAPTURE_STATUS_ACTIVE = 1
_CUDA_STREAM_IS_CAPTURING = "((int (*)(cudaStream_t, int*))cudaStreamIsCapturing)"
_CU_STREAM_GET_ID = "((int (*)(void*, unsigned long long*))cuStreamGetId)"


class StreamInfo(NamedTuple):
    handle: int
    handle_description: str
    unique_id: int | None
    device: int | None
    priority: int | None
    is_capturing: bool | None
    flags: int | None


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


def _query_stream_property(handle: int, function: str, output_type: str) -> int | None:
    output: gdb.Value | None = None
    try:
        output = gdb.parse_and_eval(f"({output_type}*)malloc(sizeof({output_type}))")
        address = int(output)
        if address == 0:
            return None
        status = gdb.parse_and_eval(
            f"(int){function}((cudaStream_t){handle:#x}, ({output_type}*){address:#x})"
        )
        if int(status) != 0:
            return None
        return int(output.dereference())
    except (gdb.error, TypeError, ValueError):
        return None
    finally:
        if output is not None:
            try:
                gdb.parse_and_eval(f"(void)free((void*){int(output):#x})")
            except (gdb.error, TypeError, ValueError):
                pass


def _query_stream_id(handle: int) -> int | None:
    unique_id = _query_stream_property(handle, _CU_STREAM_GET_ID, "unsigned long long")
    if unique_id is not None:
        return unique_id
    return _query_stream_property(handle, "cudaStreamGetId", "unsigned long long")


def _query_stream_device_from_context(handle: int) -> int | None:
    output: gdb.Value | None = None
    context_pushed = False
    try:
        output = gdb.parse_and_eval("(void**)malloc(sizeof(void*))")
        address = int(output)
        if address == 0:
            return None

        status = gdb.parse_and_eval(
            "(int)((int (*)(void*, void**))cuStreamGetCtx)"
            f"((void*){handle:#x}, (void**){address:#x})"
        )
        if int(status) != 0:
            return None
        context = int(output.dereference())

        status = gdb.parse_and_eval(
            f"(int)((int (*)(void*))cuCtxPushCurrent)((void*){context:#x})"
        )
        if int(status) != 0:
            return None
        context_pushed = True

        status = gdb.parse_and_eval(
            f"(int)((int (*)(int*))cuCtxGetDevice)((int*){address:#x})"
        )
        if int(status) != 0:
            return None
        device = gdb.Value(address).cast(gdb.lookup_type("int").pointer())
        return int(device.dereference())
    except (gdb.error, TypeError, ValueError):
        return None
    finally:
        if output is not None:
            try:
                if context_pushed:
                    gdb.parse_and_eval(
                        "(int)((int (*)(void**))cuCtxPopCurrent)"
                        f"((void**){int(output):#x})"
                    )
                gdb.parse_and_eval(f"(void)free((void*){int(output):#x})")
            except (gdb.error, TypeError, ValueError):
                pass


def _query_stream_device(handle: int) -> int | None:
    device = _query_stream_device_from_context(handle)
    if device is not None:
        return device
    return _query_stream_property(handle, "cudaStreamGetDevice", "int")


def _stream_info(handle_value: gdb.Value) -> StreamInfo:
    handle = int(handle_value)
    byte_size = handle_value.type.sizeof
    description = _handle_description(handle, byte_size)
    invalid_handle = (1 << (byte_size * 8)) - 1
    if handle == invalid_handle:
        return StreamInfo(handle, description, None, None, None, None, None)

    capture_status = _query_stream_property(handle, _CUDA_STREAM_IS_CAPTURING, "int")
    is_capturing = (
        capture_status == _CUDA_STREAM_CAPTURE_STATUS_ACTIVE
        if capture_status is not None
        else None
    )
    # Invoking the metadata query paths below as inferior calls invalidates
    # global graph capture in GDB and LLDB. Preserve it until capture ends.
    if capture_status != _CUDA_STREAM_CAPTURE_STATUS_NONE:
        return StreamInfo(handle, description, None, None, None, is_capturing, None)

    return StreamInfo(
        handle,
        description,
        _query_stream_id(handle),
        _query_stream_device(handle),
        _query_stream_property(handle, "cudaStreamGetPriority", "int"),
        is_capturing,
        _query_stream_property(handle, "cudaStreamGetFlags", "unsigned int"),
    )


class StreamPrinter:
    """Expose CUDA stream metadata to GDB."""

    def __init__(self, value: gdb.Value, type_name: str) -> None:
        self.value = value
        self.type_name = type_name
        self.info = _stream_info(value["__stream"])

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        properties = (
            ("device", self.info.device, "int"),
            ("priority", self.info.priority, "int"),
            ("is_capturing", self.info.is_capturing, "bool"),
            ("flags", self.info.flags, "unsigned int"),
        )
        for name, property_value, property_type in properties:
            if property_value is None:
                continue
            try:
                yield (
                    name,
                    gdb.Value(property_value).cast(gdb.lookup_type(property_type)),
                )
            except (gdb.error, TypeError, ValueError):
                continue

    def to_string(self) -> str:
        unique_id = (
            str(self.info.unique_id)
            if self.info.unique_id is not None
            else "unavailable"
        )
        return (
            f"{self.type_name} handle={self.info.handle_description}, "
            f"unique_id={unique_id}"
        )


class StreamPrinterLookup(gdb.printing.PrettyPrinter):
    """Select printers for cuda::stream and cuda::stream_ref."""

    def __init__(self) -> None:
        super().__init__("cuda::stream")

    def __call__(self, value: gdb.Value) -> StreamPrinter | None:
        type_name = _stream_type_name(value.type)
        if type_name is None:
            return None
        try:
            return StreamPrinter(value, type_name)
        except (gdb.error, TypeError, ValueError):
            return None


def register(objfile: ModuleType) -> None:
    """Register CUDA stream formatters with GDB."""
    gdb.printing.register_pretty_printer(objfile, StreamPrinterLookup(), replace=True)
