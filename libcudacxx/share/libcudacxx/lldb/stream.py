# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printers for cuda::stream and cuda::stream_ref."""

from __future__ import annotations

import re

import lldb

_STREAM_PATTERN = re.compile(r"^cuda::(?:stream|stream_ref)$")
# These are the public values of cudaStreamLegacy and cudaStreamPerThread.
# The debugger expression parser does not necessarily expose the macros.
_CUDA_STREAM_LEGACY_HANDLE = 1
_CUDA_STREAM_PER_THREAD_HANDLE = 2
InternalDict = dict[str, object]


def is_cuda_stream(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = (
        value_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""
    )
    return _STREAM_PATTERN.fullmatch(type_name) is not None


def _stream_handle(value: lldb.SBValue) -> lldb.SBValue:
    value = value.GetNonSyntheticValue()
    handle = value.GetChildMemberWithName("__stream")
    if handle.IsValid():
        return handle

    for index in range(value.GetNumChildren()):
        base = value.GetChildAtIndex(index)
        if not base.IsValid():
            continue
        handle = base.GetChildMemberWithName("__stream")
        if handle.IsValid():
            return handle
    return lldb.SBValue()


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


def stream_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str | None:
    handle = _stream_handle(value)
    if not handle.IsValid() or handle.GetError().Fail():
        return None
    description = _handle_description(
        handle.GetValueAsUnsigned(0), handle.GetType().GetByteSize()
    )
    return f"handle={description}"


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register CUDA stream formatters in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --python-function "
        f"{module}.stream_summary --recognizer-function {module}.is_cuda_stream"
    )
