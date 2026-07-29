# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printers for cuda::stream and cuda::stream_ref."""

from __future__ import annotations

import re
from typing import NamedTuple

import lldb

_STREAM_PATTERN = re.compile(r"^cuda::(?:stream|stream_ref)$")
# These are the public values of cudaStreamLegacy and cudaStreamPerThread.
# The debugger expression parser does not necessarily expose the macros.
_CUDA_STREAM_LEGACY_HANDLE = 1
_CUDA_STREAM_PER_THREAD_HANDLE = 2
_CUDA_STREAM_CAPTURE_STATUS_NONE = 0
_CUDA_STREAM_CAPTURE_STATUS_ACTIVE = 1
_CUDA_STREAM_IS_CAPTURING = "((int (*)(cudaStream_t, int*))cudaStreamIsCapturing)"
_CU_STREAM_GET_ID = "((int (*)(void*, unsigned long long*))cuStreamGetId)"
InternalDict = dict[str, object]


class StreamInfo(NamedTuple):
    device: int | None
    priority: int | None
    is_capturing: bool | None
    flags: int | None


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


def _evaluate(value: lldb.SBValue, expression: str) -> lldb.SBValue:
    frame = value.GetFrame()
    if not frame.IsValid():
        return lldb.SBValue()
    options = lldb.SBExpressionOptions()
    options.SetIgnoreBreakpoints(True)
    options.SetUnwindOnError(True)
    return frame.EvaluateExpression(expression, options)


def _query_stream_property(
    value: lldb.SBValue,
    handle: int,
    function: str,
    output_type: str,
    *,
    signed: bool,
) -> int | None:
    output = _evaluate(value, f"({output_type}*)malloc(sizeof({output_type}))")
    if not output.IsValid() or output.GetError().Fail():
        return None

    address = output.GetValueAsUnsigned(0)
    if address == 0:
        return None

    try:
        status = _evaluate(
            value,
            f"(int){function}((cudaStream_t){handle:#x}, ({output_type}*){address:#x})",
        )
        if (
            not status.IsValid()
            or status.GetError().Fail()
            or status.GetValueAsSigned(-1) != 0
        ):
            return None

        result = _evaluate(value, f"*({output_type}*){address:#x}")
        if not result.IsValid() or result.GetError().Fail():
            return None
        if signed:
            return result.GetValueAsSigned(0)
        return result.GetValueAsUnsigned(0)
    finally:
        _evaluate(value, f"(void)free((void*){address:#x})")


def _unique_id(value: lldb.SBValue, handle: int) -> int | None:
    unique_id = _query_stream_property(
        value,
        handle,
        _CU_STREAM_GET_ID,
        "unsigned long long",
        signed=False,
    )
    if unique_id is not None:
        return unique_id
    return _query_stream_property(
        value,
        handle,
        "cudaStreamGetId",
        "unsigned long long",
        signed=False,
    )


def _stream_device(value: lldb.SBValue, handle: int) -> int | None:
    output = _evaluate(value, "(void**)malloc(sizeof(void*))")
    if not output.IsValid() or output.GetError().Fail():
        return None

    address = output.GetValueAsUnsigned(0)
    if address == 0:
        return None

    context_pushed = False
    try:
        status = _evaluate(
            value,
            "(int)((int (*)(void*, void**))cuStreamGetCtx)"
            f"((void*){handle:#x}, (void**){address:#x})",
        )
        if (
            not status.IsValid()
            or status.GetError().Fail()
            or status.GetValueAsSigned(-1) != 0
        ):
            return None

        context_value = _evaluate(value, f"*(void**){address:#x}")
        if not context_value.IsValid() or context_value.GetError().Fail():
            return None
        context = context_value.GetValueAsUnsigned(0)

        status = _evaluate(
            value,
            f"(int)((int (*)(void*))cuCtxPushCurrent)((void*){context:#x})",
        )
        if (
            not status.IsValid()
            or status.GetError().Fail()
            or status.GetValueAsSigned(-1) != 0
        ):
            return None
        context_pushed = True

        status = _evaluate(
            value,
            f"(int)((int (*)(int*))cuCtxGetDevice)((int*){address:#x})",
        )
        if (
            not status.IsValid()
            or status.GetError().Fail()
            or status.GetValueAsSigned(-1) != 0
        ):
            return None

        device = _evaluate(value, f"*(int*){address:#x}")
        if not device.IsValid() or device.GetError().Fail():
            return None
        return device.GetValueAsSigned(0)
    finally:
        if context_pushed:
            _evaluate(
                value,
                f"(int)((int (*)(void**))cuCtxPopCurrent)((void**){address:#x})",
            )
        _evaluate(value, f"(void)free((void*){address:#x})")


def _stream_info(value: lldb.SBValue, handle: int) -> StreamInfo:
    capture_status = _query_stream_property(
        value, handle, _CUDA_STREAM_IS_CAPTURING, "int", signed=True
    )
    is_capturing = (
        capture_status == _CUDA_STREAM_CAPTURE_STATUS_ACTIVE
        if capture_status is not None
        else None
    )
    # The other stream queries can invalidate active graph capture. If capture
    # state is active, invalidated, or unavailable, preserve the inferior state.
    if capture_status != _CUDA_STREAM_CAPTURE_STATUS_NONE:
        return StreamInfo(None, None, is_capturing, None)

    device = _stream_device(value, handle)
    if device is None:
        device = _query_stream_property(
            value, handle, "cudaStreamGetDevice", "int", signed=True
        )
    return StreamInfo(
        device,
        _query_stream_property(
            value, handle, "cudaStreamGetPriority", "int", signed=True
        ),
        is_capturing,
        _query_stream_property(
            value, handle, "cudaStreamGetFlags", "unsigned int", signed=False
        ),
    )


def stream_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str | None:
    handle = _stream_handle(value)
    if not handle.IsValid() or handle.GetError().Fail():
        return None
    raw_handle = handle.GetValueAsUnsigned(0)
    byte_size = handle.GetType().GetByteSize()
    description = _handle_description(raw_handle, byte_size)
    invalid_handle = (1 << (byte_size * 8)) - 1
    capture_status = (
        None
        if raw_handle == invalid_handle
        else _query_stream_property(
            value, raw_handle, _CUDA_STREAM_IS_CAPTURING, "int", signed=True
        )
    )
    unique_id = (
        None
        if (
            raw_handle == invalid_handle
            or capture_status != _CUDA_STREAM_CAPTURE_STATUS_NONE
        )
        else _unique_id(value, raw_handle)
    )
    unique_id_description = str(unique_id) if unique_id is not None else "unavailable"
    return f"handle={description}, unique_id={unique_id_description}"


class StreamSyntheticProvider:
    """Expose CUDA stream properties as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.value = value.GetNonSyntheticValue()
        self.children: list[tuple[str, int, str]] = []
        self.update()

    def update(self) -> bool:
        self.children = []
        handle = _stream_handle(self.value)
        if not handle.IsValid() or handle.GetError().Fail():
            return False

        raw_handle = handle.GetValueAsUnsigned(0)
        byte_size = handle.GetType().GetByteSize()
        if raw_handle == (1 << (byte_size * 8)) - 1:
            return True

        info = _stream_info(self.value, raw_handle)
        properties = (
            ("device", info.device, "int"),
            ("priority", info.priority, "int"),
            ("is_capturing", info.is_capturing, "bool"),
            ("flags", info.flags, "unsigned int"),
        )
        self.children = [
            (name, property_value, property_type)
            for name, property_value, property_type in properties
            if property_value is not None
        ]
        return True

    def num_children(self) -> int:
        return len(self.children)

    def has_children(self) -> bool:
        return bool(self.children)

    def get_child_index(self, name: str) -> int:
        for index, (child_name, _, _) in enumerate(self.children):
            if child_name == name:
                return index
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index < 0 or index >= len(self.children):
            return None
        name, property_value, property_type = self.children[index]
        expression_value = (
            int(property_value) if property_type == "bool" else property_value
        )
        return self.value.CreateValueFromExpression(
            name, f"({property_type}){expression_value}"
        )


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register CUDA stream formatters in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function "
        f"{module}.stream_summary --recognizer-function {module}.is_cuda_stream"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class "
        f"{module}.StreamSyntheticProvider --recognizer-function "
        f"{module}.is_cuda_stream"
    )
