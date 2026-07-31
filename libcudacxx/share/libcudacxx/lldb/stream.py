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
_SUMMARY_UNIQUE_ID_CHILD = "__cccl_summary_unique_id"
_UINT32_BITS = 32
_UINT32_MASK = (1 << _UINT32_BITS) - 1
_UINT64_BITS = 64
_UINT64_MASK = (1 << _UINT64_BITS) - 1
InternalDict = dict[str, object]


class StreamInfo(NamedTuple):
    device: int | None
    priority: int | None
    is_capturing: bool | None
    flags: int | None


class StreamSnapshot(NamedTuple):
    unique_id: int | None
    info: StreamInfo


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


def _query_stream_property_32(
    value: lldb.SBValue,
    handle: int,
    function: str,
    output_type: str,
    *,
    signed: bool,
) -> int | None:
    # Expression evaluation dominates LLDB formatter runtime. Pack the CUDA
    # status and result into one scalar to avoid inferior malloc/read/free calls.
    result = _evaluate(
        value,
        "(unsigned long long)([](cudaStream_t stream) { "
        f"{output_type} value{{}}; "
        f"int status = (int){function}(stream, &value); "
        "return ((unsigned long long)(unsigned int)status << 32) "
        "| (unsigned int)value; "
        f"}})((cudaStream_t){handle:#x})",
    )
    if not result.IsValid() or result.GetError().Fail():
        return None

    packed = result.GetValueAsUnsigned((1 << 64) - 1)
    status = packed >> _UINT32_BITS
    if status != 0:
        return None

    output = packed & _UINT32_MASK
    if signed and output & (1 << (_UINT32_BITS - 1)):
        return output - (1 << _UINT32_BITS)
    return output


def _query_stream_property(
    value: lldb.SBValue,
    handle: int,
    function: str,
    output_type: str,
    *,
    signed: bool,
) -> int | None:
    if output_type in ("int", "unsigned int"):
        return _query_stream_property_32(
            value, handle, function, output_type, signed=signed
        )

    if output_type == "unsigned long long":
        result = _evaluate(
            value,
            "(unsigned __int128)([](cudaStream_t stream) { "
            "unsigned long long value{}; "
            f"int status = (int){function}(stream, &value); "
            "return ((unsigned __int128)(unsigned int)status << 64) "
            "| value; "
            f"}})((cudaStream_t){handle:#x})",
        )
        if result.IsValid() and not result.GetError().Fail():
            raw_value = result.GetValue()
            if raw_value is not None:
                packed = int(raw_value, 0)
                if packed >> _UINT64_BITS == 0:
                    return packed & _UINT64_MASK

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
    result = _evaluate(
        value,
        "(unsigned long long)([](void* stream) { "
        "void* context{}; "
        "int status = "
        "((int (*)(void*, void**))cuStreamGetCtx)(stream, &context); "
        "if (status != 0) "
        "return (unsigned long long)(unsigned int)status << 32; "
        "status = ((int (*)(void*))cuCtxPushCurrent)(context); "
        "if (status != 0) "
        "return (unsigned long long)(unsigned int)status << 32; "
        "int device{}; "
        "status = ((int (*)(int*))cuCtxGetDevice)(&device); "
        "void* popped{}; "
        "(void)((int (*)(void**))cuCtxPopCurrent)(&popped); "
        "return ((unsigned long long)(unsigned int)status << 32) "
        "| (unsigned int)device; "
        f"}})((void*){handle:#x})",
    )
    if not result.IsValid() or result.GetError().Fail():
        return None

    packed = result.GetValueAsUnsigned((1 << 64) - 1)
    status = packed >> _UINT32_BITS
    if status != 0:
        return None

    device = packed & _UINT32_MASK
    if device & (1 << (_UINT32_BITS - 1)):
        return device - (1 << _UINT32_BITS)
    return device


def _stream_info(value: lldb.SBValue, handle: int) -> StreamInfo:
    capture_status = _query_stream_property(
        value, handle, _CUDA_STREAM_IS_CAPTURING, "int", signed=True
    )
    is_capturing = (
        capture_status == _CUDA_STREAM_CAPTURE_STATUS_ACTIVE
        if capture_status is not None
        else None
    )
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
            value,
            handle,
            "cudaStreamGetFlags",
            "unsigned int",
            signed=False,
        ),
    )


def _query_stream_snapshot(value: lldb.SBValue, handle: int) -> StreamSnapshot | None:
    # A wide scalar carries every value and validity bit without target
    # allocations. The caller retains the regular query path as a fallback for
    # expression parsers that do not support _BitInt.
    result = _evaluate(
        value,
        "(unsigned _BitInt(256))(([](cudaStream_t stream) { "
        "using U256 = unsigned _BitInt(256); "
        "int capture_status{}; "
        f"if ((int){_CUDA_STREAM_IS_CAPTURING}(stream, &capture_status) != 0) "
        "return (U256)0; "
        "U256 result = (U256)1 << 160; "
        "if (capture_status == 1) result |= (U256)1 << 161; "
        "if (capture_status != 0) return result | ((U256)1 << 162); "
        "unsigned long long unique_id{}; "
        f"if ((int){_CU_STREAM_GET_ID}((void*)stream, &unique_id) == 0) "
        "result |= ((U256)1 << 163) | unique_id; "
        "int device{}; "
        "void* context{}; "
        "int status = "
        "((int (*)(void*, void**))cuStreamGetCtx)((void*)stream, &context); "
        "if (status == 0) { "
        "status = ((int (*)(void*))cuCtxPushCurrent)(context); "
        "if (status == 0) { "
        "status = ((int (*)(int*))cuCtxGetDevice)(&device); "
        "void* popped{}; "
        "(void)((int (*)(void**))cuCtxPopCurrent)(&popped); "
        "} "
        "} "
        "if (status != 0) status = (int)cudaStreamGetDevice(stream, &device); "
        "if (status == 0) "
        "result |= ((U256)1 << 164) | ((U256)(unsigned int)device << 64); "
        "int priority{}; "
        "status = (int)cudaStreamGetPriority(stream, &priority); "
        "if (status == 0) "
        "result |= ((U256)1 << 165) | ((U256)(unsigned int)priority << 96); "
        "unsigned int flags{}; "
        "status = (int)cudaStreamGetFlags(stream, &flags); "
        "if (status == 0) "
        "result |= ((U256)1 << 166) | ((U256)flags << 128); "
        "return result; "
        f"}})((cudaStream_t){handle:#x}))",
    )
    if not result.IsValid() or result.GetError().Fail():
        return None
    raw_value = result.GetValue()
    if raw_value is None:
        return None
    packed = int(raw_value, 0)

    if not packed & (1 << 160):
        return StreamSnapshot(None, StreamInfo(None, None, None, None))
    is_capturing = bool(packed & (1 << 161))
    if packed & (1 << 162):
        return StreamSnapshot(None, StreamInfo(None, None, is_capturing, None))

    def signed32(raw: int) -> int:
        return raw - (1 << _UINT32_BITS) if raw & (1 << 31) else raw

    unique_id = (
        packed & _UINT64_MASK
        if packed & (1 << 163)
        else _query_stream_property(
            value,
            handle,
            "cudaStreamGetId",
            "unsigned long long",
            signed=False,
        )
    )
    device = signed32((packed >> 64) & _UINT32_MASK) if packed & (1 << 164) else None
    priority = signed32((packed >> 96) & _UINT32_MASK) if packed & (1 << 165) else None
    flags = (packed >> 128) & _UINT32_MASK if packed & (1 << 166) else None
    return StreamSnapshot(unique_id, StreamInfo(device, priority, is_capturing, flags))


def _summary_unique_id(value: lldb.SBValue, handle: int) -> int | None:
    capture_status = _query_stream_property(
        value, handle, _CUDA_STREAM_IS_CAPTURING, "int", signed=True
    )
    if capture_status != _CUDA_STREAM_CAPTURE_STATUS_NONE:
        return None
    return _unique_id(value, handle)


def stream_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str | None:
    handle = _stream_handle(value)
    if not handle.IsValid() or handle.GetError().Fail():
        return None
    raw_handle = handle.GetValueAsUnsigned(0)
    byte_size = handle.GetType().GetByteSize()
    description = _handle_description(raw_handle, byte_size)
    invalid_handle = (1 << (byte_size * 8)) - 1
    unique_id = None
    if raw_handle != invalid_handle:
        synthetic_value = value.GetSyntheticValue()
        if synthetic_value.IsValid():
            unique_id_child = synthetic_value.GetChildMemberWithName(
                _SUMMARY_UNIQUE_ID_CHILD
            )
            if unique_id_child.IsValid() and not unique_id_child.GetError().Fail():
                unique_id = unique_id_child.GetValueAsUnsigned(0)
        else:
            unique_id = _summary_unique_id(value, raw_handle)
    unique_id_description = str(unique_id) if unique_id is not None else "unavailable"
    return f"handle={description}, unique_id={unique_id_description}"


class StreamSyntheticProvider:
    """Expose CUDA stream properties as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.value = value.GetNonSyntheticValue()
        self.children: list[tuple[str, int, str]] = []
        self.summary_unique_id: int | None = None
        self.stop_id: int | None = None
        self.initialized = False

    def update(self) -> bool:
        process = self.value.GetProcess()
        stop_id = process.GetStopID() if process.IsValid() else None
        if self.initialized and stop_id == self.stop_id:
            return True

        self.children = []
        self.summary_unique_id = None
        self.stop_id = stop_id
        self.initialized = True
        handle = _stream_handle(self.value)
        if not handle.IsValid() or handle.GetError().Fail():
            return False

        raw_handle = handle.GetValueAsUnsigned(0)
        byte_size = handle.GetType().GetByteSize()
        if raw_handle == (1 << (byte_size * 8)) - 1:
            return True

        snapshot = _query_stream_snapshot(self.value, raw_handle)
        if snapshot is None:
            snapshot = StreamSnapshot(
                _summary_unique_id(self.value, raw_handle),
                _stream_info(self.value, raw_handle),
            )
        self.summary_unique_id = snapshot.unique_id
        info = snapshot.info
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
        # The new children invalidate LLDB's cache. Subsequent update calls at
        # this stop return True above so LLDB can reuse them.
        return False

    def num_children(self) -> int:
        return len(self.children)

    def has_children(self) -> bool:
        return bool(self.children)

    def get_child_index(self, name: str) -> int:
        if name == _SUMMARY_UNIQUE_ID_CHILD and self.summary_unique_id is not None:
            return len(self.children)
        for index, (child_name, _, _) in enumerate(self.children):
            if child_name == name:
                return index
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index < 0 or index > len(self.children):
            return None
        if index == len(self.children):
            if self.summary_unique_id is None:
                return None
            name = _SUMMARY_UNIQUE_ID_CHILD
            property_value = self.summary_unique_id
            property_type = "unsigned long long"
        else:
            name, property_value, property_type = self.children[index]
        child_type = self.value.GetTarget().FindFirstType(property_type)
        if not child_type.IsValid():
            return None
        expression_value = (
            int(property_value) if property_type == "bool" else property_value
        )
        byte_order = self.value.GetTarget().GetByteOrder()
        python_byte_order = "big" if byte_order == lldb.eByteOrderBig else "little"
        raw_data = int(expression_value).to_bytes(
            child_type.GetByteSize(),
            byteorder=python_byte_order,
            signed=property_type == "int",
        )
        data = lldb.SBData()
        error = lldb.SBError()
        data.SetData(
            error,
            raw_data,
            byte_order,
            self.value.GetTarget().GetAddressByteSize(),
        )
        if error.Fail():
            return None
        return self.value.CreateValueFromData(name, data, child_type)


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
