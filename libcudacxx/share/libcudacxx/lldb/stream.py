# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printers for cuda::stream and cuda::stream_ref."""

from __future__ import annotations

import re
from typing import NamedTuple

import cccl_common

import lldb

_STREAM_PATTERN = re.compile(r"^cuda::(?:stream|stream_ref)$")
# These are the public values of cudaStreamLegacy and cudaStreamPerThread.
# The debugger expression parser does not necessarily expose the macros.
_CUDA_STREAM_LEGACY_HANDLE = 1
_CUDA_STREAM_PER_THREAD_HANDLE = 2
_CUDA_STREAM_CAPTURE_STATUS_NONE = 0
_CUDA_STREAM_CAPTURE_STATUS_ACTIVE = 1
_SUMMARY_UNIQUE_ID_CHILD = "__cccl_summary_unique_id"
_UINT32_BITS = 32
_UINT32_MASK = (1 << _UINT32_BITS) - 1
_CAPTURE_STATUS_VALID = 1 << 0
_UNIQUE_ID_VALID = 1 << 1
_DEVICE_VALID = 1 << 2
_PRIORITY_VALID = 1 << 3
_FLAGS_VALID = 1 << 4
InternalDict = dict[str, object]


# LLDB cannot materialize a locally declared struct returned by an expression.
# A Clang vector keeps this to one inferior call and exposes six stable lanes:
# unique ID, device, priority, capture status, flags, and a validity mask.
_STREAM_SNAPSHOT_EXPRESSION = """
(unsigned long long __attribute__((ext_vector_type(6))))(([](cudaStream_t stream) {
  using Snapshot = unsigned long long __attribute__((ext_vector_type(6)));
  constexpr unsigned long long capture_status_valid = 1 << 0;
  constexpr unsigned long long unique_id_valid = 1 << 1;
  constexpr unsigned long long device_valid = 1 << 2;
  constexpr unsigned long long priority_valid = 1 << 3;
  constexpr unsigned long long flags_valid = 1 << 4;

  void* original_context{};
  const int context_status =
    ((int (*)(void**))cuCtxGetCurrent)(&original_context);
  int capture_status{};
  unsigned long long unique_id{};
  int device{};
  int priority{};
  unsigned int flags{};
  unsigned long long validity{};

  if (context_status == 0
      && ((int (*)(cudaStream_t, int*))cudaStreamIsCapturing)(
           stream, &capture_status) == 0) {
    validity |= capture_status_valid;
    if (capture_status == 0) {
      if (((int (*)(void*, unsigned long long*))cuStreamGetId)(
            (void*)stream, &unique_id) == 0) {
        validity |= unique_id_valid;
      }
%s
      if ((int)cudaStreamGetPriority(stream, &priority) == 0) {
        validity |= priority_valid;
      }
      if ((int)cudaStreamGetFlags(stream, &flags) == 0) {
        validity |= flags_valid;
      }
    }
  }

  if (context_status == 0 && original_context == nullptr) {
    (void)((int (*)(void*))cuCtxSetCurrent)(nullptr);
  }

  return Snapshot{
    unique_id,
    (unsigned int)device,
    (unsigned int)priority,
    (unsigned int)capture_status,
    flags,
    validity,
  };
})((cudaStream_t)%#x))
"""

_STREAM_DEVICE_QUERY = """
  if ((int)cudaStreamGetDevice(stream, &device) == 0) {
    validity |= device_valid;
  }
"""


class StreamInfo(NamedTuple):
    device: int | None
    priority: int | None
    is_capturing: bool | None
    flags: int | None


class StreamSnapshot(NamedTuple):
    unique_id: int | None
    info: StreamInfo


def is_cuda_stream(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    return _STREAM_PATTERN.fullmatch(type_name) is not None


def _stream_handle(value: lldb.SBValue) -> lldb.SBValue:
    value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
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


def _snapshot_values(
    result: lldb.SBValue,
) -> tuple[int, int, int, int, int, int] | None:
    if result.GetNumChildren() != 6:
        return None

    lanes: list[int] = []
    for index in range(6):
        lane = result.GetChildAtIndex(index)
        if not lane.IsValid() or lane.GetError().Fail():
            return None
        lanes.append(lane.GetValueAsUnsigned(0))
    return lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5]


def _signed32(value: int) -> int:
    value &= _UINT32_MASK
    return value - (1 << _UINT32_BITS) if value & (1 << 31) else value


def _query_stream_snapshot(value: lldb.SBValue, handle: int) -> StreamSnapshot | None:
    # One vector-valued expression avoids target allocations and
    # debugger-visible state changes.
    result = _evaluate(
        value, _STREAM_SNAPSHOT_EXPRESSION % (_STREAM_DEVICE_QUERY, handle)
    )
    if not result.IsValid() or result.GetError().Fail():
        # cudaStreamGetDevice was added in CUDA 12.8. Older runtimes can still
        # provide the remaining metadata without changing the current context.
        result = _evaluate(value, _STREAM_SNAPSHOT_EXPRESSION % ("", handle))
    if not result.IsValid() or result.GetError().Fail():
        return None

    snapshot = _snapshot_values(result)
    if snapshot is None:
        return None
    unique_id, device, priority, capture_status, flags, validity = snapshot

    if not validity & _CAPTURE_STATUS_VALID:
        return StreamSnapshot(None, StreamInfo(None, None, None, None))

    capture_status = _signed32(capture_status)
    is_capturing = capture_status == _CUDA_STREAM_CAPTURE_STATUS_ACTIVE
    if capture_status != _CUDA_STREAM_CAPTURE_STATUS_NONE:
        return StreamSnapshot(None, StreamInfo(None, None, is_capturing, None))

    return StreamSnapshot(
        unique_id if validity & _UNIQUE_ID_VALID else None,
        StreamInfo(
            _signed32(device) if validity & _DEVICE_VALID else None,
            _signed32(priority) if validity & _PRIORITY_VALID else None,
            is_capturing,
            flags if validity & _FLAGS_VALID else None,
        ),
    )


def stream_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str | None:
    value = cccl_common.strip_reference_value(value)
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
            snapshot = _query_stream_snapshot(value, raw_handle)
            if snapshot is not None:
                unique_id = snapshot.unique_id
    unique_id_description = str(unique_id) if unique_id is not None else "unavailable"
    return f"handle={description}, unique_id={unique_id_description}"


class StreamSyntheticProvider:
    """Expose CUDA stream properties as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
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
            return True
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
