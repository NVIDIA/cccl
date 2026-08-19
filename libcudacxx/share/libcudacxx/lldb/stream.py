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
InternalDict = dict[str, object]


# Compiled once per process, then called per stream: an inferior round trip costs
# far more than the queries, so one call collects every field.
#
# The queries use the driver API. libcuda is always shared, so its symbols are
# always present; the cudart equivalents are absent from a statically linked
# runtime unless the program itself calls them, which silently drops a field.
#
# An unresolved identifier fails the whole unit, so the device query is spliced in
# only when its symbol resolves. The rest predate the oldest supported driver.
_STREAM_SNAPSHOT_DEFINITION = """
struct __cccl_stream_snapshot_result
{
  unsigned long long unique_id;
  int device;
  int priority;
  int capture_status;
  unsigned int flags;
  bool has_unique_id;
  bool has_device;
  bool has_priority;
  bool has_capture_status;
  bool has_flags;
};

extern "C" __cccl_stream_snapshot_result __cccl_stream_snapshot(void* stream)
{
  __cccl_stream_snapshot_result result = {};
  void* original_context = 0;
  const int context_status =
    ((int (*)(void**))cuCtxGetCurrent)(&original_context);

  if (context_status == 0
      && ((int (*)(void*, int*))cuStreamIsCapturing)(
           stream, &result.capture_status) == 0) {
    result.has_capture_status = true;
    if (result.capture_status == 0) {
      result.has_unique_id =
        ((int (*)(void*, unsigned long long*))cuStreamGetId)(
          stream, &result.unique_id) == 0;
%s
      result.has_priority =
        ((int (*)(void*, int*))cuStreamGetPriority)(
          stream, &result.priority) == 0;
      result.has_flags =
        ((int (*)(void*, unsigned int*))cuStreamGetFlags)(
          stream, &result.flags) == 0;
    }
  }

  if (context_status == 0 && original_context == 0) {
    (void)((int (*)(void*))cuCtxSetCurrent)(0);
  }
  return result;
}
"""

# cuStreamGetDevice arrived in CUDA 12.8. Reporting no device on an older driver
# is safer than changing the current CUDA context from a formatter.
_STREAM_DEVICE_QUERY = """
      result.has_device =
        ((int (*)(void*, int*))cuStreamGetDevice)(
          stream, &result.device) == 0;
"""
# Each value member has a has_<name> companion. int members read as signed.
_SNAPSHOT_FIELDS = {
    "unique_id": False,
    "device": True,
    "priority": True,
    "capture_status": True,
    "flags": False,
}


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


def _snapshot_fields(value: lldb.SBValue, handle: int) -> dict[str, int] | None:
    """Compile the snapshot if needed, then run it for one handle."""
    frame = value.GetFrame()
    process = value.GetProcess()
    target = value.GetTarget()
    if not frame.IsValid() or not process.IsValid() or not target.IsValid():
        return None

    process_id = process.GetUniqueID()
    if _snapshot_fields.process_id != process_id:
        _snapshot_fields.process_id = process_id
        definition = _STREAM_SNAPSHOT_DEFINITION % (
            _STREAM_DEVICE_QUERY
            if target.FindSymbols("cuStreamGetDevice").GetSize()
            else ""
        )
        top_level = lldb.SBExpressionOptions()
        top_level.SetIgnoreBreakpoints(True)
        top_level.SetUnwindOnError(True)
        top_level.SetTopLevel(True)
        top_level.SetLanguage(lldb.eLanguageTypeC_plus_plus)
        frame.EvaluateExpression(definition, top_level)
        # A top-level expression has no result, and it goes to the JIT, so it
        # appears in no symbol table. A call is the only proof that it compiled.
        probe = lldb.SBExpressionOptions()
        probe.SetIgnoreBreakpoints(True)
        probe.SetUnwindOnError(True)
        _snapshot_fields.installed = (
            frame.EvaluateExpression("__cccl_stream_snapshot((void*)0)", probe)
            .GetError()
            .Success()
        )
    if not _snapshot_fields.installed:
        return None

    options = lldb.SBExpressionOptions()
    options.SetIgnoreBreakpoints(True)
    options.SetUnwindOnError(True)
    result = frame.EvaluateExpression(
        f"__cccl_stream_snapshot((void*){handle:#x})", options
    )
    if result.GetError().Fail():
        return None

    fields: dict[str, int] = {}
    for name, signed in _SNAPSHOT_FIELDS.items():
        for member_name in (name, f"has_{name}"):
            member = result.GetChildMemberWithName(member_name)
            if not member.IsValid() or member.GetError().Fail():
                return None
            fields[member_name] = (
                member.GetValueAsSigned(0)
                if signed and member_name == name
                else member.GetValueAsUnsigned(0)
            )
    return fields


_snapshot_fields.process_id = None
_snapshot_fields.installed = False


def _query_stream_snapshot(value: lldb.SBValue, handle: int) -> StreamSnapshot | None:
    snapshot = _snapshot_fields(value, handle)
    if snapshot is None:
        return None

    if not snapshot["has_capture_status"]:
        return StreamSnapshot(None, StreamInfo(None, None, None, None))

    capture_status = snapshot["capture_status"]
    is_capturing = capture_status == _CUDA_STREAM_CAPTURE_STATUS_ACTIVE
    if capture_status != _CUDA_STREAM_CAPTURE_STATUS_NONE:
        return StreamSnapshot(None, StreamInfo(None, None, is_capturing, None))

    return StreamSnapshot(
        snapshot["unique_id"] if snapshot["has_unique_id"] else None,
        StreamInfo(
            snapshot["device"] if snapshot["has_device"] else None,
            snapshot["priority"] if snapshot["has_priority"] else None,
            is_capturing,
            snapshot["flags"] if snapshot["has_flags"] else None,
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
