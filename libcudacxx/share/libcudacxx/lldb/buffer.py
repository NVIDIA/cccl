# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::buffer."""

from __future__ import annotations

import re
from typing import NamedTuple

import cccl_common
import memory_resource

import lldb

_BUFFER_PATTERN = re.compile(r"^cuda::buffer<.+>$")
InternalDict = dict[str, object]


class BufferInfo(NamedTuple):
    size: int
    data_address: int
    value_type: lldb.SBType
    accessibility: str
    memory_resource: lldb.SBValue
    stream: lldb.SBValue
    alignment: lldb.SBValue


def is_cuda_buffer(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    return _BUFFER_PATTERN.fullmatch(type_name) is not None


def _buffer_info(value: lldb.SBValue) -> BufferInfo | None:
    value = cccl_common.strip_reference_value(value)
    value = value.GetNonSyntheticValue()
    storage = value.GetChildMemberWithName("__buf_")
    if not storage.IsValid():
        return None

    count = storage.GetChildMemberWithName("__count_")
    memory_resource = storage.GetChildMemberWithName("__mr_")
    stream_ref = storage.GetChildMemberWithName("__stream_")
    stream = stream_ref.GetChildMemberWithName("__stream")
    alignment = storage.GetChildMemberWithName("__alignment_")
    allocation = storage.GetChildMemberWithName("__buf_")
    if not all(
        child.IsValid()
        for child in (count, memory_resource, stream, alignment, allocation)
    ):
        return None

    # A source-level alias can hide the accessibility properties from
    # GetTypeName(), so use the canonical public type for all property checks.
    buffer_type = value.GetType().GetCanonicalType().GetUnqualifiedType()
    value_type = buffer_type.GetTemplateArgumentType(0)
    if not value_type.IsValid():
        return None

    type_name = buffer_type.GetDisplayTypeName() or ""
    host_accessible = "host_accessible" in type_name
    device_accessible = "device_accessible" in type_name
    if host_accessible and device_accessible:
        accessibility = "host/device"
    elif device_accessible:
        accessibility = "device"
    elif host_accessible:
        accessibility = "host"
    else:
        accessibility = "unknown"
    size = count.GetValueAsUnsigned(0)
    align = alignment.GetValueAsUnsigned(1)
    raw_address = allocation.GetValueAsUnsigned(0)
    data_address = (raw_address + align - 1) & ~(align - 1)
    return BufferInfo(
        size,
        data_address,
        value_type,
        accessibility,
        memory_resource,
        stream,
        alignment,
    )


def buffer_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str | None:
    info = _buffer_info(value)
    if info is None:
        return None
    resource = memory_resource.memory_resource_description(info.memory_resource)
    stream = info.stream.GetValueAsUnsigned(0)
    alignment = info.alignment.GetValueAsUnsigned(0)
    return (
        f"mr={resource}, stream={stream:#x}, size={info.size}, align={alignment}, "
        f"data={info.data_address:#x} ({info.accessibility})"
    )


class BufferSyntheticProvider:
    """Expose cuda::buffer elements as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.declared_type = value.GetType()
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.stop_id: int | None = None
        self.update()

    def _current_stop_id(self) -> int | None:
        process = self.value.GetProcess()
        return process.GetStopID() if process.IsValid() else None

    def update(self) -> bool:
        # update() runs before every read of a synthetic child, so one print
        # causes hundreds of calls. Restage only after the process stops again.
        # Anything that changes target memory must resume and re-stop it, so this
        # cannot report stale data.
        if self.stop_id is not None and self._current_stop_id() == self.stop_id:
            return True

        self.host_address = 0
        self.size = 0
        self.value_type = lldb.SBType()
        self.value_size = 0

        info = _buffer_info(self.value)
        if info is None:
            return False

        self.value_type = info.value_type
        self.value_size = self.value_type.GetByteSize()
        # A formatter must not raise into the debugger; report no elements.
        try:
            self.host_address = cccl_common.stage_device_memory(
                self.value, info.data_address, info.size * self.value_size
            )
            self.size = info.size
        except cccl_common.StagingError:
            pass
        # Staging advances the stop ID, so record what the next call sees.
        self.stop_id = self._current_stop_id()
        return True

    def num_children(self) -> int:
        return self.size

    def has_children(self) -> bool:
        return self.size != 0

    def get_type_name(self) -> str:
        # STL element access can preserve an alloc_traits::value_type typedef.
        # Report the canonical display name so LLDB shows cuda::buffer instead.
        return (
            self.declared_type.GetCanonicalType()
            .GetUnqualifiedType()
            .GetDisplayTypeName()
            or ""
        )

    def get_child_index(self, name: str) -> int:
        if name.startswith("[") and name.endswith("]"):
            try:
                return int(name[1:-1])
            except ValueError:
                pass
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index < 0 or index >= self.size or not self.host_address:
            return None
        address = self.host_address + index * self.value_size
        return self.value.CreateValueFromAddress(f"[{index}]", address, self.value_type)


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::buffer formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function {module}.buffer_summary "
        f"--recognizer-function {module}.is_cuda_buffer"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.BufferSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_buffer"
    )
