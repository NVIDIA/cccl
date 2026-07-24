# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::buffer."""

from __future__ import annotations

import re
from typing import NamedTuple

import memory_resource

import lldb

_BUFFER_PATTERN = re.compile(r"^cuda::buffer<.+>$")
# LLDB sees cudaMemcpyKind through cudaMemcpy's declaration, but its expression
# parser does not necessarily import the enum constants. This is the value of
# cudaMemcpyDefault from the CUDA Runtime API.
_CUDA_MEMCPY_DEFAULT = 4
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
    type_name = (
        value_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""
    )
    return _BUFFER_PATTERN.fullmatch(type_name) is not None


def _buffer_info(value: lldb.SBValue) -> BufferInfo | None:
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
        self.value = value.GetNonSyntheticValue()
        self.host_copy = lldb.SBValue()
        self.clear()
        self.update()

    def __del__(self) -> None:
        self.clear()

    def _evaluate(self, expression: str) -> lldb.SBValue:
        frame = self.value.GetFrame()
        if not frame.IsValid():
            return lldb.SBValue()
        options = lldb.SBExpressionOptions()
        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)
        return frame.EvaluateExpression(expression, options)

    def clear(self) -> None:
        """Release the staged copy and reset all cached buffer information."""
        if self.host_copy.IsValid():
            address = self.host_copy.GetValueAsUnsigned(0)
            if address:
                self._evaluate(f"(void)free((void*){address:#x})")
        self.host_copy = lldb.SBValue()
        self.size = 0
        self.data_address = 0
        self.value_type = lldb.SBType()
        self.value_size = 0

    def _copy_to_host(self) -> bool:
        if self.size == 0:
            return True

        byte_count = self.size * self.value_size
        self.host_copy = self._evaluate(f"(void*)malloc({byte_count})")
        if not self.host_copy.IsValid() or self.host_copy.GetError().Fail():
            return False

        host_address = self.host_copy.GetValueAsUnsigned(0)
        result = self._evaluate(
            "(int)cudaMemcpy((void*)"
            f"{host_address:#x}, (const void*){self.data_address:#x}, {byte_count}, "
            f"{_CUDA_MEMCPY_DEFAULT})"
        )
        if (
            not result.IsValid()
            or result.GetError().Fail()
            or result.GetValueAsSigned(-1) != 0
        ):
            self.clear()
            return False
        return True

    def update(self) -> bool:
        self.clear()

        info = _buffer_info(self.value)
        if info is None:
            return False

        self.size = info.size
        self.data_address = info.data_address
        self.value_type = info.value_type
        self.value_size = self.value_type.GetByteSize()
        self._copy_to_host()
        return True

    def num_children(self) -> int:
        return self.size

    def has_children(self) -> bool:
        return self.size != 0

    def get_type_name(self) -> str:
        # STL element access can preserve an alloc_traits::value_type typedef.
        # Report the canonical display name so LLDB shows cuda::buffer instead.
        return (
            self.value.GetType()
            .GetCanonicalType()
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
        if index < 0:
            return None
        if index >= self.size:
            return None
        offset = index * self.value_size
        return self.host_copy.CreateChildAtOffset(f"[{index}]", offset, self.value_type)


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
