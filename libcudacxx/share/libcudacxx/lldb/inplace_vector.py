# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::inplace_vector."""

from __future__ import annotations

import re

import cccl_common

import lldb

_INPLACE_VECTOR_PATTERN = re.compile(r"^cuda::std::inplace_vector<.+,\s*(\d+)>$")
InternalDict = dict[str, object]


def is_cuda_inplace_vector(
    value_type: lldb.SBType, _internal_dict: InternalDict
) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    return _INPLACE_VECTOR_PATTERN.fullmatch(type_name) is not None


def inplace_vector_summary(
    value: lldb.SBValue, _internal_dict: InternalDict
) -> str | None:
    """Summarize size and capacity like LLDB's std::vector formatter."""
    value = cccl_common.strip_reference_value(value)
    value = value.GetNonSyntheticValue()
    type_name = cccl_common.canonical_type_name(value.GetType())
    match = _INPLACE_VECTOR_PATTERN.fullmatch(type_name)
    if match is None:
        return None
    capacity = int(match.group(1))
    # The zero-capacity specialization has no members at all; do not read any.
    size = (
        0
        if capacity == 0
        else value.GetChildMemberWithName("__size_").GetValueAsUnsigned(0)
    )
    return f"size={size}, capacity={capacity}"


class InplaceVectorSyntheticProvider:
    """Expose constructed cuda::std::inplace_vector elements as LLDB children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.update()

    def update(self) -> bool:
        type_name = (
            self.value.GetType()
            .GetCanonicalType()
            .GetUnqualifiedType()
            .GetDisplayTypeName()
            or ""
        )
        self.type_name = type_name
        self.size = 0
        self.elems = None
        self.element_type = None
        match = _INPLACE_VECTOR_PATTERN.fullmatch(type_name)
        # The zero-capacity specialization has no members at all; do not read any.
        if not match or int(match.group(1)) == 0:
            return False
        size = self.value.GetChildMemberWithName("__size_")
        elems = self.value.GetChildMemberWithName("__elems_")
        element_type = (
            self.value.GetType().GetCanonicalType().GetTemplateArgumentType(0)
        )
        if not size.IsValid() or not elems.IsValid() or not element_type.IsValid():
            return False
        self.size = size.GetValueAsUnsigned(0)
        self.elems = elems
        self.element_type = element_type
        return True

    def num_children(self) -> int:
        return self.size

    def has_children(self) -> bool:
        return self.size != 0

    def get_type_name(self) -> str:
        return self.type_name

    def get_child_index(self, name: str) -> int:
        if name.startswith("[") and name.endswith("]"):
            try:
                return int(name[1:-1])
            except ValueError:
                pass
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index < 0 or index >= self.size or self.elems is None:
            return None
        # Storage is a typed array for trivial element types and a raw
        # unsigned-char array otherwise; creating children at element offsets
        # reads both layouts and never touches the slots past __size_.
        return self.elems.CreateChildAtOffset(
            f"[{index}]", index * self.element_type.GetByteSize(), self.element_type
        )


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::std::inplace_vector formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function {module}.inplace_vector_summary "
        f"--recognizer-function {module}.is_cuda_inplace_vector"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.InplaceVectorSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_inplace_vector"
    )
