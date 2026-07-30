# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::array."""

from __future__ import annotations

import re

import lldb

_ARRAY_PATTERN = re.compile(r"^cuda::std::array<.+,\s*(\d+)>$")
InternalDict = dict[str, object]


def is_cuda_array(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = (
        value_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""
    )
    return _ARRAY_PATTERN.fullmatch(type_name) is not None


class ArraySyntheticProvider:
    """Expose cuda::std::array elements as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
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
        match = _ARRAY_PATTERN.fullmatch(type_name)
        self.elems = self.value.GetChildMemberWithName("__elems_")
        self.size = 0
        if not self.elems.IsValid() or not match:
            return False
        self.size = int(match.group(1))
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
        if index < 0:
            return None
        if index >= self.size:
            return None
        return self.elems.GetChildAtIndex(index)


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::std::array formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.ArraySyntheticProvider "
        f"--recognizer-function {module}.is_cuda_array"
    )
