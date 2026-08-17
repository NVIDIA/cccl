# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::optional."""

from __future__ import annotations

import re

import cccl_common

import lldb

_OPTIONAL_PATTERN = re.compile(r"^cuda::std::optional<.+>$")
InternalDict = dict[str, object]


def is_cuda_optional(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    return _OPTIONAL_PATTERN.fullmatch(type_name) is not None


def optional_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    non_syn = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    engaged_member = non_syn.GetChildMemberWithName("__engaged_")
    if engaged_member.IsValid() and engaged_member.GetValueAsUnsigned(0) == 0:
        return "cuda::std::nullopt"
    return ""


class OptionalSyntheticProvider:
    """Expose cuda::std::optional elements as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.update()

    def update(self) -> bool:
        self.type_name = (
            self.value.GetType()
            .GetCanonicalType()
            .GetUnqualifiedType()
            .GetDisplayTypeName()
            or ""
        )
        self.engaged_member = self.value.GetChildMemberWithName("__engaged_")
        self.size = 0
        if not self.engaged_member.IsValid():
            return False
        self.engaged = self.engaged_member.GetValueAsUnsigned(0) != 0
        if self.engaged:
            storage = self.value.GetChildMemberWithName("__storage_")
            if not storage.IsValid():
                return False
            self.val = storage.GetChildMemberWithName("__val_")
            if not self.val.IsValid():
                return False
            self.size = 1
        return True

    def num_children(self) -> int:
        return self.size

    def has_children(self) -> bool:
        return self.size != 0

    def get_child_index(self, name: str) -> int:
        if name == "value":
            return 0
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index == 0 and self.size == 1:
            return self.val.Clone("value")
        return None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::std::optional formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --python-function {module}.optional_summary "
        f"--recognizer-function {module}.is_cuda_optional"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.OptionalSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_optional"
    )
