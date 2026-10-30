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
    type_name = cccl_common.public_type_name(value_type)
    return _OPTIONAL_PATTERN.fullmatch(type_name) is not None


def _optional_resolved_value(value: lldb.SBValue) -> lldb.SBValue | None:
    non_syn = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    val_member = non_syn.GetChildMemberWithName("__value_")
    if val_member.IsValid():
        # __value_ is a pointer (T*) for the optional<T&> specialization.
        # Its value is 0 (null) if disengaged, and non-zero if engaged,
        # regardless of the value of the referenced object itself.
        if val_member.GetValueAsUnsigned(0) != 0:
            return val_member.Dereference()
    else:
        engaged_member = non_syn.GetChildMemberWithName("__engaged_")
        if engaged_member.IsValid() and engaged_member.GetValueAsUnsigned(0) != 0:
            storage = non_syn.GetChildMemberWithName("__storage_")
            if storage.IsValid():
                val = storage.GetChildMemberWithName("__val_")
                if val.IsValid():
                    return val
    return None


def optional_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    resolved = _optional_resolved_value(value)
    if resolved is None:
        return "cuda::std::nullopt"
    return ""


class OptionalSyntheticProvider:
    """Expose cuda::std::optional elements as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.raw_value = value
        self.update()

    def get_type_name(self) -> str:
        # Use GetCanonicalType() to desugar any typedefs/aliases (e.g. optional_alias)
        # while preserving const/reference qualifiers.
        type_name = (
            self.raw_value.GetType().GetCanonicalType().GetDisplayTypeName() or ""
        )
        return cccl_common._ABI_NAMESPACE_PATTERN.sub("", type_name)

    def update(self) -> bool:
        self.resolved_value = _optional_resolved_value(self.raw_value)
        return True

    def num_children(self) -> int:
        return 1 if self.resolved_value is not None else 0

    def has_children(self) -> bool:
        return self.resolved_value is not None

    def get_child_index(self, name: str) -> int:
        if name == "value" and self.resolved_value is not None:
            return 0
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index == 0 and self.resolved_value is not None:
            return self.resolved_value.Clone("value")
        return None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::std::optional formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function {module}.optional_summary "
        f"--recognizer-function {module}.is_cuda_optional"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.OptionalSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_optional"
    )
