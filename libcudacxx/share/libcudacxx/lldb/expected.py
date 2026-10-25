# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::expected."""

from __future__ import annotations

import re

import cccl_common

import lldb

_EXPECTED_PATTERN = re.compile(r"^cuda::std::expected<.+>$")
InternalDict = dict[str, object]


def is_cuda_expected(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.public_type_name(value_type)
    return _EXPECTED_PATTERN.fullmatch(type_name) is not None


def _expected_child(value: lldb.SBValue) -> tuple[str, lldb.SBValue] | None:
    """Return the name and value of the held value or error, if there is one."""
    non_syn = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    # __has_val_ and __union_ are members of the __expected_destruct base class.
    has_value = non_syn.GetChildMemberWithName("__has_val_")
    storage = non_syn.GetChildMemberWithName("__union_")
    if not has_value.IsValid() or not storage.IsValid():
        return None
    if has_value.GetValueAsUnsigned(0) == 0:
        error = storage.GetChildMemberWithName("__unex_")
        return ("error", error) if error.IsValid() else None
    # expected<void, E> holds no value member.
    val = storage.GetChildMemberWithName("__val_")
    return ("value", val) if val.IsValid() else None


def expected_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    if _expected_child(value) is None:
        # expected<void, E> holding a value has nothing else to show.
        non_syn = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
        if non_syn.GetChildMemberWithName("__has_val_").GetValueAsUnsigned(0) != 0:
            return "has value"
    return ""


class ExpectedSyntheticProvider:
    """Expose the value or the error of a cuda::std::expected as a synthetic child."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.raw_value = value
        self.update()

    def get_type_name(self) -> str:
        # Use GetCanonicalType() to desugar any typedefs/aliases (e.g. expected_alias)
        # while preserving const/reference qualifiers.
        type_name = (
            self.raw_value.GetType().GetCanonicalType().GetDisplayTypeName() or ""
        )
        return cccl_common._ABI_NAMESPACE_PATTERN.sub("", type_name)

    def update(self) -> bool:
        self.child = _expected_child(self.raw_value)
        return True

    def num_children(self) -> int:
        return 1 if self.child is not None else 0

    def has_children(self) -> bool:
        return self.child is not None

    def get_child_index(self, name: str) -> int:
        if self.child is not None and name == self.child[0]:
            return 0
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index == 0 and self.child is not None:
            name, child = self.child
            return child.Clone(name)
        return None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::std::expected formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function {module}.expected_summary "
        f"--recognizer-function {module}.is_cuda_expected"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.ExpectedSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_expected"
    )
