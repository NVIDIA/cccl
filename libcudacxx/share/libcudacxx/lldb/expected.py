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
    type_name = cccl_common.canonical_type_name(value_type)
    return _EXPECTED_PATTERN.fullmatch(type_name) is not None


class ExpectedSyntheticProvider:
    """Expose the engaged value or the error of a cuda::std::expected as one child."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        # Captured before stripping the reference, so a caller's declared
        # const/reference qualifiers stay visible the way LLDB's own default
        # formatting would show them, instead of being silently dropped.
        self.type_name = value.GetType().GetDisplayTypeName() or ""
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.name = "value"
        self.has_val = False
        self.child = lldb.SBValue()
        self.update()

    def update(self) -> bool:
        self.has_val = bool(
            self.value.GetChildMemberWithName("__has_val_").GetValueAsUnsigned(0)
        )
        union = self.value.GetChildMemberWithName("__union_")
        if self.has_val:
            child = union.GetChildMemberWithName("__val_")
            self.name = "value"
            if not child.IsValid():
                # expected<void, E>, engaged: no __val_ member exists to show,
                # but a synthesized "void" child still gets LLDB's usual
                # single child "(name = ...)" summary, instead of nothing.
                child = self.value.CreateValueFromExpression(self.name, '"void"')
        else:
            child = union.GetChildMemberWithName("__unex_")
            self.name = "error"
        self.child = child.Clone(self.name) if child.IsValid() else lldb.SBValue()
        return self.child.IsValid()

    def num_children(self) -> int:
        return int(self.child.IsValid())

    def has_children(self) -> bool:
        return self.child.IsValid()

    def get_type_name(self) -> str:
        return self.type_name

    def get_child_index(self, name: str) -> int:
        return 0 if self.child.IsValid() and name == self.name else -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index == 0 and self.child.IsValid():
            return self.child
        return None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda expected formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.ExpectedSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_expected"
    )
