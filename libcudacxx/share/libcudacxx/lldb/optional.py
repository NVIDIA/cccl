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


def _is_engaged(value: lldb.SBValue) -> bool:
    """Return whether an optional holds a value, without touching the payload."""
    # optional<T&> stores a pointer that is null when disengaged; every other
    # specialization keeps an engaged flag next to a union payload.
    pointer = value.GetChildMemberWithName("__value_")
    if pointer.IsValid():
        return pointer.GetValueAsUnsigned(0) != 0
    return value.GetChildMemberWithName("__engaged_").GetValueAsUnsigned(0) != 0


def optional_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    """Summarize the engaged state like LLDB's std::optional formatter."""
    value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    return f"Has Value={'true' if _is_engaged(value) else 'false'}"


class OptionalSyntheticProvider:
    """Expose the contained value of a cuda::std::optional as an LLDB child."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.update()

    def update(self) -> bool:
        self.engaged = _is_engaged(self.value)
        return True

    def num_children(self) -> int:
        return 1 if self.engaged else 0

    def has_children(self) -> bool:
        return self.engaged

    def get_child_index(self, name: str) -> int:
        return 0 if name == "Value" else -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        # The payload is only read once the engaged state says it holds a value;
        # a disengaged optional may still carry the bytes of a previous value.
        if index != 0 or not self.engaged:
            return None
        pointer = self.value.GetChildMemberWithName("__value_")
        if pointer.IsValid():
            return pointer.Dereference().Clone("Value")
        # The union member's declared type is the ``remove_cv_t<T>`` typedef, which
        # leaves the value formatted under an unresolved alias (and lets a nested
        # optional's leaf inherit a stray summary). Recover the real value type from
        # the class template argument and read the payload through that instead.
        value_type = (
            self.value.GetType()
            .GetCanonicalType()
            .GetUnqualifiedType()
            .GetTemplateArgumentType(0)
        )
        val = self.value.GetChildMemberWithName("__storage_").GetChildMemberWithName(
            "__val_"
        )
        if value_type.IsValid():
            val = val.Cast(value_type)
        return val.Clone("Value")


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
