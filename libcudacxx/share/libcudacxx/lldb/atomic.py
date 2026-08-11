# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printers for CUDA atomic and atomic_ref types."""

from __future__ import annotations

import re

import cccl_common

import lldb

_ATOMIC_PATTERN = re.compile(r"^cuda::(?:std::)?atomic(?:_ref)?<.+>$")
_ATOMIC_REF_PATTERN = re.compile(r"^cuda::(?:std::)?atomic_ref<.+>$")
_THREAD_SCOPE_VALUE_PATTERN = re.compile(
    r"\((?:enum )?cuda(?:::std)?::thread_scope\)\s*(10|[012])(?=\s*>)"
)
_THREAD_SCOPE_NAMES = {
    0: "system",
    1: "device",
    2: "block",
    10: "thread",
}
InternalDict = dict[str, object]


def _complete_type_name(value_type: lldb.SBType) -> str:
    """Return the complete public type name, with thread scopes shown by name."""
    type_name = cccl_common.public_type_name(value_type)
    type_name = type_name.replace("cuda::std::thread_scope_", "cuda::thread_scope_")
    return _THREAD_SCOPE_VALUE_PATTERN.sub(
        lambda match: f"cuda::thread_scope_{_THREAD_SCOPE_NAMES[int(match.group(1))]}",
        type_name,
    )


def is_cuda_atomic(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    return (
        _ATOMIC_PATTERN.fullmatch(cccl_common.canonical_type_name(value_type))
        is not None
    )


def is_cuda_atomic_ref(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    return (
        _ATOMIC_REF_PATTERN.fullmatch(cccl_common.canonical_type_name(value_type))
        is not None
    )


def _reference_pointer(value: lldb.SBValue) -> lldb.SBValue:
    storage = value.GetChildMemberWithName("__a")
    if not storage.IsValid():
        return lldb.SBValue()
    return storage.GetChildMemberWithName("__a_value")


def atomic_ref_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str | None:
    pointer = _reference_pointer(
        cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    )
    if not pointer.IsValid():
        return None
    return f"ptr={pointer.GetValueAsUnsigned(0):#x}"


def _stored_value(value: lldb.SBValue, type_name: str) -> lldb.SBValue:
    value_type = cccl_common.strip_reference(value.GetType())
    stored = _reference_pointer(value)
    if not stored.IsValid():
        return lldb.SBValue()

    if _ATOMIC_REF_PATTERN.fullmatch(type_name) is not None:
        return stored.Dereference().Clone("value")

    storage = value.GetChildMemberWithName("__a")
    storage_type = cccl_common.canonical_type_name(storage.GetType())
    if "__atomic_small_storage<" in storage_type:
        stored = stored.GetChildMemberWithName("__a_value")
        if not stored.IsValid():
            return lldb.SBValue()
        address = stored.GetLoadAddress()
        if address == lldb.LLDB_INVALID_ADDRESS:
            return lldb.SBValue()
        stored = stored.CreateValueFromAddress(
            "value", address, value_type.GetTemplateArgumentType(0)
        )
    return stored.Clone("value")


class AtomicSyntheticProvider:
    """Expose the value represented by a CUDA atomic as one synthetic child."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.child = lldb.SBValue()
        self.update()

    def update(self) -> bool:
        self.type_name = _complete_type_name(self.value.GetType())
        self.child = _stored_value(self.value, self.type_name)
        return self.child.IsValid()

    def num_children(self) -> int:
        return int(self.child.IsValid())

    def has_children(self) -> bool:
        return self.child.IsValid()

    def get_type_name(self) -> str:
        return self.type_name

    def get_child_index(self, name: str) -> int:
        return 0 if name == "value" else -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index == 0 and self.child.IsValid():
            return self.child
        return None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register CUDA atomic formatters in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function "
        f"{module}.atomic_ref_summary --recognizer-function {module}.is_cuda_atomic_ref"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.AtomicSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_atomic"
    )
