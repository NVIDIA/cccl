# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::atomic and cuda::std::atomic_ref."""

from __future__ import annotations

import re

import lldb

_ATOMIC_PATTERN = re.compile(r"^cuda::std::atomic(?:_ref)?<.+>$")
InternalDict = dict[str, object]


def _type_name(value_type: lldb.SBType) -> str:
    return value_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""


def is_cuda_atomic(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    return _ATOMIC_PATTERN.fullmatch(_type_name(value_type)) is not None


def _stored_value(value: lldb.SBValue) -> lldb.SBValue:
    value_type = value.GetType().GetCanonicalType().GetUnqualifiedType()
    type_name = _type_name(value_type)
    storage = value.GetChildMemberWithName("__a")
    if not storage.IsValid():
        return lldb.SBValue()

    stored = storage.GetChildMemberWithName("__a_value")
    if not stored.IsValid():
        return lldb.SBValue()

    if type_name.startswith("cuda::std::atomic_ref<"):
        return stored.Dereference().Clone("value")

    storage_type = _type_name(storage.GetType())
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
        self.value = value.GetNonSyntheticValue()
        self.child = lldb.SBValue()
        self.update()

    def update(self) -> bool:
        self.type_name = _type_name(self.value.GetType())
        self.child = _stored_value(self.value)
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
        f"type synthetic add --category {category} --python-class {module}.AtomicSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_atomic"
    )
