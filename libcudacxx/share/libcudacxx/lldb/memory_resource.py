# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for CUDA type-erased memory resources."""

from __future__ import annotations

import re

import cccl_common

import lldb

_RESOURCE_PATTERN = re.compile(
    r"^cuda::mr::(?:basic_any_resource|any_resource|any_synchronous_resource)<.*>$"
)
InternalDict = dict[str, object]


def is_memory_resource(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.public_type_name(value_type)
    return _RESOURCE_PATTERN.fullmatch(type_name) is not None


def _resource_info(value: lldb.SBValue) -> tuple[str, lldb.SBValue | None]:
    value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    tagged_vptr_value = value.GetChildMemberWithName("__vptr_").GetChildMemberWithName(
        "__ptr_"
    )
    if not tagged_vptr_value.IsValid():
        return "unavailable", None
    tagged_vptr = tagged_vptr_value.GetValueAsUnsigned(0)
    if tagged_vptr == 0:
        return "empty", None

    buffer = value.GetChildMemberWithName("__buffer_")
    void_pointer = value.GetTarget().GetBasicType(lldb.eBasicTypeVoid).GetPointerType()
    if not buffer.IsValid() or not void_pointer.IsValid():
        return "unavailable", None
    if tagged_vptr & 1:
        resource = buffer.AddressOf().Cast(void_pointer).Clone("resource")
        return ("in-situ", resource) if resource.IsValid() else ("unavailable", None)

    resource = buffer.CreateChildAtOffset("resource", 0, void_pointer)
    return ("heap", resource) if resource.IsValid() else ("unavailable", None)


def memory_resource_description(value: lldb.SBValue) -> str:
    """Describe a type-erased resource using only public type information."""
    type_name = cccl_common.canonical_type_name(value.GetType())
    if not type_name:
        type_name = "type-erased resource"

    address = value.GetLoadAddress()
    if address == lldb.LLDB_INVALID_ADDRESS:
        return type_name
    return f"{type_name} @ {address:#x}"


def memory_resource_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    value_type = value.GetType()
    type_name = cccl_common.public_type_name(value_type)
    declared_type_name = value_type.GetDisplayTypeName() or value_type.GetName() or ""
    # LLDB already renders the declared type. Keep the canonical name only when
    # it adds information, such as when the declared type is an alias.
    type_prefix = "" if type_name in declared_type_name else f"{type_name} "
    storage, resource = _resource_info(value)
    if storage == "unavailable":
        return f"{type_prefix}storage=unavailable"
    if resource is None:
        return f"{type_prefix}storage=0x0"
    return f"{type_prefix}storage={resource.GetValueAsUnsigned(0):#x} ({storage})"


class MemoryResourceSyntheticProvider:
    """Expose the erased memory-resource object pointer as a synthetic child."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
        self.update()

    def update(self) -> bool:
        _, self.resource = _resource_info(self.value)
        return False

    def num_children(self) -> int:
        return int(self.resource is not None)

    def has_children(self) -> bool:
        return self.resource is not None

    def get_child_index(self, name: str) -> int:
        return 0 if name == "resource" and self.resource is not None else -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        return self.resource if index == 0 else None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register CUDA memory-resource formatters in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function "
        f"{module}.memory_resource_summary --recognizer-function "
        f"{module}.is_memory_resource"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class "
        f"{module}.MemoryResourceSyntheticProvider --recognizer-function "
        f"{module}.is_memory_resource"
    )
