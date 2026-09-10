# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::mr::shared_resource."""

from __future__ import annotations

import re

import cccl_common

import lldb

_SHARED_RESOURCE_PATTERN = re.compile(r"^cuda::mr::shared_resource<.+>$")
InternalDict = dict[str, object]


def is_shared_resource(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    return _SHARED_RESOURCE_PATTERN.fullmatch(type_name) is not None


def _display_type_name(value: lldb.SBValue) -> str:
    type_name = cccl_common.canonical_type_name(value.GetType())
    return type_name or "cuda::mr::shared_resource"


def _block_pointer(value: lldb.SBValue) -> lldb.SBValue:
    """Return the control-block pointer, or an invalid value if unavailable."""
    # shared_resource holds a __shared_block_ptr, and both the wrapper and the
    # pointer spell their member __block_.
    return (
        cccl_common.strip_reference_value(value)
        .GetNonSyntheticValue()
        .GetChildMemberWithName("__block_")
        .GetChildMemberWithName("__block_")
    )


def _payload(value: lldb.SBValue) -> lldb.SBValue:
    """Return the owned resource, or an invalid value if this handle is empty."""
    block_pointer = _block_pointer(value)
    if not block_pointer.IsValid() or block_pointer.GetValueAsUnsigned(0) == 0:
        return lldb.SBValue()
    return block_pointer.Dereference().GetChildMemberWithName("__payload")


def _resource_pointer(payload: lldb.SBValue) -> lldb.SBValue:
    """Return the pointer both formatters report as the owned resource.

    The summary reports the address only when this is invalid and there is no
    child to carry it, so both have to read the same signal.
    """
    return payload.AddressOf() if payload.IsValid() else payload


def shared_resource_summary(
    value: lldb.SBValue, _internal_dict: InternalDict
) -> str | None:
    """Describe the ownership state of a CUDA shared resource."""
    type_name = _display_type_name(value)
    block_pointer = _block_pointer(value)
    if not block_pointer.IsValid():
        return None
    if block_pointer.GetValueAsUnsigned(0) == 0:
        return f"{type_name} use_count=0, resource=nullptr"

    block = block_pointer.Dereference()
    payload = _payload(value)
    # cuda::std::atomic<int> keeps its value in __a.__a_value. Read the raw
    # members: the atomic formatter replaces that layout with a synthetic
    # "value" child, and one formatter should not depend on another's output.
    reference_count = (
        block.GetChildMemberWithName("__ref_count")
        .GetNonSyntheticValue()
        .GetChildMemberWithName("__a")
        .GetChildMemberWithName("__a_value")
    )
    if not payload.IsValid() or not reference_count.IsValid():
        return None

    use_count = reference_count.GetValueAsSigned(0)
    # The address of a readable resource belongs to the resource child, the way
    # std::shared_ptr keeps strong=/weak= in its summary and the pointer in its
    # child. Only report it here when there is no child to carry it: a control
    # block reached through a live pointer always has a readable resource, so
    # the branch below guards against an unreadable frame rather than against
    # any state the scenario can reach.
    if not _resource_pointer(payload).IsValid():
        return f"{type_name} use_count={use_count}, resource=<invalid address>"
    return f"{type_name} use_count={use_count}"


class SharedResourceSyntheticProvider:
    """Expose the owned resource as an LLDB synthetic child."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
        self.resource = lldb.SBValue()
        self.update()

    def update(self) -> bool:
        # Present the owned resource the way std::shared_ptr presents its
        # pointer: one step away, so that expanding a handle does not print the
        # implementation details of the resource itself.
        pointer = _resource_pointer(_payload(self.value))
        self.resource = pointer.Clone("resource") if pointer.IsValid() else pointer
        # Report no caching: a copy or a move changes which resource, if any,
        # this handle owns, and LLDB must ask again after every stop.
        return False

    def num_children(self) -> int:
        return 1 if self.has_children() else 0

    def has_children(self) -> bool:
        return self.resource.IsValid()

    def get_child_index(self, name: str) -> int:
        return 0 if name == "resource" else -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index != 0 or not self.resource.IsValid():
            return None
        return self.resource


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::mr::shared_resource formatters in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function "
        f"{module}.shared_resource_summary --recognizer-function "
        f"{module}.is_shared_resource"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class "
        f"{module}.SharedResourceSyntheticProvider --recognizer-function "
        f"{module}.is_shared_resource"
    )
