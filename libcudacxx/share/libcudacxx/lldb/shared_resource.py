# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::mr::shared_resource."""

from __future__ import annotations

import re

import lldb

_SHARED_RESOURCE_PATTERN = re.compile(r"^cuda::mr::shared_resource<.+>$")
InternalDict = dict[str, object]


def is_shared_resource(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = (
        value_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""
    )
    return _SHARED_RESOURCE_PATTERN.fullmatch(type_name) is not None


def shared_resource_summary(
    value: lldb.SBValue, _internal_dict: InternalDict
) -> str | None:
    """Describe the ownership state of a CUDA shared resource."""
    type_name = (
        value.GetType().GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName()
    )
    if not type_name:
        type_name = "cuda::mr::shared_resource"

    # shared_resource holds a __shared_block_ptr, and both the wrapper and the
    # pointer spell their member __block_.
    control_block = value.GetChildMemberWithName("__block_").GetChildMemberWithName(
        "__block_"
    )
    if not control_block.IsValid():
        return None
    if control_block.GetValueAsUnsigned(0) == 0:
        return f"{type_name} empty"

    block = control_block.Dereference()
    payload = block.GetChildMemberWithName("__payload")
    # cuda::std::atomic<int> keeps its value in __a.__a_value.
    reference_count = (
        block.GetChildMemberWithName("__ref_count")
        .GetChildMemberWithName("__a")
        .GetChildMemberWithName("__a_value")
    )
    if not payload.IsValid() or not reference_count.IsValid():
        return None

    use_count = reference_count.GetValueAsSigned(0)
    address = payload.GetLoadAddress()
    if address == lldb.LLDB_INVALID_ADDRESS:
        return f"{type_name} use_count={use_count}"
    return f"{type_name} use_count={use_count}, resource={address:#x}"


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::mr::shared_resource formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --python-function "
        f"{module}.shared_resource_summary --recognizer-function "
        f"{module}.is_shared_resource"
    )
