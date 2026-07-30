# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for CUDA type-erased memory resources."""

from __future__ import annotations

import re

import lldb

_RESOURCE_PATTERN = re.compile(
    r"^cuda::mr::(?:basic_any_resource|any_resource|any_synchronous_resource)<.+>$"
)
InternalDict = dict[str, object]


def is_memory_resource(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = (
        value_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""
    )
    return _RESOURCE_PATTERN.fullmatch(type_name) is not None


def memory_resource_description(value: lldb.SBValue) -> str:
    """Describe a type-erased resource using only public type information."""
    type_name = (
        value.GetType().GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName()
    )
    if not type_name:
        type_name = "type-erased resource"

    address = value.GetLoadAddress()
    if address == lldb.LLDB_INVALID_ADDRESS:
        return type_name
    return f"{type_name} @ {address:#x}"


def memory_resource_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    return memory_resource_description(value)


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register CUDA memory-resource formatters in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --python-function "
        f"{module}.memory_resource_summary --recognizer-function "
        f"{module}.is_memory_resource"
    )
