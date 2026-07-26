# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::annotated_ptr."""

from __future__ import annotations

import re

import lldb

_ANNOTATED_PTR_PATTERN = re.compile(r"^cuda::annotated_ptr<.+,.+>$")
InternalDict = dict[str, object]


def is_annotated_ptr(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = (
        value_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""
    )
    return _ANNOTATED_PTR_PATTERN.fullmatch(type_name) is not None


def annotated_ptr_description(value: lldb.SBValue) -> str:
    """Describe an annotated_ptr using type information and pointer value."""
    type_name = (
        value.GetType().GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName()
    )
    if not type_name:
        type_name = "annotated_ptr"

    # Try to get the pointer value
    ptr_display = None
    try:
        value_nonsynth = value.GetNonSyntheticValue()
        repr_ptr = value_nonsynth.GetChildMemberWithName("__repr")
        if repr_ptr.IsValid():
            error = lldb.SBError()
            ptr_value = repr_ptr.GetValueAsUnsigned(error)
            if error.Success():
                if ptr_value == 0:
                    ptr_display = "nullptr"
                else:
                    ptr_display = f"{ptr_value:#x}"
    except Exception:
        pass

    # If we couldn't read the pointer value, just show type-only description
    if ptr_display is None:
        return type_name
    return f"{type_name} -> {ptr_display}"


def annotated_ptr_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    return annotated_ptr_description(value)


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register CUDA annotated_ptr formatters in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --python-function "
        f"{module}.annotated_ptr_summary --recognizer-function "
        f"{module}.is_annotated_ptr"
    )
