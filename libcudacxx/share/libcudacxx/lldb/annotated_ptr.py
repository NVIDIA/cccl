# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::annotated_ptr."""

from __future__ import annotations

import re

import lldb

from . import cccl_common

_ANNOTATED_PTR_PATTERN = re.compile(r"^cuda::annotated_ptr<.+,.+>$")
InternalDict = dict[str, object]


def _annotated_ptr_type_name(value: lldb.SBValue) -> str:
    """Get type name with fallback to generic 'annotated_ptr'."""
    try:
        type_name = cccl_common.public_type_name(value.GetType())
        return type_name or "annotated_ptr"
    except Exception:
        return "annotated_ptr"


def _annotated_ptr_repr_value(value: lldb.SBValue) -> lldb.SBValue:
    """Get the __repr member containing the pointer value."""
    try:
        value_nonsynth = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
        return value_nonsynth.GetChildMemberWithName("__repr")
    except Exception:
        return lldb.SBValue()


def is_annotated_ptr(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    """Check if an LLDB type represents cuda::annotated_ptr."""
    try:
        type_name = cccl_common.canonical_type_name(value_type)
        return bool(_ANNOTATED_PTR_PATTERN.fullmatch(type_name))
    except Exception:
        return False


def annotated_ptr_description(value: lldb.SBValue) -> str:
    """Describe annotated_ptr using type and pointer value information."""
    type_name = _annotated_ptr_type_name(value)
    repr_ptr = _annotated_ptr_repr_value(value)

    if not repr_ptr.IsValid():
        return type_name

    error = lldb.SBError()
    ptr_addr = repr_ptr.GetValueAsUnsigned(error)
    if not error.Success():
        return type_name

    ptr_display = "nullptr" if ptr_addr == 0 else f"{ptr_addr:#x}"
    return f"{type_name} -> {ptr_display}"


def annotated_ptr_summary(
    value: lldb.SBValue, _internal_dict: InternalDict
) -> str | None:
    """Summarize annotated_ptr, returning None if unavailable."""
    try:
        return annotated_ptr_description(value)
    except Exception:
        return None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register CUDA annotated_ptr formatters with LLDB."""
    debugger.HandleCommand(
        f"type summary add --category {category} --python-function "
        f"{module}.annotated_ptr_summary --recognizer-function "
        f"{module}.is_annotated_ptr"
    )
