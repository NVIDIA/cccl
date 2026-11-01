# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::annotated_ptr."""

from __future__ import annotations

import re

import cccl_common

import lldb

_ANNOTATED_PTR_PATTERN = re.compile(r"^cuda::annotated_ptr<.+,.+>$")
InternalDict = dict[str, object]


def is_annotated_ptr(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    return _ANNOTATED_PTR_PATTERN.fullmatch(type_name) is not None


def annotated_ptr_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str:
    # LLDB already renders the declared type, so the summary only adds the pointer.
    repr_ptr = cccl_common.strip_reference_value(value).GetChildMemberWithName("__repr")
    if not repr_ptr.IsValid():
        return "-> unavailable"

    error = lldb.SBError()
    ptr_addr = repr_ptr.GetValueAsUnsigned(error)
    if error.Fail():
        return "-> unavailable"
    return "-> nullptr" if ptr_addr == 0 else f"-> {ptr_addr:#x}"


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register CUDA annotated_ptr formatters with LLDB."""
    debugger.HandleCommand(
        f"type summary add --category {category} --python-function "
        f"{module}.annotated_ptr_summary --recognizer-function "
        f"{module}.is_annotated_ptr"
    )
