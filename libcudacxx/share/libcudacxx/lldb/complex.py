# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::complex and cuda::complex."""

from __future__ import annotations

import re

import cccl_common

import lldb

_COMPLEX_PATTERN = re.compile(r"^cuda::(?:std::)?complex<.+>$")
_CHILD_NAMES = ("real", "imag")
InternalDict = dict[str, object]


def is_cuda_complex(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    return _COMPLEX_PATTERN.fullmatch(type_name) is not None


class ComplexSyntheticProvider:
    """Expose complex real and imaginary parts as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.update()

    def update(self) -> bool:
        self.type_name = (
            self.value.GetType()
            .GetCanonicalType()
            .GetUnqualifiedType()
            .GetDisplayTypeName()
            or ""
        )
        self.parts: list[lldb.SBValue] = []
        real = self.value.GetChildMemberWithName("__re_")
        imag = self.value.GetChildMemberWithName("__im_")
        if not real.IsValid() or not imag.IsValid():
            return False
        scalar_type = real.GetType()
        self.parts = [
            self.value.CreateChildAtOffset("real", 0, scalar_type),
            self.value.CreateChildAtOffset(
                "imag", scalar_type.GetByteSize(), scalar_type
            ),
        ]
        return True

    def num_children(self) -> int:
        return len(self.parts)

    def has_children(self) -> bool:
        return bool(self.parts)

    def get_type_name(self) -> str:
        return self.type_name

    def get_child_index(self, name: str) -> int:
        if name in _CHILD_NAMES:
            return _CHILD_NAMES.index(name)
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if 0 <= index < len(self.parts):
            return self.parts[index]
        return None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda complex formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.ComplexSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_complex"
    )
