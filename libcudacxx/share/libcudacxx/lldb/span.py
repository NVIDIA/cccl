# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::span."""

from __future__ import annotations

import re

import lldb

_SPAN_PATTERN = re.compile(r"^cuda::std::span<.+>$")
_ABI_NAMESPACE_PATTERN = re.compile(r"::__(?:\d+|version_bump_ver\d+_)(?=::)")
InternalDict = dict[str, object]


def _canonical_type_name(value_type: lldb.SBType) -> str:
    # GetName, not GetDisplayTypeName: the display name elides defaulted template
    # arguments, so a dynamic-extent span would lose its extent entirely. The full
    # name always keeps it, at the cost of spelling out the ABI inline namespace.
    name = value_type.GetCanonicalType().GetUnqualifiedType().GetName() or ""
    return _ABI_NAMESPACE_PATTERN.sub("", name)


def is_cuda_span(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    return _SPAN_PATTERN.fullmatch(_canonical_type_name(value_type)) is not None


def _dynamic_extent(span_type: lldb.SBType, target: lldb.SBTarget) -> int:
    # size_t(-1), with size_t's width read from the extent argument's type,
    # falling back to the pointer width.
    extent_type = span_type.GetTemplateArgumentType(1)
    byte_size = extent_type.GetByteSize() if extent_type.IsValid() else 0
    return (1 << (8 * (byte_size or target.GetAddressByteSize()))) - 1


class SpanSyntheticProvider:
    """Expose cuda::std::span elements as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.value = value.GetNonSyntheticValue()
        self.update()

    def update(self) -> bool:
        span_type = self.value.GetType().GetCanonicalType().GetUnqualifiedType()
        canonical_name = _canonical_type_name(self.value.GetType())
        self.type_name = canonical_name.replace(
            f", {_dynamic_extent(span_type, self.value.GetTarget())}>",
            ", dynamic_extent>",
        )
        self.size = 0
        self.value_size = 0

        self.data = self.value.GetChildMemberWithName("__data_")
        self.value_type = span_type.GetTemplateArgumentType(0)
        if not self.data.IsValid() or not self.value_type.IsValid():
            return False

        # The dynamic-extent specialization stores its size; the static-extent
        # one carries it in the type name, which is the only place LLDB exposes
        # a non-type template argument.
        size = self.value.GetChildMemberWithName("__size_")
        self.size = (
            size.GetValueAsUnsigned(0)
            if size.IsValid()
            else _static_extent(canonical_name)
        )

        self.value_size = self.value_type.GetByteSize()
        if self.data.GetValueAsUnsigned(0) == 0 or self.value_size == 0:
            self.size = 0
        return True

    def num_children(self) -> int:
        return self.size

    def has_children(self) -> bool:
        return self.size != 0

    def get_type_name(self) -> str:
        return self.type_name

    def get_child_index(self, name: str) -> int:
        if name.startswith("[") and name.endswith("]"):
            try:
                return int(name[1:-1])
            except ValueError:
                pass
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index < 0:
            return None
        if index >= self.size:
            return None
        offset = index * self.value_size
        return self.data.CreateChildAtOffset(f"[{index}]", offset, self.value_type)


def _static_extent(type_name: str) -> int:
    """Read the extent out of a static-extent span's type name."""
    match = re.search(r",\s*(\d+)>$", type_name)
    return int(match.group(1)) if match else 0


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::std::span formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.SpanSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_span"
    )
