# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::hierarchy."""

from __future__ import annotations

import re

import cccl_common
import tuple as tuple_printer

import lldb

_HIERARCHY_PATTERN = re.compile(r"^cuda::hierarchy<.*>$")
_LEVEL_DESC_PATTERN = re.compile(r"^cuda::hierarchy_level_desc<.*>$")
InternalDict = dict[str, object]


def _public_type_name(value_type: lldb.SBType) -> str:
    return cccl_common.public_type_name(value_type)


def is_cuda_hierarchy(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    return _HIERARCHY_PATTERN.fullmatch(_public_type_name(value_type)) is not None


def is_hierarchy_level_desc(
    value_type: lldb.SBType, _internal_dict: InternalDict
) -> bool:
    return _LEVEL_DESC_PATTERN.fullmatch(_public_type_name(value_type)) is not None


def _level_name(level_type: lldb.SBType) -> str:
    if not level_type.IsValid():
        raise ValueError("missing hierarchy level type")
    name = _public_type_name(level_type).rsplit("::", 1)[-1]
    return name.removesuffix("_level")


def _unsigned(value: lldb.SBValue) -> int:
    if not value.IsValid() or value.GetError().Fail() or value.GetValue() is None:
        raise ValueError("unreadable hierarchy extent")
    return value.GetValueAsUnsigned()


def _base_at_index(value: lldb.SBValue, index: int) -> lldb.SBValue:
    value_type = cccl_common.strip_reference(value.GetType())
    if index >= value_type.GetNumberOfDirectBaseClasses():
        raise ValueError(f"{_public_type_name(value_type)} has no such base class")
    member = value_type.GetDirectBaseClassAtIndex(index)
    base = value.CreateChildAtOffset(
        "__base", member.GetOffsetInBytes(), member.GetType()
    )
    if not base.IsValid() or base.GetError().Fail():
        raise ValueError("unreadable hierarchy extent storage")
    return base


def _first_base(value: lldb.SBValue) -> lldb.SBValue:
    return _base_at_index(value, 0)


def _hierarchy_level_desc(value: lldb.SBValue) -> lldb.SBValue:
    value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    value_type = cccl_common.strip_reference(value.GetType())
    # Inspect base types before materializing a value. Empty sibling bases can
    # share an offset, where CreateChildAtOffset returns the first requested type.
    pending = [(value_type, 0)]
    while pending:
        candidate_type, offset = pending.pop()
        if _LEVEL_DESC_PATTERN.fullmatch(_public_type_name(candidate_type)) is not None:
            if candidate_type == value_type:
                return value
            candidate = value.CreateChildAtOffset(
                "__hierarchy_level_desc", offset, candidate_type
            )
            if not candidate.IsValid() or candidate.GetError().Fail():
                raise ValueError("unreadable hierarchy level descriptor")
            return candidate
        for index in range(candidate_type.GetNumberOfDirectBaseClasses()):
            member = candidate_type.GetDirectBaseClassAtIndex(index)
            pending.append((member.GetType(), offset + member.GetOffsetInBytes()))
    raise ValueError(
        f"{_public_type_name(value.GetType())} is not a hierarchy level descriptor"
    )


def _dimensions(value: lldb.SBValue) -> tuple[int, int, int]:
    value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    extents = value.GetChildMemberWithName("__exts_").GetNonSyntheticValue()
    if not extents.IsValid() or extents.GetError().Fail():
        raise ValueError("missing hierarchy extents")

    extents_type = cccl_common.strip_reference(extents.GetType())
    # LLDB 20 does not expose non-type template arguments through SBType, even
    # though it retains the extents pack in the type name.
    arguments = _public_type_name(extents_type).removesuffix(">")
    static_arguments = arguments.rsplit(",", 3)[-3:]
    if len(static_arguments) != 3:
        raise ValueError(f"unexpected hierarchy extents type: {arguments}")
    static_extents = [int(argument.strip()) for argument in static_arguments]

    target = value.GetTarget()
    extent_size = target.FindFirstType("size_t").GetByteSize()
    if extent_size == 0:
        raise ValueError("unknown hierarchy extent width")
    dynamic_extent = (1 << (extent_size * 8)) - 1
    dynamic_count = static_extents.count(dynamic_extent)

    dynamic_values: lldb.SBValue | None = None
    if dynamic_count:
        storage = _first_base(_first_base(extents))
        dynamic_values = storage.GetChildMemberWithName("__vals_")
        if not dynamic_values.IsValid() or dynamic_values.GetError().Fail():
            raise ValueError("missing dynamic hierarchy extents")

    result = []
    dynamic_index = 0
    for static_extent in static_extents:
        if static_extent == dynamic_extent:
            if (
                dynamic_values is None
                or dynamic_index >= dynamic_values.GetNumChildren()
            ):
                raise ValueError("missing dynamic hierarchy extent")
            result.append(_unsigned(dynamic_values.GetChildAtIndex(dynamic_index)))
            dynamic_index += 1
        else:
            result.append(static_extent)
    return result[0], result[1], result[2]


def hierarchy_level_desc_summary(
    value: lldb.SBValue, _internal_dict: InternalDict
) -> str | None:
    try:
        x, y, z = _dimensions(value)
    except (RuntimeError, ValueError):
        return None
    return f"dims=(x={x}, y={y}, z={z})"


def hierarchy_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str | None:
    value_type = cccl_common.strip_reference(value.GetType())
    try:
        bottom_unit = _level_name(value_type.GetTemplateArgumentType(0))
    except (RuntimeError, ValueError):
        return None
    return f"bottom_unit={bottom_unit}"


class HierarchySyntheticProvider:
    """Expose cuda::hierarchy levels as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.levels: list[tuple[str, lldb.SBValue]] = []
        self.update()

    def update(self) -> bool:
        self.levels = []
        descs = self.value.GetChildMemberWithName("__descs_")
        if not descs.IsValid() or descs.GetError().Fail():
            return False
        provider = tuple_printer.TupleSyntheticProvider(descs, {})
        child_count = provider.num_children()
        if child_count == 0:
            return False
        for index in range(child_count):
            desc = provider.get_child_at_index(index)
            if desc is None or not desc.IsValid() or desc.GetError().Fail():
                self.levels = []
                return False
            try:
                desc = _hierarchy_level_desc(desc)
                desc_type = cccl_common.strip_reference(desc.GetType())
                name = _level_name(desc_type.GetTemplateArgumentType(0))
            except (RuntimeError, ValueError):
                self.levels = []
                return False
            self.levels.append((name, desc))
        return True

    def num_children(self) -> int:
        return len(self.levels)

    def has_children(self) -> bool:
        return len(self.levels) != 0

    def get_type_name(self) -> str:
        return "cuda::hierarchy"

    def get_child_index(self, name: str) -> int:
        for index, (level_name, _) in enumerate(self.levels):
            if name == level_name:
                return index
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index < 0 or index >= len(self.levels):
            return None
        name, value = self.levels[index]
        return value.Clone(name)


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::hierarchy formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --python-function "
        f"{module}.hierarchy_level_desc_summary "
        f"--recognizer-function {module}.is_hierarchy_level_desc"
    )
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function "
        f"{module}.hierarchy_summary "
        f"--recognizer-function {module}.is_cuda_hierarchy"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class "
        f"{module}.HierarchySyntheticProvider "
        f"--recognizer-function {module}.is_cuda_hierarchy"
    )
