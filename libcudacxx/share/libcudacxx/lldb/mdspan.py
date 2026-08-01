# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::mdspan."""

from __future__ import annotations

import re
from typing import NamedTuple

import lldb

_MDSPAN_PATTERN = re.compile(r"^cuda::std::mdspan<.+>$")
_EBCO_PATTERN = re.compile(r"__mdspan_ebco<")
_EBCO_IMPL_PATTERN = re.compile(r"__mdspan_ebco_impl<(\d+),")
_EXTENTS_PATTERN = re.compile(r"extents<([^>]*)>$")
_LAYOUT_KINDS = ("layout_left", "layout_right", "layout_stride")
# static_cast<size_t>(-1), the value of cuda::std::dynamic_extent.
_DYNAMIC_EXTENT = (1 << 64) - 1
InternalDict = dict[str, object]


def is_cuda_mdspan(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = (
        value_type.GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName() or ""
    )
    return _MDSPAN_PATTERN.fullmatch(type_name) is not None


def _direct_bases(sb_type: lldb.SBType) -> list[lldb.SBTypeMember]:
    return [
        sb_type.GetDirectBaseClassAtIndex(i)
        for i in range(sb_type.GetNumberOfDirectBaseClasses())
    ]


def _find_ebco_base(value: lldb.SBValue) -> lldb.SBValue | None:
    """Descend through transparent wrapper bases to an ``__mdspan_ebco<...>``.

    ``mdspan`` and ``layout_right``/``layout_left``'s ``mapping`` privately
    inherit ``__mdspan_ebco<...>`` directly. ``layout_stride``'s ``mapping``
    inherits it through one intermediate ``__mapping_base<...>`` wrapper
    base, which this recurses through.
    """
    bases = _direct_bases(value.GetType())
    for base in bases:
        if _EBCO_PATTERN.search(base.GetName() or ""):
            return value.CreateChildAtOffset(
                "__mdspan_ebco", base.GetOffsetInBytes(), base.GetType()
            )
    if len(bases) == 1:
        base = bases[0]
        wrapper = value.CreateChildAtOffset(
            "__wrapper", base.GetOffsetInBytes(), base.GetType()
        )
        return _find_ebco_base(wrapper)
    return None


def _ebco_element(
    ebco_value: lldb.SBValue, index: int, name: str
) -> lldb.SBValue | None:
    """Extract element ``index`` from an ``__mdspan_ebco<...>`` value.

    Each element is stored via an ``__mdspan_ebco_impl<index, T>`` base,
    which either holds the element in an ``__elem_`` data member, or (empty
    base class optimization, for empty types) privately inherits it
    directly. An empty element shares its offset with whatever follows it,
    but LLDB's ``CreateChildAtOffset`` caches values by (parent, offset) and
    returns the first type ever requested there, mismatching later requests
    at that offset. Building the empty element from freestanding data
    instead avoids poisoning that cache.
    """
    for base in _direct_bases(ebco_value.GetType()):
        match = _EBCO_IMPL_PATTERN.search(base.GetName() or "")
        if match is None or int(match.group(1)) != index:
            continue
        impl_type = base.GetType()
        if impl_type.GetNumberOfFields() == 0:
            element_type = impl_type.GetDirectBaseClassAtIndex(0).GetType()
            byte_size = element_type.GetByteSize() or 1
            data = lldb.SBData()
            error = lldb.SBError()
            data.SetData(error, bytes(byte_size), lldb.eByteOrderLittle, 8)
            return ebco_value.CreateValueFromData(name, data, element_type)
        impl = ebco_value.CreateChildAtOffset(name, base.GetOffsetInBytes(), impl_type)
        return impl.GetChildMemberWithName("__elem_")
    return None


def _descend_unique_base(value: lldb.SBValue, levels: int) -> lldb.SBValue | None:
    """Cast down through ``levels`` of single private-base inheritance."""
    for _ in range(levels):
        bases = _direct_bases(value.GetType())
        if len(bases) != 1:
            return None
        base = bases[0]
        value = value.CreateChildAtOffset(
            "__base", base.GetOffsetInBytes(), base.GetType()
        )
    return value


def _static_extents(extents_type: lldb.SBType) -> list[int] | None:
    """Parse the static extents (dynamic_extent sentinel kept) out of an
    ``extents<IndexType, Values...>`` type's display name."""
    type_name = extents_type.GetDisplayTypeName() or extents_type.GetName() or ""
    match = _EXTENTS_PATTERN.search(type_name)
    if match is None:
        return None
    parts = [part.strip() for part in match.group(1).split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return None
    return [int(value) for value in parts[1:]]


def _dynamic_values(extents_value: lldb.SBValue, count: int) -> list[int] | None:
    if count == 0:
        return []
    array_value = _descend_unique_base(extents_value, 2)
    if array_value is None:
        return None
    vals = array_value.GetChildMemberWithName("__vals_")
    if not vals.IsValid():
        return None
    return [vals.GetChildAtIndex(i).GetValueAsSigned(0) for i in range(count)]


def _combined_extents(static_values: list[int], dynamic_values: list[int]) -> list[int]:
    dynamic_iter = iter(dynamic_values)
    return [
        next(dynamic_iter) if value == _DYNAMIC_EXTENT else value
        for value in static_values
    ]


def _layout_kind(layout_type: lldb.SBType) -> str | None:
    name = layout_type.GetDisplayTypeName() or layout_type.GetName() or ""
    for kind in _LAYOUT_KINDS:
        if name == f"cuda::std::{kind}":
            return kind
    return None


def _strides(mapping_ebco: lldb.SBValue, rank: int) -> list[int] | None:
    stride_value = _ebco_element(mapping_ebco, 1, "__strides")
    if stride_value is None:
        return None
    vals = stride_value.GetChildMemberWithName("__vals_")
    if not vals.IsValid():
        return None
    return [vals.GetChildAtIndex(i).GetValueAsSigned(0) for i in range(rank)]


def _offset(
    kind: str, extents: list[int], strides: list[int] | None, indices: tuple[int, ...]
) -> int:
    if kind == "layout_stride":
        return sum(index * stride for index, stride in zip(indices, strides))
    positions = (
        range(len(extents)) if kind == "layout_right" else reversed(range(len(extents)))
    )
    result = 0
    for pos in positions:
        result = result * extents[pos] + indices[pos]
    return result


class MdspanInfo(NamedTuple):
    type_name: str
    extents: list[int] | None
    data: lldb.SBValue | None
    layout: str | None
    strides: list[int] | None

    def can_index(self) -> bool:
        if self.extents is None or self.data is None or self.layout is None:
            return False
        return self.layout != "layout_stride" or self.strides is not None

    def size(self) -> int:
        size = 1
        for extent in self.extents:
            size *= extent
        return size


def _display_type_name(value: lldb.SBValue) -> str:
    return (
        value.GetType().GetCanonicalType().GetUnqualifiedType().GetDisplayTypeName()
        or ""
    )


def _mdspan_info(value: lldb.SBValue) -> MdspanInfo | None:
    value = value.GetNonSyntheticValue()
    mdspan_type = value.GetType().GetCanonicalType().GetUnqualifiedType()
    type_name = _display_type_name(value)

    top_ebco = _find_ebco_base(value)
    if top_ebco is None:
        return None
    data_handle = _ebco_element(top_ebco, 0, "__data_handle")
    mapping = _ebco_element(top_ebco, 1, "__mapping")
    if data_handle is None or mapping is None:
        return MdspanInfo(type_name, None, None, None, None)

    extents_type = mdspan_type.GetTemplateArgumentType(1)
    static_values = _static_extents(extents_type)
    if static_values is None:
        return MdspanInfo(type_name, None, None, None, None)
    rank_dynamic = sum(1 for value_ in static_values if value_ == _DYNAMIC_EXTENT)

    mapping_ebco = _find_ebco_base(mapping)
    dynamic_values: list[int] = []
    if rank_dynamic > 0:
        if mapping_ebco is None:
            return MdspanInfo(type_name, None, None, None, None)
        extents_value = _ebco_element(mapping_ebco, 0, "__extents")
        if extents_value is None:
            return MdspanInfo(type_name, None, None, None, None)
        values = _dynamic_values(extents_value, rank_dynamic)
        if values is None:
            return MdspanInfo(type_name, None, None, None, None)
        dynamic_values = values
    extents = _combined_extents(static_values, dynamic_values)

    layout = _layout_kind(mdspan_type.GetTemplateArgumentType(2))
    strides = None
    if layout == "layout_stride" and len(extents) > 0:
        if mapping_ebco is None:
            return MdspanInfo(type_name, extents, None, layout, None)
        strides = _strides(mapping_ebco, len(extents))

    data = None
    if data_handle.GetType().GetCanonicalType().IsPointerType():
        data = data_handle

    return MdspanInfo(type_name, extents, data, layout, strides)


def mdspan_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str | None:
    info = _mdspan_info(value)
    if info is None or info.extents is None:
        return None
    extents_text = ", ".join(str(extent) for extent in info.extents)
    return f"of extents [{extents_text}]"


class MdspanSyntheticProvider:
    """Expose cuda::std::mdspan elements as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.value = value.GetNonSyntheticValue()
        self.info: MdspanInfo | None = None
        self.update()

    def update(self) -> bool:
        self.info = _mdspan_info(self.value)
        return True

    def _element_at(self, indices: tuple[int, ...]) -> lldb.SBValue:
        element_type = self.info.data.GetType().GetPointeeType()
        offset = _offset(
            self.info.layout, self.info.extents, self.info.strides, indices
        )
        byte_offset = offset * element_type.GetByteSize()
        return self.info.data.CreateChildAtOffset(
            self._label(indices), byte_offset, element_type
        )

    @staticmethod
    def _label(indices: tuple[int, ...]) -> str:
        return "[" + ",".join(str(index) for index in indices) + "]"

    def num_children(self) -> int:
        if self.info is None or not self.info.can_index():
            return 0
        return self.info.size()

    def has_children(self) -> bool:
        return self.num_children() != 0

    def get_type_name(self) -> str:
        return _display_type_name(self.value)

    def get_child_index(self, name: str) -> int:
        if self.info is None or not self.info.can_index():
            return -1
        stripped = name.strip("[]")
        try:
            parsed = (
                tuple(int(index) for index in stripped.split(",")) if stripped else ()
            )
        except ValueError:
            return -1
        flat = 0
        for extent, index in zip(self.info.extents, parsed):
            flat = flat * extent + index
        return flat

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if self.info is None or not self.info.can_index():
            return None
        if index < 0 or index >= self.info.size():
            return None
        extents = self.info.extents
        indices = [0] * len(extents)
        remaining = index
        for pos in reversed(range(len(extents))):
            indices[pos] = remaining % extents[pos]
            remaining //= extents[pos]
        return self._element_at(tuple(indices))


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::std::mdspan formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function {module}.mdspan_summary "
        f"--recognizer-function {module}.is_cuda_mdspan"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.MdspanSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_mdspan"
    )
