# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::mdspan."""

from __future__ import annotations

import re
from collections.abc import Iterator
from types import ModuleType

import memory_resource

import gdb
import gdb.printing

# static_cast<size_t>(-1), the value of cuda::std::dynamic_extent.
_DYNAMIC_EXTENT = (1 << 64) - 1

_EBCO_PATTERN = re.compile(r"__mdspan_ebco<")
_EBCO_IMPL_PATTERN = re.compile(r"__mdspan_ebco_impl<(\d+),")
_EXTENTS_PATTERN = re.compile(r"extents<([^>]*)>$")
_LAYOUT_KINDS = ("layout_left", "layout_right", "layout_stride")


def _template_name(value_type: gdb.Type) -> str:
    return str(value_type).split("<", 1)[0]


def _is_cuda_mdspan(value_type: gdb.Type) -> bool:
    # strip_typedefs resolves aliases that can hide accessibility properties.
    value_type = value_type.strip_typedefs().unqualified()
    template_name = _template_name(value_type)
    return (
        template_name.startswith("cuda::std::")
        and template_name.rsplit("::", 1)[-1] == "mdspan"
    )


def _direct_bases(value_type: gdb.Type) -> list[gdb.Field]:
    return [field for field in value_type.fields() if field.is_base_class]


def _find_ebco_base(value: gdb.Value) -> gdb.Value | None:
    """Descend through transparent wrapper bases to an ``__mdspan_ebco<...>``.

    ``mdspan`` and ``layout_right``/``layout_left``'s ``mapping`` privately
    inherit ``__mdspan_ebco<...>`` directly. ``layout_stride``'s ``mapping``
    inherits it through one intermediate ``__mapping_base<...>`` wrapper
    base, which this recurses through.
    """
    bases = _direct_bases(value.type)
    for field in bases:
        if _EBCO_PATTERN.search(str(field.type)):
            return value.cast(field.type)
    if len(bases) == 1:
        return _find_ebco_base(value.cast(bases[0].type))
    return None


def _ebco_element(ebco_value: gdb.Value, index: int) -> gdb.Value | None:
    """Extract element ``index`` from an ``__mdspan_ebco<...>`` value.

    Each element is stored via an ``__mdspan_ebco_impl<index, T>`` base,
    which either holds the element in an ``__elem_`` data member, or (empty
    base class optimization, for empty types) privately inherits it
    directly.
    """
    for field in _direct_bases(ebco_value.type):
        match = _EBCO_IMPL_PATTERN.search(str(field.type))
        if match is None or int(match.group(1)) != index:
            continue
        impl = ebco_value.cast(field.type)
        for impl_field in field.type.fields():
            if impl_field.name == "__elem_":
                return impl[impl_field]
            if impl_field.is_base_class:
                return impl.cast(impl_field.type)
    return None


def _descend_unique_base(value: gdb.Value, levels: int) -> gdb.Value | None:
    """Cast down through ``levels`` of single private-base inheritance."""
    for _ in range(levels):
        bases = _direct_bases(value.type)
        if len(bases) != 1:
            return None
        value = value.cast(bases[0].type)
    return value


def _split_top_level(text: str) -> list[str]:
    """Split ``text`` on top-level commas, ignoring commas nested in ``<>``."""
    parts = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current or parts:
        parts.append("".join(current).strip())
    return parts


def _static_extents(extents_type: gdb.Type) -> list[int] | None:
    """Parse the static extents (dynamic_extent sentinel kept) out of an
    ``extents<IndexType, Values...>`` type's name."""
    match = _EXTENTS_PATTERN.search(
        memory_resource.public_type_name(extents_type))
    if match is None:
        return None
    parts = _split_top_level(match.group(1))
    if not parts:
        return None
    return [int(value) for value in parts[1:]]


def _dynamic_values(extents_value: gdb.Value, count: int) -> list[int] | None:
    if count == 0:
        return []
    array_value = _descend_unique_base(extents_value, 2)
    if array_value is None:
        return None
    try:
        vals = array_value["__vals_"]
    except gdb.error:
        return None
    return [int(vals[i]) for i in range(count)]


def _combined_extents(static_values: list[int], dynamic_values: list[int]) -> list[int]:
    dynamic_iter = iter(dynamic_values)
    return [
        next(dynamic_iter) if value == _DYNAMIC_EXTENT else value
        for value in static_values
    ]


def _layout_kind(layout_type: gdb.Type) -> str | None:
    name = memory_resource.public_type_name(layout_type)
    for kind in _LAYOUT_KINDS:
        if name == f"cuda::std::{kind}":
            return kind
    return None


def _strides(mapping_ebco: gdb.Value, rank: int) -> list[int] | None:
    stride_value = _ebco_element(mapping_ebco, 1)
    if stride_value is None:
        return None
    try:
        vals = stride_value["__vals_"]
    except gdb.error:
        return None
    return [int(vals[i]) for i in range(rank)]


def _offset(
    kind: str, extents: list[int], strides: list[int] | None, indices: tuple[int, ...]
) -> int:
    if kind == "layout_stride":
        return sum(index * stride for index, stride in zip(indices, strides))
    positions = (
        range(len(extents)) if kind == "layout_right" else reversed(
            range(len(extents)))
    )
    result = 0
    for pos in positions:
        result = result * extents[pos] + indices[pos]
    return result


class MdspanPrinter:
    """Expose cuda::std::mdspan metadata and elements to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        self.value = value
        self.type = value.type.strip_typedefs().unqualified()
        self.type_name = memory_resource.public_type_name(self.type)

        self.extents: list[int] | None = None
        self.data: gdb.Value | None = None
        self.layout: str | None = None
        self.strides: list[int] | None = None

        self._resolve()

    def _resolve(self) -> None:
        top_ebco = _find_ebco_base(self.value)
        if top_ebco is None:
            return
        data_handle = _ebco_element(top_ebco, 0)
        mapping = _ebco_element(top_ebco, 1)
        if data_handle is None or mapping is None:
            return

        extents_type = self.type.template_argument(1)
        static_values = _static_extents(extents_type)
        if static_values is None:
            return
        rank_dynamic = sum(
            1 for value in static_values if value == _DYNAMIC_EXTENT)

        mapping_ebco = _find_ebco_base(mapping)
        dynamic_values: list[int] = []
        if rank_dynamic > 0:
            if mapping_ebco is None:
                return
            extents_value = _ebco_element(mapping_ebco, 0)
            if extents_value is None:
                return
            values = _dynamic_values(extents_value, rank_dynamic)
            if values is None:
                return
            dynamic_values = values
        self.extents = _combined_extents(static_values, dynamic_values)

        self.layout = _layout_kind(self.type.template_argument(2))
        if self.layout == "layout_stride" and len(self.extents) > 0:
            if mapping_ebco is None:
                return
            self.strides = _strides(mapping_ebco, len(self.extents))
            if self.strides is None:
                return

        if data_handle.type.strip_typedefs().code == gdb.TYPE_CODE_PTR:
            self.data = data_handle

    def _can_index(self) -> bool:
        if self.extents is None or self.data is None or self.layout is None:
            return False
        return self.layout != "layout_stride" or self.strides is not None

    def _size(self) -> int:
        size = 1
        for extent in self.extents:
            size *= extent
        return size

    def _element_at(self, indices: tuple[int, ...]) -> gdb.Value:
        offset = _offset(self.layout, self.extents, self.strides, indices)
        return (self.data + offset).dereference()

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        if not self._can_index():
            return
        for flat in range(self._size()):
            indices = [0] * len(self.extents)
            remaining = flat
            for pos in reversed(range(len(self.extents))):
                indices[pos] = remaining % self.extents[pos]
                remaining //= self.extents[pos]
            label = "[" + ",".join(str(index) for index in indices) + "]"
            yield label, self._element_at(tuple(indices))

    def to_string(self) -> str:
        if self.extents is None:
            return self.type_name
        extents_text = ", ".join(str(extent) for extent in self.extents)
        return f"{self.type_name} of extents [{extents_text}]"


class MdspanPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the cuda::std::mdspan printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::std::mdspan")

    def __call__(self, value: gdb.Value) -> MdspanPrinter | None:
        if _is_cuda_mdspan(value.type):
            return MdspanPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::std::mdspan printer with GDB."""
    gdb.printing.register_pretty_printer(
        objfile, MdspanPrinterLookup(), replace=True)
