# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::mdspan and its cuda:: accessibility wrappers."""

from __future__ import annotations

import re
from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing

_EBCO_PATTERN = re.compile(r"__mdspan_ebco<")
_EBCO_IMPL_PATTERN = re.compile(r"__mdspan_ebco_impl<\s*(\d+)\s*,")
_EXTENTS_PATTERN = re.compile(r"extents<\s*([^>]*)>$")
_LAYOUT_KINDS = ("layout_left", "layout_right", "layout_stride")
# cudaMemcpyDefault: infer host/device direction from the pointers themselves.
_CUDA_MEMCPY_DEFAULT = 4
# cuda:: accessibility wrappers around cuda::std::mdspan.
_WRAPPER_MDSPAN_NAMES = frozenset(
    {
        "host_mdspan",
        "device_mdspan",
        "managed_mdspan",
        "restrict_mdspan",
        "shared_memory_mdspan",
    }
)


def _dynamic_extent() -> int:
    """Return cuda::std::dynamic_extent, using the target's real size_t width."""
    try:
        size_t_bits = gdb.lookup_type("size_t").sizeof * 8
    except gdb.error:
        size_t_bits = 64
    return (1 << size_t_bits) - 1


def _is_cuda_mdspan(value_type: gdb.Type) -> bool:
    # canonical_type resolves references/typedefs that can hide accessibility
    # properties.
    value_type = cccl_common.canonical_type(value_type)
    template_name = cccl_common.template_name(value_type)
    if (
        template_name.startswith("cuda::std::")
        and template_name.rsplit("::", 1)[-1] == "mdspan"
    ):
        return True
    return (
        template_name.startswith("cuda::")
        and template_name.rsplit("::", 1)[-1] in _WRAPPER_MDSPAN_NAMES
    )


def _direct_bases(value_type: gdb.Type) -> list[gdb.Field]:
    return [field for field in value_type.fields() if field.is_base_class]


def _find_ebco_base(value: gdb.Value) -> gdb.Value | None:
    """Descend through transparent wrapper bases to an ``__mdspan_ebco<...>``.

    Recurses through layout_stride's ``__mapping_base<...>`` and cuda:: wrappers like ``host_mdspan``.
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

    Held in an ``__elem_`` member, or (EBCO) inherited directly for empty types.
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
    match = _EXTENTS_PATTERN.search(cccl_common.public_type_name(extents_type))
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


def _combined_extents(
    static_values: list[int], dynamic_values: list[int], dynamic_extent: int
) -> list[int]:
    dynamic_iter = iter(dynamic_values)
    return [
        next(dynamic_iter) if value == dynamic_extent else value
        for value in static_values
    ]


def _layout_kind(layout_type: gdb.Type) -> str | None:
    name = cccl_common.public_type_name(layout_type)
    for kind in _LAYOUT_KINDS:
        if name == f"cuda::std::{kind}":
            return kind
    return None


def _is_default_accessor(accessor_type: gdb.Type, element_type: gdb.Type) -> bool:
    """Return whether accessor_type is cuda::std::default_accessor<element_type>."""
    stripped = accessor_type.strip_typedefs().unqualified()
    name = cccl_common.public_type_name(stripped)
    if name.split("<", 1)[0] != "cuda::std::default_accessor":
        return False
    try:
        accessor_element = stripped.template_argument(0)
    except gdb.error:
        return False
    return cccl_common.public_type_name(
        accessor_element
    ) == cccl_common.public_type_name(element_type)


def _strides(mapping_ebco: gdb.Value, rank: int) -> list[int] | None:
    stride_value = _ebco_element(mapping_ebco, 1)
    if stride_value is None:
        return None
    try:
        vals = stride_value["__vals_"]
    except gdb.error:
        return None
    return [int(vals[i]) for i in range(rank)]


def _required_span_size(
    kind: str | None, extents: list[int], strides: list[int] | None
) -> int:
    """Return the element count spanned by the mapping, mirroring
    layout::mapping::required_span_size()."""
    if not extents:
        return 1
    if any(extent == 0 for extent in extents):
        return 0
    if kind == "layout_stride":
        if strides is None:
            return 0
        return 1 + sum(
            (extent - 1) * stride for extent, stride in zip(extents, strides)
        )
    size = 1
    for extent in extents:
        size *= extent
    return size


def _offset(
    kind: str, extents: list[int], strides: list[int] | None, indices: tuple[int, ...]
) -> int:
    if kind == "layout_stride":
        # Rank 0 has no indices to stride over, so strides may legitimately be None.
        if not indices:
            return 0
        return sum(index * stride for index, stride in zip(indices, strides))
    positions = (
        range(len(extents)) if kind == "layout_right" else reversed(range(len(extents)))
    )
    result = 0
    for pos in positions:
        result = result * extents[pos] + indices[pos]
    return result


class MdspanPrinter:
    """Expose cuda::std::mdspan metadata and elements to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)

        self.extents: list[int] | None = None
        self.data: gdb.Value | None = None
        self.layout: str | None = None
        self.strides: list[int] | None = None
        self.host_copy: gdb.Value | None = None
        self.copy_failed = False

        self._resolve()
        self._copy_to_host()

        self.type_name = self._build_type_name()

    def __del__(self) -> None:
        """Release the inferior-heap copy staged by _copy_to_host."""
        if self.host_copy is None:
            return
        try:
            gdb.parse_and_eval(f"(void)free((void*){int(self.host_copy):#x})")
        except gdb.error:
            pass
        self.host_copy = None

    def _build_type_name(self) -> str:
        """Elide LayoutPolicy/AccessorPolicy when they're mdspan's defaults,
        matching LLDB's own elided display name.
        """
        dynamic_extent = _dynamic_extent()
        type_name = cccl_common.public_type_name(self.type)
        type_name = re.sub(rf"\b{dynamic_extent}\b", "dynamic_extent", type_name)

        try:
            element_type = self.type.template_argument(0)
            layout_type = self.type.template_argument(2)
            accessor_type = self.type.template_argument(3)
        except gdb.error:
            return type_name
        if not _is_default_accessor(accessor_type, element_type):
            return type_name

        class_name, angle_bracket, rest = type_name.partition("<")
        if not angle_bracket or not rest.endswith(">"):
            return type_name
        parts = _split_top_level(rest[:-1])
        if len(parts) != 4:
            return type_name
        keep = 2 if _layout_kind(layout_type) == "layout_right" else 3
        return f"{class_name}<{', '.join(parts[:keep])}>"

    def _resolve(self) -> None:
        top_ebco = _find_ebco_base(self.value)
        if top_ebco is None:
            return
        data_handle = _ebco_element(top_ebco, 0)
        mapping = _ebco_element(top_ebco, 1)
        if data_handle is None or mapping is None:
            return

        # Find the real cuda::std::mdspan type, skipping any cuda:: wrapper.
        mdspan_type = self.type
        while cccl_common.template_name(mdspan_type).rsplit("::", 1)[-1] != "mdspan":
            bases = _direct_bases(mdspan_type)
            if len(bases) != 1:
                return
            mdspan_type = bases[0].type
        extents_type = mdspan_type.template_argument(1)
        layout_type = mdspan_type.template_argument(2)

        dynamic_extent = _dynamic_extent()
        static_values = _static_extents(extents_type)
        if static_values is None:
            return
        rank_dynamic = sum(1 for value in static_values if value == dynamic_extent)

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
        self.extents = _combined_extents(static_values, dynamic_values, dynamic_extent)

        self.layout = _layout_kind(layout_type)
        if self.layout == "layout_stride" and len(self.extents) > 0:
            if mapping_ebco is None:
                return
            self.strides = _strides(mapping_ebco, len(self.extents))
            if self.strides is None:
                return

        if data_handle.type.strip_typedefs().code == gdb.TYPE_CODE_PTR:
            self.data = data_handle

    def _copy_to_host(self) -> None:
        """Stage addressed elements into host memory via an inferior
        cudaMemcpy call (mirrors cuda::buffer's printer), since the data
        handle may be device memory GDB would silently read as zeros.
        """
        if self.extents is None or self.data is None or self.layout is None:
            return
        span = _required_span_size(self.layout, self.extents, self.strides)
        if span == 0:
            return
        element_type = self.data.type.target()
        byte_count = span * element_type.sizeof
        try:
            host_copy = gdb.parse_and_eval(f"(void*)malloc({byte_count})")
            if int(host_copy) == 0:
                self.copy_failed = True
                return
            status = gdb.parse_and_eval(
                "(int)cudaMemcpy((void*)"
                f"{int(host_copy):#x}, (const void*){int(self.data):#x}, {byte_count}, "
                f"{_CUDA_MEMCPY_DEFAULT})"
            )
            if int(status) != 0:
                gdb.parse_and_eval(f"(void)free((void*){int(host_copy):#x})")
                self.copy_failed = True
                return
        except gdb.error:
            self.copy_failed = True
            return
        self.host_copy = host_copy.cast(element_type.pointer())

    def _can_index(self) -> bool:
        return self.host_copy is not None

    def _size(self) -> int:
        size = 1
        for extent in self.extents:
            size *= extent
        return size

    def _element_at(self, indices: tuple[int, ...]) -> gdb.Value:
        offset = _offset(self.layout, self.extents, self.strides, indices)
        return (self.host_copy + offset).dereference()

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
        details = []
        if self.extents is not None:
            extents_text = ", ".join(str(extent) for extent in self.extents)
            details.append(f"rank={len(self.extents)}")
            details.append(f"extents=[{extents_text}]")
        if self.data is not None:
            address = int(self.data)
            data_text = f"data={address:#x}"
            if self.copy_failed:
                data_text += f" <error: Cannot access memory at address {address:#x}>"
            details.append(data_text)
        if not details:
            return self.type_name
        return f"{self.type_name} {', '.join(details)}"


class MdspanPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the cuda::std::mdspan printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::mdspan")

    def __call__(self, value: gdb.Value) -> MdspanPrinter | None:
        if _is_cuda_mdspan(value.type):
            return MdspanPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::std::mdspan printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, MdspanPrinterLookup(), replace=True)
