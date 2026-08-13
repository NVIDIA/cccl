# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::hierarchy."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common
import tuple as tuple_printer

import gdb
import gdb.printing


def _is_template(value_type: gdb.Type, name: str) -> bool:
    public_name = cccl_common.public_type_name(cccl_common.canonical_type(value_type))
    template_name = cccl_common.template_name(public_name)
    return template_name == f"cuda::{name}"


def _level_name(level_type: gdb.Type) -> str:
    name = cccl_common.public_type_name(cccl_common.canonical_type(level_type))
    name = name.rsplit("::", 1)[-1]
    return name.removesuffix("_level")


def _first_base(value: gdb.Value) -> gdb.Value:
    value_type = cccl_common.canonical_type(value.type)
    for field in value_type.fields():
        if field.is_base_class:
            return value.cast(field.type)
    raise ValueError(f"{value_type} has no base class")


def _hierarchy_level_desc(value: gdb.Value) -> gdb.Value:
    pending = [cccl_common.strip_reference_value(value)]
    while pending:
        candidate = pending.pop()
        candidate_type = cccl_common.canonical_type(candidate.type)
        if _is_template(candidate_type, "hierarchy_level_desc"):
            return candidate
        for field in candidate_type.fields():
            if field.is_base_class:
                pending.append(candidate.cast(field.type))
    raise ValueError(f"{value.type} is not a hierarchy level descriptor")


def _dimensions(value: gdb.Value) -> tuple[int, int, int]:
    value = cccl_common.strip_reference_value(value)
    extents = value["__exts_"]
    extents_type = cccl_common.canonical_type(extents.type)
    if not _is_template(extents_type, "std::extents"):
        raise ValueError(f"unexpected hierarchy extents type: {extents_type}")

    # GDB does not expose a variadic non-type template pack through
    # Type.template_argument, even though it retains the arguments in the name.
    arguments = cccl_common.public_type_name(extents_type).removesuffix(">")
    static_arguments = arguments.rsplit(",", 3)[-3:]
    if len(static_arguments) != 3:
        raise ValueError(f"unexpected hierarchy extents type: {extents_type}")
    static_extents = [int(argument.strip()) for argument in static_arguments]
    dynamic_extent = (1 << (gdb.lookup_type("size_t").sizeof * 8)) - 1
    dynamic_count = static_extents.count(dynamic_extent)

    dynamic_values: gdb.Value | None = None
    if dynamic_count:
        dynamic_values = _first_base(_first_base(extents))["__vals_"]

    result = []
    dynamic_index = 0
    for static_extent in static_extents:
        if static_extent == dynamic_extent:
            if dynamic_values is None or dynamic_index >= dynamic_count:
                raise ValueError("missing dynamic hierarchy extent")
            result.append(int(dynamic_values[dynamic_index]))
            dynamic_index += 1
        else:
            result.append(static_extent)
    return result[0], result[1], result[2]


class HierarchyLevelDescPrinter:
    """Summarize one hierarchy level's dimensions."""

    def __init__(self, value: gdb.Value) -> None:
        self.dimensions = _dimensions(value)

    def to_string(self) -> str:
        x, y, z = self.dimensions
        return f"dims=(x={x}, y={y}, z={z})"


class HierarchyPrinter:
    """Expose the semantic levels of cuda::hierarchy."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.type = cccl_common.canonical_type(value.type)
        self.bottom_unit = _level_name(self.type.template_argument(0))
        self.levels = []
        descs = value["__descs_"]
        for _, desc in tuple_printer.TuplePrinter(descs).children():
            desc = _hierarchy_level_desc(desc)
            desc_type = cccl_common.canonical_type(desc.type)
            self.levels.append((_level_name(desc_type.template_argument(0)), desc))

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        yield from self.levels

    def to_string(self) -> str:
        return f"cuda::hierarchy bottom_unit={self.bottom_unit}"


class HierarchyPrinterLookup(gdb.printing.PrettyPrinter):
    """Select cuda::hierarchy and hierarchy-level printers."""

    def __init__(self) -> None:
        super().__init__("cuda::hierarchy")

    def __call__(
        self, value: gdb.Value
    ) -> HierarchyPrinter | HierarchyLevelDescPrinter | None:
        try:
            if _is_template(value.type, "hierarchy"):
                return HierarchyPrinter(value)
            if _is_template(value.type, "hierarchy_level_desc"):
                return HierarchyLevelDescPrinter(value)
        except (gdb.error, IndexError, RuntimeError, TypeError, ValueError):
            return None
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::hierarchy printer with GDB."""
    gdb.printing.register_pretty_printer(
        objfile, HierarchyPrinterLookup(), replace=True
    )
