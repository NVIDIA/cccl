# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::tuple."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing


def _is_cuda_tuple(value_type: gdb.Type) -> bool:
    # strip_typedefs resolves aliases that can hide accessibility properties.
    value_type = cccl_common.canonical_type(value_type)
    public_name = cccl_common.public_type_name(value_type)
    template_name = cccl_common.template_name(value_type)
    return (
        public_name.startswith("cuda::std::")
        and template_name.rsplit("::", 1)[-1] == "tuple"
    )


def _leaf_fields(base_type: gdb.Type) -> list[tuple[int, gdb.Field]]:
    """Return each ``__tuple_leaf`` base field paired with its tuple index."""
    leaves = []
    for field in base_type.fields():
        if not field.is_base_class:
            continue
        if "__tuple_leaf" not in str(field.type):
            continue
        index = int(field.type.template_argument(0))
        leaves.append((index, field))
    leaves.sort(key=lambda pair: pair[0])
    return leaves


def _leaf_element(base_value: gdb.Value, field: gdb.Field) -> gdb.Value:
    """Extract the element stored in one ``__tuple_leaf`` base.

    ``__tuple_leaf`` either holds the element in a ``__value_`` data member, or
    (for empty, non-final element types) applies the empty base class
    optimization and privately inherits from the element type directly.
    """
    leaf = base_value.cast(field.type)
    element: gdb.Value = leaf
    for leaf_field in field.type.fields():
        if leaf_field.name == "__value_":
            element = leaf[leaf_field]
            break
        if leaf_field.is_base_class:
            element = leaf.cast(leaf_field.type)
            break
    # Reference elements print as a bare address unless dereferenced explicitly.
    if element.type.code == gdb.TYPE_CODE_REF:
        element = element.referenced_value()
    return element


class TuplePrinter:
    """Expose cuda::std::tuple elements to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        # A fully empty tuple (e.g. cuda::std::tuple<>) can compile to a type
        # with no debug-visible __base_ member at all.
        try:
            self.leaves = _leaf_fields(value["__base_"].type)
        except gdb.error:
            self.leaves = []

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        if not self.leaves:
            return
        base_value = self.value["__base_"]
        for index, field in self.leaves:
            yield f"[{index}]", _leaf_element(base_value, field)

    def to_string(self) -> str:
        return cccl_common.public_type_name(self.type)


class TuplePrinterLookup(gdb.printing.PrettyPrinter):
    """Select the cuda::std::tuple printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::std::tuple")

    def __call__(self, value: gdb.Value) -> TuplePrinter | None:
        if _is_cuda_tuple(value.type):
            return TuplePrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::std::tuple printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, TuplePrinterLookup(), replace=True)
