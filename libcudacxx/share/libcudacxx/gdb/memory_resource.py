# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for CUDA type-erased memory resources."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing

_RESOURCE_NAMES = frozenset(
    {"any_resource", "any_synchronous_resource", "basic_any_resource"}
)


def _is_memory_resource(value_type: gdb.Type) -> bool:
    value_type = cccl_common.canonical_type(value_type)
    type_name = cccl_common.public_type_name(value_type)
    return (
        type_name.startswith("cuda::mr::")
        and cccl_common.template_name(value_type).rsplit("::", 1)[-1] in _RESOURCE_NAMES
    )


def _resource_state(value: gdb.Value) -> tuple[str, gdb.Value | None]:
    tagged_vptr = int(value["__vptr_"]["__ptr_"])
    if tagged_vptr == 0:
        return "empty", None

    buffer = value["__buffer_"]
    void_pointer = gdb.lookup_type("void").pointer()
    if tagged_vptr & 1:
        return "in-situ", buffer.address.cast(void_pointer)

    resource = buffer.address.cast(void_pointer.pointer()).dereference()
    return "heap", resource


def memory_resource_description(value: gdb.Value) -> str:
    value = cccl_common.strip_reference_value(value)
    type_name = cccl_common.canonical_type_name(value.type)
    try:
        address = int(value.address)
    except (gdb.error, TypeError):
        return type_name
    return f"{type_name} @ {address:#x}"


class MemoryResourcePrinter:
    """Summarize a CUDA type-erased memory resource and expose its object pointer."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        value_type = cccl_common.canonical_type(value.type)
        self.type_name = cccl_common.public_type_name(value_type)
        try:
            self.storage, self.resource = _resource_state(value)
        except (gdb.error, KeyError, TypeError):
            self.storage, self.resource = "unavailable", None

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        if self.resource is not None:
            yield "resource", self.resource

    def to_string(self) -> str:
        if self.storage == "unavailable":
            return f"{self.type_name} storage=unavailable"
        if self.resource is None:
            return f"{self.type_name} storage=0x0"
        return f"{self.type_name} storage={int(self.resource):#x} ({self.storage})"


class MemoryResourcePrinterLookup(gdb.printing.PrettyPrinter):
    """Select printers for public CUDA type-erased resource types."""

    def __init__(self) -> None:
        super().__init__("cuda::mr::memory_resource")

    def __call__(self, value: gdb.Value) -> MemoryResourcePrinter | None:
        if _is_memory_resource(value.type):
            return MemoryResourcePrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register CUDA memory-resource formatters with GDB."""
    gdb.printing.register_pretty_printer(
        objfile, MemoryResourcePrinterLookup(), replace=True
    )
