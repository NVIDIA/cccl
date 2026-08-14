# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for CUDA type-erased memory resources."""

from __future__ import annotations

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


def memory_resource_description(value: gdb.Value) -> str:
    value = cccl_common.strip_reference_value(value)
    type_name = cccl_common.canonical_type_name(value.type)
    try:
        address = int(value.address)
    except (gdb.error, TypeError):
        return type_name
    return f"{type_name} @ {address:#x}"


class MemoryResourcePrinter:
    """Summarize a CUDA type-erased memory resource."""

    def __init__(self, value: gdb.Value) -> None:
        self.value = value

    def to_string(self) -> str:
        return memory_resource_description(self.value)


class MemoryResourcePrinterLookup(gdb.printing.PrettyPrinter):
    """Select printers for public CUDA type-erased resource types."""

    def __init__(self) -> None:
        super().__init__("cuda::mr::any_resource")

    def __call__(self, value: gdb.Value) -> MemoryResourcePrinter | None:
        if _is_memory_resource(value.type):
            return MemoryResourcePrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register CUDA memory-resource formatters with GDB."""
    gdb.printing.register_pretty_printer(
        objfile, MemoryResourcePrinterLookup(), replace=True
    )
