# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for CUDA type-erased memory resources."""

from __future__ import annotations

import re
from types import ModuleType

import gdb
import gdb.printing

_ABI_NAMESPACE_PATTERN = re.compile(r"::__(?:\d+|version_bump_ver\d+_)(?=::)")
_RESOURCE_NAMES = frozenset(
    {"any_resource", "any_synchronous_resource", "basic_any_resource"}
)


def public_type_name(value_type: gdb.Type) -> str:
    """Return a type name without CUDA ABI inline namespaces."""
    return _ABI_NAMESPACE_PATTERN.sub("", str(value_type))


def strip_reference(value_type: gdb.Type) -> gdb.Type:
    """Return the referenced type, or the type itself if it is not a reference.

    target() gives the referenced type, but on a non-reference it either
    resolves a typedef or raises, so test the type code first.
    """
    if value_type.code in (gdb.TYPE_CODE_REF, gdb.TYPE_CODE_RVALUE_REF):
        return value_type.target()
    return value_type


def strip_reference_value(value: gdb.Value) -> gdb.Value:
    """Return the referenced value, or the value itself if it is not a reference."""
    if value.type.code in (gdb.TYPE_CODE_REF, gdb.TYPE_CODE_RVALUE_REF):
        return value.referenced_value()
    return value


def _template_name(value_type: gdb.Type) -> str:
    return str(value_type).split("<", 1)[0]


def _is_memory_resource(value_type: gdb.Type) -> bool:
    value_type = strip_reference(value_type).strip_typedefs().unqualified()
    type_name = public_type_name(value_type)
    template_name = _template_name(value_type)
    return (
        type_name.startswith("cuda::mr::")
        and template_name.rsplit("::", 1)[-1] in _RESOURCE_NAMES
    )


def memory_resource_description(value: gdb.Value) -> str:
    value = strip_reference_value(value)
    value_type = value.type.strip_typedefs().unqualified()
    type_name = public_type_name(value_type)
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
