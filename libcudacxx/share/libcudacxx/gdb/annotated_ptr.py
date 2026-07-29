# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::annotated_ptr."""

from __future__ import annotations

import re
from types import ModuleType

import gdb
import gdb.printing

_ABI_NAMESPACE_PATTERN = re.compile(r"::__(?:\d+|version_bump_ver\d+_)(?=::)")


def public_type_name(value_type: gdb.Type) -> str:
    """Return a type name without CUDA ABI inline namespaces."""
    return _ABI_NAMESPACE_PATTERN.sub("", str(value_type))


def _template_name(value_type: gdb.Type) -> str:
    return str(value_type).split("<", 1)[0]


def _is_annotated_ptr(value_type: gdb.Type) -> bool:
    value_type = value_type.strip_typedefs().unqualified()
    type_name = public_type_name(value_type)
    template_name = _template_name(value_type)
    return (
        type_name.startswith("cuda::")
        and template_name.rsplit("::", 1)[-1] == "annotated_ptr"
    )


class AnnotatedPtrPrinter:
    """Summarize a CUDA annotated_ptr smart pointer."""

    def __init__(self, value: gdb.Value) -> None:
        self.value = value
        self.type = value.type.strip_typedefs().unqualified()
        self.type_name = public_type_name(self.type)

    def to_string(self) -> str:
        try:
            # Get the template arguments: _Tp (pointee type) and _Property
            pointee_type = self.type.template_argument(0)
            pointee_type_name = public_type_name(pointee_type)

            property_type = self.type.template_argument(1)
            property_type_name = public_type_name(property_type)

            # Access the __repr member (the wrapped pointer)
            repr_ptr = self.value["__repr"]

            # Get the pointer value
            ptr_value = int(repr_ptr)

            # Format the pointer display
            if ptr_value == 0:
                ptr_display = "nullptr"
            else:
                ptr_display = f"{ptr_value:#x}"

            # Construct the full type name with template arguments
            full_type = f"{self.type_name}<{pointee_type_name}, {property_type_name}>"

            return f"{full_type} -> {ptr_display}"
        except (gdb.error, ValueError):
            return self.type_name


class AnnotatedPtrPrinterLookup(gdb.printing.PrettyPrinter):
    """Select printers for cuda::annotated_ptr types."""

    def __init__(self) -> None:
        super().__init__("cuda::annotated_ptr")

    def __call__(self, value: gdb.Value) -> AnnotatedPtrPrinter | None:
        if _is_annotated_ptr(value.type):
            return AnnotatedPtrPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register CUDA annotated_ptr formatters with GDB."""
    gdb.printing.register_pretty_printer(
        objfile, AnnotatedPtrPrinterLookup(), replace=True
    )
