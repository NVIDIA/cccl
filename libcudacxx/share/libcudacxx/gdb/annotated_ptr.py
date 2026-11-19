# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::annotated_ptr."""

from __future__ import annotations

from types import ModuleType

import cccl_common

import gdb
import gdb.printing


def _is_annotated_ptr(value_type: gdb.Type) -> bool:
    """Check if a GDB type represents cuda::annotated_ptr."""
    value_type = cccl_common.canonical_type(value_type)
    type_name = cccl_common.public_type_name(value_type)
    template_name = cccl_common.template_name(value_type)
    return (
        type_name.startswith("cuda::")
        and template_name.rsplit("::", 1)[-1] == "annotated_ptr"
    )


class AnnotatedPtrPrinter:
    """Expose cuda::annotated_ptr type and pointer value to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        # Strip reference to access members
        self.value = cccl_common.strip_reference_value(value)
        # Store canonical type for type name display
        self.canonical_type_obj = cccl_common.canonical_type(self.value.type)
        # Extract base type name without template arguments to avoid duplication
        self.type_name = cccl_common.template_name(
            cccl_common.public_type_name(self.canonical_type_obj)
        )

    def _template_arguments(self) -> tuple[str, str]:
        """Return the pointee and property type names."""
        pointee_type = self.canonical_type_obj.template_argument(0)
        property_type = self.canonical_type_obj.template_argument(1)
        return (
            cccl_common.public_type_name(pointee_type),
            cccl_common.public_type_name(property_type),
        )

    def _pointer_display(self) -> str:
        """Return the stored pointer, spelling a null pointer as nullptr."""
        ptr_value = int(self.value["__repr"])
        return "nullptr" if ptr_value == 0 else f"{ptr_value:#x}"

    def to_string(self) -> str:
        """Combine type info and pointer value into annotated_ptr display."""
        pointee_type_name, property_type_name = self._template_arguments()
        full_type = f"{self.type_name}<{pointee_type_name}, {property_type_name}>"
        return f"{full_type} -> {self._pointer_display()}"


class AnnotatedPtrPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the cuda::annotated_ptr printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::annotated_ptr")

    def __call__(self, value: gdb.Value) -> AnnotatedPtrPrinter | None:
        if _is_annotated_ptr(value.type):
            return AnnotatedPtrPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::annotated_ptr printer with GDB."""
    gdb.printing.register_pretty_printer(
        objfile, AnnotatedPtrPrinterLookup(), replace=True
    )
