# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::optional."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing


def _is_cuda_optional(value_type: gdb.Type) -> bool:
    value_type = cccl_common.canonical_type(value_type)
    template_name = cccl_common.template_name(cccl_common.public_type_name(value_type))
    return (
        template_name.startswith("cuda::std::")
        and template_name.rsplit("::", 1)[-1] == "optional"
    )


class OptionalPrinter:
    """Expose cuda::std::optional metadata and value to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        # Use strip_typedefs() to resolve type aliases/typedefs (e.g. optional_alias)
        # while keeping const/reference qualifiers.
        self.type_name = cccl_common.public_type_name(value.type.strip_typedefs())
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        if any(field.name == "__value_" for field in self.type.fields()):
            self.engaged = int(value["__value_"]) != 0
            if self.engaged:
                self.val = value["__value_"].dereference()
        else:
            self.engaged = bool(value["__engaged_"])
            if self.engaged:
                # __storage_ is a member of the base class __optional_destruct_base
                self.val = value["__storage_"]["__val_"]

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        if not self.engaged:
            return
        yield "value", self.val

    def to_string(self) -> str:
        if not self.engaged:
            return "cuda::std::nullopt"
        return self.type_name


class OptionalPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the optional printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::std::optional")

    def __call__(self, value: gdb.Value) -> OptionalPrinter | None:
        if _is_cuda_optional(value.type):
            return OptionalPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda optional printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, OptionalPrinterLookup(), replace=True)
