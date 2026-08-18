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
    public_name = cccl_common.public_type_name(value_type)
    template_name = cccl_common.template_name(value_type)
    return (
        public_name.startswith("cuda::std::")
        and template_name.rsplit("::", 1)[-1] == "optional"
    )


def _has_field(value_type: gdb.Type, name: str) -> bool:
    return any(field.name == name for field in value_type.fields())


class OptionalPrinter:
    """Expose the contained value of a cuda::std::optional to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        self.type_name = cccl_common.public_type_name(self.type)
        # optional<T&> stores a pointer that is null when disengaged; every other
        # specialization keeps an engaged flag next to a union payload.
        self.pointer = (
            self.value["__value_"] if _has_field(self.type, "__value_") else None
        )
        if self.pointer is not None:
            self.engaged = int(self.pointer) != 0
        else:
            self.engaged = bool(self.value["__engaged_"])

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        # The payload is only read once the engaged state says it holds a value;
        # a disengaged optional may still carry the bytes of a previous value.
        if not self.engaged:
            return
        if self.pointer is not None:
            yield "[contained value]", self.pointer.dereference()
        else:
            yield "[contained value]", self.value["__storage_"]["__val_"]

    def to_string(self) -> str:
        # Match the libstdc++ std::optional printer: "T [no contained value]" when empty.
        if self.engaged:
            return self.type_name
        return f"{self.type_name} [no contained value]"


class OptionalPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the cuda::std::optional printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::std::optional")

    def __call__(self, value: gdb.Value) -> OptionalPrinter | None:
        if _is_cuda_optional(value.type):
            return OptionalPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::std::optional printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, OptionalPrinterLookup(), replace=True)
