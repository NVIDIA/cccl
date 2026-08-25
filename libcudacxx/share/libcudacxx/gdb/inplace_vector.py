# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::inplace_vector."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing


def _is_cuda_inplace_vector(value_type: gdb.Type) -> bool:
    value_type = cccl_common.canonical_type(value_type)
    template_name = cccl_common.template_name(cccl_common.public_type_name(value_type))
    return template_name == "cuda::std::inplace_vector"


class InplaceVectorPrinter:
    """Expose cuda::std::inplace_vector size and constructed elements to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        self.type_name = cccl_common.public_type_name(self.type)
        self.element_type = self.type.template_argument(0)
        self.capacity = int(self.type.template_argument(1))
        # The zero-capacity specialization has no members at all; do not read any.
        self.size = 0 if self.capacity == 0 else int(self.value["__size_"])

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        if self.size == 0:
            return
        # Storage is a typed array for trivial element types and a raw
        # unsigned-char array otherwise; casting the storage address to an
        # element pointer reads both layouts and never touches the
        # uninitialized slots past __size_.
        first = self.value["__elems_"].address.cast(self.element_type.pointer())
        for index in range(self.size):
            yield f"[{index}]", first[index]

    def to_string(self) -> str:
        # Match the libstdc++ std::vector printer: "X of length N, capacity M".
        return f"{self.type_name} of length {self.size}, capacity {self.capacity}"


class InplaceVectorPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the cuda::std::inplace_vector printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::std::inplace_vector")

    def __call__(self, value: gdb.Value) -> InplaceVectorPrinter | None:
        if _is_cuda_inplace_vector(value.type):
            return InplaceVectorPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::std::inplace_vector printer with GDB."""
    gdb.printing.register_pretty_printer(
        objfile, InplaceVectorPrinterLookup(), replace=True
    )
