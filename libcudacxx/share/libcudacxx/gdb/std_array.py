# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::array."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing


def _is_cuda_array(value_type: gdb.Type) -> bool:
    template_name = cccl_common.template_name(cccl_common.canonical_type(value_type))
    return (
        template_name.startswith("cuda::std::")
        and template_name.rsplit("::", 1)[-1] == "array"
    )


class ArrayPrinter:
    """Expose cuda::std::array metadata and elements to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        self.type_name = cccl_common.public_type_name(self.type)
        self.size = int(self.type.template_argument(1))

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        elems = self.value["__elems_"]
        for index in range(self.size):
            yield f"[{index}]", elems[index]

    def to_string(self) -> str:
        return self.type_name


class ArrayPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the cuda::std::array printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::std::array")

    def __call__(self, value: gdb.Value) -> ArrayPrinter | None:
        if _is_cuda_array(value.type):
            return ArrayPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::std::array printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, ArrayPrinterLookup(), replace=True)
