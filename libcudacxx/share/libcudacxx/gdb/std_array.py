# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::array."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import memory_resource

import gdb
import gdb.printing


def _template_name(value_type: gdb.Type) -> str:
    return str(value_type).split("<", 1)[0]


def _is_cuda_array(value_type: gdb.Type) -> bool:
    value_type = value_type.strip_typedefs().unqualified()
    template_name = _template_name(value_type)
    return (
        template_name.startswith("cuda::std::")
        and template_name.rsplit("::", 1)[-1] == "array"
    )


class ArrayPrinter:
    """Expose cuda::std::array metadata and elements to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        self.value = value
        self.type = value.type.strip_typedefs().unqualified()
        self.type_name = memory_resource.public_type_name(self.type)
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
