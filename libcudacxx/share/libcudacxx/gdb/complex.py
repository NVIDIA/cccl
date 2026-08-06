# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::complex and cuda::complex."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import memory_resource

import gdb
import gdb.printing

_COMPLEX_NAMES = frozenset({"cuda::complex", "cuda::std::complex"})


def _template_name(type_name: str) -> str:
    return type_name.split("<", 1)[0]


def _is_cuda_complex(value_type: gdb.Type) -> bool:
    value_type = value_type.strip_typedefs().unqualified()
    template_name = _template_name(memory_resource.public_type_name(value_type))
    return template_name in _COMPLEX_NAMES


class ComplexPrinter:
    """Expose cuda::std::complex and cuda::complex parts to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        self.value = value
        self.type = value.type.strip_typedefs().unqualified()
        self.type_name = memory_resource.public_type_name(self.type)

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        yield "real", self.value["__re_"]
        yield "imag", self.value["__im_"]

    def to_string(self) -> str:
        return self.type_name


class ComplexPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the complex printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::complex")

    def __call__(self, value: gdb.Value) -> ComplexPrinter | None:
        if _is_cuda_complex(value.type):
            return ComplexPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda complex printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, ComplexPrinterLookup(), replace=True)
