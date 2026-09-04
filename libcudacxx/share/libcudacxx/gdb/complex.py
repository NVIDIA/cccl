# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::complex and cuda::complex."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing

_COMPLEX_NAMES = frozenset({"cuda::complex", "cuda::std::complex"})


def _is_cuda_complex(value_type: gdb.Type) -> bool:
    value_type = cccl_common.canonical_type(value_type)
    template_name = cccl_common.template_name(cccl_common.public_type_name(value_type))
    return template_name in _COMPLEX_NAMES


class ComplexPrinter:
    """Expose cuda::std::complex and cuda::complex parts to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        self.type_name = cccl_common.public_type_name(self.type)

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        try:
            real = self.value["__re_"]
            imag = self.value["__im_"]
        except gdb.error:
            packed = self.value["__repr_"]
            real = packed["x"]
            imag = packed["y"]
        yield "real", real
        yield "imag", imag

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
