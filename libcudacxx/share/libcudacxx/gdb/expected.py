# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::expected."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing


def _is_cuda_expected(value_type: gdb.Type) -> bool:
    value_type = cccl_common.canonical_type(value_type)
    return (
        cccl_common.template_name(cccl_common.public_type_name(value_type))
        == "cuda::std::expected"
    )


class ExpectedPrinter:
    """Expose the engaged value or the error of a cuda::std::expected to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        self.type_name = cccl_common.public_type_name(self.type)
        self.has_val = bool(self.value["__has_val_"])

    def children(self) -> Iterator[tuple[str, gdb.Value | str]]:
        union = self.value["__union_"]
        if self.has_val:
            try:
                yield "value", union["__val_"]
            except gdb.error:
                # expected<void, E> has no __val_ in the engaged state.
                yield "value", "void"
        else:
            yield "error", union["__unex_"]

    def to_string(self) -> str:
        return self.type_name


class ExpectedPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the expected printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::std::expected")

    def __call__(self, value: gdb.Value) -> ExpectedPrinter | None:
        if _is_cuda_expected(value.type):
            return ExpectedPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda expected printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, ExpectedPrinterLookup(), replace=True)
