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
    template_name = cccl_common.template_name(cccl_common.public_type_name(value_type))
    return (
        template_name.startswith("cuda::std::")
        and template_name.rsplit("::", 1)[-1] == "expected"
    )


def _display_type_name(value_type: gdb.Type) -> str:
    """Return the public name of a type with aliases resolved and qualifiers kept.

    strip_typedefs() does not look through a reference, so resolve the referenced
    type separately (e.g. const expected_alias & -> const cuda::std::expected<...> &).
    """
    suffix = ""
    if value_type.code == gdb.TYPE_CODE_REF:
        suffix = " &"
        value_type = value_type.target()
    elif value_type.code == gdb.TYPE_CODE_RVALUE_REF:
        suffix = " &&"
        value_type = value_type.target()
    prefix = "const " if value_type == value_type.unqualified().const() else ""
    name = cccl_common.public_type_name(value_type.strip_typedefs().unqualified())
    return f"{prefix}{name}{suffix}"


class ExpectedPrinter:
    """Expose the value or the error of a cuda::std::expected to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        self.type_name = _display_type_name(value.type)
        value = cccl_common.strip_reference_value(value)
        # __has_val_ and __union_ are members of the __expected_destruct base class.
        self.has_value = bool(value["__has_val_"])
        storage = value["__union_"]
        self.child: tuple[str, gdb.Value] | None = None
        if not self.has_value:
            self.child = ("error", storage["__unex_"])
        elif cccl_common.canonical_type(value.type).template_argument(0).code != (
            gdb.TYPE_CODE_VOID
        ):
            self.child = ("value", storage["__val_"])

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        if self.child is not None:
            yield self.child

    def to_string(self) -> str:
        if self.has_value and self.child is None:
            # expected<void, E> holding a value has nothing else to show.
            return f"{self.type_name} has value"
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
