# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::span."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import memory_resource

import gdb
import gdb.printing

# cuda::std::dynamic_extent, spelled out because GDB reports the extent as the
# raw template argument value.
_DYNAMIC_EXTENT = 2**64 - 1


def _template_name(value_type: gdb.Type) -> str:
    return str(value_type).split("<", 1)[0]


def _is_cuda_span(value_type: gdb.Type) -> bool:
    # strip_typedefs resolves aliases that can hide accessibility properties.
    value_type = value_type.strip_typedefs().unqualified()
    template_name = _template_name(value_type)
    return (
        template_name.startswith("cuda::std::")
        and template_name.rsplit("::", 1)[-1] == "span"
    )


def _public_span_name(value_type: gdb.Type) -> str:
    name = memory_resource.public_type_name(value_type)
    return name.replace(f", {_DYNAMIC_EXTENT}>", ", dynamic_extent>")


class SpanPrinter:
    """Expose cuda::std::span metadata and elements to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        self.value = value
        self.type = value.type.strip_typedefs().unqualified()
        self.type_name = _public_span_name(self.type)
        self.data = value["__data_"]
        # The dynamic-extent specialization stores its size; the static-extent
        # one carries it in the type.
        if any(field.name == "__size_" for field in self.type.fields()):
            self.size = int(value["__size_"])
        else:
            self.size = int(self.type.template_argument(1))

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        if self.size == 0 or int(self.data) == 0:
            return
        for index in range(self.size):
            yield f"[{index}]", (self.data + index).dereference()

    def to_string(self) -> str:
        return self.type_name


class SpanPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the cuda::std::span printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::std::span")

    def __call__(self, value: gdb.Value) -> SpanPrinter | None:
        if _is_cuda_span(value.type):
            return SpanPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::std::span printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, SpanPrinterLookup(), replace=True)
