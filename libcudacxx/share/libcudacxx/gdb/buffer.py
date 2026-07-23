# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::buffer."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import memory_resource

import gdb
import gdb.printing

# GDB sees cudaMemcpyKind through cudaMemcpy's declaration, but its expression
# parser does not necessarily import the enum constants. This is the value of
# cudaMemcpyDefault from the CUDA Runtime API.
_CUDA_MEMCPY_DEFAULT = 4


def _template_name(value_type: gdb.Type) -> str:
    return str(value_type).split("<", 1)[0]


def _is_cuda_buffer(value_type: gdb.Type) -> bool:
    # strip_typedefs resolves aliases that can hide accessibility properties.
    value_type = value_type.strip_typedefs().unqualified()
    template_name = _template_name(value_type)
    return (
        template_name.startswith("cuda::")
        and template_name.rsplit("::", 1)[-1] == "buffer"
    )


class BufferPrinter:
    """Expose cuda::buffer metadata and elements to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        self.value = value
        self.type = value.type.strip_typedefs().unqualified()
        self.type_name = memory_resource.public_type_name(self.type)
        self.value_type = self.type.template_argument(0)

        storage = value["__buf_"]
        self.memory_resource = storage["__mr_"]
        self.stream = int(storage["__stream_"]["__stream"])
        self.size = int(storage["__count_"])
        self.alignment = int(storage["__alignment_"])
        raw_address = int(storage["__buf_"])
        self.data_address = (raw_address + self.alignment - 1) & ~(self.alignment - 1)

        host_accessible = "host_accessible" in self.type_name
        device_accessible = "device_accessible" in self.type_name
        if host_accessible and device_accessible:
            self.accessibility = "host/device"
        elif device_accessible:
            self.accessibility = "device"
        elif host_accessible:
            self.accessibility = "host"
        else:
            self.accessibility = "unknown"

        self.host_copy: gdb.Value | None = None
        self._copy_to_host()

    def __del__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Release state staged in the inferior for synthetic children."""
        if self.host_copy is None:
            return
        try:
            gdb.parse_and_eval(f"(void)free((void*){int(self.host_copy):#x})")
        except gdb.error:
            pass
        self.host_copy = None

    def _copy_to_host(self) -> None:
        if self.size == 0:
            return

        byte_count = self.size * self.value_type.sizeof
        self.host_copy = gdb.parse_and_eval(f"(void*)malloc({byte_count})")
        host_address = int(self.host_copy)
        status = gdb.parse_and_eval(
            "(int)cudaMemcpy((void*)"
            f"{host_address:#x}, (const void*){self.data_address:#x}, {byte_count}, "
            f"{_CUDA_MEMCPY_DEFAULT})"
        )
        if int(status) != 0:
            self.clear()

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        if self.host_copy is None:
            return

        pointer = self.host_copy.cast(self.value_type.pointer())
        for index in range(self.size):
            yield f"[{index}]", (pointer + index).dereference()

    def to_string(self) -> str:
        resource = memory_resource.memory_resource_description(self.memory_resource)
        return (
            f"{self.type_name} mr={resource}, stream={self.stream:#x}, "
            f"size={self.size}, align={self.alignment}, "
            f"data={self.data_address:#x} ({self.accessibility})"
        )


class BufferPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the cuda::buffer printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::buffer")

    def __call__(self, value: gdb.Value) -> BufferPrinter | None:
        if _is_cuda_buffer(value.type):
            return BufferPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::buffer printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, BufferPrinterLookup(), replace=True)
