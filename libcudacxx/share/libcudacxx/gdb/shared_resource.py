# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::mr::shared_resource."""

from __future__ import annotations

from types import ModuleType

import memory_resource

import gdb
import gdb.printing


def _template_name(value_type: gdb.Type) -> str:
    return str(value_type).split("<", 1)[0]


def _is_shared_resource(value_type: gdb.Type) -> bool:
    # strip_typedefs resolves aliases that can hide the public class name.
    value_type = value_type.strip_typedefs().unqualified()
    type_name = memory_resource.public_type_name(value_type)
    template_name = _template_name(value_type)
    return (
        type_name.startswith("cuda::mr::")
        and template_name.rsplit("::", 1)[-1] == "shared_resource"
    )


class SharedResourcePrinter:
    """Summarize the ownership state of a CUDA shared resource."""

    def __init__(self, value: gdb.Value) -> None:
        self.value = value

    def to_string(self) -> str:
        value_type = self.value.type.strip_typedefs().unqualified()
        type_name = memory_resource.public_type_name(value_type)
        # shared_resource holds a __shared_block_ptr, and both the wrapper and
        # the pointer spell their member __block_.
        control_block = self.value["__block_"]["__block_"]
        if int(control_block) == 0:
            return f"{type_name} empty"

        block = control_block.dereference()
        # cuda::std::atomic<int> keeps its value in __a.__a_value.
        use_count = int(block["__ref_count"]["__a"]["__a_value"])
        resource = int(block["__payload"].address)
        return f"{type_name} use_count={use_count}, resource={resource:#x}"


class SharedResourcePrinterLookup(gdb.printing.PrettyPrinter):
    """Select the cuda::mr::shared_resource printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::mr::shared_resource")

    def __call__(self, value: gdb.Value) -> SharedResourcePrinter | None:
        if _is_shared_resource(value.type):
            return SharedResourcePrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda::mr::shared_resource printer with GDB."""
    gdb.printing.register_pretty_printer(
        objfile, SharedResourcePrinterLookup(), replace=True
    )
