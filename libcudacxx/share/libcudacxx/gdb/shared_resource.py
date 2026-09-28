# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::mr::shared_resource."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing


def _is_shared_resource(value_type: gdb.Type) -> bool:
    value_type = cccl_common.canonical_type(value_type)
    type_name = cccl_common.public_type_name(value_type)
    return (
        type_name.startswith("cuda::mr::")
        and cccl_common.template_name(value_type).rsplit("::", 1)[-1]
        == "shared_resource"
    )


class SharedResourcePrinter:
    """Print the ownership state and the owned resource of a shared resource."""

    def __init__(self, value: gdb.Value) -> None:
        self.value = cccl_common.strip_reference_value(value)

    def _control_block(self) -> gdb.Value | None:
        """Return the control block, or None if this handle is empty."""
        # shared_resource holds a __shared_block_ptr, and both the wrapper and
        # the pointer spell their member __block_.
        control_block = self.value["__block_"]["__block_"]
        if int(control_block) == 0:
            return None
        return control_block.dereference()

    def to_string(self) -> str:
        type_name = cccl_common.canonical_type_name(self.value.type)
        block = self._control_block()
        if block is None:
            return f"{type_name} use_count=0, resource=nullptr"

        # cuda::std::atomic<int> keeps its value in __a.__a_value.
        use_count = int(block["__ref_count"]["__a"]["__a_value"])
        # The address of a readable resource belongs to the resource child, the
        # way std::shared_ptr keeps strong=/weak= in its summary and the pointer
        # in its child. Only report it here when there is no child to carry it:
        # a control block reached through a live pointer always has an address,
        # so the branch below guards against an unreadable frame rather than
        # against any state the scenario can reach.
        if block["__payload"].address is None:
            return f"{type_name} use_count={use_count}, resource=<invalid address>"
        return f"{type_name} use_count={use_count}"

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        # Present the owned resource the way std::shared_ptr presents its
        # pointer: one step away, so that expanding a handle does not print the
        # implementation details of the resource itself.
        block = self._control_block()
        if block is None:
            return
        address = block["__payload"].address
        if address is not None:
            yield "resource", address


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
