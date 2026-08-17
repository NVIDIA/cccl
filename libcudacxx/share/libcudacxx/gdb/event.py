# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printers for cuda event types."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing

_EVENT_NAMES = frozenset({"cuda::event", "cuda::event_ref", "cuda::timed_event"})


def _event_type_name(value_type: gdb.Type) -> str | None:
    type_name = cccl_common.canonical_type_name(value_type)
    if type_name in _EVENT_NAMES:
        return type_name
    return None


def _event_handle(value: gdb.Value) -> gdb.Value:
    value_type = cccl_common.canonical_type(value.type)
    value = value.cast(value_type)
    for field in value_type.fields():
        if field.name == "__event_":
            return value[field]
        if field.is_base_class and _event_type_name(field.type) is not None:
            try:
                return _event_handle(value.cast(field.type))
            except gdb.error:
                pass
    raise gdb.error("cuda event handle not found")


class EventPrinter:
    """Expose the native handle stored by a cuda event type."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        self.type_name = cccl_common.public_type_name(self.type)

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        yield "handle", _event_handle(self.value)

    def to_string(self) -> str:
        return self.type_name


class EventPrinterLookup(gdb.printing.PrettyPrinter):
    """Select printers for cuda event types by public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::event")

    def __call__(self, value: gdb.Value) -> EventPrinter | None:
        if _event_type_name(value.type) is not None:
            return EventPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register cuda event printers with GDB."""
    gdb.printing.register_pretty_printer(objfile, EventPrinterLookup(), replace=True)
