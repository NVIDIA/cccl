# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printers for CUDA atomic and atomic_ref types."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing

_ATOMIC_NAMES = frozenset(
    {"cuda::atomic", "cuda::atomic_ref", "cuda::std::atomic", "cuda::std::atomic_ref"}
)
_ATOMIC_REF_NAMES = frozenset({"cuda::atomic_ref", "cuda::std::atomic_ref"})
_THREAD_SCOPE_NAMES = {
    0: "system",
    1: "device",
    2: "block",
    10: "thread",
}


def _is_atomic_ref(type_name: str) -> bool:
    return cccl_common.template_name(type_name) in _ATOMIC_REF_NAMES


def _complete_type_name(value_type: gdb.Type) -> str:
    """Return the complete public type name, including CUDA thread scope."""
    value_type = cccl_common.canonical_type(value_type)
    type_name = cccl_common.public_type_name(value_type)
    template_name = cccl_common.template_name(type_name)
    if template_name not in {"cuda::atomic", "cuda::atomic_ref"}:
        return type_name

    value_type_name = cccl_common.public_type_name(value_type.template_argument(0))
    scope = int(value_type.template_argument(1))
    return f"{template_name}<{value_type_name}, cuda::thread_scope_{_THREAD_SCOPE_NAMES[scope]}>"


def _is_cuda_atomic(value_type: gdb.Type) -> bool:
    value_type = cccl_common.canonical_type(value_type)
    template_name = cccl_common.template_name(cccl_common.public_type_name(value_type))
    return template_name in _ATOMIC_NAMES


def _reference_pointer(value: gdb.Value) -> gdb.Value:
    return value["__a"]["__a_value"]


def _stored_value(value: gdb.Value, type_name: str) -> gdb.Value:
    value_type = cccl_common.canonical_type(value.type)
    storage = value["__a"]
    stored = storage["__a_value"]

    if _is_atomic_ref(type_name):
        return stored.dereference()

    storage_type = str(storage.type.strip_typedefs())
    if "__atomic_small_storage<" in storage_type:
        stored = stored["__a_value"]
        value_pointer_type = value_type.template_argument(0).pointer()
        return stored.address.reinterpret_cast(value_pointer_type).dereference()
    return stored


class AtomicPrinter:
    """Expose the value represented by a CUDA atomic without executing inferior code."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        self.type_name = _complete_type_name(self.type)

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        yield "value", _stored_value(self.value, self.type_name)

    def to_string(self) -> str:
        if _is_atomic_ref(self.type_name):
            pointer = int(_reference_pointer(self.value))
            return f"{self.type_name} ptr={pointer:#x}"
        return self.type_name


class AtomicPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the atomic printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::atomic")

    def __call__(self, value: gdb.Value) -> AtomicPrinter | None:
        if _is_cuda_atomic(value.type):
            return AtomicPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the CUDA atomic printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, AtomicPrinterLookup(), replace=True)
