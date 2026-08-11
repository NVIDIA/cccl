# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printers for cuda memory pool types."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import NamedTuple

import cccl_common

import gdb
import gdb.printing

_SHARED_POOL_NAMES = frozenset(
    {
        "cuda::shared_device_memory_pool",
        "cuda::shared_managed_memory_pool",
        "cuda::shared_pinned_memory_pool",
    }
)
_POOL_NAMES = (
    frozenset(
        {
            "cuda::device_memory_pool",
            "cuda::device_memory_pool_ref",
            "cuda::managed_memory_pool",
            "cuda::managed_memory_pool_ref",
            "cuda::pinned_memory_pool",
            "cuda::pinned_memory_pool_ref",
        }
    )
    | _SHARED_POOL_NAMES
)
# These are the public values of the CUmemPool_attribute enumerators. The
# debugger expression parser does not necessarily expose the enumerators.
# Every attribute listed here is a cuuint64_t and has existed since CUDA 11.2,
# so the rendered output does not vary across supported toolkits.
_POOL_ATTRIBUTES = (
    ("release_threshold", 4),
    ("reserved_mem_current", 5),
    ("reserved_mem_high", 6),
    ("used_mem_current", 7),
    ("used_mem_high", 8),
)
_CU_MEM_POOL_GET_ATTRIBUTE = "((int (*)(void*, int, void*))cuMemPoolGetAttribute)"
_ATTRIBUTE_TYPE = "unsigned long long"


class PoolInfo(NamedTuple):
    handle: int
    use_count: int | None
    attributes: tuple[tuple[str, int], ...]


def _pool_type_name(value_type: gdb.Type) -> str | None:
    type_name = cccl_common.canonical_type_name(value_type)
    if type_name in _POOL_NAMES:
        return type_name
    return None


def _find_member(value: gdb.Value, name: str) -> gdb.Value | None:
    """Return a member of value or of one of its base classes.

    The pool handle lives on cuda::__memory_pool_base, two levels below the
    shared pool types, so the search has to walk the whole base chain.
    """
    value_type = cccl_common.canonical_type(value.type)
    value = value.cast(value_type)
    for field in value_type.fields():
        if field.name == name:
            return value[field]
    for field in value_type.fields():
        if not field.is_base_class:
            continue
        try:
            member = _find_member(value.cast(field.type), name)
        except gdb.error:
            continue
        if member is not None:
            return member
    return None


def _pool_use_count(value: gdb.Value) -> int | None:
    """Return the number of owners sharing the pool, or None if unreadable."""
    reference = _find_member(value, "__ref_")
    if reference is None:
        return None
    try:
        block = reference["__block_"]
        # A no_init shared pool never allocates a control block.
        if int(block) == 0:
            return 0
        return int(block["__ref_count"]["__a"]["__a_value"])
    except (gdb.error, TypeError, ValueError):
        return None


def _query_pool_attributes(handle: int) -> tuple[tuple[str, int], ...]:
    """Read every pool attribute through a single scratch buffer.

    Inferior calls are expensive, so the buffer is allocated once and reused
    rather than staged per attribute.
    """
    attributes: list[tuple[str, int]] = []
    output: gdb.Value | None = None
    try:
        output = gdb.parse_and_eval(
            f"({_ATTRIBUTE_TYPE}*)calloc(1, sizeof({_ATTRIBUTE_TYPE}))"
        )
        address = int(output)
        if address == 0:
            return ()
        for name, attribute in _POOL_ATTRIBUTES:
            try:
                status = gdb.parse_and_eval(
                    f"(int){_CU_MEM_POOL_GET_ATTRIBUTE}((void*){handle:#x}, "
                    f"{attribute}, (void*){address:#x})"
                )
                if int(status) == 0:
                    attributes.append((name, int(output.dereference())))
            except (gdb.error, TypeError, ValueError):
                continue
    except (gdb.error, TypeError, ValueError):
        pass
    finally:
        if output is not None:
            try:
                gdb.parse_and_eval(f"(void)free((void*){int(output):#x})")
            except (gdb.error, TypeError, ValueError):
                pass
    return tuple(attributes)


def _pool_info(value: gdb.Value, shared: bool) -> PoolInfo:
    handle_value = _find_member(value, "__pool_")
    if handle_value is None:
        raise gdb.error("cuda memory pool handle not found")

    handle = int(handle_value)
    use_count = _pool_use_count(value) if shared else None
    if handle == 0:
        return PoolInfo(handle, use_count, ())
    return PoolInfo(handle, use_count, _query_pool_attributes(handle))


class MemoryPoolPrinter:
    """Expose CUDA memory pool metadata to GDB."""

    def __init__(self, value: gdb.Value, type_name: str) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type_name = type_name
        self.info = _pool_info(value, type_name in _SHARED_POOL_NAMES)

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        for name, attribute_value in self.info.attributes:
            try:
                yield (
                    name,
                    gdb.Value(attribute_value).cast(gdb.lookup_type(_ATTRIBUTE_TYPE)),
                )
            except (gdb.error, TypeError, ValueError):
                continue

    def to_string(self) -> str:
        description = f"{self.type_name} handle={self.info.handle:#x}"
        if self.info.use_count is not None:
            description += f", use_count={self.info.use_count}"
        return description


class MemoryPoolPrinterLookup(gdb.printing.PrettyPrinter):
    """Select printers for cuda memory pool types by public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::device_memory_pool")

    def __call__(self, value: gdb.Value) -> MemoryPoolPrinter | None:
        type_name = _pool_type_name(value.type)
        if type_name is None:
            return None
        try:
            return MemoryPoolPrinter(value, type_name)
        except (gdb.error, TypeError, ValueError):
            return None


def register(objfile: ModuleType) -> None:
    """Register CUDA memory pool formatters with GDB."""
    gdb.printing.register_pretty_printer(
        objfile, MemoryPoolPrinterLookup(), replace=True
    )
