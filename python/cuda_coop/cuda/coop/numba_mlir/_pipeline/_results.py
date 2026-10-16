# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Local result storage and result assembly for typed provider calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _bridges as nb


@dataclass(frozen=True)
class ArrayAllocation:
    """A fresh fixed-size array with storage independent of the input."""

    dtype: Any
    items: int


def argument(source, args, keywords):
    return args[source] if isinstance(source, int) else keywords[source]


def allocate(builder, allocations, args, keywords):
    """Create independent local result arrays before the provider call."""

    values = []
    for allocation in allocations:
        element = builder.get_storage_type(allocation.dtype)
        array_type = nb.MemRefType.get([allocation.items], element)
        with builder.alloca_insertion_point():
            value = nb.memref.alloca(array_type, [], [])
        values.append(value)
    return tuple(values)


def assemble(builder, recipe, allocations, returned):
    """Return the explicitly selected fresh array or provider scalar."""

    kind = recipe[0]
    if kind == "allocation":
        return allocations[recipe[1]]
    if kind == "return":
        if returned is None:
            raise AssertionError("a void provider has no scalar result")
        return returned
    raise AssertionError(f"unknown cooperative result recipe {kind!r}")
