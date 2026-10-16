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
    """A fresh array, optionally initialized from another array argument."""

    dtype: Any
    items: int
    copy_from: int | str | None = None


def argument(source, args, keywords):
    return args[source] if isinstance(source, int) else keywords[source]


def allocate(builder, allocations, args, keywords):
    """Allocate independent results and copy inputs for in-place providers."""

    values = []
    for allocation in allocations:
        element = builder.get_storage_type(allocation.dtype)
        array_type = nb.MemRefType.get([allocation.items], element)
        with builder.alloca_insertion_point():
            value = nb.memref.alloca(array_type, [], [])
        if allocation.copy_from is not None:
            source = builder.load_var(
                argument(allocation.copy_from, args, keywords)
            )
            for item in range(allocation.items):
                index = nb.arith.constant(nb.IndexType.get(), item)
                nb.memref.store(nb.memref.load(source, [index]), value, [index])
        values.append(value)
    return tuple(values)


def assemble(builder, recipe, allocations, returned):
    """Build an array, scalar, or tuple from an explicit result recipe.

    Recipes are ``("allocation", index)``, ``("return",)`` for the scalar
    provider return, or ``("tuple", (recipe, ...))``. They describe ownership
    independently of the provider ABI: output arrays remain pointer inputs
    to a void provider even when the public call returns a tuple of arrays.
    """

    kind = recipe[0]
    if kind == "allocation":
        return allocations[recipe[1]]
    if kind == "return":
        if returned is None:
            raise AssertionError("a void provider has no scalar result")
        return returned
    if kind == "tuple":
        return tuple(
            assemble(builder, item, allocations, returned) for item in recipe[1]
        )
    raise AssertionError(f"unknown cooperative result recipe {kind!r}")
