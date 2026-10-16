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
    """A fresh fixed-size array, optionally initialized from a call operand.

    ``copy_from`` is a positional index or keyword name. An array is copied;
    a scalar initializes a one-item allocation. This lets an in-place
    provider preserve the public input. Separate-input/output providers
    leave it unset. Each allocation has its own storage.
    """

    dtype: Any
    items: int
    copy_from: int | str | None = None


def argument(source, args, keywords):
    return args[source] if isinstance(source, int) else keywords[source]


def allocate(builder, allocations, args, keywords):
    """Create the local arrays before the call, retaining distinct storage."""

    values = []
    for allocation in allocations:
        element = builder.get_storage_type(allocation.dtype)
        array_type = nb.MemRefType.get([allocation.items], element)
        with builder.alloca_insertion_point():
            value = nb.memref.alloca(array_type, [], [])
        if allocation.copy_from is not None:
            source_var = argument(allocation.copy_from, args, keywords)
            source = builder.load_var(source_var)
            source_type = builder.get_numba_type(source_var.name)
            if isinstance(source_type, nb.types.Array):
                for item in range(allocation.items):
                    index = nb.arith.constant(nb.IndexType.get(), item)
                    loaded = nb.memref.load(source, [index])
                    nb.memref.store(loaded, value, [index])
            else:
                if allocation.items != 1:
                    raise AssertionError("a scalar initializes one item")
                scalar = nb.cast_value(
                    builder, source, source_type, allocation.dtype
                )
                index = nb.arith.constant(nb.IndexType.get(), 0)
                nb.memref.store(
                    builder.as_storage(allocation.dtype, scalar),
                    value,
                    [index],
                )
        values.append(value)
    return tuple(values)


def assemble(builder, recipe, allocations, returned):
    """Build an array, scalar, or tuple from an explicit result recipe.

    Recipes are ``("allocation", index)``, ``("return",)`` for the scalar
    provider return, ``("scalar", index, dtype)`` for a one-item allocation,
    or ``("tuple", (recipe, ...))``. They describe ownership
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
    if kind == "scalar":
        index = nb.arith.constant(nb.IndexType.get(), 0)
        value = nb.memref.load(allocations[recipe[1]], [index])
        return builder.from_storage(recipe[2], value)
    if kind == "tuple":
        return tuple(
            assemble(builder, item, allocations, returned) for item in recipe[1]
        )
    raise AssertionError(f"unknown cooperative result recipe {kind!r}")
