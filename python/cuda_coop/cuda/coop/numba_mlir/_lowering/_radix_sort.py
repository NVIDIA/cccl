# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adapt shared block radix specializations to Numba provider callables.

Every provider uses block scratch through the leading-pointer ABI. Sorting
passes native key dtypes to CUB; ranking converts signed keys into ordered
unsigned bits before digit extraction. Group planning owns input copies,
scalar boxing, and the public result shape.
"""

import numba_cuda_mlir.numba_cuda.types as numba_types

from cuda.coop._core import SynchronizationScope
from cuda.coop._core.block.radix_rank import (
    make_block_radix_rank_specialization,
)
from cuda.coop._core.block.radix_sort import (
    make_block_radix_sort_specialization,
)

from .._compiler._operations import StorageABI, register_factory
from .._compiler._parameters import normalize_dim_param, normalize_dtype_param
from .._types import make_invocable_from_specialization
from ._core import NumbaMlirArrayInputTransform, NumbaMlirCoreAdapter


def _materialize(adapter, specialization):
    """Bind the block storage contract and expose a radix provider callable.

    Translate the shared parameter descriptions to Numba, then create the
    invocable that supplies wrapper code and storage metadata to compilation.
    The adapter records block execution and synchronization scopes.
    """

    specialization = adapter.materialize(
        specialization,
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
    return make_invocable_from_specialization(specialization)


def radix_rank_keys(
    dtype,
    threads_per_block,
    items_per_thread,
    begin_bit,
    end_bit,
    descending=False,
    with_exclusive_digit_prefix=False,
):
    """Specialize integer digit ranking with an ordered unsigned key view.

    CUB's digit extractor operates on unsigned bits. For signed keys, request
    an input transform that flips the sign bit in a temporary array. This
    uses the signed ordering transform before digit selection without changing
    the caller's keys. Unsigned keys need no transform. The native key width
    still bounds the static digit interval.

    Build the block specialization with the chosen interval, direction, and
    optional digit-prefix output, then materialize its provider callable.
    """

    dtype = normalize_dtype_param(dtype)
    if not isinstance(dtype, numba_types.Integer) or dtype not in {
        numba_types.int32,
        numba_types.uint32,
        numba_types.int64,
        numba_types.uint64,
    }:
        raise TypeError(
            "radix_rank_keys keys require int32, uint32, int64, or uint64"
        )
    cub_dtype = dtype
    transforms = None
    if dtype.signed:
        cub_dtype = (
            numba_types.uint32 if dtype.bitwidth == 32 else numba_types.uint64
        )
        expression = (
            "(static_cast<unsigned int>({value}) ^ 0x80000000u)"
            if dtype.bitwidth == 32
            else (
                "(static_cast<unsigned long long>({value}) ^ "
                "0x8000000000000000ull)"
            )
        )
        transforms = {
            "keys": NumbaMlirArrayInputTransform(
                source_dtype=dtype, cpp_expression=expression
            )
        }
    adapter = NumbaMlirCoreAdapter(input_transforms=transforms)
    specialization = make_block_radix_rank_specialization(
        key_dtype=adapter.core_dtype(cub_dtype),
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        items_per_thread=items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=dtype.bitwidth,
        descending=descending,
        with_exclusive_digit_prefix=with_exclusive_digit_prefix,
    )
    return _materialize(adapter, specialization.specialization)


def radix_sort_keys(
    dtype,
    threads_per_block,
    items_per_thread,
    descending=False,
    blocked_to_striped=False,
    value_dtype=None,
):
    """Specialize stable block sorting with optional associated values.

    Keep the native integer or floating key dtype so CUB applies its ordering
    transformation. An optional value dtype selects the paired overloads.
    Direction and blocked/striped output select the CUB method. The shared
    ``both`` bit policy emits default-range and checked explicit-range forms;
    the public rewrite supplies its bit bounds as provider operands.
    """

    dtype = normalize_dtype_param(dtype)
    if dtype not in {
        numba_types.int32,
        numba_types.uint32,
        numba_types.int64,
        numba_types.uint64,
        numba_types.float32,
        numba_types.float64,
    }:
        raise TypeError(
            "radix_sort keys require 32- or 64-bit integers or floats"
        )
    adapter = NumbaMlirCoreAdapter()
    specialization = make_block_radix_sort_specialization(
        key_dtype=adapter.core_dtype(dtype),
        value_dtype=None
        if value_dtype is None
        else adapter.core_dtype(normalize_dtype_param(value_dtype)),
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        items_per_thread=items_per_thread,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        bit_policy="both",
    )
    return _materialize(adapter, specialization.specialization)


def radix_sort_pairs(
    dtype,
    threads_per_block,
    items_per_thread,
    value_dtype,
    descending=False,
    blocked_to_striped=False,
):
    """Delegate the pair-sort factory to the shared sort provider builder."""

    return radix_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        descending,
        blocked_to_striped,
        value_dtype,
    )


for _factory in (radix_rank_keys, radix_sort_keys, radix_sort_pairs):
    register_factory(
        _factory,
        operation=_factory.__name__,
        namespace="block",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
del _factory

__all__: tuple[str, ...] = ()
