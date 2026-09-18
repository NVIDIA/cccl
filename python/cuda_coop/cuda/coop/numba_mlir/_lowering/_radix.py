# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUB providers for block radix sorting and stable digit ranking."""

from numba_cuda_mlir import types

from cuda.coop._core import SynchronizationScope
from cuda.coop._core.block.radix_rank import make_block_radix_rank_spec
from cuda.coop._core.block.radix_sort import make_block_radix_sort_spec

from .._compiler._operations import StorageABI, register_factory
from .._compiler._parameters import normalize_dim_param, normalize_dtype_param
from .._types import make_invocable_from_specialization
from ._core import NumbaMlirArrayInputTransform, NumbaMlirCoreAdapter


def _materialize(adapter, spec):
    specialization = adapter.materialize(
        spec,
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
    return make_invocable_from_specialization(specialization)


def radix_rank(
    dtype,
    threads_per_block,
    items_per_thread,
    begin_bit,
    end_bit,
    descending=False,
    with_exclusive_digit_prefix=False,
):
    dtype = normalize_dtype_param(dtype)
    if dtype not in {types.int32, types.uint32, types.int64, types.uint64}:
        raise TypeError("radix_rank keys require int32, uint32, int64, or uint64")
    cub_dtype = dtype
    transforms = None
    if dtype.signed:
        cub_dtype = types.uint32 if dtype.bitwidth == 32 else types.uint64
        expression = (
            "(static_cast<unsigned int>({value}) ^ 0x80000000u)"
            if dtype.bitwidth == 32
            else "(static_cast<unsigned long long>({value}) ^ 0x8000000000000000ull)"
        )
        transforms = {
            "keys": NumbaMlirArrayInputTransform(
                source_dtype=dtype, cpp_expression=expression
            )
        }
    adapter = NumbaMlirCoreAdapter(input_transforms=transforms)
    spec = make_block_radix_rank_spec(
        key_dtype=adapter.core_dtype(cub_dtype),
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        items_per_thread=items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=dtype.bitwidth,
        descending=descending,
        with_exclusive_digit_prefix=with_exclusive_digit_prefix,
    )
    return _materialize(adapter, spec.specialization)


def radix_sort_keys(
    dtype,
    threads_per_block,
    items_per_thread,
    descending=False,
    blocked_to_striped=False,
    value_dtype=None,
):
    dtype = normalize_dtype_param(dtype)
    if dtype not in {
        types.int32,
        types.uint32,
        types.int64,
        types.uint64,
        types.float32,
        types.float64,
    }:
        raise TypeError("radix_sort keys require 32- or 64-bit integers or floats")
    adapter = NumbaMlirCoreAdapter()
    spec = make_block_radix_sort_spec(
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
    return _materialize(adapter, spec.specialization)


def radix_sort_pairs(
    dtype,
    threads_per_block,
    items_per_thread,
    value_dtype,
    descending=False,
    blocked_to_striped=False,
):
    return radix_sort_keys(
        dtype,
        threads_per_block,
        items_per_thread,
        descending,
        blocked_to_striped,
        value_dtype,
    )


for _factory in (radix_rank, radix_sort_keys, radix_sort_pairs):
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
