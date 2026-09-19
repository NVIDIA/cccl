# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Materialize the CUB WarpReduceBatched provider."""

from __future__ import annotations

from enum import Enum

from cuda.coop._core import (
    CxxOperator,
    Dependency,
    PythonOperator,
    SynchronizationScope,
)
from cuda.coop._core.warp.reduce_batched import make_warp_reduce_batched_spec

from .._compiler._operations import StorageABI, register_factory
from .._compiler._parameters import _validate_common_numeric_dtype, normalize_dim_param
from .._semantic import _normalize_numba_callable
from .._types import make_invocable_from_specialization, numba_type_to_wrapper
from ._core import NumbaMlirCoreAdapter
from ._reduce import normalize_reduce_operation, validate_reduce_operator_dtype

_BUILTINS = {
    "sum": "::cuda::std::plus<T>",
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}


def reduction_operator(binary_op, dtype, *, is_common_root=False):
    if (
        is_common_root
        and binary_op is not None
        and (not isinstance(binary_op, str) or isinstance(binary_op, Enum))
    ):
        raise TypeError("cuda.coop.reduce_batched binary_op must be a string")
    try:
        canonical = normalize_reduce_operation(binary_op)
    except NotImplementedError:
        return PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=_normalize_numba_callable(binary_op),
            name="binary_op",
        )
    validate_reduce_operator_dtype(canonical, dtype)
    return CxxOperator(_BUILTINS[canonical], Dependency("T"), name="binary_op")


def reduce_batched(
    dtype,
    threads_per_block,
    threads_in_warp,
    batches,
    binary_op=None,
    output_layout="striped",
):
    """Build one independent-batch reduction for each participating warp."""

    block_dim = normalize_dim_param(threads_per_block)
    dtype = _validate_common_numeric_dtype(dtype, operation="reduce_batched")
    reduce_operator = reduction_operator(binary_op, dtype)
    adapter = NumbaMlirCoreAdapter()
    spec = make_warp_reduce_batched_spec(
        dtype=adapter.core_dtype(dtype),
        batches=batches,
        threads_in_warp=threads_in_warp,
        reduce_operator=reduce_operator,
        output_layout=output_layout,
    )
    specialization = adapter.materialize(
        spec.specialization,
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.WARP,
        synchronization_scope=SynchronizationScope.WARP,
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(
        specialization, threads=threads_in_warp, block_threads=block_dim
    )


register_factory(
    reduce_batched,
    operation="reduce_batched",
    namespace="warp",
    storage_abi=StorageABI.LEADING_POINTER,
    execution_scope=SynchronizationScope.WARP,
    synchronization_scope=SynchronizationScope.WARP,
)

__all__: tuple[str, ...] = ()
