# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Planner-private warp reduction providers."""

import operator

from cuda.coop._core import CxxOperator, Dependency, PythonOperator
from cuda.coop._core.warp import make_warp_reduce_spec

from .._common import normalize_dtype_param
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import make_invocable_from_specialization, numba_type_to_wrapper

_BUILTIN_REDUCE_OPERATORS = {
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _materialize(
    *,
    dtype,
    operation,
    threads_in_warp,
    valid_items,
    threads_per_block,
    reduce_operator=None,
    methods=None,
):
    dtype = normalize_dtype_param(dtype)
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    core_spec = make_warp_reduce_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        operation=operation,
        reduce_operator=reduce_operator,
        valid_items=valid_items is not None,
        include_full_warp=valid_items is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(
        specialization,
        threads=threads_in_warp,
        block_threads=threads_per_block,
    )


def warp_sum(
    dtype,
    threads_in_warp=32,
    valid_items=None,
    threads_per_block=None,
):
    """Build the direct CUB warp-sum invocable selected by planning."""

    return _materialize(
        dtype=dtype,
        operation="sum",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
    )


def warp_reduce(
    dtype,
    binary_op,
    threads_in_warp=32,
    valid_items=None,
    methods=None,
    threads_per_block=None,
):
    """Build a direct CUB warp reduction with a device callback."""

    if not callable(binary_op):
        raise TypeError("binary_op must be a device callable")
    return _materialize(
        dtype=dtype,
        operation="reduce",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
        reduce_operator=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=binary_op,
            name="binary_op",
        ),
        methods=methods,
    )


def warp_reduce_builtin(
    dtype,
    binary_op,
    threads_in_warp=32,
    valid_items=None,
    threads_per_block=None,
):
    """Build a direct CUB warp reduction with a canonical C++ operator."""

    try:
        operator_cpp = _BUILTIN_REDUCE_OPERATORS[binary_op]
    except KeyError as exc:
        names = ", ".join(sorted(_BUILTIN_REDUCE_OPERATORS))
        raise ValueError(
            f"Unsupported built-in reduction {binary_op!r}; expected: {names}"
        ) from exc
    return _materialize(
        dtype=dtype,
        operation="reduce",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
        reduce_operator=CxxOperator(
            cpp=operator_cpp,
            dtype=Dependency("T"),
            name="binary_op",
        ),
    )
