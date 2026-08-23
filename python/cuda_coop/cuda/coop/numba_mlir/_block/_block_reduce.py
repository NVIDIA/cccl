# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Planner-private block reduction providers."""

import operator

from cuda.coop._core import CxxOperator, Dependency, PythonOperator
from cuda.coop._core.block import make_block_reduce_spec

from .._common import (
    CUB_BLOCK_REDUCE_ALGOS,
    normalize_dim_param,
    normalize_dtype_param,
)
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


def _reduce_algorithm(algorithm) -> str:
    if isinstance(algorithm, bool):
        raise TypeError("block reduce algorithm must not be bool")
    if not isinstance(algorithm, str):
        raise TypeError("block reduce algorithm must be a string")
    if algorithm.startswith("::cub::BlockReduceAlgorithm::"):
        return algorithm
    try:
        return CUB_BLOCK_REDUCE_ALGOS[algorithm.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(CUB_BLOCK_REDUCE_ALGOS))
        raise ValueError(
            f"Unsupported block reduce algorithm {algorithm!r}; "
            f"expected one of: {allowed}"
        ) from exc


def _materialize(
    *,
    dtype,
    threads_per_block,
    items_per_thread,
    operation,
    algorithm,
    reduce_operator=None,
    num_valid=None,
    methods=None,
):
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    if num_valid is not None and items_per_thread != 1:
        raise ValueError("num_valid is not supported for array inputs")
    dtype = normalize_dtype_param(dtype)
    core_spec = make_block_reduce_spec(
        dtype=dtype,
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        items_per_thread=items_per_thread,
        operation=operation,
        algorithm=_reduce_algorithm(algorithm),
        value_kind="scalar" if items_per_thread == 1 else "array",
        reduce_operator=reduce_operator,
        valid_items=num_valid is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(specialization)


def sum(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="warp_reductions",
    num_valid=None,
    methods=None,
):
    """Build the direct CUB block-sum invocable selected by planning."""

    return _materialize(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        operation="sum",
        algorithm=algorithm,
        num_valid=num_valid,
        methods=methods,
    )


def reduce(
    dtype,
    threads_per_block=None,
    binary_op=None,
    items_per_thread=1,
    algorithm="warp_reductions",
    num_valid=None,
    methods=None,
):
    """Build a direct CUB block reduction with a device callback."""

    if not callable(binary_op):
        raise TypeError("binary_op must be a device callable")
    return _materialize(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        operation="reduce",
        algorithm=algorithm,
        reduce_operator=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=binary_op,
            name="binary_op",
        ),
        num_valid=num_valid,
        methods=methods,
    )


def block_reduce_builtin(
    dtype,
    threads_per_block,
    binary_op,
    items_per_thread=1,
    algorithm="warp_reductions",
    num_valid=None,
):
    """Build a direct CUB reduction with a canonical C++ operator."""

    try:
        operator_cpp = _BUILTIN_REDUCE_OPERATORS[binary_op]
    except KeyError as exc:
        names = ", ".join(sorted(_BUILTIN_REDUCE_OPERATORS))
        raise ValueError(
            f"Unsupported built-in reduction {binary_op!r}; expected: {names}"
        ) from exc
    return _materialize(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        operation="reduce",
        algorithm=algorithm,
        reduce_operator=CxxOperator(
            cpp=operator_cpp,
            dtype=Dependency("T"),
            name="binary_op",
        ),
        num_valid=num_valid,
    )
