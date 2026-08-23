# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from .. import _require_runtime

_require_runtime()


from cuda.coop._core import CxxOperator, Dependency, PythonOperator
from cuda.coop._core.block import make_block_reduce_spec

from .._common import (
    CUB_BLOCK_REDUCE_ALGOS,
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import (
    make_invocable_from_specialization,
    numba_type_to_wrapper,
)

_BUILTIN_REDUCE_OPERATORS = {
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}


def _reduce(
    dtype,
    threads_per_block,
    items_per_thread,
    binary_op,
    algorithm,
    num_valid=None,
    methods=None,
):
    if algorithm not in CUB_BLOCK_REDUCE_ALGOS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")
    if num_valid is not None and items_per_thread != 1:
        raise ValueError("num_valid is not supported for array inputs")

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)

    reduce_operator = (
        None
        if binary_op is None
        else PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=binary_op,
            name="binary_op",
        )
    )
    core_spec = make_block_reduce_spec(
        dtype=dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        operation="sum" if binary_op is None else "reduce",
        algorithm=CUB_BLOCK_REDUCE_ALGOS[algorithm],
        value_kind="scalar" if items_per_thread == 1 else "array",
        reduce_operator=reduce_operator,
        valid_items=num_valid is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(specialization)


def block_reduce_builtin(
    dtype,
    threads_per_block,
    binary_op,
    items_per_thread=1,
    algorithm="warp_reductions",
    num_valid=None,
):
    """Build a CUB reduction with the canonical shared-core C++ operator."""

    if algorithm not in CUB_BLOCK_REDUCE_ALGOS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")
    if num_valid is not None and items_per_thread != 1:
        raise ValueError("num_valid is not supported for array inputs")
    try:
        operator_cpp = _BUILTIN_REDUCE_OPERATORS[binary_op]
    except KeyError as exc:
        names = ", ".join(sorted(_BUILTIN_REDUCE_OPERATORS))
        raise ValueError(
            f"Unsupported built-in reduction: {binary_op!r}; {names}"
        ) from exc

    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    core_spec = make_block_reduce_spec(
        dtype=dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        operation="reduce",
        algorithm=CUB_BLOCK_REDUCE_ALGOS[algorithm],
        value_kind="scalar" if items_per_thread == 1 else "array",
        reduce_operator=CxxOperator(
            cpp=operator_cpp,
            dtype=Dependency("T"),
            name="binary_op",
        ),
        valid_items=num_valid is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)


def reduce(
    dtype,
    threads_per_block=None,
    binary_op=None,
    items_per_thread=1,
    algorithm="warp_reductions",
    num_valid=None,
    methods=None,
    dim=None,
):
    """Build a block-wide reduction invocable using ``binary_op``.

    ``binary_op`` is a device callable that combines two ``dtype`` values and
    returns a ``dtype`` value. The generated invocable accepts either one scalar
    per thread or an ``items_per_thread`` local array and returns one aggregate
    value for the block. ``num_valid`` selects CUB's partial-tile scalar
    overload and is not valid with multi-item array inputs.
    """
    threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)
    return _reduce(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        binary_op=binary_op,
        algorithm=algorithm,
        num_valid=num_valid,
        methods=methods,
    )


def sum(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="warp_reductions",
    num_valid=None,
    methods=None,
    dim=None,
):
    """Build a block-wide sum invocable.

    This is the optimized CUB ``BlockReduce::Sum`` form. It accepts scalar or
    per-thread array input, depending on ``items_per_thread``, and returns the
    block aggregate through an output reference. ``num_valid`` selects CUB's
    partial-tile scalar overload and is not valid with multi-item array inputs.
    """
    threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)
    return _reduce(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        binary_op=None,
        algorithm=algorithm,
        num_valid=num_valid,
        methods=methods,
    )
