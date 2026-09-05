# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from .. import _require_runtime

_require_runtime()

from cuda.coop._core import CxxOperator, Dependency, PythonOperator
from cuda.coop._core.warp import make_warp_reduce_spec

from .._common import normalize_dtype_param
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


def warp_reduce(
    dtype,
    binary_op,
    threads_in_warp=32,
    valid_items=None,
    methods=None,
    threads_per_block=None,
):
    """Build a warp-wide reduction invocable using ``binary_op``.

    ``threads_in_warp`` selects the logical warp size. ``valid_items`` enables
    the CUB overload that ignores lanes beyond the valid item count supplied at
    the call site.
    """
    dtype = normalize_dtype_param(dtype)
    core_spec = make_warp_reduce_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        operation="reduce",
        reduce_operator=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=binary_op,
            name="binary_op",
        ),
        valid_items=valid_items is not None,
        include_full_warp=valid_items is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(
        specialization, threads=threads_in_warp, block_threads=threads_per_block
    )


def warp_reduce_builtin(
    dtype,
    binary_op,
    threads_in_warp=32,
    valid_items=None,
    threads_per_block=None,
):
    """Build a CUB reduction with the canonical shared-core C++ operator."""

    try:
        operator_cpp = _BUILTIN_REDUCE_OPERATORS[binary_op]
    except KeyError as exc:
        names = ", ".join(sorted(_BUILTIN_REDUCE_OPERATORS))
        raise ValueError(
            f"Unsupported built-in reduction: {binary_op!r}; {names}"
        ) from exc

    dtype = normalize_dtype_param(dtype)
    core_spec = make_warp_reduce_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        operation="reduce",
        reduce_operator=CxxOperator(
            cpp=operator_cpp,
            dtype=Dependency("T"),
            name="binary_op",
        ),
        valid_items=valid_items is not None,
        include_full_warp=valid_items is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(
        specialization, threads=threads_in_warp, block_threads=threads_per_block
    )


def _warp_unary_reduce(
    dtype,
    method_name,
    threads_in_warp=32,
    valid_items=None,
    threads_per_block=None,
):
    dtype = normalize_dtype_param(dtype)
    core_spec = make_warp_reduce_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        operation=method_name.lower(),
        valid_items=valid_items is not None,
        include_full_warp=valid_items is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(
        specialization, threads=threads_in_warp, block_threads=threads_per_block
    )


def warp_sum(dtype, threads_in_warp=32, valid_items=None, threads_per_block=None):
    """Build a warp-wide sum invocable."""
    return _warp_unary_reduce(
        dtype=dtype,
        method_name="Sum",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
    )


def warp_max(dtype, threads_in_warp=32, valid_items=None, threads_per_block=None):
    """Build a warp-wide max invocable."""
    return _warp_unary_reduce(
        dtype=dtype,
        method_name="Max",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
    )


def warp_min(dtype, threads_in_warp=32, valid_items=None, threads_per_block=None):
    """Build a warp-wide min invocable."""
    return _warp_unary_reduce(
        dtype=dtype,
        method_name="Min",
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
    )


reduce = warp_reduce
sum = warp_sum
max = warp_max
min = warp_min
