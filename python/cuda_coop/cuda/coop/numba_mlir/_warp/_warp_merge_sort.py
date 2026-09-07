# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from .. import _require_runtime

_require_runtime()

from cuda.coop._core import INT8, Dependency, PythonOperator
from cuda.coop._core.warp import make_warp_merge_sort_spec

from .._common import (
    _validate_common_integer_key_dtype,
    _validate_common_numeric_dtype,
    normalize_dtype_param,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import (
    make_invocable_from_specialization,
    numba_type_to_cpp,
    numba_type_to_wrapper,
)


def _normalize_warp_merge_sort_args(dtype, items_per_thread, compare_op):
    if dtype is None:
        raise ValueError("dtype must be provided")
    if compare_op is None:
        raise ValueError("compare_op must be provided")
    try:
        items_per_thread = int(items_per_thread)
    except (TypeError, ValueError) as e:
        raise ValueError("items_per_thread must be an integer") from e
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")
    return normalize_dtype_param(dtype), items_per_thread


def warp_merge_sort_keys(
    dtype,
    items_per_thread,
    compare_op,
    value_dtype=None,
    threads_in_warp=32,
    valid_items=None,
    oob_default=None,
    methods=None,
    threads_per_block=None,
):
    """Build a warp-wide merge-sort invocable.

    ``value_dtype`` enables the key/value overload while preserving the
    ``merge_sort_keys`` factory spelling. ``valid_items`` and ``oob_default``
    must be provided together to select the partial-tile overload.
    """
    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    dtype, items_per_thread = _normalize_warp_merge_sort_args(
        dtype, items_per_thread, compare_op
    )
    if value_dtype is not None:
        value_dtype = normalize_dtype_param(value_dtype)
        if numba_type_to_cpp(value_dtype) == "storage_t":
            raise TypeError(
                "warp_merge_sort_keys value_dtype does not support user-defined "
                "value dtypes yet."
            )
    core_spec = make_warp_merge_sort_spec(
        key_dtype=dtype,
        value_dtype=value_dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        compare_operator=PythonOperator(
            ret_dtype=INT8,
            arg_dtypes=(Dependency("KeyT"), Dependency("KeyT")),
            op=compare_op,
            name="compare_op",
        ),
        valid_items=valid_items,
        oob_default=oob_default,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(
        specialization, threads=threads_in_warp, block_threads=threads_per_block
    )


def warp_merge_sort_pairs(
    keys,
    values,
    items_per_thread,
    compare_op,
    threads_in_warp=32,
    valid_items=None,
    oob_default=None,
    methods=None,
    threads_per_block=None,
):
    """Build a warp-wide key/value merge-sort invocable.

    ``valid_items`` and ``oob_default`` must be provided together to select
    the partial-tile overload.
    """
    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    keys, items_per_thread = _normalize_warp_merge_sort_args(
        keys, items_per_thread, compare_op
    )
    if values is None:
        raise ValueError("values dtype must be provided")
    values = normalize_dtype_param(values)
    if numba_type_to_cpp(values) == "storage_t":
        raise TypeError(
            "warp_merge_sort_pairs does not support user-defined value dtypes yet."
        )
    core_spec = make_warp_merge_sort_spec(
        key_dtype=keys,
        value_dtype=values,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        compare_operator=PythonOperator(
            ret_dtype=INT8,
            arg_dtypes=(Dependency("KeyT"), Dependency("KeyT")),
            op=compare_op,
            name="compare_op",
        ),
        valid_items=valid_items,
        oob_default=oob_default,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(keys, methods=methods),),
    )
    return make_invocable_from_specialization(
        specialization, threads=threads_in_warp, block_threads=threads_per_block
    )


def _common_warp_merge_sort_keys(**kwargs):
    """Materialize a portable keys-only specialization after dtype inference."""

    kwargs = dict(kwargs)
    kwargs["dtype"] = _validate_common_integer_key_dtype(
        kwargs["dtype"], operation="merge_sort_keys"
    )
    return warp_merge_sort_keys(**kwargs)


def _common_warp_merge_sort_pairs(**kwargs):
    """Materialize a portable key/value specialization after dtype inference."""

    kwargs = dict(kwargs)
    kwargs["keys"] = _validate_common_integer_key_dtype(
        kwargs["keys"], operation="merge_sort_pairs"
    )
    kwargs["values"] = _validate_common_numeric_dtype(
        kwargs["values"], operation="merge_sort_pairs", parameter="value"
    )
    return warp_merge_sort_pairs(**kwargs)


merge_sort_keys = warp_merge_sort_keys
merge_sort_pairs = warp_merge_sort_pairs
