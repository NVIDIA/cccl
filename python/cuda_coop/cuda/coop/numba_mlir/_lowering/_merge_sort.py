# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Merge Sort provider lowering for Numba-CUDA-MLIR.

This module owns block and warp provider materialization. Callable hashing, sentinel validation, and fresh-result rewriting are compiler concerns.
"""

import operator

from cuda.coop._core import INT8, Dependency, PythonOperator
from cuda.coop._core.block import make_block_merge_sort_spec
from cuda.coop._core.warp import make_warp_merge_sort_spec

from .._compiler._parameters import normalize_dim_param, normalize_dtype_param
from .._types import (
    make_invocable_from_specialization,
    numba_type_to_cpp,
    numba_type_to_wrapper,
)
from ._core import NumbaMlirCoreAdapter


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_block_args(dtype, threads_per_block, items_per_thread, compare_op):
    if dtype is None:
        raise ValueError("dtype must be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    if compare_op is None:
        raise ValueError("compare_op must be provided")
    return (
        normalize_dim_param(threads_per_block),
        normalize_dtype_param(dtype),
        _positive_int(items_per_thread, name="items_per_thread"),
    )


def merge_sort_keys(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    compare_op=None,
    value_dtype=None,
    valid_items=None,
    oob_default=None,
    methods=None,
):
    """Build the block keys-only Merge Sort invocable selected by planning."""

    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    block_dim, dtype, items_per_thread = _normalize_block_args(
        dtype,
        threads_per_block,
        items_per_thread,
        compare_op,
    )
    if value_dtype is not None:
        value_dtype = normalize_dtype_param(value_dtype)
        if numba_type_to_cpp(value_dtype) == "storage_t":
            raise TypeError(
                "merge_sort_keys does not support user-defined value dtypes"
            )

    core_spec = make_block_merge_sort_spec(
        key_dtype=dtype,
        value_dtype=value_dtype,
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
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
    return make_invocable_from_specialization(specialization)


def merge_sort_pairs(
    keys,
    values,
    threads_per_block=None,
    items_per_thread=1,
    compare_op=None,
    valid_items=None,
    oob_default=None,
    methods=None,
):
    """Build the block key/value Merge Sort invocable selected by planning."""

    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    block_dim, keys, items_per_thread = _normalize_block_args(
        keys,
        threads_per_block,
        items_per_thread,
        compare_op,
    )
    if values is None:
        raise ValueError("values dtype must be provided")
    values = normalize_dtype_param(values)
    if numba_type_to_cpp(values) == "storage_t":
        raise TypeError("merge_sort_pairs does not support user-defined value dtypes")

    core_spec = make_block_merge_sort_spec(
        key_dtype=keys,
        value_dtype=values,
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
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
    return make_invocable_from_specialization(specialization)


def _normalize_warp_args(dtype, items_per_thread, compare_op):
    if dtype is None:
        raise ValueError("dtype must be provided")
    if compare_op is None:
        raise ValueError("compare_op must be provided")
    return (
        normalize_dtype_param(dtype),
        _positive_int(items_per_thread, name="items_per_thread"),
    )


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
    """Build the warp keys-only Merge Sort invocable selected by planning."""

    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    dtype, items_per_thread = _normalize_warp_args(
        dtype,
        items_per_thread,
        compare_op,
    )
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    if value_dtype is not None:
        value_dtype = normalize_dtype_param(value_dtype)
        if numba_type_to_cpp(value_dtype) == "storage_t":
            raise TypeError(
                "warp_merge_sort_keys does not support user-defined value dtypes"
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
        specialization,
        threads=threads_in_warp,
        block_threads=threads_per_block,
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
    """Build the warp key/value Merge Sort invocable selected by planning."""

    if (valid_items is None) != (oob_default is None):
        raise ValueError("valid_items and oob_default must be provided together")
    keys, items_per_thread = _normalize_warp_args(keys, items_per_thread, compare_op)
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    if values is None:
        raise ValueError("values dtype must be provided")
    values = normalize_dtype_param(values)
    if numba_type_to_cpp(values) == "storage_t":
        raise TypeError(
            "warp_merge_sort_pairs does not support user-defined value dtypes"
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
        specialization,
        threads=threads_in_warp,
        block_threads=threads_per_block,
    )


__all__: tuple[str, ...] = ()
