# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Planner-private warp scan providers."""

import operator

from cuda.coop._core import CxxFunction, CxxOperator, Dependency, PythonOperator
from cuda.coop._core.warp import make_warp_scan_spec

from .._common import make_typed_cpp_literal, normalize_dtype_param
from .._core_adapter import NumbaMlirCoreAdapter
from .._scan_op import ScanOp
from .._types import make_invocable_from_specialization, numba_type_to_wrapper


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


def _scan_operator(scan_op: ScanOp):
    if scan_op.is_sum or scan_op.is_known:
        return CxxOperator(
            cpp=scan_op.op_cpp,
            dtype=Dependency("T"),
            name="scan_op",
        )
    return PythonOperator(
        ret_dtype=Dependency("T"),
        arg_dtypes=(Dependency("T"), Dependency("T")),
        op=scan_op.op,
        name="scan_op",
    )


def _scan(
    *,
    dtype,
    mode,
    scan_op="+",
    initial_value=None,
    threads_in_warp=32,
    valid_items=None,
    warp_aggregate=None,
    methods=None,
    threads_per_block=None,
):
    dtype = normalize_dtype_param(dtype)
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    if mode == "inclusive" and initial_value is not None:
        raise ValueError("inclusive scan does not accept initial_value")
    normalized_op = ScanOp(scan_op)
    if mode == "exclusive" and not normalized_op.is_sum and initial_value is None:
        raise ValueError("non-sum exclusive scan requires initial_value")
    if (
        mode == "exclusive"
        and normalized_op.is_sum
        and valid_items is not None
        and initial_value is None
    ):
        initial_value = 0
    use_sum_method = (
        normalized_op.is_sum and initial_value is None and valid_items is None
    )
    core_spec = make_warp_scan_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        mode=mode,
        scan_operator=(None if use_sum_method else _scan_operator(normalized_op)),
        initial_value=(
            None
            if initial_value is None
            else CxxFunction(
                cpp=make_typed_cpp_literal(initial_value, dtype),
                dtype=dtype,
                name="initial_value",
            )
        ),
        valid_items=valid_items is not None,
        warp_aggregate=warp_aggregate is not None,
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


def warp_exclusive_sum(
    dtype,
    threads_in_warp=32,
    warp_aggregate=None,
    threads_per_block=None,
):
    return _scan(
        dtype=dtype,
        mode="exclusive",
        threads_in_warp=threads_in_warp,
        warp_aggregate=warp_aggregate,
        threads_per_block=threads_per_block,
    )


def warp_inclusive_sum(
    dtype,
    threads_in_warp=32,
    warp_aggregate=None,
    threads_per_block=None,
):
    return _scan(
        dtype=dtype,
        mode="inclusive",
        threads_in_warp=threads_in_warp,
        warp_aggregate=warp_aggregate,
        threads_per_block=threads_per_block,
    )


def warp_exclusive_scan(
    dtype,
    scan_op,
    initial_value=None,
    threads_in_warp=32,
    valid_items=None,
    warp_aggregate=None,
    threads_per_block=None,
):
    return _scan(
        dtype=dtype,
        mode="exclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        warp_aggregate=warp_aggregate,
        threads_per_block=threads_per_block,
    )


def warp_inclusive_scan(
    dtype,
    scan_op,
    initial_value=None,
    threads_in_warp=32,
    valid_items=None,
    warp_aggregate=None,
    threads_per_block=None,
):
    return _scan(
        dtype=dtype,
        mode="inclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        warp_aggregate=warp_aggregate,
        threads_per_block=threads_per_block,
    )
