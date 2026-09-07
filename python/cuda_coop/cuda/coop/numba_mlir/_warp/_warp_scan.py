# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from .. import _require_runtime

_require_runtime()

from cuda.coop._core import CxxFunction, CxxOperator, Dependency, PythonOperator
from cuda.coop._core.warp import make_warp_scan_spec

from .._common import (
    make_typed_cpp_literal,
    normalize_dtype_param,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._scan_op import ScanOp
from .._types import make_invocable_from_specialization


def _scan_op_param(scan_op):
    if scan_op.is_known or scan_op.is_sum:
        return CxxOperator(
            dtype=Dependency("T"),
            cpp=scan_op.op_cpp,
            name="scan_op",
        )
    if scan_op.is_callable:
        return PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=scan_op.op,
            name="scan_op",
        )
    raise RuntimeError("Unsupported scan operator for warp scan")


def _make_invocable(core_spec, threads_in_warp, threads_per_block=None):
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(
        specialization, threads=threads_in_warp, block_threads=threads_per_block
    )


def warp_exclusive_sum(
    dtype, threads_in_warp=32, warp_aggregate=None, threads_per_block=None
):
    """Build a warp-wide exclusive sum invocable.

    ``warp_aggregate`` may be a single-element local array that receives the
    aggregate for the logical warp.
    """
    dtype = normalize_dtype_param(dtype)
    core_spec = make_warp_scan_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        mode="exclusive",
        warp_aggregate=warp_aggregate is not None,
    )
    return _make_invocable(core_spec, threads_in_warp, threads_per_block)


def warp_inclusive_sum(
    dtype, threads_in_warp=32, warp_aggregate=None, threads_per_block=None
):
    """Build a warp-wide inclusive sum invocable.

    ``warp_aggregate`` may be a single-element local array that receives the
    aggregate for the logical warp.
    """
    dtype = normalize_dtype_param(dtype)
    core_spec = make_warp_scan_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        mode="inclusive",
        warp_aggregate=warp_aggregate is not None,
    )
    return _make_invocable(core_spec, threads_in_warp, threads_per_block)


def warp_exclusive_scan(
    dtype,
    scan_op,
    initial_value=None,
    threads_in_warp=32,
    valid_items=None,
    warp_aggregate=None,
    threads_per_block=None,
):
    """Build a warp-wide exclusive scan invocable.

    ``scan_op`` may be a known operator name, ``+`` for sum, or a Python device
    callable. ``initial_value`` and ``valid_items`` select the corresponding
    CUB ``WarpScan`` overloads when supplied. ``warp_aggregate`` may be a
    single-element local array that receives the aggregate for the logical warp.
    """
    dtype = normalize_dtype_param(dtype)
    scan_op = ScanOp(scan_op)
    use_sum_method = scan_op.is_sum and initial_value is None and valid_items is None
    core_spec = make_warp_scan_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        mode="exclusive",
        scan_operator=None if use_sum_method else _scan_op_param(scan_op),
        initial_value=(
            CxxFunction(
                cpp=make_typed_cpp_literal(initial_value, dtype),
                dtype=dtype,
                name="initial_value",
            )
            if initial_value is not None
            else None
        ),
        valid_items=valid_items is not None,
        warp_aggregate=warp_aggregate is not None,
    )
    return _make_invocable(core_spec, threads_in_warp, threads_per_block)


def warp_inclusive_scan(
    dtype,
    scan_op,
    initial_value=None,
    threads_in_warp=32,
    valid_items=None,
    warp_aggregate=None,
    threads_per_block=None,
):
    """Build a warp-wide inclusive scan invocable.

    ``scan_op`` may be a known operator name, ``+`` for sum, or a Python device
    callable. ``initial_value`` and ``valid_items`` select the corresponding
    CUB ``WarpScan`` overloads when supplied. ``warp_aggregate`` may be a
    single-element local array that receives the aggregate for the logical warp.
    """
    dtype = normalize_dtype_param(dtype)
    scan_op = ScanOp(scan_op)
    use_sum_method = scan_op.is_sum and initial_value is None and valid_items is None
    core_spec = make_warp_scan_spec(
        dtype=dtype,
        threads_in_warp=threads_in_warp,
        mode="inclusive",
        scan_operator=None if use_sum_method else _scan_op_param(scan_op),
        initial_value=(
            CxxFunction(
                cpp=make_typed_cpp_literal(initial_value, dtype),
                dtype=dtype,
                name="initial_value",
            )
            if initial_value is not None
            else None
        ),
        valid_items=valid_items is not None,
        warp_aggregate=warp_aggregate is not None,
    )
    return _make_invocable(core_spec, threads_in_warp, threads_per_block)


exclusive_sum = warp_exclusive_sum
inclusive_sum = warp_inclusive_sum
exclusive_scan = warp_exclusive_scan
inclusive_scan = warp_inclusive_scan
