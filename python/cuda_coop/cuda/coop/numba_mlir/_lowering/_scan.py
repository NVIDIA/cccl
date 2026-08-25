# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Scan provider lowering for Numba-CUDA-MLIR.

This module owns block and warp scan materialization, including block prefix callbacks and StatefulFunction descriptors. It does not infer launch dimensions.
"""

import operator

from cuda.coop._core import (
    CxxFunction,
    CxxOperator,
    Dependency,
    PythonOperator,
    StatefulOperator,
)
from cuda.coop._core.block import make_block_scan_spec
from cuda.coop._core.warp import make_warp_scan_spec

from .._compiler._parameters import (
    CUB_BLOCK_SCAN_ALGOS,
    make_typed_cpp_literal,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._enums import BlockScanAlgorithm
from .._scan_op import ScanOp
from .._stateful_function import StatefulFunction
from .._types import make_invocable_from_specialization, numba_type_to_wrapper
from ._core import NumbaMlirCoreAdapter


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


def _scan_algorithm(algorithm) -> str:
    if isinstance(algorithm, bool):
        raise TypeError("block scan algorithm must not be bool")
    if isinstance(algorithm, BlockScanAlgorithm):
        return CUB_BLOCK_SCAN_ALGOS[algorithm.name.lower()]
    if not isinstance(algorithm, str):
        raise TypeError("block scan algorithm must be a string or BlockScanAlgorithm")
    if algorithm.startswith("::cub::BlockScanAlgorithm::"):
        return algorithm
    try:
        return CUB_BLOCK_SCAN_ALGOS[algorithm.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(CUB_BLOCK_SCAN_ALGOS))
        raise ValueError(
            f"Unsupported block scan algorithm {algorithm!r}; "
            f"expected one of: {allowed}"
        ) from exc


def scan(
    dtype,
    threads_per_block,
    items_per_thread=1,
    mode="exclusive",
    scan_op="+",
    initial_value=None,
    block_prefix_callback_op=None,
    prefix_op=None,
    block_aggregate=None,
    algorithm="raking",
    methods=None,
):
    """Build the direct CUB block-scan invocable selected by planning."""

    if isinstance(items_per_thread, bool) or not isinstance(items_per_thread, int):
        raise TypeError("items_per_thread must be an integer")
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be a positive integer")
    if mode not in {"exclusive", "inclusive"}:
        raise ValueError("mode must be 'exclusive' or 'inclusive'")
    if mode == "inclusive" and initial_value is not None:
        raise ValueError("inclusive scan does not accept initial_value")
    if block_prefix_callback_op is not None and prefix_op is not None:
        raise ValueError(
            "block_prefix_callback_op and prefix_op are mutually exclusive"
        )

    prefix_callback = (
        block_prefix_callback_op if block_prefix_callback_op is not None else prefix_op
    )
    if prefix_callback is not None and initial_value is not None:
        raise ValueError("initial_value and prefix callback are mutually exclusive")
    if prefix_callback is not None and block_aggregate is not None:
        raise ValueError("block_aggregate and prefix callback are mutually exclusive")

    dtype = normalize_dtype_param(dtype)
    normalized_op = ScanOp(scan_op)
    if (
        mode == "exclusive"
        and not normalized_op.is_sum
        and initial_value is None
        and prefix_callback is None
    ):
        raise ValueError("non-sum exclusive scan requires initial_value")
    use_sum_method = normalized_op.is_sum and initial_value is None
    initial_descriptor = (
        None
        if initial_value is None
        else CxxFunction(
            cpp=make_typed_cpp_literal(initial_value, dtype),
            dtype=dtype,
            name="initial_value",
        )
    )
    scan_descriptor = None
    if not use_sum_method:
        scan_descriptor = (
            CxxOperator(
                cpp=normalized_op.op_cpp,
                dtype=Dependency("T"),
                name="scan_op",
            )
            if normalized_op.is_sum or normalized_op.is_known
            else PythonOperator(
                ret_dtype=Dependency("T"),
                arg_dtypes=(Dependency("T"), Dependency("T")),
                op=normalized_op.op,
                name="scan_op",
            )
        )
    prefix_descriptor = None
    if prefix_callback is not None:
        descriptor_kwargs = {
            "ret_dtype": Dependency("T"),
            "arg_dtypes": (Dependency("T"),),
            "op": prefix_callback,
            "name": "prefix_op",
        }
        prefix_descriptor = (
            StatefulOperator(
                state_dtype=prefix_callback.dtype,
                **descriptor_kwargs,
            )
            if isinstance(prefix_callback, StatefulFunction)
            else PythonOperator(**descriptor_kwargs)
        )
    core_spec = make_block_scan_spec(
        dtype=dtype,
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        items_per_thread=items_per_thread,
        mode=mode,
        algorithm=_scan_algorithm(algorithm),
        value_kind="array",
        scan_operator=scan_descriptor,
        initial_value=initial_descriptor,
        prefix_operator=prefix_descriptor,
        block_aggregate=block_aggregate is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(specialization)


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
