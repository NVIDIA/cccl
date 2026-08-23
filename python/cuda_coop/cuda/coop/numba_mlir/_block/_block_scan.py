# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Planner-private block scan provider."""

from cuda.coop._core import CxxFunction, CxxOperator, Dependency, PythonOperator
from cuda.coop._core.block import make_block_scan_spec

from .._common import (
    CUB_BLOCK_SCAN_ALGOS,
    make_typed_cpp_literal,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._enums import BlockScanAlgorithm
from .._scan_op import ScanOp
from .._types import make_invocable_from_specialization, numba_type_to_wrapper


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

    dtype = normalize_dtype_param(dtype)
    normalized_op = ScanOp(scan_op)
    if mode == "exclusive" and not normalized_op.is_sum and initial_value is None:
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
    core_spec = make_block_scan_spec(
        dtype=dtype,
        block_dim=tuple(normalize_dim_param(threads_per_block)),
        items_per_thread=items_per_thread,
        mode=mode,
        algorithm=_scan_algorithm(algorithm),
        value_kind="array",
        scan_operator=scan_descriptor,
        initial_value=initial_descriptor,
        block_aggregate=block_aggregate is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=(numba_type_to_wrapper(dtype, methods=methods),),
    )
    return make_invocable_from_specialization(specialization)
