# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from typing import Any

from .. import _require_runtime

_require_runtime()

from cuda.coop._core import (
    CxxFunction,
    CxxOperator,
    Dependency,
    PythonOperator,
    StatefulOperator,
)
from cuda.coop._core.block import make_block_scan_spec

from .._common import (
    CUB_BLOCK_SCAN_ALGOS,
    make_typed_cpp_literal,
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._enums import BlockScanAlgorithm
from .._scan_op import ScanOp
from .._types import (
    StatefulFunction,
    make_invocable_from_specialization,
    numba_type_to_wrapper,
)


def _resolve_scan_algorithm(algorithm: Any) -> str:
    if isinstance(algorithm, BlockScanAlgorithm):
        return CUB_BLOCK_SCAN_ALGOS[algorithm.name.lower()]

    if isinstance(algorithm, int):
        algo = BlockScanAlgorithm(algorithm)
        return CUB_BLOCK_SCAN_ALGOS[algo.name.lower()]

    if isinstance(algorithm, str):
        if algorithm.startswith("::cub::BlockScanAlgorithm::"):
            return algorithm
        if algorithm in CUB_BLOCK_SCAN_ALGOS:
            return CUB_BLOCK_SCAN_ALGOS[algorithm]
        upper = algorithm.upper()
        if upper in BlockScanAlgorithm.__members__:
            return CUB_BLOCK_SCAN_ALGOS[upper.lower()]

    allowed = sorted(CUB_BLOCK_SCAN_ALGOS.keys())
    raise ValueError(
        "Unsupported block scan algorithm "
        f"{algorithm!r}; expected one of {allowed} or BlockScanAlgorithm."
    )


def scan(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    mode="exclusive",
    scan_op="+",
    initial_value=None,
    block_prefix_callback_op=None,
    prefix_op=None,
    block_aggregate=None,
    algorithm="raking",
    methods=None,
    dim=None,
):
    """Build a block-wide scan invocable for per-thread item arrays.

    ``mode`` selects inclusive or exclusive scan. ``scan_op`` may be a known
    operator name, ``+`` for sum, or a Python device callable. ``initial_value``
    and ``block_prefix_callback_op`` follow the CUB ``BlockScan`` overloads and
    are mutually exclusive. ``block_aggregate`` may be a single-element local
    array that receives the block aggregate. CUB does not provide a scan
    overload that combines an explicit block aggregate with a prefix callback,
    so ``block_aggregate`` is mutually exclusive with ``prefix_op`` and
    ``block_prefix_callback_op``.
    """
    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")
    if mode not in ("exclusive", "inclusive"):
        raise ValueError(f"Unsupported mode: {mode!r}")

    threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)
    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)
    algorithm_cpp = _resolve_scan_algorithm(algorithm)
    scan_op = ScanOp(scan_op)
    if prefix_op is not None:
        if block_prefix_callback_op is not None:
            raise ValueError(
                "prefix_op and block_prefix_callback_op are mutually exclusive"
            )
        block_prefix_callback_op = prefix_op
    if initial_value is not None and block_prefix_callback_op is not None:
        raise ValueError(
            "initial_value and block_prefix_callback_op are mutually exclusive"
        )
    if block_aggregate is not None and block_prefix_callback_op is not None:
        raise ValueError("block_aggregate is not supported when prefix_op is provided")

    use_sum_method = scan_op.is_sum and initial_value is None

    initial_descriptor = None
    scan_descriptor = None
    if not use_sum_method:
        if initial_value is not None:
            initial_descriptor = CxxFunction(
                cpp=make_typed_cpp_literal(initial_value, dtype),
                dtype=dtype,
                name="initial_value",
            )
        if scan_op.is_sum or scan_op.is_known:
            scan_descriptor = CxxOperator(
                cpp=scan_op.op_cpp,
                dtype=Dependency("T"),
                name="scan_op",
            )
        elif isinstance(scan_op.op, StatefulFunction):
            scan_descriptor = StatefulOperator(
                op=scan_op.op,
                state_dtype=scan_op.op.dtype,
                ret_dtype=Dependency("T"),
                arg_dtypes=(Dependency("T"), Dependency("T")),
                name="scan_op",
            )
        else:
            scan_descriptor = PythonOperator(
                ret_dtype=Dependency("T"),
                arg_dtypes=(Dependency("T"), Dependency("T")),
                op=scan_op.op,
                name="scan_op",
            )

    prefix_descriptor = None
    if block_prefix_callback_op is not None:
        if isinstance(block_prefix_callback_op, StatefulFunction):
            prefix_descriptor = StatefulOperator(
                op=block_prefix_callback_op,
                state_dtype=block_prefix_callback_op.dtype,
                ret_dtype=Dependency("T"),
                arg_dtypes=(Dependency("T"),),
                name="prefix_op",
            )
        else:
            prefix_descriptor = PythonOperator(
                ret_dtype=Dependency("T"),
                arg_dtypes=(Dependency("T"),),
                op=block_prefix_callback_op,
                name="prefix_op",
            )

    type_definitions = ()
    if methods is not None:
        type_definitions = (numba_type_to_wrapper(dtype, methods=methods),)

    core_spec = make_block_scan_spec(
        dtype=dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        mode=mode,
        algorithm=algorithm_cpp,
        value_kind="array",
        scan_operator=scan_descriptor,
        initial_value=initial_descriptor,
        prefix_operator=prefix_descriptor,
        block_aggregate=block_aggregate is not None,
    )
    specialization = NumbaMlirCoreAdapter().materialize(
        core_spec.specialization,
        extra_type_definitions=type_definitions,
    )
    return make_invocable_from_specialization(specialization)


def exclusive_sum(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    block_prefix_callback_op=None,
    prefix_op=None,
    algorithm="raking",
    methods=None,
    dim=None,
):
    """Build an exclusive block-wide sum invocable."""
    return scan(
        dtype=dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        mode="exclusive",
        scan_op="+",
        block_prefix_callback_op=block_prefix_callback_op,
        prefix_op=prefix_op,
        algorithm=algorithm,
        methods=methods,
    )


def inclusive_sum(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    block_prefix_callback_op=None,
    prefix_op=None,
    algorithm="raking",
    methods=None,
    dim=None,
):
    """Build an inclusive block-wide sum invocable."""
    return scan(
        dtype=dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        mode="inclusive",
        scan_op="+",
        block_prefix_callback_op=block_prefix_callback_op,
        prefix_op=prefix_op,
        algorithm=algorithm,
        methods=methods,
    )


def exclusive_scan(
    dtype,
    threads_per_block=None,
    scan_op="+",
    items_per_thread=1,
    initial_value=None,
    block_prefix_callback_op=None,
    prefix_op=None,
    algorithm="raking",
    methods=None,
    dim=None,
):
    """Build an exclusive block-wide scan invocable with ``scan_op``."""
    return scan(
        dtype=dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        mode="exclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        block_prefix_callback_op=block_prefix_callback_op,
        prefix_op=prefix_op,
        algorithm=algorithm,
        methods=methods,
    )


def inclusive_scan(
    dtype,
    threads_per_block=None,
    scan_op="+",
    items_per_thread=1,
    initial_value=None,
    block_prefix_callback_op=None,
    prefix_op=None,
    algorithm="raking",
    methods=None,
    dim=None,
):
    """Build an inclusive block-wide scan invocable with ``scan_op``."""
    return scan(
        dtype=dtype,
        threads_per_block=resolve_threads_per_block_alias(threads_per_block, dim),
        items_per_thread=items_per_thread,
        mode="inclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        block_prefix_callback_op=block_prefix_callback_op,
        prefix_op=prefix_op,
        algorithm=algorithm,
        methods=methods,
    )
