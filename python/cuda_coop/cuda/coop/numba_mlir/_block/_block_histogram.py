# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Planner-private fused BlockHistogram provider."""

from .. import _require_runtime

_require_runtime()

from cuda.coop._core.block import (
    BlockHistogramOperation,
    make_block_histogram_spec,
    normalize_block_histogram_algorithm,
)

from .._common import normalize_dim_param, normalize_dtype_param
from .._core_adapter import NumbaMlirCoreAdapter
from .._enums import BlockHistogramAlgorithm
from .._types import make_invocable_from_specialization


def _resolve_histogram_algorithm(algorithm):
    if isinstance(algorithm, int) and not isinstance(algorithm, bool):
        algorithm = BlockHistogramAlgorithm(algorithm)
    try:
        return normalize_block_histogram_algorithm(algorithm)
    except ValueError as exc:
        allowed = ", ".join(value.name for value in BlockHistogramAlgorithm)
        raise ValueError(
            "Unsupported block histogram algorithm "
            f"{algorithm!r}; expected one of {{{allowed}}} or a CUB enum string."
        ) from exc


def _normalize_histogram_dtype(dtype):
    """Normalize Histogram's portable Python ``int`` spelling locally."""

    if dtype is int:
        from numba_cuda_mlir import types

        return types.int32
    return normalize_dtype_param(dtype)


def _group_histogram(
    item_dtype,
    counter_dtype,
    threads_per_block,
    items_per_thread,
    bins,
    algorithm=BlockHistogramAlgorithm.ATOMIC,
):
    """Build one compiler-private fused BlockHistogram invocable."""

    dim = normalize_dim_param(threads_per_block)
    item_dtype = _normalize_histogram_dtype(item_dtype)
    counter_dtype = _normalize_histogram_dtype(counter_dtype)
    algorithm = _resolve_histogram_algorithm(algorithm)
    core_spec = make_block_histogram_spec(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        bins=bins,
        algorithm=algorithm,
        operation=BlockHistogramOperation.HISTOGRAM,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)


__all__: tuple[str, ...] = ()
