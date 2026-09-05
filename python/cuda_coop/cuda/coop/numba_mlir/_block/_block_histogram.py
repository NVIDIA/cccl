# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from .. import _require_runtime

_require_runtime()

from cuda.coop._core.block import (
    BlockHistogramOperation,
    make_block_histogram_spec,
    normalize_block_histogram_algorithm,
)

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
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


class _HistogramParent:
    """Compile-time placeholder rewritten by the single-phase coop pass."""

    def __init__(self, items, histogram, algorithm=None):
        self._items = items
        self._histogram = histogram
        self._algorithm = algorithm

    def init(self, histogram=None):
        raise RuntimeError(
            "coop._block.histogram(...).init() placeholder was not replaced; "
            "the Numba-CUDA-MLIR coop single-phase rewrite hook did not run or did not "
            "recognize this call."
        )

    def composite(self, items=None, histogram=None):
        raise RuntimeError(
            "coop._block.histogram(...).composite(...) placeholder was not "
            "replaced; the Numba-CUDA-MLIR coop single-phase rewrite hook did not run "
            "or did not recognize this call."
        )


class _HistogramFactory:
    """Two-phase BlockHistogram factory result."""

    def __init__(
        self,
        item_dtype,
        counter_dtype,
        threads_per_block,
        items_per_thread,
        bins,
        algorithm,
    ):
        self.init = _histogram_init(
            item_dtype,
            counter_dtype,
            threads_per_block,
            items_per_thread,
            bins,
            algorithm=algorithm,
        )
        self.composite = _histogram_composite(
            item_dtype,
            counter_dtype,
            threads_per_block,
            items_per_thread,
            bins,
            algorithm=algorithm,
        )


def histogram(
    items=None,
    histogram=None,
    algorithm=BlockHistogramAlgorithm.ATOMIC,
    threads_per_block=None,
    items_per_thread=1,
    bins=256,
    dim=None,
    temp_storage=None,
    item_dtype=None,
    counter_dtype=None,
):
    """Create a BlockHistogram parent placeholder or two-phase factory.

    The returned object supports ``.init()`` and ``.composite()`` calls inside a
    JIT kernel. The Numba-CUDA-MLIR rewrite lowers those calls to CUB ``BlockHistogram``
    initialization and composite-count operations with inferred item, counter,
    bin, and tile metadata. When called as ``make_histogram(dtype,
    counter_dtype, threads_per_block=...)``, the returned object exposes
    two-phase ``init`` and ``composite`` invocables. Every input sample must
    satisfy CUB's ``0 <= sample < bins`` precondition; violating it is undefined
    behavior.
    """
    if item_dtype is not None:
        if items is not None:
            raise ValueError("items and item_dtype cannot both be provided")
        items = item_dtype
    if counter_dtype is not None:
        if histogram is not None:
            raise ValueError("histogram and counter_dtype cannot both be provided")
        histogram = counter_dtype

    if threads_per_block is not None or dim is not None:
        if temp_storage is not None:
            raise NotImplementedError(
                "Explicit temp_storage is not supported for histogram."
            )
        threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)
        if histogram is None:
            raise ValueError("counter_dtype must be provided")
        return _HistogramFactory(
            items,
            histogram,
            threads_per_block,
            items_per_thread,
            bins,
            algorithm,
        )

    if items is None:
        raise ValueError("items must be provided")
    if histogram is None:
        raise ValueError("histogram must be provided")
    _resolve_histogram_algorithm(algorithm)
    return _HistogramParent(items, histogram, algorithm=algorithm)


def _histogram_init(
    item_dtype,
    counter_dtype,
    threads_per_block,
    items_per_thread,
    bins,
    algorithm=BlockHistogramAlgorithm.ATOMIC,
):
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
        operation=BlockHistogramOperation.INIT,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)


def _histogram_composite(
    item_dtype,
    counter_dtype,
    threads_per_block,
    items_per_thread,
    bins,
    algorithm=BlockHistogramAlgorithm.ATOMIC,
):
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
        operation=BlockHistogramOperation.COMPOSITE,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)
