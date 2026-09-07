# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Regressions for common-root payload dtype and extent inference.

Two user-facing spellings used to fail during the single-phase rewrite:

* a ``ThreadData`` constructed with a numpy scalar-class dtype
  (``dtype=np.int32``) and filled element-wise raised ``Inconsistent
  inferred dtype`` once the payload reached a sort, discontinuity, or
  histogram collective, because the constructor spelling was never
  canonicalized against the trace-inferred Numba dtype;
* a ``coop.load`` result passed to ``coop.radix_sort_keys`` raised
  ``could not infer a static items_per_thread extent`` because the
  extent map did not recognize ``load`` as a shape-preserving producer.
"""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

import cuda.coop.numba_mlir as _numba_coop  # noqa: F401  (eager registration)
from cuda import coop

_THREADS = 64
_ITEMS_PER_THREAD = 2
_TOTAL_ITEMS = _THREADS * _ITEMS_PER_THREAD
_BINS = 16

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _numpy_dtype_fill_kernel(source, sorted_out, head_flags, histogram_out):
    tid = cuda.threadIdx.x
    block = coop.this_block()
    # The numpy scalar-class spelling plus element-wise fill is the
    # regression subject: it must unify with the trace-inferred dtype.
    keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=np.int32)
    for item in range(_ITEMS_PER_THREAD):
        keys[item] = source[tid * _ITEMS_PER_THREAD + item]

    ordered = coop.radix_sort_keys(block, keys)
    coop.store(block, sorted_out, ordered)

    heads = coop.discontinuity(block, ordered)
    coop.store(block, head_flags, heads)

    counts = coop.histogram(
        block,
        keys,
        bins=_BINS,
        bins_per_thread=1,
        counter_dtype=np.int32,
    )
    if tid < _BINS:
        histogram_out[tid] = counts[0]


@cuda.jit
def _load_into_radix_sort_kernel(source, sorted_out):
    block = coop.this_block()
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=np.int32)
    loaded = coop.load(block, source, items)
    ordered = coop.radix_sort_keys(block, loaded)
    coop.store(block, sorted_out, ordered)


def test_numpy_scalar_class_dtype_with_manual_fill_reaches_sort_collectives():
    rng = np.random.default_rng(11)
    source = rng.integers(0, _BINS, size=_TOTAL_ITEMS, dtype=np.int32)
    sorted_out = np.zeros_like(source)
    head_flags = np.zeros(_TOTAL_ITEMS, dtype=np.int32)
    histogram_out = np.zeros(_BINS, dtype=np.int32)

    _numpy_dtype_fill_kernel[1, _THREADS](
        source,
        sorted_out,
        head_flags,
        histogram_out,
    )
    cuda.synchronize()

    expected_sorted = np.sort(source)
    np.testing.assert_array_equal(sorted_out, expected_sorted)

    expected_heads = np.ones(_TOTAL_ITEMS, dtype=np.int32)
    expected_heads[1:] = (expected_sorted[1:] != expected_sorted[:-1]).astype(np.int32)
    np.testing.assert_array_equal(head_flags, expected_heads)

    np.testing.assert_array_equal(histogram_out, np.bincount(source, minlength=_BINS))


def test_load_result_feeds_radix_sort():
    rng = np.random.default_rng(13)
    source = rng.integers(-1000, 1000, size=_TOTAL_ITEMS, dtype=np.int32)
    sorted_out = np.zeros_like(source)

    _load_into_radix_sort_kernel[1, _THREADS](source, sorted_out)
    cuda.synchronize()

    np.testing.assert_array_equal(sorted_out, np.sort(source))
