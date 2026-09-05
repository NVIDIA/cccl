# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS_PER_THREAD = 3
_SAMPLE_COUNT = _THREADS * _ITEMS_PER_THREAD
_BINS = 97
_BINS_PER_THREAD = 2
_COUNTER_CAPACITY = _THREADS * _BINS_PER_THREAD
_PORTABLE_DTYPE_CASES = (
    (np.uint8, np.int32),
    (np.int32, np.uint32),
    (np.uint32, np.int64),
    (np.int64, np.uint64),
    (np.uint64, np.uint32),
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _common_histogram_kernel(source, atomic_output, sort_output, preserved):
    tid = (
        cuda.threadIdx.x
        + cuda.threadIdx.y * cuda.blockDim.x
        + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
    )
    samples = coop.ThreadData(_ITEMS_PER_THREAD, dtype=int)
    for index in range(_ITEMS_PER_THREAD):
        samples[index] = source[tid * _ITEMS_PER_THREAD + index]

    group = coop.this_block()
    atomic = coop.histogram(
        group,
        samples,
        bins=_BINS,
        bins_per_thread=_BINS_PER_THREAD,
        counter_dtype=int,
    )
    sorted_counts = coop.histogram(
        group,
        samples,
        bins=_BINS,
        bins_per_thread=_BINS_PER_THREAD,
        counter_dtype=np.int64,
        algorithm="sort",
    )

    for index in range(_BINS_PER_THREAD):
        output_index = tid * _BINS_PER_THREAD + index
        atomic_output[output_index] = atomic[index]
        sort_output[output_index] = sorted_counts[index]
    for index in range(_ITEMS_PER_THREAD):
        preserved[tid * _ITEMS_PER_THREAD + index] = samples[index]


@cuda.jit
def _qualified_histogram_kernel(source, atomic_output, sort_output, preserved):
    tid = (
        cuda.threadIdx.x
        + cuda.threadIdx.y * cuda.blockDim.x
        + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
    )
    samples = numba_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    for index in range(_ITEMS_PER_THREAD):
        samples[index] = source[tid * _ITEMS_PER_THREAD + index]

    group = numba_coop.this_block()
    atomic = numba_coop.histogram(
        group,
        samples,
        bins=_BINS,
        bins_per_thread=_BINS_PER_THREAD,
    )
    sorted_counts = numba_coop.histogram(
        group,
        samples,
        bins=_BINS,
        bins_per_thread=_BINS_PER_THREAD,
        counter_dtype=types.int64,
        algorithm="sort",
    )

    for index in range(_BINS_PER_THREAD):
        output_index = tid * _BINS_PER_THREAD + index
        atomic_output[output_index] = atomic[index]
        sort_output[output_index] = sorted_counts[index]
    for index in range(_ITEMS_PER_THREAD):
        preserved[tid * _ITEMS_PER_THREAD + index] = samples[index]


def _make_portable_dtype_kernel(counter_dtype):
    @cuda.jit
    def kernel(
        source,
        common_atomic_output,
        qualified_atomic_output,
        common_sort_output,
        qualified_sort_output,
        common_preserved,
        qualified_preserved,
    ):
        tid = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        common_samples = coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=source.dtype,
        )
        qualified_samples = numba_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=source.dtype,
        )
        for index in range(_ITEMS_PER_THREAD):
            value = source[tid * _ITEMS_PER_THREAD + index]
            common_samples[index] = value
            qualified_samples[index] = value

        common_group = coop.this_block()
        qualified_group = numba_coop.this_block()
        common_atomic = coop.histogram(
            common_group,
            common_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=counter_dtype,
        )
        qualified_atomic = numba_coop.histogram(
            qualified_group,
            qualified_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=counter_dtype,
        )
        common_sort = coop.histogram(
            common_group,
            common_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=counter_dtype,
            algorithm="sort",
        )
        qualified_sort = numba_coop.histogram(
            qualified_group,
            qualified_samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=counter_dtype,
            algorithm="sort",
        )

        for index in range(_BINS_PER_THREAD):
            output_index = tid * _BINS_PER_THREAD + index
            common_atomic_output[output_index] = common_atomic[index]
            qualified_atomic_output[output_index] = qualified_atomic[index]
            common_sort_output[output_index] = common_sort[index]
            qualified_sort_output[output_index] = qualified_sort[index]
        for index in range(_ITEMS_PER_THREAD):
            output_index = tid * _ITEMS_PER_THREAD + index
            common_preserved[output_index] = common_samples[index]
            qualified_preserved[output_index] = qualified_samples[index]

    return kernel


def _striped_oracle(values, dtype):
    counts = np.bincount(values.astype(np.int64), minlength=_BINS)
    expected = np.zeros(_COUNTER_CAPACITY, dtype=dtype)
    for rank in range(_THREADS):
        for item in range(_BINS_PER_THREAD):
            bin_index = rank + item * _THREADS
            if bin_index < _BINS:
                expected[rank * _BINS_PER_THREAD + item] = counts[bin_index]
    return expected


def _run_and_check(values):
    original = values.copy()
    common_atomic = np.full(_COUNTER_CAPACITY, -1, dtype=np.int32)
    common_sort = np.full(_COUNTER_CAPACITY, -1, dtype=np.int64)
    common_preserved = np.zeros(_SAMPLE_COUNT, dtype=np.uint8)
    qualified_atomic = np.full_like(common_atomic, -1)
    qualified_sort = np.full_like(common_sort, -1)
    qualified_preserved = np.zeros_like(common_preserved)

    _common_histogram_kernel[1, _BLOCK](
        values,
        common_atomic,
        common_sort,
        common_preserved,
    )
    _qualified_histogram_kernel[1, _BLOCK](
        values,
        qualified_atomic,
        qualified_sort,
        qualified_preserved,
    )
    cuda.synchronize()

    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(common_atomic, qualified_atomic)
    np.testing.assert_array_equal(common_sort, qualified_sort)
    np.testing.assert_array_equal(
        common_atomic,
        _striped_oracle(original, np.int32),
    )
    np.testing.assert_array_equal(
        common_sort,
        _striped_oracle(original, np.int64),
    )
    np.testing.assert_array_equal(common_preserved, original)
    np.testing.assert_array_equal(qualified_preserved, original)


@pytest.mark.evidence_for("group.histogram", backend="numba_mlir", evidence="runtime")
def test_common_histogram_matches_qualified_numba_and_independent_oracle_twice():
    indices = np.arange(_SAMPLE_COUNT, dtype=np.int32)
    _run_and_check(((indices * 17 + 3) % _BINS).astype(np.uint8))
    _run_and_check(((indices * 29 + 11) % _BINS).astype(np.uint8))


@pytest.mark.parametrize(
    ("sample_dtype", "counter_dtype"),
    _PORTABLE_DTYPE_CASES,
)
def test_common_histogram_portable_dtype_closure_matches_qualified_numba(
    sample_dtype,
    counter_dtype,
):
    indices = np.arange(_SAMPLE_COUNT, dtype=np.uint64)
    values = ((indices * 29 + indices // 5) % _BINS).astype(sample_dtype)
    original = values.copy()
    sentinel = np.iinfo(counter_dtype).max
    counter_outputs = [
        np.full(_COUNTER_CAPACITY, sentinel, dtype=counter_dtype) for _ in range(4)
    ]
    preserved_outputs = [np.zeros_like(values) for _ in range(2)]

    kernel = _make_portable_dtype_kernel(counter_dtype)
    kernel[1, _BLOCK](values, *counter_outputs, *preserved_outputs)
    cuda.synchronize()

    expected = _striped_oracle(original, counter_dtype)
    np.testing.assert_array_equal(values, original)
    for output in counter_outputs:
        assert output.dtype == np.dtype(counter_dtype)
        np.testing.assert_array_equal(output, expected)
    for output in preserved_outputs:
        assert output.dtype == np.dtype(sample_dtype)
        np.testing.assert_array_equal(output, original)
