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

import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._block import (
    BlockHistogramAlgorithm,
)

from ..support.runtime import (
    THREADS,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


def test_device_function_histogram_parent_child_with_kernel_block_primitives():
    threads_per_block = 64
    items_per_thread = 1
    bins = 8

    @cuda.jit(device=True)
    def device_histogram(d_in, d_hist):
        tid = cuda.threadIdx.x
        smem_histogram = coop.shared.array(bins, dtype=types.uint32)
        thread_samples = coop.local.array(items_per_thread, dtype=types.int32)
        histo = coop._block.histogram(thread_samples, smem_histogram)
        histo.init()
        cuda.syncthreads()
        coop._block.load(
            d_in,
            thread_samples,
            items_per_thread=items_per_thread,
            algorithm="direct",
        )
        histo.composite(thread_samples)
        cuda.syncthreads()
        if tid < bins:
            d_hist[tid] = smem_histogram[tid]

    @cuda.jit
    def kernel(d_in, d_hist, d_out):
        thread_samples = coop.local.array(items_per_thread, dtype=types.int32)

        coop._block.load(
            d_in,
            thread_samples,
            items_per_thread=items_per_thread,
            algorithm="direct",
        )
        coop._block.store(
            d_out,
            thread_samples,
            items_per_thread=items_per_thread,
            algorithm="direct",
        )

        device_histogram(d_in, d_hist)

    h_input = np.random.randint(0, bins, threads_per_block, dtype=np.int32)
    h_hist = np.zeros(bins, dtype=np.uint32)
    h_output = np.zeros(threads_per_block, dtype=np.int32)

    kernel[1, threads_per_block](h_input, h_hist, h_output)

    expected_hist = np.bincount(h_input, minlength=bins).astype(np.uint32)

    np.testing.assert_array_equal(h_output, h_input)
    np.testing.assert_array_equal(h_hist, expected_hist)


@cuda.jit
def _block_histogram_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    bins = 4
    thread_samples = cuda.local.array(1, cuda.int32)
    smem_histogram = cuda.shared.array(4, cuda.int32)

    histo = coop._block.histogram(thread_samples, smem_histogram)
    histo.init()
    cuda.syncthreads()

    thread_samples[0] = d_in[tid]
    histo.composite(thread_samples)
    cuda.syncthreads()

    if tid < bins:
        d_out[tid] = smem_histogram[tid]


def test_block_histogram_init_and_composite():
    h_input = np.arange(THREADS, dtype=np.int32) % np.int32(4)
    h_output = np.zeros(4, dtype=np.int32)

    _block_histogram_kernel[1, THREADS](h_input, h_output)

    expected = np.bincount(h_input, minlength=4).astype(np.int32)
    np.testing.assert_array_equal(h_output, expected)


@cuda.jit
def _block_histogram_sort_kernel(d_in, d_out):
    tid = cuda.threadIdx.x
    bins = 4
    thread_samples = cuda.local.array(1, cuda.int32)
    smem_histogram = cuda.shared.array(4, cuda.int32)

    histo = coop._block.histogram(
        thread_samples,
        smem_histogram,
        algorithm=BlockHistogramAlgorithm.SORT,
    )
    histo.init()
    cuda.syncthreads()

    thread_samples[0] = d_in[tid]
    histo.composite(thread_samples)
    cuda.syncthreads()

    if tid < bins:
        d_out[tid] = smem_histogram[tid]


def test_block_histogram_sort_algorithm():
    h_input = (np.arange(THREADS, dtype=np.int32) * np.int32(3)) % np.int32(4)
    h_output = np.zeros(4, dtype=np.int32)

    _block_histogram_sort_kernel[1, THREADS](h_input, h_output)

    expected = np.bincount(h_input, minlength=4).astype(np.int32)
    np.testing.assert_array_equal(h_output, expected)


def test_block_histogram_two_phase_factory():
    bins = 4
    histogram = coop._block.make_histogram(
        types.int32,
        types.int32,
        threads_per_block=THREADS,
        items_per_thread=1,
        bins=bins,
        algorithm=BlockHistogramAlgorithm.ATOMIC,
    )
    temp_storage_bytes = max(
        int(histogram.init.temp_storage_bytes),
        int(histogram.composite.temp_storage_bytes),
    )
    temp_storage_alignment = max(
        int(histogram.init.temp_storage_alignment),
        int(histogram.composite.temp_storage_alignment),
    )
    histogram_init = histogram.init
    histogram_composite = histogram.composite

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        samples = cuda.local.array(1, cuda.int32)
        counters = cuda.shared.array(bins, cuda.int32)
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )

        histogram_init(counters, temp_storage=temp_storage)
        cuda.syncthreads()
        samples[0] = d_in[tid]
        histogram_composite(samples, counters, temp_storage=temp_storage)
        cuda.syncthreads()
        if tid < bins:
            d_out[tid] = counters[tid]

    h_input = np.arange(THREADS, dtype=np.int32) % np.int32(bins)
    h_output = np.zeros(bins, dtype=np.int32)
    kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(
        h_output,
        np.full(bins, THREADS // bins, dtype=np.int32),
    )
