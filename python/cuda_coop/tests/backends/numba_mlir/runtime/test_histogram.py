# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ruff: noqa: E402

"""Histogram counts, striped ownership, preservation, and scratch reuse."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


@pytest.mark.parametrize("algorithm", ["atomic", "sort"])
@pytest.mark.parametrize(
    "sample_dtype", [np.uint8, np.int32, np.uint32, np.int64, np.uint64]
)
@pytest.mark.parametrize("counter_dtype", [np.int32, np.uint32, np.int64, np.uint64])
def test_histogram_counts_preservation_and_padding(
    algorithm, sample_dtype, counter_dtype
):
    @cuda.jit
    def kernel(source, destination, preserved):
        block = coop.this_block()
        samples = coop.ThreadData(3)
        coop.load(block, source, samples, offset=cuda.blockIdx.x * 192)
        counts = coop.histogram(
            block,
            samples,
            bins=65,
            bins_per_thread=2,
            counter_dtype=counter_dtype,
            algorithm=algorithm,
        )
        coop.store(
            block,
            destination,
            counts,
            algorithm="striped",
            offset=cuda.blockIdx.x * 128,
        )
        coop.store(block, preserved, samples, offset=cuda.blockIdx.x * 192)

    source = ((np.arange(384) * 19) % 65).astype(sample_dtype)
    source[:192] = 64  # Contention and an entirely different second block.
    output = np.full(256, 99, dtype=counter_dtype)
    preserved = np.empty_like(source)
    kernel[2, 64](source, output, preserved)
    cuda.synchronize()
    for block in range(2):
        expected = np.zeros(128, dtype=counter_dtype)
        expected[:65] = np.bincount(
            source[block * 192 : (block + 1) * 192].astype(np.int64), minlength=65
        )
        np.testing.assert_array_equal(output[block * 128 : (block + 1) * 128], expected)
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("threads,bins", [(1, 1), (7, 13), (32, 1), (64, 128)])
@pytest.mark.parametrize("algorithm", ["atomic", "sort"])
@pytest.mark.parametrize("manual_sync", [False, True])
def test_fresh_calls_reuse_storage(threads, bins, algorithm, manual_sync):
    bins_per_thread = (bins + threads - 1) // threads
    auto_sync = not manual_sync

    @cuda.jit
    def kernel(source, destination):
        block = coop.this_block()
        samples = coop.ThreadData(3, dtype=np.int32)
        coop.load(block, source, samples)
        scratch = coop.TempStorage(auto_sync=auto_sync)
        first = coop.histogram(
            block,
            samples,
            bins=bins,
            bins_per_thread=bins_per_thread,
            algorithm=algorithm,
            temp_storage=scratch,
        )
        if manual_sync:
            cuda.syncthreads()
        second = coop.histogram(
            block,
            samples,
            bins=bins,
            bins_per_thread=bins_per_thread,
            algorithm=algorithm,
            temp_storage=scratch,
        )
        if manual_sync:
            cuda.syncthreads()
        for i in range(bins_per_thread):
            second[i] += first[i]
        coop.store(block, destination, second, algorithm="striped", valid_items=bins)

    source = (np.arange(threads * 3) % bins).astype(np.int32)
    output = np.empty(bins, dtype=np.int32)
    kernel[1, threads](source, output)
    cuda.synchronize()
    np.testing.assert_array_equal(output, 2 * np.bincount(source, minlength=bins))


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("algorithm", ["atomic", "sort"])
def test_qualified_scalar_and_local_array_return_counter_payload(scalar, algorithm):
    @cuda.jit
    def kernel(source, destination):
        block = numba_coop.this_block()
        if scalar:
            samples = source[cuda.threadIdx.x]
        else:
            samples = cuda.local.array(2, dtype=np.uint32)
            samples[0] = source[cuda.threadIdx.x * 2]
            samples[1] = source[cuda.threadIdx.x * 2 + 1]
        counts = numba_coop.histogram(
            block,
            samples,
            bins=17,
            bins_per_thread=2,
            algorithm=algorithm,
            counter_dtype=np.uint64,
        )
        numba_coop.store(block, destination, counts, algorithm="striped")

    source = (np.arange(32 if scalar else 64) % 17).astype(np.uint32)
    output = np.empty(64, dtype=np.uint64)
    kernel[1, 32](source, output)
    cuda.synchronize()
    expected = np.zeros(64, dtype=np.uint64)
    expected[:17] = np.bincount(source, minlength=17)
    np.testing.assert_array_equal(output, expected)
