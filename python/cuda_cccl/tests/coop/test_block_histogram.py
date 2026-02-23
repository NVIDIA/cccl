# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

# example-begin imports
import numba
import numpy as np

# example-end imports
import pytest
from numba import cuda

import cuda.coop as coop
from cuda.coop import BlockHistogramAlgorithm, BlockLoadAlgorithm

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def bitwidth(np_type):
    return np.dtype(np_type).itemsize * 8


def get_histogram_bins_for_type(np_type):
    dtype = np.dtype(np_type)
    bins = 1 << (dtype.itemsize * 8)
    return bins if dtype.kind == "u" else bins >> 1


def test_block_histogram_init_composite():
    # example-begin histogram
    threads_per_block = 128
    items_per_thread = 1
    bins = 4

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        smem_histogram = cuda.shared.array(bins, numba.int32)
        thread_samples = cuda.local.array(items_per_thread, numba.int32)

        histo = coop.block.histogram(thread_samples, smem_histogram)
        histo.init()
        cuda.syncthreads()

        thread_samples[0] = d_in[tid]
        histo.composite(thread_samples)
        cuda.syncthreads()

        if tid < bins:
            d_out[tid] = smem_histogram[tid]

    # example-end histogram

    h_input = np.arange(threads_per_block, dtype=np.int32) % bins
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=np.int32)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    expected = np.zeros(bins, dtype=np.int32)
    for value in h_input:
        expected[value] += 1

    np.testing.assert_array_equal(h_output, expected)


def test_block_histogram_histo_atomic_single_phase0():
    item_dtype = np.uint8
    counter_dtype = np.uint32
    bins = 1 << bitwidth(item_dtype)
    items_per_thread = 4
    threads_per_block = 128
    # total_items = 1 << 15  # 32KB
    num_total_items = threads_per_block * items_per_thread

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    @cuda.jit
    def kernel2(d_in, d_out):
        tid = cuda.grid(1)
        if tid >= num_threads:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block

        num_valid_items = min(
            items_per_block,
            num_total_items - block_offset,
        )

        # Shared per-block histogram bin counts.
        smem_histogram = cuda.shared.array(bins, counter_dtype)
        thread_samples = cuda.local.array(items_per_thread, item_dtype)

        histo = coop.block.histogram(thread_samples, smem_histogram)
        histo.composite(d_in[block_offset : num_valid_items + block_offset])

        cuda.syncthreads()

        thread_linear = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        for bin_idx in range(thread_linear, bins, threads_per_block):
            d_out[bin_idx] = smem_histogram[bin_idx]

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.grid(1)
        if tid >= num_threads:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block

        num_valid_items = min(
            items_per_block,
            num_total_items - block_offset,
        )

        # Shared per-block histogram bin counts.
        smem_histogram = cuda.shared.array(bins, counter_dtype)
        thread_samples = cuda.local.array(items_per_thread, item_dtype)

        histo = coop.block.histogram(thread_samples, smem_histogram)

        # Initialize the histogram.
        histo.init()

        cuda.syncthreads()

        coop.block.load(
            d_in[block_offset:],
            thread_samples,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
            num_valid_items=num_valid_items,
        )

        cuda.syncthreads()

        histo.composite(thread_samples)

        cuda.syncthreads()

        thread_linear = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        for bin_idx in range(thread_linear, bins, threads_per_block):
            d_out[bin_idx] = smem_histogram[bin_idx]

    h_input = np.random.randint(0, bins, num_total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(bins, dtype=counter_dtype))
    num_blocks = blocks_per_grid
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    # k = kernel2[num_blocks, threads_per_block]
    k(d_input, d_output)

    actual = d_output.copy_to_host()
    expected = np.bincount(h_input, minlength=bins).astype(counter_dtype)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(expected) == num_total_items

    np.testing.assert_array_equal(actual, expected)


def test_block_histogram_histo_atomic_single_phase_0():
    item_dtype = np.uint8
    counter_dtype = np.uint32
    bins = 1 << bitwidth(item_dtype)
    items_per_thread = 4
    threads_per_block = 128
    # num_total_items = 1 << 15  # 32KB
    num_total_items = 1024
    # num_total_items = threads_per_block * items_per_thread

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.grid(1)
        if tid >= num_total_items:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        # Shared per-block histogram bin counts.
        smem_histogram = cuda.shared.array(bins, counter_dtype)
        thread_samples = cuda.local.array(items_per_thread, item_dtype)

        histo = coop.block.histogram(thread_samples, smem_histogram)
        histo.init()
        cuda.syncthreads()

        while block_offset < num_total_items:
            num_valid_items = min(
                items_per_block,
                num_total_items - block_offset,
            )

            # Load with padding.
            coop.block.load(
                d_in[block_offset:],
                thread_samples,
                items_per_thread=items_per_thread,
                algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
                num_valid_items=num_valid_items,
            )

            # Zero-pad invalid thread items explicitly.
            for i in range(items_per_thread):
                global_idx = block_offset + thread_offset + i
                if global_idx >= num_total_items:
                    thread_samples[i] = 0

            histo.composite(thread_samples)

            cuda.syncthreads()

            block_offset += items_per_block * cuda.gridDim.x

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        for bin_idx in range(cuda.threadIdx.x, bins, threads_per_block):
            cuda.atomic.add(d_out, bin_idx, smem_histogram[bin_idx])

    h_input = np.random.randint(0, bins, num_total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(bins, dtype=counter_dtype))
    num_blocks = blocks_per_grid
    print(f"num blocks: {num_blocks}, threads per block: {threads_per_block}")
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    # k = kernel2[num_blocks, threads_per_block]
    k(d_input, d_output)

    actual = d_output.copy_to_host()
    expected = np.bincount(h_input, minlength=bins).astype(counter_dtype)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(expected) == num_total_items
    assert np.sum(actual) == num_total_items

    np.testing.assert_array_equal(actual, expected)


def test_block_histogram_histo_atomic_single_phase_1():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        tid = cuda.grid(1)
        if tid >= num_total_items:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(
            d_out.shape,
            d_out.dtype,
            alignment=128,
        )
        thread_samples = coop.local.array(
            items_per_thread,
            d_in.dtype,
            alignment=16,
        )

        histo = coop.block.histogram(thread_samples, smem_histogram)
        histo.init()
        cuda.syncthreads()

        while block_offset < num_total_items:
            num_valid_items = min(
                items_per_block,
                num_total_items - block_offset,
            )

            # Load with padding.
            coop.block.load(
                d_in[block_offset:],
                thread_samples,
                algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
                num_valid_items=num_valid_items,
            )

            # Zero-pad invalid thread items explicitly.
            for i in range(items_per_thread):
                global_idx = block_offset + thread_offset + i
                if global_idx >= num_total_items:
                    thread_samples[i] = 0

            histo.composite(thread_samples)

            cuda.syncthreads()

            block_offset += items_per_block * cuda.gridDim.x

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        for bin_idx in range(cuda.threadIdx.x, bins, threads_per_block):
            cuda.atomic.add(d_out, bin_idx, smem_histogram[bin_idx])

    item_dtype = np.uint8
    counter_dtype = np.uint32
    bins = 1 << bitwidth(item_dtype)
    items_per_thread = 4
    threads_per_block = 128
    # num_total_items = 1 << 15  # 32KB
    num_total_items = 1024
    # num_total_items = threads_per_block * items_per_thread

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    h_input = np.random.randint(0, bins, num_total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(bins, dtype=counter_dtype))
    num_blocks = blocks_per_grid
    print(f"num blocks: {num_blocks}, threads per block: {threads_per_block}")
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    # k = kernel2[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread, num_total_items)

    actual = d_output.copy_to_host()
    expected = np.bincount(h_input, minlength=bins).astype(counter_dtype)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(expected) == num_total_items
    assert np.sum(actual) == num_total_items

    np.testing.assert_array_equal(actual, expected)


# @pytest.mark.parametrize("item_dtype", [np.int8, np.uint8])
# @pytest.mark.parametrize("counter_dtype", [np.int32, np.uint32])
# @pytest.mark.parametrize("threads_per_block", [32, 128, (4, 16), (4, 8, 16)])
# @pytest.mark.parametrize("items_per_thread", [2, 4, 8])
@pytest.mark.parametrize("item_dtype", [np.uint8])  # , np.int8])
@pytest.mark.parametrize("counter_dtype", [np.uint32])  # , np.uint32])
@pytest.mark.parametrize("threads_per_block", [32, 128, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [2, 4])  # , 8])
@pytest.mark.parametrize(
    "num_total_items",
    [
        1 << 10,  # 1KB
        1 << 12,  # 4KB
        1 << 15,  # 32KB
        1 << 19,  # 512KB
        1 << 23,  # 8MB
        1 << 28,  # 256MB
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        BlockHistogramAlgorithm.ATOMIC,
        BlockHistogramAlgorithm.SORT,
    ],
)
def test_block_histogram_histo_atomic_single_phase_2(
    item_dtype,
    counter_dtype,
    threads_per_block,
    items_per_thread,
    num_total_items,
    algorithm,
):
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        tid = cuda.grid(1)
        if tid >= num_total_items:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(
            d_out.shape,
            d_out.dtype,
            alignment=128,
        )
        thread_samples = coop.local.array(
            items_per_thread,
            d_in.dtype,
            alignment=16,
        )

        histo = coop.block.histogram(thread_samples, smem_histogram)
        histo.init()
        cuda.syncthreads()

        while block_offset < num_total_items:
            num_valid_items = min(
                items_per_block,
                num_total_items - block_offset,
            )

            # Load with padding.
            coop.block.load(
                d_in[block_offset:],
                thread_samples,
                algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
                num_valid_items=num_valid_items,
            )

            # Zero-pad invalid thread items explicitly.
            for i in range(items_per_thread):
                global_idx = block_offset + thread_offset + i
                if global_idx >= num_total_items:
                    thread_samples[i] = 0

            histo.composite(thread_samples)

            cuda.syncthreads()

            block_offset += items_per_block * cuda.gridDim.x

        cuda.syncthreads()

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        for bin_idx in range(cuda.threadIdx.x, bins, threads_per_block):
            cuda.atomic.add(d_out, bin_idx, smem_histogram[bin_idx])

        cuda.syncthreads()

    bins = get_histogram_bins_for_type(item_dtype)

    threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    h_input = np.random.randint(0, bins, num_total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(bins, dtype=counter_dtype))
    num_blocks = blocks_per_grid
    print(
        f"num_blocks: {num_blocks}, "
        f"threads_per_block: {threads_per_block}, "
        f"items_per_thread: {items_per_thread}, "
        f"num_total_items: {num_total_items}, "
        f"items_per_block: {items_per_block}, "
        f"blocks_per_grid: {blocks_per_grid}, "
        f"bins: {bins}"
    )

    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread, num_total_items)

    actual = d_output.copy_to_host()
    expected = np.bincount(h_input, minlength=bins).astype(counter_dtype)
    num_invalid = blocks_per_grid * items_per_block - num_total_items
    if num_invalid:
        # Invalid items are explicitly zero-padded, so they contribute to bin 0.
        expected[0] += num_invalid
    # Sanity check sum of histo bins matches total items + padding.
    assert np.sum(actual) == num_total_items + num_invalid
    assert np.sum(expected) == num_total_items + num_invalid

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("item_dtype", [np.uint8])  # , np.int8])
@pytest.mark.parametrize("counter_dtype", [np.int32])  # , np.uint32])
@pytest.mark.parametrize("threads_per_block", [128, 223, (8, 8, 16)])
@pytest.mark.parametrize("items_per_thread", [3, 4, 7])
@pytest.mark.parametrize(
    "num_total_items",
    [
        # Add a little odd fudge to ensure we handle awkward sizes properly.
        (1 << 10) + 7,  # 1KB
        (1 << 12) + 9,  # 4KB
        (1 << 15) + 13,  # 32KB
        (1 << 19) + 1,  # 512KB
        (1 << 23) + 23,  # 8MB
        (1 << 28) + 5,  # 256MB
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        BlockHistogramAlgorithm.ATOMIC,
        BlockHistogramAlgorithm.SORT,
    ],
)
def foo():
    pass


@pytest.mark.parametrize("item_dtype", [np.uint8])  # , np.int8])
@pytest.mark.parametrize("counter_dtype", [np.int32])  # , np.uint32])
@pytest.mark.parametrize("threads_per_block", [128, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [2, 4, 8])
@pytest.mark.parametrize(
    "num_total_items",
    [
        1 << 10,  # 1KB
        1 << 12,  # 4KB
        1 << 15,  # 32KB
        1 << 19,  # 512KB
        1 << 23,  # 8MB
        1 << 28,  # 256MB
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        BlockHistogramAlgorithm.ATOMIC,
        BlockHistogramAlgorithm.SORT,
    ],
)
def test_block_histogram_histo_atomic_single_phase_3(
    item_dtype,
    counter_dtype,
    threads_per_block,
    items_per_thread,
    num_total_items,
    algorithm,
):
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        tid = cuda.grid(1)
        if tid >= num_total_items:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(
            d_out.shape,
            d_out.dtype,
            alignment=128,
        )
        thread_samples = coop.local.array(
            items_per_thread,
            d_in.dtype,
            alignment=16,
        )

        histo = coop.block.histogram(thread_samples, smem_histogram)
        histo.init()
        cuda.syncthreads()

        while block_offset < num_total_items:
            num_valid_items = min(
                items_per_block,
                num_total_items - block_offset,
            )

            coop.block.load(
                d_in[block_offset:],
                thread_samples,
                algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
                num_valid_items=num_valid_items,
            )

            # Explicitly zero-pad invalid thread items.
            for i in range(items_per_thread):
                global_idx = block_offset + thread_offset + i
                if global_idx >= num_total_items:
                    thread_samples[i] = 0

            # Composite into shared histogram.
            histo.composite(thread_samples)
            cuda.syncthreads()

            # Atomically accumulate block-local histogram to global memory.
            for bin_idx in range(cuda.threadIdx.x, bins, threads_per_block):
                cuda.atomic.add(d_out, bin_idx, smem_histogram[bin_idx])

            cuda.syncthreads()

            # Reset shared histogram bins to zero before next iteration.
            histo.init()
            cuda.syncthreads()

            # Increment block offset for next iteration
            block_offset += items_per_block * cuda.gridDim.x

    bins = get_histogram_bins_for_type(item_dtype)

    threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    h_input = np.random.randint(0, bins, num_total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(bins, dtype=counter_dtype))
    num_blocks = blocks_per_grid
    print(
        f"num_blocks: {num_blocks}, "
        f"threads_per_block: {threads_per_block}, "
        f"items_per_thread: {items_per_thread}, "
        f"num_total_items: {num_total_items}, "
        f"items_per_block: {items_per_block}, "
        f"blocks_per_grid: {blocks_per_grid}, "
        f"bins: {bins}"
    )

    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread, num_total_items)

    actual = d_output.copy_to_host()
    expected = np.bincount(h_input, minlength=bins).astype(counter_dtype)
    num_invalid = blocks_per_grid * items_per_block - num_total_items
    if num_invalid:
        # Invalid items are explicitly zero-padded, so they contribute to bin 0.
        expected[0] += num_invalid
    # Sanity check sum of histo bins matches total items + padding.
    assert np.sum(actual) == num_total_items + num_invalid
    assert np.sum(expected) == num_total_items + num_invalid

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("item_dtype", [np.uint8])  # , np.int8])
@pytest.mark.parametrize("counter_dtype", [np.uint32])  # , np.uint32])
@pytest.mark.parametrize("threads_per_block", [32, 128, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [2, 4])  # , 8])
@pytest.mark.parametrize(
    "num_total_items",
    [
        1 << 10,  # 1KB
        1 << 12,  # 4KB
        1 << 15,  # 32KB
        1 << 19,  # 512KB
        1 << 23,  # 8MB
        1 << 28,  # 256MB
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        BlockHistogramAlgorithm.ATOMIC,
        BlockHistogramAlgorithm.SORT,
    ],
)
def test_block_histogram_histo_atomic_single_phase_4(
    item_dtype,
    counter_dtype,
    threads_per_block,
    items_per_thread,
    num_total_items,
    algorithm,
):
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        tid = cuda.grid(1)
        if tid >= num_total_items:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(
            d_out.shape,
            d_out.dtype,
            alignment=128,
        )
        thread_samples = coop.local.array(
            items_per_thread,
            d_in.dtype,
            alignment=16,
        )

        histo = coop.block.histogram(thread_samples, smem_histogram)
        histo.init()
        cuda.syncthreads()

        while block_offset < num_total_items:
            num_valid_items = min(
                items_per_block,
                num_total_items - block_offset,
            )

            coop.block.load(
                d_in[block_offset:],
                thread_samples,
                algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
                num_valid_items=num_valid_items,
            )

            # Zero-pad invalid thread items explicitly.
            for i in range(items_per_thread):
                global_idx = block_offset + thread_offset + i
                if global_idx >= num_total_items:
                    thread_samples[i] = 0

            histo.composite(thread_samples)

            cuda.syncthreads()

            block_offset += items_per_block * cuda.gridDim.x

        cuda.syncthreads()

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        for bin_idx in range(cuda.threadIdx.x, bins, threads_per_block):
            cuda.atomic.add(d_out, bin_idx, smem_histogram[bin_idx])

        cuda.syncthreads()

    bins = get_histogram_bins_for_type(item_dtype)

    threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    h_input = np.random.randint(0, bins, num_total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(bins, dtype=counter_dtype))
    num_blocks = blocks_per_grid
    print(
        f"num_blocks: {num_blocks}, "
        f"threads_per_block: {threads_per_block}, "
        f"items_per_thread: {items_per_thread}, "
        f"num_total_items: {num_total_items}, "
        f"items_per_block: {items_per_block}, "
        f"blocks_per_grid: {blocks_per_grid}, "
        f"bins: {bins}"
    )

    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread, num_total_items)

    actual = d_output.copy_to_host()
    expected = np.bincount(h_input, minlength=bins).astype(counter_dtype)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(actual) == num_total_items
    assert np.sum(expected) == num_total_items

    np.testing.assert_array_equal(actual, expected)


def test_block_histogram_histo_atomic_single_phase1():
    bins = 256
    item_dtype = np.uint8
    counter_dtype = np.uint32
    items_per_thread = 4
    threads_per_block = 128
    # total_items = 1 << 15  # 32KB
    algorithm = BlockHistogramAlgorithm.ATOMIC

    block_load = coop.block.load(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )
    # block_histogram = coop.block.histogram(
    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        dim=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        bins=bins,
    )

    @cuda.jit
    def kernel(d_in, d_out):
        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(bins, dtype=d_in.dtype)
        thread_samples = coop.local.array(items_per_thread, dtype=item_dtype)

        block_load(d_in, thread_samples)

        block_histogram(thread_samples, smem_histogram)

        cuda.syncthreads()

        # Store the histogram bin counts to the output.
        thread_linear = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        for bin_idx in range(thread_linear, bins, threads_per_block):
            d_out[bin_idx] = smem_histogram[bin_idx]


def test_block_histogram_histo_sort_two_phase0():
    bins = 256
    item_dtype = np.uint8
    counter_dtype = np.uint32
    items_per_thread = 4
    threads_per_block = 128
    total_items = 1 << 15  # 32KB
    algorithm = BlockHistogramAlgorithm.SORT

    block_load = coop.block.load(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )
    # block_histogram = coop.block.histogram(
    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        dim=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        bins=bins,
    )
    block_store = coop.block.store(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )

    @dataclass
    class KernelParams:
        block_load: Any
        block_histogram: Any
        block_store: Any

    kp = KernelParams(
        block_load=block_load,
        block_histogram=block_histogram,
        block_store=block_store,
    )
    kp = coop.gpu_dataclass(kp)

    temp_storage_bytes = max(
        (
            block_load.temp_storage_bytes,
            block_store.temp_storage_bytes,
            block_histogram.temp_storage_bytes,
        )
    )
    temp_storage_bytes_sum = sum(
        (
            block_load.temp_storage_bytes,
            block_store.temp_storage_bytes,
            block_histogram.temp_storage_bytes,
        )
    )
    temp_storage_alignment = max(
        (
            block_load.temp_storage_alignment,
            block_store.temp_storage_alignment,
            block_histogram.temp_storage_alignment,
        )
    )

    assert kp.temp_storage_bytes_max == temp_storage_bytes
    assert kp.temp_storage_bytes_sum == temp_storage_bytes_sum
    assert kp.temp_storage_alignment == temp_storage_alignment

    bl0ck_histogram = block_histogram

    @cuda.jit
    def kernel(d_in, d_out):
        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(bins, dtype=counter_dtype)
        # temp_storage = coop.shared.array(temp_storage_bytes, np.uint8,
        #                                 alignment=temp_storage_alignment)
        # thread_samples = cuda.local.array(items_per_thread, item_dtype)
        thread_samples = coop.local.array(items_per_thread, dtype=item_dtype)
        # thread_samples = coop.local.array(items_per_thread, item_dtype)

        # histo = block_histogram(temp_storage)
        histo = bl0ck_histogram()

        # Initialize the histogram.

        histo.init(smem_histogram)

        block_load(d_in, thread_samples)

        histo.composite(thread_samples, smem_histogram)

        cuda.syncthreads()

        # Store the histogram bin counts to the output.
        thread_linear = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        for bin_idx in range(thread_linear, bins, threads_per_block):
            d_out[bin_idx] = smem_histogram[bin_idx]

    h_input = np.random.randint(0, bins, total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(bins, dtype=counter_dtype))
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output)

    items_per_block = threads_per_block * items_per_thread
    expected_histo = np.bincount(h_input[:items_per_block], minlength=bins)
    # Sanity check sum of histo bins matches loaded items.
    assert np.sum(expected_histo) == items_per_block

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, expected_histo)


@pytest.mark.xfail(
    reason="Two-phase BlockHistogram SORT accumulation not supported yet"
)
def test_block_histogram_histo_sort_two_phase1():
    bins = 256
    item_dtype = np.uint8
    counter_dtype = np.uint32
    items_per_thread = 4
    threads_per_block = 128
    total_items = 1 << 15  # 32KB
    algorithm = BlockHistogramAlgorithm.SORT

    block_load = coop.block.load(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )
    # block_histogram = coop.block.histogram(
    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        dim=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        bins=bins,
    )
    block_store = coop.block.store(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )

    @dataclass
    class KernelParams:
        block_load: Any
        block_histogram: Any
        block_store: Any

    kp = KernelParams(
        block_load=block_load,
        block_histogram=block_histogram,
        block_store=block_store,
    )
    kp = coop.gpu_dataclass(kp)

    temp_storage_bytes = max(
        (
            block_load.temp_storage_bytes,
            block_store.temp_storage_bytes,
            block_histogram.temp_storage_bytes,
        )
    )
    temp_storage_bytes_sum = sum(
        (
            block_load.temp_storage_bytes,
            block_store.temp_storage_bytes,
            block_histogram.temp_storage_bytes,
        )
    )
    temp_storage_alignment = max(
        (
            block_load.temp_storage_alignment,
            block_store.temp_storage_alignment,
            block_histogram.temp_storage_alignment,
        )
    )

    assert kp.temp_storage_bytes_max == temp_storage_bytes
    assert kp.temp_storage_bytes_sum == temp_storage_bytes_sum
    assert kp.temp_storage_alignment == temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.grid(1)
        num_threads = cuda.gridsize(1)
        start = tid * items_per_thread
        step = num_threads * items_per_thread

        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(bins, counter_dtype)
        # temp_storage = coop.shared.array(temp_storage_bytes, np.uint8,
        #                                 alignment=temp_storage_alignment)
        thread_samples = coop.local.array(items_per_thread, item_dtype)

        # histo = block_histogram(temp_storage)
        histo = block_histogram

        # Initialize the histogram.

        # N.B. The C++ `CounterT` template parameter (`counter_dtype` in
        #      Python) is inferred from the `smem_histogram` dtype.
        #       `bins` is inferred from the `smem_histogram.shape`.
        histo.init(smem_histogram)
        cuda.syncthreads()

        index = start
        while index < total_items:
            block_load(
                d_in[index:],
                thread_samples,
            )
            cuda.syncthreads()
            histo.composite(thread_samples, smem_histogram)
            cuda.syncthreads()
            index += step

        cuda.syncthreads()

        # Store the histogram bin counts to the output.
        thread_linear = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        for bin_idx in range(thread_linear, bins, threads_per_block):
            d_out[bin_idx] = smem_histogram[bin_idx]

    h_input = np.random.randint(0, bins, total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(bins, dtype=counter_dtype))
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output)

    expected_histo = np.bincount(h_input, minlength=bins)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(expected_histo) == total_items

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, expected_histo)


@pytest.mark.xfail(
    reason="Two-phase BlockHistogram SORT accumulation not supported yet"
)
def test_block_histogram_histo_sort_two_phase2():
    bins = 256
    item_dtype = np.uint8
    counter_dtype = np.uint32
    items_per_thread = 4
    threads_per_block = 128
    total_items = 1 << 15  # 32KB
    algorithm = BlockHistogramAlgorithm.SORT

    block_load = coop.block.load(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )
    # block_histogram = coop.block.histogram(
    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        dim=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        bins=bins,
    )
    block_store = coop.block.store(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )

    @dataclass
    class KernelParams:
        block_load: Any
        block_histogram: Any
        block_store: Any

    kp = KernelParams(
        block_load=block_load,
        block_histogram=block_histogram,
        block_store=block_store,
    )
    kp = coop.gpu_dataclass(kp)

    temp_storage_bytes = max(
        (
            block_load.temp_storage_bytes,
            block_store.temp_storage_bytes,
            block_histogram.temp_storage_bytes,
        )
    )
    temp_storage_bytes_sum = sum(
        (
            block_load.temp_storage_bytes,
            block_store.temp_storage_bytes,
            block_histogram.temp_storage_bytes,
        )
    )
    temp_storage_alignment = max(
        (
            block_load.temp_storage_alignment,
            block_store.temp_storage_alignment,
            block_histogram.temp_storage_alignment,
        )
    )

    assert kp.temp_storage_bytes_max == temp_storage_bytes
    assert kp.temp_storage_bytes_sum == temp_storage_bytes_sum
    assert kp.temp_storage_alignment == temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.grid(1)
        num_threads = cuda.gridsize(1)
        start = tid * items_per_thread
        step = num_threads * items_per_thread

        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(bins, counter_dtype)
        # temp_storage = coop.shared.array(temp_storage_bytes, np.uint8,
        #                                 alignment=temp_storage_alignment)
        thread_samples = coop.local.array(items_per_thread, item_dtype)

        # histo = block_histogram(temp_storage)
        histo = block_histogram

        # Initialize the histogram.

        # N.B. The C++ `CounterT` template parameter (`counter_dtype` in
        #      Python) is inferred from the `smem_histogram` dtype.
        #       `bins` is inferred from the `smem_histogram.shape`.
        histo.init(smem_histogram)
        cuda.syncthreads()

        index = start
        while index < total_items:
            block_load(
                d_in[index:],
                thread_samples,
            )
            cuda.syncthreads()
            histo.composite(thread_samples, smem_histogram)
            cuda.syncthreads()
            index += step

        cuda.syncthreads()

        # Store the histogram bin counts to the output.
        thread_linear = (
            cuda.threadIdx.x
            + cuda.threadIdx.y * cuda.blockDim.x
            + cuda.threadIdx.z * cuda.blockDim.x * cuda.blockDim.y
        )
        for bin_idx in range(thread_linear, bins, threads_per_block):
            d_out[bin_idx] = smem_histogram[bin_idx]

    h_input = np.random.randint(0, bins, total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(bins, dtype=counter_dtype))
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output)

    expected_histo = np.bincount(h_input, minlength=bins)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(expected_histo) == total_items

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, expected_histo)


# Consolidated from tests/coop/test_histo2.py
@pytest.mark.parametrize("item_dtype", [np.uint8, np.int8])
@pytest.mark.parametrize("counter_dtype", [np.int32, np.uint32])
@pytest.mark.parametrize("threads_per_block", [32, 128, (4, 16), (4, 8, 16)])
@pytest.mark.parametrize("items_per_thread", [2, 4, 8])
@pytest.mark.parametrize(
    "num_total_items",
    [
        1 << 10,  # 1KB
        1 << 12,  # 4KB
        1 << 15,  # 32KB
        1 << 19,  # 512KB
        1 << 23,  # 8MB
        1 << 28,  # 256MB
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        BlockHistogramAlgorithm.ATOMIC,
        BlockHistogramAlgorithm.SORT,
    ],
)
def test_block_histogram_histo_single_phase_2(
    item_dtype,
    counter_dtype,
    threads_per_block,
    items_per_thread,
    num_total_items,
    algorithm,
):
    # Example of what a two-phase histogram instance creation would look like,
    # purely for the purpose of demonstrating how much simpler the single-phase
    # construction is because we can infer so many of the other parameters
    # on the fly.
    histo_two_phase = coop.block.histogram(
        item_dtype=np.int8,
        counter_dtype=np.int32,
        dim=128,
        items_per_thread=4,
        algorithm=BlockHistogramAlgorithm.ATOMIC,
        bins=256,
    )

    # N.B. The example above is still valid; the single-phase functionality
    #      does not remove the ability to construct these primitives outside
    #      of kernels--which you would do if you needed to access temp storage
    #      or alignment ahead of time.
    print(f"temp storage bytes: {histo_two_phase.temp_storage_bytes}")
    print(f"temp storage alignment: {histo_two_phase.temp_storage_alignment}")

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        tid = cuda.grid(1)
        if tid >= num_total_items:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block

        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(
            d_out.shape,
            d_out.dtype,
            alignment=128,
        )
        thread_samples = coop.local.array(
            items_per_thread,
            d_in.dtype,
            alignment=16,
        )

        # Create the "parent" histogram instance.  Note that we only need
        # to provide two parameters--the local register array from which we
        # plan to read, and the corresponding shared-memory bin count array.
        #
        # We can infer all of the other arguments required by the two-phase
        # constructor (see `histo_two_phase` above) automatically, as follows:
        #
        #   item_dtype: inferred from thread_samples.dtype (which was inferred
        #               from d_in.dtype).
        #
        #   counter_dtype: inferred from smem_histogram.dtype (which was inferred
        #                  from d_out.dtype).
        #
        #   dim: inferred from the grid launch dimensions of the current kernel.
        #
        #   items_per_thread: inferred from the thread_samples.shape (which was
        #                     obtained via the items_per_thread kernel param).
        #
        #   algorithm: defaults to ATOMIC.
        #
        #   bins: inferred from smem_histogram.shape (which was inferred from
        #         d_out.shape).
        histo = coop.block.histogram(thread_samples, smem_histogram)

        # Initialize the histogram.  This corresponds to the CUB instance
        # method BlockHistogram::InitHistogram.  It is a "child" call of a
        # parent instance.  (Our shared-memory array also comes pre-zeroed
        # so this technically isn't necessary.)
        histo.init()

        cuda.syncthreads()

        # Loop through tiles of the input data, loading chunks optimally via
        # block load, then updating the histogram accordingly.  To simplify
        # the example, we don't have any corner-case handling for undesirable
        # shapes (e.g. items per block (and thus, block offset) not a perfect
        # multiple of total items, etc.).
        while block_offset < num_total_items:
            coop.block.load(
                d_in[block_offset:],
                thread_samples,
                algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
            )

            # This is the second "child" call against the parent histogram
            # instance, corresponding to the the CUB C++ instance method:
            #
            #   BlockHistogram::Compute(
            #       T& thread_samples[ITEMS_PER_THREAD],
            #       CounterT& histogram[BINS]
            #   )
            #
            # Note that we don't need to furnish the smem_histogram as the
            # second parameter here--that was already provided to the
            # histogram's constructor, and thus, is accessible to us behind
            # the scenes when we need to wire up this composite call.
            histo.composite(thread_samples)

            cuda.syncthreads()

            block_offset += items_per_block * cuda.gridDim.x

        # Final block atomic update to merge our block counts into the user's
        # output results array (d_out).
        for bin_idx in range(cuda.threadIdx.x, bins, threads_per_block):
            cuda.atomic.add(d_out, bin_idx, smem_histogram[bin_idx])

    # Kernel ends.  Remaining code is test scaffolding.

    bins = get_histogram_bins_for_type(item_dtype)

    threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    items_per_block = threads_per_block * items_per_thread
    if num_total_items < items_per_block or num_total_items % items_per_block != 0:
        pytest.skip(
            "BlockHistogram requires full tiles (num_total_items must be a "
            "multiple of items_per_block)."
        )
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    h_input = np.random.randint(0, bins, num_total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    # Ensure deterministic output; device_array is not guaranteed zeroed.
    d_output = cuda.to_device(np.zeros(bins, dtype=counter_dtype))
    num_blocks = blocks_per_grid

    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread, num_total_items)

    actual = d_output.copy_to_host()
    expected = np.bincount(h_input, minlength=bins).astype(counter_dtype)

    # Sanity check sum of histo bins matches total items.
    assert np.sum(actual) == num_total_items
    assert np.sum(expected) == num_total_items

    # Verify arrays match.
    np.testing.assert_array_equal(actual, expected)
