# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

from helpers import (
    row_major_tid,
)

import cuda.cccl.cooperative.experimental as coop

def test_block_histogram_histo_sort_two_phase():

    bins = 256
    item_dtype = np.uint8
    counter_dtype = np.uint32
    items_per_thread = 4
    threads_per_block = 128
    total_items = 1 << 15 # 32KB
    algorithm = coop.BlockHistogramAlgorithm.BLOCK_HISTO_SORT

    block_load = coop.BlockLoad(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )
    block_histogram = coop.block.histogram.create(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        bins=bins,
    )
    block_store = coop.BlockStore(
        item_dtype,
        dim,
        items_per_thread,
    )

    temp_storage_bytes = max((
        block_load.temp_storage_bytes,
        block_store.temp_storage_bytes,
        block_histogram.temp_storage_bytes
    ))
    temp_storage_alignment = max((
        block_load.temp_storage_alignment,
        block_store.temp_storage_alignment,
        block_histogram.temp_storage_alignment,
    ))

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        tid = cuda.grid(1)
        num_threads = cuda.gridsize(1)
        start = tid * items_per_thread
        step = num_threads * items_per_thread

        # Shared per-block histogram bin counts.
        smem_histogram = cuda.shared.array(bins, counter_dtype)
        temp_storage = cuda.shared.array(temp_storage_bytes, np.uint8,
                                         alignment=temp_storage_alignment)
        thread_samples = cuda.local.array(items_per_thread, item_dtype)

        histo = block_histogram(temp_storage)

        # Initialize the histogram.
        histo.init(smem_histogram, counter_dtype)

        index = start
        while index < total_items:
            block_load(
                d_in[index:],
                thread_data,
                items_per_thread,
            )
            cuda.syncthreads()
            histo.composite(thread_samples, smem_histogram)
            index += step

        cuda.syncthreads()

        # Store the histogram bin counts to the output.
        if tid < bins:
            d_out[tid] = smem_histogram[tid]

    h_input = np.random.randint(
        0, bins, total_items, dtype=item_dtype
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=counter_dtype)
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread)

    expected_histo = np.bincount(h_input, minlength=bins)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(expected_histo) == total_items

    np.testing.assert_array_equal(h_output, h_input)


def test_block_histogram_histo_sort_single_phase():

    item_dtype = np.uint8
    counter_dtype = np.uint32
    # Is bins necessary?  Isn't it max `item_dtype`?
    bins = 256
    items_per_thread = 4
    threads_per_block = 128
    total_items = 1 << 15 # 32KB
    algorithm = coop.BlockHistogramAlgorithm.BLOCK_HISTO_SORT

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        # Shared per-block histogram bin counts.
        smem_histogram = cuda.shared.array(bins, counter_dtype)
        thread_samples = cuda.local.array(shape=items_per_thread, item_dtype)

        # Does implicit init().  `item_dtype` inferred from
        # smem_histogram.dtype.  `bins` inferred from `smem_histogram.shape`.
        histo = coop.block.histogram(
            smem_histogram, # CounterT histogram[BINS]
            items_per_thread,
            algorithm,
        )

        # Initialize the histogram.
        histo.init(smem_histogram, dtype)

        stride = cuda.blockDim.x * items_per_thread
        index = row_major_tid()

        # Enumerate the input data and update the histogram composite.
        while index < total_items:
            coop.block.load(
                d_in,
                thread_data,
                items_per_thread,
            )
            cuda.syncwarp()
            histo.composite(thread_samples, smem_histogram)
            index += stride

