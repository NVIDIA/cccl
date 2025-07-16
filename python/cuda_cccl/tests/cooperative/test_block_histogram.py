# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from helpers import (
    row_major_tid,
)
from numba import cuda

import cuda.cccl.cooperative.experimental as coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_histogram_histo_sort_two_phase0():
    bins = 256
    item_dtype = np.uint8
    counter_dtype = np.uint32
    items_per_thread = 4
    threads_per_block = 128
    total_items = 1 << 15  # 32KB
    algorithm = coop.BlockHistogramAlgorithm.SORT

    block_load = coop.block.load(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )
    # block_histogram = coop.block.histogram(
    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        # counter_dtype=counter_dtype,
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
        tid = cuda.grid(1)

        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(bins, counter_dtype)
        # smem_histogram = cuda.shared.array(bins, counter_dtype)
        # temp_storage = coop.shared.array(temp_storage_bytes, np.uint8,
        #                                 alignment=temp_storage_alignment)
        # thread_samples = cuda.local.array(items_per_thread, item_dtype)
        thread_samples = coop.local.array(items_per_thread, item_dtype)

        # histo = block_histogram(temp_storage)
        histo = bl0ck_histogram()

        # Initialize the histogram.

        # N.B. The C++ `CounterT` template parameter (`counter_dtype` in
        #      Python) is inferred from the `smem_histogram` dtype.
        #       `bins` is inferred from the `smem_histogram.shape`.
        histo.init(smem_histogram)

        block_load(d_in, thread_samples, items_per_thread)

        histo.composite(thread_samples, smem_histogram)

        cuda.syncthreads()

        # Store the histogram bin counts to the output.
        if tid < bins:
            d_out[tid] = smem_histogram[tid]

    h_input = np.random.randint(0, bins, total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=counter_dtype)
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread)

    expected_histo = np.bincount(h_input, minlength=bins)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(expected_histo) == total_items

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_input)


def test_block_histogram_histo_sort_two_phase1():
    bins = 256
    item_dtype = np.uint8
    counter_dtype = np.uint32
    items_per_thread = 4
    threads_per_block = 128
    total_items = 1 << 15  # 32KB
    algorithm = coop.BlockHistogramAlgorithm.SORT

    block_load = coop.block.load(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )
    # block_histogram = coop.block.histogram(
    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        # counter_dtype=counter_dtype,
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
    def kernel(d_in, d_out, items_per_thread):
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
        histo = block_histogram()

        # Initialize the histogram.

        # N.B. The C++ `CounterT` template parameter (`counter_dtype` in
        #      Python) is inferred from the `smem_histogram` dtype.
        #       `bins` is inferred from the `smem_histogram.shape`.
        histo.init(smem_histogram)

        index = start
        while index < total_items:
            block_load(
                d_in[index:],
                thread_samples,
                items_per_thread,
            )
            cuda.syncthreads()
            histo.composite(thread_samples, smem_histogram)
            index += step

        cuda.syncthreads()

        # Store the histogram bin counts to the output.
        if tid < bins:
            d_out[tid] = smem_histogram[tid]

    h_input = np.random.randint(0, bins, total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=counter_dtype)
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread)

    expected_histo = np.bincount(h_input, minlength=bins)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(expected_histo) == total_items

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_input)


def test_block_histogram_histo_sort_two_phase2():
    bins = 256
    item_dtype = np.uint8
    counter_dtype = np.uint32
    items_per_thread = 4
    threads_per_block = 128
    total_items = 1 << 15  # 32KB
    algorithm = coop.BlockHistogramAlgorithm.SORT

    block_load = coop.block.load(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )
    # block_histogram = coop.block.histogram(
    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        # counter_dtype=counter_dtype,
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

    histo = block_histogram

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
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
        # histo = block_histogram()

        # Initialize the histogram.

        # N.B. The C++ `CounterT` template parameter (`counter_dtype` in
        #      Python) is inferred from the `smem_histogram` dtype.
        #       `bins` is inferred from the `smem_histogram.shape`.
        histo.init(smem_histogram)

        index = start
        while index < total_items:
            block_load(
                d_in[index:],
                thread_samples,
                items_per_thread,
            )
            cuda.syncthreads()
            histo.composite(thread_samples, smem_histogram)
            index += step

        cuda.syncthreads()

        # Store the histogram bin counts to the output.
        if tid < bins:
            d_out[tid] = smem_histogram[tid]

    h_input = np.random.randint(0, bins, total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=counter_dtype)
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread)

    expected_histo = np.bincount(h_input, minlength=bins)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(expected_histo) == total_items

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_input)


def test_block_histogram_histo_sort_single_phase():
    item_dtype = np.uint8
    counter_dtype = np.uint32
    # Is bins necessary?  Isn't it max `item_dtype`?
    bins = 256
    items_per_thread = 4
    threads_per_block = 128
    total_items = 1 << 15  # 32KB
    algorithm = coop.BlockHistogramAlgorithm.SORT

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(bins, counter_dtype)
        thread_samples = coop.local.array(shape=items_per_thread, dtype=item_dtype)

        # Does implicit init().  `item_dtype` inferred from
        # smem_histogram.dtype.  `bins` inferred from `smem_histogram.shape`.
        histo = coop.block.histogram(
            thread_samples.dtype,
            smem_histogram,  # CounterT histogram[BINS]
            items_per_thread,
            algorithm,
        )

        # Initialize the histogram.
        # histo.init(smem_histogram)

        stride = cuda.blockDim.x * items_per_thread
        index = row_major_tid()

        # Enumerate the input data and update the histogram composite.
        while index < total_items:
            coop.block.load(
                d_in,
                thread_samples,
                items_per_thread,
            )
            cuda.syncwarp()
            histo.composite(thread_samples, smem_histogram)
            index += stride

    h_input = np.random.randint(0, bins, total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=counter_dtype)
    num_blocks = 1
    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread)

    expected_histo = np.bincount(h_input, minlength=bins)
    # Sanity check sum of histo bins matches total items.
    assert np.sum(expected_histo) == total_items

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_input)
