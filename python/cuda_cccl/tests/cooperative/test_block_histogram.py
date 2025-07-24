# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import reduce
from operator import mul

import numba
import numpy as np
import pytest
from helpers import (
    NUMBA_TYPES_TO_NP,
    random_int,
)
from numba import cuda, types

import cuda.cccl.cooperative.experimental as coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@pytest.mark.parametrize("item_dtype", [types.int8, types.uint8])
@pytest.mark.parametrize("counter_dtype", [types.uint32, types.uint64])
@pytest.mark.parametrize("threads_per_block", [128, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "algorithm",
    [
        coop.block.BlockHistogramAlgorithm.ATOMIC,
        coop.block.BlockHistogramAlgorithm.SORT,
    ],
)
def test_block_histogram(
    item_dtype, counter_dtype, threads_per_block, items_per_thread, algorithm
):
    bins = 1 << item_dtype.bitwidth

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    block_load = coop.block.load(
        item_dtype,
        threads_per_block,
        items_per_thread,
    )

    block_histogram = coop.block.histogram(
        item_dtype=item_dtype,
        counter_dtype=counter_dtype,
        dim=threads_per_block,
        items_per_thread=items_per_thread,
        bins=bins,
        algorithm=algorithm,
    )

    files = block_load.files + block_histogram.files

    @cuda.jit(link=files)
    def kernel(d_in, d_out):
        tid = cuda.grid(1)

        smem_histogram = cuda.shared.array(bins, dtype=counter_dtype)
        thread_samples = cuda.local.array(items_per_thread, dtype=item_dtype)

        block_load(d_in, thread_samples)

        block_histogram(thread_samples, smem_histogram)

        cuda.syncthreads()

        if tid < bins:
            d_out[tid] = smem_histogram[tid]

    item_dtype_np = NUMBA_TYPES_TO_NP[item_dtype]
    counter_dtype_np = NUMBA_TYPES_TO_NP[counter_dtype]

    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, dtype=item_dtype_np)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=counter_dtype_np)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    # Calculate reference histogram.
    expected = np.bincount(h_input, minlength=bins).astype(counter_dtype_np)
    assert np.sum(expected) == total_items

    actual = d_output.copy_to_host()
    np.testing.assert_array_equal(actual, expected)
