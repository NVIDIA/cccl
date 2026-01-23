# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import numba
import numpy as np
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-end imports


def _validate_ranks(keys, ranks, begin_bit, end_bit, descending):
    radix_bits = end_bit - begin_bit
    mask = (1 << radix_bits) - 1
    digits = (keys >> begin_bit) & mask
    max_digit = 1 << radix_bits

    counts = np.zeros(max_digit, dtype=np.int32)
    for digit in digits:
        counts[int(digit)] += 1

    offsets = {}
    prefix = 0
    if descending:
        digit_iter = range(max_digit - 1, -1, -1)
    else:
        digit_iter = range(max_digit)
    for digit in digit_iter:
        offsets[digit] = prefix
        prefix += int(counts[digit])

    sorted_ranks = np.sort(ranks)
    np.testing.assert_array_equal(
        sorted_ranks, np.arange(len(ranks), dtype=ranks.dtype)
    )

    for idx, rank in enumerate(ranks):
        digit = int(digits[idx])
        start = offsets[digit]
        end = start + int(counts[digit])
        assert start <= rank < end


def test_block_radix_rank():
    # example-begin radix-rank
    threads_per_block = 64
    items_per_thread = 1
    begin_bit = 0
    end_bit = 4

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        items = cuda.local.array(items_per_thread, numba.uint32)
        ranks = cuda.local.array(items_per_thread, numba.int32)
        items[0] = d_in[tid]
        coop.block.radix_rank(
            items,
            ranks,
            items_per_thread,
            begin_bit,
            end_bit,
            False,
        )
        d_out[tid] = ranks[0]

    # example-end radix-rank

    h_input = np.random.randint(0, 2**16, threads_per_block, dtype=np.uint32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(threads_per_block, dtype=np.int32)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    _validate_ranks(h_input, h_output, begin_bit, end_bit, False)
