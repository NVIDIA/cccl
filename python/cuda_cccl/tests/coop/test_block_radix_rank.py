# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import reduce
from operator import mul

import numba
import numpy as np
from helpers import row_major_tid
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


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


def _run_radix_rank_test(
    threads_per_block, items_per_thread, begin_bit, end_bit, descending
):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )
    total_items = num_threads_per_block * items_per_thread

    @cuda.jit
    def kernel(d_in, d_out):
        tid = row_major_tid()
        items = cuda.local.array(items_per_thread, numba.uint32)
        ranks = cuda.local.array(items_per_thread, numba.int32)
        for i in range(items_per_thread):
            items[i] = d_in[tid * items_per_thread + i]
        coop.block.radix_rank(
            items,
            ranks,
            items_per_thread,
            begin_bit,
            end_bit,
            descending,
        )
        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = ranks[i]

    h_input = np.random.randint(0, 2**16, total_items, dtype=np.uint32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=np.int32)

    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    _validate_ranks(h_input, h_output, begin_bit, end_bit, descending)


def _run_radix_rank_two_phase_test(
    threads_per_block, items_per_thread, begin_bit, end_bit, descending
):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )
    total_items = num_threads_per_block * items_per_thread

    radix_rank = coop.block.radix_rank(
        numba.uint32,
        threads_per_block,
        items_per_thread,
        begin_bit,
        end_bit,
        descending=descending,
    )

    @cuda.jit
    def kernel(d_in, d_out):
        tid = row_major_tid()
        items = cuda.local.array(items_per_thread, numba.uint32)
        ranks = cuda.local.array(items_per_thread, numba.int32)
        for i in range(items_per_thread):
            items[i] = d_in[tid * items_per_thread + i]
        radix_rank(
            items,
            ranks,
            items_per_thread,
            begin_bit,
            end_bit,
            descending,
        )
        for i in range(items_per_thread):
            d_out[tid * items_per_thread + i] = ranks[i]

    h_input = np.random.randint(0, 2**16, total_items, dtype=np.uint32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=np.int32)

    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    _validate_ranks(h_input, h_output, begin_bit, end_bit, descending)


def test_block_radix_rank_ascending():
    _run_radix_rank_test(
        threads_per_block=128,
        items_per_thread=4,
        begin_bit=0,
        end_bit=5,
        descending=False,
    )


def test_block_radix_rank_descending():
    _run_radix_rank_test(
        threads_per_block=64,
        items_per_thread=3,
        begin_bit=2,
        end_bit=6,
        descending=True,
    )


def test_block_radix_rank_two_phase_ascending():
    _run_radix_rank_two_phase_test(
        threads_per_block=64,
        items_per_thread=2,
        begin_bit=0,
        end_bit=5,
        descending=False,
    )


def test_block_radix_rank_two_phase_descending():
    _run_radix_rank_two_phase_test(
        threads_per_block=32,
        items_per_thread=3,
        begin_bit=1,
        end_bit=6,
        descending=True,
    )
