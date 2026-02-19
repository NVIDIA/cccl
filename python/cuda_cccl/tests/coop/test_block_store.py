# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import reduce
from operator import mul

import numba
import pytest
from helpers import NUMBA_TYPES_TO_NP, random_int, row_major_tid
from numba import cuda, types

from cuda import coop
from cuda.coop import BlockStoreAlgorithm

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@pytest.mark.parametrize("T", [types.int8, types.int16, types.uint32, types.uint64])
@pytest.mark.parametrize("threads_per_block", [32, 128, 256, (8, 16), (2, 4, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 3])
@pytest.mark.parametrize(
    "algorithm",
    [
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    ],
)
def test_block_store(T, threads_per_block, items_per_thread, algorithm):
    block_store = coop.block.store(T, threads_per_block, items_per_thread, algorithm)
    store_algorithm = BlockStoreAlgorithm[algorithm.upper()]

    num_threads_per_block = (
        threads_per_block
        if type(threads_per_block) is int
        else reduce(mul, threads_per_block)
    )

    if algorithm == "striped":

        @cuda.jit(device=True)
        def input_index(i):
            return row_major_tid() + num_threads_per_block * i
    else:

        @cuda.jit(device=True)
        def input_index(i):
            return row_major_tid() * items_per_thread + i

    @cuda.jit
    def kernel(d_input, d_output_two_phase, d_output_single_phase):
        thread_data_two_phase = cuda.local.array(shape=items_per_thread, dtype=dtype)
        thread_data_single_phase = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            value = d_input[input_index(i)]
            thread_data_two_phase[i] = value
            thread_data_single_phase[i] = value
        block_store(d_output_two_phase, thread_data_two_phase)
        coop.block.store(
            d_output_single_phase,
            thread_data_single_phase,
            items_per_thread=items_per_thread,
            algorithm=store_algorithm,
        )

    dtype = NUMBA_TYPES_TO_NP[T]
    items_per_tile = num_threads_per_block * items_per_thread
    h_input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(h_input)
    d_output_two_phase = cuda.device_array(items_per_tile, dtype=dtype)
    d_output_single_phase = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_per_block](d_input, d_output_two_phase, d_output_single_phase)
    cuda.synchronize()

    output_two_phase = d_output_two_phase.copy_to_host()
    output_single_phase = d_output_single_phase.copy_to_host()
    reference = h_input
    for i in range(items_per_tile):
        assert output_two_phase[i] == reference[i]
        assert output_single_phase[i] == reference[i]
        assert output_two_phase[i] == output_single_phase[i]

    sig = (T[::1], T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass
