# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import pytest
from helpers import NUMBA_TYPES_TO_NP, random_int
from numba import cuda, types
from pynvjitlink import patch

import cuda.cooperative.experimental as cudax

patch.patch_numba_linker(lto=True)
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@pytest.mark.parametrize("T", [types.int8, types.int16, types.uint32, types.uint64])
@pytest.mark.parametrize("threads_in_block", [32, 128, 256])
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
def test_block_store(T, threads_in_block, items_per_thread, algorithm):
    block_store = cudax.block.store(T, threads_in_block, items_per_thread, algorithm)
    temp_storage_bytes = block_store.temp_storage_bytes

    if algorithm == "striped":

        @cuda.jit(device=True)
        def input_index(i):
            return cuda.threadIdx.x + threads_in_block * i
    else:

        @cuda.jit(device=True)
        def input_index(i):
            return cuda.threadIdx.x * items_per_thread + i

    @cuda.jit(link=block_store.files)
    def kernel(d_input, d_output):
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = d_input[input_index(i)]
        block_store(temp_storage, d_output, thread_data)

    dtype = NUMBA_TYPES_TO_NP[T]
    items_per_tile = threads_in_block * items_per_thread
    h_input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_in_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = h_input
    for i in range(items_per_tile):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass
