# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pynvjitlink import patch
import numpy as np
import cuda.cooperative.experimental as cudax
from helpers import random_int, NUMBA_TYPES_TO_NP
import pytest
from numba import cuda, types
import numba
patch.patch_numba_linker(lto=True)
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@pytest.mark.parametrize('T', [types.int8, types.int16, types.uint32, types.uint64])
@pytest.mark.parametrize('items_per_thread', [1, 3])
def test_warp_merge_sort(T, items_per_thread):
    def op(a, b):
        return a < b

    warp_merge_sort = cudax.warp.merge_sort_keys(
        T, items_per_thread, op)
    temp_storage_bytes = warp_merge_sort.temp_storage_bytes

    @cuda.jit(link=warp_merge_sort.files)
    def kernel(input, output):
        tid = cuda.threadIdx.x
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype='uint8')
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        warp_merge_sort(temp_storage, thread_data)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    items_per_tile = 32 * items_per_thread
    input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = sorted(input)
    for i in range(items_per_tile):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass
