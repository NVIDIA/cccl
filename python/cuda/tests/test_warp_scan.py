# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pynvjitlink import patch
import cuda.cooperative.experimental as cudax
import numba
from helpers import random_int, NUMBA_TYPES_TO_NP
import numpy as np
import pytest
from numba import cuda, types
import numba

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


patch.patch_numba_linker(lto=True)


@pytest.mark.parametrize('T', [types.uint32, types.uint64])
def test_warp_exclusive_sum(T):
    warp_exclusive_sum = cudax.warp.exclusive_sum(dtype=T)
    temp_storage_bytes = warp_exclusive_sum.temp_storage_bytes

    @cuda.jit(link=warp_exclusive_sum.files)
    def kernel(input, output):
        tid = cuda.threadIdx.x
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype='uint8')
        output[tid] = warp_exclusive_sum(temp_storage, input[tid])

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(32, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = np.cumsum(h_input) - h_input
    for i in range(32):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass
