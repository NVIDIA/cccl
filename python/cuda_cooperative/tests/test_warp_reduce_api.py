# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda
from pynvjitlink import patch

import cuda.cooperative.experimental as cudax

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-begin imports
patch.patch_numba_linker(lto=True)
# example-end imports


def test_warp_reduction():
    def op(a, b):
        return a if a > b else b

    # example-begin reduce
    warp_reduce = cudax.warp.reduce(numba.int32, op)

    @cuda.jit(link=warp_reduce.files)
    def kernel(input, output):
        warp_output = warp_reduce(input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = warp_output

    # example-end reduce

    h_input = np.random.randint(0, 42, 32, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, 32](d_input, d_output)
    h_output = d_output.copy_to_host()
    h_expected = np.max(h_input)

    assert h_output[0] == h_expected


def test_warp_sum():
    # example-begin sum
    warp_sum = cudax.warp.sum(numba.int32)

    @cuda.jit(link=warp_sum.files)
    def kernel(input, output):
        warp_output = warp_sum(input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = warp_output

    # example-end sum

    h_input = np.ones(32, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)
    kernel[1, 32](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == 32
