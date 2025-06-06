# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda
from pynvjitlink import patch

import cuda.cccl.cooperative.experimental as cudax

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-begin imports
patch.patch_numba_linker(lto=True)
# example-end imports


def test_warp_exclusive_sum():
    # example-begin exclusive-sum
    # Specialize exclusive sum for a warp of threads
    warp_exclusive_sum = cudax.warp.exclusive_sum(numba.int32)

    # Link the exclusive sum to a CUDA kernel
    @cuda.jit(link=warp_exclusive_sum.files)
    def kernel(data):
        # Collectively compute the warp-wide exclusive prefix sum
        data[cuda.threadIdx.x] = warp_exclusive_sum(data[cuda.threadIdx.x])

    # example-end exclusive-sum

    tile_size = 32

    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, 32](d_keys)
    h_keys = d_keys.copy_to_host()
    for i in range(tile_size):
        assert h_keys[i] == i


test_warp_exclusive_sum()
