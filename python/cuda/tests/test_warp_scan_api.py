# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import numba
from numba import cuda
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# example-begin imports
import cuda.cooperative.experimental as cudax
from pynvjitlink import patch
patch.patch_numba_linker(lto=True)
# example-end imports


def test_warp_exclusive_sum():
    # example-begin exclusive-sum
    # Specialize exclusive sum for a warp of threads
    warp_exclusive_sum = cudax.warp.exclusive_sum(numba.int32)
    temp_storage_bytes = warp_exclusive_sum.temp_storage_bytes

    # Link the exclusive sum to a CUDA kernel
    @cuda.jit(link=warp_exclusive_sum.files)
    def kernel(data):
        # Allocate shared memory for exclusive sum
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)

        # Collectively compute the warp-wide exclusive prefix sum
        data[cuda.threadIdx.x] = warp_exclusive_sum(temp_storage, data[cuda.threadIdx.x])
    # example-end exclusive-sum

    tile_size = 32

    h_keys = np.ones(tile_size, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    kernel[1, 32](d_keys)
    h_keys = d_keys.copy_to_host()
    for i in range(tile_size):
        assert h_keys[i] == i
