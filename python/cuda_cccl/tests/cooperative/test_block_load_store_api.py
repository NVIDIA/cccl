# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin imports
import numba
import numpy as np
from numba import cuda
from pynvjitlink import patch

import cuda.cccl.cooperative.experimental as cudax

patch.patch_numba_linker(lto=True)
# example-end imports

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_block_load_store():
    # example-begin load_store
    threads_per_block = 32
    items_per_thread = 4
    block_load = cudax.block.load(
        numba.int32, threads_per_block, items_per_thread, "striped"
    )
    block_store = cudax.block.store(
        numba.int32, threads_per_block, items_per_thread, "striped"
    )

    @cuda.jit(link=block_load.files + block_store.files)
    def kernel(input, output):
        tmp = cuda.local.array(items_per_thread, numba.int32)
        block_load(input, tmp)
        block_store(output, tmp)

    # example-end load_store

    h_input = np.random.randint(
        0, 42, threads_per_block * items_per_thread, dtype=np.int32
    )
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_per_block](d_input, d_output)
    h_output = d_output.copy_to_host()

    np.testing.assert_allclose(h_output, h_input)
