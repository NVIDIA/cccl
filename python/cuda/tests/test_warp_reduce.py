# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pynvjitlink import patch
import cuda.cooperative.experimental as cudax
from numba.core.extending import (lower_builtin, make_attribute_wrapper,
                                  models, register_model, type_callable,
                                  typeof_impl)
from numba.core import cgutils
import numpy as np
from helpers import random_int, NUMBA_TYPES_TO_NP
import pytest
from numba import cuda, types
import numba
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


patch.patch_numba_linker(lto=True)


@pytest.mark.parametrize('T', [types.uint32, types.uint64])
def test_warp_reduction_of_integral_type(T):
    def op(a, b):
        return a if a < b else b

    warp_reduce = cudax.warp.reduce(T, op)
    temp_storage_bytes = warp_reduce.temp_storage_bytes

    @cuda.jit(link=warp_reduce.files)
    def kernel(input, output):
        temp_storage = cuda.shared.array(
            shape=temp_storage_bytes, dtype='uint8')
        warp_output = warp_reduce(temp_storage, input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = warp_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.min(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass


@pytest.mark.parametrize('T', [types.uint32, types.uint64])
def test_warp_sum(T):
    warp_reduce = cudax.warp.sum(T)
    temp_storage_bytes = warp_reduce.temp_storage_bytes

    @cuda.jit(link=warp_reduce.files)
    def kernel(input, output):
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype='uint8')
        warp_output = warp_reduce(temp_storage, input[cuda.threadIdx.x])

        if cuda.threadIdx.x == 0:
            output[0] = warp_output

    dtype = NUMBA_TYPES_TO_NP[T]
    h_input = random_int(32, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=dtype)
    kernel[1, 32](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_expected = np.sum(h_input)

    assert h_output[0] == h_expected

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass
