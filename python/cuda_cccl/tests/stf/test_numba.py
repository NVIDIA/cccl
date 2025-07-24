# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from cuda.cccl.experimental.stf._stf_bindings_impl import logical_data, context, AccessMode, read, rw, write
import ctypes
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import driver, devicearray

@cuda.jit
def axpy(a, x, y):
    i = cuda.grid(1)
    if i < x.size:
        y[i] = a * x[i] + y[i]

@cuda.jit
def scale(a, x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = a * x[i]

def test_numba():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Y)

    with ctx.task(rw(lX)) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        # dX = t.get_arg_numba(0)
        dX = cuda.from_cuda_array_interface(t.get_arg_cai(0), owner=None, sync=False)
        scale[32, 64, nb_stream](2.0, dX)
        pass

    with ctx.task(read(lX), rw(lY)) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        print(nb_stream)
        dX = t.get_arg_numba(0)
        dY = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dX, dY)
        pass

    with ctx.task(read(lX), rw(lZ)) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = t.get_arg_numba(0)
        dZ = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dX, dZ)
        pass

    with ctx.task(read(lY), rw(lZ)) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dY = t.get_arg_numba(0)
        dZ = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dY, dZ)
        pass

    del ctx

if __name__ == "__main__":
    print("Running CUDASTF examples...")
    test_numba()
