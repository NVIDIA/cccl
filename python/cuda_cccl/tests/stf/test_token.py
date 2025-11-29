# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
from numba import cuda

import cuda.stf as stf

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def test_token():
    ctx = stf.context()
    lX = ctx.token()
    lY = ctx.token()
    lZ = ctx.token()

    with ctx.task(lX.rw()):
        pass

    with ctx.task(lX.read(), lY.rw()):
        pass

    with ctx.task(lX.read(), lZ.rw()):
        pass

    with ctx.task(lY.read(), lZ.rw()):
        pass

    ctx.finalize()


@cuda.jit
def axpy(a, x, y):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.size, stride):
        y[i] = a * x[i] + y[i]


def test_numba_token():
    n = 1024 * 1024
    X = np.ones(n, dtype=np.float32)
    Y = np.ones(n, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    token = ctx.token()

    # Use a reasonable grid size - kernel loop will handle all elements
    blocks = 32
    threads_per_block = 256

    with ctx.task(lX.read(), lY.rw(), token.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = t.get_arg_numba(0)
        dY = t.get_arg_numba(1)
        axpy[blocks, threads_per_block, nb_stream](2.0, dX, dY)

    with ctx.task(lX.read(), lY.rw(), token.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        print(nb_stream)
        dX, dY = t.numba_arguments()
        axpy[blocks, threads_per_block, nb_stream](2.0, dX, dY)

    ctx.finalize()

    # Sanity checks: verify the results after finalize
    # First task: Y = 2.0 * X + Y = 2.0 * 1.0 + 1.0 = 3.0
    # Second task: Y = 2.0 * X + Y = 2.0 * 1.0 + 3.0 = 5.0
    assert np.allclose(X, 1.0), f"X should still be 1.0 (read-only), but got {X[0]}"
    assert np.allclose(Y, 5.0), (
        f"Y should be 5.0 after two axpy operations, but got {Y[0]}"
    )


if __name__ == "__main__":
    test_token()
