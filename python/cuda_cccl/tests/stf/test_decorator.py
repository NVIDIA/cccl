import numpy as np

import numba
from numba import cuda

numba.config.CUDA_ENABLE_PYNVJITLINK = 1
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

import cuda.cccl.experimental.stf as cudastf

@cudastf.jit
def axpy(a, x, y):
    i = cuda.grid(1)
    if i < x.size:
        y[i] = a * x[i] + y[i]


@cudastf.jit
def scale(a, x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = a * x[i]


X, Y, Z = (np.ones(16, np.float32) for _ in range(3))

ctx = cudastf.context()
lX = ctx.logical_data(X)
lY = ctx.logical_data(Y)
lZ = ctx.logical_data(Z)

scale[32, 64, ctx](2.0, lX.rw())
axpy[32, 64, ctx](2.0, lX.read(), lY.rw())  # default device
axpy[32, 64, ctx, cudastf.exec_place.device(0)](
    2.0, lX.read(), lZ.rw()
)  # explicit exec place
axpy[32, 64, ctx](
    2.0, lY.read(), lZ.rw(cudastf.data_place.device(0))
)  # per-dep placement override
