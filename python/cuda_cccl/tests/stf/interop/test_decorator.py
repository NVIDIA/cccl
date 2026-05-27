# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

numba = pytest.importorskip("numba")
pytest.importorskip("numba.cuda")
from numba import cuda  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop.numba import jit  # noqa: E402

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@jit
def axpy(a, x, y):
    i = cuda.grid(1)
    if i < x.size:
        y[i] = a * x[i] + y[i]


@jit
def scale(a, x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = a * x[i]


@pytest.mark.parametrize("use_graph", [True, False])
def test_decorator(use_graph):
    X, Y, Z = (np.ones(16, np.float32) for _ in range(3))

    ctx = stf.context(use_graph=use_graph)
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    scale[32, 64](2.0, lX.rw())
    axpy[32, 64](2.0, lX.read(), lY.rw())
    axpy[32, 64, stf.exec_place.device(0)](
        2.0, lX.read(), lZ.rw()
    )  # explicit exec place
    axpy[32, 64](
        2.0, lY.read(), lZ.rw(stf.data_place.device(0))
    )  # per-dep placement override

    ctx.finalize()

    assert np.allclose(X, 2.0)
    assert np.allclose(Y, 5.0)
    assert np.allclose(Z, 15.0)


if __name__ == "__main__":
    test_decorator(False)
    test_decorator(True)
