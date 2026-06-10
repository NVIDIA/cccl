# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import math

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402

try:
    from cuda.core import Program

    _HAS_CUDA_CORE = True
except ImportError:
    _HAS_CUDA_CORE = False

pytestmark = pytest.mark.skipif(not _HAS_CUDA_CORE, reason="cuda.core not available")

AXPY_SOURCE = r"""
extern "C" __global__
void axpy(int n, double alpha, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += alpha * x[i];
    }
}
"""


def _compile_axpy():
    prog = Program(AXPY_SOURCE, "c++")
    mod = prog.compile("cubin")
    return mod.get_kernel("axpy")


def test_cuda_kernel_axpy():
    """AXPY via cuda_kernel: Y = Y + alpha * X, verify after finalize."""
    N = 1024
    alpha = 3.14
    X = np.array([math.sin(i) for i in range(N)], dtype=np.float64)
    Y = np.array([math.cos(i) for i in range(N)], dtype=np.float64)
    Y_expected = Y + alpha * X

    kernel = _compile_axpy()

    ctx = stf.context()
    lX = ctx.logical_data(X, name="X")
    lY = ctx.logical_data(Y, name="Y")

    with ctx.cuda_kernel(lX.read(), lY.rw(), symbol="axpy") as k:
        dX = k.get_arg(0)
        dY = k.get_arg(1)
        k.launch(
            kernel,
            grid=((N + 255) // 256,),
            block=(256,),
            args=[ctypes.c_int(N), ctypes.c_double(alpha), dX, dY],
        )

    ctx.finalize()

    np.testing.assert_allclose(Y, Y_expected, rtol=1e-12)


def test_cuda_kernel_axpy_graph():
    """AXPY via cuda_kernel with graph backend."""
    N = 1024
    alpha = 2.0
    X = np.ones(N, dtype=np.float64) * 3.0
    Y = np.ones(N, dtype=np.float64) * 5.0
    Y_expected = Y + alpha * X

    kernel = _compile_axpy()

    ctx = stf.context(use_graph=True)
    lX = ctx.logical_data(X, name="X")
    lY = ctx.logical_data(Y, name="Y")

    with ctx.cuda_kernel(lX.read(), lY.rw(), symbol="axpy") as k:
        dX = k.get_arg(0)
        dY = k.get_arg(1)
        k.launch(
            kernel,
            grid=((N + 255) // 256,),
            block=(256,),
            args=[ctypes.c_int(N), ctypes.c_double(alpha), dX, dY],
        )

    ctx.finalize()

    np.testing.assert_allclose(Y, Y_expected, rtol=1e-12)


def test_cuda_kernel_chained():
    """Two chained cuda_kernel tasks on the same data."""
    N = 512
    X = np.ones(N, dtype=np.float64)
    Y = np.zeros(N, dtype=np.float64)

    kernel = _compile_axpy()

    ctx = stf.context()
    lX = ctx.logical_data(X, name="X")
    lY = ctx.logical_data(Y, name="Y")

    for alpha in [1.0, 2.0]:
        with ctx.cuda_kernel(lX.read(), lY.rw()) as k:
            dX = k.get_arg(0)
            dY = k.get_arg(1)
            k.launch(
                kernel,
                grid=((N + 255) // 256,),
                block=(256,),
                args=[ctypes.c_int(N), ctypes.c_double(alpha), dX, dY],
            )

    ctx.finalize()

    np.testing.assert_allclose(Y, 3.0 * np.ones(N), rtol=1e-12)


def test_cuda_kernel_raw_handle():
    """Accept a raw CUfunction handle (int) instead of cuda.core.Kernel."""
    N = 256
    X = np.ones(N, dtype=np.float64)
    Y = np.zeros(N, dtype=np.float64)

    kernel = _compile_axpy()
    raw_handle = int(kernel._handle)

    ctx = stf.context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)

    with ctx.cuda_kernel(lX.read(), lY.rw()) as k:
        dX = k.get_arg(0)
        dY = k.get_arg(1)
        k.launch(
            raw_handle,
            grid=(1,),
            block=(256,),
            args=[ctypes.c_int(N), ctypes.c_double(1.0), dX, dY],
        )

    ctx.finalize()

    np.testing.assert_allclose(Y, X, rtol=1e-12)


def test_cuda_kernel_loop_accumulate():
    """Y += i * X in a loop for i in 0..K-1, verifying per-iteration scalar lifetime.

    Each iteration creates a fresh ParamHolder with a different alpha=i.
    If argument storage is not kept alive correctly, STF would see stale
    values and the final sum would be wrong.
    Expected result: Y_final = Y_init + sum(0..K-1) * X = 0 + K*(K-1)/2 * 1.
    """
    N = 256
    K = 50
    X = np.ones(N, dtype=np.float64)
    Y = np.zeros(N, dtype=np.float64)

    kernel = _compile_axpy()

    ctx = stf.context()
    lX = ctx.logical_data(X, name="X")
    lY = ctx.logical_data(Y, name="Y")

    for i in range(K):
        with ctx.cuda_kernel(lX.read(), lY.rw()) as k:
            dX = k.get_arg(0)
            dY = k.get_arg(1)
            k.launch(
                kernel,
                grid=(1,),
                block=(256,),
                args=[ctypes.c_int(N), ctypes.c_double(float(i)), dX, dY],
            )

    ctx.finalize()

    expected = float(K * (K - 1) // 2)
    np.testing.assert_allclose(Y, expected * np.ones(N), rtol=1e-12)


def test_cuda_kernel_multi_launch():
    """Two launch() calls inside a single cuda_kernel task.

    The cuda_kernel task accumulates kernel descriptors, so calling
    launch() twice should execute both kernels with the correct args.
    Y = Y + alpha1*X + alpha2*X = 0 + 1*1 + 2*1 = 3.
    """
    N = 256
    X = np.ones(N, dtype=np.float64)
    Y = np.zeros(N, dtype=np.float64)

    kernel = _compile_axpy()

    ctx = stf.context()
    lX = ctx.logical_data(X, name="X")
    lY = ctx.logical_data(Y, name="Y")

    with ctx.cuda_kernel(lX.read(), lY.rw()) as k:
        dX = k.get_arg(0)
        dY = k.get_arg(1)
        k.launch(
            kernel,
            grid=(1,),
            block=(256,),
            args=[ctypes.c_int(N), ctypes.c_double(1.0), dX, dY],
        )
        k.launch(
            kernel,
            grid=(1,),
            block=(256,),
            args=[ctypes.c_int(N), ctypes.c_double(2.0), dX, dY],
        )

    ctx.finalize()

    np.testing.assert_allclose(Y, 3.0 * np.ones(N), rtol=1e-12)


if __name__ == "__main__":
    test_cuda_kernel_axpy()
