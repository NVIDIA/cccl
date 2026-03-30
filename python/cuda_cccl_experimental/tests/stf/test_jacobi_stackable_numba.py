# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Jacobi iteration with stackable context and while_loop — Numba version.
Python equivalent of cudax/examples/stf/jacobi_stackable_raii.cu

Requires CUDA 12.4+ for conditional graph nodes.
"""

import numba
import numpy as np
from numba import cuda
from numba_helpers import get_arg_numba, numba_arguments

import cuda.stf as stf

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@cuda.jit
def init_kernel(A, Anew, m, n):
    i, j = cuda.grid(2)
    if i < m and j < n:
        if i == j:
            A[i, j] = 1.0
        else:
            A[i, j] = -1.0
        Anew[i, j] = A[i, j]


@cuda.jit
def reset_residual(residual):
    residual[0] = 0.0


@cuda.jit
def jacobi_step(A, Anew, residual, m, n):
    """Compute Anew from 4-neighbors of A (interior only), reduce max error."""
    i, j = cuda.grid(2)
    if i <= 0 or i >= m - 1 or j <= 0 or j >= n - 1:
        return

    Anew[i, j] = 0.25 * (A[i - 1, j] + A[i + 1, j] + A[i, j - 1] + A[i, j + 1])
    error = abs(A[i, j] - Anew[i, j])
    cuda.atomic.max(residual, 0, error)


@cuda.jit
def copy_back(A, Anew, m, n):
    """Copy Anew -> A for interior points."""
    i, j = cuda.grid(2)
    if i > 0 and i < m - 1 and j > 0 and j < n - 1:
        A[i, j] = Anew[i, j]


def test_jacobi_stackable_numba():
    m, n = 256, 256
    tol = 0.1

    A_host = np.zeros((m, n), dtype=np.float64)
    Anew_host = np.zeros((m, n), dtype=np.float64)

    ctx = stf.stackable_context()

    lA = ctx.logical_data(A_host, name="A")
    lAnew = ctx.logical_data(Anew_host, name="Anew")
    lresidual = ctx.logical_data_empty((1,), np.float64, name="residual")

    threads = (16, 16)
    blocks = ((m + 15) // 16, (n + 15) // 16)

    # Initialize
    with ctx.task(lA.write(), lAnew.write()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dA, dAnew = numba_arguments(t)
        init_kernel[blocks, threads, nb_stream](dA, dAnew, m, n)

    # Iterative solve with while loop
    with ctx.while_loop() as loop:
        # Reset residual to 0
        with ctx.task(lresidual.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dres = numba_arguments(t)
            reset_residual[1, 1, nb_stream](dres)

        # Jacobi step: compute Anew, reduce max error into residual
        with ctx.task(lA.read(), lAnew.write(), lresidual.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dA = get_arg_numba(t, 0)
            dAnew = get_arg_numba(t, 1)
            dres = get_arg_numba(t, 2)
            jacobi_step[blocks, threads, nb_stream](dA, dAnew, dres, m, n)

        # Copy Anew -> A
        with ctx.task(lA.rw(), lAnew.read()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dA = get_arg_numba(t, 0)
            dAnew = get_arg_numba(t, 1)
            copy_back[blocks, threads, nb_stream](dA, dAnew, m, n)

        # Continue while residual > tolerance
        loop.continue_while(lresidual, ">", tol)

    ctx.finalize()

    print(f"Jacobi converged (Numba) with tolerance {tol}")


@cuda.jit
def scale_kernel(x, alpha):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = x[i] * alpha


@cuda.jit
def add_kernel(x, val):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = x[i] + val


def test_graph_scope_numba():
    """Test basic graph_scope nesting with Numba kernels."""
    n = 1024
    X_host = np.ones(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    # Nested graph scope: X *= 2, then X += 1
    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            scale_kernel[bpg, tpb, nb_stream](dX, 2.0)

        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 1.0)

    # Another graph scope: X *= 3
    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            scale_kernel[bpg, tpb, nb_stream](dX, 3.0)

    ctx.finalize()

    # Expected: (1.0 * 2.0 + 1.0) * 3.0 = 9.0
    assert np.allclose(X_host, 9.0), f"Expected 9.0, got {X_host[0]}"


def test_repeat_numba():
    """Test repeat scope with Numba — increment X by 1 ten times."""
    n = 1024
    X_host = np.zeros(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    with ctx.repeat(10):
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 1.0)

    ctx.finalize()

    # Expected: 0.0 + 10 * 1.0 = 10.0
    assert np.allclose(X_host, 10.0), f"Expected 10.0, got {X_host[0]}"


if __name__ == "__main__":
    test_graph_scope_numba()
    test_repeat_numba()
    test_jacobi_stackable_numba()
