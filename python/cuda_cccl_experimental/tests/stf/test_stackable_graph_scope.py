# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Test basic stackable context operations: graph_scope, repeat, read_only.
These tests exercise the stackable context without while_loop (no CUDA 12.4+ conditional nodes).
"""

import numba
import numpy as np
from numba import cuda
from numba_helpers import numba_arguments

import cuda.stf as stf

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


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


@cuda.jit
def axpy_kernel(y, alpha, x):
    """y = y + alpha * x"""
    i = cuda.grid(1)
    if i < y.size:
        y[i] = y[i] + alpha * x[i]


def test_single_graph_scope():
    """Single graph scope: scale X by 2."""
    n = 1024
    X_host = np.ones(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            scale_kernel[bpg, tpb, nb_stream](dX, 2.0)

    ctx.finalize()

    assert np.allclose(X_host, 2.0), f"Expected 2.0, got {X_host[0]}"


def test_nested_graph_scopes():
    """Two sequential graph scopes: scale then add."""
    n = 512
    X_host = np.ones(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    # First scope: X *= 3
    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            scale_kernel[bpg, tpb, nb_stream](dX, 3.0)

    # Second scope: X += 5
    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 5.0)

    ctx.finalize()

    # Expected: 1.0 * 3.0 + 5.0 = 8.0
    assert np.allclose(X_host, 8.0), f"Expected 8.0, got {X_host[0]}"


def test_multi_data_graph_scope():
    """Graph scope with two data items: Y = Y + alpha * X."""
    n = 256
    X_host = np.full(n, 2.0, dtype=np.float32)
    Y_host = np.full(n, 1.0, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")
    lY = ctx.logical_data(Y_host, name="Y")

    lX.set_read_only()

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    with ctx.graph_scope():
        with ctx.task(lY.rw(), lX.read()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            from numba_helpers import get_arg_numba

            dY = get_arg_numba(t, 0)
            dX = get_arg_numba(t, 1)
            axpy_kernel[bpg, tpb, nb_stream](dY, 3.0, dX)

    ctx.finalize()

    # Expected: Y = 1.0 + 3.0 * 2.0 = 7.0
    assert np.allclose(Y_host, 7.0), f"Expected 7.0, got {Y_host[0]}"


def test_graph_scope_for_loop():
    """Multiple graph scopes in a Python for loop (like C++ examples)."""
    n = 1024
    X_host = np.ones(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    for _ in range(5):
        with ctx.graph_scope():
            with ctx.task(lX.rw()) as t:
                nb_stream = cuda.external_stream(t.stream_ptr())
                dX = numba_arguments(t)
                add_kernel[bpg, tpb, nb_stream](dX, 1.0)

    ctx.finalize()

    # Expected: 1.0 + 5 * 1.0 = 6.0
    assert np.allclose(X_host, 6.0), f"Expected 6.0, got {X_host[0]}"


def test_repeat_scope():
    """Repeat scope: add 1 to X ten times."""
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


def test_fence():
    """Test fence() returns to host between scopes."""
    n = 256
    X_host = np.ones(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            scale_kernel[bpg, tpb, nb_stream](dX, 5.0)

    # Fence synchronizes back to host
    fence_stream = ctx.fence()
    assert fence_stream is not None

    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 3.0)

    ctx.finalize()

    # Expected: 1.0 * 5.0 + 3.0 = 8.0
    assert np.allclose(X_host, 8.0), f"Expected 8.0, got {X_host[0]}"


if __name__ == "__main__":
    test_single_graph_scope()
    test_nested_graph_scopes()
    test_multi_data_graph_scope()
    test_graph_scope_for_loop()
    test_repeat_scope()
    test_fence()
    print("All graph_scope tests passed!")
