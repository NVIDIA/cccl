# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Jacobi iteration with stackable context and while_loop — PyTorch version.
Python equivalent of cudax/examples/stf/jacobi_stackable_raii.cu

Requires CUDA 12.4+ for conditional graph nodes.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop.pytorch import pytorch_task  # noqa: E402


def test_jacobi_stackable_pytorch():
    m, n = 256, 256
    tol = 0.1

    A_host = np.zeros((m, n), dtype=np.float64)
    Anew_host = np.zeros((m, n), dtype=np.float64)
    # Host-backed residual so we can assert convergence after finalize.
    residual_host = np.zeros((1,), dtype=np.float64)

    ctx = stf.stackable_context()

    lA = ctx.logical_data(A_host, name="A")
    lAnew = ctx.logical_data(Anew_host, name="Anew")
    lresidual = ctx.logical_data(residual_host, name="residual")

    # Initialize: A(i,j) = 1.0 if i==j else -1.0
    with pytorch_task(ctx, lA.write(), lAnew.write()) as (tA, tAnew):
        tA.fill_(-1.0)
        tA.fill_diagonal_(1.0)
        tAnew.copy_(tA)

    # Iterative solve with while loop
    with ctx.while_loop() as loop:
        # Jacobi step: compute Anew from A neighbors, measure residual
        with pytorch_task(ctx, lA.read(), lAnew.write(), lresidual.write()) as (
            tA,
            tAnew,
            tres,
        ):
            tAnew[1:-1, 1:-1] = 0.25 * (
                tA[:-2, 1:-1] + tA[2:, 1:-1] + tA[1:-1, :-2] + tA[1:-1, 2:]
            )
            tres[0] = torch.max(torch.abs(tA[1:-1, 1:-1] - tAnew[1:-1, 1:-1]))

        # Copy Anew -> A (interior points)
        with pytorch_task(ctx, lA.rw(), lAnew.read()) as (tA, tAnew):
            tA[1:-1, 1:-1] = tAnew[1:-1, 1:-1]

        # Continue while residual > tolerance
        loop.continue_while(lresidual, ">", tol)

    ctx.finalize()

    # The loop can only exit with residual <= tol; a non-finite or too-large
    # final residual means the solve diverged or never ran.
    assert np.isfinite(residual_host[0]), (
        f"Jacobi residual is non-finite ({residual_host[0]})"
    )
    assert residual_host[0] <= tol, (
        f"Jacobi residual {residual_host[0]} exceeds tolerance {tol}"
    )

    # Interior-only updates leave the boundary at its initial values
    # (A[i,j] = 1 on the diagonal else -1).
    init_full = np.where(np.eye(m, n, dtype=bool), 1.0, -1.0)
    assert np.allclose(A_host[0, :], init_full[0, :])
    assert np.allclose(A_host[-1, :], init_full[-1, :])
    assert np.allclose(A_host[:, 0], init_full[:, 0])
    assert np.allclose(A_host[:, -1], init_full[:, -1])

    # Interior must have evolved and stayed finite.
    interior = (slice(1, m - 1), slice(1, n - 1))
    assert np.all(np.isfinite(A_host))
    assert not np.allclose(A_host[interior], init_full[interior]), (
        "Jacobi interior did not evolve from its initial state"
    )

    print(f"Jacobi converged (PyTorch) with residual {residual_host[0]} <= {tol}")


def test_graph_scope_pytorch():
    """Test basic graph_scope nesting with PyTorch operations."""
    n = 1024
    X_host = np.ones(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    # Nested graph scope: X *= 2, then X += 1
    with ctx.graph_scope():
        with pytorch_task(ctx, lX.rw()) as (tX,):
            tX[:] = tX * 2.0

        with pytorch_task(ctx, lX.rw()) as (tX,):
            tX[:] = tX + 1.0

    # Another graph scope: X *= 3
    with ctx.graph_scope():
        with pytorch_task(ctx, lX.rw()) as (tX,):
            tX[:] = tX * 3.0

    ctx.finalize()

    # Expected: (1.0 * 2.0 + 1.0) * 3.0 = 9.0
    assert np.allclose(X_host, 9.0), f"Expected 9.0, got {X_host[0]}"


def test_repeat_pytorch():
    """Test repeat scope with PyTorch — increment X by 1 ten times."""
    n = 1024
    X_host = np.zeros(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    with ctx.repeat(10):
        with pytorch_task(ctx, lX.rw()) as (tX,):
            tX[:] = tX + 1.0

    ctx.finalize()

    # Expected: 0.0 + 10 * 1.0 = 10.0
    assert np.allclose(X_host, 10.0), f"Expected 10.0, got {X_host[0]}"
