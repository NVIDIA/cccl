# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Conjugate-gradient solver — STF with PyTorch on a stackable context.

Demonstrates:
  - **stackable_context** for nested asynchronous scopes
  - **while_loop** for data-dependent iteration (conditional CUDA graph nodes)
  - **pytorch_task** for expressing GPU linear algebra with PyTorch tensors
  - **host_launch** for asynchronous host-side observation of GPU results

Solves A * x = b where A is a random diagonally-dominant tridiagonal SPD
matrix, using the standard CG algorithm:

  stackable_context
    [setup: x=0, r=b, p=r, rsold=dot(r,r)]
    while_loop (rsnew > tol²):
      Ap    = A @ p
      pAp   = dot(p, Ap)
      x    += alpha * p        alpha = rsold / pAp
      r    -= alpha * Ap
      rsnew = dot(r, r)
      p     = r + beta * p     beta  = rsnew / rsold
      rsold = rsnew

Python port of cudax/examples/stf/linear_algebra/cg_csr_stackable.cu,
simplified to use a dense matrix.

Requires CUDA 12.4+ (conditional graph nodes).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf as stf  # noqa: E402

# --- Linear-algebra building blocks (PyTorch, graph-capture safe) ----------


def stf_dot(ctx, la, lb, lres):
    """res = dot(a, b).  Uses copy_ (not indexed write) for graph safety."""
    with pytorch_task(ctx, la.read(), lb.read(), lres.write()) as (tA, tB, tRes):
        tRes.copy_(torch.dot(tA, tB).unsqueeze(0))


def stf_matvec(ctx, lA, lx, ly):
    """y = A @ x  (dense matrix-vector product)."""
    with pytorch_task(ctx, lA.read(), lx.read(), ly.write()) as (tA, tX, tY):
        tY[:] = torch.mv(tA, tX)


# --- CG solver -----------------------------------------------------------


def cg_solver(ctx, lA, lX, lB, N, tol=1e-10):
    """
    Solve A * X = B with the Conjugate Gradient method.

    All temporaries are created as stackable logical data so they are
    automatically managed across while_loop iterations.
    """
    lR = ctx.logical_data_empty((N,), np.float64, name="R")
    lP = ctx.logical_data_empty((N,), np.float64, name="P")
    lrsold = ctx.logical_data_empty((1,), np.float64, name="rsold")

    # X = 0 (initial guess)
    with pytorch_task(ctx, lX.write()) as (tX,):
        tX.zero_()

    # R = B  (residual r = b − A·0 = b)
    with pytorch_task(ctx, lR.write(), lB.read()) as (tR, tB):
        tR[:] = tB

    # P = R
    with pytorch_task(ctx, lP.write(), lR.read()) as (tP, tR):
        tP[:] = tR

    # rsold = R'R
    stf_dot(ctx, lR, lR, lrsold)

    tol_sq = tol * tol

    # --- CG while loop (conditional CUDA graph node) ----------------------
    with ctx.while_loop() as loop:
        lAp = ctx.logical_data_empty((N,), np.float64, name="Ap")
        lpAp = ctx.logical_data_empty((1,), np.float64, name="pAp")
        lrsnew = ctx.logical_data_empty((1,), np.float64, name="rsnew")

        # Ap = A @ P
        stf_matvec(ctx, lA, lP, lAp)

        # pAp = P'Ap
        stf_dot(ctx, lP, lAp, lpAp)

        # X += alpha·P   (alpha = rsold / pAp)
        # Alpha is recomputed in the R update task below because each
        # pytorch_task is an independent graph node with its own closure.
        with pytorch_task(ctx, lX.rw(), lrsold.read(), lpAp.read(), lP.read()) as (
            tX,
            tRsold,
            tPAp,
            tP,
        ):
            alpha = tRsold.squeeze() / tPAp.squeeze()
            tX += alpha * tP

        # R -= alpha·Ap
        with pytorch_task(ctx, lR.rw(), lrsold.read(), lpAp.read(), lAp.read()) as (
            tR,
            tRsold,
            tPAp,
            tAp,
        ):
            alpha = tRsold.squeeze() / tPAp.squeeze()
            tR -= alpha * tAp

        # rsnew = R'R
        stf_dot(ctx, lR, lR, lrsnew)

        # Condition: continue while residual norm² exceeds tolerance².
        # This sets the predicate for the *next* replay — the P and rsold
        # updates below still execute in the current iteration.
        loop.continue_while(lrsnew, ">", tol_sq)

        # P = R + beta·P   (beta = rsnew / rsold)
        with pytorch_task(ctx, lP.rw(), lR.read(), lrsnew.read(), lrsold.read()) as (
            tP,
            tR,
            tRsnew,
            tRsold,
        ):
            beta = tRsnew.squeeze() / tRsold.squeeze()
            tP[:] = tR + beta * tP

        # rsold = rsnew
        with pytorch_task(ctx, lrsold.write(), lrsnew.read()) as (tRsold, tRsnew):
            tRsold.copy_(tRsnew)


# --- Test ----------------------------------------------------------------


def test_cg_solver():
    """Solve a random dense SPD system with CG; verify against numpy."""
    N = 2560

    # Random diagonally-dominant tridiagonal SPD matrix (same structure as
    # genTridiag in the C++ cg_csr_stackable.cu example).
    rng = np.random.default_rng(42)
    A_host = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        A_host[i, i] = 2.0 + rng.random()
        if i > 0:
            off = rng.random()
            A_host[i, i - 1] = off
            A_host[i - 1, i] = off

    B_host = np.ones(N, dtype=np.float64)
    X_host = np.zeros(N, dtype=np.float64)

    X_ref = np.linalg.solve(A_host, B_host)

    ctx = stf.stackable_context()
    lA = ctx.logical_data(A_host, name="A")
    lB = ctx.logical_data(B_host, name="B")
    lX = ctx.logical_data(X_host, name="X")

    lA.set_read_only()
    lB.set_read_only()

    cg_solver(ctx, lA, lX, lB, N, tol=1e-10)

    ctx.finalize()

    error = np.max(np.abs(X_host - X_ref))
    print("=== CG solver (PyTorch + stackable_context) ===")
    print(f"Matrix: {N}x{N} tridiagonal SPD")
    print(f"Max error vs numpy.linalg.solve: {error:.2e}")

    assert not np.any(np.isnan(X_host)), "NaN in solution"
    assert not np.any(np.isinf(X_host)), "Inf in solution"
    assert np.allclose(X_host, X_ref, atol=1e-6), (
        f"CG solution does not match reference (max error = {error:.2e})"
    )

    print("PASSED")


if __name__ == "__main__":
    test_cg_solver()
