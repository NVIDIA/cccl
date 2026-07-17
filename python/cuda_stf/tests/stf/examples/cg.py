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

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop.pytorch import pytorch_task  # noqa: E402

torch = pytest.importorskip("torch")

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


def cg_solver(ctx, lA, lX, lB, N, tol=1e-10, max_iter=None):
    """
    Solve A * X = B with the Conjugate Gradient method.

    All temporaries are created as stackable logical data so they are
    automatically managed across while_loop iterations.

    ``max_iter`` bounds the device-side while loop so a non-converging or
    ill-conditioned system terminates instead of replaying forever. CG
    converges in at most ``N`` steps in exact arithmetic, so the default
    leaves headroom for round-off; convergence is verified on the host after
    :meth:`finalize`.
    """
    if max_iter is None:
        max_iter = 2 * N + 50

    lR = ctx.logical_data_empty((N,), np.float64, name="R")
    lP = ctx.logical_data_empty((N,), np.float64, name="P")
    lrsold = ctx.logical_data_empty((1,), np.float64, name="rsold")
    # Device-side iteration counter used to enforce the iteration cap.
    liter = ctx.logical_data_empty((1,), np.float64, name="iter")

    # X = 0 (initial guess)
    with pytorch_task(ctx, lX.write()) as (tX,):
        tX.zero_()

    # iter = 0
    with pytorch_task(ctx, liter.write()) as (tIt,):
        tIt.zero_()

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

        # Iteration guard: control = rsnew while iterating, forced to 0
        # (<= tol²) once the iteration cap is reached so the loop terminates
        # instead of hanging on a non-converging system.
        lctrl = ctx.logical_data_empty((1,), np.float64, name="ctrl")
        with pytorch_task(ctx, lctrl.write(), lrsnew.read(), liter.rw()) as (
            tCtrl,
            tRsnew,
            tIt,
        ):
            tIt += 1.0
            capped = tIt.squeeze() >= float(max_iter)
            tCtrl.copy_(torch.where(capped, torch.zeros_like(tRsnew), tRsnew))

        # Condition: continue while residual norm² exceeds tolerance² and the
        # iteration cap has not been hit. This sets the predicate for the
        # *next* replay — the P and rsold updates below still execute in the
        # current iteration.
        loop.continue_while(lctrl, ">", tol_sq)

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
    residual_norm = float(np.linalg.norm(B_host - A_host @ X_host))
    b_norm = float(np.linalg.norm(B_host))
    print("=== CG solver (PyTorch + stackable_context) ===")
    print(f"Matrix: {N}x{N} tridiagonal SPD")
    print(f"Max error vs numpy.linalg.solve: {error:.2e}")
    print(f"Residual norm ||b - A x||: {residual_norm:.2e}")

    # Finite check first: a non-finite result means the iteration diverged.
    assert np.all(np.isfinite(X_host)), "CG produced non-finite values"
    # A large residual means CG stalled or hit the iteration cap without
    # converging; report it explicitly rather than only comparing to X_ref.
    assert residual_norm <= 1e-5 * max(1.0, b_norm), (
        f"CG did not converge: residual norm {residual_norm:.2e} "
        f"(relative {residual_norm / max(1.0, b_norm):.2e}); "
        "it may have hit the iteration cap or stalled"
    )
    assert np.allclose(X_host, X_ref, atol=1e-6), (
        f"CG solution does not match reference (max error = {error:.2e})"
    )

    print("PASSED")


def main():
    test_cg_solver()


if __name__ == "__main__":
    main()
