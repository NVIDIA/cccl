# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
BiCGSTAB solver example — STF with PyTorch on a stackable context.

Demonstrates:
  - stackable_context task orchestration
  - pytorch_task interop for dense matvec/dot updates
  - solving a non-symmetric linear system with BiCGSTAB

This complements ``cg.py`` by covering a solver suitable for non-symmetric
matrices.
"""

import numpy as np
import pytest

import cuda.stf._experimental as stf
from cuda.stf._experimental.interop.pytorch import pytorch_task

torch = pytest.importorskip("torch")


def stf_dot(ctx, la, lb, lres):
    """res = dot(a, b)."""
    with pytorch_task(ctx, la.read(), lb.read(), lres.write()) as (tA, tB, tRes):
        tRes.copy_(torch.dot(tA, tB).unsqueeze(0))


def stf_matvec(ctx, lA, lx, ly):
    """y = A @ x (dense matrix-vector)."""
    with pytorch_task(ctx, lA.read(), lx.read(), ly.write()) as (tA, tX, tY):
        tY[:] = torch.mv(tA, tX)


def bicgstab_solver(ctx, lA, lX, lB, N, tol=1e-10, maxiter=400):
    """Solve A * X = B with BiCGSTAB."""
    # Vector temporaries
    lR = ctx.logical_data_empty((N,), np.float64, name="R")
    lRhat = ctx.logical_data_empty((N,), np.float64, name="Rhat")
    lP = ctx.logical_data_empty((N,), np.float64, name="P")
    lV = ctx.logical_data_empty((N,), np.float64, name="V")
    lS = ctx.logical_data_empty((N,), np.float64, name="S")
    lT = ctx.logical_data_empty((N,), np.float64, name="T")

    # Scalar temporaries
    lrho = ctx.logical_data_empty((1,), np.float64, name="rho")
    lrho_prev = ctx.logical_data_empty((1,), np.float64, name="rho_prev")
    lalpha = ctx.logical_data_empty((1,), np.float64, name="alpha")
    lomega = ctx.logical_data_empty((1,), np.float64, name="omega")
    ltmp = ctx.logical_data_empty((1,), np.float64, name="tmp")
    liter = ctx.logical_data_empty((1,), np.float64, name="iter")
    lcond = ctx.logical_data_empty((1,), np.float64, name="cond")

    with pytorch_task(ctx, lX.write()) as (tX,):
        tX.zero_()
    with pytorch_task(ctx, lR.write(), lB.read()) as (tR, tB):
        tR[:] = tB
    with pytorch_task(ctx, lRhat.write(), lR.read()) as (tRhat, tR):
        tRhat[:] = tR
    with pytorch_task(ctx, lP.write(), lV.write()) as (tP, tV):
        tP.zero_()
        tV.zero_()
    with pytorch_task(ctx, lrho_prev.write(), lalpha.write(), lomega.write()) as (
        tRhoPrev,
        tAlpha,
        tOmega,
    ):
        tRhoPrev[:] = 1.0
        tAlpha[:] = 1.0
        tOmega[:] = 1.0
    with pytorch_task(ctx, liter.write(), lcond.write()) as (tIter, tCond):
        tIter[:] = 0.0
        tCond[:] = 1.0

    tol_sq = tol * tol

    # --- BiCGSTAB while loop (conditional CUDA graph node) -----------------
    with ctx.while_loop() as loop:
        # rho = dot(rhat, r)
        stf_dot(ctx, lRhat, lR, lrho)

        # p = r + beta * (p - omega * v)
        with pytorch_task(
            ctx,
            lP.rw(),
            lR.read(),
            lrho.read(),
            lrho_prev.read(),
            lalpha.read(),
            lomega.read(),
            lV.read(),
        ) as (tP, tR, tRho, tRhoPrev, tAlpha, tOmega, tV):
            beta = (tRho.squeeze() / tRhoPrev.squeeze()) * (
                tAlpha.squeeze() / tOmega.squeeze()
            )
            tP[:] = tR + beta * (tP - tOmega.squeeze() * tV)

        # v = A @ p
        stf_matvec(ctx, lA, lP, lV)

        # alpha = rho / dot(rhat, v)
        stf_dot(ctx, lRhat, lV, ltmp)
        with pytorch_task(ctx, lalpha.write(), lrho.read(), ltmp.read()) as (
            tAlpha,
            tRho,
            tTmp,
        ):
            tAlpha[:] = tRho.squeeze() / tTmp.squeeze()

        # s = r - alpha * v
        with pytorch_task(ctx, lS.write(), lR.read(), lalpha.read(), lV.read()) as (
            tS,
            tR,
            tAlpha,
            tV,
        ):
            tS[:] = tR - tAlpha.squeeze() * tV

        # t = A @ s
        stf_matvec(ctx, lA, lS, lT)

        # omega = dot(t, s) / dot(t, t)
        stf_dot(ctx, lT, lS, lomega)
        stf_dot(ctx, lT, lT, ltmp)
        with pytorch_task(ctx, lomega.rw(), ltmp.read()) as (tOmega, tTmp):
            tOmega[:] = tOmega.squeeze() / tTmp.squeeze()

        # x = x + alpha*p + omega*s
        with pytorch_task(
            ctx, lX.rw(), lalpha.read(), lP.read(), lomega.read(), lS.read()
        ) as (tX, tAlpha, tP, tOmega, tS):
            tX[:] = tX + tAlpha.squeeze() * tP + tOmega.squeeze() * tS

        # r = s - omega*t
        with pytorch_task(ctx, lR.rw(), lS.read(), lomega.read(), lT.read()) as (
            tR,
            tS,
            tOmega,
            tT,
        ):
            tR[:] = tS - tOmega.squeeze() * tT

        # rho_prev = rho
        with pytorch_task(ctx, lrho_prev.write(), lrho.read()) as (tRhoPrev, tRho):
            tRhoPrev.copy_(tRho)

        # Continue while residual norm² > tol² and iter < maxiter.
        stf_dot(ctx, lR, lR, ltmp)
        with pytorch_task(ctx, liter.rw(), lcond.write(), ltmp.read()) as (
            tIter,
            tCond,
            tRes,
        ):
            tIter += 1.0
            tCond[:] = (
                ((tRes.squeeze() > tol_sq) & (tIter.squeeze() < float(maxiter)))
                .to(tCond.dtype)
                .unsqueeze(0)
            )

        loop.continue_while(lcond, ">", 0.5)


def test_bicgstab_solver():
    """Solve a random non-symmetric system with BiCGSTAB; verify against numpy."""
    N = 1024
    rng = np.random.default_rng(1234)

    A_host = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        lower = rng.uniform(-0.2, 0.2) if i > 0 else 0.0
        upper = rng.uniform(-0.2, 0.2) if i < N - 1 else 0.0
        # Strict diagonal dominance -> robust solve target.
        A_host[i, i] = 2.5 + abs(lower) + abs(upper) + rng.uniform(0.0, 0.5)
        if i > 0:
            A_host[i, i - 1] = lower
        if i < N - 1:
            A_host[i, i + 1] = upper

    B_host = np.ones(N, dtype=np.float64)
    X_host = np.zeros(N, dtype=np.float64)
    X_ref = np.linalg.solve(A_host, B_host)

    ctx = stf.stackable_context()
    lA = ctx.logical_data(A_host, name="A")
    lB = ctx.logical_data(B_host, name="B")
    lX = ctx.logical_data(X_host, name="X")
    lA.set_read_only()
    lB.set_read_only()

    bicgstab_solver(ctx, lA, lX, lB, N, tol=1e-10, maxiter=600)
    ctx.finalize()

    error = np.max(np.abs(X_host - X_ref))
    print("=== BiCGSTAB solver (PyTorch + stackable_context) ===")
    print(f"Matrix: {N}x{N} tridiagonal non-symmetric")
    print(f"Max error vs numpy.linalg.solve: {error:.2e}")

    assert not np.any(np.isnan(X_host)), "NaN in solution"
    assert not np.any(np.isinf(X_host)), "Inf in solution"
    assert np.allclose(X_host, X_ref, atol=1e-6), (
        f"BiCGSTAB solution does not match reference (max error = {error:.2e})"
    )


def main():
    test_bicgstab_solver()


if __name__ == "__main__":
    main()
