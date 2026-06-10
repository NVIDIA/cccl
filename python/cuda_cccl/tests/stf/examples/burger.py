# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Full Burger equation solver using stackable context + PyTorch.

Solves the viscous Burger equation using an implicit time-stepping scheme
with Newton + BiCGSTAB, expressed entirely as PyTorch tensor operations inside
pytorch_task context managers.

Nesting structure (5 levels, fully graph-captured):
  with ctx.repeat(outer_iters):           # level 1 (conditional)
    with ctx.graph_scope():               # level 2
      with ctx.repeat(substeps):          # level 3 (conditional)
        newton_solver(ctx, ...)
          with ctx.while_loop():          # level 4 (Newton)
            compute_residual / assemble_jacobian / ...
            bicgstab_solver(ctx, ...)
              with ctx.while_loop():      # level 5 (BiCGSTAB)
                spmv / dot / axpy / ...
      pytorch_task: snapshot copy

Snapshots are collected via a regular GPU task (pytorch_task) that copies
the solution into a pre-allocated buffer using index_copy_.  This avoids
host_launch, which creates host callback nodes that are not supported
inside CUDA conditional graph bodies (repeat / while_loop).

Requires CUDA 12.4+ (conditional graph nodes).

NOTE: All scalar writes use slice ops / .copy_() / .fill_() instead of
      tensor[0] = val, because PyTorch's single-element indexed writes
      use cudaMemcpyAsync which is incompatible with CUDA graph capture.
"""

import os

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop.pytorch import pytorch_task  # noqa: E402

torch = pytest.importorskip("torch")

BURGER_PLOT = os.environ.get("BURGER_PLOT", "") != ""


# ---------------------------------------------------------------------------
# Linear-algebra building blocks (all pure PyTorch, graph-capture safe)
# ---------------------------------------------------------------------------


def stf_dot(ctx, la, lb, lres):
    """res = dot(a, b).  Graph-safe: uses copy_ instead of [0] = ..."""
    with pytorch_task(ctx, la.read(), lb.read(), lres.write()) as (tA, tB, tRes):
        tRes.copy_(torch.dot(tA, tB).unsqueeze(0))


def stf_spmv(ctx, lA_val, lx, ly, N):
    """
    y = A * x  (tridiagonal matvec via direct tensor slicing).

    Exploits the known CSR layout of the Burger Jacobian:
      values[0]                                -> boundary row 0
      values[1+3*k], [2+3*k], [3+3*k]  k=0..N-3  -> interior row k+1
      values[1+3*(N-2)]                        -> boundary row N-1
    """
    interior = N - 2
    last = 1 + 3 * interior
    with pytorch_task(ctx, lA_val.read(), lx.read(), ly.write()) as (tVal, tX, tY):
        # Boundary rows (use slicing, not scalar indexing)
        tY[:1] = tVal[:1] * tX[:1]
        tY[N - 1 : N] = tVal[last : last + 1] * tX[N - 1 : N]

        # Interior rows: extract tridiagonal bands from packed CSR values
        lower = tVal[1 : 1 + 3 * interior : 3]
        diag = tVal[2 : 2 + 3 * interior : 3]
        upper = tVal[3 : 3 + 3 * interior : 3]

        tY[1 : N - 1] = lower * tX[0 : N - 2] + diag * tX[1 : N - 1] + upper * tX[2:N]


# ---------------------------------------------------------------------------
# Physics: Burger residual and Jacobian
# ---------------------------------------------------------------------------


def compute_residual(ctx, lU, lU_prev, lresidual, N, h, dt, nu):
    """
    F(U) for the implicit Burger discretisation.

    Boundary rows enforce homogeneous Dirichlet (u=0).
    Interior: F_i = (u_i - u_prev_i)/dt + u_i*(u_{i+1}-u_{i-1})/(2h)
                     - nu*(u_{i-1} - 2u_i + u_{i+1})/h^2
    """
    with pytorch_task(ctx, lresidual.write(), lU.read(), lU_prev.read()) as (
        tRes,
        tU,
        tUp,
    ):
        # Boundary residual = U (for Dirichlet u=0, residual = u - 0)
        tRes[:] = tU

        u = tU[1 : N - 1]
        u_left = tU[0 : N - 2]
        u_right = tU[2:N]
        u_prev = tUp[1 : N - 1]

        term_time = (u - u_prev) / dt
        term_conv = u * (u_right - u_left) / (2.0 * h)
        term_diff = -nu * (u_left - 2.0 * u + u_right) / (h * h)
        tRes[1 : N - 1] = term_time + term_conv + term_diff


def assemble_jacobian(ctx, lU, lA_val, N, h, dt, nu):
    """
    Fill CSR values for the Jacobian J = dF/dU.

    Layout: boundary rows get 1.0 on the diagonal; interior rows get the
    tridiagonal stencil packed as (left, center, right) with stride-3
    indexing.
    """
    interior = N - 2
    last = 1 + 3 * interior
    with pytorch_task(ctx, lU.read(), lA_val.write()) as (tU, tVal):
        # Boundary diagonals (slice, not scalar index)
        tVal[:1].fill_(1.0)
        tVal[last : last + 1].fill_(1.0)

        u = tU[1 : N - 1]
        u_left = tU[0 : N - 2]
        u_right = tU[2:N]

        left = -u / (2.0 * h) - nu / (h * h)
        center = 1.0 / dt + (u_right - u_left) / (2.0 * h) + 2.0 * nu / (h * h)
        right = u / (2.0 * h) - nu / (h * h)

        tVal[1 : 1 + 3 * interior : 3] = left
        tVal[2 : 2 + 3 * interior : 3] = center
        tVal[3 : 3 + 3 * interior : 3] = right


# ---------------------------------------------------------------------------
# BiCGSTAB solver
# ---------------------------------------------------------------------------


def bicgstab_solver(ctx, lA_val, lX, lB, N, tol=1e-8, max_iter=100):
    """
    BiCGSTAB solver: A * X = B for generally non-symmetric Jacobians.

    Uses a stackable while_loop for the iteration, with a compound
    condition scalar (convergence AND iteration cap).
    """
    # --- Data created before the while scope ---
    lR = ctx.logical_data_empty((N,), np.float64, name="R")
    lRhat = ctx.logical_data_empty((N,), np.float64, name="Rhat")
    lP = ctx.logical_data_empty((N,), np.float64, name="P")
    lV = ctx.logical_data_empty((N,), np.float64, name="V")
    lS = ctx.logical_data_empty((N,), np.float64, name="S")
    lT = ctx.logical_data_empty((N,), np.float64, name="T")
    lrho = ctx.logical_data_empty((1,), np.float64, name="rho")
    lrho_prev = ctx.logical_data_empty((1,), np.float64, name="rho_prev")
    lalpha = ctx.logical_data_empty((1,), np.float64, name="alpha")
    lomega = ctx.logical_data_empty((1,), np.float64, name="omega")
    liter = ctx.logical_data_empty((1,), np.float64, name="bicg_iter")
    ltmp = ctx.logical_data_empty((1,), np.float64, name="tmp")

    # X = 0
    with pytorch_task(ctx, lX.write()) as (tX,):
        tX.zero_()

    # R = B
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
    with pytorch_task(ctx, liter.write()) as (tIter,):
        tIter.fill_(0.0)

    # --- BiCGSTAB while loop ---
    tol_sq = tol * tol

    with ctx.while_loop() as loop:
        lcond = ctx.logical_data_empty((1,), np.float64, name="bicg_cond")

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

        # v = A*p
        stf_spmv(ctx, lA_val, lP, lV, N)

        # alpha = rho / dot(rhat, v)
        stf_dot(ctx, lRhat, lV, ltmp)
        with pytorch_task(ctx, lalpha.write(), lrho.read(), ltmp.read()) as (
            tAlpha,
            tRho,
            tTmp,
        ):
            tAlpha[:] = tRho.squeeze() / tTmp.squeeze()

        # s = r - alpha*v
        with pytorch_task(ctx, lS.write(), lR.read(), lalpha.read(), lV.read()) as (
            tS,
            tR,
            tAlpha,
            tV,
        ):
            tS[:] = tR - tAlpha.squeeze() * tV

        # t = A*s
        stf_spmv(ctx, lA_val, lS, lT, N)

        # omega = dot(t,s)/dot(t,t)
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

        # Compound condition: continue if residual > tol and iter < max.
        stf_dot(ctx, lR, lR, ltmp)
        with pytorch_task(ctx, ltmp.read(), liter.rw(), lcond.write()) as (
            tRes,
            tIter,
            tCond,
        ):
            tIter += 1
            not_converged = (tRes.squeeze() > tol_sq).to(torch.float64)
            not_max = (tIter.squeeze() < max_iter).to(torch.float64)
            tCond.copy_((not_converged * not_max).unsqueeze(0))

        loop.continue_while(lcond, ">", 0.5)


# ---------------------------------------------------------------------------
# Newton solver
# ---------------------------------------------------------------------------


def newton_solver(
    ctx, lU, lA_val, N, h, dt, nu, max_newton=20, newton_tol=1e-10, max_cg=100
):
    """
    Newton solver for the implicit Burger time step.

    Each iteration: compute residual, assemble Jacobian, solve the
    linear system J * delta = -F(U) with BiCGSTAB, then U += delta.
    Uses a compound condition scalar for the while loop.
    """
    # --- Data created before the while scope ---
    lU_prev = ctx.logical_data_empty((N,), np.float64, name="U_prev")
    lnewton_norm2 = ctx.logical_data_empty((1,), np.float64, name="newton_norm2")
    lnewton_iter = ctx.logical_data_empty((1,), np.float64, name="newton_iter")

    # U_prev = U
    with pytorch_task(ctx, lU_prev.write(), lU.read()) as (tUp, tU):
        tUp[:] = tU

    # iter = 0
    with pytorch_task(ctx, lnewton_iter.write()) as (tIter,):
        tIter.fill_(0.0)

    newton_tol_sq = newton_tol * newton_tol

    with ctx.while_loop() as loop:
        # Data scoped to the while body
        lresidual = ctx.logical_data_empty((N,), np.float64, name="residual")
        ldelta = ctx.logical_data_empty((N,), np.float64, name="delta")
        lrhs = ctx.logical_data_empty((N,), np.float64, name="rhs")
        lnewton_cond = ctx.logical_data_empty((1,), np.float64, name="newton_cond")

        # Compute residual F(U)
        compute_residual(ctx, lU, lU_prev, lresidual, N, h, dt, nu)

        # newton_norm2 = residual' * residual
        stf_dot(ctx, lresidual, lresidual, lnewton_norm2)

        # Assemble Jacobian J = dF/dU
        assemble_jacobian(ctx, lU, lA_val, N, h, dt, nu)

        # rhs = -residual
        with pytorch_task(ctx, lrhs.write(), lresidual.read()) as (tRhs, tRes):
            tRhs[:] = -tRes

        # Solve J * delta = rhs  with BiCGSTAB
        bicgstab_solver(ctx, lA_val, ldelta, lrhs, N, tol=1e-8, max_iter=max_cg)

        # U += delta
        with pytorch_task(ctx, lU.rw(), ldelta.read()) as (tU, tDelta):
            tU += tDelta

        # Compound Newton condition
        with pytorch_task(
            ctx, lnewton_norm2.read(), lnewton_iter.rw(), lnewton_cond.write()
        ) as (tNorm2, tIter, tCond):
            tIter += 1
            not_converged = (tNorm2.squeeze() > newton_tol_sq).to(torch.float64)
            not_max = (tIter.squeeze() < max_newton).to(torch.float64)
            tCond.copy_((not_converged * not_max).unsqueeze(0))

        loop.continue_while(lnewton_cond, ">", 0.5)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def test_burger():
    """
    Full Burger equation test.

    Nesting structure:
      for outer in range(outer_iters):
        graph_scope:
          repeat(substeps):
            newton_solver(...)
    """
    N = 2560
    nsteps = 300
    substeps = 10
    outer_iters = nsteps // substeps
    nu = 0.05
    h = 1.0 / (N - 1)
    dt = max(0.5 * h * h / nu, 0.001)
    nz = 3 * N - 4

    print("=== Burger equation solver (PyTorch + stackable_context) ===")
    print(f"Grid: N={N}, h={h:.4e}")
    print(f"Time: dt={dt:.4e}, nsteps={nsteps}, substeps={substeps}")
    print(f"Physics: nu={nu}")

    # Initial condition: sin(pi*x) with homogeneous Dirichlet BCs
    U_host = np.zeros(N, dtype=np.float64)
    x_grid = np.linspace(0, 1, N)
    U_host[1:-1] = np.sin(np.pi * x_grid[1:-1])

    U_init_max = np.max(np.abs(U_host))
    U_init_snap = U_host.copy()

    ctx = stf.stackable_context()
    lU = ctx.logical_data(U_host, name="U")
    lA_val = ctx.logical_data_empty((nz,), np.float64, name="csr_val")

    # Snapshot buffer: one row per outer iteration, filled by a GPU task.
    # We use a regular pytorch_task (kernel node) instead of host_launch
    # because host callback nodes are not supported inside CUDA conditional
    # graph bodies (repeat / while_loop).
    snapshots_host = np.zeros((outer_iters, N), dtype=np.float64)
    lSnapshots = ctx.logical_data(snapshots_host, name="snapshots")
    snap_iter_host = np.zeros(1, dtype=np.int64)
    lSnapIter = ctx.logical_data(snap_iter_host, name="snap_iter")

    # Time-stepping: repeat > graph_scope > repeat > newton_solver
    with ctx.repeat(outer_iters):
        with ctx.graph_scope():
            with ctx.repeat(substeps):
                newton_solver(
                    ctx,
                    lU,
                    lA_val,
                    N,
                    h,
                    dt,
                    nu,
                    max_newton=20,
                    newton_tol=1e-10,
                    max_cg=100,
                )

            # Store snapshot via GPU copy (graph-safe, no host callback)
            with pytorch_task(ctx, lU.read(), lSnapshots.rw(), lSnapIter.rw()) as (
                tU,
                tSnap,
                tIter,
            ):
                idx = tIter[0:1].long()
                tSnap.index_copy_(0, idx, tU.unsqueeze(0))
                tIter.add_(1)

    ctx.finalize()

    # Build snapshot list from the GPU-filled buffer (after finalize)
    snapshots = [(0, U_init_snap)]
    for i in range(outer_iters):
        step = (i + 1) * substeps
        snapshots.append((step, snapshots_host[i].copy()))
        print(
            f"Timestep {step}, t={step * dt:.4e}, max(U)={np.max(snapshots_host[i]):.6f}"
        )

    # --- Validation ---
    assert not np.any(np.isnan(U_host)), "NaN detected in solution"
    assert not np.any(np.isinf(U_host)), "Inf detected in solution"
    assert np.isclose(U_host[0], 0.0, atol=1e-10), f"Left BC violated: U[0]={U_host[0]}"
    assert np.isclose(U_host[-1], 0.0, atol=1e-10), (
        f"Right BC violated: U[N-1]={U_host[-1]}"
    )
    assert np.max(np.abs(U_host)) < 2.0, (
        f"Solution unbounded: max|U|={np.max(np.abs(U_host))}"
    )

    U_final_max = np.max(np.abs(U_host))
    assert U_final_max < U_init_max, (
        f"Solution did not dissipate: initial max={U_init_max}, final max={U_final_max}"
    )

    print(f"Dissipation: {U_init_max:.6f} -> {U_final_max:.6f}")
    print("Burger test PASSED")

    # --- Plot (set BURGER_PLOT=1 to display) ---
    if BURGER_PLOT:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for step, U_snap in snapshots:
            label = f"t={step * dt:.4f}" if step > 0 else "initial"
            alpha = 0.4 if step == 0 else 0.5 + 0.5 * step / (nsteps)
            ax.plot(x_grid, U_snap, label=label, alpha=alpha)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"Viscous Burger equation  (N={N}, nu={nu}, dt={dt:.2e})")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("burger_solution.png", dpi=150)
        print("Saved burger_solution.png")
        plt.show()


def main():
    test_burger()


if __name__ == "__main__":
    main()
