# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Efficient Burger equation solver using stackable context + fused Numba kernels.

Same physics and STF nesting as test_burger_stackable.py, but replaces the
many-kernel PyTorch elementwise operations with fused Numba @cuda.jit kernels
and band storage for the tridiagonal Jacobian.

Key differences from the PyTorch version:
  - Band storage (lower, diag, upper arrays) instead of stride-3 CSR.
  - Each physics operation is a single fused kernel instead of 6-12 PyTorch ops.
  - Dot products use shared-memory block reduction + atomicAdd.
  - CG update pairs (X += alpha*P, R -= alpha*Ap) fused into one kernel.

Nesting structure (5 levels, fully graph-captured):
  with ctx.repeat(outer_iters):           # level 1 (conditional)
    with ctx.graph_scope():               # level 2
      with ctx.repeat(substeps):          # level 3 (conditional)
        newton_solver(ctx, ...)
          with ctx.while_loop():          # level 4 (Newton)
            cg_solver(ctx, ...)
              with ctx.while_loop():      # level 5 (CG)

Requires CUDA 12.4+ (conditional graph nodes) and numba-cuda.
"""

import os
import time

import numba
import numpy as np
from numba import cuda
from numba_helpers import get_arg_numba

import cuda.stf as stf

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

BURGER_PLOT = os.environ.get("BURGER_PLOT", "") != ""
TPB = 256
DOT_BLOCKS = 256


# ---------------------------------------------------------------------------
# Fused Numba CUDA kernels
# ---------------------------------------------------------------------------


@cuda.jit
def tridiag_spmv_kernel(lower, diag, upper, x, y, N):
    """y = tridiag(lower, diag, upper) * x — one thread per row."""
    i = cuda.grid(1)
    if i >= N:
        return
    val = diag[i] * x[i]
    if i > 0:
        val += lower[i] * x[i - 1]
    if i < N - 1:
        val += upper[i] * x[i + 1]
    y[i] = val


@cuda.jit
def compute_residual_kernel(U, U_prev, residual, N, inv_dt, inv_2h, nu_inv_h2):
    """Fused Burger residual F(U): boundary + interior in one pass."""
    i = cuda.grid(1)
    if i >= N:
        return
    if i == 0 or i == N - 1:
        residual[i] = U[i]
    else:
        u = U[i]
        residual[i] = (
            (u - U_prev[i]) * inv_dt
            + u * (U[i + 1] - U[i - 1]) * inv_2h
            - nu_inv_h2 * (U[i - 1] - 2.0 * u + U[i + 1])
        )


@cuda.jit
def assemble_jacobian_kernel(U, lower, diag, upper, N, inv_dt, inv_2h, nu_inv_h2):
    """Fused Jacobian J = dF/dU into band storage — one pass over U."""
    i = cuda.grid(1)
    if i >= N:
        return
    if i == 0 or i == N - 1:
        lower[i] = 0.0
        diag[i] = 1.0
        upper[i] = 0.0
    else:
        u = U[i]
        lower[i] = -u * inv_2h - nu_inv_h2
        diag[i] = inv_dt + (U[i + 1] - U[i - 1]) * inv_2h + 2.0 * nu_inv_h2
        upper[i] = u * inv_2h - nu_inv_h2


@cuda.jit
def dot_zero_kernel(out):
    """Zero a scalar before atomic accumulation."""
    if cuda.grid(1) == 0:
        out[0] = 0.0


@cuda.jit
def dot_accum_kernel(a, b, out, N):
    """Block-level dot product with shared-memory reduction + atomicAdd."""
    tid = cuda.threadIdx.x
    shared = cuda.shared.array(256, dtype=numba.float64)

    acc = 0.0
    i = cuda.blockIdx.x * 256 + tid
    stride = 256 * cuda.gridDim.x
    while i < N:
        acc += a[i] * b[i]
        i += stride
    shared[tid] = acc
    cuda.syncthreads()

    s = 128
    while s > 0:
        if tid < s:
            shared[tid] += shared[tid + s]
        cuda.syncthreads()
        s >>= 1

    if tid == 0:
        cuda.atomic.add(out, 0, shared[0])


@cuda.jit
def axpy_pair_kernel(X, R, P, Ap, rsold, pAp, N):
    """Fused: X += alpha*P and R -= alpha*Ap where alpha = rsold/pAp."""
    i = cuda.grid(1)
    if i >= N:
        return
    alpha = rsold[0] / pAp[0]
    X[i] += alpha * P[i]
    R[i] -= alpha * Ap[i]


@cuda.jit
def p_update_kernel(P, R, rsnew, rsold, N):
    """P = R + (rsnew/rsold) * P."""
    i = cuda.grid(1)
    if i >= N:
        return
    P[i] = R[i] + (rsnew[0] / rsold[0]) * P[i]


@cuda.jit
def convergence_check_kernel(metric, iter_ctr, cond, tol_sq, max_iter, global_ctr):
    """Increment iter + global counter, set cond = 1 if (metric > tol^2 AND iter < max)."""
    if cuda.grid(1) == 0:
        iter_ctr[0] += 1.0
        global_ctr[0] += 1.0
        not_converged = 1.0 if metric[0] > tol_sq else 0.0
        not_max = 1.0 if iter_ctr[0] < max_iter else 0.0
        cond[0] = not_converged * not_max


@cuda.jit
def negate_kernel(dst, src, N):
    """dst = -src."""
    i = cuda.grid(1)
    if i < N:
        dst[i] = -src[i]


@cuda.jit
def axpy_kernel(y, x, N):
    """y += x."""
    i = cuda.grid(1)
    if i < N:
        y[i] += x[i]


@cuda.jit
def sub_kernel(y, x, N):
    """y -= x."""
    i = cuda.grid(1)
    if i < N:
        y[i] -= x[i]


@cuda.jit
def copy_vec_kernel(dst, src, N):
    """dst[:] = src[:]."""
    i = cuda.grid(1)
    if i < N:
        dst[i] = src[i]


@cuda.jit
def zero_kernel(x, N):
    """x[:] = 0."""
    i = cuda.grid(1)
    if i < N:
        x[i] = 0.0


@cuda.jit
def fill_scalar_kernel(x, val):
    """x[0] = val."""
    if cuda.grid(1) == 0:
        x[0] = val


@cuda.jit
def copy_scalar_kernel(dst, src):
    """dst[0] = src[0]."""
    if cuda.grid(1) == 0:
        dst[0] = src[0]


@cuda.jit
def snapshot_copy_kernel(snapshots, U, snap_iter, N):
    """snapshots[snap_iter, :] = U[:]; snap_iter += 1."""
    i = cuda.grid(1)
    if i < N:
        row = snap_iter[0]
        snapshots[row, i] = U[i]
    if i == 0:
        snap_iter[0] += 1


# ---------------------------------------------------------------------------
# STF wrapper functions
# ---------------------------------------------------------------------------


def _bpg(N):
    return (N + TPB - 1) // TPB


def stf_spmv(ctx, l_lower, l_diag, l_upper, lx, ly, N):
    """y = tridiag(lower, diag, upper) * x."""
    with ctx.task(
        l_lower.read(), l_diag.read(), l_upper.read(), lx.read(), ly.write()
    ) as t:
        s = cuda.external_stream(t.stream_ptr())
        tridiag_spmv_kernel[_bpg(N), TPB, s](
            get_arg_numba(t, 0),
            get_arg_numba(t, 1),
            get_arg_numba(t, 2),
            get_arg_numba(t, 3),
            get_arg_numba(t, 4),
            N,
        )


def stf_dot(ctx, la, lb, lout, N):
    """out = dot(a, b) via block reduction + atomicAdd."""
    with ctx.task(la.read(), lb.read(), lout.write()) as t:
        s = cuda.external_stream(t.stream_ptr())
        da, db, dout = get_arg_numba(t, 0), get_arg_numba(t, 1), get_arg_numba(t, 2)
        dot_zero_kernel[1, 1, s](dout)
        dot_accum_kernel[DOT_BLOCKS, TPB, s](da, db, dout, N)


def stf_compute_residual(ctx, lU, lU_prev, lresidual, N, inv_dt, inv_2h, nu_inv_h2):
    """Fused Burger residual."""
    with ctx.task(lU.read(), lU_prev.read(), lresidual.write()) as t:
        s = cuda.external_stream(t.stream_ptr())
        compute_residual_kernel[_bpg(N), TPB, s](
            get_arg_numba(t, 0),
            get_arg_numba(t, 1),
            get_arg_numba(t, 2),
            N,
            inv_dt,
            inv_2h,
            nu_inv_h2,
        )


def stf_assemble_jacobian(
    ctx, lU, l_lower, l_diag, l_upper, N, inv_dt, inv_2h, nu_inv_h2
):
    """Fused Jacobian into band storage."""
    with ctx.task(lU.read(), l_lower.write(), l_diag.write(), l_upper.write()) as t:
        s = cuda.external_stream(t.stream_ptr())
        assemble_jacobian_kernel[_bpg(N), TPB, s](
            get_arg_numba(t, 0),
            get_arg_numba(t, 1),
            get_arg_numba(t, 2),
            get_arg_numba(t, 3),
            N,
            inv_dt,
            inv_2h,
            nu_inv_h2,
        )


# ---------------------------------------------------------------------------
# CG solver
# ---------------------------------------------------------------------------


def cg_solver(
    ctx, l_lower, l_diag, l_upper, lX, lB, N, ltotal_cg, cg_tol=1e-8, max_cg=100
):
    """CG solver: tridiag(lower,diag,upper) * X = B."""
    lR = ctx.logical_data_empty((N,), np.float64, name="R")
    lP = ctx.logical_data_empty((N,), np.float64, name="P")
    lAx = ctx.logical_data_empty((N,), np.float64, name="Ax")
    lrsold = ctx.logical_data_empty((1,), np.float64, name="rsold")
    lcg_iter = ctx.logical_data_empty((1,), np.float64, name="cg_iter")

    # X = 0
    with ctx.task(lX.write()) as t:
        s = cuda.external_stream(t.stream_ptr())
        zero_kernel[_bpg(N), TPB, s](get_arg_numba(t, 0), N)

    # R = B
    with ctx.task(lR.write(), lB.read()) as t:
        s = cuda.external_stream(t.stream_ptr())
        copy_vec_kernel[_bpg(N), TPB, s](get_arg_numba(t, 0), get_arg_numba(t, 1), N)

    # Ax = A*X (X=0, but keeps structural fidelity)
    stf_spmv(ctx, l_lower, l_diag, l_upper, lX, lAx, N)

    # R -= Ax
    with ctx.task(lR.rw(), lAx.read()) as t:
        s = cuda.external_stream(t.stream_ptr())
        sub_kernel[_bpg(N), TPB, s](get_arg_numba(t, 0), get_arg_numba(t, 1), N)

    # P = R
    with ctx.task(lP.write(), lR.read()) as t:
        s = cuda.external_stream(t.stream_ptr())
        copy_vec_kernel[_bpg(N), TPB, s](get_arg_numba(t, 0), get_arg_numba(t, 1), N)

    # rsold = dot(R, R)
    stf_dot(ctx, lR, lR, lrsold, N)

    # cg_iter = 0
    with ctx.task(lcg_iter.write()) as t:
        s = cuda.external_stream(t.stream_ptr())
        fill_scalar_kernel[1, 1, s](get_arg_numba(t, 0), 0.0)

    cg_tol_sq = cg_tol * cg_tol

    with ctx.while_loop() as loop:
        lAp = ctx.logical_data_empty((N,), np.float64, name="Ap")
        lpAp = ctx.logical_data_empty((1,), np.float64, name="pAp")
        lrsnew = ctx.logical_data_empty((1,), np.float64, name="rsnew")
        lcond = ctx.logical_data_empty((1,), np.float64, name="cg_cond")

        # Ap = A*P
        stf_spmv(ctx, l_lower, l_diag, l_upper, lP, lAp, N)

        # pAp = dot(P, Ap)
        stf_dot(ctx, lP, lAp, lpAp, N)

        # Fused: X += alpha*P, R -= alpha*Ap
        with ctx.task(
            lX.rw(), lR.rw(), lP.read(), lAp.read(), lrsold.read(), lpAp.read()
        ) as t:
            s = cuda.external_stream(t.stream_ptr())
            axpy_pair_kernel[_bpg(N), TPB, s](
                get_arg_numba(t, 0),
                get_arg_numba(t, 1),
                get_arg_numba(t, 2),
                get_arg_numba(t, 3),
                get_arg_numba(t, 4),
                get_arg_numba(t, 5),
                N,
            )

        # rsnew = dot(R, R)
        stf_dot(ctx, lR, lR, lrsnew, N)

        # Convergence check: iter++, cond = (rsnew > tol^2) && (iter < max)
        with ctx.task(lrsnew.read(), lcg_iter.rw(), lcond.write(), ltotal_cg.rw()) as t:
            s = cuda.external_stream(t.stream_ptr())
            convergence_check_kernel[1, 1, s](
                get_arg_numba(t, 0),
                get_arg_numba(t, 1),
                get_arg_numba(t, 2),
                cg_tol_sq,
                float(max_cg),
                get_arg_numba(t, 3),
            )

        loop.continue_while(lcond, ">", 0.5)

        # P = R + beta*P
        with ctx.task(lP.rw(), lR.read(), lrsnew.read(), lrsold.read()) as t:
            s = cuda.external_stream(t.stream_ptr())
            p_update_kernel[_bpg(N), TPB, s](
                get_arg_numba(t, 0),
                get_arg_numba(t, 1),
                get_arg_numba(t, 2),
                get_arg_numba(t, 3),
                N,
            )

        # rsold = rsnew
        with ctx.task(lrsold.write(), lrsnew.read()) as t:
            s = cuda.external_stream(t.stream_ptr())
            copy_scalar_kernel[1, 1, s](get_arg_numba(t, 0), get_arg_numba(t, 1))


# ---------------------------------------------------------------------------
# Newton solver
# ---------------------------------------------------------------------------


def newton_solver(
    ctx,
    lU,
    l_lower,
    l_diag,
    l_upper,
    N,
    inv_dt,
    inv_2h,
    nu_inv_h2,
    ltotal_newton,
    ltotal_cg,
    max_newton=20,
    newton_tol=1e-10,
    max_cg=100,
):
    """Newton solver for implicit Burger time step with fused kernels."""
    lU_prev = ctx.logical_data_empty((N,), np.float64, name="U_prev")
    lnewton_norm2 = ctx.logical_data_empty((1,), np.float64, name="newton_norm2")
    lnewton_iter = ctx.logical_data_empty((1,), np.float64, name="newton_iter")

    # U_prev = U
    with ctx.task(lU_prev.write(), lU.read()) as t:
        s = cuda.external_stream(t.stream_ptr())
        copy_vec_kernel[_bpg(N), TPB, s](get_arg_numba(t, 0), get_arg_numba(t, 1), N)

    # newton_iter = 0
    with ctx.task(lnewton_iter.write()) as t:
        s = cuda.external_stream(t.stream_ptr())
        fill_scalar_kernel[1, 1, s](get_arg_numba(t, 0), 0.0)

    newton_tol_sq = newton_tol * newton_tol

    with ctx.while_loop() as loop:
        lresidual = ctx.logical_data_empty((N,), np.float64, name="residual")
        ldelta = ctx.logical_data_empty((N,), np.float64, name="delta")
        lrhs = ctx.logical_data_empty((N,), np.float64, name="rhs")
        lnewton_cond = ctx.logical_data_empty((1,), np.float64, name="newton_cond")

        # Residual F(U)
        stf_compute_residual(ctx, lU, lU_prev, lresidual, N, inv_dt, inv_2h, nu_inv_h2)

        # newton_norm2 = dot(residual, residual)
        stf_dot(ctx, lresidual, lresidual, lnewton_norm2, N)

        # Jacobian J = dF/dU
        stf_assemble_jacobian(
            ctx, lU, l_lower, l_diag, l_upper, N, inv_dt, inv_2h, nu_inv_h2
        )

        # rhs = -residual
        with ctx.task(lrhs.write(), lresidual.read()) as t:
            s = cuda.external_stream(t.stream_ptr())
            negate_kernel[_bpg(N), TPB, s](get_arg_numba(t, 0), get_arg_numba(t, 1), N)

        # CG solve: J * delta = rhs
        cg_solver(
            ctx,
            l_lower,
            l_diag,
            l_upper,
            ldelta,
            lrhs,
            N,
            ltotal_cg,
            cg_tol=1e-8,
            max_cg=max_cg,
        )

        # U += delta
        with ctx.task(lU.rw(), ldelta.read()) as t:
            s = cuda.external_stream(t.stream_ptr())
            axpy_kernel[_bpg(N), TPB, s](get_arg_numba(t, 0), get_arg_numba(t, 1), N)

        # Newton convergence: iter++, cond = (norm2 > tol^2) && (iter < max)
        with ctx.task(
            lnewton_norm2.read(),
            lnewton_iter.rw(),
            lnewton_cond.write(),
            ltotal_newton.rw(),
        ) as t:
            s = cuda.external_stream(t.stream_ptr())
            convergence_check_kernel[1, 1, s](
                get_arg_numba(t, 0),
                get_arg_numba(t, 1),
                get_arg_numba(t, 2),
                newton_tol_sq,
                float(max_newton),
                get_arg_numba(t, 3),
            )

        loop.continue_while(lnewton_cond, ">", 0.5)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def test_burger_fast():
    """
    Efficient Burger equation solver with fused Numba kernels.

    Same physics and graph nesting as test_burger_stackable.py.
    """
    N = 2560
    nsteps = 300
    substeps = 10
    outer_iters = nsteps // substeps
    nu = 0.05
    h = 1.0 / (N - 1)
    dt = max(0.5 * h * h / nu, 0.001)

    inv_dt = 1.0 / dt
    inv_2h = 1.0 / (2.0 * h)
    nu_inv_h2 = nu / (h * h)

    print("=== Burger solver (Numba fused kernels + stackable_context) ===")
    print(f"Grid: N={N}, h={h:.4e}")
    print(f"Time: dt={dt:.4e}, nsteps={nsteps}, substeps={substeps}")
    print(f"Physics: nu={nu}")

    U_host = np.zeros(N, dtype=np.float64)
    x_grid = np.linspace(0, 1, N)
    U_host[1:-1] = np.sin(np.pi * x_grid[1:-1])

    U_init_max = np.max(np.abs(U_host))
    U_init_snap = U_host.copy()

    cuda.get_current_device().reset()

    t_start = time.perf_counter()

    ctx = stf.stackable_context()
    lU = ctx.logical_data(U_host, name="U")

    # Band storage for tridiagonal Jacobian
    l_lower = ctx.logical_data_empty((N,), np.float64, name="J_lower")
    l_diag = ctx.logical_data_empty((N,), np.float64, name="J_diag")
    l_upper = ctx.logical_data_empty((N,), np.float64, name="J_upper")

    # Iteration counters (accumulated on GPU across all graph replays)
    newton_count = np.zeros(1, dtype=np.float64)
    cg_count = np.zeros(1, dtype=np.float64)
    ltotal_newton = ctx.logical_data(newton_count, name="total_newton")
    ltotal_cg = ctx.logical_data(cg_count, name="total_cg")

    # Snapshot buffer
    snapshots_host = np.zeros((outer_iters, N), dtype=np.float64)
    lSnapshots = ctx.logical_data(snapshots_host, name="snapshots")
    snap_iter_host = np.zeros(1, dtype=np.int64)
    lSnapIter = ctx.logical_data(snap_iter_host, name="snap_iter")

    t_submit = time.perf_counter()

    with ctx.repeat(outer_iters):
        with ctx.graph_scope():
            with ctx.repeat(substeps):
                newton_solver(
                    ctx,
                    lU,
                    l_lower,
                    l_diag,
                    l_upper,
                    N,
                    inv_dt,
                    inv_2h,
                    nu_inv_h2,
                    ltotal_newton,
                    ltotal_cg,
                    max_newton=20,
                    newton_tol=1e-10,
                    max_cg=100,
                )

            # Snapshot: copy U into buffer row, increment counter
            with ctx.task(lSnapshots.rw(), lU.read(), lSnapIter.rw()) as t:
                s = cuda.external_stream(t.stream_ptr())
                snapshot_copy_kernel[_bpg(N), TPB, s](
                    get_arg_numba(t, 0),
                    get_arg_numba(t, 1),
                    get_arg_numba(t, 2),
                    N,
                )

    t_submit_end = time.perf_counter()

    ctx.finalize()
    cuda.synchronize()
    t_end = time.perf_counter()

    submit_ms = (t_submit_end - t_submit) * 1000
    total_ms = (t_end - t_start) * 1000
    total_newton = int(newton_count[0])
    total_cg = int(cg_count[0])
    avg_newton = total_newton / nsteps
    avg_cg_per_newton = total_cg / total_newton if total_newton > 0 else 0

    exec_ms = (t_end - t_submit_end) * 1000

    print(f"Submit: {submit_ms:.1f}ms  Exec: {exec_ms:.1f}ms  Total: {total_ms:.1f}ms")
    print(f"Newton iters: {total_newton} total ({avg_newton:.1f}/step)")
    print(f"CG iters:     {total_cg} total ({avg_cg_per_newton:.1f}/Newton)")

    # Bandwidth analysis.
    # Bytes per CG iteration (read+write, fp64):
    #   SpMV:       5N (3 bands + x read, y write)
    #   dot(P,Ap):  2N
    #   axpy_pair:  6N (read X,R,P,Ap; write X,R)
    #   dot(R,R):   1N (same array, served from cache)
    #   p_update:   3N (read P,R; write P)
    #   Total:     17N fp64 = 136N bytes
    # Bytes per Newton overhead (outside CG loop):
    #   residual:3N + dot:1N + jacobian:4N + negate:2N + CG_init:13N + axpy:2N = 25N
    #   Total: 25N fp64 = 200N bytes
    bytes_cg = total_cg * 17 * N * 8
    bytes_newton_overhead = total_newton * 25 * N * 8
    bytes_total = bytes_cg + bytes_newton_overhead

    peak_bw_gbs = 912.0  # RTX 3080 Ti GDDR6X theoretical peak
    achieved_bw = bytes_total / (exec_ms / 1000) / 1e9

    print(f"Data moved:   {bytes_total / 1e9:.2f} GB")
    print(
        f"Achieved BW:  {achieved_bw:.1f} GB/s ({100 * achieved_bw / peak_bw_gbs:.1f}% of {peak_bw_gbs:.0f} GB/s peak)"
    )

    # Snapshots
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
    print("Burger fast test PASSED")

    if BURGER_PLOT:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for step, U_snap in snapshots:
            label = f"t={step * dt:.4f}" if step > 0 else "initial"
            alpha = 0.4 if step == 0 else 0.5 + 0.5 * step / nsteps
            ax.plot(x_grid, U_snap, label=label, alpha=alpha)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"Viscous Burger equation  (N={N}, nu={nu}, dt={dt:.2e})")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("burger_solution_fast.png", dpi=150)
        print("Saved burger_solution_fast.png")
        plt.show()


if __name__ == "__main__":
    test_burger_fast()
