# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Optimized PyTorch reference implementation of the viscous Burger solver.

NOTE: this is the only file under ``stf/examples/`` that intentionally
does **not** use CUDASTF. It is the non-STF baseline kept here for
direct, side-by-side comparison with :mod:`burger`, which solves the
exact same problem through STF stackable contexts and graph_scope /
while_loop / repeat composition.

This is an "as fast as we can make it without STF" implementation with
the same discretisation, parameters, and validation checks as the STF
Burger variants.

  * every numerical kernel (spmv, residual, Jacobian, ...) is
    wrapped with ``@torch.compile`` so TorchInductor can fuse the small
    elementwise / reduction ops into a handful of Triton kernels.
    Compilation happens once at import / warmup time -- *never* inside
    any capture region -- so we stay clear of Dynamo's
    ``CUDAGeneratorImpl::current_seed`` issue.

  * the BiCGSTAB inner loop only syncs every ``SOLVER_CHECK_EVERY`` iterations
    (default 4) instead of once per iteration.  A Python ``while`` is
    still what drives the loop, but the sync frequency is cut by 4x.

Loop/composition note
---------------------

This reference keeps plain Python outer loops (with reduced sync cadence)
for robustness across PyTorch versions. The STF variant in ``burger.py``
uses graph-native conditional loops.
"""

import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

BURGER_PLOT = os.environ.get("BURGER_PLOT", "") != ""

SOLVER_CHECK_EVERY = 4  # sync every N inner-solver iterations
NEWTON_CHECK_EVERY = 1  # Newton converges in a few iterations; fine to sync each


# ---------------------------------------------------------------------------
# Compiled kernels -- functional style (returns new tensors) so Inductor
# has maximum room to fuse.  Called only outside any CUDA graph capture.
# ---------------------------------------------------------------------------


@torch.compile(fullgraph=True, dynamic=False)
def spmv_fn(tVal: torch.Tensor, tX: torch.Tensor, N: int) -> torch.Tensor:
    """y = A * x, tridiagonal, CSR layout packed by assemble_jacobian_fn."""
    interior = N - 2
    last = 1 + 3 * interior

    lower = tVal[1 : 1 + 3 * interior : 3]
    diag = tVal[2 : 2 + 3 * interior : 3]
    upper = tVal[3 : 3 + 3 * interior : 3]

    y_boundary_left = tVal[:1] * tX[:1]
    y_boundary_right = tVal[last : last + 1] * tX[N - 1 : N]
    y_interior = lower * tX[0 : N - 2] + diag * tX[1 : N - 1] + upper * tX[2:N]

    return torch.cat([y_boundary_left, y_interior, y_boundary_right])


@torch.compile(fullgraph=True, dynamic=False)
def residual_fn(
    tU: torch.Tensor, tUp: torch.Tensor, N: int, h: float, dt: float, nu: float
) -> torch.Tensor:
    """F(U) for implicit Burger; Dirichlet u=0 on the boundary rows."""
    u = tU[1 : N - 1]
    u_left = tU[0 : N - 2]
    u_right = tU[2:N]
    u_prev = tUp[1 : N - 1]

    term_time = (u - u_prev) / dt
    term_conv = u * (u_right - u_left) / (2.0 * h)
    term_diff = -nu * (u_left - 2.0 * u + u_right) / (h * h)
    interior = term_time + term_conv + term_diff

    return torch.cat([tU[:1], interior, tU[N - 1 : N]])


@torch.compile(fullgraph=True, dynamic=False)
def assemble_jacobian_fn(
    tU: torch.Tensor, N: int, h: float, dt: float, nu: float
) -> torch.Tensor:
    """Packed CSR values of J = dF/dU (length 3N-4)."""
    u = tU[1 : N - 1]
    u_left = tU[0 : N - 2]
    u_right = tU[2:N]

    left = -u / (2.0 * h) - nu / (h * h)
    center = 1.0 / dt + (u_right - u_left) / (2.0 * h) + 2.0 * nu / (h * h)
    right = u / (2.0 * h) - nu / (h * h)

    band = torch.stack([left, center, right], dim=1).reshape(-1)
    one = torch.ones(1, device=tU.device, dtype=tU.dtype)
    return torch.cat([one, band, one])


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def bicgstab_solve(
    tA_val: torch.Tensor,
    tB: torch.Tensor,
    N: int,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> tuple[torch.Tensor, int]:
    """BiCGSTAB with batched-sync convergence checks."""
    device = tB.device
    dtype = tB.dtype

    tX = torch.zeros(N, device=device, dtype=dtype)
    tR = tB - spmv_fn(tA_val, tX, N)
    tRhat = tR.clone()
    tP = torch.zeros_like(tR)
    tV = torch.zeros_like(tR)
    rho_prev = torch.tensor(1.0, device=device, dtype=dtype)
    alpha = torch.tensor(1.0, device=device, dtype=dtype)
    omega = torch.tensor(1.0, device=device, dtype=dtype)

    tol_sq = tol * tol
    it = 0
    while it < max_iter:
        for _ in range(SOLVER_CHECK_EVERY):
            rho = torch.dot(tRhat, tR)
            beta = (rho / rho_prev) * (alpha / omega)
            tP = tR + beta * (tP - omega * tV)
            tV = spmv_fn(tA_val, tP, N)
            alpha = rho / torch.dot(tRhat, tV)
            tS = tR - alpha * tV
            tT = spmv_fn(tA_val, tS, N)
            omega = torch.dot(tT, tS) / torch.dot(tT, tT)
            tX = tX + alpha * tP + omega * tS
            tR = tS - omega * tT
            rho_prev = rho
            it += 1
            if it >= max_iter:
                break
        if torch.dot(tR, tR).item() <= tol_sq:
            break
    return tX, it


def newton_solve(
    tU: torch.Tensor,
    N: int,
    h: float,
    dt: float,
    nu: float,
    max_newton: int = 20,
    newton_tol: float = 1e-10,
    max_cg: int = 100,
) -> tuple[torch.Tensor, int]:
    """One implicit Burger time step."""
    tU_prev = tU.clone()
    newton_tol_sq = newton_tol * newton_tol

    it = 0
    while it < max_newton:
        tRes = residual_fn(tU, tU_prev, N, h, dt, nu)
        norm2 = torch.dot(tRes, tRes)
        tA_val = assemble_jacobian_fn(tU, N, h, dt, nu)

        tDelta, _ = bicgstab_solve(tA_val, -tRes, N, tol=1e-8, max_iter=max_cg)
        tU = tU + tDelta
        it += 1

        if it % NEWTON_CHECK_EVERY == 0 and norm2.item() <= newton_tol_sq:
            break
    return tU, it


# ---------------------------------------------------------------------------
# Warmup: force every @torch.compile cache entry to be populated BEFORE
# the main timing region, so we're not measuring Inductor codegen.
# ---------------------------------------------------------------------------


def _warmup(N: int, h: float, dt: float, nu: float) -> None:
    device = torch.device("cuda")
    dtype = torch.float64

    tU = torch.randn(N, device=device, dtype=dtype)
    tUp = torch.randn(N, device=device, dtype=dtype)
    tVal = torch.randn(3 * N - 4, device=device, dtype=dtype)
    tX = torch.randn(N, device=device, dtype=dtype)
    tB = torch.randn(N, device=device, dtype=dtype)

    spmv_fn(tVal, tX, N)
    residual_fn(tU, tUp, N, h, dt, nu)
    assemble_jacobian_fn(tU, N, h, dt, nu)
    bicgstab_solve(tVal, tB, N, tol=1e-8, max_iter=50)

    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main test -- compiled kernels + check-every-K BiCGSTAB loop
# ---------------------------------------------------------------------------


def _run_burger(N=None, nsteps=None, substeps=None, nu=0.05):
    if N is None:
        N = int(os.environ.get("BURGER_N", "2560"))
    if nsteps is None:
        nsteps = int(os.environ.get("BURGER_NSTEPS", "300"))
    if substeps is None:
        substeps = int(os.environ.get("BURGER_SUBSTEPS", "10"))
    outer_iters = nsteps // substeps
    h = 1.0 / (N - 1)
    dt = max(0.5 * h * h / nu, 0.001)

    print("=== Burger equation solver (PyTorch reference, no STF) ===")
    print(f"Grid: N={N}, h={h:.4e}")
    print(f"Time: dt={dt:.4e}, nsteps={nsteps}, substeps={substeps}")
    print(f"Physics: nu={nu}")
    print(f"BiCGSTAB sync period: every {SOLVER_CHECK_EVERY} iter")

    device = torch.device("cuda")
    dtype = torch.float64

    U_host = np.zeros(N, dtype=np.float64)
    x_grid = np.linspace(0, 1, N)
    U_host[1:-1] = np.sin(np.pi * x_grid[1:-1])

    U_init_max = float(np.max(np.abs(U_host)))
    U_init_snap = U_host.copy()

    tU = torch.from_numpy(U_host).to(device=device, dtype=dtype)
    snapshots_gpu = torch.zeros((outer_iters, N), device=device, dtype=dtype)

    t_warm = time.perf_counter()
    _warmup(N, h, dt, nu)
    t_warm = time.perf_counter() - t_warm
    print(f"Warmup (compile):  {t_warm:.2f} s")

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for outer in range(outer_iters):
        for _ in range(substeps):
            tU, _ = newton_solve(tU, N, h, dt, nu)
        snapshots_gpu[outer].copy_(tU)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    snapshots_host = snapshots_gpu.cpu().numpy()
    for i in range(outer_iters):
        step = (i + 1) * substeps
        print(
            f"Timestep {step}, t={step * dt:.4e}, max(U)={np.max(snapshots_host[i]):.6f}"
        )

    U_final = tU.detach().cpu().numpy()

    assert not np.any(np.isnan(U_final)), "NaN in solution"
    assert not np.any(np.isinf(U_final)), "Inf in solution"
    assert np.isclose(U_final[0], 0.0, atol=1e-10)
    assert np.isclose(U_final[-1], 0.0, atol=1e-10)
    assert np.max(np.abs(U_final)) < 2.0
    U_final_max = float(np.max(np.abs(U_final)))
    assert U_final_max < U_init_max

    print(f"Dissipation: {U_init_max:.6f} -> {U_final_max:.6f}")
    print(
        f"Wall time:   {elapsed:.3f} s  ({nsteps} steps, {elapsed / nsteps * 1e3:.2f} ms/step)"
    )
    print(
        f"BENCH variant=reference N={N} nsteps={nsteps} "
        f"total_s={elapsed:.6f} ms_per_step={elapsed / nsteps * 1e3:.6f}"
    )
    print("Burger reference test PASSED")

    if BURGER_PLOT:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        snapshots = [(0, U_init_snap)] + [
            ((i + 1) * substeps, snapshots_host[i].copy()) for i in range(outer_iters)
        ]
        for step, U_snap in snapshots:
            label = f"t={step * dt:.4f}" if step > 0 else "initial"
            alpha = 0.4 if step == 0 else 0.5 + 0.5 * step / nsteps
            ax.plot(x_grid, U_snap, label=label, alpha=alpha)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(
            f"Viscous Burger equation - PyTorch reference  (N={N}, nu={nu}, dt={dt:.2e})"
        )
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("burger_solution_reference.png", dpi=150)
        print("Saved burger_solution_reference.png")
        plt.show()

    return elapsed


def test_burger_pytorch_reference():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test_burger_pytorch_reference")
        return
    _run_burger()


def main():
    test_burger_pytorch_reference()


if __name__ == "__main__":
    main()
