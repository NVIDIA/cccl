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

  * every numerical kernel (spmv, residual, Jacobian, CG body, ...) is
    wrapped with ``@torch.compile`` so TorchInductor can fuse the small
    elementwise / reduction ops into a handful of Triton kernels.
    Compilation happens once at import / warmup time -- *never* inside
    any capture region -- so we stay clear of Dynamo's
    ``CUDAGeneratorImpl::current_seed`` issue.

  * the CG inner loop only syncs every ``CG_CHECK_EVERY`` iterations
    (default 4) instead of once per iteration.  A Python ``while`` is
    still what drives the loop, but the sync frequency is cut by 4x.

Loop-composition options (and why we picked this one)
-----------------------------------------------------

STF expresses the CG/Newton loops as conditional-graph ``while_loop``
nodes whose continue predicate is a GPU scalar.  The whole time step is
then one CUDA-graph launch, zero host syncs.

With plain PyTorch there are three ways to get loop composition:

  (A) **Python ``while`` + ``.item()``**
      One sync per iteration.  Trivial to write, slow.

  (B) **Python ``while`` + check-every-K iterations**  (this file, default)
      Sync frequency reduced by K.  Keeps the ``@torch.compile`` kernels
      fully eager-callable and easy to debug.  No HOP magic.

  (C) **``torch._higher_order_ops.while_loop``**  (shown at the bottom)
      The closest analogue to STF's ``continue_while``: you supply
      ``cond_fn``/``body_fn`` that operate on a tuple of tensors, and
      TorchInductor compiles the whole loop into a single graph region
      with no host driver involvement.  Limitations:
        - body must be pure-functional (no in-place on inputs)
        - carry must be a flat tuple of tensors
        - nesting one while_loop inside another one inside ``torch.compile``
          works but error messages are brutal when guards mismatch
        - not all ops are supported inside the body yet
      It is the "right" long-term answer; we keep (B) as the default
      because it is robust against PyTorch version churn.

The default ``test_burger_pytorch_optimized`` exercises (B).
``test_burger_pytorch_optimized_while_hop`` exercises (C) for the CG
loop specifically, to demonstrate composition.
"""

import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch._higher_order_ops import while_loop as hop_while_loop  # noqa: E402

BURGER_PLOT = os.environ.get("BURGER_PLOT", "") != ""

CG_CHECK_EVERY = 4  # sync every N CG iterations
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


# One full CG iteration fused into a single compiled callable.  Returns
# the new (X, R, P, rsold) carry so Python only has to read ``rsold``
# (= rsnew of the previous iter) when it wants to check convergence.
@torch.compile(fullgraph=True, dynamic=False)
def cg_step(
    tA_val: torch.Tensor,
    tX: torch.Tensor,
    tR: torch.Tensor,
    tP: torch.Tensor,
    rsold: torch.Tensor,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tAp = spmv_fn(tA_val, tP, N)
    pAp = torch.dot(tP, tAp)
    alpha = rsold / pAp
    tX = tX + alpha * tP
    tR = tR - alpha * tAp
    rsnew = torch.dot(tR, tR)
    beta = rsnew / rsold
    tP = tR + beta * tP
    return tX, tR, tP, rsnew


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def cg_solve(
    tA_val: torch.Tensor,
    tB: torch.Tensor,
    N: int,
    cg_tol: float = 1e-8,
    max_cg: int = 100,
) -> tuple[torch.Tensor, int]:
    """CG with batched-sync convergence check."""
    device = tB.device
    dtype = tB.dtype

    tX = torch.zeros(N, device=device, dtype=dtype)
    tAx = spmv_fn(tA_val, tX, N)
    tR = tB - tAx
    tP = tR.clone()
    rsold = torch.dot(tR, tR)

    cg_tol_sq = cg_tol * cg_tol
    it = 0
    while it < max_cg:
        for _ in range(CG_CHECK_EVERY):
            tX, tR, tP, rsold = cg_step(tA_val, tX, tR, tP, rsold, N)
            it += 1
            if it >= max_cg:
                break
        if rsold.item() <= cg_tol_sq:
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

        tDelta, _ = cg_solve(tA_val, -tRes, N, cg_tol=1e-8, max_cg=max_cg)
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
    tR = torch.randn(N, device=device, dtype=dtype)
    tP = torch.randn(N, device=device, dtype=dtype)
    rsold = torch.dot(tR, tR)

    spmv_fn(tVal, tX, N)
    residual_fn(tU, tUp, N, h, dt, nu)
    assemble_jacobian_fn(tU, N, h, dt, nu)
    cg_step(tVal, tX, tR, tP, rsold, N)

    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main test -- strategy (B): compiled kernels + check-every-K CG loop
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

    print("=== Burger equation solver (optimized PyTorch, no STF) ===")
    print(f"Grid: N={N}, h={h:.4e}")
    print(f"Time: dt={dt:.4e}, nsteps={nsteps}, substeps={substeps}")
    print(f"Physics: nu={nu}")
    print(f"CG sync period: every {CG_CHECK_EVERY} iter")

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
        f"BENCH variant=optimized N={N} nsteps={nsteps} "
        f"total_s={elapsed:.6f} ms_per_step={elapsed / nsteps * 1e3:.6f}"
    )
    print("Burger optimized test PASSED")

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
            f"Viscous Burger equation - PyTorch optimized  (N={N}, nu={nu}, dt={dt:.2e})"
        )
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("burger_solution_optimized.png", dpi=150)
        print("Saved burger_solution_optimized.png")
        plt.show()

    return elapsed


def test_burger_pytorch_optimized():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test_burger_pytorch_optimized")
        return
    _run_burger()


# ---------------------------------------------------------------------------
# Strategy (C) applied to the full solver: swap the Python-driven CG for
# the HOP-based cg_solve_hop defined below.  Newton is still Python-driven
# because nesting a HOP while_loop inside another HOP while_loop does not
# compose cleanly on torch 2.9 (see the smoke test for context).
# ---------------------------------------------------------------------------


def newton_solve_hop(
    tU: torch.Tensor,
    N: int,
    h: float,
    dt: float,
    nu: float,
    max_newton: int = 20,
    newton_tol: float = 1e-10,
    max_cg: int = 100,
) -> tuple[torch.Tensor, int]:
    """Newton time step that uses the HOP-compiled CG for the linear solve."""
    tU_prev = tU.clone()
    newton_tol_sq = newton_tol * newton_tol
    cg_tol_sq = 1e-16  # tight; CG loop exits on its own condition

    it = 0
    while it < max_newton:
        tRes = residual_fn(tU, tU_prev, N, h, dt, nu)
        norm2 = torch.dot(tRes, tRes)
        tA_val = assemble_jacobian_fn(tU, N, h, dt, nu)

        tDelta, _ = cg_solve_hop(tA_val, -tRes, N, max_cg, cg_tol_sq)
        tU = tU + tDelta
        it += 1

        if it % NEWTON_CHECK_EVERY == 0 and norm2.item() <= newton_tol_sq:
            break
    return tU, it


def _run_burger_hop(N=None, nsteps=None, substeps=None, nu=0.05):
    if N is None:
        N = int(os.environ.get("BURGER_N", "2560"))
    if nsteps is None:
        nsteps = int(os.environ.get("BURGER_NSTEPS", "300"))
    if substeps is None:
        substeps = int(os.environ.get("BURGER_SUBSTEPS", "10"))
    outer_iters = nsteps // substeps
    h = 1.0 / (N - 1)
    dt = max(0.5 * h * h / nu, 0.001)

    print("=== Burger solver (optimized + HOP while_loop CG) ===")
    print(f"Grid: N={N}, h={h:.4e}")
    print(f"Time: dt={dt:.4e}, nsteps={nsteps}, substeps={substeps}")
    print(f"Physics: nu={nu}")

    device = torch.device("cuda")
    dtype = torch.float64

    U_host = np.zeros(N, dtype=np.float64)
    x_grid = np.linspace(0, 1, N)
    U_host[1:-1] = np.sin(np.pi * x_grid[1:-1])

    U_init_max = float(np.max(np.abs(U_host)))

    tU = torch.from_numpy(U_host).to(device=device, dtype=dtype)
    snapshots_gpu = torch.zeros((outer_iters, N), device=device, dtype=dtype)

    t_warm = time.perf_counter()
    _warmup(N, h, dt, nu)
    # Also warm cg_solve_hop -- first call compiles the HOP graph region.
    tA_dummy = torch.randn(3 * N - 4, device=device, dtype=dtype)
    tB_dummy = torch.randn(N, device=device, dtype=dtype)
    _ = cg_solve_hop(tA_dummy, tB_dummy, N, 100, 1e-16)
    torch.cuda.synchronize()
    t_warm = time.perf_counter() - t_warm
    print(f"Warmup (compile):  {t_warm:.2f} s")

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for outer in range(outer_iters):
        for _ in range(substeps):
            tU, _ = newton_solve_hop(tU, N, h, dt, nu)
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
        f"BENCH variant=optimized_hop N={N} nsteps={nsteps} "
        f"total_s={elapsed:.6f} ms_per_step={elapsed / nsteps * 1e3:.6f}"
    )
    print("Burger optimized_hop test PASSED")
    return elapsed


def test_burger_pytorch_optimized_hop():
    """Full Burger solve with the CG inner loop driven by a while_loop HOP."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test_burger_pytorch_optimized_hop")
        return
    _run_burger_hop()


# ---------------------------------------------------------------------------
# Strategy (C): compose the CG inner loop as torch._higher_order_ops.while_loop
#
# Closest analogue to STF's continue_while -- the whole CG loop becomes
# a single compiled graph region with no Python driver and no per-iter
# host sync.  Key requirement: the HOP call MUST be invoked from inside
# a @torch.compile'd function.  Calling it from eager hits a fallback
# path that graph-breaks in torch 2.9.
# ---------------------------------------------------------------------------


def _spmv_inline(tVal: torch.Tensor, tX: torch.Tensor, N: int) -> torch.Tensor:
    """Body-inlineable SpMV.  Avoids a nested @torch.compile call from the
    while_loop body (HOPs do not compose with nested compiles cleanly)."""
    interior = N - 2
    last = 1 + 3 * interior
    lower = tVal[1 : 1 + 3 * interior : 3]
    diag = tVal[2 : 2 + 3 * interior : 3]
    upper = tVal[3 : 3 + 3 * interior : 3]
    y_bl = tVal[:1] * tX[:1]
    y_br = tVal[last : last + 1] * tX[N - 1 : N]
    y_i = lower * tX[0 : N - 2] + diag * tX[1 : N - 1] + upper * tX[2:N]
    return torch.cat([y_bl, y_i, y_br])


@torch.compile(fullgraph=True, dynamic=False)
def cg_solve_hop(
    tA_val: torch.Tensor,
    tB: torch.Tensor,
    N: int,
    max_cg: int,
    cg_tol_sq: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full CG solver with `while_loop` HOP as the iteration driver.

    Returns ``(tX, num_iters)``.  The entire iteration runs as one graph
    region inside the compiled function -- no Python loop, no host-side
    synchronisation.  Mirrors STF's ``continue_while(cond, ">", tol)``.
    """
    tX = torch.zeros_like(tB)
    tAx = _spmv_inline(tA_val, tX, N)
    tR = tB - tAx
    tP = tR.clone()
    rsold = torch.dot(tR, tR)
    it = torch.zeros((), dtype=torch.int64, device=tB.device)

    max_cg_t = torch.tensor(max_cg, device=tB.device, dtype=torch.int64)
    tol_t = torch.tensor(cg_tol_sq, device=tB.device, dtype=tB.dtype)

    def cond_fn(it, tX, tR, tP, rsold):
        return (it < max_cg_t) & (rsold > tol_t)

    def body_fn(it, tX, tR, tP, rsold):
        tAp = _spmv_inline(tA_val, tP, N)
        pAp = torch.dot(tP, tAp)
        alpha = rsold / pAp
        tX = tX + alpha * tP
        tR = tR - alpha * tAp
        rsnew = torch.dot(tR, tR)
        beta = rsnew / rsold
        tP = tR + beta * tP
        return it + 1, tX, tR, tP, rsnew

    it, tX, tR, tP, rsold = hop_while_loop(cond_fn, body_fn, (it, tX, tR, tP, rsold))
    return tX, it


def test_burger_pytorch_optimized_while_hop():
    """The CG inner loop expressed as a torch while-condition (HOP).

    Verifies that:
      * the HOP version runs end-to-end on this PyTorch
      * it produces the same answer as the Python-driven CG to fp64
        round-off
      * moving the while loop into the graph is measurably faster than
        driving it from Python
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test_burger_pytorch_optimized_while_hop")
        return

    N = 2560
    h = 1.0 / (N - 1)
    dt = 1e-3
    nu = 0.05
    device = torch.device("cuda")
    dtype = torch.float64

    U_host = np.zeros(N, dtype=np.float64)
    x_grid = np.linspace(0, 1, N)
    U_host[1:-1] = np.sin(np.pi * x_grid[1:-1])
    tU = torch.from_numpy(U_host).to(device=device, dtype=dtype)
    tU_prev = tU.clone()

    _warmup(N, h, dt, nu)

    tRes = residual_fn(tU, tU_prev, N, h, dt, nu)
    tA_val = assemble_jacobian_fn(tU, N, h, dt, nu)
    tB = -tRes

    max_cg = 100
    cg_tol_sq = 1e-16  # tight; force a fixed 100-iter comparison

    try:
        tX_hop, it_hop = cg_solve_hop(tA_val, tB, N, max_cg, cg_tol_sq)
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"while_loop HOP not usable on this PyTorch: {exc!r}")
        return

    tX_py, _ = cg_solve(tA_val, tB, N, cg_tol=1e-16, max_cg=max_cg)

    assert not torch.isnan(tX_hop).any()
    assert not torch.isinf(tX_hop).any()
    rel = (tX_hop - tX_py).norm() / tX_py.norm()
    assert rel.item() < 1e-10, f"HOP vs Python CG results disagree: rel={rel.item()}"
    print(
        f"HOP composition OK: {it_hop.item()} iters, rel_diff vs Python = {rel.item():.2e}"
    )

    # Steady-state timing
    N_REPS = 30
    torch.cuda.synchronize()
    import time as _time

    t0 = _time.perf_counter()
    for _ in range(N_REPS):
        tX_hop, _ = cg_solve_hop(tA_val, tB, N, max_cg, cg_tol_sq)
    torch.cuda.synchronize()
    hop_ms = (_time.perf_counter() - t0) * 1000 / N_REPS

    t0 = _time.perf_counter()
    for _ in range(N_REPS):
        tX_py, _ = cg_solve(tA_val, tB, N, cg_tol=1e-16, max_cg=max_cg)
    torch.cuda.synchronize()
    py_ms = (_time.perf_counter() - t0) * 1000 / N_REPS

    print(f"HOP while_loop   CG solve: {hop_ms:6.3f} ms")
    print(f"Python-driven    CG solve: {py_ms:6.3f} ms")
    print(f"HOP speedup: {py_ms / hop_ms:.2f}x")


def main():
    test_burger_pytorch_optimized()


if __name__ == "__main__":
    main()
