# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Test multi-level nesting with stackable context.

Mimics the structure of burger.cu:
  for outer in range(outer_iterations):   # Python for loop
    with graph_scope():                   # level 1
      with repeat(substeps):             # level 2 (nested)
        tasks...                          # loop body
      tasks...                            # after repeat, still inside graph_scope

Also tests: graph_scope inside graph_scope, while_loop inside graph_scope.

Requires CUDA 12.4+ for repeat/while_loop tests.
"""

import numba
import numpy as np
from numba import cuda
from numba_helpers import get_arg_numba, numba_arguments

import cuda.stf as stf

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@cuda.jit
def add_kernel(x, val):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = x[i] + val


@cuda.jit
def scale_kernel(x, alpha):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = x[i] * alpha


@cuda.jit
def copy_kernel(dst, src):
    i = cuda.grid(1)
    if i < dst.size:
        dst[i] = src[i]


@cuda.jit
def diffusion_step_kernel(u_new, u_old, n, nu_dt_over_h2):
    """Simple 1D diffusion: u_new[i] = u_old[i] + nu*dt/h^2 * (u[i-1] - 2*u[i] + u[i+1])"""
    i = cuda.grid(1)
    if i <= 0 or i >= n - 1:
        return
    u_new[i] = u_old[i] + nu_dt_over_h2 * (u_old[i - 1] - 2.0 * u_old[i] + u_old[i + 1])


@cuda.jit
def compute_max_diff_kernel(residual, a, b):
    """Compute max |a[i] - b[i]| using atomic max (for convergence check)."""
    i = cuda.grid(1)
    if i < a.size:
        diff = abs(a[i] - b[i])
        cuda.atomic.max(residual, 0, diff)


@cuda.jit
def reset_scalar_kernel(s):
    s[0] = 0.0


def test_graph_scope_with_repeat():
    """
    Burger-like pattern: Python for loop > graph_scope > repeat > tasks.

    Structure:
      for outer in range(3):
        graph_scope:
          repeat(5):
            X += 1.0       (repeated 5 times)
          X *= 2.0          (once, after repeat, inside graph_scope)

    Expected: after each outer iteration, X = (X + 5) * 2
      iter 0: (0 + 5) * 2 = 10
      iter 1: (10 + 5) * 2 = 30
      iter 2: (30 + 5) * 2 = 70
    """
    n = 1024
    X_host = np.zeros(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    for outer in range(3):
        with ctx.graph_scope():
            with ctx.repeat(5):
                with ctx.task(lX.rw()) as t:
                    nb_stream = cuda.external_stream(t.stream_ptr())
                    dX = numba_arguments(t)
                    add_kernel[bpg, tpb, nb_stream](dX, 1.0)

            with ctx.task(lX.rw()) as t:
                nb_stream = cuda.external_stream(t.stream_ptr())
                dX = numba_arguments(t)
                scale_kernel[bpg, tpb, nb_stream](dX, 2.0)

    ctx.finalize()

    assert np.allclose(X_host, 70.0), f"Expected 70.0, got {X_host[0]}"


def test_graph_scope_with_while_loop():
    """
    Newton-like pattern: Python for loop > graph_scope > while_loop > tasks.

    Simple iterative refinement: X += 0.1 until X > 1.0.
    Uses a scalar residual to control the while loop.

    Structure:
      for outer in range(2):
        graph_scope:
          while_loop (residual > threshold):
            X += 0.1
            residual = max(|target - X|)
          X *= -1    (after convergence, negate)

    Starting from X=0, first loop: X reaches ~1.0, then negated to ~-1.0
    Second loop: X reaches ~0.0, then negated to ~0.0
    """
    n = 256
    X_host = np.zeros(n, dtype=np.float64)
    target = 1.0
    step = 0.1
    tol = 0.05

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")
    lresidual = ctx.logical_data_empty((1,), np.float64, name="residual")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    # Outer iteration with graph_scope > while_loop nesting
    with ctx.graph_scope():
        with ctx.while_loop() as loop:
            # Reset residual
            with ctx.task(lresidual.write()) as t:
                nb_stream = cuda.external_stream(t.stream_ptr())
                dres = numba_arguments(t)
                reset_scalar_kernel[1, 1, nb_stream](dres)

            # Step toward target
            with ctx.task(lX.rw()) as t:
                nb_stream = cuda.external_stream(t.stream_ptr())
                dX = numba_arguments(t)
                add_kernel[bpg, tpb, nb_stream](dX, step)

            # Compute max |target - X| as residual
            with ctx.task(lX.read(), lresidual.rw()) as t:
                nb_stream = cuda.external_stream(t.stream_ptr())
                dX = get_arg_numba(t, 0)
                dres = get_arg_numba(t, 1)
                compute_max_diff_target[bpg, tpb, nb_stream](dres, dX, target)

            loop.continue_while(lresidual, ">", tol)

        # After convergence, scale by 2 (still inside graph_scope)
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            scale_kernel[bpg, tpb, nb_stream](dX, 2.0)

    ctx.finalize()

    # X should be ~2.0 (converged to ~1.0, then scaled by 2)
    assert np.allclose(X_host, 2.0, atol=0.2), f"Expected ~2.0, got {X_host[0]}"


@cuda.jit
def compute_max_diff_target(residual, x, target):
    """max |target - x[i]| via atomic max."""
    i = cuda.grid(1)
    if i < x.size:
        diff = abs(target - x[i])
        cuda.atomic.max(residual, 0, diff)


def test_diffusion_timestep_nesting():
    """
    Full burger-like structure for 1D diffusion:
      for outer in range(outer_iters):    # Python for loop
        graph_scope:                       # level 1
          repeat(substeps):               # level 2
            diffusion_step(U)             # inner loop body
          copy U_snap <- U                # snapshot after substeps

    This is the exact nesting pattern from burger.cu, applied to
    a simpler PDE (heat equation instead of Burgers').
    """
    n = 256
    nu = 0.1
    h = 1.0 / (n - 1)
    dt = 0.4 * h * h / nu  # stable explicit time step

    U_host = np.zeros(n, dtype=np.float64)
    U_snap_host = np.zeros(n, dtype=np.float64)

    # Initial condition: sin(pi*x)
    x = np.linspace(0, 1, n)
    U_host[:] = np.sin(np.pi * x)
    U_host[0] = 0.0
    U_host[-1] = 0.0

    ctx = stf.stackable_context()
    lU = ctx.logical_data(U_host, name="U")
    lU_new = ctx.logical_data_empty((n,), np.float64, name="U_new")
    lU_snap = ctx.logical_data(U_snap_host, name="U_snap")

    tpb = 256
    bpg = (n + tpb - 1) // tpb
    coeff = nu * dt / (h * h)

    outer_iters = 3
    substeps = 10

    for outer in range(outer_iters):
        with ctx.graph_scope():
            with ctx.repeat(substeps):
                # Diffusion step: U_new = U + coeff * laplacian(U)
                with ctx.task(lU.read(), lU_new.write()) as t:
                    nb_stream = cuda.external_stream(t.stream_ptr())
                    dU = get_arg_numba(t, 0)
                    dU_new = get_arg_numba(t, 1)
                    diffusion_step_kernel[bpg, tpb, nb_stream](dU_new, dU, n, coeff)

                # Copy back: U <- U_new
                with ctx.task(lU.write(), lU_new.read()) as t:
                    nb_stream = cuda.external_stream(t.stream_ptr())
                    dU = get_arg_numba(t, 0)
                    dU_new = get_arg_numba(t, 1)
                    copy_kernel[bpg, tpb, nb_stream](dU, dU_new)

            # Snapshot after substeps (still inside graph_scope, after repeat)
            with ctx.task(lU_snap.write(), lU.read()) as t:
                nb_stream = cuda.external_stream(t.stream_ptr())
                dU_snap = get_arg_numba(t, 0)
                dU = get_arg_numba(t, 1)
                copy_kernel[bpg, tpb, nb_stream](dU_snap, dU)

    ctx.finalize()

    total_steps = outer_iters * substeps
    # Analytical solution of heat equation: exp(-pi^2 * nu * t) * sin(pi*x)
    t_final = total_steps * dt
    analytical = np.exp(-(np.pi**2) * nu * t_final) * np.sin(np.pi * x)

    # Check convergence toward analytical solution (won't be exact due to discretization)
    error = np.max(np.abs(U_host - analytical))
    print(
        f"Diffusion test: {total_steps} steps, t={t_final:.4f}, max error vs analytical = {error:.6e}"
    )
    assert error < 0.1, f"Error too large: {error}"

    # Snapshot should match final U
    assert np.allclose(U_host, U_snap_host), "Snapshot doesn't match final state"


def test_nested_graph_scopes():
    """Two levels of graph_scope nesting (no repeat/while)."""
    n = 512
    X_host = np.ones(n, dtype=np.float32)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    tpb = 256
    bpg = (n + tpb - 1) // tpb

    with ctx.graph_scope():  # level 1
        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 1.0)  # X = 2.0

        with ctx.graph_scope():  # level 2 (nested graph inside graph)
            with ctx.task(lX.rw()) as t:
                nb_stream = cuda.external_stream(t.stream_ptr())
                dX = numba_arguments(t)
                scale_kernel[bpg, tpb, nb_stream](dX, 3.0)  # X = 6.0

        with ctx.task(lX.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dX = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dX, 4.0)  # X = 10.0

    ctx.finalize()

    assert np.allclose(X_host, 10.0), f"Expected 10.0, got {X_host[0]}"


def test_repeat_with_while_inside():
    """
    Burger-like 3-level nesting: graph_scope > repeat > while_loop.

    Mimics the burger.cu pattern where each repeated substep runs
    an iterative solver (Newton/CG) until convergence.

    Structure:
      graph_scope:                          # level 1
        repeat(3):                          # level 2 (conditional)
          while_loop (residual > tol):      # level 3 (conditional inside conditional)
            X += 0.25
            residual = max(|target - X|)
          target += 1.0                     # after convergence, raise target (inside repeat)

    Starting from X=0, target=1.0:
      repeat iter 0: while converges X to ~1.0, target becomes 2.0
      repeat iter 1: while converges X to ~2.0, target becomes 3.0
      repeat iter 2: while converges X to ~3.0, target becomes 4.0
    Final X ~ 3.0, target ~ 4.0
    """
    n = 256
    X_host = np.zeros(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")
    lresidual = ctx.logical_data_empty((1,), np.float64, name="residual")

    # Target is a scalar on device — starts at 1.0
    target_host = np.array([1.0], dtype=np.float64)
    ltarget = ctx.logical_data(target_host, name="target")

    tpb = 256
    bpg = (n + tpb - 1) // tpb
    step = 0.25
    tol = 0.1

    with ctx.graph_scope():  # level 1
        with ctx.repeat(3):  # level 2
            with ctx.while_loop() as loop:  # level 3
                # Reset residual
                with ctx.task(lresidual.write()) as t:
                    nb_stream = cuda.external_stream(t.stream_ptr())
                    dres = numba_arguments(t)
                    reset_scalar_kernel[1, 1, nb_stream](dres)

                # Step X toward current target
                with ctx.task(lX.rw()) as t:
                    nb_stream = cuda.external_stream(t.stream_ptr())
                    dX = numba_arguments(t)
                    add_kernel[bpg, tpb, nb_stream](dX, step)

                # Compute residual = max |target - X|
                with ctx.task(lX.read(), ltarget.read(), lresidual.rw()) as t:
                    nb_stream = cuda.external_stream(t.stream_ptr())
                    dX = get_arg_numba(t, 0)
                    dtarget = get_arg_numba(t, 1)
                    dres = get_arg_numba(t, 2)
                    compute_max_diff_arrays[bpg, tpb, nb_stream](dres, dX, dtarget, n)

                loop.continue_while(lresidual, ">", tol)

            # After while converges, bump target by 1 (inside repeat, after while)
            with ctx.task(ltarget.rw()) as t:
                nb_stream = cuda.external_stream(t.stream_ptr())
                dtarget = numba_arguments(t)
                add_kernel[1, 1, nb_stream](dtarget, 1.0)

    ctx.finalize()

    # After 3 repeat iterations, X should have converged to ~3.0
    assert np.allclose(X_host, 3.0, atol=step + tol), f"Expected ~3.0, got {X_host[0]}"
    print(f"repeat > while test passed: X = {X_host[0]:.4f}")


@cuda.jit
def compute_max_diff_arrays(residual, a, b, n):
    """max |a[i] - b[0]| via atomic max (b is a scalar broadcast)."""
    i = cuda.grid(1)
    if i < n:
        diff = abs(a[i] - b[0])
        cuda.atomic.max(residual, 0, diff)


def test_while_with_repeat_inside():
    """
    Inverted nesting: graph_scope > while_loop > repeat.

    Each while iteration runs a fixed batch of repeat steps,
    then checks convergence.

    Structure:
      graph_scope:                              # level 1
        while_loop (residual > tol):            # level 2 (conditional)
          repeat(5):                            # level 3 (conditional inside conditional)
            X += 0.1                            # small steps
          residual = max(|target - X|)          # check after 5 steps

    Starting from X=0, target=2.0, step=0.1:
      Each while iteration does X += 5*0.1 = 0.5, then checks.
      iter 0: X=0.5, residual=1.5 > 0.1 → continue
      iter 1: X=1.0, residual=1.0 > 0.1 → continue
      iter 2: X=1.5, residual=0.5 > 0.1 → continue
      iter 3: X=2.0, residual=0.0 ≤ 0.1 → stop
    Final X ~ 2.0
    """
    n = 256
    X_host = np.zeros(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")
    lresidual = ctx.logical_data_empty((1,), np.float64, name="residual")

    target_host = np.array([2.0], dtype=np.float64)
    ltarget = ctx.logical_data(target_host, name="target")
    ltarget.set_read_only()

    tpb = 256
    bpg = (n + tpb - 1) // tpb
    step = 0.1
    tol = 0.1

    with ctx.graph_scope():  # level 1
        with ctx.while_loop() as loop:  # level 2
            with ctx.repeat(5):  # level 3
                with ctx.task(lX.rw()) as t:
                    nb_stream = cuda.external_stream(t.stream_ptr())
                    dX = numba_arguments(t)
                    add_kernel[bpg, tpb, nb_stream](dX, step)

            # After 5 small steps, measure residual
            with ctx.task(lresidual.write()) as t:
                nb_stream = cuda.external_stream(t.stream_ptr())
                dres = numba_arguments(t)
                reset_scalar_kernel[1, 1, nb_stream](dres)

            with ctx.task(lX.read(), ltarget.read(), lresidual.rw()) as t:
                nb_stream = cuda.external_stream(t.stream_ptr())
                dX = get_arg_numba(t, 0)
                dtarget = get_arg_numba(t, 1)
                dres = get_arg_numba(t, 2)
                compute_max_diff_arrays[bpg, tpb, nb_stream](dres, dX, dtarget, n)

            loop.continue_while(lresidual, ">", tol)

    ctx.finalize()

    assert np.allclose(X_host, 2.0, atol=5 * step + tol), (
        f"Expected ~2.0, got {X_host[0]}"
    )
    print(f"while > repeat test passed: X = {X_host[0]:.4f}")


def test_repeat_with_while_inside_pytorch():
    """
    Same 3-level nesting as test_repeat_with_while_inside but using PyTorch.

    Structure:
      graph_scope:                          # level 1
        repeat(3):                          # level 2 (conditional)
          while_loop (residual > tol):      # level 3 (conditional inside conditional)
            X += step
            residual = max(|target - X|)
          target += 1.0                     # after convergence, raise target

    Starting from X=0, target=1.0:
      repeat iter 0: while converges X to ~1.0, target becomes 2.0
      repeat iter 1: while converges X to ~2.0, target becomes 3.0
      repeat iter 2: while converges X to ~3.0, target becomes 4.0
    """
    import torch
    from pytorch_task import pytorch_task

    n = 256
    X_host = np.zeros(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")
    lresidual = ctx.logical_data_empty((1,), np.float64, name="residual")

    target_host = np.array([1.0], dtype=np.float64)
    ltarget = ctx.logical_data(target_host, name="target")

    step = 0.25
    tol = 0.1

    with ctx.graph_scope():
        with ctx.repeat(3):
            with ctx.while_loop() as loop:
                with pytorch_task(ctx, lX.rw()) as (tX,):
                    tX[:] += step

                with pytorch_task(
                    ctx, lX.read(), ltarget.read(), lresidual.write()
                ) as (tX, tTarget, tRes):
                    tRes[0] = torch.max(torch.abs(tX - tTarget[0]))

                loop.continue_while(lresidual, ">", tol)

            with pytorch_task(ctx, ltarget.rw()) as (tTarget,):
                tTarget[0] += 1.0

    ctx.finalize()

    assert np.allclose(X_host, 3.0, atol=step + tol), f"Expected ~3.0, got {X_host[0]}"
    print(f"repeat > while (PyTorch) passed: X = {X_host[0]:.4f}")


def test_while_with_repeat_inside_pytorch():
    """
    Same 3-level inverted nesting using PyTorch.

    Structure:
      graph_scope:                              # level 1
        while_loop (residual > tol):            # level 2
          repeat(5):                            # level 3
            X += 0.1
          residual = max(|target - X|)
    """
    import torch
    from pytorch_task import pytorch_task

    n = 256
    X_host = np.zeros(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")
    lresidual = ctx.logical_data_empty((1,), np.float64, name="residual")

    target_host = np.array([2.0], dtype=np.float64)
    ltarget = ctx.logical_data(target_host, name="target")
    ltarget.set_read_only()

    step = 0.1
    tol = 0.1

    with ctx.graph_scope():
        with ctx.while_loop() as loop:
            with ctx.repeat(5):
                with pytorch_task(ctx, lX.rw()) as (tX,):
                    tX[:] += step

            with pytorch_task(ctx, lX.read(), ltarget.read(), lresidual.write()) as (
                tX,
                tTarget,
                tRes,
            ):
                tRes[0] = torch.max(torch.abs(tX - tTarget[0]))

            loop.continue_while(lresidual, ">", tol)

    ctx.finalize()

    assert np.allclose(X_host, 2.0, atol=5 * step + tol), (
        f"Expected ~2.0, got {X_host[0]}"
    )
    print(f"while > repeat (PyTorch) passed: X = {X_host[0]:.4f}")


if __name__ == "__main__":
    print("=== test_nested_graph_scopes ===")
    test_nested_graph_scopes()

    print("\n=== test_graph_scope_with_repeat ===")
    test_graph_scope_with_repeat()

    print("\n=== test_diffusion_timestep_nesting ===")
    test_diffusion_timestep_nesting()

    print("\n=== test_graph_scope_with_while_loop ===")
    test_graph_scope_with_while_loop()

    print("\n=== test_repeat_with_while_inside ===")
    test_repeat_with_while_inside()

    print("\n=== test_while_with_repeat_inside ===")
    test_while_with_repeat_inside()

    print("\n=== test_repeat_with_while_inside_pytorch ===")
    test_repeat_with_while_inside_pytorch()

    print("\n=== test_while_with_repeat_inside_pytorch ===")
    test_while_with_repeat_inside_pytorch()

    print("\nAll nested stackable tests passed!")
