# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Jacobi iteration with stackable context and while_loop -- Warp version.

Structural twin of ``test_jacobi_stackable_numba.py``: identical control
flow (``ctx.while_loop`` + residual-driven exit), identical numerics
(float64, same kernel logic), but kernels are ``wp.kernel`` launched via
``wp.launch(..., stream=s)`` on the stream STF hands to each task.

This is the missing piece between:
  * ``test_jacobi_stackable_numba.py``   -- numba + conditional while_loop
  * ``test_stf_in_scoped_capture.py``    -- warp + graph capture

Combining them gives the pattern Newton-style physics codebases need: a
non-linear outer loop (Newton / L-BFGS / MuJoCo constraint solver) with
data-dependent termination, whose bodies launch Warp kernels, all
captured into a single conditional CUDA graph.

Requires CUDA 12.4+ for conditional graph nodes.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import cuda.stf._experimental as stf

# ---------------------------------------------------------------------------
# STF <-> Warp glue: wp.Stream adapter cache and CAI -> wp.array helpers.
#
# Double-registering the same raw cudaStream_t with Warp corrupts its
# internal state, so we memoize one ``wp.Stream`` wrapper per (device, raw
# ptr) pair. STF's stream pool is small, so the cache stays small.
# ---------------------------------------------------------------------------


_wp_stream_cache: dict[tuple[int, int], wp.Stream] = {}


def wrap_stream(raw_ptr: int, device) -> wp.Stream:
    """Return a cached ``wp.Stream`` wrapping ``raw_ptr`` on ``device``."""
    key = (id(device), int(raw_ptr))
    s = _wp_stream_cache.get(key)
    if s is None:
        s = wp.Stream(device, cuda_stream=int(raw_ptr))
        _wp_stream_cache[key] = s
    return s


def get_arg_warp(task, index: int, dtype, shape=None, device=None) -> wp.array:
    """Alias a single task argument as a ``wp.array`` (no copy).

    STF returns each argument as an ``stf_cai`` exposing ``ptr``, ``shape``
    and ``dtype``. Warp's ``wp.array`` constructor is strict about its
    ``data=`` path (rejects non-ndarray objects even if they expose
    ``__cuda_array_interface__``), so we go through ``ptr=`` which maps
    an external allocation without taking ownership. The returned array
    is only valid for the duration of the enclosing ``with ctx.task(...)``
    block.
    """
    cai = task.get_arg_cai(index)
    cai_shape = tuple(cai.shape) if shape is None else tuple(shape)
    return wp.array(
        ptr=int(cai.ptr),
        dtype=dtype,
        shape=cai_shape,
        device=device if device is not None else wp.get_device(),
    )


# ---------------------------------------------------------------------------
# Warp kernels. Line-for-line equivalents of the Numba versions in
# ``test_jacobi_stackable_numba.py``.
# ---------------------------------------------------------------------------


@wp.kernel
def init_kernel(
    A: wp.array2d(dtype=wp.float64),
    Anew: wp.array2d(dtype=wp.float64),
):
    i, j = wp.tid()
    if i == j:
        A[i, j] = wp.float64(1.0)
    else:
        A[i, j] = wp.float64(-1.0)
    Anew[i, j] = A[i, j]


@wp.kernel
def reset_residual(residual: wp.array(dtype=wp.float64)):
    residual[0] = wp.float64(0.0)


@wp.kernel
def jacobi_step(
    A: wp.array2d(dtype=wp.float64),
    Anew: wp.array2d(dtype=wp.float64),
    residual: wp.array(dtype=wp.float64),
):
    """Interior 4-neighbour average with atomic-max residual reduction."""
    i, j = wp.tid()
    m = A.shape[0]
    n = A.shape[1]
    if i <= 0 or i >= m - 1 or j <= 0 or j >= n - 1:
        return

    Anew[i, j] = wp.float64(0.25) * (
        A[i - 1, j] + A[i + 1, j] + A[i, j - 1] + A[i, j + 1]
    )
    error = wp.abs(A[i, j] - Anew[i, j])
    wp.atomic_max(residual, 0, error)


@wp.kernel
def copy_back(
    A: wp.array2d(dtype=wp.float64),
    Anew: wp.array2d(dtype=wp.float64),
):
    i, j = wp.tid()
    m = A.shape[0]
    n = A.shape[1]
    if i > 0 and i < m - 1 and j > 0 and j < n - 1:
        A[i, j] = Anew[i, j]


# Extra kernels used by the sanity tests below (independent of Jacobi).


@wp.kernel
def scale_kernel(x: wp.array(dtype=wp.float32), alpha: wp.float32):
    i = wp.tid()
    x[i] = x[i] * alpha


@wp.kernel
def add_kernel(x: wp.array(dtype=wp.float32), val: wp.float32):
    i = wp.tid()
    x[i] = x[i] + val


@wp.kernel
def branch_kernel(
    x: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    bias: wp.float32,
):
    i = wp.tid()
    out[i] = x[i] + bias


@wp.kernel
def join4_kernel(
    b0: wp.array(dtype=wp.float32),
    b1: wp.array(dtype=wp.float32),
    b2: wp.array(dtype=wp.float32),
    b3: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    residual: wp.array(dtype=wp.float32),
    loop_iters: wp.float32,
):
    i = wp.tid()
    x[i] = b0[i] + b1[i] + b2[i] + b3[i]
    if i == 0:
        residual[0] = loop_iters


@wp.kernel
def while_body_kernel(
    x: wp.array(dtype=wp.float32),
    residual: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    x[i] = x[i] + wp.float32(1.0)
    if i == 0:
        residual[0] = residual[0] - wp.float32(1.0)


# ---------------------------------------------------------------------------
# Main Jacobi test: conditional while_loop driven by the residual.
# ---------------------------------------------------------------------------


def test_jacobi_stackable_warp():
    m, n = 256, 256
    tol = 0.1

    wp.init()
    device = wp.get_device()

    A_host = np.zeros((m, n), dtype=np.float64)
    Anew_host = np.zeros((m, n), dtype=np.float64)

    ctx = stf.stackable_context()

    lA = ctx.logical_data(A_host, name="A")
    lAnew = ctx.logical_data(Anew_host, name="Anew")
    lresidual = ctx.logical_data_empty((1,), np.float64, name="residual")

    # Initialize A and Anew.
    with ctx.task(lA.write(), lAnew.write()) as t:
        s = wrap_stream(t.stream_ptr(), device)
        dA = get_arg_warp(t, 0, wp.float64, (m, n))
        dAnew = get_arg_warp(t, 1, wp.float64, (m, n))
        wp.launch(
            init_kernel,
            dim=(m, n),
            inputs=[dA, dAnew],
            device=device,
            stream=s,
        )

    # Iterative solve with while_loop. Each iteration resets the
    # residual, does a Jacobi sweep (which atomic-maxes into residual),
    # and copies the new grid back. ``continue_while`` checks the
    # residual on-device and lets the conditional graph keep going.
    with ctx.while_loop() as loop:
        with ctx.task(lresidual.write()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            dres = get_arg_warp(t, 0, wp.float64, (1,))
            wp.launch(
                reset_residual,
                dim=1,
                inputs=[dres],
                device=device,
                stream=s,
            )

        with ctx.task(lA.read(), lAnew.write(), lresidual.rw()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            dA = get_arg_warp(t, 0, wp.float64, (m, n))
            dAnew = get_arg_warp(t, 1, wp.float64, (m, n))
            dres = get_arg_warp(t, 2, wp.float64, (1,))
            wp.launch(
                jacobi_step,
                dim=(m, n),
                inputs=[dA, dAnew, dres],
                device=device,
                stream=s,
            )

        with ctx.task(lA.rw(), lAnew.read()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            dA = get_arg_warp(t, 0, wp.float64, (m, n))
            dAnew = get_arg_warp(t, 1, wp.float64, (m, n))
            wp.launch(
                copy_back,
                dim=(m, n),
                inputs=[dA, dAnew],
                device=device,
                stream=s,
            )

        loop.continue_while(lresidual, ">", tol)

    ctx.finalize()

    print(f"Jacobi converged (Warp) with tolerance {tol}")


# ---------------------------------------------------------------------------
# Sanity tests: graph_scope and repeat with Warp kernels (no while_loop).
# These mirror the Numba test's secondary tests so the two backends can
# be compared one-for-one.
# ---------------------------------------------------------------------------


def test_graph_scope_warp():
    n = 1024
    X_host = np.ones(n, dtype=np.float32)

    wp.init()
    device = wp.get_device()

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            dX = get_arg_warp(t, 0, wp.float32, (n,))
            wp.launch(
                scale_kernel,
                dim=n,
                inputs=[dX, wp.float32(2.0)],
                device=device,
                stream=s,
            )
        with ctx.task(lX.rw()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            dX = get_arg_warp(t, 0, wp.float32, (n,))
            wp.launch(
                add_kernel,
                dim=n,
                inputs=[dX, wp.float32(1.0)],
                device=device,
                stream=s,
            )

    with ctx.graph_scope():
        with ctx.task(lX.rw()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            dX = get_arg_warp(t, 0, wp.float32, (n,))
            wp.launch(
                scale_kernel,
                dim=n,
                inputs=[dX, wp.float32(3.0)],
                device=device,
                stream=s,
            )

    ctx.finalize()

    # (1.0 * 2.0 + 1.0) * 3.0 == 9.0
    assert np.allclose(X_host, 9.0), f"Expected 9.0, got {X_host[0]}"


def test_repeat_warp():
    n = 1024
    X_host = np.zeros(n, dtype=np.float32)

    wp.init()
    device = wp.get_device()

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")

    with ctx.repeat(10):
        with ctx.task(lX.rw()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            dX = get_arg_warp(t, 0, wp.float32, (n,))
            wp.launch(
                add_kernel,
                dim=n,
                inputs=[dX, wp.float32(1.0)],
                device=device,
                stream=s,
            )

    ctx.finalize()

    assert np.allclose(X_host, 10.0), f"Expected 10.0, got {X_host[0]}"


def _submit_branch_task(ctx, lX, lout, branch_id: int, n: int, device):
    with ctx.task(lX.read(), lout.write()) as t:
        s = wrap_stream(t.stream_ptr(), device)
        dX = get_arg_warp(t, 0, wp.float32, (n,))
        dout = get_arg_warp(t, 1, wp.float32, (n,))
        wp.launch(
            branch_kernel,
            dim=n,
            inputs=[dX, dout, wp.float32(branch_id + 1)],
            device=device,
            stream=s,
        )


def _submit_join4_task(ctx, branch_lds, lX, lresidual, n: int, loop_iters: int, device):
    if len(branch_lds) != 4:
        raise ValueError("join4_kernel expects exactly four branch outputs")

    branch_deps = [ld.read() for ld in branch_lds]
    with ctx.task(*branch_deps, lX.write(), lresidual.write()) as t:
        s = wrap_stream(t.stream_ptr(), device)
        branches = [
            get_arg_warp(t, index, wp.float32, (n,)) for index in range(len(branch_lds))
        ]
        dX = get_arg_warp(t, len(branch_lds), wp.float32, (n,))
        dres = get_arg_warp(t, len(branch_lds) + 1, wp.float32, (1,))
        wp.launch(
            join4_kernel,
            dim=n,
            inputs=[
                branches[0],
                branches[1],
                branches[2],
                branches[3],
                dX,
                dres,
                wp.float32(loop_iters),
            ],
            device=device,
            stream=s,
        )


def _submit_while_body_task(ctx, lX, lresidual, n: int, device):
    with ctx.task(lX.rw(), lresidual.rw()) as t:
        s = wrap_stream(t.stream_ptr(), device)
        dX = get_arg_warp(t, 0, wp.float32, (n,))
        dres = get_arg_warp(t, 1, wp.float32, (1,))
        wp.launch(
            while_body_kernel,
            dim=n,
            inputs=[dX, dres],
            device=device,
            stream=s,
        )


def test_launchable_graph_k_branches_then_while_warp():
    """Build one launchable graph from K branch scopes, a join, and a while body."""
    n = 512
    k_branches = 4
    graph_replays = 3
    while_iters = 3

    wp.init()
    device = wp.get_device()

    X_host = np.zeros(n, dtype=np.float32)

    graph = stf.task_graph()
    ctx = graph.context
    lX = ctx.logical_data(X_host, name="X")
    branch_lds = [
        ctx.logical_data_empty((n,), np.float32, name=f"branch_{branch_id}")
        for branch_id in range(k_branches)
    ]
    lresidual = ctx.logical_data_empty((1,), np.float32, name="residual")

    with graph:
        for branch_id, lbranch in enumerate(branch_lds):
            with ctx.graph_scope():
                lX.push(stf.AccessMode.READ)
                _submit_branch_task(ctx, lX, lbranch, branch_id, n, device)

        _submit_join4_task(ctx, branch_lds, lX, lresidual, n, while_iters, device)

        with ctx.while_loop() as loop:
            _submit_while_body_task(ctx, lX, lresidual, n, device)
            loop.continue_while(lresidual, ">", 0.5)

    for _ in range(graph_replays):
        graph.launch()
    graph.reset()

    graph.finalize()

    expected = np.zeros(n, dtype=np.float32)
    branch_bias_sum = sum(range(1, k_branches + 1))
    for _ in range(graph_replays):
        expected = k_branches * expected + branch_bias_sum
        expected = expected + while_iters

    assert np.allclose(X_host, expected), f"Expected {expected[0]}, got {X_host[0]}"


if __name__ == "__main__":
    test_graph_scope_warp()
    test_repeat_warp()
    test_jacobi_stackable_warp()
    test_launchable_graph_k_branches_then_while_warp()
