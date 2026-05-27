# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Compiled variant of ``test_fdtd_pytorch_simplified.py``.

Each of the six FDTD stencil updates is factored into a module-level
function decorated with ``@torch.compile``. Each task opens an STF
``pytorch_task`` block, obtains torch tensor views over the STF logical
data, and hands them to the compiled kernel. The single-cell source
scatter and the host-side finite check are intentionally left
uncompiled (no fusion gain, JIT cost would dominate).

Scheduling
----------
The time loop runs inside ``ctx.repeat(chunk)`` scopes on a
``stackable_context``. Each scope body is recorded once and replayed
on device as a CUDA conditional graph. This amortizes task-launch
overhead across all iterations of the chunk.

``output_freq == 0`` runs the entire simulation as a single
``ctx.repeat(timesteps)`` block. ``output_freq > 0`` runs blocks of
``output_freq`` steps (with a trailing partial block if ``timesteps``
is not a multiple of ``output_freq``) and interleaves Python-side
diagnostic output between blocks.

Because the repeat body is *captured once* and replayed, any Python
value referenced from inside the body is frozen at its recording-time
value. The time-dependent point source therefore cannot use a
Python-side step index. Instead, a 1-element ``int64`` logical data
(``lstep``) is kept as a device-side counter: the in-graph source task
reads the counter, computes ``sin(kx - ω·step·dt)`` on device, adds it
to ``ez[cx, cy, cz]``, then increments the counter. The counter
increment is itself part of the captured graph, so it ticks correctly
on every replay.

Notes
-----
* ``fullgraph=True`` is used on the stencils so that any unintended
  Python-side op inside a stencil produces a loud Dynamo error instead
  of silently graph-breaking.
* ``mode`` is left at its default. ``mode="reduce-overhead"`` (CUDA
  graphs) must not be used here: STF hands a different stream to each
  task, which would force re-capture (or fail outright). STF's own
  ``ctx.repeat`` graph capture is what provides graph-level
  amortization for this file.
* Stencil slices are written as literal ``1:-1`` / ``0:-2`` / ``1:``
  forms. This is the most Dynamo-friendly phrasing and keeps the code
  close to the pencil-and-paper FDTD update equations.
* Scalars ``dt, dx, dy, dz`` are passed by value; Dynamo specializes on
  them. Because they are constants for a given run, no recompilation
  occurs.
* ``ctx.repeat`` requires CUDA 12.4+ (conditional CUDA graphs).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop.pytorch import pytorch_task  # noqa: E402

try:
    import matplotlib.pyplot as plt

    has_matplotlib = True
except ImportError:
    has_matplotlib = False


# ---------------------------------------------------------------------------
# Compiled stencil kernels.
#
# Each kernel is a pure elementwise update over a 3D slice. All of them are
# memory-bound; Inductor fuses the chain of temporaries into a single
# elementwise kernel, eliminating intermediate allocations.
# ---------------------------------------------------------------------------


@torch.compile(fullgraph=True)
def _update_ex(
    ex: torch.Tensor,
    hy: torch.Tensor,
    hz: torch.Tensor,
    eps: torch.Tensor,
    dt: float,
    dx: float,
) -> None:
    ex[1:-1, 1:-1, 1:-1] += (dt / (eps[1:-1, 1:-1, 1:-1] * dx)) * (
        (hz[1:-1, 1:-1, 1:-1] - hz[1:-1, 0:-2, 1:-1])
        - (hy[1:-1, 1:-1, 1:-1] - hy[1:-1, 1:-1, 0:-2])
    )


@torch.compile(fullgraph=True)
def _update_ey(
    ey: torch.Tensor,
    hx: torch.Tensor,
    hz: torch.Tensor,
    eps: torch.Tensor,
    dt: float,
    dy: float,
) -> None:
    ey[1:-1, 1:-1, 1:-1] += (dt / (eps[1:-1, 1:-1, 1:-1] * dy)) * (
        (hx[1:-1, 1:-1, 1:-1] - hx[1:-1, 1:-1, 0:-2])
        - (hz[1:-1, 1:-1, 1:-1] - hz[0:-2, 1:-1, 1:-1])
    )


@torch.compile(fullgraph=True)
def _update_ez(
    ez: torch.Tensor,
    hx: torch.Tensor,
    hy: torch.Tensor,
    eps: torch.Tensor,
    dt: float,
    dz: float,
) -> None:
    ez[1:-1, 1:-1, 1:-1] += (dt / (eps[1:-1, 1:-1, 1:-1] * dz)) * (
        (hy[1:-1, 1:-1, 1:-1] - hy[0:-2, 1:-1, 1:-1])
        - (hx[1:-1, 1:-1, 1:-1] - hx[1:-1, 0:-2, 1:-1])
    )


@torch.compile(fullgraph=True)
def _update_hx(
    hx: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
    mu: torch.Tensor,
    dt: float,
    dy: float,
) -> None:
    hx[0:-1, 0:-1, 0:-1] -= (dt / (mu[0:-1, 0:-1, 0:-1] * dy)) * (
        (ez[0:-1, 1:, 0:-1] - ez[0:-1, 0:-1, 0:-1])
        - (ey[0:-1, 0:-1, 1:] - ey[0:-1, 0:-1, 0:-1])
    )


@torch.compile(fullgraph=True)
def _update_hy(
    hy: torch.Tensor,
    ex: torch.Tensor,
    ez: torch.Tensor,
    mu: torch.Tensor,
    dt: float,
    dz: float,
) -> None:
    hy[0:-1, 0:-1, 0:-1] -= (dt / (mu[0:-1, 0:-1, 0:-1] * dz)) * (
        (ex[0:-1, 0:-1, 1:] - ex[0:-1, 0:-1, 0:-1])
        - (ez[1:, 0:-1, 0:-1] - ez[0:-1, 0:-1, 0:-1])
    )


@torch.compile(fullgraph=True)
def _update_hz(
    hz: torch.Tensor,
    ex: torch.Tensor,
    ey: torch.Tensor,
    mu: torch.Tensor,
    dt: float,
    dx: float,
) -> None:
    hz[0:-1, 0:-1, 0:-1] -= (dt / (mu[0:-1, 0:-1, 0:-1] * dx)) * (
        (ey[1:, 0:-1, 0:-1] - ey[0:-1, 0:-1, 0:-1])
        - (ex[0:-1, 1:, 0:-1] - ex[0:-1, 0:-1, 0:-1])
    )


# ---------------------------------------------------------------------------
# Visualization helper (same as the eager variant).
# ---------------------------------------------------------------------------


def show_slice(t3d, plane="xy", index=None):
    """Display a 2D slice of a 3D tensor (requires matplotlib)."""
    if not has_matplotlib:
        return

    if plane == "xy":
        idx = t3d.shape[2] // 2 if index is None else index
        slice2d = t3d[:, :, idx]
    elif plane == "xz":
        idx = t3d.shape[1] // 2 if index is None else index
        slice2d = t3d[:, idx, :]
    elif plane == "yz":
        idx = t3d.shape[0] // 2 if index is None else index
        slice2d = t3d[idx, :, :]
    else:
        raise ValueError("plane must be 'xy', 'xz' or 'yz'")

    arr = slice2d.detach().cpu().numpy()

    plt.imshow(arr, origin="lower", cmap="seismic", vmin=-1e-2, vmax=1e-2)
    plt.show(block=False)
    plt.pause(0.01)


# ---------------------------------------------------------------------------
# Test driver.
# ---------------------------------------------------------------------------


def test_fdtd_3d_pytorch_simplified_compiled(
    size_x: int = 150,
    size_y: int = 150,
    size_z: int = 150,
    timesteps: int = 10,
    output_freq: int = 0,
    dx: float = 0.01,
    dy: float = 0.01,
    dz: float = 0.01,
    epsilon0: float = 8.85e-12,
    mu0: float = 1.256e-6,
) -> None:
    """
    FDTD 3D with per-task stencils compiled via ``torch.compile`` and
    the time loop running inside ``ctx.repeat(chunk)`` CUDA-graph scopes.
    """
    ctx = stf.stackable_context()

    shape = (size_x, size_y, size_z)

    # stackable_context has no logical_data_zeros / logical_data_full;
    # allocate host buffers and let STF migrate them on first device use.
    ex_host = np.zeros(shape, dtype=np.float64)
    ey_host = np.zeros(shape, dtype=np.float64)
    ez_host = np.zeros(shape, dtype=np.float64)
    hx_host = np.zeros(shape, dtype=np.float64)
    hy_host = np.zeros(shape, dtype=np.float64)
    hz_host = np.zeros(shape, dtype=np.float64)
    epsilon_host = np.full(shape, float(epsilon0), dtype=np.float64)
    mu_host = np.full(shape, float(mu0), dtype=np.float64)
    step_host = np.zeros((1,), dtype=np.int64)

    lex = ctx.logical_data(ex_host, name="ex")
    ley = ctx.logical_data(ey_host, name="ey")
    lez = ctx.logical_data(ez_host, name="ez")
    lhx = ctx.logical_data(hx_host, name="hx")
    lhy = ctx.logical_data(hy_host, name="hy")
    lhz = ctx.logical_data(hz_host, name="hz")
    lepsilon = ctx.logical_data(epsilon_host, name="epsilon")
    lmu = ctx.logical_data(mu_host, name="mu")
    # Device-side step counter; incremented once per captured replay.
    lstep = ctx.logical_data(step_host, name="step")

    dt = 0.25 * min(dx, dy, dz) * math.sqrt(epsilon0 * mu0)

    cx, cy, cz = size_x // 2, size_y // 10, size_z // 2

    # Source-term constants pulled out so the in-graph body stays tight.
    freq = 1.0e9
    omega = 2.0 * math.pi * freq
    wavelength = 3.0e8 / freq
    kw = 2.0 * math.pi / wavelength
    kx_phase = kw * (cx * dx)

    # One full FDTD timestep, factored so it can be used both for a
    # pre-repeat warmup pass (to trigger torch.compile tracing outside
    # any CUDA graph capture) and inside ctx.repeat() for replay.
    def _timestep_body():
        # --- E updates ---
        with pytorch_task(ctx, lex.rw(), lhy.read(), lhz.read(), lepsilon.read()) as (
            ex,
            hy,
            hz,
            epsilon,
        ):
            _update_ex(ex, hy, hz, epsilon, dt, dx)

        with pytorch_task(ctx, ley.rw(), lhx.read(), lhz.read(), lepsilon.read()) as (
            ey,
            hx,
            hz,
            epsilon,
        ):
            _update_ey(ey, hx, hz, epsilon, dt, dy)

        with pytorch_task(ctx, lez.rw(), lhx.read(), lhy.read(), lepsilon.read()) as (
            ez,
            hx,
            hy,
            epsilon,
        ):
            _update_ez(ez, hx, hy, epsilon, dt, dz)

        # Time-dependent point source: use the device counter so the
        # sinusoidal amplitude is correct across captured-graph replays.
        # Not worth compiling (single-cell scatter).
        with pytorch_task(ctx, lez.rw(), lstep.rw()) as (ez, step):
            t_dev = step.to(torch.float64) * dt
            ez[cx, cy, cz] = ez[cx, cy, cz] + torch.sin(kx_phase - omega * t_dev[0])
            step.add_(1)

        # --- H updates ---
        with pytorch_task(ctx, lhx.rw(), ley.read(), lez.read(), lmu.read()) as (
            hx,
            ey,
            ez,
            mu,
        ):
            _update_hx(hx, ey, ez, mu, dt, dy)

        with pytorch_task(ctx, lhy.rw(), lex.read(), lez.read(), lmu.read()) as (
            hy,
            ex,
            ez,
            mu,
        ):
            _update_hy(hy, ex, ez, mu, dt, dz)

        with pytorch_task(ctx, lhz.rw(), lex.read(), ley.read(), lmu.read()) as (
            hz,
            ex,
            ey,
            mu,
        ):
            _update_hz(hz, ex, ey, mu, dt, dx)

    total = int(timesteps)

    # Warmup pass: runs the first real timestep outside any ctx.repeat
    # scope so that torch.compile's Dynamo tracing (which internally
    # calls torch.cuda.get_rng_state(), illegal during CUDA stream
    # capture) happens here, not inside the captured graph. After this,
    # all six stencils are cached and the repeat scopes can capture
    # cleanly. The warmup step is a physically valid timestep, not a
    # throwaway - it advances the simulation by exactly one step.
    if total > 0:
        _timestep_body()
        total -= 1

    # One pass if output_freq == 0, else output_freq-sized chunks plus a
    # final short chunk if remaining steps are not a multiple of
    # output_freq.
    chunk_size = output_freq if output_freq > 0 else total
    n = 0
    while n < total:
        chunk = min(chunk_size, total - n)

        with ctx.repeat(chunk):
            # Explicitly import epsilon and mu as read-only for this
            # scope. Without this, STF auto-pushes them as RW (the
            # conservative default), which forces serialization of the
            # six sibling stencil tasks that only read them. The
            # "no write access on data pushed with a write mode"
            # warnings are STF telling us this is happening.
            lepsilon.push(stf.AccessMode.READ)
            lmu.push(stf.AccessMode.READ)

            _timestep_body()

        n += chunk

        # Diagnostics live outside the repeat so the print / matplotlib
        # calls run once per chunk (not once per replay).
        if output_freq > 0:
            with pytorch_task(ctx, lez.read()) as (ez,):
                # n counts steps after the warmup; add 1 for the warmup
                # step so the printed index matches absolute sim time.
                print(f"{n + 1}\t{ez[cx, cy, cz].item():.6e}")
                if has_matplotlib:
                    show_slice(ez, plane="xy")

    def _check_finite(*arrays):
        for arr in arrays:
            assert np.isfinite(arr).all(), "FDTD produced non-finite values"

    ctx.host_launch(
        lex.read(),
        ley.read(),
        lez.read(),
        lhx.read(),
        lhy.read(),
        lhz.read(),
        fn=_check_finite,
    )

    ctx.finalize()


if __name__ == "__main__":
    output_freq = 50 if has_matplotlib else 0
    if not has_matplotlib and output_freq > 0:
        print("Warning: matplotlib not available, running without visualization")
        output_freq = 0
    test_fdtd_3d_pytorch_simplified_compiled(timesteps=1000, output_freq=output_freq)
