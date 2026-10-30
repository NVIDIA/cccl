# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

numba = pytest.importorskip("numba")
pytest.importorskip("numba.cuda")
from numba import cuda  # noqa: E402

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop.numba import jit  # noqa: E402


@jit
def laplacian_5pt_kernel(u_in, u_out, dx, dy):
    """
    Compute a 5-point Laplacian on u_in and write the result to u_out.

    Grid-stride 2-D kernel. Assumes C-contiguous (row-major) inputs.
    Boundary cells are copied unchanged.
    """
    coef_x = 1.0 / (dx * dx)
    coef_y = 1.0 / (dy * dy)

    i, j = cuda.grid(2)  # i <-> row (x-index), j <-> col (y-index)
    nx, ny = u_in.shape

    if i >= nx or j >= ny:
        return  # out-of-bounds threads do nothing

    if 0 < i < nx - 1 and 0 < j < ny - 1:
        u_out[i, j] = (u_in[i - 1, j] - 2.0 * u_in[i, j] + u_in[i + 1, j]) * coef_x + (
            u_in[i, j - 1] - 2.0 * u_in[i, j] + u_in[i, j + 1]
        ) * coef_y
    else:
        # simple Dirichlet/Neumann placeholder: copy input to output
        u_out[i, j] = u_in[i, j]


def test_numba2d(monkeypatch):
    monkeypatch.setattr(numba.cuda.config, "CUDA_LOW_OCCUPANCY_WARNINGS", 0)

    nx, ny = 1024, 1024
    dx = 2.0 * np.pi / (nx - 1)
    dy = 2.0 * np.pi / (ny - 1)

    # a smooth test field: f(x,y) = sin(x) * cos(y)
    x = np.linspace(0, 2 * np.pi, nx, dtype=np.float64)
    y = np.linspace(0, 2 * np.pi, ny, dtype=np.float64)

    u = np.sin(x)[:, None] * np.cos(y)[None, :]  # shape = (nx, ny)
    u_out = np.zeros_like(u)

    ctx = stf.context()
    lu = ctx.logical_data(u)
    lu_out = ctx.logical_data(u_out)

    threads_per_block = (16, 16)  # 256 threads per block is a solid starting point
    blocks_per_grid = (
        (nx + threads_per_block[0] - 1) // threads_per_block[0],
        (ny + threads_per_block[1] - 1) // threads_per_block[1],
    )

    laplacian_5pt_kernel[blocks_per_grid, threads_per_block](
        lu.read(), lu_out.write(), dx, dy
    )

    ctx.finalize()

    # Vectorized reference: starting from a copy leaves the boundary cells at
    # their input values (matching the kernel's copy-through), and the interior
    # is a single fused slice expression instead of a ~1M-iteration Python loop.
    u_out_ref = u.copy()
    u_out_ref[1:-1, 1:-1] = (
        u[:-2, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[2:, 1:-1]
    ) / dx**2 + (u[1:-1, :-2] - 2.0 * u[1:-1, 1:-1] + u[1:-1, 2:]) / dy**2

    # compare with the GPU result
    assert np.allclose(u_out, u_out_ref, rtol=1e-6, atol=1e-6)
