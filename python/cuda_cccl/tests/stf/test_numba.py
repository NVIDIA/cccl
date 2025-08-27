# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba
import numpy as np
import pytest
from numba import cuda

numba.config.CUDA_ENABLE_PYNVJITLINK = 1

from cuda.cccl.experimental.stf._stf_bindings import (
    context,
    data_place,
    exec_place,
    read,
    rw,
    write,
)


@cuda.jit
def axpy(a, x, y):
    i = cuda.grid(1)
    if i < x.size:
        y[i] = a * x[i] + y[i]


@cuda.jit
def scale(a, x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = a * x[i]


def test_numba():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = context(use_graph=True)
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    with ctx.task(rw(lX)) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        # dX = t.get_arg_numba(0)
        dX = cuda.from_cuda_array_interface(t.get_arg_cai(0), owner=None, sync=False)
        scale[32, 64, nb_stream](2.0, dX)
        pass

    with ctx.task(read(lX), rw(lY)) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        print(nb_stream)
        dX = t.get_arg_numba(0)
        dY = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dX, dY)
        pass

    with ctx.task(read(lX), rw(lZ)) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = t.get_arg_numba(0)
        dZ = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dX, dZ)
        pass

    with ctx.task(read(lY), rw(lZ)) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dY = t.get_arg_numba(0)
        dZ = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dY, dZ)
        pass


@cuda.jit
def laplacian_5pt_kernel(u_in, u_out, dx, dy):
    """
    Compute a 5‑point Laplacian on u_in and write the result to u_out.

    Grid‑stride 2‑D kernel.  Assumes C‑contiguous (row‑major) inputs.
    Boundary cells are copied unchanged.
    """
    coef_x = 1.0 / (dx * dx)
    coef_y = 1.0 / (dy * dy)

    i, j = cuda.grid(2)  # i ↔ row (x‑index), j ↔ col (y‑index)
    nx, ny = u_in.shape

    if i >= nx or j >= ny:
        return  # out‑of‑bounds threads do nothing

    if 0 < i < nx - 1 and 0 < j < ny - 1:
        u_out[i, j] = (u_in[i - 1, j] - 2.0 * u_in[i, j] + u_in[i + 1, j]) * coef_x + (
            u_in[i, j - 1] - 2.0 * u_in[i, j] + u_in[i, j + 1]
        ) * coef_y
    else:
        # simple Dirichlet/Neumann placeholder: copy input to output
        u_out[i, j] = u_in[i, j]


def test_numba2d():
    nx, ny = 1024, 1024
    dx = 2.0 * np.pi / (nx - 1)
    dy = 2.0 * np.pi / (ny - 1)

    # a smooth test field: f(x,y) = sin(x) * cos(y)
    x = np.linspace(0, 2 * np.pi, nx, dtype=np.float64)
    y = np.linspace(0, 2 * np.pi, ny, dtype=np.float64)

    u = np.sin(x)[:, None] * np.cos(y)[None, :]  # shape = (nx, ny)
    u_out = np.zeros_like(u)

    ctx = context()
    lu = ctx.logical_data(u)
    lu_out = ctx.logical_data(u_out)

    with ctx.task(read(lu), write(lu_out)) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        du = t.get_arg_numba(0)
        du_out = t.get_arg_numba(1)
        threads_per_block = (16, 16)  # 256 threads per block is a solid starting point
        blocks_per_grid = (
            (nx + threads_per_block[0] - 1) // threads_per_block[0],
            (ny + threads_per_block[1] - 1) // threads_per_block[1],
        )
        laplacian_5pt_kernel[blocks_per_grid, threads_per_block, nb_stream](
            du, du_out, dx, dy
        )
        pass

    ctx.finalize()

    u_out_ref = np.zeros_like(u)

    for i in range(1, nx - 1):  # skip boundaries
        for j in range(1, ny - 1):
            u_out_ref[i, j] = (u[i - 1, j] - 2.0 * u[i, j] + u[i + 1, j]) / dx**2 + (
                u[i, j - 1] - 2.0 * u[i, j] + u[i, j + 1]
            ) / dy**2

    # copy boundaries
    u_out_ref[0, :] = u[0, :]
    u_out_ref[-1, :] = u[-1, :]
    u_out_ref[:, 0] = u[:, 0]
    u_out_ref[:, -1] = u[:, -1]

    # compare with the GPU result
    max_abs_diff = np.abs(u_out - u_out_ref).max()
    print(f"max(|gpu - ref|) = {max_abs_diff:.3e}")


def test_numba_exec_place():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    with ctx.task(exec_place.device(0), lX.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        # dX = t.get_arg_numba(0)
        dX = cuda.from_cuda_array_interface(t.get_arg_cai(0), owner=None, sync=False)
        scale[32, 64, nb_stream](2.0, dX)
        pass

    with ctx.task(exec_place.device(0), lX.read(), lY.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        print(nb_stream)
        dX = t.get_arg_numba(0)
        dY = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dX, dY)
        pass

    with ctx.task(
        exec_place.device(0), lX.read(data_place.managed()), lZ.rw(data_place.managed())
    ) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = t.get_arg_numba(0)
        dZ = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dX, dZ)
        pass

    with ctx.task(exec_place.device(0), lY.read(), lZ.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dY = t.get_arg_numba(0)
        dZ = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dY, dZ)
        pass


def test_numba_places():
    if len(list(cuda.gpus)) < 2:
        pytest.skip("Need at least 2 GPUs")
        return

    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    with ctx.task(lX.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = t.get_arg_numba(0)
        scale[32, 64, nb_stream](2.0, dX)
        pass

    with ctx.task(lX.read(), lY.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        print(nb_stream)
        dX = t.get_arg_numba(0)
        dY = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dX, dY)
        pass

    with ctx.task(exec_place.device(1), lX.read(), lZ.rw()) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dX = t.get_arg_numba(0)
        dZ = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dX, dZ)
        pass

    with ctx.task(lY.read(), lZ.rw(data_place.device(1))) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        dY = t.get_arg_numba(0)
        dZ = t.get_arg_numba(1)
        axpy[32, 64, nb_stream](2.0, dY, dZ)
        pass


if __name__ == "__main__":
    print("Running CUDASTF examples...")
    test_numba_exec_place()
