import numba
import numpy as np
from numba import cuda

numba.config.CUDA_ENABLE_PYNVJITLINK = 1
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

import cuda.cccl.experimental.stf as cudastf


@cudastf.jit
def laplacian_5pt_kernel(u_in, u_out, dx, dy):
    """
    Compute a 5?~@~Qpoint Laplacian on u_in and write the result to u_out.

    Grid?~@~Qstride 2?~@~QD kernel.  Assumes C?~@~Qcontiguous (row?~@~Qmajor) inputs.
    Boundary cells are copied unchanged.
    """
    coef_x = 1.0 / (dx * dx)
    coef_y = 1.0 / (dy * dy)

    i, j = cuda.grid(2)  # i ?~F~T row (x?~@~Qindex), j ?~F~T col (y?~@~Qindex)
    nx, ny = u_in.shape

    if i >= nx or j >= ny:
        return  # out?~@~Qof?~@~Qbounds threads do nothing

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

    ctx = cudastf.context()
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
