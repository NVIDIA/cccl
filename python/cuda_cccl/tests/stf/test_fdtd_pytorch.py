import math
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda as tc

from cuda.cccl.experimental.stf._stf_bindings import (
    context,
)

Plane = Literal["xy", "xz", "yz"]


def show_slice(t3d, plane="xy", index=None):
    # grab a 2D view
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

    # move to cpu numpy array
    arr = slice2d.detach().cpu().numpy()

    # imshow = "imshow" not "imread"
    plt.imshow(
        arr,
        origin="lower",
        cmap="seismic",
        vmin=-1e-2,
        vmax=1e-2,
        #        norm=SymLogNorm(linthresh=1e-8, vmin=-1e-0, vmax=1e-0)
        #         norm=LogNorm(vmin=1e-12, vmax=1e-6)
    )
    # plt.colorbar()
    plt.show(block=False)
    plt.pause(0.01)


def test_fdtd_3d_pytorch(
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
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    ctx = context()

    # allocate and initialize fields
    shape = (size_x, size_y, size_z)

    # Electric field components (initialized to zero)
    lex = ctx.logical_data_zeros(shape, dtype=np.float64)
    ley = ctx.logical_data_zeros(shape, dtype=np.float64)
    lez = ctx.logical_data_zeros(shape, dtype=np.float64)

    # Magnetic field components (initialized to zero)
    lhx = ctx.logical_data_zeros(shape, dtype=np.float64)
    lhy = ctx.logical_data_zeros(shape, dtype=np.float64)
    lhz = ctx.logical_data_zeros(shape, dtype=np.float64)

    # Material properties
    lepsilon = ctx.logical_data_full(shape, float(epsilon0), dtype=np.float64)
    lmu = ctx.logical_data_full(shape, float(mu0), dtype=np.float64)

    # CFL (same formula as example)
    dt = 0.25 * min(dx, dy, dz) * math.sqrt(epsilon0 * mu0)

    # Es (interior) = [1..N-2] along all dims -> enables i-1, j-1, k-1
    i_es, j_es, k_es = slice(1, -1), slice(1, -1), slice(1, -1)
    i_es_m, j_es_m, k_es_m = slice(0, -2), slice(0, -2), slice(0, -2)

    # Hs (base) = [0..N-2] along all dims -> enables i+1, j+1, k+1
    i_hs, j_hs, k_hs = slice(0, -1), slice(0, -1), slice(0, -1)
    i_hs_p, j_hs_p, k_hs_p = slice(1, None), slice(1, None), slice(1, None)

    # source location (single cell at center)
    cx, cy, cz = size_x // 2, size_y // 10, size_z // 2

    def source(t: float, x: float, y: float, z: float) -> float:
        # sin(k*x - omega*t) with f = 1e9 Hz
        pi = math.pi
        freq = 1.0e9
        omega = 2.0 * pi * freq
        wavelength = 3.0e8 / freq
        k = 2.0 * pi / wavelength
        return math.sin(k * x - omega * t)

    for n in range(int(timesteps)):
        # -------------------------
        # update electric fields (Es)
        # Ex(i,j,k) += (dt/(ε*dx)) * [(Hz(i,j,k)-Hz(i,j-1,k)) - (Hy(i,j,k)-Hy(i,j,k-1))]
        with (
            ctx.task(lex.rw(), lhy.read(), lhz.read(), lepsilon.read()) as t,
            tc.stream(tc.ExternalStream(t.stream_ptr())),
        ):
            ex, hy, hz, epsilon = t.tensor_arguments()
            ex[i_es, j_es, k_es] = ex[i_es, j_es, k_es] + (
                dt / (epsilon[i_es, j_es, k_es] * dx)
            ) * (
                (hz[i_es, j_es, k_es] - hz[i_es, j_es_m, k_es])
                - (hy[i_es, j_es, k_es] - hy[i_es, j_es, k_es_m])
            )

        # Ey(i,j,k) += (dt/(ε*dy)) * [(Hx(i,j,k)-Hx(i,j,k-1)) - (Hz(i,j,k)-Hz(i-1,j,k))]
        with (
            ctx.task(ley.rw(), lhx.read(), lhz.read(), lepsilon.read()) as t,
            tc.stream(tc.ExternalStream(t.stream_ptr())),
        ):
            ey, hx, hz, epsilon = t.tensor_arguments()
            ey[i_es, j_es, k_es] = ey[i_es, j_es, k_es] + (
                dt / (epsilon[i_es, j_es, k_es] * dy)
            ) * (
                (hx[i_es, j_es, k_es] - hx[i_es, j_es, k_es_m])
                - (hz[i_es, j_es, k_es] - hz[i_es_m, j_es, k_es])
            )

        # Ez(i,j,k) += (dt/(ε*dz)) * [(Hy(i,j,k)-Hy(i-1,j,k)) - (Hx(i,j,k)-Hx(i,j-1,k))]
        with (
            ctx.task(lez.rw(), lhx.read(), lhy.read(), lepsilon.read()) as t,
            tc.stream(tc.ExternalStream(t.stream_ptr())),
        ):
            ez, hx, hy, epsilon = t.tensor_arguments()
            ez[i_es, j_es, k_es] = ez[i_es, j_es, k_es] + (
                dt / (epsilon[i_es, j_es, k_es] * dz)
            ) * (
                (hy[i_es, j_es, k_es] - hy[i_es_m, j_es, k_es])
                - (hx[i_es, j_es, k_es] - hx[i_es, j_es_m, k_es])
            )

        # source at center cell
        with (
            ctx.task(lez.rw()) as t,
            tc.stream(tc.ExternalStream(t.stream_ptr())),
        ):
            ez = t.tensor_arguments()
            ez[cx, cy, cz] = ez[cx, cy, cz] + source(n * dt, cx * dx, cy * dy, cz * dz)

        # -------------------------
        # update magnetic fields (Hs)
        # Hx(i,j,k) -= (dt/(μ*dy)) * [(Ez(i,j+1,k)-Ez(i,j,k)) - (Ey(i,j,k+1)-Ey(i,j,k))]
        with (
            ctx.task(lhx.rw(), ley.read(), lez.read(), lmu.read()) as t,
            tc.stream(tc.ExternalStream(t.stream_ptr())),
        ):
            hx, ey, ez, mu = t.tensor_arguments()
            hx[i_hs, j_hs, k_hs] = hx[i_hs, j_hs, k_hs] - (
                dt / (mu[i_hs, j_hs, k_hs] * dy)
            ) * (
                (ez[i_hs, j_hs_p, k_hs] - ez[i_hs, j_hs, k_hs])
                - (ey[i_hs, j_hs, k_hs_p] - ey[i_hs, j_hs, k_hs])
            )

        # Hy(i,j,k) -= (dt/(μ*dz)) * [(Ex(i,j,k+1)-Ex(i,j,k)) - (Ez(i+1,j,k)-Ez(i,j,k))]
        with (
            ctx.task(lhy.rw(), lex.read(), lez.read(), lmu.read()) as t,
            tc.stream(tc.ExternalStream(t.stream_ptr())),
        ):
            hy, ex, ez, mu = t.tensor_arguments()
            hy[i_hs, j_hs, k_hs] = hy[i_hs, j_hs, k_hs] - (
                dt / (mu[i_hs, j_hs, k_hs] * dz)
            ) * (
                (ex[i_hs, j_hs, k_hs_p] - ex[i_hs, j_hs, k_hs])
                - (ez[i_hs_p, j_hs, k_hs] - ez[i_hs, j_hs, k_hs])
            )

        # Hz(i,j,k) -= (dt/(μ*dx)) * [(Ey(i+1,j,k)-Ey(i,j,k)) - (Ex(i,j+1,k)-Ex(i,j,k))]
        with (
            ctx.task(lhz.rw(), lex.read(), ley.read(), lmu.read()) as t,
            tc.stream(tc.ExternalStream(t.stream_ptr())),
        ):
            hz, ex, ey, mu = t.tensor_arguments()
            hz[i_hs, j_hs, k_hs] = hz[i_hs, j_hs, k_hs] - (
                dt / (mu[i_hs, j_hs, k_hs] * dx)
            ) * (
                (ey[i_hs_p, j_hs, k_hs] - ey[i_hs, j_hs, k_hs])
                - (ex[i_hs, j_hs_p, k_hs] - ex[i_hs, j_hs, k_hs])
            )

        if output_freq > 0 and (n % output_freq) == 0:
            with (
                ctx.task(lez.read()) as t,
                tc.stream(tc.ExternalStream(t.stream_ptr())),
            ):
                ez = t.tensor_arguments()
                print(f"{n}\t{ez[cx, cy, cz].item():.6e}")
                show_slice(ez, plane="xy")
            pass

    ctx.finalize()


if __name__ == "__main__":
    # Run FDTD simulation
    print("Running FDTD 3D PyTorch example...")
    test_fdtd_3d_pytorch(timesteps=1000, output_freq=5)
