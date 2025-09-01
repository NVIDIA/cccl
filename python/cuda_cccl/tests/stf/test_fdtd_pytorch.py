import math
from typing import Tuple, Optional

import torch

def fdtd_3d_pytorch(
    size_x: int = 100,
    size_y: int = 100,
    size_z: int = 100,
    timesteps: int = 10,
    output_freq: int = 0,
    dx: float = 0.01,
    dy: float = 0.01,
    dz: float = 0.01,
    epsilon0: float = 8.85e-12,
    mu0: float = 1.256e-6,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # allocate fields
    shape = (size_x, size_y, size_z)
    ex = torch.zeros(shape, dtype=dtype, device=device)
    ey = torch.zeros_like(ex)
    ez = torch.zeros_like(ex)

    hx = torch.zeros_like(ex)
    hy = torch.zeros_like(ex)
    hz = torch.zeros_like(ex)

    epsilon = torch.full(shape, float(epsilon0), dtype=dtype, device=device)
    mu = torch.full(shape, float(mu0), dtype=dtype, device=device)

    # CFL (same formula as example)
    dt = 0.25 * min(dx, dy, dz) * math.sqrt(epsilon0 * mu0)

    # Es (interior) = [1..N-2] along all dims -> enables i-1, j-1, k-1
    i_es, j_es, k_es = slice(1, -1), slice(1, -1), slice(1, -1)
    i_es_m, j_es_m, k_es_m = slice(0, -2), slice(0, -2), slice(0, -2)

    # Hs (base) = [0..N-2] along all dims -> enables i+1, j+1, k+1
    i_hs, j_hs, k_hs = slice(0, -1), slice(0, -1), slice(0, -1)
    i_hs_p, j_hs_p, k_hs_p = slice(1, None), slice(1, None), slice(1, None)

    # source location (single cell at center)
    cx, cy, cz = size_x // 2, size_y // 2, size_z // 2

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
        ex[i_es, j_es, k_es] = ex[i_es, j_es, k_es] + (dt / (epsilon[i_es, j_es, k_es] * dx)) * (
            (hz[i_es, j_es, k_es] - hz[i_es, j_es_m, k_es])
            - (hy[i_es, j_es, k_es] - hy[i_es, j_es, k_es_m])
        )

        # Ey(i,j,k) += (dt/(ε*dy)) * [(Hx(i,j,k)-Hx(i,j,k-1)) - (Hz(i,j,k)-Hz(i-1,j,k))]
        ey[i_es, j_es, k_es] = ey[i_es, j_es, k_es] + (dt / (epsilon[i_es, j_es, k_es] * dy)) * (
            (hx[i_es, j_es, k_es] - hx[i_es, j_es, k_es_m])
            - (hz[i_es, j_es, k_es] - hz[i_es_m, j_es, k_es])
        )

        # Ez(i,j,k) += (dt/(ε*dz)) * [(Hy(i,j,k)-Hy(i-1,j,k)) - (Hx(i,j,k)-Hx(i,j-1,k))]
        ez[i_es, j_es, k_es] = ez[i_es, j_es, k_es] + (dt / (epsilon[i_es, j_es, k_es] * dz)) * (
            (hy[i_es, j_es, k_es] - hy[i_es_m, j_es, k_es])
            - (hx[i_es, j_es, k_es] - hx[i_es, j_es_m, k_es])
        )

        # source at center cell
        ez[cx, cy, cz] = ez[cx, cy, cz] + source(n * dt, cx * dx, cy * dy, cz * dz)

        # -------------------------
        # update magnetic fields (Hs)
        # Hx(i,j,k) -= (dt/(μ*dy)) * [(Ez(i,j+1,k)-Ez(i,j,k)) - (Ey(i,j,k+1)-Ey(i,j,k))]
        hx[i_hs, j_hs, k_hs] = hx[i_hs, j_hs, k_hs] - (dt / (mu[i_hs, j_hs, k_hs] * dy)) * (
            (ez[i_hs, j_hs_p, k_hs] - ez[i_hs, j_hs, k_hs])
            - (ey[i_hs, j_hs, k_hs_p] - ey[i_hs, j_hs, k_hs])
        )

        # Hy(i,j,k) -= (dt/(μ*dz)) * [(Ex(i,j,k+1)-Ex(i,j,k)) - (Ez(i+1,j,k)-Ez(i,j,k))]
        hy[i_hs, j_hs, k_hs] = hy[i_hs, j_hs, k_hs] - (dt / (mu[i_hs, j_hs, k_hs] * dz)) * (
            (ex[i_hs, j_hs, k_hs_p] - ex[i_hs, j_hs, k_hs])
            - (ez[i_hs_p, j_hs, k_hs] - ez[i_hs, j_hs, k_hs])
        )

        # Hz(i,j,k) -= (dt/(μ*dx)) * [(Ey(i+1,j,k)-Ey(i,j,k)) - (Ex(i,j+1,k)-Ex(i,j,k))]
        hz[i_hs, j_hs, k_hs] = hz[i_hs, j_hs, k_hs] - (dt / (mu[i_hs, j_hs, k_hs] * dx)) * (
            (ey[i_hs_p, j_hs, k_hs] - ey[i_hs, j_hs, k_hs])
            - (ex[i_hs, j_hs_p, k_hs] - ex[i_hs, j_hs, k_hs])
        )

        if output_freq > 0 and (n % output_freq) == 0:
            print(f"{n}\t{ez[cx, cy, cz].item():.6e}")

    return ex, ey, ez, hx, hy, hz


if __name__ == "__main__":
    # quick check
    ex, ey, ez, hx, hy, hz = fdtd_3d_pytorch(timesteps=1000, output_freq=5)
    print("done; Ez(center) =", ez[50, 50, 50].item())
