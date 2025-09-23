//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief FDTD example using the repeat_n helper function
 *
 * This shows how to refactor the original fdtd_while.cu example
 * to use the new repeat_n helper for cleaner loop patterns.
 */

#include <cuda/experimental/__stf/stackable/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

#include <stdlib.h>

using namespace cuda::experimental::stf;

// Define the source function
_CCCL_DEVICE double Source(double t, double x, double y, double z)
{
  constexpr double pi         = 3.14159265358979323846;
  constexpr double freq       = 1e9;
  constexpr double omega      = (2 * pi * freq);
  constexpr double wavelength = 3e8 / freq;
  constexpr double k          = 2 * pi / wavelength;
  return sin(k * x - omega * t);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving test: conditional nodes are only available since CUDA 12.4.\n");
  return 0;
#else
  stackable_ctx ctx;

  // Initialize the time loop
  size_t timesteps = 10;
  if (argc > 1)
  {
    timesteps = (size_t) atol(argv[1]);
  }

  // Domain dimensions (smaller for this example)
  const size_t SIZE_X = 50;
  const size_t SIZE_Y = 50;
  const size_t SIZE_Z = 50;

  // Grid spacing
  const double DX = 0.01;
  const double DY = 0.01;
  const double DZ = 0.01;

  // Define the electric and magnetic fields
  auto data_shape = shape_of<slice<double, 3>>(SIZE_X, SIZE_Y, SIZE_Z);
  auto lEx        = ctx.logical_data(data_shape);
  auto lEy        = ctx.logical_data(data_shape);
  auto lEz        = ctx.logical_data(data_shape);
  auto lHx        = ctx.logical_data(data_shape);
  auto lHy        = ctx.logical_data(data_shape);
  auto lHz        = ctx.logical_data(data_shape);

  // Define the permittivity and permeability of the medium
  auto lepsilon = ctx.logical_data(data_shape);
  auto lmu      = ctx.logical_data(data_shape);

  const double EPSILON = 8.85e-12; // Permittivity of free space
  const double MU      = 1.256e-6; // Permeability of free space

  // CFL condition DT <= min(DX, DY, DZ) * sqrt(epsilon_max * mu_max)
  double DT = 0.25 * min(min(DX, DY), DZ) * sqrt(EPSILON * MU);

  // Initialize E fields
  ctx.parallel_for(data_shape, lEx.write(), lEy.write(), lEz.write())
      ->*[] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Ex, auto Ey, auto Ez) {
            Ex(i, j, k) = 0.0;
            Ey(i, j, k) = 0.0;
            Ez(i, j, k) = 0.0;
          };

  // Initialize H fields
  ctx.parallel_for(data_shape, lHx.write(), lHy.write(), lHz.write())
      ->*[] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Hx, auto Hy, auto Hz) {
            Hx(i, j, k) = 0.0;
            Hy(i, j, k) = 0.0;
            Hz(i, j, k) = 0.0;
          };

  // Initialize permittivity and permeability fields
  ctx.parallel_for(data_shape, lepsilon.write(), lmu.write())
      ->*[=] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto epsilon, auto mu) {
            epsilon(i, j, k) = EPSILON;
            mu(i, j, k)      = MU;
          };

  // Set the source location
  const size_t center_x = SIZE_X / 2;
  const size_t center_y = SIZE_Y / 2;
  const size_t center_z = SIZE_Z / 2;

  // Index shapes for Electric fields, Magnetic fields, and the source
  box Es({1ul, SIZE_X - 1}, {1ul, SIZE_Y - 1}, {1ul, SIZE_Z - 1});
  box Hs({0ul, SIZE_X - 1}, {0ul, SIZE_Y - 1}, {0ul, SIZE_Z - 1});
  box source_s({center_x, center_x + 1}, {center_y, center_y + 1}, {center_z, center_z + 1});

  std::cout << "Running FDTD simulation for " << timesteps << " timesteps" << std::endl;
  std::cout << "Grid size: " << SIZE_X << "x" << SIZE_Y << "x" << SIZE_Z << std::endl;

  {
    auto repeat_guard = ctx.repeat_graph_scope(timesteps);

    // Update Ex
    ctx.parallel_for(Es, lEx.rw(), lHy.read(), lHz.read(), lepsilon.read())
        ->*[=]
      _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Ex, auto Hy, auto Hz, auto epsilon) {
        Ex(i, j, k) = Ex(i, j, k)
                    + (DT / (epsilon(i, j, k) * DX)) * (Hz(i, j, k) - Hz(i, j - 1, k) - Hy(i, j, k) + Hy(i, j, k - 1));
      };

    // Update Ey
    ctx.parallel_for(Es, lEy.rw(), lHx.read(), lHz.read(), lepsilon.read())
        ->*[=]
      _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Ey, auto Hx, auto Hz, auto epsilon) {
        Ey(i, j, k) = Ey(i, j, k)
                    + (DT / (epsilon(i, j, k) * DY)) * (Hx(i, j, k) - Hx(i, j, k - 1) - Hz(i, j, k) + Hz(i - 1, j, k));
      };

    // Update Ez
    ctx.parallel_for(Es, lEz.rw(), lHx.read(), lHy.read(), lepsilon.read())
        ->*[=]
      _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Ez, auto Hx, auto Hy, auto epsilon) {
        Ez(i, j, k) = Ez(i, j, k)
                    + (DT / (epsilon(i, j, k) * DZ)) * (Hy(i, j, k) - Hy(i - 1, j, k) - Hx(i, j, k) + Hx(i, j - 1, k));
      };

    // Add the source function at the center of the grid
    // Note: We could add a current iteration tracker if needed for time-dependent sources
    ctx.parallel_for(source_s, lEz.rw())->*[=] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Ez) {
      // For simplicity, using a constant source in this example
      // In the full version, you'd want to track the current timestep
      Ez(i, j, k) = Ez(i, j, k) + 0.1 * sin(0.1 * (i + j + k));
    };

    // Update Hx
    ctx.parallel_for(Hs, lHx.rw(), lEy.read(), lEz.read(), lmu.read())
        ->*[=] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Hx, auto Ey, auto Ez, auto mu) {
              Hx(i, j, k) = Hx(i, j, k)
                          - (DT / (mu(i, j, k) * DY)) * (Ez(i, j + 1, k) - Ez(i, j, k) - Ey(i, j, k + 1) + Ey(i, j, k));
            };

    // Update Hy
    ctx.parallel_for(Hs, lHy.rw(), lEx.read(), lEz.read(), lmu.read())
        ->*[=] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Hy, auto Ex, auto Ez, auto mu) {
              Hy(i, j, k) = Hy(i, j, k)
                          - (DT / (mu(i, j, k) * DZ)) * (Ex(i, j, k + 1) - Ex(i, j, k) - Ez(i + 1, j, k) + Ez(i, j, k));
            };

    // Update Hz
    ctx.parallel_for(Hs, lHz.rw(), lEx.read(), lEy.read(), lmu.read())
        ->*[=] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Hz, auto Ex, auto Ey, auto mu) {
              Hz(i, j, k) = Hz(i, j, k)
                          - (DT / (mu(i, j, k) * DX)) * (Ey(i + 1, j, k) - Ey(i, j, k) - Ex(i, j + 1, k) + Ex(i, j, k));
            };
  } // repeat_guard

  // Print final result at center
  ctx.host_launch(lEz.read())->*[=](auto Ez) {
    std::cout << "Final Ez at center: " << Ez(center_x, center_y, center_z) << std::endl;
  };

  ctx.finalize();

  std::cout << "FDTD simulation completed!" << std::endl;
  return 0;
#endif
}
