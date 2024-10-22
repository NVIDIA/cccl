//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>

using namespace cuda::experimental::stf;

// FIXME : MSVC has trouble with box constructors
#if !defined(_CCCL_COMPILER_MSVC)
void write_vtk_2D(const std::string& filename, slice<double, 3> Ez, double dx, double dy, double /*unused*/)
{
  FILE* f = fopen(filename.c_str(), "w");

  const size_t pos_z = Ez.extent(2) / 2;
  const size_t nx    = Ez.extent(0);

  const size_t size = Ez.extent(0) * Ez.extent(1);

  fprintf(f, "# vtk DataFile Version 3.0\n");
  fprintf(f, "vtk output\n");
  fprintf(f, "ASCII\n");
  fprintf(f, "DATASET UNSTRUCTURED_GRID\n");
  fprintf(f, "POINTS %ld float\n", 4 * size);

  for (size_t y = 0; y < Ez.extent(1); y++)
  {
    for (size_t x = 0; x < Ez.extent(0); x++)
    {
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(x + 0), dy * static_cast<float>(y + 0));
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(x + 1), dy * static_cast<float>(y + 0));
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(x + 1), dy * static_cast<float>(y + 1));
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(x + 0), dy * static_cast<float>(y + 1));
    }
  }

  fprintf(f, "CELLS %ld %ld\n", size, 5 * size);

  size_t cell_id = 0;
  for (size_t y = 0; y < Ez.extent(1); y++)
  {
    for (size_t x = 0; x < Ez.extent(0); x++)
    {
      const size_t point_offset = cell_id * 4;
      fprintf(f,
              "4 %d %d %d %d\n",
              (int) (point_offset + 0),
              (int) (point_offset + 1),
              (int) (point_offset + 2),
              (int) (point_offset + 3));

      cell_id++;
    }
  }

  fprintf(f, "CELL_TYPES %ld\n", size);

  for (size_t ii = 0; ii < size; ii++)
  {
    fprintf(f, "5\n");
  }

  fprintf(f, "CELL_DATA %ld\n", size);
  fprintf(f, "SCALARS Ez double 1\n");
  fprintf(f, "LOOKUP_TABLE default\n");

  for (size_t y = 0; y < Ez.extent(1); y++)
  {
    for (size_t x = 0; x < Ez.extent(0); x++)
    {
      fprintf(f, "%lf\n", Ez(x, y, pos_z));
    }
  }

  fclose(f);
}

// Define the source function
__device__ double Source(double t, double x, double y, double z)
{
  constexpr double pi         = 3.14159265358979323846;
  constexpr double freq       = 1e9;
  constexpr double omega      = (2 * pi * freq);
  constexpr double wavelength = 3e8 / freq;
  constexpr double k          = 2 * pi / wavelength;
  return sin(k * x - omega * t);
}
#endif // !defined(_CCCL_COMPILER_MSVC)

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
#if !defined(_CCCL_COMPILER_MSVC)
  stream_ctx ctx;

  // Domain dimensions
  const size_t SIZE_X = 100;
  const size_t SIZE_Y = 100;
  const size_t SIZE_Z = 100;

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

  // Initialize E
  ctx.parallel_for(data_shape, lEx.write(), lEy.write(), lEz.write())
      ->*[] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Ex, auto Ey, auto Ez) {
            Ex(i, j, k) = 0.0;
            Ey(i, j, k) = 0.0;
            Ez(i, j, k) = 0.0;
          };

  // Initialize H
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

  // Set the source function at the center of the grid

  const size_t center_x = SIZE_X / 2;
  const size_t center_y = SIZE_Y / 2;
  const size_t center_z = SIZE_Z / 2;

  // Initialize the time loop
  size_t timesteps = 10;
  if (argc > 1)
  {
    timesteps = (size_t) atol(argv[1]);
  }

  // No output by default
  int64_t output_freq = -1;
  if (argc > 2)
  {
    output_freq = (int64_t) atol(argv[2]);
  }

  /* Index shapes for Electric fields, Magnetic fields, and the indices where there is a source */
  box<3> Es({1ul, SIZE_X - 1}, {1ul, SIZE_Y - 1}, {1ul, SIZE_Z - 1});
  box<3> Hs({0ul, SIZE_X - 1}, {0ul, SIZE_Y - 1}, {0ul, SIZE_Z - 1});
  box<3> source_s({center_x, center_x + 1}, {center_y, center_y + 1}, {center_z, center_z + 1});

  for (size_t n = 0; n < timesteps; n++)
  {
    // Update the electric fields

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
    ctx.parallel_for(source_s, lEz.rw())->*[=] _CCCL_DEVICE(size_t i, size_t j, size_t k, auto Ez) {
      Ez(i, j, k) = Ez(i, j, k) + Source(n * DT, i * DX, j * DY, k * DZ);
    };

    // Update the magnetic fields

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

    if (output_freq > 0)
    {
      ctx.host_launch(lEz.read())->*[=](auto Ez) {
        // Output the electric field at the center of the grid
        fprintf(stderr, "%ld\t%le\n", n, Ez(center_x, center_y, center_z));

        if (output_freq > 0 && n % output_freq == 0)
        {
          std::string filename = "Ez" + std::to_string(n) + ".vtk";

          // Dump a 2D slice of Ez in VTK
          write_vtk_2D(filename, Ez, DX, DY, DZ);
        }
      };
    }
  }

  ctx.finalize();
#endif // !defined(_CCCL_COMPILER_MSVC)
}
