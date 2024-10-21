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
 * @brief An example solving heat equation with finite differences using the
 * parallel_for construct.
 *
 * A multi-gpu version is shown in the heat_mgpu.cu example.
 *
 * This example also illustrate how to annotate resources with set_symbol
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

void dump_iter(slice<double, 2> sUn, int iter)
{
  /* Create a binary file in the PPM format */
  char name[64];
  snprintf(name, 64, "heat_%06d.ppm", iter);
  FILE* f = fopen(name, "wb");
  fprintf(f, "P6\n%zu %zu\n255\n", sUn.extent(0), sUn.extent(1));
  for (size_t j = 0; j < sUn.extent(1); j++)
  {
    for (size_t i = 0; i < sUn.extent(0); i++)
    {
      int v = (int) (255.0 * sUn(i, j) / 100.0);
      // we assume values between 0.0 and 100.0 : max value is in red,
      // min is in blue
      unsigned char color[3];
      color[0] = static_cast<char>(v); /* red */
      color[1] = static_cast<char>(0); /* green */
      color[2] = static_cast<char>(255 - v); /* blue */
      fwrite(color, 1, 3, f);
    }
  }
  fclose(f);
}

int main()
{
  context ctx;

  const size_t N = 800;

  auto lU  = ctx.logical_data(shape_of<slice<double, 2>>(N, N));
  auto lU1 = ctx.logical_data(lU.shape());

  // Initialize the Un field with boundary conditions, and a disk at a lower
  // temperature in the middle.
  ctx.parallel_for(lU.shape(), lU.write())->*[=] _CCCL_DEVICE(size_t i, size_t j, auto U) {
    double rad = U.extent(0) / 8.0;
    double dx  = (double) i - U.extent(0) / 2;
    double dy  = (double) j - U.extent(1) / 2;

    U(i, j) = (dx * dx + dy * dy < rad * rad) ? 100.0 : 0.0;

    /* Set up boundary conditions */
    if (j == 0.0)
    {
      U(i, j) = 100.0;
    }
    if (j == U.extent(1) - 1)
    {
      U(i, j) = 0.0;
    }
    if (i == 0.0)
    {
      U(i, j) = 0.0;
    }
    if (i == U.extent(0) - 1)
    {
      U(i, j) = 0.0;
    }
  };

  // diffusion constant
  double a = 0.5;

  double dx  = 0.1;
  double dy  = 0.1;
  double dx2 = dx * dx;
  double dy2 = dy * dy;

  // time step
  double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

  double c = a * dt;

  int nsteps     = 1000;
  int image_freq = -1;

  for (int iter = 0; iter < nsteps; iter++)
  {
    if (image_freq > 0 && iter % image_freq == 0)
    {
      // Dump Un in a PPM file
      ctx.host_launch(lU.read())->*[=](auto U) {
        dump_iter(U, iter);
      };
    }

    // Update Un using Un1 value with a finite difference scheme
    ctx.parallel_for(inner<1>(lU.shape()), lU.read(), lU1.write())
        ->*[=]
      _CCCL_DEVICE(size_t i, size_t j, auto U, auto U1) {
        U1(i, j) =
          U(i, j)
          + c * ((U(i - 1, j) - 2 * U(i, j) + U(i + 1, j)) / dx2 + (U(i, j - 1) - 2 * U(i, j) + U(i, j + 1)) / dy2);
      };

    std::swap(lU, lU1);
  }

  ctx.finalize();
}
