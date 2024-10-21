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
 * @brief Reference OpenACC implementation of the FDTD example
 */
#include <string>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define Ex(i, j, k)      (Ex_array[(i) + (j) * SIZE_X + (k) * SIZE_X * SIZE_Y])
#define Ey(i, j, k)      (Ey_array[(i) + (j) * SIZE_X + (k) * SIZE_X * SIZE_Y])
#define Ez(i, j, k)      (Ez_array[(i) + (j) * SIZE_X + (k) * SIZE_X * SIZE_Y])
#define Hx(i, j, k)      (Hx_array[(i) + (j) * SIZE_X + (k) * SIZE_X * SIZE_Y])
#define Hy(i, j, k)      (Hy_array[(i) + (j) * SIZE_X + (k) * SIZE_X * SIZE_Y])
#define Hz(i, j, k)      (Hz_array[(i) + (j) * SIZE_X + (k) * SIZE_X * SIZE_Y])
#define mu(i, j, k)      (mu_array[(i) + (j) * SIZE_X + (k) * SIZE_X * SIZE_Y])
#define epsilon(i, j, k) (epsilon_array[(i) + (j) * SIZE_X + (k) * SIZE_X * SIZE_Y])

void write_vtk_2D(
  const std::string& filename,
  double* Ez_array,
  double dx,
  double dy,
  double dz,
  size_t SIZE_X,
  size_t SIZE_Y,
  size_t SIZE_Z)
{
  FILE* f = fopen(filename.c_str(), "w");

  const size_t pos_z = SIZE_Z / 2;
  const size_t size  = SIZE_X * SIZE_Y;

  fprintf(f, "# vtk DataFile Version 3.0\n");
  fprintf(f, "vtk output\n");
  fprintf(f, "ASCII\n");
  fprintf(f, "DATASET UNSTRUCTURED_GRID\n");
  fprintf(f, "POINTS %ld float\n", 4 * size);

  for (size_t y = 0; y < SIZE_Y; y++)
  {
    for (size_t x = 0; x < SIZE_X; x++)
    {
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(x + 0), dy * static_cast<float>(y + 0));
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(x + 1), dy * static_cast<float>(y + 0));
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(x + 1), dy * static_cast<float>(y + 1));
      fprintf(f, "%lf %lf 0.0\n", dx * static_cast<float>(x + 0), dy * static_cast<float>(y + 1));
    }
  }

  fprintf(f, "CELLS %ld %ld\n", size, 5 * size);

  size_t cell_id = 0;
  for (size_t y = 0; y < SIZE_Y; y++)
  {
    for (size_t x = 0; x < SIZE_X; x++)
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

  for (size_t y = 0; y < SIZE_Y; y++)
  {
    for (size_t x = 0; x < SIZE_X; x++)
    {
      fprintf(f, "%lf\n", Ez(x, y, pos_z));
    }
  }

  fclose(f);
}

// Define the source function
double Source(double t, double x, double y, double z)
{
  const double pi         = 3.14159265358979323846;
  const double freq       = 1e9;
  const double omega      = (2 * pi * freq);
  const double wavelength = 3e8 / freq;
  const double k          = 2 * pi / wavelength;
  return sin(k * x - omega * t);
}

int main(int argc, char** argv)
{
  // Domain dimensions
  const size_t SIZE_X = 100;
  const size_t SIZE_Y = 100;
  const size_t SIZE_Z = 100;

  // Grid spacing
  const double DX = 0.01;
  const double DY = 0.01;
  const double DZ = 0.01;

  // Define the electric and magnetic fields
  double* Ex_array = new double[SIZE_X * SIZE_Y * SIZE_Z];
  double* Ey_array = new double[SIZE_X * SIZE_Y * SIZE_Z];
  double* Ez_array = new double[SIZE_X * SIZE_Y * SIZE_Z];
  double* Hx_array = new double[SIZE_X * SIZE_Y * SIZE_Z];
  double* Hy_array = new double[SIZE_X * SIZE_Y * SIZE_Z];
  double* Hz_array = new double[SIZE_X * SIZE_Y * SIZE_Z];

  // Define the permittivity and permeability of the medium
  double* epsilon_array = new double[SIZE_X * SIZE_Y * SIZE_Z];
  double* mu_array      = new double[SIZE_X * SIZE_Y * SIZE_Z];

  const double EPSILON = 8.85e-12; // Permittivity of free space
  const double MU      = 1.256e-6; // Permeability of free space

  // CFL condition DT <= min(DX, DY, DZ) * sqrt(epsilon_max * mu_max)
  double DT = 0.25 * std::min(std::min(DX, DY), DZ) * sqrt(EPSILON * MU);

// Initialize the fields, permittivity, permeability, and conductivity
#pragma acc parallel loop collapse(3)
  for (size_t k = 0; k < SIZE_Z; k++)
  {
    for (size_t j = 0; j < SIZE_Y; j++)
    {
      for (size_t i = 0; i < SIZE_X; i++)
      {
        Ex(i, j, k)      = 0.0;
        Ey(i, j, k)      = 0.0;
        Ez(i, j, k)      = 0.0;
        Hx(i, j, k)      = 0.0;
        Hy(i, j, k)      = 0.0;
        Hz(i, j, k)      = 0.0;
        mu(i, j, k)      = MU;
        epsilon(i, j, k) = EPSILON;
      }
    }
  }

  // Set the source function at the center of the grid
  const size_t center_x = SIZE_X / 2;
  const size_t center_y = SIZE_Y / 2;
  const size_t center_z = SIZE_Z / 2;

  // Initialize the time loop
  size_t timesteps = 1000;
  if (argc > 1)
  {
    timesteps = (size_t) atol(argv[1]);
  }

  // No output by default
  ssize_t output_freq = -1;
  if (argc > 2)
  {
    output_freq = (ssize_t) atol(argv[2]);
  }

  for (size_t n = 0; n < timesteps; n++)
  {
// Update the electric fields

// Update Ex
#pragma acc parallel loop collapse(3) async(1)
    for (size_t k = 1; k < SIZE_Z - 1; k++)
    {
      for (size_t j = 1; j < SIZE_Y - 1; j++)
      {
        for (size_t i = 1; i < SIZE_X - 1; i++)
        {
          Ex(i, j, k) =
            Ex(i, j, k)
            + (DT / (epsilon(i, j, k) * DX)) * (Hz(i, j, k) - Hz(i, j - 1, k) - Hy(i, j, k) + Hy(i, j, k - 1));
        }
      }
    }

#pragma acc parallel loop collapse(3) async(2)
    for (size_t k = 1; k < SIZE_Z - 1; k++)
    {
      for (size_t j = 1; j < SIZE_Y - 1; j++)
      {
        for (size_t i = 1; i < SIZE_X - 1; i++)
        {
          Ey(i, j, k) =
            Ey(i, j, k)
            + (DT / (epsilon(i, j, k) * DY)) * (Hx(i, j, k) - Hx(i, j, k - 1) - Hz(i, j, k) + Hz(i - 1, j, k));
        }
      }
    }

#pragma acc parallel loop collapse(3) async(3)
    for (size_t k = 1; k < SIZE_Z - 1; k++)
    {
      for (size_t j = 1; j < SIZE_Y - 1; j++)
      {
        for (size_t i = 1; i < SIZE_X - 1; i++)
        {
          Ez(i, j, k) =
            Ez(i, j, k)
            + (DT / (epsilon(i, j, k) * DZ)) * (Hy(i, j, k) - Hy(i - 1, j, k) - Hx(i, j, k) + Hx(i, j - 1, k));

          // Add the source function at the center of the grid
          if (i == center_x && j == center_y && k == center_z)
          {
            Ez(i, j, k) = Ez(i, j, k) + Source(n * DT, i * DX, j * DY, k * DZ);
          }
        }
      }
    }

#pragma acc wait(1)
#pragma acc wait(2)
#pragma acc wait(3)

#pragma acc parallel loop collapse(3) async(1)
    for (size_t k = 0; k < SIZE_Z - 1; k++)
    {
      for (size_t j = 0; j < SIZE_Y - 1; j++)
      {
        for (size_t i = 0; i < SIZE_X - 1; i++)
        {
          Hx(i, j, k) =
            Hx(i, j, k) - (DT / (mu(i, j, k) * DY)) * (Ez(i, j + 1, k) - Ez(i, j, k) - Ey(i, j, k + 1) + Ey(i, j, k));
        }
      }
    }

#pragma acc parallel loop collapse(3) async(2)
    for (size_t k = 0; k < SIZE_Z - 1; k++)
    {
      for (size_t j = 0; j < SIZE_Y - 1; j++)
      {
        for (size_t i = 0; i < SIZE_X - 1; i++)
        {
          Hy(i, j, k) =
            Hy(i, j, k) - (DT / (mu(i, j, k) * DZ)) * (Ex(i, j, k + 1) - Ex(i, j, k) - Ez(i + 1, j, k) + Ez(i, j, k));
        }
      }
    }

#pragma acc parallel loop collapse(3) async(3)
    for (size_t k = 0; k < SIZE_Z - 1; k++)
    {
      for (size_t j = 0; j < SIZE_Y - 1; j++)
      {
        for (size_t i = 0; i < SIZE_X - 1; i++)
        {
          Hz(i, j, k) =
            Hz(i, j, k) - (DT / (mu(i, j, k) * DX)) * (Ey(i + 1, j, k) - Ey(i, j, k) - Ex(i, j + 1, k) + Ex(i, j, k));
        }
      }
    }

#pragma acc wait(1)
#pragma acc wait(2)
#pragma acc wait(3)

    if (output_freq > 0)
    {
      // Output the electric field at the center of the grid
      fprintf(stderr, "%ld\t%le\n", n, Ez(center_x, center_y, center_z));

      if (output_freq > 0 && n % output_freq == 0)
      {
        std::string filename = "Ez" + std::to_string(n) + ".vtk";

        // Dump a 2D slice of Ez in VTK
        write_vtk_2D(filename, Ez_array, DX, DY, DZ, SIZE_X, SIZE_Y, SIZE_Z);
      }
    }
  }

  return 0;
}
