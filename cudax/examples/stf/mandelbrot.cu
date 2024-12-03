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
 * @brief A transparent multi-GPU implementation of Mandelbrot fractal using parallel_for
 */

#include <cuda/experimental/stf.cuh>

#include <fstream>
#include <iostream>

using namespace cuda::experimental::stf;

int main(int argc, char** argv)
{
  context ctx;

  // Image dimensions
  size_t width  = 2000;
  size_t height = 1000;

  // Complex plane boundaries
  double xMin = -2.0;
  double xMax = 1.0;
  double yMin = -1.5;
  double yMax = 1.5;

  // Maximum number of iterations
  int maxIterations = 256;

  // Describe a 2D array of integers of size (width x heigth)
  auto lbuffer = ctx.logical_data(shape_of<slice<int, 2>>(width, height));

  cudaEvent_t start, stop;
  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));

  cuda_safe_call(cudaEventRecord(start, ctx.task_fence()));

  // Compute each pixel
  ctx.parallel_for(blocked_partition(), exec_place::all_devices(), lbuffer.shape(), lbuffer.write())
      ->*[=] _CCCL_DEVICE(size_t x, size_t y, auto buffer) {
            // Map pixel coordinates to complex plane
            // c = cr + i ci
            double cr = x * (xMax - xMin) / width + xMin;
            double ci = y * (yMax - yMin) / height + yMin;

            // z = zr + i zi
            double zr = 0.0;
            double zi = 0.0;

            int iterations = 0;

            // Evaluate depth
            while (zr * zr + zi * zi < 4 && iterations < maxIterations)
            {
              // compute : z = z * z + c;
              //
              // z = (zr + i zi) (zr + i zi) + cr + i ci
              // z = zr zr - zi zi + 2 i zi zr + cr + i ci
              // zr = (zr zr - zi zi + cr)
              // zi = (2 zi zr + ci)
              double zr_prev = zr;
              double zi_prev = zi;
              zr             = zr_prev * zr_prev - zi_prev * zi_prev + cr;
              zi             = 2.0 * zr_prev * zi_prev + ci;

              iterations++;
            }

            buffer(x, y) = iterations;
          };

  cuda_safe_call(cudaEventRecord(stop, ctx.task_fence()));

  if (argc > 1)
  {
    auto fileName = std::string(argv[1]);

    // Generate a PPM file from the buffer
    ctx.host_launch(lbuffer.read())->*[&](auto buffer) {
      std::ofstream imageFile(fileName, std::ios::binary);
      if (!imageFile)
      {
        std::cerr << "Failed to create image file: " << fileName << std::endl;
        return;
      }

      imageFile << "P6\n";
      imageFile << width << " " << height << "\n";
      imageFile << "255\n";

      for (size_t y = 0; y < height; y++)
      {
        for (size_t x = 0; x < width; x++)
        {
          int iterations = buffer(x, y);
          // Convert iterations to RGB values
          unsigned char r = (iterations % 8) * 32;
          unsigned char g = (iterations % 16) * 16;
          unsigned char b = (iterations % 32) * 8;

          // Write pixel data to file
          imageFile << r << g << b;
        }
      }

      imageFile.close();
      std::cout << "Mandelbrot image generated and saved as " << fileName << std::endl;
    };
  }

  ctx.finalize();

  // Must call this first, see e.g.
  // https://stackoverflow.com/questions/6551121/cuda-cudaeventelapsedtime-returns-device-not-ready-error
  cuda_safe_call(cudaEventSynchronize(stop));

  fprintf(stderr, "Mandelbrot took %.2f ms\n", cuda_try<cudaEventElapsedTime>(start, stop));
}
