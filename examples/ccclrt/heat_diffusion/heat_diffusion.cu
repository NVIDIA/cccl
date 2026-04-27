//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// MIT License
//
// Copyright (c) 2016 CSC Training
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//===----------------------------------------------------------------------===//

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/mdspan>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>

//
// Based on CSC materials from:
//
// https://github.com/csc-training/openacc/tree/master/exercises/heat
//

//! @brief Initializes the data with a pattern of a disk of radius of 1/6 of the width.
void init_grid(cuda::host_mdspan<float, cuda::std::dims<2, int>> grid)
{
  const float radius2 = (grid.extent(0) / 6.f) * (grid.extent(1) / 6.f);

  for (int i = 0; i < grid.extent(0); ++i)
  {
    for (int j = 0; j < grid.extent(1); ++j)
    {
      const float distance2 =
        (i - grid.extent(0) / 2) * (i - grid.extent(0) / 2) + (j - grid.extent(1) / 2) * (j - grid.extent(1) / 2);
      grid(i, j) = (distance2 < radius2) ? 65.f : 5.f;
    }
  }
}

//! @brief Kernel functor for heat diffusion computation.
class HeatDiffusionKernelFunctor
{
  int nx_; //!< Number of grid points in X axis.
  int ny_; //!< Number of grid points in Y axis.
  float dx2_; //!< Horizontal grid spacing squared.
  float dy2_; //!< Vertical grid spacing squared.
  float a_times_dt_; //!< Diffusion coefficient multiplied by time difference.

public:
  //! @brief The number of threads per dimension in a thread block.
  static constexpr int threads_per_side = 16;

  //! @brief Constructor.
  //!
  //! @param nx Number of grid points in X axis.
  //! @param ny Number of grid points in Y axis.
  //! @param a The diffusion coefficient.
  //! @param dx Horizontal grid spacing.
  //! @param dy Vertical grid spacing.
  //! @param dt Time difference.
  constexpr HeatDiffusionKernelFunctor(int nx, int ny, float a, float dx, float dy, float dt) noexcept
      : nx_{nx}
      , ny_{ny}
      , dx2_{dx * dx}
      , dy2_(dy * dy)
      , a_times_dt_{a * dt}
  {}

  //! @brief Gets the default configuration with filled block dimensions. Should be combined with grid dimensions
  //!        config.
  [[nodiscard]] constexpr auto default_config() const noexcept
  {
    const auto blocks_per_grid_x = cuda::ceil_div(nx_, threads_per_side);
    const auto blocks_per_grid_y = cuda::ceil_div(ny_, threads_per_side);
    return cuda::make_config(
      cuda::block_dims<threads_per_side, threads_per_side>(),
      cuda::grid_dims(dim3{static_cast<unsigned>(blocks_per_grid_x), static_cast<unsigned>(blocks_per_grid_y)}));
  }

  //! @brief Computes 1 step of heat diffusion simulation.
  //!
  //! @param config The kernel configuration.
  //! @param src The old data.
  //! @param dst The new data.
  template <class Config>
  __device__ void
  operator()(const Config& config, cuda::std::span<const float> src, cuda::std::span<float> dst) const noexcept
  {
    // Shared memory to exchange the loaded data.
    __shared__ float smem[(threads_per_side + 2) * (threads_per_side + 2)];

    // Create mdspan views for source, destination and shared memory data.
    cuda::device_mdspan src_view{src.data(), cuda::std::dims<2, int>{ny_, nx_}};
    cuda::device_mdspan dst_view{dst.data(), cuda::std::dims<2, int>{ny_, nx_}};
    cuda::device_mdspan smem_view{smem, cuda::std::extents<int, threads_per_side + 2, threads_per_side + 2>{}};

    // The index of this thread in the grid.
    const auto y_in_grid = cuda::gpu_thread.index(cuda::grid, config).y;
    const auto x_in_grid = cuda::gpu_thread.index(cuda::grid, config).x;

    // The index of this thread in the block.
    const auto y_in_block = cuda::gpu_thread.index(cuda::block, config).y;
    const auto x_in_block = cuda::gpu_thread.index(cuda::block, config).x;

    // The index of the shared member owned by this thread.
    const auto y_in_smem = y_in_block + 1;
    const auto x_in_smem = x_in_block + 1;

    // Is this thread part of the grid border.
    const bool is_y_grid_border = (y_in_grid == 0) || (y_in_grid == ny_ - 1);
    const bool is_x_grid_border = (x_in_grid == 0) || (x_in_grid == nx_ - 1);
    const bool is_grid_border   = is_y_grid_border || is_x_grid_border;

    // Load the central square to shared memory.
    smem_view(y_in_smem, x_in_smem) = src_view(y_in_grid, x_in_grid);

    // Load the bottom border to shared memory.
    if (y_in_block == 0 && !is_grid_border)
    {
      smem_view(0, x_in_smem) = src_view(y_in_grid - 1, x_in_grid);
    }

    // Load the top border to shared memory.
    if (y_in_block == threads_per_side - 1 && !is_grid_border)
    {
      smem_view(y_in_smem + 1, x_in_smem) = src_view(y_in_grid + 1, x_in_grid);
    }

    // Load the left border to shared memory.
    if (x_in_block == 0 && !is_grid_border)
    {
      smem_view(y_in_smem, 0) = src_view(y_in_grid, x_in_grid - 1);
    }

    // Load the right border to shared memory.
    if (y_in_block == threads_per_side - 1 && !is_grid_border)
    {
      smem_view(y_in_smem, x_in_smem + 1) = src_view(y_in_grid, x_in_grid + 1);
    }

    // Make sure all the data is loaded before computing.
    __syncthreads();

    // If the thread is not on the grid border, update the destination value.
    if (!is_grid_border)
    {
      const auto current = smem_view(y_in_smem, x_in_smem);
      const auto bottom  = smem_view(y_in_smem - 1, x_in_smem);
      const auto top     = smem_view(y_in_smem + 1, x_in_smem);
      const auto left    = smem_view(y_in_smem, x_in_smem - 1);
      const auto right   = smem_view(y_in_smem, x_in_smem + 1);

      dst_view(y_in_grid, x_in_grid) =
        current + a_times_dt_ * ((bottom - 2.f * current + top) / dy2_ + (left - 2.f * current + right) / dx2_);
    }
  }
};

//! @brief Outputs the data in a given time step. The result is stored in a PGM format.
void write_output(cuda::host_mdspan<float, cuda::std::dims<2, int>> grid, int step)
{
  constexpr int min_value = 0;
  constexpr int max_value = 80;

  std::ofstream file{"heat_map_step_" + std::to_string(step) + ".pgm"};
  file << "P2\n";
  file << std::to_string(grid.extent(1)) << " " << std::to_string(grid.extent(0)) << "\n";
  file << max_value << "\n";

  for (int i = 0; i < grid.extent(0); ++i)
  {
    for (int j = 0; j < grid.extent(1); ++j)
    {
      if (j > 0)
      {
        file << " ";
      }
      file << cuda::std::clamp(static_cast<int>(grid(i, j)), min_value, max_value);
    }
    file << "\n";
  }
}

int main(int argc, const char* argv[])
try
{
  // Width.
  constexpr int nx = 256;
  // Height.
  constexpr int ny = 256;
  // Number of grid points.
  constexpr int ngrid_points = nx * ny;
  // Default number of time steps.
  constexpr int default_nt = 5000;

  // Diffusion constant.
  constexpr float a = 0.5f;
  // Horizontal grid spacing.
  constexpr float dx = 0.1f;
  // Vertical grid spacing.
  constexpr float dy = 0.1f;
  // Horizontal grid spacing squared.
  constexpr float dx2 = dx * dx;
  // Vertical grid spacing squared.
  constexpr float dy2 = dy * dy;
  // Largest stable time step.
  constexpr float dt = dx2 * dy2 / (2.f * a * (dx2 + dy2));

  // Write the output each n-th time step.
  constexpr int output_freq = 1000;

  // Number of type steps.
  int nt = default_nt;
  if (argc >= 2)
  {
    nt = std::stoi(argv[1]);
  }

  // Check that there is a CUDA device.
  if (cuda::devices.size() == 0)
  {
    std::fprintf(stderr, "No CUDA devices were found.\n");
    return 1;
  }

  // We will work with the first device.
  cuda::device_ref device = cuda::devices[0];

  // Create the a stream for memory allocations and computations.
  cuda::stream main_stream{device};

  // Allocate a pinned host buffer.
  const auto pinned_mempool = cuda::pinned_default_memory_pool();
  auto host_buffer          = cuda::make_buffer<float>(main_stream, pinned_mempool, ngrid_points, cuda::no_init);
  cuda::host_mdspan<float, cuda::std::dims<2, int>> host_buffer_view{host_buffer.data(), ny, nx};

  // Wait for the allocation to complete before initializing the buffer.
  main_stream.sync();

  // Initialize the host buffer with values.
  init_grid(host_buffer_view);

  // Allocate first device buffer.
  const auto device_mempool = cuda::device_default_memory_pool(device);
  auto device_buffer_1      = cuda::make_buffer<float>(main_stream, device_mempool, ngrid_points, cuda::no_init);

  // Copy the initial heat map to device buffer 1.
  cuda::copy_bytes(main_stream, host_buffer, device_buffer_1);

  // Create second device buffer and initialize it with the data from the first device buffer.
  auto device_buffer_2 = cuda::make_buffer<float>(main_stream, device_mempool, device_buffer_1);

  // Create additional stream for device -> host copies and output writing.
  cuda::stream output_stream{device};

  // Create an instance of the kernel functor for computing the heat diffusion step.
  HeatDiffusionKernelFunctor kernel_functor{nx, ny, a, dx, dy, dt};

  // Callable to determine if the results should be outputted.
  const auto should_output = [](int ti) {
    return ti % output_freq == 0;
  };

  // Record the start of the computation.
  const auto start = main_stream.record_timed_event();

  // The main compute loop.
  for (int ti = 0; ti <= nt; ti++)
  {
    if (should_output(ti))
    {
      // Make output_stream wait for the main_stream to make the data ready so the copy to host can start.
      output_stream.wait(main_stream);

      // Enqueue the copy from device to host buffer.
      cuda::copy_bytes(output_stream, device_buffer_1, host_buffer);
    }

    // Enqueue the computation to the main stream. We use empty config, because the functor default config sets all of
    // the properties we need.
    cuda::launch(main_stream, cuda::make_config(), kernel_functor, device_buffer_1, device_buffer_2);

    if (should_output(ti))
    {
      // Make the main stream wait for the copy operation to complete, so the data isn't overwritten too early.
      main_stream.wait(output_stream);

      // Enqueue the output callback to output stream.
      cuda::host_launch(output_stream, write_output, host_buffer_view, ti);
    }

    device_buffer_1.swap(device_buffer_2);
  }

  // Record the end of the computation.
  const auto end = main_stream.record_timed_event();

  // Wait for all of the asynchronous work to complete.
  main_stream.sync();
  output_stream.sync();

  // Get the elapsed time.
  const auto elapsed_time = end - start;

  // Print the results.
  std::printf("Elapsed time [ms]: %f\n", static_cast<float>(elapsed_time.count()) / 1'000'000);
}
catch (const cuda::cuda_error& e)
{
  std::fprintf(stderr, "CUDA error: %s\n", e.what());
  return 1;
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "An unknown error: %s\n", e.what());
  return 1;
}
catch (...)
{
  std::fprintf(stderr, "An unknown error occurred\n");
  return 1;
}
