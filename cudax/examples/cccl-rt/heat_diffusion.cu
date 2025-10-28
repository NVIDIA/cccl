#include <cuda/cmath>
#include <cuda/std/mdspan>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/algorithm.cuh>
#include <cuda/experimental/container.cuh>
#include <cuda/experimental/hierarchy.cuh>
#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <cstdio>
#include <exception>
#include <fstream>

/*
 * Based on CSC materials from:
 *
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 *
 */
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

namespace cudax = cuda::experimental;

//! @brief Initializes the data with a pattern of a disk of radius of 1/6 of the width.
void init_grid(cuda::std::mdspan<float, cuda::std::dims<2, int>> grid)
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
    return cudax::make_config(cudax::make_hierarchy(cudax::block_dims<threads_per_side, threads_per_side>(),
                                                    cudax::grid_dims(dim3{blocks_per_grid_x, blocks_per_grid_y})));
  }

  //! @brief Computes 1 step of heat diffusion simulation.
  //!
  //! @param config The kernel configuration.
  //! @param src The old data.
  //! @param dst The new data.
  template <class Config>
  __device__ void operator()(const Config& config, const float* src, float* dst) const noexcept
  {
    // Shared memory to exchange the loaded data.
    __shared__ float smem[(threads_per_side + 2) * (threads_per_side + 2)];

    // Create mdspan views for source, destination and shared memory data.
    cuda::std::mdspan src_view{src, cuda::std::dims<2, int>{ny_, nx_}};
    cuda::std::mdspan dst_view{dst, cuda::std::dims<2, int>{ny_, nx_}};
    cuda::std::mdspan smem_view{smem, cuda::std::extents<int, threads_per_side + 2, threads_per_side + 2>{}};

    // The index of this thread in the grid.
    const auto y_in_grid = config.dims.index(cudax::grid).y;
    const auto x_in_grid = config.dims.index(cudax::grid).x;

    // The index of this thread in the block.
    const auto y_in_block = config.dims.index(cudax::block).y;
    const auto x_in_block = config.dims.index(cudax::block).x;

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
void write_output(cuda::std::mdspan<float, cuda::std::dims<2, int>> grid, int step)
{
  constexpr int min_value = 0;
  constexpr int max_value = 80;

  ::std::ofstream file{"heat_map_step_" + std::to_string(step) + ".pgm"};
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

int main()
try
{
  // Width.
  constexpr int nx = 256;
  // Height.
  constexpr int ny = 256;
  // Number of grid points.
  constexpr int ngrid_points = nx * ny;
  // Number of time steps.
  constexpr int nt = 5000;

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
  auto host_buffer =
    cudax::make_async_buffer<float>(main_stream, cudax::pinned_default_memory_pool(), ngrid_points, cudax::no_init);
  cuda::std::mdspan<float, cuda::std::dims<2, int>> host_buffer_view{host_buffer.data(), ny, nx};

  // Wait for the allocation to complete before initializing the buffer.
  main_stream.sync();

  // Initialize the host buffer with values.
  init_grid(host_buffer_view);

  // Allocate device buffers.
  auto device_buffer_1 = cudax::make_async_buffer<float>(
    main_stream, cudax::device_default_memory_pool(device), ngrid_points, cudax::no_init);
  auto device_buffer_2 = cudax::make_async_buffer<float>(
    main_stream, cudax::device_default_memory_pool(device), ngrid_points, cudax::no_init);

  // Copy the initial heat map to device buffer 1.
  cudax::copy_bytes(main_stream, host_buffer, device_buffer_1);

  // Copy the initial heat map also to the device buffer 2.
  cudax::copy_bytes(main_stream, device_buffer_1, device_buffer_2);

  // Create additional stream for device -> host copies and output writing.
  cuda::stream output_stream{device};

  // Create events for main and output streams synchronizations.
  cuda::event copy_output_event{device};
  cuda::event copy_done_event{device};
  cuda::event output_done_event{device};

  // Create an instance of the kernel functor for computing the heat diffusion step.
  HeatDiffusionKernelFunctor kernel_functor{nx, ny, a, dx, dy, dt};

  // Callable to determine if the results should be outputted.
  auto should_output = [](int ti) {
    return ti % output_freq == 0;
  };

  // Record the start of the computation.
  const auto start = main_stream.record_timed_event();

  // The main compute loop.
  for (int ti = 0; ti <= nt; ti++)
  {
    if (should_output(ti))
    {
      // Record an event to main stream signaling that the data is ready and the copy operation can start.
      copy_output_event.record(main_stream);
    }

    // Enqueue the computation to the main stream.
    cudax::launch(main_stream, {}, kernel_functor, device_buffer_1.data(), device_buffer_2.data());

    if (should_output(ti))
    {
      // Make the output stream wait for the data to be ready.
      output_stream.wait(copy_output_event);

      // Enqueue the copy from device to host buffer.
      cudax::copy_bytes(output_stream, device_buffer_1, host_buffer);

      // Record an event signaling that the copy operation has finished.
      copy_done_event.record(output_stream);

      // Make the main stream wait for the copy operation to complete, so the data isn't overwritten too early.
      main_stream.wait(copy_done_event);

      // Enqueue the output callback to output stream.
      cudax::host_launch(output_stream, write_output, host_buffer_view, ti);
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
