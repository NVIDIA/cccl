//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstring>
#include <cuda/stream>

#include <cuda/experimental/hierarchy.cuh>
#include <cuda/experimental/kernel.cuh>
#include <cuda/experimental/launch.cuh>

#include <cstdio>
#include <stdexcept>

#include <cuda.h>

// Create an alias for the experimental namespace to shorten the code.
namespace cudax = cuda::experimental;

// A helper type for storing kernel launch patter name.
struct name_buffer
{
  // Size of the buffer.
  static constexpr cuda::std::size_t size = 128;

  // Buffer data.
  char data[size];

  // Constructor from string literal.
  template <cuda::std::size_t N>
  name_buffer(const char (&str)[N])
  {
    static_assert(N <= size, "string literal is too long");
    cuda::std::memcpy(data, str, N);
  }
};

// A helper function for printing the Hello world! message.
__device__ void say_hello(dim3 tid, const name_buffer& name)
{
  printf("Hello world from thread [%u, %u] launched as %s!\n", tid.x, tid.y, name.data);

  // Wait for all threads in block to print the output.
  __syncthreads();

  // Print additional new line once.
  if (tid.x == 0 && tid.y == 0)
  {
    printf("\n");
  }
}

// This is the traditional way to define a kernel, a void function decorated with __global__ attribute.
__global__ void kernel(name_buffer name)
{
  say_hello(threadIdx, name);
}

// This is a kernel functor, a callable object with operator() decorated with __device__ attribute. When launched, the
// object is copied to the device and operator() is invoked on the device.
struct kernel_functor
{
  // The functor object can be set on host before the launch. Keep in mind that the functor (thus all the members) must
  // be trivially copyable.
  int member;

  // The operator() must be decorated with __device__ attribute. It can also be a template.
  __device__ void operator()(name_buffer name)
  {
    say_hello(threadIdx, name);

    // Check that the member was copied correctly to the device.
    assert(member == 42);
  }
};

// This is again a kernel functor, but this time the operator() takes the implicit kernel configuration parameter. This
// parameter is a cudax::kernel_config object that contains the launch configuration.
struct kernel_functor_with_config
{
  // A type that represents the layout of the dynamic shared memory used by this kernel functor.
  struct dynamic_smem_layout
  {
    int value;
  };

  // The operator() must be again decorated with __device__ attribute. It can also be a template or take additional
  // parameters after the kernel configuration parameter. Return type must be void.
  template <class Dims, class... Opts>
  __device__ void operator()(cudax::kernel_config<Dims, Opts...> config, name_buffer name)
  {
    say_hello(config.dims.index(cudax::thread, cudax::block), name);

    // Call other demo methods.
    demo_extents(config);
    demo_static_extents(config);
    demo_dynamic_shared_memory(config);
  }

  template <class Config>
  __device__ void demo_index(const Config& config)
  {
    // dims.index(entity, in_level) queries the index of an entity in a hierarchy level. Query of the thread entity
    // index in the block hierarchy level results in the same value as blockIdx.
    const auto thread_idx_in_block = config.dims.index(cudax::thread, cudax::block);
    assert(thread_idx_in_block.x == threadIdx.x);
    assert(thread_idx_in_block.y == threadIdx.y);

    // One of the advantages compared to the traditional approach is that the index can be very easily queried for
    // any hierarchy level.
    const auto thread_idx_in_grid = config.dims.index(cudax::thread, cudax::grid);
    assert(thread_idx_in_grid.x == blockIdx.x * blockDim.x + threadIdx.x);
    assert(thread_idx_in_grid.y == blockIdx.y * blockDim.y + threadIdx.y);
  }

  template <class Config>
  __device__ void demo_extents(const Config& config)
  {
    // Similarly dims.extents(entity, level) queries the extents of an entity in a hierarchy level. Query of the thread
    // entity extent in the block hierarchy results in the same value as blockDim.
    const auto nthreads_in_block = config.dims.extents(cudax::thread, cudax::block);
    assert(nthreads_in_block.x == blockDim.x);
    assert(nthreads_in_block.y == blockDim.y);

    // Again, the extents can be queried for any hierarchy levels.
    const auto nthreads_in_grid = config.dims.extents(cudax::thread, cudax::grid);
    assert(nthreads_in_grid.x == blockDim.x * gridDim.x);
    assert(nthreads_in_grid.y == blockDim.y * gridDim.y);
  }

  template <class Config>
  __device__ void demo_static_extents(const Config& config)
  {
    // The main advantage of using kernel config instead of the ordinary kernel parameters is that the config can carry
    // statically defined extents. That means it is easier to generate kernels specialized for certain block sizes.
    static_assert(
      decltype(config.dims.extents(cudax::thread, cudax::block))::static_extent(0) == cuda::std::dynamic_extent);
    static_assert(
      decltype(config.dims.extents(cudax::thread, cudax::block))::static_extent(1) == cuda::std::dynamic_extent);

    // The static extents can be alternatively queried by the static_extents(...) static method.
    static_assert(decltype(config.dims)::static_extents(cudax::block, cudax::grid)[0] == 1);
    static_assert(decltype(config.dims)::static_extents(cudax::block, cudax::grid)[1] == 1);
  }

  template <class Config>
  __device__ void demo_dynamic_shared_memory(const Config& config)
  {
    // If the config contains a cudax::dynamic_shared_memory_option option, the cudax::dynamic_smem_ref function can be
    // used to get a reference to the dynamic shared memory type. Keep in mind that the object is not constructed, so
    // one of the threads must construct it before it is used.
    auto& dyn_smem = cudax::dynamic_smem_ref(config);

    // Construct the dynamic_smem_layout object in the shared memory by the first thread in the block.
    if (config.dims.rank(cudax::thread, cudax::block) == 0)
    {
      new (&dyn_smem) dynamic_smem_layout{42};
    }

    // Wait until the write is finished.
    __syncthreads();

    // All threads should see the same value in the shared memory.
    assert(dyn_smem.value == 42);
  }
};

int main()
try
{
  // Check we have at least one device.
  if (cuda::devices.size() == 0)
  {
    std::fprintf(stderr, "No CUDA devices found\n");
    return 1;
  }

  // We will use the first device.
  cuda::device_ref device = cuda::devices[0];

  // cudax::launch always requires a work submitter, so let's create a CUDA stream.
  cuda::stream stream{device};

  // Create a custom hierarchy to be used in cudax::launch. We will be launching a 1D grid of 1 block. The block will be
  // a 2D grid of 2 threads in x and y axis.
  //
  // Note that the grid dimensions are passed as template parameters in this example. That means the value can be used
  // in constexpr context inside the kernel. Block dimensions will be constructed at runtime as usually.
  const auto hierarchy = cudax::make_hierarchy(cudax::grid_dims<1>(), cudax::block_dims(dim3{2, 2}));

  // Launch an ordinary kernel. cudax::launch takes a stream as the first argument followed by the kernel configuration,
  // kernel and kernel parameters.
  cudax::launch(stream, cudax::make_config(hierarchy), kernel, "kernel");

  // Launch a kernel functor. Here, we use cudax::distribute to create the kernel_config for us. This function creates
  // a simple 1D grid of 1D blocks of a given size.
  cudax::launch(stream, cudax::distribute<4>(4), kernel_functor{42}, name_buffer{"kernel functor"});

  // Launch a kernel functor that takes a cudax::kernel_config. Note that the kernel config is passed automatically as
  // the first argument by the cudax::launch function.
  const auto config =
    cudax::make_config(hierarchy, cudax::dynamic_shared_memory<kernel_functor_with_config::dynamic_smem_layout>());
  cudax::launch(stream, config, kernel_functor_with_config{}, name_buffer{"kernel functor with config"});

#if defined(__CUDACC_EXTENDED_LAMBDA__)
  // Kernel lambda is another form of the kernel functor. It can optionally take the kernel_config as the first
  // argument. Extended lambda are required to use this feature.
  // See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambdas for more info.
  cudax::launch(
    stream,
    cudax::make_config(hierarchy),
    [] __device__(auto config, name_buffer name) {
      say_hello(config.dims.index(cudax::thread, cudax::block), name);
    },
    name_buffer{"kernel lambda"});
#endif // defined(__CUDACC_EXTENDED_LAMBDA__)

#if CUDA_VERSION >= 12010
  // Launch a cudax::kernel_ref object which is a wrapper of cudaKernel_t. The type is available since CUDA 12.0, but
  // the cudaGetKernel function used to get the handle of a CUDA Runtime kernel is available since CUDA 12.1.
  cudax::launch(stream, cudax::make_config(hierarchy), cudax::kernel_ref{kernel}, "kernel reference");
#endif // CUDA_VERSION >= 12010

  // Wait for all of the tasks in the stream to complete.
  stream.sync();
}
catch (const cuda::cuda_error& e)
{
  std::fprintf(stderr, "CUDA error: %s", e.what());
  return 1;
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "Error: %s", e.what());
  return 1;
}
catch (...)
{
  std::fprintf(stderr, "An unknown error was encountered");
  return 1;
}
