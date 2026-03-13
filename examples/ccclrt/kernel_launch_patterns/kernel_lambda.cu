//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This example demonstrates how kernel lambdas can be launched using cuda::launch. Kernel lambdas behave mostly the
// same way as other kernel functors, but there are some differences.

#if !defined(__CUDACC_EXTENDED_LAMBDA__)
#  error "This example requires extended lambda support."
#endif // !__CUDACC_EXTENDED_LAMBDA__

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/stream>

#include <cstdio>
#include <exception>

#include "common.cuh"

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

  // cuda::launch always requires a work submitter, so let's create a CUDA stream.
  cuda::stream stream{device};

  // Set block and grid dimensions to be used with the kernel config. Dimensions specified as template parameters will
  // be statically known in the kernel.
  const auto block_dims = cuda::block_dims<2, 2>();
  const auto grid_dims  = cuda::grid_dims(dim3{1});

  // Make the kernel config.
  const auto kernel_config = cuda::make_config(grid_dims, block_dims);

  // For kernels that use dynamic shared memory, we need a dynamic shared memory option to be passed in the kernel
  // config.
  const auto dyn_smem_opt = cuda::dynamic_shared_memory<uint3[]>(cuda::gpu_thread.count(cuda::block, kernel_config));

  // Make the kernel config with dynamic shared memory option.
  const auto kernel_config_with_dyn_smem = cuda::make_config(grid_dims, block_dims, dyn_smem_opt);

  // Launch the kernel lambda using the kernel config. Unlike with kernel functors, the config is not automatically
  // passed as the first parameter and must be passed explicitly.
  cuda::launch(
    stream,
    kernel_config,
    [] __device__(auto config, auto kernel_name) {
      say_hello(cuda::gpu_thread.index(cuda::block, config), kernel_name);
    },
    kernel_config,
    KernelName{"kernel lambda"});

  // The kernel lambda with captures be launched the same way. All parameters must be captured by value and each thread
  // will get a copy of the lambda.
  cuda::launch(
    stream,
    kernel_config,
    [kernel_name = KernelName{"kernel lambda with capture"}] __device__(auto config) {
      say_hello(cuda::gpu_thread.index(cuda::block, config), kernel_name);
    },
    kernel_config);

  // The kernel lambda can use dynamic shared memory in the same way as kernel functors.
  cuda::launch(
    stream,
    kernel_config_with_dyn_smem,
    [] __device__(auto config, auto kernel_name) {
      // Retrieve the dynamic shared memory view. Since we passed uint3[4], we will get cuda::std::span<uint3>.
      const auto smem = cuda::dynamic_shared_memory(config);

      // Get all necessary hierarchy values.
      const auto tindex = cuda::gpu_thread.index(cuda::block, config);
      const auto trank  = cuda::gpu_thread.rank(cuda::block, config);
      const auto tcount = cuda::gpu_thread.count(cuda::block, config);

      // Each thread will write it's index to the next thread's index in the shared memory.
      smem[(trank + 1) % tcount] = tindex;

      // Wait for the all threads to finish the write to shared memory.
      __syncthreads();

      // Call say hello with received previous thread's index.
      say_hello(smem[trank], kernel_name);
    },
    kernel_config_with_dyn_smem,
    KernelName{"kernel lambda with dynamic shared memory"});

  // Wait for all of the tasks in the stream to complete.
  stream.sync();
}
catch (const cuda::cuda_error& e)
{
  std::fprintf(stderr, "CUDA error: %s\n", e.what());
  return 1;
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "Error: %s\n", e.what());
  return 1;
}
catch (...)
{
  std::fprintf(stderr, "An unknown error was encountered\n");
  return 1;
}
