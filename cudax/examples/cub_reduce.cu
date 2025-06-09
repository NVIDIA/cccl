//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Example of using `cub::DeviceReduce::Reduce` with cudax environment.

#include <cub/device/device_reduce.cuh>

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include <iostream>

namespace cudax = cuda::experimental;

int main()
{
  constexpr int num_items = 50000;

  // A CUDA stream on which to execute the reduction
  cudax::stream stream{};

  // An environment we use to pass all necessary information to the containers
  cudax::env_t<cuda::mr::device_accessible> env{cudax::device_memory_resource{}, stream};

  // Allocate input and output, but do not zero initialize output (`cudax::no_init`)
  cudax::async_device_buffer<int> d_in{env, num_items, 1};
  cudax::async_device_buffer<float> d_out{env, 1, cudax::no_init};

  cub::DeviceReduce::Reduce(d_in.begin(), d_out.begin(), num_items, cuda::std::plus{}, 0, env);

  cudax::env_t<cuda::mr::host_accessible> host_env{cudax::pinned_memory_resource{}, stream};
  cudax::async_host_buffer<float> h_out{host_env, d_out};

  stream.sync();

  if (h_out.get_unsynchronized(0) != num_items)
  {
    std::cerr << "Result verification failed: " << h_out.get_unsynchronized(0) << " != " << num_items << "\n";
    exit(EXIT_FAILURE);
  }
}
