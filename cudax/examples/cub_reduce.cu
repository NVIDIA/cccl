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
  cuda::stream stream{cuda::devices[0]};
  cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(cuda::devices[0]);

  // Allocate input and output, but do not zero initialize output (`cudax::no_init`)
  auto d_in  = cudax::make_async_buffer<int>(stream, mr, num_items, 1);
  auto d_out = cudax::make_async_buffer<float>(stream, mr, 1, cudax::no_init);

  // An environment we use to pass all necessary information to CUB
  cudax::env_t<cuda::mr::device_accessible> env{mr, stream};
  auto error = cub::DeviceReduce::Reduce(d_in.begin(), d_out.begin(), num_items, cuda::std::plus{}, 0, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Reduce failed: " << cudaGetErrorString(error) << "\n";
    exit(EXIT_FAILURE);
  }

  auto h_out = cudax::make_async_buffer<float>(stream, cuda::pinned_default_memory_pool(), d_out);

  stream.sync();

  if (h_out.get_unsynchronized(0) != num_items)
  {
    std::cerr << "Result verification failed: " << h_out.get_unsynchronized(0) << " != " << num_items << "\n";
    exit(EXIT_FAILURE);
  }
}
