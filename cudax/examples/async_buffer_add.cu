//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include <iostream>

namespace cudax = cuda::experimental;

constexpr int numElements = 50000;

struct generator
{
  thrust::default_random_engine gen{};
  thrust::uniform_real_distribution<float> dist{-10.0f, 10.0f};

  __host__ __device__ generator(const unsigned seed)
      : gen{seed}
  {}

  __host__ __device__ float operator()() noexcept
  {
    return dist(gen);
  }
};

int main()
{
  // A CUDA stream on which to execute the vector addition kernel
  cudax::stream stream{};

  // The execution policy we want to use to run all work on the same stream
  auto policy = thrust::cuda::par_nosync.on(stream.get());

  // An environment we use to pass all necessary information to the containers
  cudax::env_t<cuda::mr::device_accessible> env{cudax::device_memory_resource{}, stream};

  // Allocate the two inputs and output, but do not zero initialize via `cudax::uninit`
  cudax::async_device_buffer<float> A{env, numElements, cudax::uninit};
  cudax::async_device_buffer<float> B{env, numElements, cudax::uninit};
  cudax::async_device_buffer<float> C{env, numElements, cudax::uninit};

  // Fill both vectors on stream using a random number generator
  thrust::generate(policy, A.begin(), A.end(), generator{42});
  thrust::generate(policy, B.begin(), B.end(), generator{1337});

  // Add the vectors together
  thrust::transform(policy, A.begin(), A.end(), B.begin(), C.begin(), cuda::std::plus<>{});

  // Verify that the result vector is correct, by copying it to host
  cudax::env_t<cuda::mr::host_accessible> host_env{cudax::pinned_memory_resource{}, stream};
  cudax::async_host_buffer<float> h_A{host_env, A};
  cudax::async_host_buffer<float> h_B{host_env, B};
  cudax::async_host_buffer<float> h_C{host_env, C};

  // Do not forget to sync afterwards
  stream.wait();

  for (int i = 0; i < numElements; ++i)
  {
    if (cuda::std::abs(h_A.get_unsynchronized(i) + h_B.get_unsynchronized(i) - h_C.get_unsynchronized(i)) > 1e-5)
    {
      std::cerr << "Result verification failed at element " << i << "\n";
      exit(EXIT_FAILURE);
    }
  }

  return 0;
}
