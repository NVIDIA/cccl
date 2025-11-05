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

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/tabulate.h>
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

  __host__ __device__ float operator()(cuda::std::size_t idx) noexcept
  {
    gen.discard(idx);
    return dist(gen);
  }
};

int main()
{
  // A CUDA stream on which to execute the vector addition kernel
  cudax::stream stream{cuda::device_ref{0}};

  // The execution policy we want to use to run all work on the same stream
  auto policy = thrust::cuda::par_nosync.on(stream.get());

  cuda::device_memory_pool_ref device_resource = cuda::device_default_memory_pool(cuda::device_ref{0});

  // Allocate the two inputs and output, but do not zero initialize via `cudax::no_init`
  cudax::async_device_buffer<float> A{stream, device_resource, numElements, cudax::no_init};
  cudax::async_device_buffer<float> B{stream, device_resource, numElements, cudax::no_init};
  cudax::async_device_buffer<float> C{stream, device_resource, numElements, cudax::no_init};

  // Fill both vectors on stream using a random number generator
  thrust::tabulate(policy, A.begin(), A.end(), generator{42});
  thrust::tabulate(policy, B.begin(), B.end(), generator{1337});

  // Add the vectors together
  thrust::transform(policy, A.begin(), A.end(), B.begin(), C.begin(), cuda::std::plus<>{});

  cuda::pinned_memory_pool_ref pinned_resource = cuda::pinned_default_memory_pool();

  // Verify that the result vector is correct, by copying it to host
  cudax::async_host_buffer<float> h_A{stream, pinned_resource, A};
  cudax::async_host_buffer<float> h_B{stream, pinned_resource, B};
  cudax::async_host_buffer<float> h_C{stream, pinned_resource, C};

  // Do not forget to sync afterwards
  stream.sync();

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
