//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/device/device_transform.cuh>

#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cuda/memory_resource>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/algorithm.cuh>
#include <cuda/experimental/container.cuh>
#include <cuda/experimental/launch.cuh>

#include <algorithm>

#include "helper.h"
#include "types.h"

C2H_TEST("DeviceTransform::Transform cudax::async_device_buffer", "[device][device_transform]")
{
  using type          = int;
  const int num_items = 1 << 24;

  cudax::stream stream{cuda::device_ref{0}};
  cuda::device_memory_pool_ref resource = cuda::device_default_memory_pool(cuda::device_ref{0});

  cudax::async_device_buffer<type> a{stream, resource, num_items, cudax::no_init};
  cudax::async_device_buffer<type> b{stream, resource, num_items, cudax::no_init};
  thrust::sequence(thrust::cuda::par_nosync.on(stream.get()), a.begin(), a.end());
  thrust::sequence(thrust::cuda::par_nosync.on(stream.get()), b.begin(), b.end());

  cudax::async_device_buffer<type> result{stream, resource, num_items, cudax::no_init};

  cub::DeviceTransform::Transform(
    ::cuda::std::make_tuple(a.begin(), b.begin()), result.begin(), num_items, ::cuda::std::plus<type>{});

  // copy back to host
  thrust::host_vector<type> a_h(num_items);
  thrust::host_vector<type> b_h(num_items);
  thrust::host_vector<type> result_h(num_items);
  REQUIRE(cudaMemcpyAsync(a_h.data(), a.data(), num_items * sizeof(type), cudaMemcpyDeviceToHost, stream.get())
          == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(b_h.data(), b.data(), num_items * sizeof(type), cudaMemcpyDeviceToHost, stream.get())
          == cudaSuccess);
  REQUIRE(
    cudaMemcpyAsync(result_h.data(), result.data(), num_items * sizeof(type), cudaMemcpyDeviceToHost, stream.get())
    == cudaSuccess);
  stream.sync();

  // compute reference and verify
  thrust::host_vector<type> reference_h(num_items);
  std::transform(a_h.begin(), a_h.end(), b_h.begin(), reference_h.begin(), std::plus<type>{});
  REQUIRE(reference_h == result_h);
}

struct add_kernel
{
  template <typename T>
  __device__ void operator()(cuda::std::span<T> a, cuda::std::span<const T> b)
  {
    for (int i = cudax::hierarchy::rank(cudax::thread, cudax::grid); i < a.size();
         i += cudax::hierarchy::count(cudax::thread, cudax::grid))
    {
      a[i] += b[i];
    }
  }
};

C2H_CCCLRT_TEST("cudax::async_buffer launch transform", "[container][async_buffer]")
{
  cudax::stream stream{cuda::device_ref{0}};
  cuda::device_memory_pool_ref resource = cuda::device_default_memory_pool(cuda::device_ref{0});

  const cuda::std::array array = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  cudax::async_device_buffer<int> a       = cudax::make_async_buffer<int>(stream, resource, array);
  const cudax::async_device_buffer<int> b = cudax::make_async_buffer(stream, resource, a.size(), 1);

  cudax::launch(stream, cudax::make_config(cudax::grid_dims<1>, cudax::block_dims<32>), add_kernel{}, a, b);

  std::vector<int> host_result(a.size());
  cudax::copy_bytes(stream, a, host_result);

  stream.sync();

  for (size_t i = 0; i < array.size(); ++i)
  {
    REQUIRE(host_result[i] == array[i] + 1);
  }
}
