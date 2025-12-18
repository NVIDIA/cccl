//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/device/device_transform.cuh>

#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_resource>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <algorithm>

#include "helper.h"
#include "types.h"

C2H_TEST("DeviceTransform::Transform cuda::device_buffer", "[device][launch_transform]")
{
  using type          = int;
  const int num_items = 1 << 24;

  cuda::stream stream{cuda::device_ref{0}};
  cuda::device_memory_pool_ref resource = cuda::device_default_memory_pool(cuda::device_ref{0});

  cuda::device_buffer<type> a{stream, resource, num_items, cuda::no_init};
  cuda::device_buffer<type> b{stream, resource, num_items, cuda::no_init};
  thrust::sequence(thrust::cuda::par_nosync.on(stream.get()), a.begin(), a.end());
  thrust::sequence(thrust::cuda::par_nosync.on(stream.get()), b.begin(), b.end());

  cuda::device_buffer<type> result{stream, resource, num_items, cuda::no_init};

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
    for (int i = cuda::hierarchy::rank(cuda::thread, cuda::grid); i < a.size();
         i += cuda::hierarchy::count(cuda::thread, cuda::grid))
    {
      a[i] += b[i];
    }
  }
};

C2H_CCCLRT_TEST("cuda::buffer launch transform", "[container][buffer]")
{
  cuda::stream stream{cuda::device_ref{0}};
  cuda::device_memory_pool_ref resource = cuda::device_default_memory_pool(cuda::device_ref{0});

  const cuda::std::array array = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  cuda::device_buffer<int> a       = cuda::make_buffer<int>(stream, resource, array);
  const cuda::device_buffer<int> b = cuda::make_buffer(stream, resource, a.size(), 1);

  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<32>()), add_kernel{}, a, b);

  std::vector<int> host_result(a.size());
  cuda::copy_bytes(stream, a, host_result);

  stream.sync();

  for (size_t i = 0; i < array.size(); ++i)
  {
    REQUIRE(host_result[i] == array[i] + 1);
  }
}
