// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_memcpy.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceMemcpy::Batched accepts env with stream", "[memcpy][env]")
{
  // example-begin memcpy-batched-env
  // Source data: 3 buffers of different sizes laid out contiguously
  // Buffer 0: [10, 20]     Buffer 1: [30, 40, 50]     Buffer 2: [60]
  auto d_src = thrust::device_vector<int>{10, 20, 30, 40, 50, 60};

  // Copy into two separate destination buffers to highlight the API's flexibility
  auto d_dst_a = thrust::device_vector<int>(5, 0);
  auto d_dst_b = thrust::device_vector<int>(1, 0);

  // Source pointers: one per buffer, pointing into d_src
  auto d_src_ptrs = thrust::device_vector<const int*>{
    thrust::raw_pointer_cast(d_src.data()) + 0,
    thrust::raw_pointer_cast(d_src.data()) + 2,
    thrust::raw_pointer_cast(d_src.data()) + 5};

  // Destination pointers: buffers 0,1 go to d_dst_a, buffer 2 goes to d_dst_b
  auto d_dst_ptrs = thrust::device_vector<int*>{
    thrust::raw_pointer_cast(d_dst_a.data()) + 0,
    thrust::raw_pointer_cast(d_dst_a.data()) + 2,
    thrust::raw_pointer_cast(d_dst_b.data()) + 0};

  // Sizes in bytes for each buffer
  auto d_sizes = thrust::device_vector<int>{
    2 * static_cast<int>(sizeof(int)), 3 * static_cast<int>(sizeof(int)), 1 * static_cast<int>(sizeof(int))};

  int num_buffers = 3;

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceMemcpy::Batched(
    thrust::raw_pointer_cast(d_src_ptrs.data()),
    thrust::raw_pointer_cast(d_dst_ptrs.data()),
    thrust::raw_pointer_cast(d_sizes.data()),
    num_buffers,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceMemcpy::Batched failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_a{10, 20, 30, 40, 50};
  thrust::device_vector<int> expected_b{60};
  // example-end memcpy-batched-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_dst_a == expected_a);
  REQUIRE(d_dst_b == expected_b);
}
