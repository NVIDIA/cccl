// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Allow nested NVTX ranges for this algorithm
// #include "insert_nested_NVTX_range_guard.h"

#include <cub/config.cuh>

// MSVC doesn't support __device__ lambdas
#if !_CCCL_COMPILER(MSVC)

#  include <cub/device/device_for.cuh>

#  include <thrust/detail/raw_pointer_cast.h>
#  include <thrust/device_vector.h>
#  include <thrust/equal.h>
#  include <thrust/fill.h>
#  include <thrust/host_vector.h>

#  include <cuda/std/array>
#  include <cuda/std/mdspan>
#  include <cuda/std/span>

#  include <cstdlib>
#  include <iostream>

#  include <c2h/catch2_test_helper.h>

// example-begin for-each-in-extents-op
struct linear_store_3D
{
  using data_t = cuda::std::array<int, 3>;

  cuda::std::span<data_t> d_output1_raw;

  __device__ void operator()(int idx, int x, int y, int z)
  {
    d_output1_raw[idx] = {x, y, z};
  }
};
// example-end for-each-in-extents-op

// clang-format off
C2H_TEST("Device ForEachInExtents", "[ForEachInExtents][device]")
{
  // example-begin for-each-in-extents-example
  using                            data_t = cuda::std::array<int, 3>;
  cuda::std::extents<int, 3, 2, 2> extents{};
  thrust::device_vector<data_t>    d_output1(cub::detail::size(extents), thrust::no_init);
  thrust::device_vector<data_t>    d_output2(cub::detail::size(extents), thrust::no_init);
  auto                             d_output1_raw = cuda::std::span<data_t>{thrust::raw_pointer_cast(d_output1.data()),
                                                                          3 * 2 * 2};
  auto                             d_output2_raw = cuda::std::span<data_t>{thrust::raw_pointer_cast(d_output2.data()),
                                                                          3 * 2 * 2};
  thrust::host_vector<data_t>      expected = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                               {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
                                               {2, 0, 0}, {2, 0, 1}, {2, 1, 0}, {2, 1, 1}};

  cub::DeviceFor::ForEachInExtents(extents, [=] __device__ (int idx, int x, int y, int z) {
    d_output1_raw[idx] = {x, y, z};
  });
  // d_output1 is now filled with the expected values

  thrust::fill(d_output2.begin(), d_output2.end(), data_t{});
  cub::DeviceFor::ForEachInExtents(extents, linear_store_3D{d_output2_raw});
  // d_output2 is now filled with the expected values

  // example-end for-each-in-extents-example
  thrust::host_vector<data_t> h_output1(cub::detail::size(extents), thrust::no_init);
  thrust::host_vector<data_t> h_output2(cub::detail::size(extents), thrust::no_init);
  h_output1 = d_output1;
  h_output2 = d_output2;
  REQUIRE(h_output1 == expected);
  REQUIRE(h_output2 == expected);
}
// clang-format on

#endif // !_CCCL_COMPILER(MSVC)
