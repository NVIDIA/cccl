// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Allow nested NVTX ranges for this algorithm
// #include "insert_nested_NVTX_range_guard.h"

#include <cub/config.cuh>

// MSVC doesn't support __device__ lambdas
#if !_CCCL_COMPILER(MSVC)

#  include <cub/device/device_for.cuh>

#  include <thrust/device_vector.h>
#  include <thrust/fill.h>
#  include <thrust/host_vector.h>

#  include <cuda/std/array>
#  include <cuda/std/span>

#  include <c2h/catch2_test_helper.h>

// example-begin for-each-in-layout-op
struct layout_store_3D
{
  using data_t = cuda::std::array<int, 3>;

  cuda::std::span<data_t> d_output_raw;

  __device__ void operator()(int idx, int x, int y, int z)
  {
    d_output_raw[idx] = {x, y, z};
  }
};
// example-end for-each-in-layout-op

// clang-format off
C2H_TEST("Device ForEachInLayout", "[ForEachInLayout][device]")
{
  // example-begin for-each-in-layout-example
  using                            data_t = cuda::std::array<int, 3>;
  cuda::std::extents<int, 3, 2, 2> extents{};
  thrust::device_vector<data_t>    d_output(cub::detail::size(extents));
  thrust::host_vector<data_t>      h_output(cub::detail::size(extents));
  auto                             d_output_raw = cuda::std::span<data_t>{thrust::raw_pointer_cast(d_output.data()),
                                                                          3 * 2 * 2};
  thrust::host_vector<data_t>      expected = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0},
                                               {1, 1, 0}, {2, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                               {2, 0, 1}, {0, 1, 1}, {1, 1, 1}, {2, 1, 1}};

  auto status = cub::DeviceFor::ForEachInLayout(cuda::std::layout_left{}, extents,
                                                [=] __device__ (int idx, int x, int y, int z) {
    d_output_raw[idx] = {x, y, z};
  });
  REQUIRE(status == cudaSuccess);
  h_output = d_output;
  REQUIRE(h_output == expected);

  thrust::fill(d_output.begin(), d_output.end(), data_t{});
  status = cub::DeviceFor::ForEachInLayout(cuda::std::layout_left{}, extents, layout_store_3D{d_output_raw});
  REQUIRE(status == cudaSuccess);
  h_output = d_output;
  REQUIRE(h_output == expected);
  // example-end for-each-in-layout-example
}
// clang-format on

#endif // !_CCCL_COMPILER(MSVC)
