// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allow nested NVTX ranges for this algorithm
// #include "insert_nested_NVTX_range_guard.h"

#include <cub/config.cuh>

// MSVC doesn't support __device__ lambdas
#if !_CCCL_COMPILER(MSVC)

#  include <cub/device/device_for.cuh>

#  include <thrust/device_vector.h>
#  include <thrust/equal.h>
#  include <thrust/fill.h>
#  include <thrust/host_vector.h>

#  include <cuda/std/array>
#  include <cuda/std/span>

#  include <cstdlib>
#  include <iostream>

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
  using data_t             = cuda::std::array<int, 3>;
  using extents_type       = cuda::std::extents<int, 3, 2, 2>;
  using mapping_left_type  = cuda::std::layout_left::mapping<extents_type>;
  extents_type                  extents{};
  thrust::device_vector<data_t> d_output(cub::detail::size(extents), thrust::no_init);
  thrust::host_vector<data_t>   h_output(cub::detail::size(extents), thrust::no_init);
  auto                          d_output_raw = cuda::std::span<data_t>{thrust::raw_pointer_cast(d_output.data()),
                                                                       3 * 2 * 2};
  thrust::host_vector<data_t>   expected = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0},
                                            {0, 1, 0}, {1, 1, 0}, {2, 1, 0},
                                            {0, 0, 1}, {1, 0, 1}, {2, 0, 1},
                                            {0, 1, 1}, {1, 1, 1}, {2, 1, 1}};

  auto status = cub::DeviceFor::ForEachInLayout(mapping_left_type{extents},
                                                [=] __device__ (int idx, int x, int y, int z) {
    d_output_raw[idx] = {x, y, z};
  });
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceFor::ForEachInLayout failed with status: " << status << std::endl;
    std::exit(EXIT_FAILURE);
  }
  h_output = d_output;
  if (!thrust::equal(h_output.begin(), h_output.end(), expected.begin()))
  {
    std::cerr << "error: h_output != expected" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  thrust::fill(d_output.begin(), d_output.end(), data_t{});
  status = cub::DeviceFor::ForEachInLayout(mapping_left_type{extents}, layout_store_3D{d_output_raw});
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceFor::ForEachInLayout failed with status: " << status << std::endl;
    std::exit(EXIT_FAILURE);
  }
  h_output = d_output;
  if (!thrust::equal(h_output.begin(), h_output.end(), expected.begin()))
  {
    std::cerr << "error: h_output != expected" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // example-end for-each-in-layout-example
}
// clang-format on

#endif // !_CCCL_COMPILER(MSVC)
