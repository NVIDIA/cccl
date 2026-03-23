// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_copy.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda/devices>
#include <cuda/std/mdspan>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cub::DeviceCopy::Batched accepts env with stream", "[copy][env]")
{
  // example-begin copy-batched-env
  // 3 contiguous ranges copied via Batched API
  constexpr int num_ranges = 3;
  constexpr int range_size = 4;
  constexpr int num_items  = num_ranges * range_size;

  thrust::device_vector<int> d_src(num_items);
  thrust::device_vector<int> d_dst(num_items, 0);
  thrust::sequence(d_src.begin(), d_src.end(), 1);

  // Each range is a contiguous slice of range_size elements
  const int* src_base = thrust::raw_pointer_cast(d_src.data());
  int* dst_base       = thrust::raw_pointer_cast(d_dst.data());

  thrust::device_vector<const int*> d_input_ptrs{src_base, src_base + range_size, src_base + 2 * range_size};
  thrust::device_vector<int*> d_output_ptrs{dst_base, dst_base + range_size, dst_base + 2 * range_size};
  thrust::device_vector<int> d_sizes{range_size, range_size, range_size};

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceCopy::Batched(
    thrust::raw_pointer_cast(d_input_ptrs.data()),
    thrust::raw_pointer_cast(d_output_ptrs.data()),
    thrust::raw_pointer_cast(d_sizes.data()),
    num_ranges,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceCopy::Batched failed with status: " << error << std::endl;
  }
  // example-end copy-batched-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_dst == d_src);
}

C2H_TEST("cub::DeviceCopy::Copy mdspan accepts env with stream", "[copy][env]")
{
  // example-begin copy-mdspan-env
  // Copy a 2D array using mdspan
  constexpr int N = 4;
  constexpr int M = 3;

  thrust::device_vector<float> d_input(N * M);
  thrust::device_vector<float> d_output(N * M, 0.0f);

  // Fill input with sequential values
  thrust::sequence(d_input.begin(), d_input.end(), 1.0f);

  using extents_t = cuda::std::extents<int, N, M>;
  using mdspan_t  = cuda::std::mdspan<float, extents_t, cuda::std::layout_right>; // row-major

  mdspan_t mdspan_in(thrust::raw_pointer_cast(d_input.data()), extents_t{});
  mdspan_t mdspan_out(thrust::raw_pointer_cast(d_output.data()), extents_t{});

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceCopy::Copy(mdspan_in, mdspan_out, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceCopy::Copy failed with status: " << error << std::endl;
  }
  // example-end copy-mdspan-env

  REQUIRE(error == cudaSuccess);
  // Verify the data was copied
  REQUIRE(d_input == d_output);
}
