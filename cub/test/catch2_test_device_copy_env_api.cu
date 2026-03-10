// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_copy.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>

#include <cuda/devices>
#include <cuda/std/mdspan>
#include <cuda/stream>

#include <iostream>

#include <c2h/catch2_test_helper.h>

template <typename T>
struct index_to_ptr
{
  T* base;
  const int* offsets;
  __host__ __device__ __forceinline__ T* operator()(int index) const
  {
    return base + offsets[index];
  }
};

struct get_size
{
  const int* offsets;
  __host__ __device__ __forceinline__ int operator()(int index) const
  {
    return offsets[index + 1] - offsets[index];
  }
};

C2H_TEST("cub::DeviceCopy::Batched accepts env with stream", "[copy][env]")
{
  // example-begin copy-batched-env
  // 3 ranges of different sizes: [10, 20], [30, 40, 50], [60]
  auto d_src     = thrust::device_vector<int>{10, 20, 30, 40, 50, 60};
  auto d_dst     = thrust::device_vector<int>(6, 0);
  auto d_offsets = thrust::device_vector<int>{0, 2, 5, 6};

  int num_ranges = 3;

  thrust::counting_iterator<int> iota(0);
  auto input_it = thrust::make_transform_iterator(
    iota, index_to_ptr<const int>{thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto output_it = thrust::make_transform_iterator(
    iota, index_to_ptr<int>{thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_offsets.data())});
  auto sizes = thrust::make_transform_iterator(iota, get_size{thrust::raw_pointer_cast(d_offsets.data())});

  cuda::stream stream{cuda::devices[0]};
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref};

  auto error = cub::DeviceCopy::Batched(input_it, output_it, sizes, num_ranges, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceCopy::Batched failed with status: " << error << std::endl;
  }

  thrust::device_vector<int> expected{10, 20, 30, 40, 50, 60};
  // example-end copy-batched-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_dst == expected);
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
