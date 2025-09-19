// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_copy.cuh>
#include <cuda/std/mdspan>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <c2h/catch2_test_helper.h>

C2H_TEST("Device ForEachInExtents", "[ForEachInExtents][device]")
{
  // clang-format off
// example-begin copy-mdspan-example-op
  // Example: Copy a 2D array from row-major to column-major layout
  constexpr int N = 10;
  constexpr int M = 8;
  
  // Allocate device memory using thrust::device_vector
  thrust::device_vector<float> d_input(N * M);
  thrust::device_vector<float> d_output(N * M);
  
  using extents_t    = cuda::std::extents<int, N, M>;
  using mdspan_in_t  = cuda::std::mdspan<float, extents_t, cuda::std::layout_right>; // row-major
  using mdspan_out_t = cuda::std::mdspan<float, extents_t, cuda::std::layout_left>; // column-major
  // Create mdspans with different layouts
  mdspan_in_t mdspan_in(thrust::raw_pointer_cast(d_input.data()), extents_t{});  
  mdspan_out_t mdspan_out(thrust::raw_pointer_cast(d_output.data()), extents_t{});  
  
  // Determine temporary device storage requirements
  void*  d_temp_storage     = nullptr;
  size_t temp_storage_bytes = 0;
  assert(cub::DeviceCopy::Copy(d_temp_storage, temp_storage_bytes, mdspan_in, mdspan_out) == cudaSuccess);
  
  // Allocate temporary storage using thrust::device_vector
  thrust::device_vector<char> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  
  // Run copy algorithm
  assert(cub::DeviceCopy::Copy(d_temp_storage, temp_storage_bytes, mdspan_in, mdspan_out) == cudaSuccess);
// example-end copy-mdspan-example-op
  // clang-format on
  REQUIRE(d_input == d_output);
  static_cast<void>(d_temp_storage);
}
