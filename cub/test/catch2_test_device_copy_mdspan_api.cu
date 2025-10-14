// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_copy.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/std/mdspan>

#include <c2h/catch2_test_helper.h>

void check_status(cudaError_t status)
{
  REQUIRE(status == cudaSuccess);
}

C2H_TEST("DeviceCopy::Copy Mdspan API example", "[copy][mdspan]")
{
  // clang-format off
// example-begin copy-mdspan-example-op
  // Example: Copy a 2D array from row-major to column-major layout
  constexpr int N = 10;
  constexpr int M = 8;

  // Allocate device memory using thrust::device_vector
  thrust::device_vector<float> d_input(N * M);
  thrust::device_vector<float> d_output(N * M, thrust::no_init);

  using extents_t    = cuda::std::extents<int, N, M>;
  using mdspan_in_t  = cuda::std::mdspan<float, extents_t, cuda::std::layout_right>; // row-major
  using mdspan_out_t = cuda::std::mdspan<float, extents_t, cuda::std::layout_left>; // column-major
  // Create mdspans with different layouts
  mdspan_in_t mdspan_in(thrust::raw_pointer_cast(d_input.data()), extents_t{});
  mdspan_out_t mdspan_out(thrust::raw_pointer_cast(d_output.data()), extents_t{});

  // Determine temporary device storage requirements
  void*  d_temp_storage     = nullptr;
  size_t temp_storage_bytes = 0;
  auto status = cub::DeviceCopy::Copy(d_temp_storage, temp_storage_bytes, mdspan_in, mdspan_out);
  check_status(status);

  // Allocate temporary storage using thrust::device_vector
  thrust::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run copy algorithm
  status = cub::DeviceCopy::Copy(d_temp_storage, temp_storage_bytes, mdspan_in, mdspan_out);
  check_status(status);
// example-end copy-mdspan-example-op
  // clang-format on
  REQUIRE(d_input == d_output);
}
