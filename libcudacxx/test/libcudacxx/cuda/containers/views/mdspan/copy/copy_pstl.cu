//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/mdspan>
#include <cuda/std/execution>
#include <cuda/stream>

#include <testing.cuh>

template <class Policy>
void test_pstl_mdspan_copy(const Policy& policy, cuda::stream_ref stream)
{
  constexpr int rows = 37;
  constexpr int cols = 53;

  thrust::host_vector<int> input(rows * cols);
  thrust::host_vector<int> expected(rows * cols);
  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      input[row * cols + col]    = row * cols + col;
      expected[row + col * rows] = row * cols + col;
    }
  }

  thrust::device_vector<int> device_input(input);
  thrust::device_vector<int> device_output(rows * cols, 0);

  using extents_t    = cuda::std::extents<int, rows, cols>;
  using src_mdspan_t = cuda::device_mdspan<const int, extents_t, cuda::std::layout_right>;
  using dst_mdspan_t = cuda::device_mdspan<int, extents_t, cuda::std::layout_left>;

  const src_mdspan_t src(thrust::raw_pointer_cast(device_input.data()));
  const dst_mdspan_t dst(thrust::raw_pointer_cast(device_output.data()));

  cuda::std::copy(policy, src, dst);
  REQUIRE(cudaStreamQuery(stream.get()) == cudaSuccess);

  const thrust::host_vector<int> result(device_output);
  REQUIRE(result == expected);
}

TEST_CASE("cuda::std::copy copies device mdspans with a CUDA execution policy", "[copy][pstl]")
{
  SECTION("default stream")
  {
    test_pstl_mdspan_copy(cuda::execution::gpu, cuda::stream_ref{cudaStream_t{}});
  }

  SECTION("provided stream")
  {
    const cuda::stream stream{cuda::device_ref{0}};
    test_pstl_mdspan_copy(cuda::execution::gpu.with(cuda::get_stream, stream), stream);
  }
}
