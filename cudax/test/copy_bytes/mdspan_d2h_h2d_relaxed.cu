//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/mdspan>
#include <cuda/stream>

#include <cuda/experimental/copy_bytes.cuh>

#include "testing.cuh"

static const cuda::stream stream{cuda::device_ref{0}};

// Source uses layout_stride_relaxed, destination uses layout_right with the same extents.
template <typename T, typename SrcMapping>
void test_impl_relaxed(
  const thrust::host_vector<T>& input, const thrust::host_vector<T>& expected, const SrcMapping& src_mapping)
{
  using extents_t     = typename SrcMapping::extents_type;
  using dst_mapping_t = cuda::std::layout_right::mapping<extents_t>;

  const dst_mapping_t dst_mapping(src_mapping.extents());

  thrust::device_vector<T> device_data(expected.size(), 0);
  {
    // host to device
    cuda::host_mdspan<const T, extents_t, cuda::layout_stride_relaxed> host_md(input.data(), src_mapping);
    cuda::device_mdspan<T, extents_t> device_md(thrust::raw_pointer_cast(device_data.data()), dst_mapping);
    cuda::experimental::copy_bytes(host_md, device_md, stream);
    stream.sync();
    CUDAX_REQUIRE(thrust::host_vector<T>(device_data) == expected);
  }
  {
    // device to host
    thrust::host_vector<T> host_output(expected.size(), 0);
    thrust::device_vector<T> device_input(input.begin(), input.end());
    cuda::device_mdspan<const T, extents_t, cuda::layout_stride_relaxed> device_md(
      thrust::raw_pointer_cast(device_input.data()), src_mapping);
    cuda::host_mdspan<T, extents_t> host_md(host_output.data(), dst_mapping);
    cuda::experimental::copy_bytes(device_md, host_md, stream);
    stream.sync();
    CUDAX_REQUIRE(host_output == expected);
  }
}

/***********************************************************************************************************************
 * layout_stride_relaxed Tests
 **********************************************************************************************************************/

TEST_CASE("copy_bytes layout_stride_relaxed, sliding window", "[copy_bytes][relaxed][window]")
{
  // A sliding window (Toeplitz matrix) using strides {1, 1}:
  // data = [0, 1, 2, ..., 2N-2], src(i, j) = data[i + j]
  constexpr int N       = 5;
  constexpr int DataLen = 2 * N - 1; // = 9
  thrust::host_vector<int> input(DataLen);
  for (int i = 0; i < DataLen; ++i)
  {
    input[i] = i;
  }
  thrust::host_vector<int> expected(N * N);
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      expected[i * N + j] = i + j;
    }
  }

  using extents_t = cuda::std::extents<int, N, N>;
  using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;

  test_impl_relaxed(input, expected, mapping_t(extents_t{}, cuda::dstrides<int, 2>(1, 1)));
}

TEST_CASE("copy_bytes layout_stride_relaxed, reversed 1D", "[copy_bytes][relaxed][1d]")
{
  constexpr int N = 8;
  thrust::host_vector<int> input(N);
  for (int i = 0; i < N; ++i)
  {
    input[i] = i;
  }
  // strides = {-1}, offset = N-1 -> src(i) = input[(N-1) - i]
  thrust::host_vector<int> expected(N);
  for (int i = 0; i < N; ++i)
  {
    expected[i] = N - 1 - i;
  }
  using extents_t = cuda::std::extents<int, N>;
  using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;
  test_impl_relaxed(input, expected, mapping_t(extents_t{}, cuda::dstrides<int, 1>(-1), N - 1));
}

TEST_CASE("copy_bytes layout_stride_relaxed, 2D column-reversed", "[copy_bytes][relaxed][2d]")
{
  constexpr int M = 3;
  constexpr int N = 4;
  // row-major input: {0, 1, 2, ..., 11}
  thrust::host_vector<int> input(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    input[i] = i;
  }
  // strides = {N, -1}, offset = N-1 -> src(i,j) = input[(N-1) + i*N - j]
  // Each row is reversed: row i reads input[i*N + (N-1)], input[i*N + (N-2)], ...
  thrust::host_vector<int> expected(M * N);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      expected[i * N + j] = i * N + (N - 1 - j);
    }
  }
  using extents_t = cuda::std::extents<int, M, N>;
  using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;
  test_impl_relaxed(input, expected, mapping_t(extents_t{}, cuda::dstrides<int, 2>(N, -1), N - 1));
}

TEST_CASE("copy_bytes layout_stride_relaxed, offset subarray", "[copy_bytes][relaxed][offset]")
{
  constexpr int Rows    = 3;
  constexpr int Cols    = 2;
  constexpr int Ld      = 4;
  constexpr int Offset  = 2;
  constexpr int DataLen = Offset + (Rows - 1) * Ld + (Cols - 1) + 1; // = 12
  // input: [0, 1, 2, ..., 11]
  thrust::host_vector<int> input(DataLen);
  for (int i = 0; i < DataLen; ++i)
  {
    input[i] = i;
  }
  // strides = {Ld, 1}, src(i,j) = input[Offset + i*Ld + j]
  // src(0,0) = input[2]  = 2,  src(0,1) = input[3]  = 3
  // src(1,0) = input[6]  = 6,  src(1,1) = input[7]  = 7
  // src(2,0) = input[10] = 10, src(2,1) = input[11] = 11
  thrust::host_vector<int> expected = {2, 3, 6, 7, 10, 11};

  using extents_t = cuda::std::extents<int, Rows, Cols>;
  using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;
  test_impl_relaxed(input, expected, mapping_t(extents_t{}, cuda::dstrides<int, 2>(Ld, 1), Offset));
}
