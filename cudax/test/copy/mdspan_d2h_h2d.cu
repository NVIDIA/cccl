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

#include <cuda/std/mdspan>
#include <cuda/stream>

#include <cuda/experimental/__copy/mdspan_d2h_h2d.cuh>

#include "testing.cuh"

static const cuda::stream stream{cuda::device_ref{0}};

template <typename SrcLayout = cuda::std::layout_right,
          typename DstLayout = cuda::std::layout_right,
          typename T,
          typename I,
          size_t... SrcExtents,
          size_t... DstExtents>
void test_impl(const thrust::host_vector<T>& input,
               const thrust::host_vector<T>& expected_data,
               cuda::std::extents<I, SrcExtents...> src_extents,
               cuda::std::extents<I, DstExtents...> dst_extents)
{
  using src_extents_t = cuda::std::extents<I, SrcExtents...>;
  using dst_extents_t = cuda::std::extents<I, DstExtents...>;
  using src_mapping_t = typename SrcLayout::template mapping<src_extents_t>;
  using dst_mapping_t = typename DstLayout::template mapping<dst_extents_t>;
  thrust::device_vector<T> device_data(input.size(), 0);
  {
    // host to device
    using host_mdspan_t   = cuda::host_mdspan<const T, src_extents_t, SrcLayout>;
    using device_mdspan_t = cuda::device_mdspan<T, dst_extents_t, DstLayout>;
    host_mdspan_t host_md(input.data(), src_mapping_t(src_extents));
    device_mdspan_t device_md(thrust::raw_pointer_cast(device_data.data()), dst_mapping_t(dst_extents));
    cuda::experimental::copy_bytes(host_md, device_md, stream);
    stream.sync();
    CUDAX_REQUIRE(thrust::host_vector<T>(device_data) == expected_data);
  }
  {
    // device to host
    using device_mdspan_t = cuda::device_mdspan<const T, src_extents_t, SrcLayout>;
    using host_mdspan_t   = cuda::host_mdspan<T, dst_extents_t, DstLayout>;
    thrust::host_vector<T> host_data(input.size(), 0);
    device_data = input;
    device_mdspan_t device_md(thrust::raw_pointer_cast(device_data.data()), src_mapping_t(src_extents));
    host_mdspan_t host_md(host_data.data(), dst_mapping_t(dst_extents));
    cuda::experimental::copy_bytes(device_md, host_md, stream);
    stream.sync();
    CUDAX_REQUIRE(host_data == expected_data);
  }
}

template <typename T, typename I, size_t... SrcExtents, size_t... DstExtents>
void test_impl_stride(
  const thrust::host_vector<T>& input,
  const thrust::host_vector<T>& expected_data,
  cuda::std::extents<I, SrcExtents...> src_extents,
  cuda::std::extents<I, DstExtents...> dst_extents,
  const cuda::std::array<I, sizeof...(SrcExtents)>& src_strides,
  const cuda::std::array<I, sizeof...(DstExtents)>& dst_strides)
{
  using src_extents_t = cuda::std::extents<I, SrcExtents...>;
  using dst_extents_t = cuda::std::extents<I, DstExtents...>;
  using src_mapping_t = cuda::std::layout_stride::mapping<src_extents_t>;
  using dst_mapping_t = cuda::std::layout_stride::mapping<dst_extents_t>;
  thrust::device_vector<T> device_data(input.size(), 0);
  {
    // host to device
    using host_mdspan_t   = cuda::host_mdspan<const T, src_extents_t, cuda::std::layout_stride>;
    using device_mdspan_t = cuda::device_mdspan<T, dst_extents_t, cuda::std::layout_stride>;
    host_mdspan_t host_md(input.data(), src_mapping_t(src_extents, src_strides));
    device_mdspan_t device_md(thrust::raw_pointer_cast(device_data.data()), dst_mapping_t(dst_extents, dst_strides));
    cuda::experimental::copy_bytes(host_md, device_md, stream);
    stream.sync();
    CUDAX_REQUIRE(thrust::host_vector<T>(device_data) == expected_data);
  }
  {
    // device to host
    using device_mdspan_t = cuda::device_mdspan<const T, src_extents_t, cuda::std::layout_stride>;
    using host_mdspan_t   = cuda::host_mdspan<T, dst_extents_t, cuda::std::layout_stride>;
    thrust::host_vector<T> host_data(input.size(), 0);
    device_data = input;
    device_mdspan_t device_md(thrust::raw_pointer_cast(device_data.data()), src_mapping_t(src_extents, src_strides));
    host_mdspan_t host_md(host_data.data(), dst_mapping_t(dst_extents, dst_strides));
    cuda::experimental::copy_bytes(device_md, host_md, stream);
    stream.sync();
    CUDAX_REQUIRE(host_data == expected_data);
  }
}

/***********************************************************************************************************************
 * 1D Tests
 **********************************************************************************************************************/

// tensorA:      (16):(1)
// tensorB:      (16):(1)
// stride order: true
// tile size:    16
// num tiles:    1
TEST_CASE("copy_bytes 1D", "[copy_bytes][1d]")
{
  constexpr int N = 16;
  thrust::host_vector<int> host_data(N);
  for (int i = 0; i < N; ++i)
  {
    host_data[i] = i;
  }
  using extents = cuda::std::extents<int, N>;
  test_impl(host_data, host_data, extents(), extents());
  test_impl<cuda::std::layout_left, cuda::std::layout_left>(host_data, host_data, extents(), extents());
}

/***********************************************************************************************************************
 * 2D Tests
 **********************************************************************************************************************/

TEST_CASE("copy_bytes 2D", "[copy_bytes][2d][basic]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> host_data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    host_data[i] = i;
  }
  using extents = cuda::std::extents<int, M, N>;
  // row major to row major
  // tensorA:      (4,8):(8,1)
  // tensorB:      (4,8):(8,1)
  // stride order: true
  // tile size:    32
  // num tiles:    1
  test_impl(host_data, host_data, extents(), extents());
  // column major to column major
  // tensorA:      (4,8):(1,4)
  // tensorB:      (4,8):(1,4)
  // stride order: true
  // tile size:    32
  // num tiles:    1
  test_impl<cuda::std::layout_left, cuda::std::layout_left>(host_data, host_data, extents(), extents());
  // row major to column major
  thrust::host_vector<int> expected(M * N);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      expected[i + j * M] = i * N + j;
    }
  }
  // tensorA:      (4,8):(8,1)
  // tensorB:      (4,8):(1,4)
  // stride order: false
  // tile size:    1
  // num tiles:    32
  test_impl<cuda::std::layout_right, cuda::std::layout_left>(host_data, expected, extents(), extents());
  // column major to row major
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      expected[i * N + j] = i + j * M;
    }
  }
  // tensorA:      (4,8):(1,4)
  // tensorB:      (4,8):(8,1)
  // stride order: false
  // tile size:    1
  // num tiles:    32
  test_impl<cuda::std::layout_left, cuda::std::layout_right>(host_data, expected, extents(), extents());
}

TEST_CASE("copy_bytes 2D swapped extents", "[copy_bytes][2d][swapped]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> host_data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    host_data[i] = i;
  }
  using src_extents = cuda::std::extents<int, M, N>;
  using dst_extents = cuda::std::extents<int, N, M>;
  // row major to row major
  // tensorA:      (4,8):(8,1)
  // tensorB:      (8,4):(4,1)
  // stride order: true
  // tile size:    32
  // num tiles:    1
  test_impl(host_data, host_data, src_extents(), dst_extents());
  // column major to column major
  // tensorA:      (4,8):(1,4)
  // tensorB:      (8,4):(1,8)
  // stride order: true
  // tile size:    32
  // num tiles:    1
  test_impl<cuda::std::layout_left, cuda::std::layout_left>(host_data, host_data, src_extents(), dst_extents());
  thrust::host_vector<int> expected(M * N);
  // row major to column major (swapped extents)
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < M; ++j)
    {
      expected[i + j * N] = i * M + j;
    }
  }
  // row major to column major
  // tensorA:      (4,8):(8,1)
  // tensorB:      (8,4):(1,8)
  // stride order: false
  // tile size:    1
  // num tiles:    32
  test_impl<cuda::std::layout_right, cuda::std::layout_left>(host_data, expected, src_extents(), dst_extents());
}

// row major to row major
// tensorA:      (4,8):(8,1)
// tensorB:      (32):(1)
// stride order: true
// tile size:    32
// num tiles:    1
// --------------------------------
// column major to column major
// tensorA:      (4,8):(1,4)
// tensorB:      (32):(1)
// stride order: false
// tile size:    1
// num tiles:    32
TEST_CASE("copy_bytes 2D mixed ranks", "[copy_bytes][2d][ranks]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> host_data(M * N);
  for (int i = 0; i < static_cast<int>(M * N); ++i)
  {
    host_data[i] = i;
  }
  using src_extents = cuda::std::extents<int, M, N>; // 2D
  using dst_extents = cuda::std::extents<int, M * N>; // 1D
  // row major to row major
  test_impl(host_data, host_data, src_extents(), dst_extents());
  // column major to column major
  thrust::host_vector<int> expected(M * N);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      expected[i * N + j] = i + j * M;
    }
  }
  test_impl<cuda::std::layout_left, cuda::std::layout_left>(host_data, expected, src_extents(), dst_extents());
}

// row major to row major
// tensorA:      (1280,2564):(2564,1)
// tensorB:      (1280,2564):(2564,1)
// stride order: true
// tile size:    3281920
// num tiles:    1
TEST_CASE("copy_bytes 2D large", "[copy_bytes][2d][large]")
{
  constexpr int M = 1280;
  constexpr int N = 2564;
  thrust::host_vector<int> host_data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    host_data[i] = i;
  }
  using extents = cuda::std::extents<int, M, N>;
  test_impl(host_data, host_data, extents(), extents());
}

/***********************************************************************************************************************
 * 3D Test
 **********************************************************************************************************************/

// row major to row major
// tensorA:      (2,3,4):(12,4,1)
// tensorB:      (2,3,4):(12,4,1)
// stride order: true
// tile size:    24
// num tiles:    1
//--------------------------------
// row major to column major
// tensorA:      (2,3,4):(12,4,1)
// tensorB:      (2,3,4):(1,2,6)
// stride order: false
// tile size:    1
// num tiles:    24
TEST_CASE("copy_bytes 3D", "[copy_bytes][3d]")
{
  constexpr int D0    = 2;
  constexpr int D1    = 3;
  constexpr int D2    = 4;
  constexpr int total = D0 * D1 * D2;
  thrust::host_vector<int> host_data(total);
  for (int i = 0; i < total; ++i)
  {
    host_data[i] = i;
  }
  using extents = cuda::std::extents<int, D0, D1, D2>;
  // row major to row major
  test_impl(host_data, host_data, extents(), extents());
  // row major to column major
  thrust::host_vector<int> expected(total);
  for (int i = 0, p = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        expected[i + j * D0 + k * D0 * D1] = p++;
      }
    }
  }
  test_impl<cuda::std::layout_right, cuda::std::layout_left>(host_data, expected, extents(), extents());
}

/***********************************************************************************************************************
 * Edge Cases
 **********************************************************************************************************************/

TEST_CASE("copy_bytes rank 0", "[copy_bytes][0d]")
{
  thrust::host_vector<int> host_data(1, 42);
  using extents = cuda::std::extents<int>;
  test_impl(host_data, host_data, extents(), extents());
}

TEST_CASE("copy_bytes size 0", "[copy_bytes][zero_size]")
{
  int value = 42;
  thrust::device_vector<int> device_data(1, 0);
  cuda::host_mdspan<int, cuda::std::dims<2>> src(&value, 0, 0);
  cuda::device_mdspan<int, cuda::std::dims<2>> device_md(thrust::raw_pointer_cast(device_data.data()), 0, 0);
  cuda::experimental::copy_bytes(src, device_md, stream);
  stream.sync();
}

TEST_CASE("copy_bytes size mismatch throws", "[copy_bytes][throw]")
{
  constexpr int N = 16;
  thrust::host_vector<int> host_data(N, 0);
  thrust::device_vector<int> device_data(N / 2, 0);
  using extents = cuda::std::dims<1>;
  cuda::host_mdspan<int, extents> src(host_data.data(), extents(N));
  cuda::device_mdspan<int, extents> dst(thrust::raw_pointer_cast(device_data.data()), extents(N / 2));
  REQUIRE_THROWS_AS(cuda::experimental::copy_bytes(src, dst, stream), std::invalid_argument);
}

/***********************************************************************************************************************
 * Strided Layout Tests
 **********************************************************************************************************************/

// tensorA:      (4,8):(16,1)
// tensorB:      (4,8):(16,1)
// stride order: true
// tile size:    8
// num tiles:    4
TEST_CASE("copy_bytes 2D strided, padded layout", "[copy_bytes][2d][stride][row]")
{
  constexpr int M  = 4;
  constexpr int N  = 8;
  constexpr int Ld = 16;
  thrust::host_vector<int> host_data(Ld * N, 0);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      host_data[i * N * 2 + j] = i * N + j;
    }
  }
  using src_extents                = cuda::std::extents<int, M, N>;
  using dst_extents                = cuda::std::extents<int, M, N>;
  cuda::std::array<int, 2> strides = {Ld, 1};
  test_impl_stride(host_data, host_data, src_extents(), dst_extents(), strides, strides);
}

// tensorA:      (4,8):(1,16)
// tensorB:      (4,8):(1,16)
// stride order: true
// tile size:    4
// num tiles:    8
TEST_CASE("copy_bytes 2D strided, padded column-major", "[copy_bytes][2d][stride][column]")
{
  constexpr int M  = 4;
  constexpr int N  = 8;
  constexpr int Ld = 16;
  thrust::host_vector<int> input_data(Ld * N, 0);
  int k = 0;
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      input_data[i + j * Ld] = k++;
    }
  }
  using extents                    = cuda::std::extents<int, M, N>;
  cuda::std::array<int, 2> strides = {1, Ld};
  test_impl_stride(input_data, input_data, extents(), extents(), strides, strides);
}

// tensorA:      (2,3,4):(12,4,1)
// tensorB:      (2,3,4):(1,8,2)
// stride order: false
// tile size:    1
// num tiles:    24
TEST_CASE("copy_bytes 3D strided, permutation", "[copy_bytes][3d][stride][permutation]")
{
  // src: (2, 3, 4):(12, 4, 1) -> order [0, 1, 2]
  // dst: (2, 3, 4):(1, 8, 2)  -> order [1, 2, 0]
  constexpr int D0 = 2;
  constexpr int D1 = 3;
  constexpr int D2 = 4;
  constexpr int N  = D0 * D1 * D2;
  thrust::host_vector<int> input_data(N);
  for (int i = 0; i < N; ++i)
  {
    input_data[i] = i;
  }
  thrust::host_vector<int> expected(N, -1);
  for (int i = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        const int src_offset = i * (D1 * D2) + j * D2 + k; // row-major
        const int dst_offset = i + j * (D2 * D0) + k * D0; // strides {1, 8, 2}
        expected[dst_offset] = input_data[src_offset];
      }
    }
  }
  using extents                        = cuda::std::extents<int, D0, D1, D2>;
  cuda::std::array<int, 3> src_strides = {D1 * D2, D2, 1};
  cuda::std::array<int, 3> dst_strides = {1, D2 * D0, D0};
  test_impl_stride(input_data, expected, extents(), extents(), src_strides, dst_strides);
}

// tensorA:      (2,3,4):(12,4,1)
// tensorB:      (2,3,4):(8,16,1)
// stride order: false
// tile size:    4
// num tiles:    6
TEST_CASE("copy_bytes 3D strided, tile_size > 1 with different stride order", "[copy_bytes][3d][stride][tile]")
{
  // src: (2, 3, 4):(12, 4, 1)  -> order [0, 1, 2]
  // dst: (2, 3, 4):(8, 16, 1)  -> order [1, 0, 2]
  constexpr int D0 = 2;
  constexpr int D1 = 3;
  constexpr int D2 = 4;
  constexpr int N  = 44; // codomain size: (2 - 1) * 8 + (3 - 1) * 16 + (4 - 1) * 1 = 44
  thrust::host_vector<int> input_data(N, -1);
  for (int i = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        const int src_offset   = i * (D1 * D2) + j * D2 + k; // row-major
        input_data[src_offset] = src_offset;
      }
    }
  }
  thrust::host_vector<int> expected(N, 0);
  for (int i = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        const int src_offset = i * (D1 * D2) + j * D2 + k; // row-major
        const int dst_offset = i * 8 + j * 16 + k; // strides {8, 16, 1}
        expected[dst_offset] = src_offset;
      }
    }
  }
  using extents                        = cuda::std::extents<int, D0, D1, D2>;
  cuda::std::array<int, 3> src_strides = {D1 * D2, D2, 1};
  cuda::std::array<int, 3> dst_strides = {8, 16, 1};
  test_impl_stride(input_data, expected, extents(), extents(), src_strides, dst_strides);
}
