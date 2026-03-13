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

#include <cuda/std/array>
#include <cuda/std/mdspan>
#include <cuda/stream>

#include <cuda/experimental/__copy/mdspan_d2d.cuh>

#include "testing.cuh"

static const cuda::stream stream{cuda::device_ref{0}};

template <typename SrcLayout, typename DstLayout, typename T, typename... Ints>
void test_copy(const thrust::host_vector<T>& input, const thrust::host_vector<T>& expected, Ints... shape)
{
  constexpr size_t Rank = sizeof...(Ints);
  using extents_t       = cuda::std::dextents<int, Rank>;
  extents_t ext(static_cast<int>(shape)...);
  typename SrcLayout::template mapping<extents_t> src_mapping(ext);
  typename DstLayout::template mapping<extents_t> dst_mapping(ext);

  thrust::device_vector<T> d_src(input.begin(), input.end());
  thrust::device_vector<T> d_dst(expected.size(), T{0});

  cuda::device_mdspan<T, extents_t, SrcLayout> src(thrust::raw_pointer_cast(d_src.data()), src_mapping);
  cuda::device_mdspan<T, extents_t, DstLayout> dst(thrust::raw_pointer_cast(d_dst.data()), dst_mapping);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<T> result(d_dst);
  CUDAX_REQUIRE(result == expected);
}

template <typename Layout, typename T, typename... Ints>
void test_copy(const thrust::host_vector<T>& data, Ints... shape)
{
  test_copy<Layout, Layout>(data, data, shape...);
}

template <typename T, size_t Rank>
void test_copy_strided(
  const thrust::host_vector<T>& input,
  const thrust::host_vector<T>& expected,
  const cuda::std::array<int, Rank>& shape,
  const cuda::std::array<int, Rank>& src_strides,
  const cuda::std::array<int, Rank>& dst_strides)
{
  using extents_t = cuda::std::dextents<int, Rank>;
  using mapping_t = cuda::std::layout_stride::mapping<extents_t>;
  extents_t ext(shape);
  mapping_t src_mapping(ext, src_strides);
  mapping_t dst_mapping(ext, dst_strides);

  thrust::device_vector<T> d_src(input.begin(), input.end());
  thrust::device_vector<T> d_dst(expected.size(), T{0});

  cuda::device_mdspan<T, extents_t, cuda::std::layout_stride> src(thrust::raw_pointer_cast(d_src.data()), src_mapping);
  cuda::device_mdspan<T, extents_t, cuda::std::layout_stride> dst(thrust::raw_pointer_cast(d_dst.data()), dst_mapping);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<T> result(d_dst);
  CUDAX_REQUIRE(result == expected);
}

template <typename T, size_t Rank>
void test_copy_strided(const thrust::host_vector<T>& data,
                       const cuda::std::array<int, Rank>& shape,
                       const cuda::std::array<int, Rank>& strides)
{
  test_copy_strided(data, data, shape, strides, strides);
}

using cuda::std::layout_left;
using cuda::std::layout_right;

/***********************************************************************************************************************
 * 1D Tests
 **********************************************************************************************************************/

// src: (16):(1)
// dst: (16):(1)
TEST_CASE("copy d2d 1D", "[copy][d2d][1d]")
{
  constexpr int N = 16;
  thrust::host_vector<int> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = i;
  }
  test_copy<layout_right>(data, N);
}

/***********************************************************************************************************************
 * 2D Tests
 **********************************************************************************************************************/

// src: (4,8):(8,1)
// dst: (4,8):(8,1)
TEST_CASE("copy d2d 2D row-major to row-major", "[copy][d2d][2d][basic]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = i;
  }
  test_copy<layout_right>(data, M, N);
}

// src: (4,8):(1,4)
// dst: (4,8):(1,4)
TEST_CASE("copy d2d 2D column-major to column-major", "[copy][d2d][2d][basic]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = i;
  }
  test_copy<layout_left>(data, M, N);
}

// src: (4,8):(8,1)
// dst: (4,8):(1,4)
TEST_CASE("copy d2d 2D row-major to column-major", "[copy][d2d][2d][basic]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = i;
  }
  thrust::host_vector<int> expected(M * N);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      expected[i + j * M] = i * N + j;
    }
  }
  test_copy<layout_right, layout_left>(data, expected, M, N);
}

// src: (4,8):(1,4)
// dst: (4,8):(8,1)
TEST_CASE("copy d2d 2D column-major to row-major", "[copy][d2d][2d][basic]")
{
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = i;
  }
  thrust::host_vector<int> expected(M * N);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      expected[i * N + j] = i + j * M;
    }
  }
  test_copy<layout_left, layout_right>(data, expected, M, N);
}

// src: (1280,2564):(2564,1)
// dst: (1280,2564):(2564,1)
TEST_CASE("copy d2d 2D large", "[copy][d2d][2d][large]")
{
  constexpr int M = 1280;
  constexpr int N = 2564;
  thrust::host_vector<int> data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = i;
  }
  test_copy<layout_right>(data, M, N);
}

/***********************************************************************************************************************
 * 3D Tests
 **********************************************************************************************************************/

// src: (2,3,4):(12,4,1)
// dst: (2,3,4):(12,4,1)
TEST_CASE("copy d2d 3D row-major", "[copy][d2d][3d]")
{
  constexpr int D0 = 2;
  constexpr int D1 = 3;
  constexpr int D2 = 4;
  thrust::host_vector<int> data(D0 * D1 * D2);
  for (int i = 0; i < D0 * D1 * D2; ++i)
  {
    data[i] = i;
  }
  test_copy<layout_right>(data, D0, D1, D2);
}

// src: (2,3,4):(12,4,1)
// dst: (2,3,4):(1,2,6)
TEST_CASE("copy d2d 3D row-major to column-major", "[copy][d2d][3d]")
{
  constexpr int D0    = 2;
  constexpr int D1    = 3;
  constexpr int D2    = 4;
  constexpr int total = D0 * D1 * D2;
  thrust::host_vector<int> data(total);
  for (int i = 0; i < total; ++i)
  {
    data[i] = i;
  }
  thrust::host_vector<int> expected(total);
  for (int i = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        expected[i + j * D0 + k * D0 * D1] = i * (D1 * D2) + j * D2 + k;
      }
    }
  }
  test_copy<layout_right, layout_left>(data, expected, D0, D1, D2);
}

/***********************************************************************************************************************
 * Strided Layout Tests
 **********************************************************************************************************************/

// src: (4,8):(16,1)
// dst: (4,8):(16,1)
TEST_CASE("copy d2d 2D strided padded row-major", "[copy][d2d][2d][stride][row]")
{
  constexpr int M  = 4;
  constexpr int N  = 8;
  constexpr int Ld = 16;
  cuda::std::array<int, 2> shape{M, N};
  cuda::std::array<int, 2> strides{Ld, 1};

  using extents_t      = cuda::std::dextents<int, 2>;
  using mapping_t      = cuda::std::layout_stride::mapping<extents_t>;
  const auto span_size = static_cast<int>(mapping_t(extents_t(shape), strides).required_span_size());

  thrust::host_vector<int> data(span_size, 0);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      data[i * Ld + j] = i * N + j;
    }
  }
  test_copy_strided(data, shape, strides);
}

// src: (4,8):(1,16)
// dst: (4,8):(1,16)
TEST_CASE("copy d2d 2D strided padded column-major", "[copy][d2d][2d][stride][column]")
{
  constexpr int M  = 4;
  constexpr int N  = 8;
  constexpr int Ld = 16;
  cuda::std::array<int, 2> shape{M, N};
  cuda::std::array<int, 2> strides{1, Ld};

  using extents_t      = cuda::std::dextents<int, 2>;
  using mapping_t      = cuda::std::layout_stride::mapping<extents_t>;
  const auto span_size = static_cast<int>(mapping_t(extents_t(shape), strides).required_span_size());

  thrust::host_vector<int> data(span_size, 0);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      data[i + j * Ld] = i * N + j;
    }
  }
  test_copy_strided(data, shape, strides);
}

// src: (2,3,4):(12,4,1)
// dst: (2,3,4):(1,8,2)
TEST_CASE("copy d2d 3D strided permutation", "[copy][d2d][3d][stride][permutation]")
{
  constexpr int D0 = 2;
  constexpr int D1 = 3;
  constexpr int D2 = 4;
  thrust::host_vector<int> input(D0 * D1 * D2);
  for (int i = 0; i < D0 * D1 * D2; ++i)
  {
    input[i] = i;
  }
  cuda::std::array<int, 3> shape{D0, D1, D2};
  cuda::std::array<int, 3> src_strides{D1 * D2, D2, 1};
  cuda::std::array<int, 3> dst_strides{1, D2 * D0, D0};

  using extents_t     = cuda::std::dextents<int, 3>;
  using mapping_t     = cuda::std::layout_stride::mapping<extents_t>;
  const auto dst_span = static_cast<int>(mapping_t(extents_t(shape), dst_strides).required_span_size());

  thrust::host_vector<int> expected(dst_span, 0);
  for (int i = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        expected[i + j * (D2 * D0) + k * D0] = input[i * (D1 * D2) + j * D2 + k];
      }
    }
  }
  test_copy_strided(input, expected, shape, src_strides, dst_strides);
}

// src: (2,3,4):(12,4,1)
// dst: (2,3,4):(8,16,1)
TEST_CASE("copy d2d 3D strided different stride order", "[copy][d2d][3d][stride][tile]")
{
  constexpr int D0 = 2;
  constexpr int D1 = 3;
  constexpr int D2 = 4;
  thrust::host_vector<int> input(D0 * D1 * D2);
  for (int i = 0; i < D0 * D1 * D2; ++i)
  {
    input[i] = i;
  }
  cuda::std::array<int, 3> shape{D0, D1, D2};
  cuda::std::array<int, 3> src_strides{D1 * D2, D2, 1};
  cuda::std::array<int, 3> dst_strides{8, 16, 1};

  using extents_t     = cuda::std::dextents<int, 3>;
  using mapping_t     = cuda::std::layout_stride::mapping<extents_t>;
  const auto dst_span = static_cast<int>(mapping_t(extents_t(shape), dst_strides).required_span_size());

  thrust::host_vector<int> expected(dst_span, 0);
  for (int i = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        expected[i * 8 + j * 16 + k] = input[i * (D1 * D2) + j * D2 + k];
      }
    }
  }
  test_copy_strided(input, expected, shape, src_strides, dst_strides);
}

/***********************************************************************************************************************
 * Different Element Types
 **********************************************************************************************************************/

// src: (1024):(1)
// dst: (1024):(1)
TEST_CASE("copy d2d 1D double", "[copy][d2d][types][double]")
{
  constexpr int N = 1024;
  thrust::host_vector<double> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = static_cast<double>(i) * 0.5;
  }
  test_copy<layout_right>(data, N);
}

// src: (2048):(1)
// dst: (2048):(1)
TEST_CASE("copy d2d 1D short", "[copy][d2d][types][short]")
{
  constexpr int N = 2048;
  thrust::host_vector<short> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = static_cast<short>(i % 1000);
  }
  test_copy<layout_right>(data, N);
}

// src: (4096):(1)
// dst: (4096):(1)
TEST_CASE("copy d2d 1D char", "[copy][d2d][types][char]")
{
  constexpr int N = 4096;
  thrust::host_vector<char> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = static_cast<char>(i % 128);
  }
  test_copy<layout_right>(data, N);
}

/***********************************************************************************************************************
 * Large / Multi-Block Tests
 **********************************************************************************************************************/

// src: (100000):(1)
// dst: (100000):(1)
TEST_CASE("copy d2d 1D large", "[copy][d2d][1d][large]")
{
  constexpr int N = 100000;
  thrust::host_vector<float> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  test_copy<layout_right>(data, N);
}

// src: (13,17):(17,1)
// dst: (13,17):(1,13)
TEST_CASE("copy d2d 2D transposition non-tile-divisible", "[copy][d2d][2d][boundary]")
{
  constexpr int M = 13;
  constexpr int N = 17;
  thrust::host_vector<float> data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  thrust::host_vector<float> expected(M * N);
  for (int r = 0; r < M; ++r)
  {
    for (int c = 0; c < N; ++c)
    {
      expected[r + c * M] = data[r * N + c];
    }
  }
  test_copy<layout_right, layout_left>(data, expected, M, N);
}

// src: (100,200):(200,1)
// dst: (100,200):(1,100)
TEST_CASE("copy d2d 2D large transposition", "[copy][d2d][2d][large][transpose]")
{
  constexpr int M = 100;
  constexpr int N = 200;
  thrust::host_vector<int> data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = i;
  }
  thrust::host_vector<int> expected(M * N);
  for (int r = 0; r < M; ++r)
  {
    for (int c = 0; c < N; ++c)
    {
      expected[r + c * M] = data[r * N + c];
    }
  }
  test_copy<layout_right, layout_left>(data, expected, M, N);
}
