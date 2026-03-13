//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/host_vector.h>

#include "copy_bytes_common.cuh"
#include <cute/layout.hpp>

/***********************************************************************************************************************
 * 1D Tests
 **********************************************************************************************************************/

// tensorA: (16):(1)
// tensorB: (16):(1)
TEST_CASE("copy_bytes 1D", "[copy_bytes][1d]")
{
  using namespace cute;
  constexpr int N = 16;
  thrust::host_vector<int> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = i;
  }
  test_impl(data, make_layout(make_shape(N), make_stride(1)));
}

/***********************************************************************************************************************
 * 2D Tests
 **********************************************************************************************************************/

// tensorA: (4,8)
// tensorB: (4,8)
// All row-major/column-major combinations
TEST_CASE("copy_bytes 2D", "[copy_bytes][2d][basic]")
{
  using namespace cute;
  constexpr int M = 4;
  constexpr int N = 8;
  thrust::host_vector<int> data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = i;
  }
  auto shape = make_shape(M, N);

  // row major to row major: (4,8):(8,1) -> (4,8):(8,1)
  test_impl(data, make_layout(shape, make_stride(N, 1)));

  // column major to column major: (4,8):(1,4) -> (4,8):(1,4)
  test_impl(data, make_layout(shape, make_stride(1, M)));

  // row major to column major: (4,8):(8,1) -> (4,8):(1,4)
  thrust::host_vector<int> expected_r2c(M * N);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      expected_r2c[i + j * M] = i * N + j;
    }
  }
  test_impl(data, expected_r2c, make_layout(shape, make_stride(N, 1)), make_layout(shape, make_stride(1, M)));

  // column major to row major: (4,8):(1,4) -> (4,8):(8,1)
  thrust::host_vector<int> expected_c2r(M * N);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      expected_c2r[i * N + j] = i + j * M;
    }
  }
  test_impl(data, expected_c2r, make_layout(shape, make_stride(1, M)), make_layout(shape, make_stride(N, 1)));
}

// tensorA: (1280,2564):(2564,1)
// tensorB: (1280,2564):(2564,1)
TEST_CASE("copy_bytes 2D large", "[copy_bytes][2d][large]")
{
  using namespace cute;
  constexpr int M = 1280;
  constexpr int N = 2564;
  thrust::host_vector<int> data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    data[i] = i;
  }
  test_impl(data, make_layout(make_shape(M, N), make_stride(N, 1)));
}

/***********************************************************************************************************************
 * 3D Tests
 **********************************************************************************************************************/

// tensorA: (2,3,4)
// tensorB: (2,3,4)
// row-major to row-major, column-major to column-major
TEST_CASE("copy_bytes 3D", "[copy_bytes][3d]")
{
  using namespace cute;
  constexpr int D0    = 2;
  constexpr int D1    = 3;
  constexpr int D2    = 4;
  constexpr int total = D0 * D1 * D2;
  thrust::host_vector<int> data(total);
  for (int i = 0; i < total; ++i)
  {
    data[i] = i;
  }
  auto shape = make_shape(D0, D1, D2);

  // row major to row major: (2,3,4):(12,4,1) -> same
  test_impl(data, make_layout(shape, make_stride(D1 * D2, D2, 1)));

  // row major to column major: (2,3,4):(12,4,1) -> (2,3,4):(1,2,6)
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
  test_impl(
    data, expected, make_layout(shape, make_stride(D1 * D2, D2, 1)), make_layout(shape, make_stride(1, D0, D0 * D1)));
}

/***********************************************************************************************************************
 * Strided Layout Tests
 **********************************************************************************************************************/

// tensorA: (4,8):(16,1)
// tensorB: (4,8):(16,1)
TEST_CASE("copy_bytes 2D strided, padded row-major", "[copy_bytes][2d][stride][row]")
{
  using namespace cute;
  constexpr int M  = 4;
  constexpr int N  = 8;
  constexpr int Ld = 16;
  thrust::host_vector<int> data(Ld * M, 0);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      data[i * Ld + j] = i * N + j;
    }
  }
  auto layout = make_layout(make_shape(M, N), make_stride(Ld, 1));
  test_impl(data, layout);
}

// tensorA: (4,8):(1,16)
// tensorB: (4,8):(1,16)
TEST_CASE("copy_bytes 2D strided, padded column-major", "[copy_bytes][2d][stride][column]")
{
  using namespace cute;
  constexpr int M  = 4;
  constexpr int N  = 8;
  constexpr int Ld = 16;
  thrust::host_vector<int> data(Ld * N, 0);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      data[i + j * Ld] = i * N + j;
    }
  }
  auto layout = make_layout(make_shape(M, N), make_stride(1, Ld));
  test_impl(data, layout);
}

// src: (2,3,4):(12,4,1) -> dst: (2,3,4):(1,8,2)
TEST_CASE("copy_bytes 3D strided, permutation", "[copy_bytes][3d][stride][permutation]")
{
  using namespace cute;
  constexpr int D0 = 2;
  constexpr int D1 = 3;
  constexpr int D2 = 4;
  constexpr int N  = D0 * D1 * D2;
  thrust::host_vector<int> input(N);
  for (int i = 0; i < N; ++i)
  {
    input[i] = i;
  }
  thrust::host_vector<int> expected(N, -1);
  for (int i = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        const int src_off = i * (D1 * D2) + j * D2 + k;
        const int dst_off = i + j * (D2 * D0) + k * D0;
        expected[dst_off] = input[src_off];
      }
    }
  }
  auto shape = make_shape(D0, D1, D2);
  test_impl(
    input, expected, make_layout(shape, make_stride(D1 * D2, D2, 1)), make_layout(shape, make_stride(1, D2 * D0, D0)));
}

// TensorA: (2,3,4):(12,4,1)
// TensorB: (2,3,4):(8,16,1)
TEST_CASE("copy_bytes 3D strided, different stride order", "[copy_bytes][3d][stride][tile]")
{
  using namespace cute;
  constexpr int D0         = 2;
  constexpr int D1         = 3;
  constexpr int D2         = 4;
  constexpr int alloc_size = (D0 - 1) * 8 + (D1 - 1) * 16 + (D2 - 1) * 1 + 1; // 44
  thrust::host_vector<int> input(alloc_size, -1);
  for (int i = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        input[i * (D1 * D2) + j * D2 + k] = i * (D1 * D2) + j * D2 + k;
      }
    }
  }
  thrust::host_vector<int> expected(alloc_size, 0);
  for (int i = 0; i < D0; ++i)
  {
    for (int j = 0; j < D1; ++j)
    {
      for (int k = 0; k < D2; ++k)
      {
        const int src_off = i * (D1 * D2) + j * D2 + k;
        const int dst_off = i * 8 + j * 16 + k;
        expected[dst_off] = src_off;
      }
    }
  }
  auto shape = make_shape(D0, D1, D2);
  test_impl(input, expected, make_layout(shape, make_stride(D1 * D2, D2, 1)), make_layout(shape, make_stride(8, 16, 1)));
}

/***********************************************************************************************************************
 * Different Element Types
 **********************************************************************************************************************/

// tensorA: (1024):(1)
// tensorB: (1024):(1)
TEST_CASE("copy_bytes 1D double", "[copy_bytes][types][double]")
{
  using namespace cute;
  constexpr int N = 1024;
  thrust::host_vector<double> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = static_cast<double>(i) * 0.5;
  }
  test_impl(data, make_layout(make_shape(N), make_stride(1)));
}

// tensorA: (2048):(1)
// tensorB: (2048):(1)
TEST_CASE("copy_bytes 1D short", "[copy_bytes][types][short]")
{
  using namespace cute;
  constexpr int N = 2048;
  thrust::host_vector<short> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = static_cast<short>(i % 1000);
  }
  test_impl(data, make_layout(make_shape(N), make_stride(1)));
}

// tensorA: (4096):(1)
// tensorB: (4096):(1)
TEST_CASE("copy_bytes 1D char", "[copy_bytes][types][char]")
{
  using namespace cute;
  constexpr int N = 4096;
  thrust::host_vector<char> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = static_cast<char>(i % 128);
  }
  test_impl(data, make_layout(make_shape(N), make_stride(1)));
}

/***********************************************************************************************************************
 * Large / Multi-Block Tests
 **********************************************************************************************************************/

// tensorA: (100000):(1)
// tensorB: (100000):(1)
TEST_CASE("copy_bytes 1D large", "[copy_bytes][1d][large]")
{
  using namespace cute;
  constexpr int N = 100000;
  thrust::host_vector<float> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = static_cast<float>(i);
  }
  test_impl(data, make_layout(make_shape(N), make_stride(1)));
}

// tensorA: (13,17):(17,1)
// tensorB: (13,17):(1,13)
TEST_CASE("copy_bytes 2D transposition non-tile-divisible", "[copy_bytes][2d][boundary]")
{
  using namespace cute;
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
      expected[c * M + r] = data[r * N + c];
    }
  }
  auto shape = make_shape(M, N);
  test_impl(data, expected, make_layout(shape, make_stride(N, 1)), make_layout(shape, make_stride(1, M)));
}

// tensorA: (100,200):(200,1)
// tensorB: (100,200):(1,100)
TEST_CASE("copy_bytes 2D large transposition", "[copy_bytes][2d][large][transpose]")
{
  using namespace cute;
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
      expected[c * M + r] = data[r * N + c];
    }
  }
  auto shape = make_shape(M, N);
  test_impl(data, expected, make_layout(shape, make_stride(N, 1)), make_layout(shape, make_stride(1, M)));
}
