//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/stream>

#include <cuda/experimental/copy_bytes.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/layout.hpp>

#include <testing.cuh>

C2H_TEST("copy_bytes 1D contiguous", "[copy_bytes]")
{
  using namespace cute;
  using T = float;

  constexpr int n = 128;
  thrust::host_vector<T> h_src(n);
  for (int i = 0; i < n; ++i)
  {
    h_src[i] = static_cast<T>(i);
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n, T{0});

  cuda::stream stream{cuda::device_ref{0}};

  auto layout = make_layout(make_shape(n));
  cudax::copy_bytes(
    stream, thrust::raw_pointer_cast(d_dst.data()), layout, thrust::raw_pointer_cast(d_src.data()), layout);
  stream.sync();

  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_src);
}

C2H_TEST("copy_bytes 2D same layout", "[copy_bytes]")
{
  using namespace cute;
  using T = int;

  constexpr int rows = 16;
  constexpr int cols = 32;
  constexpr int n    = rows * cols;

  thrust::host_vector<T> h_src(n);
  for (int i = 0; i < n; ++i)
  {
    h_src[i] = i;
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n, 0);

  cuda::stream stream{cuda::device_ref{0}};

  // Row-major layout: stride = (cols, 1)
  auto layout = make_layout(make_shape(rows, cols), make_stride(cols, 1));
  cudax::copy_bytes(
    stream, thrust::raw_pointer_cast(d_dst.data()), layout, thrust::raw_pointer_cast(d_src.data()), layout);
  stream.sync();

  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_src);
}

C2H_TEST("copy_bytes 2D layout transposition", "[copy_bytes]")
{
  using namespace cute;
  using T = int;

  constexpr int rows = 8;
  constexpr int cols = 16;
  constexpr int n    = rows * cols;

  // Source: row-major (rows x cols), stride = (cols, 1)
  // Destination: column-major (rows x cols), stride = (1, rows)
  // Element (r, c) in source at offset r*cols + c
  // Element (r, c) in destination at offset c*rows + r

  thrust::host_vector<T> h_src(n);
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      h_src[r * cols + c] = r * 100 + c;
    }
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n, 0);

  cuda::stream stream{cuda::device_ref{0}};

  auto shape      = make_shape(rows, cols);
  auto src_layout = make_layout(shape, make_stride(cols, 1));  // row-major
  auto dst_layout = make_layout(shape, make_stride(1, rows));  // column-major

  cudax::copy_bytes(
    stream, thrust::raw_pointer_cast(d_dst.data()), dst_layout, thrust::raw_pointer_cast(d_src.data()), src_layout);
  stream.sync();

  // Build expected column-major output: element (r, c) at offset c*rows + r
  thrust::host_vector<T> h_expected(n);
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      h_expected[c * rows + r] = r * 100 + c;
    }
  }

  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_expected);
}

C2H_TEST("copy_bytes 3D dynamic shapes", "[copy_bytes]")
{
  using namespace cute;
  using T = float;

  int d0 = 4;
  int d1 = 6;
  int d2 = 8;
  int n  = d0 * d1 * d2;

  thrust::host_vector<T> h_src(n);
  for (int i = 0; i < n; ++i)
  {
    h_src[i] = static_cast<T>(i);
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n, T{0});

  cuda::stream stream{cuda::device_ref{0}};

  // Contiguous row-major layout for 3D: stride = (d1*d2, d2, 1)
  auto layout = make_layout(make_shape(d0, d1, d2), make_stride(d1 * d2, d2, 1));
  cudax::copy_bytes(
    stream, thrust::raw_pointer_cast(d_dst.data()), layout, thrust::raw_pointer_cast(d_src.data()), layout);
  stream.sync();

  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_src);
}

// ------------------------------------------------------------------
// Large-size tests (exercise vectorization and multi-block dispatch)
// ------------------------------------------------------------------

C2H_TEST("copy_bytes 1D large contiguous", "[copy_bytes]")
{
  using namespace cute;
  using T = float;

  constexpr int n = 100000;
  thrust::host_vector<T> h_src(n);
  for (int i = 0; i < n; ++i)
  {
    h_src[i] = static_cast<T>(i);
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n, T{0});

  cuda::stream stream{cuda::device_ref{0}};

  auto layout = make_layout(make_shape(n));
  cudax::copy_bytes(
    stream, thrust::raw_pointer_cast(d_dst.data()), layout, thrust::raw_pointer_cast(d_src.data()), layout);
  stream.sync();

  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_src);
}

// ------------------------------------------------------------------
// Non-tile-divisible sizes for diff-layout (boundary tile handling)
// ------------------------------------------------------------------

C2H_TEST("copy_bytes 2D transposition non-tile-divisible", "[copy_bytes]")
{
  using namespace cute;
  using T = float;

  // 13x17: not a multiple of tile size (32x32)
  constexpr int rows = 13;
  constexpr int cols = 17;
  constexpr int n    = rows * cols;

  thrust::host_vector<T> h_src(n);
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      h_src[r * cols + c] = static_cast<T>(r * 100 + c);
    }
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n, T{0});

  cuda::stream stream{cuda::device_ref{0}};

  auto shape      = make_shape(rows, cols);
  auto src_layout = make_layout(shape, make_stride(cols, 1));  // row-major
  auto dst_layout = make_layout(shape, make_stride(1, rows));  // column-major

  cudax::copy_bytes(
    stream, thrust::raw_pointer_cast(d_dst.data()), dst_layout, thrust::raw_pointer_cast(d_src.data()), src_layout);
  stream.sync();

  thrust::host_vector<T> h_expected(n);
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      h_expected[c * rows + r] = static_cast<T>(r * 100 + c);
    }
  }

  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_expected);
}

C2H_TEST("copy_bytes 2D large transposition", "[copy_bytes]")
{
  using namespace cute;
  using T = int;

  constexpr int rows = 100;
  constexpr int cols = 200;
  constexpr int n    = rows * cols;

  thrust::host_vector<T> h_src(n);
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      h_src[r * cols + c] = r * 1000 + c;
    }
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n, 0);

  cuda::stream stream{cuda::device_ref{0}};

  auto shape      = make_shape(rows, cols);
  auto src_layout = make_layout(shape, make_stride(cols, 1));
  auto dst_layout = make_layout(shape, make_stride(1, rows));

  cudax::copy_bytes(
    stream, thrust::raw_pointer_cast(d_dst.data()), dst_layout, thrust::raw_pointer_cast(d_src.data()), src_layout);
  stream.sync();

  thrust::host_vector<T> h_expected(n);
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      h_expected[c * rows + r] = r * 1000 + c;
    }
  }

  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_expected);
}

// ------------------------------------------------------------------
// Different element types
// ------------------------------------------------------------------

C2H_TEST("copy_bytes 1D contiguous double", "[copy_bytes]")
{
  using namespace cute;
  using T = double;

  constexpr int n = 1024;
  thrust::host_vector<T> h_src(n);
  for (int i = 0; i < n; ++i)
  {
    h_src[i] = static_cast<T>(i) * 0.5;
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n, T{0});

  cuda::stream stream{cuda::device_ref{0}};

  auto layout = make_layout(make_shape(n));
  cudax::copy_bytes(
    stream, thrust::raw_pointer_cast(d_dst.data()), layout, thrust::raw_pointer_cast(d_src.data()), layout);
  stream.sync();

  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_src);
}

C2H_TEST("copy_bytes 1D contiguous short", "[copy_bytes]")
{
  using namespace cute;
  using T = short;

  constexpr int n = 2048;
  thrust::host_vector<T> h_src(n);
  for (int i = 0; i < n; ++i)
  {
    h_src[i] = static_cast<T>(i % 1000);
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(n, T{0});

  cuda::stream stream{cuda::device_ref{0}};

  auto layout = make_layout(make_shape(n));
  cudax::copy_bytes(
    stream, thrust::raw_pointer_cast(d_dst.data()), layout, thrust::raw_pointer_cast(d_src.data()), layout);
  stream.sync();

  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == h_src);
}

// ------------------------------------------------------------------
// 2D same layout with non-contiguous outer stride (strided)
// ------------------------------------------------------------------

C2H_TEST("copy_bytes 2D same layout strided", "[copy_bytes]")
{
  using namespace cute;
  using T = float;

  // 16 rows x 32 cols, but outer stride = 64 (gap of 32 elements per row)
  constexpr int rows         = 16;
  constexpr int cols         = 32;
  constexpr int outer_stride = 64;
  constexpr int alloc_size   = rows * outer_stride;

  thrust::host_vector<T> h_src(alloc_size, T{-1});
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      h_src[r * outer_stride + c] = static_cast<T>(r * 100 + c);
    }
  }

  thrust::device_vector<T> d_src = h_src;
  thrust::device_vector<T> d_dst(alloc_size, T{0});

  cuda::stream stream{cuda::device_ref{0}};

  auto layout = make_layout(make_shape(rows, cols), make_stride(outer_stride, 1));
  cudax::copy_bytes(
    stream, thrust::raw_pointer_cast(d_dst.data()), layout, thrust::raw_pointer_cast(d_src.data()), layout);
  stream.sync();

  // Verify only the strided locations
  thrust::host_vector<T> h_dst = d_dst;
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      CUDAX_REQUIRE(h_dst[r * outer_stride + c] == static_cast<T>(r * 100 + c));
    }
  }
}
