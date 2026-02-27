//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <climits>
#include <stdexcept>

#include "copy_bytes_common.cuh"
#include <cute/layout.hpp>

/***********************************************************************************************************************
 * Edge Cases
 **********************************************************************************************************************/

TEST_CASE("copy_bytes scalar", "[copy_bytes][0d]")
{
  using namespace cute;
  thrust::host_vector<int> data(1, 42);
  test_impl(data, make_layout(make_shape(1)));
}

TEST_CASE("copy_bytes size 0", "[copy_bytes][zero_size]")
{
  using namespace cute;
  thrust::host_vector<int> data(1, 0);
  thrust::host_vector<int> expected(1, 0);
  auto layout = make_layout(make_shape(0, 0));
  test_impl(data, expected, layout, layout);
}

/***********************************************************************************************************************
 * Tile Boundary
 **********************************************************************************************************************/

TEST_CASE("copy_bytes tile boundary exact", "[copy_bytes][tile_boundary]")
{
  using namespace cute;
  constexpr int N = 1024;
  thrust::host_vector<int> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = i;
  }
  test_impl(data, make_layout(make_shape(N), make_stride(1)));
}

TEST_CASE("copy_bytes tile boundary sub-tile", "[copy_bytes][tile_boundary]")
{
  using namespace cute;
  constexpr int N = 1020; // fallback loop only
  thrust::host_vector<int> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = i;
  }
  test_impl(data, make_layout(make_shape(N), make_stride(1)));
}

TEST_CASE("copy_bytes tile boundary partial", "[copy_bytes][tile_boundary]")
{
  using namespace cute;
  constexpr int N = 1028; // block 1 fallback (1)
  thrust::host_vector<int> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = i;
  }
  test_impl(data, make_layout(make_shape(N), make_stride(1)));
}

/***********************************************************************************************************************
 * Negative: mismatched shapes
 **********************************************************************************************************************/

TEST_CASE("copy_bytes mismatched shapes", "[copy_bytes][negative]")
{
  using namespace cute;
  namespace cudax = cuda::experimental;
  constexpr int N = 64;
  thrust::device_vector<float> d_src(N);
  thrust::device_vector<float> d_dst(N);
  auto* src_ptr   = thrust::raw_pointer_cast(d_src.data());
  auto* dst_ptr   = thrust::raw_pointer_cast(d_dst.data());
  auto src_layout = make_layout(make_shape(8, 8), make_stride(1, 8));
  auto dst_layout = make_layout(make_shape(4, 16), make_stride(1, 4));
  CHECK_THROWS_AS(cudax::copy_bytes_registers(src_ptr, src_layout, dst_ptr, dst_layout, stream), std::invalid_argument);
}

/***********************************************************************************************************************
 * Misaligned pointer + vectorization
 **********************************************************************************************************************/

TEST_CASE("copy_bytes misaligned pointer", "[copy_bytes][alignment]")
{
  using namespace cute;
  namespace cudax = cuda::experimental;
  constexpr int N = 512;

  thrust::host_vector<char> h_src(N);
  for (int i = 0; i < N; ++i)
  {
    h_src[i] = static_cast<char>(i % 128);
  }

  // Pad by 1 element so that ptr+1 is misaligned (1-byte aligned instead of 256)
  thrust::host_vector<char> h_src_padded(N + 1, char{0});
  thrust::copy(h_src.begin(), h_src.end(), h_src_padded.begin() + 1);

  thrust::device_vector<char> d_src_buf = h_src_padded;
  thrust::device_vector<char> d_dst_buf(N + 1, char{0});

  auto* src_ptr = thrust::raw_pointer_cast(d_src_buf.data()) + 1;
  auto* dst_ptr = thrust::raw_pointer_cast(d_dst_buf.data()) + 1;
  auto layout   = make_layout(make_shape(N), make_stride(1));

  auto check = [&] {
    thrust::host_vector<char> h_dst_buf(d_dst_buf);
    thrust::host_vector<char> result(h_dst_buf.begin() + 1, h_dst_buf.begin() + 1 + N);
    CUDAX_REQUIRE(result == h_src);
  };

  thrust::fill(d_dst_buf.begin(), d_dst_buf.end(), char{0});
  cudax::copy_bytes_naive(src_ptr, layout, dst_ptr, layout, stream);
  stream.sync();
  check();

  thrust::fill(d_dst_buf.begin(), d_dst_buf.end(), char{0});
  cudax::copy_bytes_registers(src_ptr, layout, dst_ptr, layout, stream);
  stream.sync();
  check();
}

/***********************************************************************************************************************
 * Large count > INT_MAX
 **********************************************************************************************************************/

TEST_CASE("copy_bytes large count > INT_MAX", "[copy_bytes][large][.]")
{
  using namespace cute;
  namespace cudax = cuda::experimental;

  const auto N        = static_cast<size_t>(INT_MAX) + 257;
  const auto required = 2 * N;

  size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  if (free_mem < required + (size_t{256} << 20))
  {
    SKIP("Not enough GPU memory (" << (free_mem >> 20) << " MB free, need ~" << (required >> 20) << " MB)");
  }

  thrust::device_vector<char> d_src(N, static_cast<char>(0x42));
  thrust::device_vector<char> d_dst(N, static_cast<char>(0x00));

  auto* src_ptr = thrust::raw_pointer_cast(d_src.data());
  auto* dst_ptr = thrust::raw_pointer_cast(d_dst.data());

  auto layout = make_layout(make_shape(static_cast<long long>(N)), make_stride(1));
  cudax::copy_bytes_registers(src_ptr, layout, dst_ptr, layout, stream);
  stream.sync();

  CUDAX_REQUIRE(d_dst[0] == static_cast<char>(0x42));
  CUDAX_REQUIRE(d_dst[N / 2] == static_cast<char>(0x42));
  CUDAX_REQUIRE(d_dst[N - 1] == static_cast<char>(0x42));
}
