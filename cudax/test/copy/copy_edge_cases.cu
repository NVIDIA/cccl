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

#include <cuda/std/climits>

#include <stdexcept>

#include "copy_common.cuh"

/***********************************************************************************************************************
 * Edge Cases
 **********************************************************************************************************************/

// src: (1):(1)
// dst: (1):(1)
TEST_CASE("copy d2d scalar", "[copy][d2d][0d]")
{
  thrust::host_vector<int> data(1, 42);
  test_copy<layout_right>(data, 1);
}

// src: (0,0):(0,1)
// dst: (0,0):(0,1)
TEST_CASE("copy d2d size 0", "[copy][d2d][zero_size]")
{
  namespace cudax = cuda::experimental;
  thrust::device_vector<int> d_src(1, 0);
  thrust::device_vector<int> d_dst(1, 0);

  using extents_t = cuda::std::dextents<int, 2>;
  extents_t ext(0, 0);
  cuda::std::layout_right::mapping<extents_t> mapping(ext);

  cuda::device_mdspan<int, extents_t, layout_right> src(thrust::raw_pointer_cast(d_src.data()), mapping);
  cuda::device_mdspan<int, extents_t, layout_right> dst(thrust::raw_pointer_cast(d_dst.data()), mapping);

  cudax::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<int> result(d_dst);
  CUDAX_REQUIRE(result[0] == 0);
}

/***********************************************************************************************************************
 * Large element type (64 bytes)
 **********************************************************************************************************************/

struct alignas(64) large_type_64
{
  char data[64];

  bool operator==(const large_type_64& other) const
  {
    for (int i = 0; i < 64; ++i)
    {
      if (data[i] != other.data[i])
      {
        return false;
      }
    }
    return true;
  }
};

// src: (128):(1)
// dst: (128):(1)
TEST_CASE("copy d2d large element 64 bytes", "[copy][d2d][large_element]")
{
  constexpr int N = 128;
  thrust::host_vector<large_type_64> data(N);
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < 64; ++j)
    {
      data[i].data[j] = static_cast<char>((i * 64 + j) % 128);
    }
  }
  test_copy<layout_right>(data, N);
}

/***********************************************************************************************************************
 * Tile Boundary
 **********************************************************************************************************************/

// src: (1024):(1)
// dst: (1024):(1)
TEST_CASE("copy d2d tile boundary exact", "[copy][d2d][tile_boundary]")
{
  constexpr int N = 1024;
  thrust::host_vector<int> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = i;
  }
  test_copy<layout_right>(data, N);
}

// src: (1020):(1)
// dst: (1020):(1)
TEST_CASE("copy d2d tile boundary sub-tile", "[copy][d2d][tile_boundary]")
{
  constexpr int N = 1020;
  thrust::host_vector<int> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = i;
  }
  test_copy<layout_right>(data, N);
}

// src: (1028):(1)
// dst: (1028):(1)
TEST_CASE("copy d2d tile boundary partial", "[copy][d2d][tile_boundary]")
{
  constexpr int N = 1028;
  thrust::host_vector<int> data(N);
  for (int i = 0; i < N; ++i)
  {
    data[i] = i;
  }
  test_copy<layout_right>(data, N);
}

/***********************************************************************************************************************
 * Negative: mismatched shapes
 **********************************************************************************************************************/

TEST_CASE("copy d2d mismatched shapes", "[copy][d2d][negative]")
{
  namespace cudax = cuda::experimental;
  constexpr int N = 64;
  thrust::device_vector<float> d_src(N);
  thrust::device_vector<float> d_dst(N);

  using extents_src_t = cuda::std::dextents<int, 2>;
  using extents_dst_t = cuda::std::dextents<int, 2>;
  extents_src_t src_ext(8, 8);
  extents_dst_t dst_ext(4, 16);

  cuda::device_mdspan<float, extents_src_t, layout_right> src(
    thrust::raw_pointer_cast(d_src.data()), layout_right::mapping<extents_src_t>(src_ext));
  cuda::device_mdspan<float, extents_dst_t, layout_right> dst(
    thrust::raw_pointer_cast(d_dst.data()), layout_right::mapping<extents_dst_t>(dst_ext));

  CHECK_THROWS_AS(cudax::copy(src, dst, stream), std::invalid_argument);
}

/***********************************************************************************************************************
 * Mismatched extents/strides types between src and dst
 **********************************************************************************************************************/

// src: dextents<int, 2>(4, 8), layout_right
// dst: dextents<long long, 2>(4, 8), layout_right
TEST_CASE("copy d2d different extent types", "[copy][d2d][mixed_types]")
{
  namespace cudax = cuda::experimental;
  constexpr int M = 4;
  constexpr int N = 8;

  thrust::host_vector<float> h_data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    h_data[i] = static_cast<float>(i);
  }
  thrust::device_vector<float> d_src(h_data.begin(), h_data.end());
  thrust::device_vector<float> d_dst(M * N, 0.0f);

  using src_extents_t = cuda::std::dextents<int, 2>;
  using dst_extents_t = cuda::std::dextents<long long, 2>;
  src_extents_t src_ext(M, N);
  dst_extents_t dst_ext(M, N);

  cuda::device_mdspan<float, src_extents_t, layout_right> src(
    thrust::raw_pointer_cast(d_src.data()), layout_right::mapping<src_extents_t>(src_ext));
  cuda::device_mdspan<float, dst_extents_t, layout_right> dst(
    thrust::raw_pointer_cast(d_dst.data()), layout_right::mapping<dst_extents_t>(dst_ext));

  cudax::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<float> result(d_dst);
  CUDAX_REQUIRE(result == h_data);
}

// src: dextents<int, 2>(4, 8), layout_stride with strides array<int, 2>
// dst: dextents<long long, 2>(4, 8), layout_stride with strides array<long long, 2>
TEST_CASE("copy d2d different extent and stride types", "[copy][d2d][mixed_types]")
{
  namespace cudax = cuda::experimental;
  constexpr int M = 4;
  constexpr int N = 8;

  thrust::host_vector<float> h_data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    h_data[i] = static_cast<float>(i);
  }
  thrust::device_vector<float> d_src(h_data.begin(), h_data.end());
  thrust::device_vector<float> d_dst(M * N, 0.0f);

  using src_extents_t = cuda::std::dextents<int, 2>;
  using dst_extents_t = cuda::std::dextents<long long, 2>;
  using src_mapping_t = cuda::std::layout_stride::mapping<src_extents_t>;
  using dst_mapping_t = cuda::std::layout_stride::mapping<dst_extents_t>;

  src_mapping_t src_mapping(src_extents_t(M, N), cuda::std::array<int, 2>{N, 1});
  dst_mapping_t dst_mapping(dst_extents_t(M, N), cuda::std::array<long long, 2>{N, 1});

  cuda::device_mdspan<float, src_extents_t, cuda::std::layout_stride> src(
    thrust::raw_pointer_cast(d_src.data()), src_mapping);
  cuda::device_mdspan<float, dst_extents_t, cuda::std::layout_stride> dst(
    thrust::raw_pointer_cast(d_dst.data()), dst_mapping);

  cudax::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<float> result(d_dst);
  CUDAX_REQUIRE(result == h_data);
}

/***********************************************************************************************************************
 * Misaligned pointer + vectorization
 **********************************************************************************************************************/

TEST_CASE("copy d2d misaligned pointer", "[copy][d2d][alignment]")
{
  namespace cudax = cuda::experimental;
  constexpr int N = 512;

  thrust::host_vector<char> h_src(N);
  for (int i = 0; i < N; ++i)
  {
    h_src[i] = static_cast<char>(i % 128);
  }

  thrust::host_vector<char> h_src_padded(N + 1, char{0});
  thrust::copy(h_src.begin(), h_src.end(), h_src_padded.begin() + 1);

  thrust::device_vector<char> d_src_buf = h_src_padded;
  thrust::device_vector<char> d_dst_buf(N + 1, char{0});

  auto* src_ptr = thrust::raw_pointer_cast(d_src_buf.data()) + 1;
  auto* dst_ptr = thrust::raw_pointer_cast(d_dst_buf.data()) + 1;

  using extents_t = cuda::std::dextents<int, 1>;
  extents_t ext(N);
  layout_right::mapping<extents_t> mapping(ext);

  cuda::device_mdspan<char, extents_t, layout_right> src(src_ptr, mapping);
  cuda::device_mdspan<char, extents_t, layout_right> dst(dst_ptr, mapping);

  cudax::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<char> h_dst_buf(d_dst_buf);
  thrust::host_vector<char> result(h_dst_buf.begin() + 1, h_dst_buf.begin() + 1 + N);
  CUDAX_REQUIRE(result == h_src);
}

/***********************************************************************************************************************
 * Large count > INT_MAX
 **********************************************************************************************************************/

TEST_CASE("copy d2d large count > INT_MAX", "[copy][d2d][large][.]")
{
  namespace cudax = cuda::experimental;

  const auto N        = static_cast<size_t>(INT_MAX) + 257;
  const auto required = 2 * N;

  size_t free_mem  = 0;
  size_t total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  if (free_mem < required + (size_t{256} << 20))
  {
    SKIP("Not enough GPU memory (" << (free_mem >> 20) << " MB free, need ~" << (required >> 20) << " MB)");
  }

  thrust::device_vector<char> d_src(N, static_cast<char>(0x42));
  thrust::device_vector<char> d_dst(N, static_cast<char>(0x00));

  using extents_t = cuda::std::dextents<long long, 1>;
  extents_t ext(static_cast<long long>(N));
  layout_right::mapping<extents_t> mapping(ext);

  cuda::device_mdspan<char, extents_t, layout_right> src(thrust::raw_pointer_cast(d_src.data()), mapping);
  cuda::device_mdspan<char, extents_t, layout_right> dst(thrust::raw_pointer_cast(d_dst.data()), mapping);

  cudax::copy(src, dst, stream);
  stream.sync();

  CUDAX_REQUIRE(d_dst[0] == static_cast<char>(0x42));
  CUDAX_REQUIRE(d_dst[N / 2] == static_cast<char>(0x42));
  CUDAX_REQUIRE(d_dst[N - 1] == static_cast<char>(0x42));
}
