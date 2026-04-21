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
#include <cuda/std/linalg>

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

// src: int   (8):(1)
// dst: float (8):(1)
// __to_raw_tensor removes singleton dims, so we use N > 1 to avoid rank-0 tensors.
// __are_byte_copyable is false when types differ → bypasses memcpy fast path.
// __have_default_accessors is true, tile_size == tensor_size → path (1) DeviceTransform
TEST_CASE("copy d2d different types", "[copy][d2d][1d][mixed_types]")
{
  namespace cudax = cuda::experimental;
  constexpr int N = 8;
  thrust::host_vector<int> h_src(N);
  for (int i = 0; i < N; ++i)
  {
    h_src[i] = i * 10;
  }
  thrust::device_vector<int> d_src(h_src.begin(), h_src.end());
  thrust::device_vector<float> d_dst(N, 0.0f);

  using extents_t = cuda::std::dextents<int, 1>;
  extents_t ext(N);
  layout_right::mapping<extents_t> mapping(ext);

  cuda::device_mdspan<int, extents_t, layout_right> src(thrust::raw_pointer_cast(d_src.data()), mapping);
  cuda::device_mdspan<float, extents_t, layout_right> dst(thrust::raw_pointer_cast(d_dst.data()), mapping);

  cudax::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<float> h_expected(N);
  for (int i = 0; i < N; ++i)
  {
    h_expected[i] = static_cast<float>(i * 10);
  }
  thrust::host_vector<float> result(d_dst);
  CUDAX_REQUIRE(result == h_expected);
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
 * Non-default accessor (scaled_accessor: scales element values by 2 on read)
 **********************************************************************************************************************/

// src: (128):(1) with scaled_accessor (scale factor = 2)
// dst: (128):(1) with default_accessor
// Bypasses the DeviceTransform contiguous path since __have_default_accessors is false
TEST_CASE("copy d2d contiguous scaled_accessor", "[copy][d2d][1d][accessor]")
{
  namespace cudax = cuda::experimental;
  constexpr int N = 128;

  auto h_src = make_iota<int>(N);
  thrust::device_vector<int> d_src(h_src.begin(), h_src.end());
  thrust::device_vector<int> d_dst(N, 0);

  using extents_t    = cuda::std::dextents<int, 1>;
  using scaled_acc_t = cuda::std::linalg::scaled_accessor<int, cuda::std::default_accessor<int>>;
  using dev_acc_t    = cuda::device_accessor<scaled_acc_t>;
  using src_mdspan_t = cuda::device_mdspan<const int, extents_t, layout_right, scaled_acc_t>;
  using dst_mdspan_t = cuda::device_mdspan<int, extents_t, layout_right>;
  extents_t ext(N);
  layout_right::mapping<extents_t> mapping(ext);

  src_mdspan_t src(
    thrust::raw_pointer_cast(d_src.data()), mapping, dev_acc_t{scaled_acc_t{2, cuda::std::default_accessor<int>{}}});
  dst_mdspan_t dst(thrust::raw_pointer_cast(d_dst.data()), mapping);

  cudax::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<int> h_expected(N);
  for (int i = 0; i < N; ++i)
  {
    h_expected[i] = i * 2;
  }
  thrust::host_vector<int> result(d_dst);
  CUDAX_REQUIRE(result == h_expected);
}

/***********************************************************************************************************************
 * Large element type (64 bytes)
 **********************************************************************************************************************/

struct alignas(64) large_type_64
{
  cuda::std::array<char, 64> data;

  friend bool operator==(const large_type_64& __lhs, const large_type_64& __rhs)
  {
    return __lhs.data == __rhs.data;
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
 * Large element type (128 bytes), non-vectorizable contiguous kernel
 **********************************************************************************************************************/

struct alignas(64) large_type_128
{
  cuda::std::array<char, 128> data;

  friend bool operator==(const large_type_128& __lhs, const large_type_128& __rhs)
  {
    return __lhs.data == __rhs.data;
  }
};

// src: (4,512):(512,1), layout_right, sizeof(T) = 128
// dst: (4,512):(512,1), layout_right, sizeof(T) = 128
// sizeof(T) > __max_vector_access → __are_vectorizable_copy is false
// inner_extent_bytes = 512 * 128 = 64KB >= __bytes_in_flight → path (2b)
// 2D with outer dim > 1 → tile_size != tensor_size → bypasses DeviceTransform path (1)
TEST_CASE("copy d2d large element 128 bytes 2D contiguous", "[copy][d2d][large_element]")
{
  constexpr int M = 4;
  constexpr int N = 512;
  thrust::host_vector<large_type_128> data(M * N);
  for (int i = 0; i < M * N; ++i)
  {
    for (int j = 0; j < 128; ++j)
    {
      data[i].data[j] = static_cast<char>((i * 128 + j) % 128);
    }
  }
  test_copy<layout_right>(data, M, N);
}

// src: (2,513):(1024,1), layout_stride, sizeof(T) = 128
// dst: (2,513):(1024,1), layout_stride
// Padded outer stride (1024 > 513) prevents coalescing → tile_size = 513 != tensor_size = 1026
// inner_extent_bytes = 513 * 128 = 65664 >= __bytes_in_flight → path (2b)
// __are_vectorizable_copy is false (alignof(128) > __max_vector_access)
// _TileSize = 256, inner_size = 513 = 2 * 256 + 1 → last tile has 1 remaining element
TEST_CASE("copy d2d contiguous kernel remainder", "[copy][d2d][contiguous][remainder]")
{
  constexpr int M  = 2;
  constexpr int N  = 513;
  constexpr int Ld = 1024;

  cuda::std::array<int, 2> shape{M, N};
  cuda::std::array<int, 2> strides{Ld, 1};

  using extents_t     = cuda::std::dextents<int, 2>;
  using mapping_t     = cuda::std::layout_stride::mapping<extents_t>;
  const int span_size = static_cast<int>(mapping_t(extents_t(shape), strides).required_span_size());

  thrust::host_vector<large_type_128> h_src(span_size);
  for (int i = 0; i < span_size; ++i)
  {
    for (int j = 0; j < 128; ++j)
    {
      h_src[i].data[j] = static_cast<char>((i * 128 + j) % 128);
    }
  }

  large_type_128 zero{};
  thrust::host_vector<large_type_128> h_expected(span_size, zero);
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      h_expected[i * Ld + j] = h_src[i * Ld + j];
    }
  }

  test_copy_strided(h_src, h_expected, shape, strides, strides);
}

/***********************************************************************************************************************
 * Overaligned type (alignof > natural alignment, vectorizable via alignment-based check)
 **********************************************************************************************************************/

struct alignas(16) overaligned_int
{
  int value;

  bool operator==(const overaligned_int& other) const
  {
    return value == other.value;
  }
};

static_assert(sizeof(overaligned_int) == 16);
static_assert(alignof(overaligned_int) == 16);
static_assert(alignof(overaligned_int) >= sizeof(overaligned_int));

// src: (2,4096):(4096,1), layout_right, sizeof(T) = 16, alignof(T) = 16
// dst: (2,4096):(4096,1), layout_right
// alignof(T) <= __max_vector_access → __are_vectorizable_copy is true
// inner_extent_bytes = 4096 * 16 = 64KB >= __bytes_in_flight → path (2a)
// 2D with outer dim > 1 → tile_size != tensor_size → bypasses DeviceTransform path (1)
TEST_CASE("copy d2d overaligned type vectorized", "[copy][d2d][overaligned]")
{
  constexpr int M     = 2;
  constexpr int N     = 4096;
  constexpr int total = M * N;
  thrust::host_vector<overaligned_int> data(total);
  for (int i = 0; i < total; ++i)
  {
    data[i].value = i;
  }
  test_copy<layout_right>(data, M, N);
}

/***********************************************************************************************************************
 * Tile Boundary
 **********************************************************************************************************************/

// src: (1024):(1)
// dst: (1024):(1)
TEST_CASE("copy d2d tile boundary exact", "[copy][d2d][tile_boundary]")
{
  constexpr int N = 1024;
  test_copy<layout_right>(make_iota<int>(N), N);
}

// src: (1020):(1)
// dst: (1020):(1)
TEST_CASE("copy d2d tile boundary sub-tile", "[copy][d2d][tile_boundary]")
{
  constexpr int N = 1020;
  test_copy<layout_right>(make_iota<int>(N), N);
}

// src: (1028):(1)
// dst: (1028):(1)
TEST_CASE("copy d2d tile boundary partial", "[copy][d2d][tile_boundary]")
{
  constexpr int N = 1028;
  test_copy<layout_right>(make_iota<int>(N), N);
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

  auto h_data = make_iota<float>(M * N);
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

  auto h_data = make_iota<float>(M * N);
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
