//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_COPY_COMMON_CUH
#define CUDAX_TEST_COPY_COMMON_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/stream>

#include <cuda/experimental/copy.cuh>

#include "testing.cuh"

using cuda::std::layout_left;
using cuda::std::layout_right;

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

template <typename T, typename... Ints>
void test_copy_iota(Ints... shape)
{
  const int total = (static_cast<int>(shape) * ...);
  thrust::host_vector<T> data(total);
  for (int i = 0; i < total; ++i)
  {
    data[i] = static_cast<T>(i);
  }
  test_copy<layout_right>(data, shape...);
}

template <typename T, size_t Rank>
void test_copy_offset(
  int src_alloc,
  int src_offset,
  const cuda::std::array<int, Rank>& shape,
  const cuda::std::array<int, Rank>& src_strides,
  int dst_alloc,
  int dst_offset,
  const cuda::std::array<int, Rank>& dst_strides)
{
  thrust::host_vector<T> h_src(src_alloc);
  for (int i = 0; i < src_alloc; ++i)
  {
    h_src[i] = static_cast<T>(i);
  }
  int total = 1;
  for (size_t d = 0; d < Rank; ++d)
  {
    total *= shape[d];
  }
  thrust::host_vector<T> expected(dst_alloc, T{0});
  for (int flat = 0; flat < total; ++flat)
  {
    cuda::std::array<int, Rank> idx{};
    int tmp = flat;
    for (int d = static_cast<int>(Rank) - 1; d >= 0; --d)
    {
      idx[d] = tmp % shape[d];
      tmp /= shape[d];
    }
    int src_linear = src_offset;
    int dst_linear = dst_offset;
    for (size_t d = 0; d < Rank; ++d)
    {
      src_linear += idx[d] * src_strides[d];
      dst_linear += idx[d] * dst_strides[d];
    }
    expected[dst_linear] = h_src[src_linear];
  }

  thrust::device_vector<T> d_src(h_src.begin(), h_src.end());
  thrust::device_vector<T> d_dst(dst_alloc, T{0});

  using extents_t = cuda::std::dextents<int, Rank>;
  using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;
  extents_t ext(shape);
  auto* src_ptr = thrust::raw_pointer_cast(d_src.data()) + src_offset;
  auto* dst_ptr = thrust::raw_pointer_cast(d_dst.data()) + dst_offset;
  mapping_t src_map(ext, cuda::dstrides<int, Rank>(src_strides));
  mapping_t dst_map(ext, cuda::dstrides<int, Rank>(dst_strides));

  cuda::device_mdspan<T, extents_t, cuda::layout_stride_relaxed> src(src_ptr, src_map);
  cuda::device_mdspan<T, extents_t, cuda::layout_stride_relaxed> dst(dst_ptr, dst_map);

  cuda::experimental::copy(src, dst, stream);
  stream.sync();

  thrust::host_vector<T> result(d_dst);
  CUDAX_REQUIRE(result == expected);
}

template <typename T, size_t Rank>
void test_copy_offset(
  int alloc, int offset, const cuda::std::array<int, Rank>& shape, const cuda::std::array<int, Rank>& strides)
{
  test_copy_offset<T>(alloc, offset, shape, strides, alloc, offset, strides);
}

#endif // CUDAX_TEST_COPY_COMMON_CUH
