//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_COPY_BYTES_COMMON_CUH
#define CUDAX_TEST_COPY_BYTES_COMMON_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/stream>

#include <cuda/experimental/__copy_bytes/copy_bytes_naive.cuh>
#include <cuda/experimental/__copy_bytes/copy_bytes_registers.cuh>

#include "testing.cuh"
#include <cute/layout.hpp>

static const cuda::stream stream{cuda::device_ref{0}};

template <typename T, typename SrcLayout, typename DstLayout>
void test_impl(const thrust::host_vector<T>& input,
               const thrust::host_vector<T>& expected,
               const SrcLayout& src_layout,
               const DstLayout& dst_layout,
               int src_offset = 0,
               int dst_offset = 0)
{
  namespace cudax                = cuda::experimental;
  thrust::device_vector<T> d_src = input;
  thrust::device_vector<T> d_dst(expected.size(), T{0});
  auto* src_ptr = thrust::raw_pointer_cast(d_src.data()) + src_offset;
  auto* dst_ptr = thrust::raw_pointer_cast(d_dst.data()) + dst_offset;

  d_dst.assign(expected.size(), T{0});
  cudax::copy_bytes_naive(src_ptr, src_layout, dst_ptr, dst_layout, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == expected);

  d_dst.assign(expected.size(), T{0});
  cudax::copy_bytes_registers(src_ptr, src_layout, dst_ptr, dst_layout, stream);
  stream.sync();
  CUDAX_REQUIRE(thrust::host_vector<T>(d_dst) == expected);
}

template <typename T, typename Layout>
void test_impl(const thrust::host_vector<T>& input, const Layout& layout)
{
  test_impl(input, input, layout, layout);
}

template <typename T, typename Layout>
void test_impl(int alloc_size, int offset, const Layout& layout)
{
  thrust::host_vector<T> h_src(alloc_size);
  for (int i = 0; i < alloc_size; ++i)
  {
    h_src[i] = static_cast<T>(i);
  }
  test_impl(h_src, h_src, layout, layout, offset, offset);
}

template <typename T, typename SrcLayout, typename DstLayout>
void test_impl(
  int src_alloc, int src_offset, const SrcLayout& src_layout, int dst_alloc, int dst_offset, const DstLayout& dst_layout)
{
  thrust::host_vector<T> h_src(src_alloc);
  for (int i = 0; i < src_alloc; ++i)
  {
    h_src[i] = static_cast<T>(i);
  }
  thrust::host_vector<T> expected(dst_alloc, T{0});
  const int num_items = static_cast<int>(cute::size(src_layout));
  for (int i = 0; i < num_items; ++i)
  {
    expected[dst_offset + dst_layout(i)] = h_src[src_offset + src_layout(i)];
  }
  test_impl(h_src, expected, src_layout, dst_layout, src_offset, dst_offset);
}

#endif // CUDAX_TEST_COPY_BYTES_COMMON_CUH
