//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_COPY_BYTES_NAIVE
#define _CUDAX__COPY_BYTES_COPY_BYTES_NAIVE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/launch>

#include <cuda/experimental/__copy_bytes/layout_utils.cuh>

#include <cuda/std/__cccl/prologue.h>
#include <cute/tensor.hpp>

namespace cuda::experimental
{
template <typename Config, typename SrcTensor, typename DstTensor>
__global__ void copy_bytes_naive_kernel(Config config, SrcTensor src, DstTensor dst, int num_items)
{
  const auto idx    = ::cuda::gpu_thread.rank(::cuda::grid, config);
  const auto stride = ::cuda::gpu_thread.count(::cuda::grid, config);
  for (auto i = idx; i < num_items; i += stride)
  {
    dst(i) = src(i);
  }
}

//! @brief Launch the naive fallback kernel.
template <typename T, typename SrcLayout, typename DstLayout>
void copy_bytes_naive(
  const T* src, const SrcLayout& src_layout, T* dst, const DstLayout& dst_layout, ::cuda::stream_ref stream)
{
  namespace cudax          = ::cuda::experimental;
  constexpr int block_size = 256;
  const auto num_items     = static_cast<int>(::cute::size(src_layout));
  if (num_items == 0)
  {
    return;
  }
  const auto src_tensor = cudax::make_gmem_tensor(src, src_layout);
  const auto dst_tensor = cudax::make_gmem_tensor(dst, dst_layout);
  const auto grid_size     = ::cuda::ceil_div(num_items, block_size);
  const auto config        = ::cuda::make_config(::cuda::block_dims<block_size>(), ::cuda::grid_dims(grid_size));
  const auto& kernel = cudax::copy_bytes_naive_kernel<decltype(config), decltype(src_tensor), decltype(dst_tensor)>;

  ::cuda::launch(stream, config, kernel, src_tensor, dst_tensor, num_items);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_BYTES_NAIVE
