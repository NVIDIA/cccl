//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_COPY_SAME_LAYOUT
#define _CUDAX__COPY_BYTES_COPY_SAME_LAYOUT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cute/numeric/int.hpp>
#include <cute/tensor.hpp>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/launch>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief Vectorized same-layout copy kernel using CuTe recast.
//!
//! When source and destination have the same layout, shared memory adds no benefit.
//! Instead, recast both tensors to a wider VecType for vectorized loads/stores and
//! use a grid-stride loop over the recast elements.
//!
//! @tparam SrcTensor CuTe tensor type for source (already recast to VecType)
//! @tparam DstTensor CuTe tensor type for destination (already recast to VecType)
template <typename Config, typename SrcTensor, typename DstTensor>
__global__ void copy_same_layout_kernel(Config config, SrcTensor src, DstTensor dst, int n)
{
  int idx    = ::cuda::gpu_thread.rank(::cuda::grid, config);
  int stride = ::cuda::gpu_thread.count(::cuda::grid, config);
  for (int i = idx; i < n; i += stride)
  {
    dst(i) = src(i);
  }
}

//! @brief Launch the vectorized same-layout copy kernel.
//!
//! Recasts the tensors to a wider VecType based on the computed vectorization
//! width, then launches a grid-stride kernel over the recast elements.
//!
//! @tparam VecBytes Vectorization width in bytes (compile-time template parameter)
//! @tparam T Element type of the original tensors
//! @tparam Layout CuTe layout type (same for src and dst)
template <int VecBytes, typename T, typename Layout>
void launch_copy_same_layout(::cuda::stream_ref stream, T* dst_ptr, const T* src_ptr, const Layout& layout)
{
  using VecType = ::cute::uint_bit_t<VecBytes * 8>;

  auto src_tensor = ::cute::recast<VecType>(::cute::make_tensor(::cute::make_gmem_ptr(src_ptr), layout));
  auto dst_tensor = ::cute::recast<VecType>(::cute::make_tensor(::cute::make_gmem_ptr(dst_ptr), layout));

  int n                    = ::cute::size(src_tensor);
  constexpr int block_size = 256;
  int grid_size            = ::cuda::ceil_div(n, block_size);

  auto config = ::cuda::make_config(::cuda::block_dims<block_size>(), ::cuda::grid_dims(grid_size));
  ::cuda::launch(
    stream, config, copy_same_layout_kernel<decltype(config), decltype(src_tensor), decltype(dst_tensor)>,
    src_tensor, dst_tensor, n);
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_SAME_LAYOUT
