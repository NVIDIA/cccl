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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/launch>

#include <cute/numeric/int.hpp>
#include <cute/tensor.hpp>

#include <cuda/experimental/__copy_bytes/layout_utils.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief Tiled same-layout copy kernel using cooperative_copy.
//!
//! Each block processes one tile of TileSize vectorized elements.
//! Full tiles use cooperative_copy with static shapes for optimal thread
//! partitioning. The last (boundary) tile uses a manual bounds-checked loop.
//!
//! This kernel is used for rank-1 tensors (fully contiguous after coalescing
//! and recast to VecType).
//!
//! @tparam Config   Kernel launch configuration type (encodes block dimensions)
//! @tparam SrcTensor CuTe tensor type for source (rank-1, recast to VecType)
//! @tparam DstTensor CuTe tensor type for destination (rank-1, recast to VecType)
//! @tparam TileSize Number of VecType elements per block (compile-time constant)
template <typename Config, typename SrcTensor, typename DstTensor, int TileSize>
__global__ void copy_same_layout_tiled_kernel(Config config, SrcTensor src, DstTensor dst, int n)
{
  constexpr int NumThreads = __config_block_threads<Config>;

  uint32_t thread_idx = ::cuda::gpu_thread.rank(::cuda::block, config);
  int block_idx       = ::cuda::block.rank(::cuda::grid, config);
  int offset          = block_idx * TileSize;
  int remaining       = n - offset;

  if (remaining >= TileSize)
  {
    auto tile_layout = ::cute::make_layout(::cute::Int<TileSize>{}, ::cute::Int<1>{});
    auto src_tile    = ::cute::make_tensor(::cute::make_gmem_ptr(&src(offset)), tile_layout);
    auto dst_tile    = ::cute::make_tensor(::cute::make_gmem_ptr(&dst(offset)), tile_layout);

    ::cute::cooperative_copy<NumThreads, ::cute::sizeof_bits_v<typename SrcTensor::value_type>>(
      thread_idx, src_tile, dst_tile);
  }
  else
  {
    for (int i = thread_idx; i < remaining; i += NumThreads)
    {
      dst(offset + i) = src(offset + i);
    }
  }
}

//! @brief Grid-stride same-layout copy kernel (fallback for multi-rank tensors).
//!
//! When the recast tensor has rank > 1, tiling into 1D static-shaped tiles is
//! not possible because the memory is non-contiguous. Falls back to a simple
//! grid-stride loop over the linear index space.
//!
//! @tparam Config   Kernel launch configuration type
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
//! width. For rank-1 tensors (fully contiguous), uses a tiled cooperative_copy
//! kernel for optimal thread partitioning. For multi-rank tensors, falls back
//! to a grid-stride loop.
//!
//! @tparam VecBytes Vectorization width in bytes (compile-time template parameter)
//! @tparam T Element type of the original tensors
//! @tparam Layout CuTe layout type (same for src and dst)
template <int VecBytes, typename T, typename Layout>
void launch_copy_same_layout(::cuda::stream_ref stream, T* dst_ptr, const T* src_ptr, const Layout& layout)
{
  using VecType            = ::cute::uint_bit_t<VecBytes * 8>;
  auto src_tensor          = ::cute::recast<VecType>(::cute::make_tensor(::cute::make_gmem_ptr(src_ptr), layout));
  auto dst_tensor          = ::cute::recast<VecType>(::cute::make_tensor(::cute::make_gmem_ptr(dst_ptr), layout));
  int n                    = ::cute::size(src_tensor);
  constexpr int block_size = 256;
  constexpr int Rank       = decltype(::cute::rank(src_tensor))::value;

  if constexpr (Rank == 1)
  {
    constexpr int TileSize = block_size * 4;
    int grid_size          = ::cuda::ceil_div(n, TileSize);
    auto config            = ::cuda::make_config(::cuda::block_dims<block_size>(), ::cuda::grid_dims(grid_size));
    const auto& kernel =
      copy_same_layout_tiled_kernel<decltype(config), decltype(src_tensor), decltype(dst_tensor), TileSize>;
    ::cuda::launch(stream, config, kernel, src_tensor, dst_tensor, n);
  }
  else
  {
    int grid_size   = ::cuda::ceil_div(n, block_size);
    auto config     = ::cuda::make_config(::cuda::block_dims<block_size>(), ::cuda::grid_dims(grid_size));
    const auto& kernel = copy_same_layout_kernel<decltype(config), decltype(src_tensor), decltype(dst_tensor)>;
    ::cuda::launch(stream, config, kernel, src_tensor, dst_tensor, n);
  }
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_SAME_LAYOUT
