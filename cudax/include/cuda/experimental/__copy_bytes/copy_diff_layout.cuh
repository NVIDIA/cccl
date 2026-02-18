//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_COPY_DIFF_LAYOUT
#define _CUDAX__COPY_BYTES_COPY_DIFF_LAYOUT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/launch>
#include <cuda/std/__exception/cuda_error.h>

#include <cute/layout.hpp>
#include <cute/numeric/int.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief Compute the product of static extents from a cuda::std::extents type.
template <typename>
struct __extents_product;

template <typename IndexType, ::cuda::std::size_t... Es>
struct __extents_product<::cuda::std::extents<IndexType, Es...>>
{
  static constexpr int value = static_cast<int>((Es * ...));
};

//! @brief Extract the compile-time block thread count from a kernel_config type.
//!
//! Uses the hierarchy type embedded in Config to find the block-level descriptor,
//! then computes the product of its static extents.
template <typename Config>
inline constexpr int __config_block_threads = __extents_product<
  typename Config::hierarchy_type::template level_desc_type<::cuda::block_level>::extents_type>::value;

//! @brief Shared-memory tiled copy kernel for tensors with different src/dst layouts.
//!
//! Uses shared memory to decouple load and store access patterns. Each block
//! processes one (TileM x TileN) tile: loads from source into swizzled shared
//! memory, then stores from shared memory to destination.
//!
//! Full tiles use cooperative_copy for optimal thread partitioning.
//! Boundary tiles (when dimensions are not multiples of tile sizes) use a
//! manual bounds-checked loop.
//!
//! @tparam T Element type
//! @tparam SrcTensor CuTe tensor type for source (rank-2, gmem)
//! @tparam DstTensor CuTe tensor type for destination (rank-2, gmem)
//! @tparam TileM Tile height (compile-time constant)
//! @tparam TileN Tile width (compile-time constant)
template <typename Config, typename T, typename SrcTensor, typename DstTensor, int TileM, int TileN>
__global__ void copy_diff_layout_kernel(Config config, SrcTensor src, DstTensor dst)
{
  // Extract compile-time thread count from Config's hierarchy type
  constexpr int NumThreads = __config_block_threads<Config>;

  uint32_t thread_idx = ::cuda::gpu_thread.rank(::cuda::block, config);
  int block_idx       = ::cuda::block.rank(::cuda::grid, config);

  int M = ::cute::get<0>(::cute::shape(src));
  int N = ::cute::get<1>(::cute::shape(src));

  static __shared__ T smem_buf[TileM * TileN];

  // Swizzled shared memory layout for bank-conflict-free access
  auto smem_base =
    ::cute::make_layout(::cute::make_shape(::cute::Int<TileM>{}, ::cute::Int<TileN>{}),
                        ::cute::make_stride(::cute::Int<TileN>{}, ::cute::Int<1>{}));
  auto smem_layout = ::cute::composition(::cute::Swizzle<3, 3, 3>{}, smem_base);
  auto smem        = ::cute::make_tensor(::cute::make_smem_ptr(smem_buf), smem_layout);

  // 2D tile coordinates from flat block index
  int num_tiles_n = (N + TileN - 1) / TileN;
  int tile_m_idx  = block_idx / num_tiles_n;
  int tile_n_idx  = block_idx % num_tiles_n;

  int m_begin  = tile_m_idx * TileM;
  int n_begin  = tile_n_idx * TileN;
  int actual_m = min(TileM, M - m_begin);
  int actual_n = min(TileN, N - n_begin);

  // Extract strides from tensors
  auto src_stride_m = ::cute::get<0>(::cute::stride(src));
  auto src_stride_n = ::cute::get<1>(::cute::stride(src));
  auto dst_stride_m = ::cute::get<0>(::cute::stride(dst));
  auto dst_stride_n = ::cute::get<1>(::cute::stride(dst));

  bool is_full_tile = (actual_m == TileM && actual_n == TileN);

  if (is_full_tile)
  {
    // Construct tile tensors with STATIC shapes and dynamic strides
    auto src_tile = ::cute::make_tensor(
      ::cute::make_gmem_ptr(&src(m_begin, n_begin)),
      ::cute::make_layout(
        ::cute::make_shape(::cute::Int<TileM>{}, ::cute::Int<TileN>{}),
        ::cute::make_stride(src_stride_m, src_stride_n)));
    auto dst_tile = ::cute::make_tensor(
      ::cute::make_gmem_ptr(&dst(m_begin, n_begin)),
      ::cute::make_layout(
        ::cute::make_shape(::cute::Int<TileM>{}, ::cute::Int<TileN>{}),
        ::cute::make_stride(dst_stride_m, dst_stride_n)));

    // gmem -> smem (source-coalesced access)
    ::cute::cooperative_copy<NumThreads, ::cute::sizeof_bits_v<T>>(thread_idx, src_tile, smem);
    __syncthreads();

    // smem -> gmem (dest-coalesced access)
    ::cute::cooperative_copy<NumThreads, ::cute::sizeof_bits_v<T>>(thread_idx, smem, dst_tile);
  }
  else
  {
    // Boundary tile: manual bounds-checked copy via tensor indexing
    int tile_elems = actual_m * actual_n;
    for (int i = thread_idx; i < tile_elems; i += NumThreads)
    {
      int r                   = i / actual_n;
      int c                   = i % actual_n;
      smem_buf[r * TileN + c] = src(m_begin + r, n_begin + c);
    }
    __syncthreads();
    for (int i = thread_idx; i < tile_elems; i += NumThreads)
    {
      int r                         = i / actual_n;
      int c                         = i % actual_n;
      dst(m_begin + r, n_begin + c) = smem_buf[r * TileN + c];
    }
  }
}

//! @brief Launch the shared-memory tiled copy kernel for different layouts.
//!
//! Creates full CuTe tensors on the host and passes them to the kernel,
//! which uses tensor indexing for element access and tile construction.
//!
//! @tparam T Element type
//! @tparam SrcLayout CuTe layout type for source (rank-2)
//! @tparam DstLayout CuTe layout type for destination (rank-2)
template <typename T, typename SrcLayout, typename DstLayout>
void launch_copy_diff_layout(
  ::cuda::stream_ref stream,
  T* dst_ptr,
  const T* src_ptr,
  const SrcLayout& src_layout,
  const DstLayout& dst_layout)
{
  constexpr int TileM      = 32;
  constexpr int TileN      = 32;
  constexpr int block_size = 256;

  auto src_tensor = ::cute::make_tensor(::cute::make_gmem_ptr(src_ptr), src_layout);
  auto dst_tensor = ::cute::make_tensor(::cute::make_gmem_ptr(dst_ptr), dst_layout);

  int M         = ::cute::get<0>(::cute::shape(src_layout));
  int N         = ::cute::get<1>(::cute::shape(src_layout));
  int grid_size = ((M + TileM - 1) / TileM) * ((N + TileN - 1) / TileN);

  auto config = ::cuda::make_config(::cuda::block_dims<block_size>(), ::cuda::grid_dims(grid_size));
  ::cuda::launch(
    stream, config,
    copy_diff_layout_kernel<decltype(config), T, decltype(src_tensor), decltype(dst_tensor), TileM, TileN>,
    src_tensor, dst_tensor);
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_DIFF_LAYOUT
