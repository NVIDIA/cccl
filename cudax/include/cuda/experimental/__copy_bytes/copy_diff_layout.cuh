//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__exception/cuda_error.h>

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/int.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::detail
{

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
//! @tparam SrcLayout CuTe layout type for source (rank-2, strides may be dynamic)
//! @tparam DstLayout CuTe layout type for destination (rank-2, strides may be dynamic)
//! @tparam TileM Tile height (compile-time constant)
//! @tparam TileN Tile width (compile-time constant)
template <typename T, typename SrcLayout, typename DstLayout, int TileM, int TileN>
__global__ void copy_diff_layout_kernel(
  const T* __restrict__ src_ptr,
  T* __restrict__ dst_ptr,
  SrcLayout src_layout,
  DstLayout dst_layout,
  int M,
  int N)
{
  using namespace cute;
  constexpr int NumThreads = 256;

  static __shared__ T smem_buf[TileM * TileN];

  // Swizzled shared memory layout for bank-conflict-free access
  auto smem_base   = make_layout(make_shape(Int<TileM>{}, Int<TileN>{}), make_stride(Int<TileN>{}, Int<1>{}));
  auto smem_layout = composition(Swizzle<3, 3, 3>{}, smem_base);
  auto smem        = make_tensor(make_smem_ptr(smem_buf), smem_layout);

  // 2D tile coordinates from flat block index
  int num_tiles_n = (N + TileN - 1) / TileN;
  int tile_m_idx  = blockIdx.x / num_tiles_n;
  int tile_n_idx  = blockIdx.x % num_tiles_n;

  int m_begin  = tile_m_idx * TileM;
  int n_begin  = tile_n_idx * TileN;
  int actual_m = min(TileM, M - m_begin);
  int actual_n = min(TileN, N - n_begin);

  // Extract strides from layouts
  auto src_stride_m = get<0>(stride(src_layout));
  auto src_stride_n = get<1>(stride(src_layout));
  auto dst_stride_m = get<0>(stride(dst_layout));
  auto dst_stride_n = get<1>(stride(dst_layout));

  int src_offset = m_begin * static_cast<int>(src_stride_m) + n_begin * static_cast<int>(src_stride_n);
  int dst_offset = m_begin * static_cast<int>(dst_stride_m) + n_begin * static_cast<int>(dst_stride_n);

  bool is_full_tile = (actual_m == TileM && actual_n == TileN);

  if (is_full_tile)
  {
    // Construct tile tensors with STATIC shapes and dynamic strides
    auto src_tile =
      make_tensor(make_gmem_ptr(src_ptr + src_offset),
                  make_layout(make_shape(Int<TileM>{}, Int<TileN>{}), make_stride(src_stride_m, src_stride_n)));
    auto dst_tile =
      make_tensor(make_gmem_ptr(dst_ptr + dst_offset),
                  make_layout(make_shape(Int<TileM>{}, Int<TileN>{}), make_stride(dst_stride_m, dst_stride_n)));

    // gmem -> smem (source-coalesced access)
    cooperative_copy<NumThreads, sizeof_bits_v<T>>(threadIdx.x, src_tile, smem);
    __syncthreads();

    // smem -> gmem (dest-coalesced access)
    cooperative_copy<NumThreads, sizeof_bits_v<T>>(threadIdx.x, smem, dst_tile);
  }
  else
  {
    // Boundary tile: manual bounds-checked copy without swizzle
    int tile_elems = actual_m * actual_n;
    for (int i = threadIdx.x; i < tile_elems; i += NumThreads)
    {
      int r = i / actual_n;
      int c = i % actual_n;
      smem_buf[r * TileN + c] =
        src_ptr[src_offset + r * static_cast<int>(src_stride_m) + c * static_cast<int>(src_stride_n)];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < tile_elems; i += NumThreads)
    {
      int r = i / actual_n;
      int c = i % actual_n;
      dst_ptr[dst_offset + r * static_cast<int>(dst_stride_m) + c * static_cast<int>(dst_stride_n)] =
        smem_buf[r * TileN + c];
    }
  }
}

//! @brief Launch the shared-memory tiled copy kernel for different layouts.
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
  const DstLayout& dst_layout,
  int M,
  int N)
{
  constexpr int TileM      = 32;
  constexpr int TileN      = 32;
  constexpr int block_size = 256;

  int num_tiles_m = (M + TileM - 1) / TileM;
  int num_tiles_n = (N + TileN - 1) / TileN;
  int grid_size   = num_tiles_m * num_tiles_n;

  copy_diff_layout_kernel<T, SrcLayout, DstLayout, TileM, TileN>
    <<<grid_size, block_size, 0, stream.get()>>>(src_ptr, dst_ptr, src_layout, dst_layout, M, N);
}

} // namespace cuda::experimental::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_DIFF_LAYOUT
