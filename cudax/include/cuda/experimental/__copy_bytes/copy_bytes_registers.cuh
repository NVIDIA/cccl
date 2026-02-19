//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS
#define _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS

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
#include <cuda/std/__type_traits/make_nbit_int.h>

#include <cuda/experimental/__copy_bytes/layout_utils.cuh>

#include <cuda/std/__cccl/prologue.h>
#include <cute/tensor.hpp>

namespace cuda::experimental
{
//! @brief Tiled same-layout copy kernel using cooperative_copy.
//!
//! Each block processes one tile of TileSize vectorized elements along the
//! innermost (stride-1) dimension. Blocks are distributed over
//! (tiles_per_row * outer_size) total tiles, where outer_size is the product
//! of all dimensions except mode 0.
//!
//! Full tiles use cooperative_copy with static shapes for optimal thread
//! partitioning. Boundary tiles (last tile per row) use a bounds-checked loop.
//!
//! Works for arbitrary-rank tensors whose mode 0 has stride Int<1>.
//! For rank-1 tensors, outer_size == 1 and this reduces to a simple 1D tiling.
//!
//! @tparam Config    Kernel launch configuration type (encodes block dimensions)
//! @tparam SrcTensor CuTe tensor type for source (recast to VecType)
//! @tparam DstTensor CuTe tensor type for destination (recast to VecType)
//! @tparam TileSize  Number of VecType elements per tile (compile-time constant)
template <typename Config, typename SrcTensor, typename DstTensor, int TileSize>
__global__ void copy_bytes_registers_tiled_kernel(Config config, SrcTensor src, DstTensor dst, int inner_size)
{
  constexpr int NumThreads = ::cuda::gpu_thread.count(::cuda::block, config);
  uint32_t thread_idx      = ::cuda::gpu_thread.rank(::cuda::block, config);
  int block_idx            = ::cuda::block.rank(::cuda::grid, config);
  int tiles_per_row        = ::cuda::ceil_div(inner_size, TileSize);
  int inner_tile           = block_idx % tiles_per_row;
  int outer_idx            = block_idx / tiles_per_row;
  int inner_offset         = inner_tile * TileSize;
  int flat_offset          = inner_offset + outer_idx * inner_size;
  int remaining            = inner_size - inner_offset;

  if (remaining >= TileSize)
  {
    auto tile_layout = make_static_layout<TileSize>();
    auto src_tile    = make_gmem_tensor(&src(flat_offset), tile_layout);
    auto dst_tile    = make_gmem_tensor(&dst(flat_offset), tile_layout);

    ::cute::cooperative_copy<NumThreads, ::cute::sizeof_bits_v<typename SrcTensor::value_type>>(
      thread_idx, src_tile, dst_tile);
  }
  else
  {
    for (int i = thread_idx; i < remaining; i += NumThreads)
    {
      dst(flat_offset + i) = src(flat_offset + i);
    }
  }
}

//! @brief 2D tiled copy kernel using cooperative_copy for different layouts.
//!
//! Each block processes one (TileM x TileN) tile. Full tiles use
//! cooperative_copy with static shapes and per-tensor dynamic strides.
//! Boundary tiles use a manual bounds-checked loop.
//!
//! cooperative_copy automatically selects a good permutation and
//! vectorization width based on both source and destination layouts.
template <typename Config, typename SrcTensor, typename DstTensor, int TileM, int TileN>
__global__ void copy_bytes_registers_2d_kernel(Config config, SrcTensor src, DstTensor dst)
{
  constexpr int NumThreads = ::cuda::gpu_thread.count(::cuda::block, config);
  uint32_t thread_idx      = ::cuda::gpu_thread.rank(::cuda::block, config);
  int block_idx            = ::cuda::block.rank(::cuda::grid, config);

  int M = ::cute::get<0>(::cute::shape(src));
  int N = ::cute::get<1>(::cute::shape(src));

  int num_tiles_n = ::cuda::ceil_div(N, TileN);
  int tile_m_idx  = block_idx / num_tiles_n;
  int tile_n_idx  = block_idx % num_tiles_n;

  int m_begin  = tile_m_idx * TileM;
  int n_begin  = tile_n_idx * TileN;
  int actual_m = min(TileM, M - m_begin);
  int actual_n = min(TileN, N - n_begin);

  if (actual_m == TileM && actual_n == TileN)
  {
    auto src_tile = make_gmem_tensor(
      &src(m_begin, n_begin),
      ::cute::make_layout(
        ::cute::make_shape(::cute::Int<TileM>{}, ::cute::Int<TileN>{}),
        ::cute::make_stride(::cute::get<0>(::cute::stride(src)), ::cute::get<1>(::cute::stride(src)))));
    auto dst_tile = make_gmem_tensor(
      &dst(m_begin, n_begin),
      ::cute::make_layout(
        ::cute::make_shape(::cute::Int<TileM>{}, ::cute::Int<TileN>{}),
        ::cute::make_stride(::cute::get<0>(::cute::stride(dst)), ::cute::get<1>(::cute::stride(dst)))));

    ::cute::cooperative_copy<NumThreads, ::cute::sizeof_bits_v<typename SrcTensor::value_type>>(
      thread_idx, src_tile, dst_tile);
  }
  else
  {
    int tile_elems = actual_m * actual_n;
    for (int i = thread_idx; i < tile_elems; i += NumThreads)
    {
      int r                         = i / actual_n;
      int c                         = i % actual_n;
      dst(m_begin + r, n_begin + c) = src(m_begin + r, n_begin + c);
    }
  }
}

//! @brief Launch the 2D tiled copy kernel for different layouts.
template <typename T, typename SrcLayout, typename DstLayout>
void launch_copy_bytes_registers_2d(
  ::cuda::stream_ref stream, const T* src_ptr, const SrcLayout& src_layout, T* dst_ptr, const DstLayout& dst_layout)
{
  constexpr int TileM      = 32;
  constexpr int TileN      = 32;
  constexpr int block_size = 256;

  auto src_tensor = make_gmem_tensor(src_ptr, src_layout);
  auto dst_tensor = make_gmem_tensor(dst_ptr, dst_layout);

  int M         = ::cute::get<0>(::cute::shape(src_layout));
  int N         = ::cute::get<1>(::cute::shape(src_layout));
  int grid_size = ::cuda::ceil_div(M, TileM) * ::cuda::ceil_div(N, TileN);

  auto config = ::cuda::make_config(::cuda::block_dims<block_size>(), ::cuda::grid_dims(grid_size));
  ::cuda::launch(
    stream,
    config,
    copy_bytes_registers_2d_kernel<decltype(config), decltype(src_tensor), decltype(dst_tensor), TileM, TileN>,
    src_tensor,
    dst_tensor);
}

//! @brief Launch the vectorized same-layout copy kernel.
//!
//! Recasts the tensors to a wider VecType based on the computed vectorization
//! width. Tiles the innermost (stride-1) dimension with cooperative_copy for
//! all ranks. Blocks are distributed over (tiles_per_row * outer_size).
//!
//! @pre Mode 0 of @p layout must have stride Int<1>.
//!
//! @tparam VecBytes Vectorization width in bytes (compile-time template parameter)
//! @tparam T Element type of the original tensors
//! @tparam Layout CuTe layout type (same for src and dst)
template <int VecBytes, typename T, typename Layout>
void launch_copy_bytes_registers(::cuda::stream_ref stream, const T* src_ptr, T* dst_ptr, const Layout& layout)
{
  using VecType            = ::cuda::std::__make_nbit_uint_t<VecBytes * 8>;
  auto src_tensor          = ::cute::recast<VecType>(make_gmem_tensor(src_ptr, layout));
  auto dst_tensor          = ::cute::recast<VecType>(make_gmem_tensor(dst_ptr, layout));
  constexpr int block_size = 256;
  constexpr int TileSize   = block_size * 4;
  int inner_size           = ::cute::size<0>(src_tensor);
  int outer_size           = ::cute::size(src_tensor) / inner_size;
  int tiles_per_row        = ::cuda::ceil_div(inner_size, TileSize);
  int grid_size            = tiles_per_row * outer_size;
  auto config              = ::cuda::make_config(::cuda::block_dims<block_size>(), ::cuda::grid_dims(grid_size));
  const auto& kernel =
    copy_bytes_registers_tiled_kernel<decltype(config), decltype(src_tensor), decltype(dst_tensor), TileSize>;

  ::cuda::launch(stream, config, kernel, src_tensor, dst_tensor, inner_size);
}

//! @brief Register-based copy using cooperative_copy.
//!
//! Preprocesses both layouts independently (sort, coalesce, filter).
//! - **Same layout**: vectorized recast + 1D tiling with cooperative_copy
//! - **Different layout, rank-2**: 2D tiling with cooperative_copy (no shared memory)
//! - **Fallback**: copy_bytes_naive for unsupported configurations
template <typename T, typename SrcLayout, typename DstLayout>
void copy_bytes_registers(
  const T* src, const SrcLayout& src_layout, T* dst, const DstLayout& dst_layout, ::cuda::stream_ref stream)
{
  constexpr int MaxRank     = 8;
  constexpr int SrcR        = decltype(::cute::rank(src_layout))::value;
  constexpr int DstR        = decltype(::cute::rank(dst_layout))::value;
  constexpr auto __max_rank = ::cuda::std::max(SrcR, DstR);
  static_assert(__max_rank <= MaxRank, "Layout rank exceeds maximum supported rank");

  ::cuda::std::array<int, MaxRank> src_shapes{};
  ::cuda::std::array<int, MaxRank> src_strides{};
  __extract_layout(src_layout, src_shapes, src_strides);

  ::cuda::std::array<int, MaxRank> dst_shapes{};
  ::cuda::std::array<int, MaxRank> dst_strides{};
  __extract_layout(dst_layout, dst_shapes, dst_strides);

  // Sort into canonical (ascending stride) order for comparison.
  // Do NOT coalesce before comparing: coalescing erases the distinction
  // between different physical layouts (e.g. row-major vs col-major).
  auto src_sorted_s  = src_shapes;
  auto src_sorted_st = src_strides;
  auto dst_sorted_s  = dst_shapes;
  auto dst_sorted_st = dst_strides;
  sort_modes_by_stride(src_sorted_s, src_sorted_st, SrcR);
  sort_modes_by_stride(dst_sorted_s, dst_sorted_st, DstR);

  bool same_layout = layouts_match(src_sorted_s, src_sorted_st, dst_sorted_s, dst_sorted_st, MaxSrcDst);

  if (same_layout)
  {
    runtime_coalesce(src_sorted_s, src_sorted_st, SrcR);

    ::cuda::std::array<int, MaxRank> eff_shape{};
    ::cuda::std::array<int, MaxRank> eff_stride{};
    int eff_rank = 0;
    for (int i = 0; i < SrcR; ++i)
    {
      if (src_sorted_s[i] > 1)
      {
        eff_shape[eff_rank]  = src_sorted_s[i];
        eff_stride[eff_rank] = src_sorted_st[i];
        ++eff_rank;
      }
    }

    if (eff_rank == 0)
    {
      auto one = make_static_layout<1>();
      copy_bytes_naive(src, one, dst, one, stream);
      return;
    }

    auto src_vec   = __max_vector_size(src, eff_shape, eff_stride);
    auto dst_vec   = __max_vector_size(dst, eff_shape, eff_stride);
    auto vec_bytes = ::cuda::std::min(src_vec, dst_vec);

    auto dispatch = [&](const auto& opt_layout) {
      switch (vec_bytes)
      {
        case 16:
          launch_copy_bytes_registers<16>(stream, src, dst, opt_layout);
          break;
        case 8:
          launch_copy_bytes_registers<8>(stream, src, dst, opt_layout);
          break;
        case 4:
          launch_copy_bytes_registers<4>(stream, src, dst, opt_layout);
          break;
        case 2:
          launch_copy_bytes_registers<2>(stream, src, dst, opt_layout);
          break;
        default:
          launch_copy_bytes_registers<1>(stream, src, dst, opt_layout);
          break;
      }
    };

    if (eff_rank == 1 && eff_stride[0] == 1)
    {
      dispatch(::cute::make_layout(::cute::make_shape(eff_shape[0]), ::cute::make_stride(::cute::Int<1>{})));
    }
    else if (eff_rank == 2 && eff_stride[0] == 1)
    {
      dispatch(::cute::make_layout(
        ::cute::make_shape(eff_shape[0], eff_shape[1]), ::cute::make_stride(::cute::Int<1>{}, eff_stride[1])));
    }
    else if (eff_rank == 3 && eff_stride[0] == 1)
    {
      dispatch(::cute::make_layout(::cute::make_shape(eff_shape[0], eff_shape[1], eff_shape[2]),
                                   ::cute::make_stride(::cute::Int<1>{}, eff_stride[1], eff_stride[2])));
    }
    else
    {
      copy_bytes_naive(src, src_layout, dst, dst_layout, stream);
    }
  }
  else if (SrcR == 2 && DstR == 2)
  {
    int M        = src_shapes[0];
    int N        = src_shapes[1];
    auto src_opt = ::cute::make_layout(::cute::make_shape(M, N), ::cute::make_stride(src_strides[0], src_strides[1]));
    auto dst_opt = ::cute::make_layout(::cute::make_shape(M, N), ::cute::make_stride(dst_strides[0], dst_strides[1]));
    launch_copy_bytes_registers_2d(stream, src, src_opt, dst, dst_opt);
  }
  else
  {
    copy_bytes_naive(src, src_layout, dst, dst_layout, stream);
  }
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS
