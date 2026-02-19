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

#include <cuda/experimental/__copy_bytes/copy_bytes_naive.cuh>
#include <cuda/experimental/__copy_bytes/layout_utils.cuh>

#include <cuda/std/__cccl/prologue.h>
#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>

namespace cuda::experimental
{
//! @brief Unified tiled copy kernel.
//!
//! Each block processes one tile of TileSize elements along the innermost
//! dimension. Blocks are distributed over (tiles_per_row * outer_size) tiles.
//!
//! When both tensors have compile-time stride Int<1> in mode 0 (vectorized
//! path), each thread copies a contiguous chunk of EPT elements via
//! cute::copy, which auto-vectorizes through static layout analysis.
//!
//! Otherwise (element-wise path), each thread copies elements via a strided
//! loop using CuTe's linear index decomposition.
template <typename Config, typename SrcTensor, typename DstTensor, int TileSize>
__global__ void copy_bytes_kernel(Config config, SrcTensor src, DstTensor dst, int inner_size, int tiles_per_row)
{
  constexpr int NumThreads = ::cuda::gpu_thread.count(::cuda::block, config);
  uint32_t tid             = ::cuda::gpu_thread.rank(::cuda::block, config);
  int bid                  = ::cuda::block.rank(::cuda::grid, config);
  int inner_tile           = bid % tiles_per_row;
  int outer_idx            = bid / tiles_per_row;
  int inner_offset         = inner_tile * TileSize;
  int flat_offset          = inner_offset + outer_idx * inner_size;
  int remaining            = inner_size - inner_offset;

  constexpr bool __stride1 =
    ::cute::is_constant<1, decltype(::cute::stride<0>(::cuda::std::declval<SrcTensor>()))>::value
    && ::cute::is_constant<1, decltype(::cute::stride<0>(::cuda::std::declval<DstTensor>()))>::value;

  if constexpr (__stride1)
  {
    if (remaining >= TileSize)
    {
      constexpr int EPT     = TileSize / NumThreads;
      const auto thr_layout = ::cute::make_layout(::cute::Int<EPT>{});
      auto thr_src          = make_gmem_tensor(&src(flat_offset + tid * EPT), thr_layout);
      auto thr_dst          = make_gmem_tensor(&dst(flat_offset + tid * EPT), thr_layout);
      ::cute::copy(thr_src, thr_dst);
      return;
    }
  }
  for (int i = tid; i < remaining; i += NumThreads)
  {
    dst(flat_offset + i) = src(flat_offset + i);
  }
}

//! @brief Launch the unified copy kernel with pre-built tensors.
template <int TileSize, typename SrcTensor, typename DstTensor>
void __launch_copy_bytes_kernel(
  ::cuda::stream_ref stream, SrcTensor src_tensor, DstTensor dst_tensor, int inner_size, int outer_size)
{
  constexpr int block_size = 256;
  const int tiles_per_row  = ::cuda::ceil_div(inner_size, TileSize);
  const int grid_size      = tiles_per_row * outer_size;
  auto config              = ::cuda::make_config(::cuda::block_dims<block_size>(), ::cuda::grid_dims(grid_size));
  ::cuda::launch(
    stream, config, copy_bytes_kernel<decltype(config), SrcTensor, DstTensor, TileSize>, src_tensor, dst_tensor, inner_size, tiles_per_row);
}

//! @brief Register-based copy using cute::copy.
//!
//! Preprocesses both layouts to determine the optimal copy strategy:
//!
//! 1. Sort both layouts by src's ascending stride (common permutation).
//! 2. If both have stride-1 in mode 0 (vectorized path):
//!    - Compute the contiguous extent for each tensor, use the minimum as
//!      inner_size to avoid crossing mode boundaries.
//!    - Compute the maximum compatible vectorization width and recast.
//!    - Launch the unified kernel (cute::copy auto-vectorizes).
//! 3. If mode 0 is not stride-1 for both (element-wise path, rank <= 2):
//!    - Launch the unified kernel (element-wise through CuTe indexing).
//! 4. Fallback: copy_bytes_naive for rank > 2 or unsupported configurations.
template <typename T, typename SrcLayout, typename DstLayout>
void copy_bytes_registers(
  const T* src, const SrcLayout& src_layout, T* dst, const DstLayout& dst_layout, ::cuda::stream_ref stream)
{
  constexpr int MaxRank = 8;
  constexpr int SrcR    = decltype(::cute::rank(src_layout))::value;
  constexpr int DstR    = decltype(::cute::rank(dst_layout))::value;
  static_assert(SrcR <= MaxRank && DstR <= MaxRank, "Layout rank exceeds maximum supported rank");
  static_assert(SrcR == DstR, "Source and destination layouts must have the same rank");
  constexpr int __rank = SrcR;
  const int total      = static_cast<int>(::cute::size(src_layout));
  if (total == 0)
  {
    return;
  }
  // NOTE:: src/dst shapes_sorted are equal
  ::cuda::std::array<int, MaxRank> shapes_sorted{};
  ::cuda::std::array<int, MaxRank> src_strides_sorted{};
  ::cuda::std::array<int, MaxRank> dst_strides_sorted{};
  ::cuda::std::array<int, MaxRank> dst_shapes_sorted{};
  __extract_layout(src_layout, shapes_sorted, src_strides_sorted);
  __extract_layout(dst_layout, dst_shapes_sorted, dst_strides_sorted);
  // Sort both by src's ascending absolute stride (common permutation).
  // After this, src has stride-1 in mode 0 (if any mode is stride-1).
  __sort_by_stride_paired(shapes_sorted, src_strides_sorted, dst_strides_sorted, __rank);

  bool both_stride1 = (::cuda::std::abs(src_strides_sorted[0]) == 1) && (::cuda::std::abs(dst_strides_sorted[0]) == 1);

  if (both_stride1)
  {
    int src_extent = __contiguous_extent(shapes_sorted, src_strides_sorted, __rank);
    int dst_extent = __contiguous_extent(shapes_sorted, dst_strides_sorted, __rank);
    int inner_size = ::cuda::std::min(src_extent, dst_extent);
    int outer_size = total / inner_size;

    auto src_vector_bytes    = __max_vector_size_bytes(src, shapes_sorted, src_strides_sorted);
    auto dst_vector_bytes    = __max_vector_size_bytes(dst, shapes_sorted, dst_strides_sorted);
    auto common_vector_bytes = ::cuda::std::min(src_vector_bytes, dst_vector_bytes);

    // Recast to VecType and launch the vectorized kernel.
    // The lambda captures the optimized layouts and dispatches on common_vector_bytes.
    auto __dispatch_vec = [&](const auto& src_opt, const auto& dst_opt) {
      auto __launch = [&](auto __vec_c) {
        constexpr int VecBytes   = decltype(__vec_c)::value;
        constexpr int VecBitsInt = VecBytes * 8;
        using VecType            = ::cuda::std::__make_nbit_uint_t<VecBitsInt>;
        auto src_recast          = ::cute::recast<VecType>(make_gmem_tensor(src, src_opt));
        auto dst_recast          = ::cute::recast<VecType>(make_gmem_tensor(dst, dst_opt));
        const int vec_inner      = ::cute::size<0>(src_recast);
        const int vec_outer      = ::cute::size(src_recast) / vec_inner;
        constexpr int TileSize   = 256 * (16 / VecBytes);
        __launch_copy_bytes_kernel<TileSize>(stream, src_recast, dst_recast, vec_inner, vec_outer);
      };
      switch (common_vector_bytes)
      {
        case 16:
          __launch(::cuda::std::integral_constant<int, 16>{});
          break;
        case 8:
          __launch(::cuda::std::integral_constant<int, 8>{});
          break;
        case 4:
          __launch(::cuda::std::integral_constant<int, 4>{});
          break;
        case 2:
          __launch(::cuda::std::integral_constant<int, 2>{});
          break;
        default:
          __launch(::cuda::std::integral_constant<int, 1>{});
          break;
      }
    };

    if (__rank == 1 || inner_size == total)
    {
      auto opt = ::cute::make_layout(::cute::make_shape(total), ::cute::make_stride(::cute::_1{}));
      __dispatch_vec(opt, opt);
    }
    else if (__rank == 2)
    {
      auto shape   = ::cute::make_shape(shapes_sorted[0], shapes_sorted[1]);
      auto src_opt = ::cute::make_layout(shape, ::cute::make_stride(::cute::_1{}, src_strides_sorted[1]));
      auto dst_opt = ::cute::make_layout(shape, ::cute::make_stride(::cute::_1{}, dst_strides_sorted[1]));
      __dispatch_vec(src_opt, dst_opt);
    }
    else if (__rank == 3)
    {
      auto shape = ::cute::make_shape(shapes_sorted[0], shapes_sorted[1], shapes_sorted[2]);
      auto src_opt =
        ::cute::make_layout(shape, ::cute::make_stride(::cute::_1{}, src_strides_sorted[1], src_strides_sorted[2]));
      auto dst_opt =
        ::cute::make_layout(shape, ::cute::make_stride(::cute::_1{}, dst_strides_sorted[1], dst_strides_sorted[2]));
      __dispatch_vec(src_opt, dst_opt);
    }
    else
    {
      copy_bytes_naive(src, src_layout, dst, dst_layout, stream);
    }
  }
  else if (__rank <= 3)
  {
    constexpr int TileSize = 256 * 4;
    if (__rank == 1)
    {
      auto src_opt = ::cute::make_layout(::cute::make_shape(total), ::cute::make_stride(src_strides_sorted[0]));
      auto dst_opt = ::cute::make_layout(::cute::make_shape(total), ::cute::make_stride(dst_strides_sorted[0]));
      __launch_copy_bytes_kernel<TileSize>(
        stream, make_gmem_tensor(src, src_opt), make_gmem_tensor(dst, dst_opt), total, 1);
    }
    else if (__rank == 2)
    {
      auto shape   = ::cute::make_shape(shapes_sorted[0], shapes_sorted[1]);
      auto src_opt = ::cute::make_layout(shape, ::cute::make_stride(src_strides_sorted[0], src_strides_sorted[1]));
      auto dst_opt = ::cute::make_layout(shape, ::cute::make_stride(dst_strides_sorted[0], dst_strides_sorted[1]));
      __launch_copy_bytes_kernel<TileSize>(
        stream, make_gmem_tensor(src, src_opt), make_gmem_tensor(dst, dst_opt), total, 1);
    }
    else
    {
      auto shape   = ::cute::make_shape(shapes_sorted[0], shapes_sorted[1], shapes_sorted[2]);
      auto src_opt = ::cute::make_layout(
        shape, ::cute::make_stride(src_strides_sorted[0], src_strides_sorted[1], src_strides_sorted[2]));
      auto dst_opt = ::cute::make_layout(
        shape, ::cute::make_stride(dst_strides_sorted[0], dst_strides_sorted[1], dst_strides_sorted[2]));
      __launch_copy_bytes_kernel<TileSize>(
        stream, make_gmem_tensor(src, src_opt), make_gmem_tensor(dst, dst_opt), total, 1);
    }
  }
  else
  {
    copy_bytes_naive(src, src_layout, dst, dst_layout, stream);
  }
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS
