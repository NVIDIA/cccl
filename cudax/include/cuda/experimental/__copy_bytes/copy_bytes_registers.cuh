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

#include <cuda/experimental/__copy/types.cuh>
#include <cuda/experimental/__copy/utils.cuh>
#include <cuda/experimental/__copy_bytes/copy_bytes_naive.cuh>
#include <cuda/experimental/__copy_bytes/layout_utils.cuh>

#include <cuda/std/__cccl/prologue.h>
#include <cute/algorithm/copy.hpp>
#include <cute/layout.hpp>
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
template <typename Config, typename SrcTensor, typename DstTensor, int TileSize, int VecBitsInt>
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
      ::cute::copy(::cute::AutoVectorizingCopyWithAssumedAlignment<VecBitsInt>{}, thr_src, thr_dst);
      return;
    }
  }
  for (int i = tid; i < remaining; i += NumThreads)
  {
    dst(flat_offset + i) = src(flat_offset + i);
  }
}

//! @brief Launch the unified copy kernel with pre-built tensors.
template <int TileSize, int VecBitsInt, typename SrcTensor, typename DstTensor>
void __launch_copy_bytes_kernel(
  ::cuda::stream_ref stream, SrcTensor src_tensor, DstTensor dst_tensor, int inner_size, int outer_size)
{
  constexpr int block_size = 256;
  const int tiles_per_row  = ::cuda::ceil_div(inner_size, TileSize);
  const int grid_size      = tiles_per_row * outer_size;
  auto config              = ::cuda::make_config(::cuda::block_dims<block_size>(), ::cuda::grid_dims(grid_size));
  ::cuda::launch(
    stream,
    config,
    copy_bytes_kernel<decltype(config), SrcTensor, DstTensor, TileSize, VecBitsInt>,
    src_tensor,
    dst_tensor,
    inner_size,
    tiles_per_row);
}

//! @brief Dispatch a vectorized copy kernel based on the common vector size in bytes.
//!
//! Recasts the source and destination tensors to the appropriate vector type
//! and launches the unified tiled copy kernel.
//!
//! @param[in] __stream    CUDA stream to launch on
//! @param[in] __src       Source CuTe tensor
//! @param[in] __dst       Destination CuTe tensor
//! @param[in] __common_vector_bytes  Common vectorization width in bytes (1, 2, 4, 8, or 16)
template <typename _SrcTensor, typename _DstTensor>
void __dispatch_vectorized_copy(
  ::cuda::stream_ref __stream, const _SrcTensor& __src, const _DstTensor& __dst, int __common_vector_bytes)
{
  auto __launch = [&](auto __vec_c) {
    constexpr int VecBytes   = decltype(__vec_c)::value;
    constexpr int VecBitsInt = VecBytes * 8;
    using VecType            = ::cuda::std::__make_nbit_uint_t<VecBitsInt>;
    auto src_recast          = ::cute::recast<VecType>(__src);
    auto dst_recast          = ::cute::recast<VecType>(__dst);
    const int vec_inner      = ::cute::size<0>(src_recast);
    const int vec_outer      = ::cute::size(src_recast) / vec_inner;
    constexpr int TileSize   = 256 * (16 / VecBytes);
    __launch_copy_bytes_kernel<TileSize, VecBitsInt>(__stream, src_recast, dst_recast, vec_inner, vec_outer);
  };
  switch (__common_vector_bytes)
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
}

//! @brief Register-based copy using cute::copy.
//!
//! Preprocesses both layouts to determine the optimal copy strategy:
//!
//! 1. Sort both layouts by dst's ascending stride (common permutation).
//! 2. If both have stride-1 in mode 0 (vectorized path):
//!    - Compute the contiguous extent for each tensor, use the minimum as
//!      inner_size to avoid crossing mode boundaries.
//!    - Compute the maximum compatible vectorization width and recast.
//!    - Launch the unified kernel (cute::copy auto-vectorizes).
//! 3. Fallback: copy_bytes_naive for non-vectorizable or unsupported configurations.
template <typename T, typename SrcLayout, typename DstLayout>
void copy_bytes_registers(
  const T* src, const SrcLayout& src_layout, T* dst, const DstLayout& dst_layout, ::cuda::stream_ref stream)
{
  constexpr int SrcR = decltype(::cute::rank(src_layout))::value;
  constexpr int DstR = decltype(::cute::rank(dst_layout))::value;
  static_assert(SrcR == DstR, "Source and destination layouts must have the same rank");
  const int total_size = static_cast<int>(::cute::size(src_layout));
  if (total_size == 0)
  {
    return;
  }
  if constexpr (SrcR == 0)
  {
    return;
  }
  else
  {
    // NOTE: source and destination extents are idential
    auto __src = __to_raw_tensor<SrcR>(src, src_layout, __remove_extent1_mode);
    auto __dst = __to_raw_tensor<DstR>(dst, dst_layout, __remove_extent1_mode);
    _CCCL_ASSERT(__src.__rank == __dst.__rank,
                 "Source and destination ranks must be equal after removing extent-1 modes");
    _CCCL_ASSERT(__src.__shapes == __dst.__shapes, "Source and destination shapes must be identical");
    // Sort both by dst's ascending absolute stride (common permutation).
    // After this, dst has stride-1 in mode 0 (if any mode is stride-1).
    // Shapes are kept in sync (both tensors share the same shape because they are ordered by the same permutation).
    __sort_by_stride_paired(__src, __dst);
    // Merge adjacent modes that are contiguous in both tensors, reducing effective rank.
    __coalesce_paired(__src, __dst);

    const int __actual_rank        = static_cast<int>(__src.__rank);
    const bool are_both_contiguous = (__src.__strides[0] == 1) && (__dst.__strides[0] == 1);

    if (are_both_contiguous)
    {
      const int inner_size           = static_cast<int>(__src.__shapes[0]);
      const auto src_vector_bytes    = __max_vector_size_bytes(__src);
      const auto dst_vector_bytes    = __max_vector_size_bytes(__dst);
      const auto common_vector_bytes = ::cuda::std::min(src_vector_bytes, dst_vector_bytes);

      printf("--------------------------------\n");
      cute::print(src_layout);
      printf("\n");
      cute::print(dst_layout);
      printf("\n");
      printf("common_vector_bytes: %d\n", (int) common_vector_bytes);
      __println(__src);
      __println(__dst);

      if (__actual_rank <= 1 || inner_size == total_size)
      {
        auto opt        = ::cute::make_layout(::cute::make_shape(total_size), ::cute::make_stride(::cute::_1{}));
        auto src_tensor = ::cuda::experimental::make_gmem_tensor(__src.__data, opt);
        auto dst_tensor = ::cuda::experimental::make_gmem_tensor(__dst.__data, opt);
        __dispatch_vectorized_copy(stream, src_tensor, dst_tensor, common_vector_bytes);
      }
      else if (__actual_rank == 2)
      {
        auto shape      = ::cute::make_shape(__src.__shapes[0], __src.__shapes[1]);
        auto src_tensor = ::cuda::experimental::make_gmem_tensor(
          __src.__data, ::cute::make_layout(shape, ::cute::make_stride(::cute::_1{}, __src.__strides[1])));
        auto dst_tensor = ::cuda::experimental::make_gmem_tensor(
          __dst.__data, ::cute::make_layout(shape, ::cute::make_stride(::cute::_1{}, __dst.__strides[1])));
        __dispatch_vectorized_copy(stream, src_tensor, dst_tensor, common_vector_bytes);
      }
      else if (__actual_rank == 3)
      {
        auto shape      = ::cute::make_shape(__src.__shapes[0], __src.__shapes[1], __src.__shapes[2]);
        auto src_tensor = ::cuda::experimental::make_gmem_tensor(
          __src.__data,
          ::cute::make_layout(shape, ::cute::make_stride(::cute::_1{}, __src.__strides[1], __src.__strides[2])));
        auto dst_tensor = ::cuda::experimental::make_gmem_tensor(
          __dst.__data,
          ::cute::make_layout(shape, ::cute::make_stride(::cute::_1{}, __dst.__strides[1], __dst.__strides[2])));
        __dispatch_vectorized_copy(stream, src_tensor, dst_tensor, common_vector_bytes);
      }
      else
      {
        ::cuda::experimental::copy_bytes_naive(src, src_layout, dst, dst_layout, stream);
      }
    }
    else
    {
      ::cuda::experimental::copy_bytes_naive(src, src_layout, dst, dst_layout, stream);
    }
  }
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_BYTES_REGISTERS
