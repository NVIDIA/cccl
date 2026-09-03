//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_COPY_SHARED_MEMORY_H
#define _CUDAX__COPY_COPY_SHARED_MEMORY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__launch/configuration.h>
#include <cuda/__launch/launch.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/array>

#include <cuda/experimental/__copy/copy_shared_memory_utils.cuh>
#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy_bytes/types.cuh>

#include <cuda/std/__cccl/prologue.h>

//! Shared-memory tiled transpose for arbitrary-rank tensor copies.
//!
//! The overall idea is to decompose the tensors into tiles that can fit in shared memory.
//! Each tile is assigned to a thread block. A tile can entirely represent a dimension or split the respective extent.
//! The algorithm creates tiles over dimensions that provide coalesced accesses in the source and destination tensors.
//!
//! (1) Grid decomposition
//! The tensor is partitioned into tiles whose per-dimension sizes are capped by warp size and shared-memory capacity.
//! The total number of tiles (product of ceil(extent[d] / tile_size[d]) over all dimensions) becomes the 1-D grid size.
//!
//! (2) Block processing
//! Each block handles one tile in two phases:
//!   1. *Load*: threads cooperatively read source elements into shared memory.
//!      This requires additional logic to "transpose" the source tensor into a row-major order.
//!      The mapping is determined by using the source-tile permutation obtained by sorting by |src stride|.
//!   2. *Store*: after a barrier, threads read shared memory in destination-coalesced order by using the
//!      destination-tile permutation obtained by sorting by |dst stride|.
//!
//! Boundary tiles that extend past the tensor extents fall back to a direct element-wise copy without shared memory.

namespace cuda::experimental
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Compute an optimized shared-memory offset.
//!
//! @param[in] __offset The offset in the shared-memory tile.
//! @return The physical offset in the optimized shared-memory layout.
template <bool _UseOptimizedSmemLayout, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_DEVICE_API __tile_extent_t __smem_offset(__tile_extent_t __offset) noexcept
{
  if constexpr (_UseOptimizedSmemLayout && _MaxRank == 2 && (sizeof(_Tp) == 1 || sizeof(_Tp) == 2))
  {
    // Align each row to the next 32-bit bank word. The resulting 36-char and 34-short pitches distribute transposed
    // accesses across all 32 shared-memory banks.
    constexpr __tile_extent_t __smem_padding = sizeof(unsigned) / sizeof(_Tp);
    return __offset + (__offset / __max_tile_size_32) * __smem_padding;
  }
  else if constexpr (_UseOptimizedSmemLayout)
  {
    static_assert(__max_tile_size_32 == 32, "XOR shared-memory swizzle assumes 32 banks and 32-element tile modes");
    constexpr __tile_extent_t __swizzle_tile_size = __max_tile_size_32 * __max_tile_size_32;
    const auto __outer                            = __offset / __swizzle_tile_size;
    const auto __offset_tile_rounded              = __outer * __swizzle_tile_size;
    const auto __inner                            = __offset - __offset_tile_rounded;
    const auto __row                              = __inner / __max_tile_size_32;
    const auto __row_tile_rounded                 = __row * __max_tile_size_32;
    const auto __col                              = __inner - __row_tile_rounded;
    return __offset_tile_rounded + __row_tile_rounded + (__col ^ __row);
  }
  return __offset;
}

//! @brief Shared-memory tiled transpose kernel for arbitrary-rank tensors.
//!
//! Each block processes one tile. Threads cooperatively iterate over tile elements with a stride loop. Full (interior)
//! tiles use a two-phase shared-memory transpose: load source data into shared memory using source-coalesced ordering,
//! then store from shared memory to destination using destination-coalesced ordering. Partial (boundary) tiles copy
//! elements directly without shared memory.
//!
//! @param[in]  __config                 Kernel launch configuration
//! @param[in]  __src_ptr                Pointer to source data
//! @param[in]  __src_accessor           Accessor for reading source elements
//! @param[out] __dst_ptr                Pointer to destination data
//! @param[in]  __dst_accessor           Accessor for writing destination elements
//! @param[in]  __grid_iter              Coordinate iterator for grid tile decomposition
//! @param[in]  __grid_tile_src_strides  Per-dimension source strides scaled by tile sizes
//! @param[in]  __grid_tile_dst_strides  Per-dimension destination strides scaled by tile sizes
//! @param[in]  __tile_perm_iter         Coordinate iterator for src-permuted tile decomposition
//! @param[in]  __src_perm_src_strides   Src-permuted source strides for loading
//! @param[in]  __tile_src_perm_smem_strides Src-permuted shared memory strides for loading
//! @param[in]  __tile_dst_perm_iter     Coordinate iterator for dst-permuted tile decomposition
//! @param[in]  __dst_perm_dst_strides   Dst-permuted destination strides for storing
//! @param[in]  __tile_dst_smem_strides  Dst-permuted shared memory strides for storing
//! @param[in]  __dst_strides            Per-dimension destination strides for partial tiles
//! @param[in]  __tile_total_size        Total number of elements in one tile
//! @param[in]  __tile_sizes             Per-dimension tile extents
//! @param[in]  __extents                Per-dimension tensor extents (for partial-tile bounds)
//! @param[in]  __src_strides            Per-dimension source strides (for partial-tile access)
template <bool _UseOptimizedSmemLayout,
          typename _Config,
          ::cuda::std::size_t _MaxRankUZ,
          typename _TpSrc,
          typename _TpDst,
          typename _SrcAccessor,
          typename _DstAccessor,
          typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut>
_CCCL_KERNEL_ATTRIBUTES void __copy_shared_mem_kernel(
  _CCCL_GRID_CONSTANT const _Config __config,
  const _TpSrc* _CCCL_RESTRICT __src_ptr,
  _CCCL_GRID_CONSTANT const _SrcAccessor __src_accessor,
  _TpDst* _CCCL_RESTRICT __dst_ptr,
  _CCCL_GRID_CONSTANT const _DstAccessor __dst_accessor,
  _CCCL_GRID_CONSTANT const __tensor_coord_iterator<_ExtentT, _MaxRankUZ> __grid_iter,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTIn, _MaxRankUZ> __grid_tile_src_strides,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTOut, _MaxRankUZ> __grid_tile_dst_strides,
  _CCCL_GRID_CONSTANT const __tensor_coord_iterator<__tile_extent_t, _MaxRankUZ> __tile_perm_iter,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTIn, _MaxRankUZ> __src_perm_src_strides,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<__tile_extent_t, _MaxRankUZ> __tile_src_perm_smem_strides,
  _CCCL_GRID_CONSTANT const __tensor_coord_iterator<__tile_extent_t, _MaxRankUZ> __tile_dst_perm_iter,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTOut, _MaxRankUZ> __dst_perm_dst_strides,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<__tile_extent_t, _MaxRankUZ> __tile_dst_perm_smem_strides,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTOut, _MaxRankUZ> __dst_strides,
  _CCCL_GRID_CONSTANT const int __tile_total_size,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<__tile_extent_t, _MaxRankUZ> __tile_sizes,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_ExtentT, _MaxRankUZ> __extents,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTIn, _MaxRankUZ> __src_strides)
{
  constexpr auto __max_rank = int{_MaxRankUZ};
  // Grid tile decomposition: map linearized block index to src/dst base offsets
  // __grid_coords: linear tile index -> multi-dimensional coordinates (array)
  const auto __grid_index  = ::cuda::block.index_as<_ExtentT>(::cuda::grid).x;
  const auto __grid_coords = __grid_iter(__grid_index);

  {
    _StrideTIn __src_base  = 0;
    _StrideTOut __dst_base = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __k = 0; __k < __max_rank; ++__k)
    {
      __src_base += static_cast<_StrideTIn>(__grid_coords[__k]) * __grid_tile_src_strides[__k];
      __dst_base += static_cast<_StrideTOut>(__grid_coords[__k]) * __grid_tile_dst_strides[__k];
    }
    __src_ptr += __src_base;
    __dst_ptr += __dst_base;
  }

  // Partial tile detection: is the current tile full or partial?
  bool __is_full_tile = true;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __k = 0; __k < __max_rank; ++__k)
  {
    const auto __block_start = __grid_coords[__k] * __tile_sizes[__k];
    if (__block_start + __tile_sizes[__k] > __extents[__k])
    {
      __is_full_tile = false;
      break;
    }
  }

  // Dispatch to Full-tile or Boundary case
  const auto __tid           = ::cuda::gpu_thread.rank_as<int>(::cuda::block, __config);
  const auto __block_stride  = ::cuda::gpu_thread.count_as<int>(::cuda::block, __config);
  using __partial_tensor_src = __partial_tensor<const _TpSrc, _StrideTIn, _MaxRankUZ, _SrcAccessor>;
  using __partial_tensor_dst = __partial_tensor<_TpDst, _StrideTOut, _MaxRankUZ, _DstAccessor>;

  //--------------------------------------------------------------------------------------------------------------------
  // Full-tile shared-memory transpose
  if (__is_full_tile)
  {
    using _Tp = ::cuda::std::remove_cv_t<_TpSrc>;
    constexpr bool __use_rank2_padded_smem =
      _UseOptimizedSmemLayout && _MaxRankUZ == 2 && (sizeof(_Tp) == 1 || sizeof(_Tp) == 2);

    extern __shared__ char __smem_bytes[];
    auto* __smem = reinterpret_cast<_Tp*>(__smem_bytes);

    // (1) load src to shared memory by using the src/tile-permuted ordering
    if constexpr (__use_rank2_padded_smem)
    {
      using __src_value_type = ::cuda::std::remove_const_t<_TpSrc>;
      for (auto __i = __tid; __i < __tile_total_size; __i += __block_stride)
      {
        const auto __inner      = static_cast<__tile_extent_t>(__i) % __max_tile_size_32;
        const auto __outer      = static_cast<__tile_extent_t>(__i) / __max_tile_size_32;
        const auto __src_offset = static_cast<_StrideTIn>(__inner) * __src_perm_src_strides[0]
                                + static_cast<_StrideTIn>(__outer) * __src_perm_src_strides[1];
        const auto __raw_offset = __inner * __tile_src_perm_smem_strides[0] + __outer * __tile_src_perm_smem_strides[1];
        const auto __smem_offset = ::cuda::experimental::__smem_offset<true, _Tp, _MaxRankUZ>(__raw_offset);
        __smem[__smem_offset]    = __src_accessor.access(const_cast<__src_value_type*>(__src_ptr), __src_offset);
      }
    }
    else
    {
      using __partial_tensor_smem =
        __partial_tensor<_Tp, __tile_extent_t, _MaxRankUZ, ::cuda::std::default_accessor<_Tp>>;
      const __partial_tensor_src __src_tensor{__src_ptr, __src_perm_src_strides, __src_accessor};
      const __partial_tensor_smem __smem_tensor{
        __smem, __tile_src_perm_smem_strides, ::cuda::std::default_accessor<_Tp>{}};

      for (auto __i = __tid; __i < __tile_total_size; __i += __block_stride)
      {
        const auto __coords     = __tile_perm_iter(__i);
        const auto __raw_offset = __smem_tensor.__offset(__coords);
        const auto __optimized_offset =
          ::cuda::experimental::__smem_offset<_UseOptimizedSmemLayout, _Tp, _MaxRankUZ>(__raw_offset);
        __smem[__optimized_offset] = __src_tensor(__coords);
      }
    }
    __syncthreads();

    // (2) store from shared memory to destination by using the dst/tile-permuted ordering
    if constexpr (__use_rank2_padded_smem)
    {
      for (auto __i = __tid; __i < __tile_total_size; __i += __block_stride)
      {
        const auto __inner      = static_cast<__tile_extent_t>(__i) % __max_tile_size_32;
        const auto __outer      = static_cast<__tile_extent_t>(__i) / __max_tile_size_32;
        const auto __dst_offset = static_cast<_StrideTOut>(__inner) * __dst_perm_dst_strides[0]
                                + static_cast<_StrideTOut>(__outer) * __dst_perm_dst_strides[1];
        const auto __raw_offset = __inner * __tile_dst_perm_smem_strides[0] + __outer * __tile_dst_perm_smem_strides[1];
        const auto __smem_offset = ::cuda::experimental::__smem_offset<true, _Tp, _MaxRankUZ>(__raw_offset);
        __dst_accessor.access(__dst_ptr, __dst_offset) = __smem[__smem_offset];
      }
    }
    else
    {
      using __partial_tensor_smem =
        __partial_tensor<_Tp, __tile_extent_t, _MaxRankUZ, ::cuda::std::default_accessor<_Tp>>;
      const __partial_tensor_dst __dst_tensor{__dst_ptr, __dst_perm_dst_strides, __dst_accessor};
      const __partial_tensor_smem __smem_dst_tensor{
        __smem, __tile_dst_perm_smem_strides, ::cuda::std::default_accessor<_Tp>{}};

      for (auto __i = __tid; __i < __tile_total_size; __i += __block_stride)
      {
        const auto __coords     = __tile_dst_perm_iter(__i);
        const auto __raw_offset = __smem_dst_tensor.__offset(__coords);
        const auto __optimized_offset =
          ::cuda::experimental::__smem_offset<_UseOptimizedSmemLayout, _Tp, _MaxRankUZ>(__raw_offset);
        __dst_tensor(__coords) = __smem[__optimized_offset];
      }
    }
  }

  //--------------------------------------------------------------------------------------------------------------------
  // Boundary direct-copy (no shared memory)
  else
  {
    using __uextent_t = ::cuda::std::make_unsigned_t<_ExtentT>;
    const __partial_tensor_src __src_tensor{__src_ptr, __src_strides, __src_accessor};
    const __partial_tensor_dst __dst_tensor{__dst_ptr, __dst_strides, __dst_accessor};

    // Find the partial tile sizes and total number of elements
    ::cuda::std::array<__tile_extent_t, __max_rank> __partial_tile_sizes{};
    int __partial_tile_total = 1;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __k = 0; __k < __max_rank; ++__k)
    {
      const auto __block_start  = static_cast<__uextent_t>(__grid_coords[__k] * __tile_sizes[__k]);
      const auto __diff         = static_cast<__tile_extent_t>(__extents[__k] - __block_start);
      __partial_tile_sizes[__k] = ::cuda::std::min(__tile_sizes[__k], __diff);
      __partial_tile_total *= __partial_tile_sizes[__k];
    }

    // map the linear index to the multi-dimensional coordinates and copy the elements
    for (auto __i = __tid; __i < __partial_tile_total; __i += __block_stride)
    {
      __tile_extent_t __linear = __i;
      ::cuda::std::array<__tile_extent_t, __max_rank> __coords;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __k = 0; __k < __max_rank; ++__k)
      {
        __coords[__k] = __linear % __partial_tile_sizes[__k];
        __linear /= __partial_tile_sizes[__k];
      }
      __dst_tensor(__coords) = __src_tensor(__coords);
    }
  }
}

#if !_CCCL_COMPILER(NVRTC)

//! @brief Launch the shared-memory tiled transpose kernel.
//!
//! Precomputes the source/destination-coalesced permutations and tile shapes, constructs coordinate iterators, then
//! launches one block per tile.
//!
//! @pre `__src.__rank >= 2`
//!
//! @param[in]  __src          Source raw tensor descriptor
//! @param[out] __dst          Destination raw tensor descriptor
//! @param[in]  __stream       CUDA stream for asynchronous execution
//! @param[in]  __src_accessor Accessor for reading source elements
//! @param[in]  __dst_accessor Accessor for writing destination elements
template <typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _MaxRank,
          typename _SrcAccessor,
          typename _DstAccessor>
_CCCL_HOST_API void __launch_copy_shared_mem_kernel(
  const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _MaxRank>& __src,
  const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _MaxRank>& __dst,
  ::cuda::stream_ref __stream,
  const _SrcAccessor& __src_accessor = {},
  const _DstAccessor& __dst_accessor = {})
{
  namespace cudax = ::cuda::experimental;
  using ::cuda::std::size_t;
  _CCCL_ASSERT(__src.__rank >= 2, "Rank must be at least 2 for shared memory transpose");

  const auto __tiling               = cudax::__find_shared_mem_tiling<_TpIn>(__src, __dst, __stream.device());
  const auto __tile_sizes           = __tiling.__tile_sizes;
  const auto __rank                 = __src.__rank;
  const auto __tile_total_size      = __tiling.__tile_total_size;
  const auto __smem_allocation_size = __tiling.__smem_allocation_size;

  //
  //--------------------------------------------------------------------------------------------------------------------
  // Find the grid size (number of blocks) and strides for block index decomposition

  // __grid_perm represents the order in which the grid dimensions are visited
  ::cuda::std::array<size_t, _MaxRank> __grid_perm{};
  // __grid_perm is initialized to the identity permutation (source tensor order)
  for (size_t __i = 0; __i < _MaxRank; ++__i)
  {
    __grid_perm[__i] = __i;
  }
  // Optimization: improve destination cache locality.
  // When a destination dimension memory location is not divisible by the tile size, it means that the partial tile at
  // the end can be reused in the next iteration. In this case, visit the tiles by following the destination order.

  // On the other hand, this could hurt source cache locality. A solution is to enable this optimization depending on a
  // heuristic that compares how many tiles fit in the fastest-changing dimension of the source and destination tensors.
  // The heuristic requires the destination tensor to have no more than 32x more tiles than the source tensor (in the
  // fastest-changing dimension).

  const auto __src_inner_dim = __tiling.__src_perm[0];
  const auto __dst_inner_dim = __tiling.__dst_perm[0];
  const auto __src_inner_grid_size =
    ::cuda::ceil_div(__src.__extents[__src_inner_dim], static_cast<_ExtentT>(__tile_sizes[__src_inner_dim]));
  const auto __dst_inner_grid_size =
    ::cuda::ceil_div(__dst.__extents[__dst_inner_dim], static_cast<_ExtentT>(__tile_sizes[__dst_inner_dim]));
  const auto __dst_to_src_grid_ratio = ::cuda::ceil_div(__dst_inner_grid_size, __src_inner_grid_size);

  if (constexpr auto __max_dst_to_src_grid_ratio = 32; __dst_to_src_grid_ratio <= __max_dst_to_src_grid_ratio)
  {
    for (size_t __i = 1; __i < __rank; ++__i)
    {
      // For each dimension, compute the span (or offset) where the current tile starts in the destination tensor
      const auto __dst_prev_dim = __tiling.__dst_perm[__i - 1];
      const auto __dst_tile_span =
        static_cast<_StrideTOut>(__tile_sizes[__dst_prev_dim]) * __dst.__strides[__dst_prev_dim];
      // If the current destination stride is not divisible by the tile span, use the destination grid permutation
      const auto __dst_curr_dim = __tiling.__dst_perm[__i];
      if (__dst.__strides[__dst_curr_dim] % __dst_tile_span != 0)
      {
        __grid_perm = __tiling.__dst_perm;
        break;
      }
    }
  }

  ::cuda::std::array<_ExtentT, _MaxRank> __grid_tile_sizes{};
  ::cuda::std::array<_StrideTIn, _MaxRank> __grid_tile_src_strides{};
  ::cuda::std::array<_StrideTOut, _MaxRank> __grid_tile_dst_strides{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __grid_tile_extents{};
  ::cuda::std::array<_ExtentT, _MaxRank> __grid_extents{};
  ::cuda::std::array<_StrideTIn, _MaxRank> __grid_src_strides{};
  ::cuda::std::array<_StrideTOut, _MaxRank> __grid_dst_strides{};
  _ExtentT __grid_size = 1;
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    const auto __p               = __grid_perm[__i];
    __grid_tile_sizes[__i]       = ::cuda::ceil_div(__src.__extents[__p], static_cast<_ExtentT>(__tile_sizes[__p]));
    __grid_tile_src_strides[__i] = static_cast<_StrideTIn>(__tile_sizes[__p]) * __src.__strides[__p];
    __grid_tile_dst_strides[__i] = static_cast<_StrideTOut>(__tile_sizes[__p]) * __dst.__strides[__p];
    __grid_tile_extents[__i]     = __tile_sizes[__p];
    __grid_extents[__i]          = __dst.__extents[__p];
    __grid_src_strides[__i]      = __src.__strides[__p];
    __grid_dst_strides[__i]      = __dst.__strides[__p];
    __grid_size *= __grid_tile_sizes[__i];
  }
  // remaining unused dimensions
  for (size_t __i = __rank; __i < _MaxRank; ++__i)
  {
    __grid_tile_sizes[__i]   = 1;
    __grid_tile_extents[__i] = 1;
    __grid_extents[__i]      = 1;
  }

  //--------------------------------------------------------------------------------------------------------------------
  // Reordered arrays for loading src and storing dst based on coalesced permutations
  ::cuda::std::array<_StrideTIn, _MaxRank> __src_perm_src_strides{};
  ::cuda::std::array<_StrideTOut, _MaxRank> __dst_perm_dst_strides{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_src_perm_sizes{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_dst_perm_sizes{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_src_perm_smem_strides{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_dst_perm_smem_strides{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __canonical_strides{};
  __canonical_strides[0] = 1;
  for (size_t __i = 1; __i < __rank; ++__i)
  {
    __canonical_strides[__i] = __canonical_strides[__i - 1] * __tile_sizes[__i - 1];
  }
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    const auto __p                    = __tiling.__src_perm[__i];
    __tile_src_perm_sizes[__i]        = __tile_sizes[__p];
    __src_perm_src_strides[__i]       = __src.__strides[__p];
    __tile_src_perm_smem_strides[__i] = __canonical_strides[__p];

    const auto __q                    = __tiling.__dst_perm[__i];
    __tile_dst_perm_sizes[__i]        = __tile_sizes[__q];
    __dst_perm_dst_strides[__i]       = __dst.__strides[__q];
    __tile_dst_perm_smem_strides[__i] = __canonical_strides[__q];
  }
  for (size_t __i = __rank; __i < _MaxRank; ++__i)
  {
    __tile_src_perm_sizes[__i] = 1;
    __tile_dst_perm_sizes[__i] = 1;
  }

  //--------------------------------------------------------------------------------------------------------------------
  // Construct coordinate iterators on the host (precomputed fast modulo/division)
  // namely, given a linear index, compute the multi-dimensional coordinates
  const __tensor_coord_iterator<_ExtentT, _MaxRank> __grid_iter{__grid_tile_sizes}; // grid tile index
  const __tensor_coord_iterator<__tile_extent_t, _MaxRank> __tile_perm_iter{__tile_src_perm_sizes}; // src -> shared
                                                                                                    // memory
  const __tensor_coord_iterator<__tile_extent_t, _MaxRank> __tile_dst_perm_iter{__tile_dst_perm_sizes}; // shared memory
                                                                                                        // -> dst

  //--------------------------------------------------------------------------------------------------------------------
  // Launch the kernel
  using __value_type                = ::cuda::std::remove_cv_t<_TpIn>;
  constexpr int __thread_block_size = 256;

  const auto __config = ::cuda::make_config(
    ::cuda::block_dims(__thread_block_size),
    ::cuda::grid_dims(__grid_size),
    ::cuda::dynamic_shared_memory<__value_type[]>(__smem_allocation_size));

  if (__tiling.__use_xor_swizzle || __tiling.__use_padded_smem)
  {
    const auto __kernel = cudax::__copy_shared_mem_kernel<
      true,
      decltype(__config),
      _MaxRank,
      _TpIn,
      _TpOut,
      _SrcAccessor,
      _DstAccessor,
      _ExtentT,
      _StrideTIn,
      _StrideTOut>;

    ::cuda::launch(
      __stream,
      __config,
      __kernel,
      __src.__data,
      __src_accessor,
      __dst.__data,
      __dst_accessor,
      __grid_iter,
      __grid_tile_src_strides,
      __grid_tile_dst_strides,
      __tile_perm_iter,
      __src_perm_src_strides,
      __tile_src_perm_smem_strides,
      __tile_dst_perm_iter,
      __dst_perm_dst_strides,
      __tile_dst_perm_smem_strides,
      __grid_dst_strides,
      static_cast<int>(__tile_total_size),
      __grid_tile_extents,
      __grid_extents,
      __grid_src_strides);
  }
  else
  {
    const auto __kernel = cudax::__copy_shared_mem_kernel<
      false,
      decltype(__config),
      _MaxRank,
      _TpIn,
      _TpOut,
      _SrcAccessor,
      _DstAccessor,
      _ExtentT,
      _StrideTIn,
      _StrideTOut>;

    ::cuda::launch(
      __stream,
      __config,
      __kernel,
      __src.__data,
      __src_accessor,
      __dst.__data,
      __dst_accessor,
      __grid_iter,
      __grid_tile_src_strides,
      __grid_tile_dst_strides,
      __tile_perm_iter,
      __src_perm_src_strides,
      __tile_src_perm_smem_strides,
      __tile_dst_perm_iter,
      __dst_perm_dst_strides,
      __tile_dst_perm_smem_strides,
      __grid_dst_strides,
      static_cast<int>(__tile_total_size),
      __grid_tile_extents,
      __grid_extents,
      __grid_src_strides);
  }
}

#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_COPY_SHARED_MEMORY_H
