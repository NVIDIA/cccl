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
#include <cuda/std/__algorithm/stable_sort.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/array>

#include <cuda/experimental/__copy/copy_shared_memory_utils.cuh>
#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy_bytes/tensor_query.cuh>
#include <cuda/experimental/__copy_bytes/types.cuh>

#include <cuda/std/__cccl/prologue.h>

//! Shared-memory tiled transpose for arbitrary-rank tensor copies.
//!
//! The overall idea is to decompose the tensors into tiles that can fit in shared memory.
//! Each tile is assigned to a thread block. A tile can entirely represent a dimension or split the respective extent.
//! The algorithm creates tiles only for contiguous dimensions in the destination tensor.
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
//!   2. *Store*: after a barrier, threads read shared memory in the destination-natural
//!      (row-major) order and write to global memory, achieving coalesced destination writes.
//!
//! Boundary tiles that extend past the tensor extents fall back to a direct element-wise copy without shared memory.

namespace cuda::experimental
{
//! @brief Shared-memory tiled transpose kernel for arbitrary-rank tensors.
//!
//! Each block processes one tile. Threads cooperatively iterate over tile elements with a stride loop. Full (interior)
//! tiles use a two-phase shared-memory transpose: load source data into shared memory using source-coalesced ordering,
//! then store from shared memory to destination using destination-natural ordering. Partial (boundary) tiles copy
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
//! @param[in]  __tile_perm_smem_strides Src-permuted shared memory strides for loading
//! @param[in]  __tile_natural_iter      Coordinate iterator for dst-natural tile decomposition
//! @param[in]  __dst_strides            Per-dimension destination strides for storing
//! @param[in]  __tile_total_size        Total number of elements in one tile
//! @param[in]  __tile_sizes             Per-dimension tile extents
//! @param[in]  __extents                Per-dimension tensor extents (for partial-tile bounds)
//! @param[in]  __src_strides            Per-dimension source strides (for partial-tile access)
template <typename _Config,
          ::cuda::std::size_t _MaxRankUZ,
          typename _TpSrc,
          typename _TpDst,
          typename _SrcAccessor,
          typename _DstAccessor,
          typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut>
__global__ void __copy_shared_mem_kernel(
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
  _CCCL_GRID_CONSTANT const ::cuda::std::array<__tile_extent_t, _MaxRankUZ> __tile_perm_smem_strides,
  _CCCL_GRID_CONSTANT const __tensor_coord_iterator<__tile_extent_t, _MaxRankUZ> __tile_natural_iter,
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
    using __partial_tensor_smem =
      __partial_tensor<_Tp, __tile_extent_t, _MaxRankUZ, ::cuda::std::default_accessor<_Tp>>;

    extern __shared__ char __smem_bytes[];
    auto* __smem = reinterpret_cast<_Tp*>(__smem_bytes);

    // (1) load src to shared memory by using the src/tile-permuted ordering
    const __partial_tensor_src __src_tensor{__src_ptr, __src_perm_src_strides, __src_accessor};
    const __partial_tensor_smem __smem_tensor{__smem, __tile_perm_smem_strides, ::cuda::std::default_accessor<_Tp>{}};

    for (auto __i = __tid; __i < __tile_total_size; __i += __block_stride)
    {
      const auto __coords     = __tile_perm_iter(__i);
      __smem_tensor(__coords) = __src_tensor(__coords);
    }
    __syncthreads();

    // (2) store from shared memory to destination by using the dst-natural ordering (row-major)
    const __partial_tensor_dst __dst_tensor{__dst_ptr, __dst_strides, __dst_accessor};

    for (auto __i = __tid; __i < __tile_total_size; __i += __block_stride)
    {
      const auto __coords    = __tile_natural_iter(__i);
      __dst_tensor(__coords) = __smem[__i];
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
//! Precomputes the src-coalesced permutation and tile shapes, constructs coordinate iterators, then launches one
//! block per tile.
//!
//! @pre `__src.__rank >= 2`
//! @pre `__dst.__strides[0] == 1`
//! @pre `__src.__strides[0] != 1`
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
  _CCCL_ASSERT(__src.__strides[0] != 1, "Source must not have stride-1 in mode 0");
  _CCCL_ASSERT(__dst.__strides[0] == 1, "Destination must have stride-1 in mode 0");

  size_t __tile_total_size = 0;
  const auto __tile_sizes  = cudax::__find_shared_mem_tiling<_TpIn>(__dst, __tile_total_size);
  const auto __rank        = __src.__rank;

  //--------------------------------------------------------------------------------------------------------------------
  // Find the grid size (number of blocks) and strides for block index decomposition
  ::cuda::std::array<_ExtentT, _MaxRank> __grid_tile_sizes{};
  ::cuda::std::array<_StrideTIn, _MaxRank> __grid_tile_src_strides{};
  ::cuda::std::array<_StrideTOut, _MaxRank> __grid_tile_dst_strides{};
  _ExtentT __grid_size = 1;
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    __grid_tile_sizes[__i]       = ::cuda::ceil_div(__src.__extents[__i], static_cast<_ExtentT>(__tile_sizes[__i]));
    __grid_tile_src_strides[__i] = static_cast<_StrideTIn>(__tile_sizes[__i]) * __src.__strides[__i];
    __grid_tile_dst_strides[__i] = static_cast<_StrideTOut>(__tile_sizes[__i]) * __dst.__strides[__i];
    __grid_size *= __grid_tile_sizes[__i];
  }
  for (size_t __i = __rank; __i < _MaxRank; ++__i)
  {
    __grid_tile_sizes[__i] = 1;
  }

  //--------------------------------------------------------------------------------------------------------------------
  // source-coalesced permutation: sort modes by ascending |src_stride|
  ::cuda::std::array<size_t, _MaxRank> __src_perm{};
  for (size_t __i = 0; __i < _MaxRank; ++__i)
  {
    __src_perm[__i] = __i;
  }
  ::cuda::std::stable_sort(
    __src_perm.begin(), __src_perm.begin() + __rank, __stride_compare<_StrideTIn, _MaxRank>{__src.__strides});

  //--------------------------------------------------------------------------------------------------------------------
  // Reordered arrays for loading src to shared memory based on the source-coalesced permutation
  ::cuda::std::array<_StrideTIn, _MaxRank> __src_perm_src_strides{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_perm_sizes{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_perm_smem_strides{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __canonical_strides{};
  __canonical_strides[0] = 1;
  for (size_t __i = 1; __i < __rank; ++__i)
  {
    __canonical_strides[__i] = __canonical_strides[__i - 1] * __tile_sizes[__i - 1];
  }
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    const auto __p                = __src_perm[__i];
    __tile_perm_sizes[__i]        = __tile_sizes[__p];
    __src_perm_src_strides[__i]   = __src.__strides[__p];
    __tile_perm_smem_strides[__i] = __canonical_strides[__p];
  }
  for (size_t __i = __rank; __i < _MaxRank; ++__i)
  {
    __tile_perm_sizes[__i] = 1;
  }

  //--------------------------------------------------------------------------------------------------------------------
  // Construct coordinate iterators on the host (precomputed fast modulo/division)
  // namely, given a linear index, compute the multi-dimensional coordinates
  const __tensor_coord_iterator<_ExtentT, _MaxRank> __grid_iter{__grid_tile_sizes}; // grid tile index
  const __tensor_coord_iterator<__tile_extent_t, _MaxRank> __tile_perm_iter{__tile_perm_sizes}; // src -> shared memory
  const __tensor_coord_iterator<__tile_extent_t, _MaxRank> __tile_natural_iter{__tile_sizes}; // shared memory -> dst

  //--------------------------------------------------------------------------------------------------------------------
  // Launch the kernel
  using __value_type            = ::cuda::std::remove_cv_t<_TpIn>;
  const int __thread_block_size = cudax::__find_thread_block_size(__tile_total_size * sizeof(__value_type));

  const auto __config = ::cuda::make_config(
    ::cuda::block_dims(__thread_block_size),
    ::cuda::grid_dims(__grid_size),
    ::cuda::dynamic_shared_memory<__value_type[]>(__tile_total_size));
  const auto __kernel = cudax::__copy_shared_mem_kernel<
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
    __tile_perm_smem_strides,
    __tile_natural_iter,
    __dst.__strides,
    static_cast<int>(__tile_total_size),
    __tile_sizes,
    __dst.__extents,
    __src.__strides);
}

#endif // !_CCCL_COMPILER(NVRTC)
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_COPY_SHARED_MEMORY_H
