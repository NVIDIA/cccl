//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_CONTIGUOUS_H
#define _CUDAX__COPY_CONTIGUOUS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/launch>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/array>

#include <cuda/experimental/__copy/tensor_copy_utils.cuh>
#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy_bytes/types.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Tiled copy kernel for contiguous innermost dimension.
//!
//! Each block processes one tile of the innermost dimension for one outer index. Threads within a
//! block stride over the tile, reading from the source and writing to the destination via accessors.
//!
//! @param[in]  __config          Kernel launch configuration
//! @param[in]  __src_ptr         Pointer to source data
//! @param[in]  __src_strides     Per-dimension strides for the source tensor
//! @param[in]  __src_accessor    Accessor for reading source elements
//! @param[out] __dst_ptr         Pointer to destination data
//! @param[in]  __dst_strides     Per-dimension strides for the destination tensor
//! @param[in]  __dst_accessor    Accessor for writing destination elements
//! @param[in]  __coord_iter      Coordinate iterator for multi-dimensional index mapping
//! @param[in]  __inner_size      Extent of the contiguous innermost dimension
//! @param[in]  __num_inner_tiles Number of tiles along the innermost dimension
template <typename _Config,
          int _TileSize,
          typename _TpSrc,
          typename _TpDst,
          typename _SrcAccessor,
          typename _DstAccessor,
          typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          ::cuda::std::size_t _Rank>
__global__ void __copy_contiguous_kernel(
  _CCCL_GRID_CONSTANT const _Config __config,
  _CCCL_GRID_CONSTANT const _TpSrc* const _CCCL_RESTRICT __src_ptr,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTIn, _Rank> __src_strides,
  _CCCL_GRID_CONSTANT const _SrcAccessor __src_accessor,
  _CCCL_GRID_CONSTANT _TpDst* const _CCCL_RESTRICT __dst_ptr,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTOut, _Rank> __dst_strides,
  _CCCL_GRID_CONSTANT const _DstAccessor __dst_accessor,
  _CCCL_GRID_CONSTANT const __tensor_coord_iterator<_ExtentT, _Rank> __coord_iter,
  _CCCL_GRID_CONSTANT const _ExtentT __inner_size,
  _CCCL_GRID_CONSTANT const int __num_inner_tiles)
{
  const auto __thread_id      = ::cuda::gpu_thread.rank_as<_ExtentT>(::cuda::grid, __config);
  const auto __stride         = ::cuda::gpu_thread.count_as<_ExtentT>(::cuda::grid, __config);
  const auto __block_id       = ::cuda::block.rank_as<int>(::cuda::grid, __config);
  constexpr auto __block_size = ::cuda::gpu_thread.count(::cuda::block, __config);
  __partial_tensor __src{__src_ptr, __src_strides, __src_accessor};
  __partial_tensor __dst{__dst_ptr, __dst_strides, __dst_accessor};

  const int __tile_idx    = __block_id % __num_inner_tiles;
  const int __outer_idx   = __block_id / __num_inner_tiles;
  const int __tile_offset = __tile_idx * _TileSize;
  const int __flat_offset = __tile_offset + __outer_idx * __inner_size;
  const int __remaining   = __inner_size - __tile_offset;

  const auto __thr_offset = __flat_offset + __thread_id;
  __src.__ptr += __thr_offset;
  __dst.__ptr += __thr_offset;
  if (__remaining >= _TileSize)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _TileSize; __i += __block_size)
    {
      const auto __coord = __coord_iter(__i);
      __dst(__coord)     = __src(__coord);
    }
  }
  else
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _TileSize; __i += __block_size)
    {
      if (__thr_offset + __i < __remaining)
      {
        const auto __coord = __coord_iter(__i);
        __dst(__coord)     = __src(__coord);
      }
    }
  }
}

inline constexpr int __bytes_in_flight = 64 * 1024; // 64KB

//! @brief Compute the number of elements each thread copies for a given vector width.
//!
//! @param[in] __access_bytes Size in bytes of one vectorized access
//! @return Number of elements per thread
[[nodiscard]] _CCCL_HOST_API constexpr int __elem_per_thread(int __access_bytes) noexcept
{
  constexpr auto __threads_per_sm = 2048; // 2048 threads per SM
  return (__bytes_in_flight / __access_bytes) / __threads_per_sm;
}

//! @brief Launch the tiled copy kernel for contiguous innermost dimension.
//!
//! Computes tile size and inner/outer dimensions from the raw tensor descriptors, then dispatches
//! the @ref __copy_contiguous_kernel.
//!
//! @param[in] __src          Source raw tensor descriptor
//! @param[in] __dst          Destination raw tensor descriptor
//! @param[in] __stream       CUDA stream for asynchronous execution
//! @param[in] __src_accessor Accessor for reading source elements
//! @param[in] __dst_accessor Accessor for writing destination elements
template <typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _Rank,
          typename _SrcAccessor = ::cuda::std::default_accessor<_TpIn>,
          typename _DstAccessor = ::cuda::std::default_accessor<_TpOut>>
_CCCL_HOST_API void __launch_copy_contiguous_kernel(
  const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _Rank>& __src,
  const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _Rank>& __dst,
  ::cuda::stream_ref __stream,
  const _SrcAccessor& __src_accessor = {},
  const _DstAccessor& __dst_accessor = {})
{
  namespace cudax                    = ::cuda::experimental;
  constexpr int __block_size         = 256;
  constexpr auto __elems_per_thread1 = cudax::__elem_per_thread(sizeof(_TpIn));
  constexpr auto __tile_size         = __block_size * __elems_per_thread1;
  const auto __inner_size            = __src.__extents[0];
  const auto __outer_size            = cudax::__total_size(__src) / __inner_size;
  const auto __num_inner_tiles       = ::cuda::ceil_div(__inner_size, __tile_size);
  const auto __grid_size             = __num_inner_tiles * __outer_size;
  const auto __config = ::cuda::make_config(::cuda::block_dims<__block_size>(), ::cuda::grid_dims(__grid_size));

  const __tensor_coord_iterator<_ExtentT, _Rank> __coord_iter(__src.__extents);
  const auto __kernel = ::cuda::experimental::__copy_contiguous_kernel<
    decltype(__config),
    __tile_size,
    _TpIn,
    _TpOut,
    _SrcAccessor,
    _DstAccessor,
    _ExtentT,
    _StrideTIn,
    _StrideTOut,
    _Rank>;

  ::cuda::launch(
    __stream,
    __config,
    __kernel,
    __src.__data,
    __src.__strides,
    __src_accessor,
    __dst.__data,
    __dst.__strides,
    __dst_accessor,
    __coord_iter,
    __inner_size,
    __num_inner_tiles);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_CONTIGUOUS_H
