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

#include <cub/device/dispatch/tuning/tuning_transform.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__device/all_devices.h>
#include <cuda/__device/arch_id.h>
#include <cuda/__device/arch_traits.h>
#include <cuda/__launch/configuration.h>
#include <cuda/__launch/launch.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/array>

#include <cuda/experimental/__copy/tensor_copy_utils.cuh>
#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy_bytes/types.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Tiled copy kernel for contiguous innermost dimension.
//!
//! Uses a 2D grid: blockIdx.x = tile along inner dimension, blockIdx.y = outer index.
//! Threads within a block stride over the tile, reading from the source and writing to the
//! destination via accessors. The coordinate iterator maps linear indices to multi-dimensional
//! coordinates, which are then used with per-tensor strides for the actual memory access.
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
  _CCCL_GRID_CONSTANT const _ExtentT __inner_size)
{
  using __partial_tensor_src  = __partial_tensor<const _TpSrc, _StrideTIn, _Rank, _SrcAccessor>;
  using __partial_tensor_dst  = __partial_tensor<_TpDst, _StrideTOut, _Rank, _DstAccessor>;
  const auto __thread_id      = ::cuda::gpu_thread.rank_as<_ExtentT>(::cuda::block, __config);
  const auto __block_idx      = ::cuda::block.index_as<_ExtentT>(::cuda::grid);
  constexpr auto __block_size = ::cuda::gpu_thread.count_as<int>(::cuda::block, __config);
  const __partial_tensor_src __src{__src_ptr, __src_strides, __src_accessor};
  const __partial_tensor_dst __dst{__dst_ptr, __dst_strides, __dst_accessor};

  const auto __tile_offset = __block_idx.x * _TileSize;
  const auto __outer_idx   = __block_idx.y;
  const auto __remaining   = __inner_size - __tile_offset;
  const auto __base_idx    = __outer_idx * __inner_size + __tile_offset + __thread_id;

  if (__remaining >= _TileSize)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _TileSize; __i += __block_size)
    {
      const auto __coord = __coord_iter(__base_idx + __i);
      __dst(__coord)     = __src(__coord);
    }
  }
  else
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _TileSize; __i += __block_size)
    {
      if (__thread_id + __i < __remaining)
      {
        const auto __coord = __coord_iter(__base_idx + __i);
        __dst(__coord)     = __src(__coord);
      }
    }
  }
}

//! @brief Query the minimum bytes-in-flight target for the current GPU architecture.
//!
//! Delegates to CUB's architecture-specific tuning.
//! @return Bytes-in-flight target (e.g. 12KB for V100, 16KB for A100, 48KB for H200, 64KB for B200)
[[nodiscard]] _CCCL_HOST_API inline int __bytes_in_flight() noexcept
{
  const auto __dev_id = ::cuda::__driver::__cudevice_to_ordinal(::cuda::__driver::__ctxGetDevice());
  const auto __dev    = ::cuda::devices[__dev_id];
  const auto __cc     = ::cuda::device_attributes::compute_capability(__dev);
  return CUB_NS_QUALIFIER::detail::transform::cc_to_min_bytes_in_flight(__cc);
}

// Compute the number of elements each thread copies for a given vector width.
[[nodiscard]] _CCCL_HOST_API inline int __elem_per_thread(int __access_bytes, int __bytes_in_flight) noexcept
{
  constexpr auto __threads_per_sm = 2048;
  return ::cuda::std::max(__bytes_in_flight / (__access_bytes * __threads_per_sm), 1);
}

// Dispatch a callable with a compile-time tile size derived from a runtime value.
template <typename _Op>
_CCCL_HOST_API void __dispatch_tile_size(int __tile_size, _Op __op) noexcept
{
  if (__tile_size >= 2048)
  {
    __op(::cuda::std::integral_constant<int, 2048>{});
  }
  else if (__tile_size >= 1024)
  {
    __op(::cuda::std::integral_constant<int, 1024>{});
  }
  else if (__tile_size >= 512)
  {
    __op(::cuda::std::integral_constant<int, 512>{});
  }
  else
  {
    __op(::cuda::std::integral_constant<int, 256>{});
  }
}

//! @brief Launch the tiled copy kernel for contiguous innermost dimension.
//!
//! Computes tile size from the architecture-specific bytes-in-flight target, then dispatches
//! the @ref __copy_contiguous_kernel with a compile-time tile size.
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
  constexpr int __block_size   = 256;
  const auto __bytes_in_flight = ::cuda::experimental::__bytes_in_flight();
  const auto __elems_per_thread =
    ::cuda::experimental::__elem_per_thread(static_cast<int>(sizeof(_TpIn)), __bytes_in_flight);
  const auto __tile_size_rt = __block_size * __elems_per_thread;

  ::cuda::experimental::__dispatch_tile_size(__tile_size_rt, [&](auto __tile_constant) {
    constexpr int __tile_size    = decltype(__tile_constant)::value;
    const auto __inner_size      = __src.__extents[0];
    const auto __outer_size      = ::cuda::experimental::__total_size(__src) / __inner_size;
    const auto __num_inner_tiles = ::cuda::ceil_div(__inner_size, __tile_size);
    constexpr auto __arch_limits = ::cuda::__common_arch_traits(::cuda::arch_id::sm_90);
    _CCCL_ASSERT(__num_inner_tiles <= _ExtentT(__arch_limits.max_grid_dim_x),
                 "grid x-dimension exceeds the maximum grid size");
    _CCCL_ASSERT(__outer_size <= _ExtentT(__arch_limits.max_grid_dim_y),
                 "grid y-dimension exceeds the maximum grid size");
    const auto __grid_dims = ::dim3(static_cast<unsigned>(__num_inner_tiles), static_cast<unsigned>(__outer_size));
    const auto __config    = ::cuda::make_config(::cuda::block_dims<__block_size>(), ::cuda::grid_dims(__grid_dims));

    const __tensor_coord_iterator<_ExtentT, _Rank> __coord_iter{__src.__extents};
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
      __inner_size);
  });
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_CONTIGUOUS_H
