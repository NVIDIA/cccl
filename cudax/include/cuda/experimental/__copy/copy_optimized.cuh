//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_OPTIMIZED_H
#define _CUDAX__COPY_OPTIMIZED_H

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

#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy_bytes/types.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Element-wise copy kernel for strided tensor data.
//!
//! Each thread copies one element at a time using a grid-stride loop, mapping linear indices to
//! multi-dimensional coordinates via @ref __tensor_coord_iterator.
//!
//! @param[in]  __config        Kernel launch configuration
//! @param[in]  __src_ptr       Pointer to source data
//! @param[in]  __src_strides   Per-dimension strides for the source tensor
//! @param[in]  __src_accessor  Accessor for reading source elements
//! @param[out] __dst_ptr       Pointer to destination data
//! @param[in]  __dst_strides   Per-dimension strides for the destination tensor
//! @param[in]  __dst_accessor  Accessor for writing destination elements
//! @param[in]  __coord_iter    Coordinate iterator for multi-dimensional index mapping
//! @param[in]  __tensor_size   Total number of elements to copy
template <typename _Config,
          typename _TpSrc,
          typename _TpDst,
          typename _SrcAccessor,
          typename _DstAccessor,
          typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          ::cuda::std::size_t _Rank>
__global__ void __copy_optimized_kernel(
  _CCCL_GRID_CONSTANT const _Config __config,
  _CCCL_GRID_CONSTANT const _TpSrc* const _CCCL_RESTRICT __src_ptr,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTIn, _Rank> __src_strides,
  _CCCL_GRID_CONSTANT const _SrcAccessor __src_accessor,
  _CCCL_GRID_CONSTANT _TpDst* const _CCCL_RESTRICT __dst_ptr,
  _CCCL_GRID_CONSTANT const ::cuda::std::array<_StrideTOut, _Rank> __dst_strides,
  _CCCL_GRID_CONSTANT const _DstAccessor __dst_accessor,
  _CCCL_GRID_CONSTANT const __tensor_coord_iterator<_ExtentT, _Rank> __coord_iter,
  _CCCL_GRID_CONSTANT const _ExtentT __tensor_size)
{
  using __partial_tensor_src = __partial_tensor<const _TpSrc, _StrideTIn, _Rank, _SrcAccessor>;
  using __partial_tensor_dst = __partial_tensor<_TpDst, _StrideTOut, _Rank, _DstAccessor>;
  const auto __idx           = ::cuda::gpu_thread.rank_as<_ExtentT>(::cuda::grid, __config);
  const auto __stride        = ::cuda::gpu_thread.count_as<_ExtentT>(::cuda::grid, __config);
  const __partial_tensor_src __src{__src_ptr, __src_strides, __src_accessor};
  const __partial_tensor_dst __dst{__dst_ptr, __dst_strides, __dst_accessor};

  for (auto __i = __idx; __i < __tensor_size; __i += __stride)
  {
    const auto __coord = __coord_iter(__i);
    __dst(__coord)     = __src(__coord);
  }
}

//! @brief Launch a naive element-wise copy kernel for strided tensor data.
//!
//! Each thread copies one element at a time using a grid-stride loop. Coordinates are
//! computed from linear indices via @ref __tensor_coord_iterator.
//!
//! @param[in]  __src          Source raw tensor descriptor
//! @param[out] __dst          Destination raw tensor descriptor
//! @param[in]  __tensor_size  Total number of elements to copy
//! @param[in]  __stream       CUDA stream for asynchronous execution
//! @param[in]  __src_accessor Accessor for reading source elements
//! @param[in]  __dst_accessor Accessor for writing destination elements
template <typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _Rank,
          typename _SrcAccessor = ::cuda::std::default_accessor<_TpIn>,
          typename _DstAccessor = ::cuda::std::default_accessor<_TpOut>>
_CCCL_HOST_API void __copy_optimized(
  const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _Rank>& __src,
  const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _Rank>& __dst,
  _ExtentT __tensor_size,
  ::cuda::stream_ref __stream,
  const _SrcAccessor& __src_accessor = {},
  const _DstAccessor& __dst_accessor = {}) noexcept
{
  constexpr int __block_size = 256;
  const __tensor_coord_iterator<_ExtentT, _Rank> __coord_iter(__src.__extents);
  const auto __grid_size = ::cuda::ceil_div(__tensor_size, _ExtentT{__block_size});
  const auto __config    = ::cuda::make_config(::cuda::block_dims<__block_size>(), ::cuda::grid_dims(__grid_size));
  const auto& __kernel   = ::cuda::experimental::__copy_optimized_kernel<
      decltype(__config),
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
    __tensor_size);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_OPTIMIZED_H
