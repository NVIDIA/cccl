//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_COPY_BYTES_NAIVE
#define _CUDAX__COPY_BYTES_COPY_BYTES_NAIVE

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
#include <cuda/std/array>

#include <cuda/experimental/__copy/tensor_copy_utils.cuh>
#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy_bytes/types.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
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
  _CCCL_GRID_CONSTANT const _ExtentT __num_items)
{
  const auto __idx    = ::cuda::gpu_thread.rank_as<_ExtentT>(::cuda::grid, __config);
  const auto __stride = ::cuda::gpu_thread.count_as<_ExtentT>(::cuda::grid, __config);
  const __partial_tensor __src{__src_ptr, __src_strides, __src_accessor};
  const __partial_tensor __dst{__dst_ptr, __dst_strides, __dst_accessor};
  for (auto __i = __idx; __i < __num_items; __i += __stride)
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
//! @param[in]  __src_accessor Accessor for reading source elements
//! @param[out] __dst          Destination raw tensor descriptor
//! @param[in]  __dst_accessor Accessor for writing destination elements
//! @param[in]  __num_items    Total number of elements to copy
//! @param[in]  __stream       CUDA stream for asynchronous execution
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
  _ExtentT __num_items,
  ::cuda::stream_ref __stream,
  const _SrcAccessor& __src_accessor = {},
  const _DstAccessor& __dst_accessor = {}) noexcept
{
  constexpr int __block_size = 256;
  if (__num_items == 0)
  {
    return;
  }
  const __tensor_coord_iterator<_ExtentT, _Rank> __coord_iter(__src.__extents);
  const auto __grid_size = ::cuda::ceil_div(__num_items, _ExtentT{__block_size});
  const auto __config    = ::cuda::make_config(::cuda::block_dims<__block_size>(), ::cuda::grid_dims(__grid_size));
  const auto& __kernel   = ::cuda::experimental::__copy_optimized_kernel<
    decltype(__config),
    const _TpIn,
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
    __num_items);
}

template <int _VectorSize>
constexpr auto __const_vector_size = ::cuda::std::integral_constant<int, _VectorSize>{};

template <typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          typename _SrcAccessor,
          typename _DstAccessor,
          ::cuda::std::size_t _Rank>
_CCCL_HOST_API void __copy_vectorized_dispatch(
  const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _Rank>& __src,
  const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _Rank>& __dst,
  ::cuda::std::size_t __vector_size_bytes,
  ::cuda::stream_ref __stream) noexcept
{
  namespace cudax                  = ::cuda::experimental;
  const auto __total_size          = cudax::__total_size(__src);
  const auto __call_copy_optimized = [&](auto __const_vector_size) {
    const auto __src_recast = cudax::__reshape_vectorized<__const_vector_size>(__src);
    const auto __dst_recast = cudax::__reshape_vectorized<__const_vector_size>(__dst);
    cudax::__copy_optimized(__src_recast, __dst_recast, __total_size, __stream);
  };
  if constexpr (sizeof(_TpIn) <= 32)
  {
    if (__vector_size_bytes == 32)
    {
      __call_copy_optimized(__const_vector_size<32>);
    }
  }
  else if constexpr (sizeof(_TpIn) <= 16)
  {
    if (__vector_size == 16)
    {
      __call_copy_optimized(__const_vector_size<16>);
    }
  }
  else if constexpr (sizeof(_TpIn) <= 8)
  {
    if (__vector_size_bytes == 8)
    {
      __call_copy_optimized(__const_vector_size<8>);
    }
  }
  else if constexpr (sizeof(_TpIn) <= 4)
  {
    if (__vector_size_bytes == 4)
    {
      __call_copy_optimized(__const_vector_size<4>);
    }
  }
  else if constexpr (sizeof(_TpIn) <= 2)
  {
    if (__vector_size_bytes == 2)
    {
      __call_copy_optimized(__const_vector_size<2>);
    }
  }
  else
  {
    __call_copy_optimized(__const_vector_size<1>);
  }
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_BYTES_NAIVE
