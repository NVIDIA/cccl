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
#include <cuda/launch>

#include <cuda/experimental/__copy_bytes/cute_utils.cuh>

#include <cute/tensor.hpp>
//
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Naive element-by-element copy kernel.
//!
//! @param[in]  __config    Kernel launch configuration
//! @param[in]  __src       Source tensor
//! @param[out] __dst       Destination tensor
//! @param[in]  __num_items Number of elements to copy
template <typename _Config, typename _SrcTensor, typename _DstTensor>
__global__ void __copy_bytes_naive_kernel(_Config __config, _SrcTensor __src, _DstTensor __dst, int __num_items)
{
  const auto __idx    = ::cuda::gpu_thread.rank(::cuda::grid, __config);
  const auto __stride = ::cuda::gpu_thread.count(::cuda::grid, __config);
  for (auto __i = __idx; __i < __num_items; __i += __stride)
  {
    __dst(__i) = __src(__i);
  }
}

//! @brief Launch a naive fallback copy kernel for device-to-device tensor copies.
//!
//! Each thread copies one element at a time using a grid-stride loop.
//!
//! @param[in]  __src        Pointer to source data
//! @param[in]  __src_layout CuTe layout of the source tensor
//! @param[out] __dst        Pointer to destination data
//! @param[in]  __dst_layout CuTe layout of the destination tensor
//! @param[in]  __stream     CUDA stream for asynchronous execution
template <typename _Tp, typename _SrcLayout, typename _DstLayout>
_CCCL_HOST_API void copy_bytes_naive(
  const _Tp* __src,
  const _SrcLayout& __src_layout,
  _Tp* __dst,
  const _DstLayout& __dst_layout,
  ::cuda::stream_ref __stream)
{
  constexpr int __block_size = 256;
  const auto __num_items     = static_cast<int>(::cute::size(__src_layout));
  if (__num_items == 0)
  {
    return;
  }
  const auto __src_tensor = ::cuda::experimental::__make_gmem_tensor(__src, __src_layout);
  const auto __dst_tensor = ::cuda::experimental::__make_gmem_tensor(__dst, __dst_layout);
  const auto __grid_size  = ::cuda::ceil_div(__num_items, __block_size);
  const auto __config     = ::cuda::make_config(::cuda::block_dims<__block_size>(), ::cuda::grid_dims(__grid_size));
  const auto& __kernel =
    ::cuda::experimental::__copy_bytes_naive_kernel<decltype(__config), decltype(__src_tensor), decltype(__dst_tensor)>;
  ::cuda::launch(__stream, __config, __kernel, __src_tensor, __dst_tensor, __num_items);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_COPY_BYTES_NAIVE
