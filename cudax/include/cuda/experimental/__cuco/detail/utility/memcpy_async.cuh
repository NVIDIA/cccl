//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_UTILITY_MEMCPY_ASYNC_CUH
#define _CUDAX___CUCO_DETAIL_UTILITY_MEMCPY_ASYNC_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__runtime/api_wrapper.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/types.h>

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::detail
{
//! @brief Enqueues a byte-wise copy of @p __count bytes from @p __src to @p __dst in stream order.
//!
//! Dispatches to `cudaMemcpyBatchAsync` starting with CTK 13.0 to avoid the driver-side locking of the legacy memcpy
//! path, and falls back to `cudaMemcpyAsync` otherwise.
//!
//! @param __dst Destination address
//! @param __src Source address
//! @param __count Number of bytes to copy
//! @param __stream Stream the copy is enqueued into
//!
//! @throws cuda_error if the copy cannot be enqueued
_CCCL_HOST_API inline void
__memcpy_async(void* __dst, const void* __src, ::cuda::std::size_t __count, ::cuda::stream_ref __stream)
{
#if _CCCL_CTK_AT_LEAST(13, 0)
  if (__stream.get() != nullptr)
  {
    void* __dsts[]                        = {__dst};
    const void* __srcs[]                  = {__src};
    const ::cuda::std::size_t __sizes[]   = {__count};
    ::cuda::std::size_t __attrs_indices[] = {0};

    ::cudaMemcpyAttributes __attrs[1]{};
    __attrs[0].srcAccessOrder = ::cudaMemcpySrcAccessOrderStream;
    __attrs[0].flags          = ::cudaMemcpyFlagPreferOverlapWithCompute;

    _CCCL_TRY_CUDA_API(
      ::cudaMemcpyBatchAsync,
      "cuco: failed to enqueue a batched memcpy",
      __dsts,
      __srcs,
      __sizes,
      1,
      __attrs,
      __attrs_indices,
      1,
      __stream.get());
    return;
  }
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyAsync, "cuco: failed to enqueue a memcpy", __dst, __src, __count, ::cudaMemcpyDefault, __stream.get());
}
} // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_UTILITY_MEMCPY_ASYNC_CUH
