//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CONTAINER_MAKE_BUFFER_WITH_POOL_H
#define _CUDA___CONTAINER_MAKE_BUFFER_WITH_POOL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__container/buffer.h>
#  include <cuda/__memory_pool/device_memory_pool.h>

#  if _CCCL_CTK_AT_LEAST(12, 9)
#    include <cuda/__memory_pool/pinned_memory_pool.h>
#  endif // _CCCL_CTK_AT_LEAST(12, 9)

#  if _CCCL_CTK_AT_LEAST(13, 0)
#    include <cuda/__memory_pool/managed_memory_pool.h>
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Creates a buffer backed by the default device memory pool.
//! @param __stream The stream used for allocation.
//! @param __device The device whose default memory pool will be used.
//! @param __args Remaining arguments forwarded to `make_buffer`.
//! @see make_buffer for the full set of supported argument combinations.
template <class _Tp, class... _Args>
_CCCL_HOST_API auto make_device_buffer(stream_ref __stream, device_ref __device, _Args&&... __args)
{
  return ::cuda::make_buffer<_Tp>(
    __stream, ::cuda::device_default_memory_pool(__device), ::cuda::std::forward<_Args>(__args)...);
}

#  if _CCCL_CTK_AT_LEAST(12, 9)

//! @brief Creates a buffer backed by the default pinned memory pool.
//! @param __stream The stream used for allocation.
//! @param __args Remaining arguments forwarded to `make_buffer`.
//! @see make_buffer for the full set of supported argument combinations.
template <class _Tp, class... _Args>
_CCCL_HOST_API auto make_pinned_buffer(stream_ref __stream, _Args&&... __args)
{
  return ::cuda::make_buffer<_Tp>(
    __stream, ::cuda::pinned_default_memory_pool(), ::cuda::std::forward<_Args>(__args)...);
}

#  endif // _CCCL_CTK_AT_LEAST(12, 9)

#  if _CCCL_CTK_AT_LEAST(13, 0)

//! @brief Creates a buffer backed by the default managed memory pool.
//! @param __stream The stream used for allocation.
//! @param __args Remaining arguments forwarded to `make_buffer`.
//! @see make_buffer for the full set of supported argument combinations.
template <class _Tp, class... _Args>
_CCCL_HOST_API auto make_managed_buffer(stream_ref __stream, _Args&&... __args)
{
  return ::cuda::make_buffer<_Tp>(
    __stream, ::cuda::managed_default_memory_pool(), ::cuda::std::forward<_Args>(__args)...);
}

#  endif // _CCCL_CTK_AT_LEAST(13, 0)

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___CONTAINER_MAKE_BUFFER_WITH_POOL_H
