//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINERS_ALLOCATION_INFO_H
#define __CUDAX__CONTAINERS_ALLOCATION_INFO_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__memory_resource/any_resource.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__cstddef/types.h>

#  include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c __allocation_info class provides information about a memory
//! allocation.
_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Struct containing allocation info used to acquire a buffer.
//! Returned when a buffer is released.
template <class _Tp, class... _Properties>
struct __allocation_info
{
  using __resource_type _CCCL_NODEBUG = ::cuda::mr::any_resource<_Properties...>;
  using __pointer_type _CCCL_NODEBUG  = _Tp*;
  using __size_type _CCCL_NODEBUG     = ::cuda::std::size_t;

  ::cuda::stream_ref stream;
  __resource_type mr;
  __pointer_type ptr;
  __size_type capacity;
  __size_type size;
  __size_type alignment;

  _CCCL_HOST_API __allocation_info(
    ::cuda::stream_ref __stream,
    __resource_type __mr,
    __pointer_type __ptr,
    __size_type __capacity,
    __size_type __size,
    __size_type __alignment) noexcept
      : stream(__stream)
      , mr(::cuda::std::move(__mr))
      , ptr(__ptr)
      , capacity(__capacity)
      , size(__size)
      , alignment(__alignment)
  {}
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //__CUDAX__CONTAINERS_ALLOCATION_INFO_H
