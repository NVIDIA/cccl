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

template <class _Tp, class... _Properties>
struct __allocation_info
{
  using __resource_t = ::cuda::mr::any_resource<_Properties...>;
  using __pointer_t  = _Tp*;
  using __size_type  = ::cuda::std::size_t;

  ::cuda::stream_ref stream;
  __resource_t mr;
  __pointer_t ptr       = nullptr;
  __size_type capacity  = 0;
  __size_type size      = 0;
  __size_type alignment = alignof(_Tp);
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //__CUDAX__CONTAINERS_ALLOCATION_INFO_H
