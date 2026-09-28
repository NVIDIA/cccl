//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_AS_UNINITIALIZED_BYTES_H
#define _CUDA___MEMORY_AS_UNINITIALIZED_BYTES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <::cuda::std::size_t _SizeOfT, ::cuda::std::size_t _AlignOfT>
struct __uninitialized_bytes_t
{
  alignas(_AlignOfT) unsigned char __bytes[_SizeOfT];

  static_assert(_SizeOfT % _AlignOfT == 0, "Storage size must be a multiple of its alignment");

  template <class _Tp>
  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE _Tp& __alias() noexcept
  {
    static_assert(sizeof(_Tp) <= _SizeOfT, "T does not fit in uninitialized storage");
    static_assert(alignof(_Tp) <= _AlignOfT, "T is more aligned than uninitialized storage");
    return reinterpret_cast<_Tp&>(*this);
  }
};

template <class _Tp>
using __as_uninitialized_bytes = __uninitialized_bytes_t<sizeof(_Tp), alignof(_Tp)>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_AS_UNINITIALIZED_BYTES_H
