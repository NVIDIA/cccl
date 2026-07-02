// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024-25 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MEMORY_TEMPORARY_BUFFER_H
#define _CUDA_STD___MEMORY_TEMPORARY_BUFFER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__new/allocate.h>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/__type_traits/alignment_of.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/climits>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]] _CCCL_NO_CFI _CCCL_API inline pair<_Tp*, ptrdiff_t> get_temporary_buffer(ptrdiff_t __n) noexcept
{
  pair<_Tp*, ptrdiff_t> __r(0, 0);
  const ptrdiff_t __m = (~ptrdiff_t(0) ^ ptrdiff_t(ptrdiff_t(1) << (sizeof(ptrdiff_t) * CHAR_BIT - 1))) / sizeof(_Tp);
  if (__n > __m)
  {
    __n = __m;
  }
  while (__n > 0)
  {
    if constexpr (alignof(_Tp) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
      __r.first = static_cast<_Tp*>(::cuda::std::__cccl_operator_new(__n * sizeof(_Tp), align_val_t{alignof(_Tp)}));
    }
    else
    {
      __r.first = static_cast<_Tp*>(::cuda::std::__cccl_operator_new(__n * sizeof(_Tp)));
    }

    if (__r.first)
    {
      __r.second = __n;
      break;
    }
    __n /= 2;
  }
  return __r;
}

template <class _Tp>
_CCCL_API inline void return_temporary_buffer(_Tp* __p) noexcept
{
  ::cuda::std::__cccl_deallocate_unsized((void*) __p, alignof(_Tp));
}

struct __return_temporary_buffer
{
  template <class _Tp>
  _CCCL_API void operator()(_Tp* __p) const noexcept
  {
    ::cuda::std::return_temporary_buffer(__p);
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_TEMPORARY_BUFFER_H
