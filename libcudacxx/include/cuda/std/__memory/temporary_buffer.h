// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_TEMPORARY_BUFFER_H
#define _LIBCUDACXX___MEMORY_TEMPORARY_BUFFER_H

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
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__new_>
#include <cuda/std/__type_traits/alignment_of.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstddef>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_NO_CFI _LIBCUDACXX_HIDE_FROM_ABI pair<_Tp*, ptrdiff_t>
get_temporary_buffer(ptrdiff_t __n) noexcept
{
  pair<_Tp*, ptrdiff_t> __r(0, 0);
  const ptrdiff_t __m =
    (~ptrdiff_t(0) ^ ptrdiff_t(ptrdiff_t(1) << (sizeof(ptrdiff_t) * __CHAR_BIT__ - 1))) / sizeof(_Tp);
  if (__n > __m)
  {
    __n = __m;
  }
  while (__n > 0)
  {
#if !defined(_LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION)
    if (__is_overaligned_for_new(_LIBCUDACXX_ALIGNOF(_Tp)))
    {
      _CUDA_VSTD::align_val_t __al = _CUDA_VSTD::align_val_t(_CUDA_VSTD::alignment_of<_Tp>::value);
      __r.first                    = static_cast<_Tp*>(::operator new(__n * sizeof(_Tp), __al));
    }
    else
    {
      __r.first = static_cast<_Tp*>(::operator new(__n * sizeof(_Tp)));
    }
#else // ^^^ !_LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION ^^^ / vvv _LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION vvv
    if (__is_overaligned_for_new(_LIBCUDACXX_ALIGNOF(_Tp)))
    {
      // Since aligned operator new is unavailable, return an empty
      // buffer rather than one with invalid alignment.
      return __r;
    }

    __r.first = static_cast<_Tp*>(::operator new(__n * sizeof(_Tp)));
#endif // _LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION

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
_LIBCUDACXX_HIDE_FROM_ABI void return_temporary_buffer(_Tp* __p) noexcept
{
  _CUDA_VSTD::__cccl_deallocate_unsized((void*) __p, _LIBCUDACXX_ALIGNOF(_Tp));
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MEMORY_TEMPORARY_BUFFER_H
