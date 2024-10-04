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

#ifndef _LIBCUDACXX___MEMORY_ASSUME_ALIGNED_H
#define _LIBCUDACXX___MEMORY_ASSUME_ALIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// MSVC provides __builtin_assume_aligned only for version >= 1923 and needs <memory> header
#if defined(_CCCL_COMPILER_MSVC) && _CCCL_MSVC_VERSION >= 1923 && __has_include(<memory>)
#  include <memory>
#endif

#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/cstddef> // size_t
#include <cuda/std/cstdint> // uintptr_t

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <_CUDA_VSTD::size_t _Align, class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp* assume_aligned(_Tp* __ptr) noexcept
{
  static_assert(_CUDA_VSTD::has_single_bit(_Align));
  if (_CUDA_VSTD::__libcpp_is_constant_evaluated())
  {
    return __ptr;
  }
  else
  {
    _CCCL_ASSERT(reinterpret_cast<_CUDA_VSTD::uintptr_t>(__ptr) % _Align == 0, "Alignment assumption is violated");
#if defined(_CCCL_BUILTIN_ASSUME_ALIGNED)
    return static_cast<_Tp*>(__builtin_assume_aligned(__ptr, _Align));
#else // ^^^ _CCCL_BUILTIN_ASSUME_ALIGNED ^^^ / vvv !_CCCL_BUILTIN_ASSUME_ALIGNED vvv
    return __ptr;
#endif // !_CCCL_BUILTIN_ASSUME_ALIGNED
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MEMORY_ASSUME_ALIGNED_H
