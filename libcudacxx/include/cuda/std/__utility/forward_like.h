// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_FORWARD_LIKE_H
#define _LIBCUDACXX___UTILITY_FORWARD_LIKE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <utility> // IWYU pragma: keep

#if (_CCCL_COMPILER(CLANG, >=, 17) || _CCCL_COMPILER(GCC, >=, 15) || _CCCL_COMPILER(MSVC, >=, 19, 36)) \
  && defined(__cpp_lib_forward_like) && (__cpp_lib_forward_like >= 202217L) && !defined(__CUDA_ARCH__)
#  define _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE() 1
#else
#  define _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE() 0
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE()

// The compiler treats ::std::forward_like as a builtin function so it does not need to be
// instantiated and will be compiled away even at -O0.
using ::std::forward_like;

#else // ^^^ _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE() ^^^ / vvv !_CCCL_HAS_BUILTIN_STD_FORWARD_LIKE() vvv

template <class _Ap, class _Bp>
using _ForwardLike = __copy_cvref_t<_Ap&&, remove_reference_t<_Bp>>;

template <class _Tp, class _Up>
[[nodiscard]] _CCCL_INTRINSIC _LIBCUDACXX_HIDE_FROM_ABI constexpr auto forward_like(_Up&& __ux) noexcept
  -> _ForwardLike<_Tp, _Up>
{
  return static_cast<_ForwardLike<_Tp, _Up>>(__ux);
}

#endif // _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE()

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_FORWARD_LIKE_H
