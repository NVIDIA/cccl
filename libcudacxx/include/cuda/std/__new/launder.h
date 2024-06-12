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

#ifndef _LIBCUDACXX___NEW_LAUNDER_H
#define _LIBCUDACXX___NEW_LAUNDER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_CCCL_NODISCARD inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 _Tp*
launder(_Tp* __p) noexcept
{
  static_assert(!_CCCL_TRAIT(is_function, _Tp), "can't launder functions");
  static_assert(!_CCCL_TRAIT(is_same, void, __remove_cv_t<_Tp>), "can't launder cv-void");
#if defined(_LIBCUDACXX_LAUNDER)
  return _LIBCUDACXX_LAUNDER(__p);
#else
  return __p;
#endif // defined(_LIBCUDACXX_LAUNDER)
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NEW_LAUNDER_H
