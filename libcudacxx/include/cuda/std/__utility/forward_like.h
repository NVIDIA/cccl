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
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_reference.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Ap, class _Bp>
using _CopyConst = _If<_CCCL_TRAIT(is_const, _Ap), const _Bp, _Bp>;

template <class _Ap, class _Bp>
using _OverrideRef = _If<_CCCL_TRAIT(is_rvalue_reference, _Ap), remove_reference_t<_Bp>&&, _Bp&>;

template <class _Ap, class _Bp>
using _ForwardLike = _OverrideRef<_Ap&&, _CopyConst<remove_reference_t<_Ap>, remove_reference_t<_Bp>>>;

template <class _Tp, class _Up>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto forward_like(_Up&& __ux) noexcept -> _ForwardLike<_Tp, _Up>
{
  return static_cast<_ForwardLike<_Tp, _Up>>(__ux);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_FORWARD_LIKE_H
