// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_UNREACHABLE_SENTINEL_H
#define _LIBCUDACXX___ITERATOR_UNREACHABLE_SENTINEL_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/concepts.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_STD
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

struct unreachable_sentinel_t {
  _LIBCUDACXX_TEMPLATE(class _Iter)
    (requires(weakly_incrementable<_Iter>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator==(unreachable_sentinel_t, const _Iter&) noexcept {
    return false;
  }
#if _LIBCUDACXX_STD_VER < 20
  _LIBCUDACXX_TEMPLATE(class _Iter)
    (requires(weakly_incrementable<_Iter>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator==(const _Iter&, unreachable_sentinel_t) noexcept {
    return false;
  }
  _LIBCUDACXX_TEMPLATE(class _Iter)
    (requires(weakly_incrementable<_Iter>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator!=(unreachable_sentinel_t, const _Iter&) noexcept {
    return true;
  }
  _LIBCUDACXX_TEMPLATE(class _Iter)
    (requires(weakly_incrementable<_Iter>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator!=(const _Iter&, unreachable_sentinel_t) noexcept {
    return true;
  }
#endif
};
_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

_LIBCUDACXX_CPO_ACCESSIBILITY unreachable_sentinel_t unreachable_sentinel{};
_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX_STD_VER > 11

#endif // _LIBCUDACXX___ITERATOR_UNREACHABLE_SENTINEL_H
