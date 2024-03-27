// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___VARIANT_MONOSTATE_H
#define _LIBCUDACXX___VARIANT_MONOSTATE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include "../__compare/ordering.h"
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include "../__functional/hash.h"

#include "../cstddef"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER >= 2011

struct _LIBCUDACXX_TEMPLATE_VIS monostate {};

_LIBCUDACXX_INLINE_VISIBILITY constexpr bool operator==(monostate, monostate) noexcept { return true; }

#if _CCCL_STD_VER < 2020

_LIBCUDACXX_INLINE_VISIBILITY constexpr bool operator!=(monostate, monostate) noexcept { return false; }

#endif // _CCCL_STD_VER < 2020

#if _CCCL_STD_VER >= 2020 && !defined(_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR)

_LIBCUDACXX_INLINE_VISIBILITY constexpr strong_ordering operator<=>(monostate, monostate) noexcept {
  return strong_ordering::equal;
}

#else // _CCCL_STD_VER >= 2020

_LIBCUDACXX_INLINE_VISIBILITY constexpr bool operator<(monostate, monostate) noexcept { return false; }

_LIBCUDACXX_INLINE_VISIBILITY constexpr bool operator>(monostate, monostate) noexcept { return false; }

_LIBCUDACXX_INLINE_VISIBILITY constexpr bool operator<=(monostate, monostate) noexcept { return true; }

_LIBCUDACXX_INLINE_VISIBILITY constexpr bool operator>=(monostate, monostate) noexcept { return true; }

#  endif // _CCCL_STD_VER >= 2020

#ifndef __cuda_std__
template <>
struct _LIBCUDACXX_TEMPLATE_VIS hash<monostate> {
  using argument_type = monostate;
  using result_type = size_t;

  inline _LIBCUDACXX_INLINE_VISIBILITY result_type operator()(const argument_type&) const noexcept {
    return 66740831; // return a fundamentally attractive random value.
  }
};
#endif // __cuda_std__

#endif // _CCCL_STD_VER >= 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___VARIANT_MONOSTATE_H
