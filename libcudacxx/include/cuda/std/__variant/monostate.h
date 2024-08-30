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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/__compare/ordering.h>
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include <cuda/std/__functional/hash.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT monostate
{};

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator==(monostate, monostate) noexcept
{
  return true;
}

#if _CCCL_STD_VER < 2020

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator!=(monostate, monostate) noexcept
{
  return false;
}

#endif // _CCCL_STD_VER < 2020

#if _CCCL_STD_VER >= 2020 && !defined(_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR)

_LIBCUDACXX_HIDE_FROM_ABI constexpr strong_ordering operator<=>(monostate, monostate) noexcept
{
  return strong_ordering::equal;
}

#else // _CCCL_STD_VER >= 2020

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator<(monostate, monostate) noexcept
{
  return false;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator>(monostate, monostate) noexcept
{
  return false;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator<=(monostate, monostate) noexcept
{
  return true;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator>=(monostate, monostate) noexcept
{
  return true;
}

#endif // _CCCL_STD_VER >= 2020

#ifndef __cuda_std__
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT hash<monostate>
{
  using argument_type = monostate;
  using result_type   = size_t;

  _LIBCUDACXX_HIDE_FROM_ABI result_type operator()(const argument_type&) const noexcept
  {
    return 66740831; // return a fundamentally attractive random value.
  }
};
#endif // __cuda_std__

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___VARIANT_MONOSTATE_H
