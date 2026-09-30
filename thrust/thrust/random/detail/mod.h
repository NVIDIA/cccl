// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

THRUST_NAMESPACE_BEGIN

namespace random::detail
{
template <typename T, T A, T C, T M, bool = (M == 0)>
struct static_mod
{
  static const T q = M / A;
  static const T r = M % A;

  _CCCL_HOST_DEVICE T operator()(T x) const
  {
    if constexpr (A == 1)
    {
      x %= M;
    }
    else
    {
      T t1 = A * (x % q);
      T t2 = r * (x / q);
      if (t1 >= t2)
      {
        x = t1 - t2;
      }
      else
      {
        x = M - t2 + t1;
      }
    }

    if constexpr (C != 0)
    {
      const T d = M - x;
      if (d > C)
      {
        x += C;
      }
      else
      {
        x = C - d;
      }
    }

    return x;
  }
}; // end static_mod

// Rely on machine overflow handling
template <typename T, T A, T C, T M>
struct static_mod<T, A, C, M, true>
{
  _CCCL_HOST_DEVICE T operator()(T x) const
  {
    return A * x + C;
  }
}; // end static_mod

template <typename T, T A, T C, T M>
_CCCL_HOST_DEVICE T mod(T x)
{
  const static_mod<T, A, C, M> f;
  return f(x);
} // end static_mod
} // namespace random::detail

THRUST_NAMESPACE_END
