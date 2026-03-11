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
template <typename T, T a, T c, T m, bool = (m == 0)>
struct static_mod
{
  static const T q = m / a;
  static const T r = m % a;

  _CCCL_HOST_DEVICE T operator()(T x) const
  {
    if constexpr (a == 1)
    {
      x %= m;
    }
    else
    {
      T t1 = a * (x % q);
      T t2 = r * (x / q);
      if (t1 >= t2)
      {
        x = t1 - t2;
      }
      else
      {
        x = m - t2 + t1;
      }
    }

    if constexpr (c != 0)
    {
      const T d = m - x;
      if (d > c)
      {
        x += c;
      }
      else
      {
        x = c - d;
      }
    }

    return x;
  }
}; // end static_mod

// Rely on machine overflow handling
template <typename T, T a, T c, T m>
struct static_mod<T, a, c, m, true>
{
  _CCCL_HOST_DEVICE T operator()(T x) const
  {
    return a * x + c;
  }
}; // end static_mod

template <typename T, T a, T c, T m>
_CCCL_HOST_DEVICE T mod(T x)
{
  static_mod<T, a, c, m> f;
  return f(x);
} // end static_mod
} // namespace random::detail

THRUST_NAMESPACE_END
