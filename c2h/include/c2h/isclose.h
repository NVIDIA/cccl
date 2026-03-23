// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

template <typename T>
bool isclose(T a, T b, T r_tol, T a_tol)
{
  if constexpr (std::is_floating_point_v<T>)
  {
    if (a == b)
    {
      return true;
    }
    return std::abs(a - b) <= std::max(a_tol, r_tol * std::max(std::abs(a), std::abs(b)));
  }
  else
  {
    static_assert(std::is_integral_v<T>, "isclose: unsupported type, expected floating point or integral");
    return a == b;
  }
}

template <typename T>
bool isclose(T a, T b, T r_tol)
{
  return isclose(a, b, r_tol, T(0));
}

template <typename T>
bool isclose(T a, T b)
{
  if constexpr (std::is_floating_point_v<T>)
  {
    return isclose(a, b, T(1 << 8) * std::numeric_limits<T>::epsilon(), T(0));
  }
  else
  {
    static_assert(std::is_integral_v<T>, "isclose: unsupported type, expected floating point or integral");
    return a == b;
  }
}
