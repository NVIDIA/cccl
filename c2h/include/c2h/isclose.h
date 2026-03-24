// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

template <typename T>
bool isclose(T a, T b, T r_tol, T a_tol)
{
  if constexpr (cuda::is_floating_point_v<T>)
  {
    assert(cuda::std::isfinite(r_tol) && r_tol >= T{} && "r_tol must be finite and non-negative");
    assert(cuda::std::isfinite(a_tol) && a_tol >= T{} && "a_tol must be finite and non-negative");
    if (cuda::std::isnan(a) || cuda::std::isnan(b))
    {
      return false;
    }
    if (cuda::std::isinf(a) || cuda::std::isinf(b))
    {
      return a == b;
    }
    return cuda::std::abs(a - b) <= cuda::std::max(a_tol, r_tol * cuda::std::max(cuda::std::abs(a), cuda::std::abs(b)));
  }
  else
  {
    static_assert(cuda::std::is_integral_v<T>, "isclose: unsupported type, expected floating point or integral");
    return a == b;
  }
}

template <typename T>
bool isclose(T a, T b, T r_tol)
{
  return isclose(a, b, r_tol, T{});
}

template <typename T>
bool isclose(T a, T b)
{
  if constexpr (cuda::is_floating_point_v<T>)
  {
    return isclose(a, b, T(1 << 8) * cuda::std::numeric_limits<T>::epsilon(), T{});
  }
  else
  {
    static_assert(cuda::std::is_integral_v<T>, "isclose: unsupported type, expected floating point or integral");
    return a == b;
  }
}
