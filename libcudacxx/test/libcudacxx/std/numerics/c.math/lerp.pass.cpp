//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

// constexpr float lerp(float a, float b, float t) noexcept;
// constexpr double lerp(double a, double b, double t) noexcept;
// constexpr long double lerp(long double a, long double b, long double t) noexcept;
// constexpr __float128 lerp(__float128 a, __float128 b, __float128 t) noexcept; // extension

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "fp_compare.h"
#include "test_macros.h"

template <typename T>
TEST_FUNC constexpr bool constexpr_test()
{
  return cuda::std::lerp(T(0.0), T(12), T(0.0)) == T(0.0) && cuda::std::lerp(T(12), T(0.0), T(0.5)) == T(6)
      && cuda::std::lerp(T(0.0), T(12), T(2)) == T(24);
}

template <typename T>
TEST_FUNC void test()
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::lerp(T(), T(), T()))>);
  static_assert(noexcept(cuda::std::lerp(T(), T(), T())));

  const T maxV = cuda::std::numeric_limits<T>::max();
  const T inf  = cuda::std::numeric_limits<T>::infinity();

  //  Things that can be compared exactly
  assert((cuda::std::lerp(T(0.0), T(12), T(0.0)) == T(0.0)));
  assert((cuda::std::lerp(T(0.0), T(12), T(1)) == T(12)));
  assert((cuda::std::lerp(T(12), T(0.0), T(0.0)) == T(12)));
  assert((cuda::std::lerp(T(12), T(0.0), T(1)) == T(0.0)));

  assert((cuda::std::lerp(T(0.0), T(12), T(0.5)) == T(6)));
  assert((cuda::std::lerp(T(12), T(0.0), T(0.5)) == T(6)));
  assert((cuda::std::lerp(T(0.0), T(12), T(2)) == T(24)));
  assert((cuda::std::lerp(T(12), T(0.0), T(2)) == T(-12)));

  assert((cuda::std::lerp(maxV, maxV / T(10), T(0.0)) == maxV));
  assert((cuda::std::lerp(maxV / T(10), maxV, T(1)) == maxV));

  assert((cuda::std::lerp(T(2.3), T(2.3), inf) == T(2.3)));

  assert(cuda::std::lerp(T(0.0), T(0.0), T(23)) == T(0.0));

  // __half and __nvbfloat have precision issues here
  if constexpr (!cuda::std::__is_extended_floating_point_v<T>)
  {
    assert(cuda::std::isnan(cuda::std::lerp(T(0.0), T(0.0), T(inf))));
  }
}

#if _CCCL_HAS_FLOAT128()
TEST_FUNC constexpr bool test_float128_constexpr()
{
  using T = __float128;
  // 2^-100, representable in __float128 but not in double when added to 1
  constexpr T eps = T(1) / (T(1ull << 50) * T(1ull << 50));

  return cuda::std::lerp(T(-4), T(8), T(0)) == T(-4) && cuda::std::lerp(T(-4), T(8), T(1)) == T(8)
      && cuda::std::lerp(T(-4), T(8), T(0.25)) == T(-1) && cuda::std::lerp(T(1), T(1) + eps, T(0.5)) == T(1) + eps / 2
      && cuda::std::lerp(T(0), 12, 0.5f) == T(6);
}

TEST_FUNC void test_float128()
{
  using T = __float128;

  // 2^-100, representable in __float128 but not in double when added to 1
  const T eps  = T(1) / (T(1ull << 50) * T(1ull << 50));
  const T maxV = cuda::std::numeric_limits<T>::max();
  const T minV = cuda::std::numeric_limits<T>::min();

  // endpoints are returned exactly
  assert(cuda::std::lerp(T(1) + eps, T(3), T(0)) == T(1) + eps);
  assert(cuda::std::lerp(T(1) + eps, T(3), T(1)) == T(3));
  assert(cuda::std::lerp(T(-3), T(5) + eps, T(0)) == T(-3));
  assert(cuda::std::lerp(T(-3), T(5) + eps, T(1)) == T(5) + eps);
  assert(cuda::std::lerp(maxV, -maxV, T(0)) == maxV);
  assert(cuda::std::lerp(maxV, -maxV, T(1)) == -maxV);

  // interpolation and extrapolation
  assert(cuda::std::lerp(T(2), T(10), T(0.25)) == T(4));
  assert(cuda::std::lerp(T(10), T(2), T(0.25)) == T(8));
  assert(cuda::std::lerp(T(2), T(10), T(2)) == T(18));
  assert(cuda::std::lerp(T(2), T(10), T(-1)) == T(-6));
  assert(cuda::std::lerp(T(10), T(2), T(-0.5)) == T(14));

  // opposite signs
  assert(cuda::std::lerp(T(-4), T(8), T(0.25)) == T(-1));
  assert(cuda::std::lerp(T(8), T(-4), T(0.25)) == T(5));
  assert(cuda::std::lerp(T(-4), T(8), T(1.5)) == T(14));
  assert(cuda::std::lerp(T(-4), T(8), T(-0.5)) == T(-10));

  // large and small values
  assert(cuda::std::lerp(-maxV, maxV, T(0.5)) == T(0));
  assert(cuda::std::lerp(maxV, maxV, T(0.5)) == maxV);
  assert(cuda::std::lerp(minV, T(3) * minV, T(0.5)) == T(2) * minV);
  assert(cuda::std::lerp(-minV, minV, T(0.5)) == T(0));

  // precision beyond double
  assert(cuda::std::lerp(T(1), T(1) + eps, T(0.5)) == T(1) + eps / 2);
  assert(cuda::std::lerp(T(1), T(1) + eps, T(0.5)) != T(1));
  assert(static_cast<double>(cuda::std::lerp(T(1), T(1) + eps, T(0.5))) == 1.0);
  assert(cuda::std::lerp(-eps, eps, T(0.75)) == eps / 2);
  assert(cuda::std::lerp(T(1), T(1) + eps, T(2)) == T(1) + T(2) * eps);

  // non-finite inputs
  const T inf = cuda::std::numeric_limits<T>::infinity();
  assert(cuda::std::lerp(T(2), T(2), inf) == T(2));
  assert(cuda::std::isnan(cuda::std::lerp(T(0), T(0), inf)));

  // return types and mixed-type promotion
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::lerp(T(), T(), T()))>);
  static_assert(noexcept(cuda::std::lerp(T(), T(), T())));
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::lerp(T(), 0.f, 0.f))>);
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::lerp(0.f, T(), 0.f))>);
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::lerp(0.f, 0.f, T()))>);
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::lerp(T(), 0.0, 0.f))>);
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::lerp(T(), 0, 0))>);
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::lerp(0, T(), 0u))>);
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::lerp(0, 0L, T()))>);
  static_assert(noexcept(cuda::std::lerp(T(), 0.0, 0.f)));
  static_assert(cuda::std::is_same_v<double, decltype(cuda::std::lerp(0, 0.0, 0.f))>);

  assert(cuda::std::lerp(T(0), 12.0, 0.5f) == T(6));
  assert(cuda::std::lerp(0.0, T(12), 0.5f) == T(6));
  assert(cuda::std::lerp(0, 12, T(0.5)) == T(6));
  assert(cuda::std::lerp(T(-4), 8, 0.25) == T(-1));
  assert(cuda::std::lerp(T(1), T(1) + eps, 0.5) == T(1) + eps / 2);
}
#endif // _CCCL_HAS_FLOAT128()

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
#if _CCCL_HAS_FLOAT128()
  test<__float128>();
  test_float128();
#endif // _CCCL_HAS_FLOAT128()

  static_assert(constexpr_test<float>());
  static_assert(constexpr_test<double>());
#if _CCCL_HAS_FLOAT128()
  static_assert(constexpr_test<__float128>());
  static_assert(test_float128_constexpr());
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
