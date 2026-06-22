//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__complex_>
#include <cuda/numeric>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U, class = void>
inline constexpr bool has_isclose_v = false;

template <class T, class U>
inline constexpr bool
  has_isclose_v<T, U, cuda::std::void_t<decltype(cuda::isclose(cuda::std::declval<T>(), cuda::std::declval<U>()))>> =
    true;

template <class T, class AbsTol, class = void>
inline constexpr bool has_isclose_abs_tol_v = false;

template <class T, class AbsTol>
inline constexpr bool has_isclose_abs_tol_v<
  T,
  AbsTol,
  cuda::std::void_t<decltype(cuda::isclose(
    cuda::std::declval<T>(), cuda::std::declval<T>(), 0.0f, cuda::std::declval<AbsTol>()))>> = true;

template <class T>
TEST_FUNC constexpr float default_rel_tol()
{
  constexpr auto digits = (cuda::std::numeric_limits<T>::max_digits10 + 1) / 2;
  auto tol              = 1.0f;
  for (int i = 0; i < digits; ++i)
  {
    tol /= 10.0f;
  }
  return tol;
}

template <class T>
TEST_FUNC constexpr bool test_floating_point()
{
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(T{}, T{}))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(T{}, T{}, 0.0f))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(T{}, T{}, 0.0f, T{}))>);
  static_assert(noexcept(cuda::isclose(T{}, T{})));
  static_assert(noexcept(cuda::isclose(T{}, T{}, 0.0f)));
  static_assert(noexcept(cuda::isclose(T{}, T{}, 0.0f, T{})));

  constexpr auto tol = default_rel_tol<T>();
  assert(cuda::isclose(T{1}, T{1}));
  assert(cuda::isclose(T{1}, T{1} + tol / T{2}));
  assert(!cuda::isclose(T{1}, T{1} + tol * T{2}));

  assert(cuda::isclose(T{10}, T{11}, 0.1f));
  assert(cuda::isclose(T{11}, T{10}, 0.1f));
  assert(!cuda::isclose(T{10}, T{12}, 0.1f));

  assert(!cuda::isclose(T{0}, tol / T{2}));
  assert(cuda::isclose(T{0}, T{0.5}, 0.0f, T{0.5}));
  assert(!cuda::isclose(T{0}, T{0.5}, 0.0f, T{0.25}));

  const auto inf = cuda::std::numeric_limits<T>::infinity();
  const auto nan = cuda::std::numeric_limits<T>::quiet_NaN();
  assert(cuda::isclose(inf, inf));
  assert(cuda::isclose(-inf, -inf));
  assert(!cuda::isclose(inf, -inf));
  assert(!cuda::isclose(inf, T{1}, 10.0f));
  assert(!cuda::isclose(nan, nan));
  assert(!cuda::isclose(nan, T{}));

  return true;
}

TEST_FUNC constexpr bool test_integral()
{
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(0, 0))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(0, 0, 0.0f))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(0, 0, 0.0f, 0.0))>);
  static_assert(noexcept(cuda::isclose(0, 0)));
  static_assert(noexcept(cuda::isclose(0, 0, 0.0f)));
  static_assert(noexcept(cuda::isclose(0, 0, 0.0f, 0.0)));

  assert(cuda::isclose(1, 1));
  assert(cuda::isclose(1u, 1u));
  assert(!cuda::isclose(1, 2));
  assert(cuda::isclose(100, 101, 0.02f));
  assert(cuda::isclose(101, 100, 0.02f));
  assert(!cuda::isclose(100, 103, 0.02f));
  assert(cuda::isclose(0, 1, 0.0f, 1.0));
  assert(!cuda::isclose(0, 1, 0.0f, 0.5));

  static_assert(!has_isclose_v<int, unsigned int>);
  static_assert(!has_isclose_v<float, double>);
  static_assert(has_isclose_abs_tol_v<double, float>);
  static_assert(has_isclose_abs_tol_v<double, double>);
  static_assert(!has_isclose_abs_tol_v<float, double>);
  static_assert(has_isclose_abs_tol_v<int, float>);
  static_assert(has_isclose_abs_tol_v<int, double>);

  return true;
}

template <class Complex>
TEST_FUNC void test_complex()
{
  using T = typename Complex::value_type;

  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(Complex{}, Complex{}))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(Complex{}, Complex{}, 0.0f))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(Complex{}, Complex{}, 0.0f, T{}))>);
  static_assert(noexcept(cuda::isclose(Complex{}, Complex{})));
  static_assert(noexcept(cuda::isclose(Complex{}, Complex{}, 0.0f)));
  static_assert(noexcept(cuda::isclose(Complex{}, Complex{}, 0.0f, T{})));

  assert(cuda::isclose(Complex{T{1}, T{2}}, Complex{T{1}, T{2}}));
  assert(cuda::isclose(Complex{T{3}, T{4}}, Complex{T{3}, T{4.4}}, 0.1f));
  assert(!cuda::isclose(Complex{T{3}, T{4}}, Complex{T{3}, T{5}}, 0.1f));

  // PEP 485 uses complex magnitudes, not component-wise scalar comparisons.
  assert(cuda::isclose(Complex{T{1}, T{1}}, Complex{T{2}, T{0}}, 0.75f));

  assert(!cuda::isclose(Complex{T{0}, T{0}}, Complex{T{0.3}, T{0.4}}));
  assert(cuda::isclose(Complex{T{0}, T{0}}, Complex{T{0.3}, T{0.4}}, 0.0f, T{0.5}));
  assert(!cuda::isclose(Complex{T{0}, T{0}}, Complex{T{0.3}, T{0.4}}, 0.0f, T{0.25}));

  const auto inf = cuda::std::numeric_limits<T>::infinity();
  const auto nan = cuda::std::numeric_limits<T>::quiet_NaN();
  assert(cuda::isclose(Complex{inf, T{1}}, Complex{inf, T{1}}));
  assert(!cuda::isclose(Complex{inf, T{1}}, Complex{inf, T{2}}, 10.0f));
  assert(!cuda::isclose(Complex{nan, T{}}, Complex{nan, T{}}));
  assert(!cuda::isclose(Complex{nan, T{}}, Complex{}));
}

TEST_FUNC void test_mixed_complex()
{
  static_assert(!has_isclose_v<cuda::std::complex<float>, cuda::std::complex<double>>);
  static_assert(!has_isclose_v<cuda::complex<float>, cuda::complex<double>>);
  static_assert(!has_isclose_v<cuda::complex<float>, cuda::std::complex<float>>);
  static_assert(has_isclose_abs_tol_v<cuda::std::complex<double>, float>);
  static_assert(has_isclose_abs_tol_v<cuda::complex<double>, float>);
  static_assert(!has_isclose_abs_tol_v<cuda::std::complex<float>, double>);
  static_assert(!has_isclose_abs_tol_v<cuda::complex<float>, double>);
}

TEST_FUNC constexpr bool test()
{
  test_floating_point<float>();
  test_floating_point<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_floating_point<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  test_integral();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  test_complex<cuda::std::complex<float>>();
  test_complex<cuda::std::complex<double>>();
  test_complex<cuda::complex<float>>();
  test_complex<cuda::complex<double>>();
  test_mixed_complex();

  return 0;
}
