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
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <complex>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

template <class T, class AbsTol, class = void>
inline constexpr bool has_isclose_abs_tol_v = false;

template <class T, class AbsTol>
inline constexpr bool has_isclose_abs_tol_v<
  T,
  AbsTol,
  cuda::std::void_t<decltype(cuda::isclose(
    cuda::std::declval<T>(), cuda::std::declval<T>(), 0.0f, cuda::std::declval<AbsTol>()))>> = true;

template <class T>
TEST_FUNC bool test_floating_point()
{
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(T{}, T{}))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(T{}, T{}, 0.0f))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(T{}, T{}, 0.0f, T{}))>);
  static_assert(noexcept(cuda::isclose(T{}, T{})));
  static_assert(noexcept(cuda::isclose(T{}, T{}, 0.0f)));
  static_assert(noexcept(cuda::isclose(T{}, T{}, 0.0f, T{})));

  constexpr auto tol = cuda::__isclose_default_relative_tolerance<T>();
  assert(cuda::isclose(T{1}, T{1}));
  assert(cuda::isclose(T{1}, T{1} + static_cast<T>(tol / 2.0f)));
  assert(!cuda::isclose(T{1}, T{1} + static_cast<T>(tol * 2.0f)));

  assert(cuda::isclose(T{10}, T{11}, 0.1f));
  assert(cuda::isclose(T{11}, T{10}, 0.1f));
  assert(!cuda::isclose(T{10}, T{12}, 0.1f));

  const auto inf = cuda::std::numeric_limits<T>::infinity();
  const auto nan = cuda::std::numeric_limits<T>::quiet_NaN();
  assert(cuda::isclose(inf, inf));
  assert(cuda::isclose(-inf, -inf));
  assert(!cuda::isclose(inf, -inf));
  assert(!cuda::isclose(nan, nan));
  assert(!cuda::isclose(nan, T{}));
  return true;
}

template <typename T>
TEST_FUNC bool test_integral()
{
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(T{}, T{}))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(T{}, T{}, 0.0f))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(T{}, T{}, 0.0f, T{}))>);
  static_assert(noexcept(cuda::isclose(T{}, T{})));
  static_assert(noexcept(cuda::isclose(T{}, T{}, 0.0f)));
  static_assert(noexcept(cuda::isclose(T{}, T{}, 0.0f, T{})));

  // equal values
  assert(cuda::isclose(T{1}, T{1}));
  assert(!cuda::isclose(T{1}, T{2}));

  // relative tolerance values
  assert(cuda::isclose(T{100}, T{101}, 0.02f));
  assert(cuda::isclose(T{101}, T{100}, 0.02f));
  assert(!cuda::isclose(T{100}, T{103}, 0.02f));

  //  absolute tolerance values, positive/negative values
  assert(cuda::isclose(-3, 4, 0.0f, 7));
  assert(!cuda::isclose(-3, 4, 0.0f, 6));
  assert(cuda::isclose(3, -4, 0.0f, 7));
  assert(!cuda::isclose(3, -4, 0.0f, 6));

  //  absolute tolerance values, negative values
  assert(cuda::isclose(-3, -5, 0.0f, 2));
  assert(!cuda::isclose(-3, -5, 0.0f, 1));
  assert(cuda::isclose(-5, -3, 0.0f, 2));
  assert(!cuda::isclose(-5, -3, 0.0f, 1));

  // absolute tolerance values, positive values
  assert(cuda::isclose(3u, 5u, 0.0f, 2u));
  assert(!cuda::isclose(3u, 5u, 0.0f, 1u));
  assert(cuda::isclose(5u, 3u, 0.0f, 2u));
  assert(!cuda::isclose(5u, 3u, 0.0f, 1u));

  // edge cases
  constexpr auto max = cuda::std::numeric_limits<T>::max();
  assert(cuda::isclose(max, max, 1.0f));
  assert(cuda::isclose(T{0}, max, 1.0f));
  return true;
}

TEST_FUNC bool test_integral_boundaries()
{
  constexpr uint64_t rhs       = 18446744073586094827ull;
  constexpr uint64_t threshold = 1844674434846400176ull;
  constexpr uint64_t lhs       = rhs - threshold;

  assert(cuda::__float_ratio{0.1f} * rhs == threshold);
  assert(cuda::isclose(lhs + 1, rhs, 0.1f));
  assert(cuda::isclose(lhs, rhs, 0.1f));
  assert(!cuda::isclose(lhs - 1, rhs, 0.1f));

  constexpr auto max = cuda::std::numeric_limits<uint64_t>::max();
  assert(cuda::__float_ratio{0.0f} * max == uint64_t{0});
  assert(cuda::__float_ratio{0.5f} * max == (max >> 1));
  assert(cuda::__float_ratio{cuda::std::numeric_limits<float>::denorm_min()} * max == 0);

#if _CCCL_HAS_INT128()
  constexpr auto max128       = cuda::std::numeric_limits<__uint128_t>::max();
  constexpr auto threshold128 = (__uint128_t{13421773} << 101) - 1;
  constexpr auto lhs128       = max128 - threshold128;

  assert(cuda::__float_ratio{0.1f} * max128 == threshold128);
  assert(cuda::isclose(lhs128 + 1, max128, 0.1f));
  assert(cuda::isclose(lhs128, max128, 0.1f));
  assert(!cuda::isclose(lhs128 - 1, max128, 0.1f));
  assert(cuda::__float_ratio{0.5f} * max128 == (max128 >> 1));
  assert(cuda::__float_ratio{cuda::std::numeric_limits<float>::denorm_min()} * max128 == 0);
#endif // _CCCL_HAS_INT128()
  return true;
}

template <class Complex>
TEST_FUNC void test_complex()
{
  using T = typename Complex::value_type;
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(Complex{}, Complex{}))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(Complex{}, Complex{}, 0.0f))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::isclose(Complex{}, Complex{}, 0.0f, T{}))>);
  static_assert(noexcept(cuda::isclose(cuda::std::declval<Complex>(), cuda::std::declval<Complex>())));
  static_assert(noexcept(cuda::isclose(cuda::std::declval<Complex>(), cuda::std::declval<Complex>(), 0.0f)));
  static_assert(noexcept(
    cuda::isclose(cuda::std::declval<Complex>(), cuda::std::declval<Complex>(), 0.0f, cuda::std::declval<T>())));

  assert(cuda::isclose(Complex{T{1}, T{2}}, Complex{T{1}, T{2}}));
  assert(cuda::isclose(Complex{T{3}, T{4}}, Complex{T{3}, T{4.4}}, 0.1f));
  assert(!cuda::isclose(Complex{T{3}, T{4}}, Complex{T{3}, T{5}}, 0.1f));

  assert(!cuda::isclose(Complex{T{0}, T{0}}, Complex{T{0.3}, T{0.4}}));
  auto abs_tol = T{0.5};
#if _LIBCUDACXX_HAS_NVBF16()
  if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>)
  {
    abs_tol = T{0.51};
  }
#endif // _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::isclose(Complex{T{0}, T{0}}, Complex{T{0.3}, T{0.4}}, 0.0f, abs_tol));
  assert(!cuda::isclose(Complex{T{0}, T{0}}, Complex{T{0.3}, T{0.4}}, 0.0f, T{0.25}));

  const auto inf = cuda::std::numeric_limits<T>::infinity();
  const auto nan = cuda::std::numeric_limits<T>::quiet_NaN();
  assert(cuda::isclose(Complex{inf, T{1}}, Complex{inf, T{1}}));
  assert(!cuda::isclose(Complex{inf, T{1}}, Complex{inf, T{2}}, 1.0f));
  assert(!cuda::isclose(Complex{nan, T{}}, Complex{nan, T{}}));
  assert(!cuda::isclose(Complex{nan, T{}}, Complex{}));
}

TEST_FUNC constexpr void test_invalid_complex_cases()
{
  static_assert(!has_isclose_abs_tol_v<cuda::std::complex<double>, float>);
  static_assert(!has_isclose_abs_tol_v<cuda::complex<double>, float>);
  static_assert(!has_isclose_abs_tol_v<cuda::std::complex<float>, double>);
  static_assert(!has_isclose_abs_tol_v<cuda::complex<float>, double>);
#if _CCCL_HAS_HOST_STD_LIB()
  static_assert(!has_isclose_abs_tol_v<std::complex<double>, float>);
  static_assert(!has_isclose_abs_tol_v<std::complex<float>, double>);
#endif // _CCCL_HAS_HOST_STD_LIB()
}

TEST_FUNC bool test_standard_types()
{
  test_floating_point<float>();
  test_floating_point<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_floating_point<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_floating_point<__float128>();
#endif // _CCCL_HAS_FLOAT128()
  test_integral<signed char>();
  test_integral<unsigned char>();
  test_integral<short>();
  test_integral<unsigned short>();
  test_integral<int>();
  test_integral<unsigned>();
  test_integral<long>();
  test_integral<unsigned long>();
  test_integral<long long>();
  test_integral<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_integral<__int128_t>();
  test_integral<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  test_integral_boundaries();

  test_invalid_complex_cases();
  return true;
}

template <template <typename> class Complex>
TEST_FUNC void test_complex_types()
{
  test_complex<Complex<float>>();
  test_complex<Complex<double>>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_complex<Complex<long double>>();
#endif // _CCCL_HAS_LONG_DOUBLE()
  // complex__float128 support requires std::hypot overload
}

TEST_FUNC bool test_extended_fp()
{
#if _LIBCUDACXX_HAS_NVFP16()
  test_floating_point<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_floating_point<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  test_complex_types<cuda::std::complex>();
#if _LIBCUDACXX_HAS_NVFP16()
  test_complex<cuda::std::complex<__half>>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_complex<cuda::std::complex<__nv_bfloat16>>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  test_complex_types<cuda::complex>();
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, (test_complex_types<std::complex>();))
#endif // _CCCL_HAS_HOST_STD_LIB()
  return true;
}

int main(int, char**)
{
  assert(test_standard_types());
  assert(test_extended_fp());
  return 0;
}
