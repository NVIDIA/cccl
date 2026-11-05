//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a host device function in tile mode

// <cuda/std/__simd_>
// [simd.ctor], broadcast constraints and converting constructor explicitness

#include <cuda/__complex_>
#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <complex>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

namespace simd = cuda::std::simd;

struct implicit_conversion
{
  TEST_HOST_DEVICE_FUNC constexpr operator double() const
  {
    return 3.0;
  }
};

struct explicit_conversion
{
  TEST_HOST_DEVICE_FUNC explicit constexpr operator int() const
  {
    return 3;
  }
};

template <int Numerator, int Denominator = 1>
struct floating_constant
{
  static constexpr double value = double(Numerator) / Denominator;
  TEST_HOST_DEVICE_FUNC constexpr operator double() const
  {
    return value;
  }
};

struct large_floating_constant
{
  static constexpr double value = 1.0e100;
  TEST_HOST_DEVICE_FUNC constexpr operator double() const
  {
    return value;
  }
};

template <typename T, typename U, bool Expected, int N>
TEST_HOST_DEVICE_FUNC constexpr void test_participation()
{
  using Vec = simd::vec<T, N>;
  static_assert(cuda::std::is_constructible_v<Vec, U> == Expected);
  static_assert(cuda::std::is_convertible_v<U, Vec> == Expected);
  static_assert(cuda::std::is_constructible_v<Vec, U&> == Expected);
  static_assert(cuda::std::is_convertible_v<U&, Vec> == Expected);
  static_assert(cuda::std::is_constructible_v<Vec, const U&> == Expected);
  static_assert(cuda::std::is_convertible_v<const U&, Vec> == Expected);
}

template <typename T, typename U, int N>
TEST_HOST_DEVICE_FUNC constexpr void test_value(U value, T expected)
{
  test_participation<T, U, true, N>();

  const simd::vec<T, N> vec = value;
  for (int i = 0; i < N; ++i)
  {
    if constexpr (cuda::std::is_arithmetic_v<T>)
    {
      assert(vec[i] == expected);
    }
    else
    {
      assert(vec[i].real() == expected.real());
      assert(vec[i].imag() == expected.imag());
    }
  }
}

template <typename Char, int N>
TEST_HOST_DEVICE_FUNC constexpr void test_character_rank()
{
  using Integer = cuda::std::
    conditional_t<cuda::std::is_signed_v<Char>, cuda::std::make_signed_t<Char>, cuda::std::make_unsigned_t<Char>>;
  using CharVec = simd::vec<Char, N>;
  using IntVec  = simd::vec<Integer, N>;
  static_assert(cuda::std::is_convertible_v<IntVec, CharVec>);
  static_assert(cuda::std::is_convertible_v<CharVec, IntVec>);

  const CharVec chars = IntVec(Integer{5});
  const IntVec ints   = chars;
  for (int i = 0; i < N; ++i)
  {
    assert(chars[i] == Char{5});
    assert(ints[i] == Integer{5});
  }
  using cuda::std::integral_constant;
  test_value<Char, integral_constant<int, 5>, N>({}, Char{5});
  test_value<int, integral_constant<Char, Char{5}>, N>({}, 5);
  if constexpr (!cuda::std::is_signed_v<Char>)
  {
    test_participation<Char, integral_constant<int, -1>, false, N>();
  }
}

template <typename Complex, int N>
TEST_HOST_DEVICE_FUNC constexpr void test_complex()
{
  using Real = typename Complex::value_type;
  test_value<Complex, float, N>(2.0f, Complex{2});
  test_value<Complex, short, N>(3, Complex{3});
  test_value<Complex, floating_constant<3, 2>, N>({}, Complex{Real{1.5}});
  test_value<Complex, cuda::std::integral_constant<int, 5>, N>({}, Complex{5});
  // construct and convert simd::vec<Complex, N> from/to simd::vec<short, N>
  test_value<Complex, simd::vec<short, N>, N>(simd::vec<short, N>{short{6}}, Complex{6});

  if constexpr (cuda::std::is_same_v<Real, float>)
  {
    test_participation<Complex, double, false, N>();
    test_participation<Complex, cuda::std::integral_constant<int, 16777217>, false, N>();
    static_assert(cuda::std::is_constructible_v<simd::vec<Complex, N>, simd::vec<double, N>>);
    static_assert(!cuda::std::is_convertible_v<simd::vec<double, N>, simd::vec<Complex, N>>);
  }
  else
  {
    test_value<Complex, double, N>(4.0, Complex{4});
  }
}

template <int N>
TEST_HOST_DEVICE_FUNC constexpr void test_broadcast_participation()
{
  // integer and floating-point types
  test_participation<int, double, false, N>();
  test_participation<float, double, false, N>();
  test_participation<short, int, false, N>();
  test_participation<unsigned, int, false, N>();
  test_participation<int, unsigned, false, N>();
  // broadcast from an explicit conversion is not allowed
  test_participation<int, explicit_conversion, false, N>();

  using cuda::std::integral_constant;
  using IMin = integral_constant<int, cuda::std::numeric_limits<int>::min()>;
  using UMax = integral_constant<unsigned, cuda::std::numeric_limits<unsigned>::max()>;
  test_participation<unsigned, IMin, false, N>();
  test_participation<int, UMax, false, N>();
  test_participation<unsigned, integral_constant<int, -1>, false, N>();
  test_participation<signed char, integral_constant<int, 128>, false, N>();
  test_participation<float, integral_constant<int, 16777217>, false, N>();
  test_participation<float, integral_constant<uint64_t, UINT64_MAX>, false, N>();
  test_participation<int, floating_constant<3, 2>, false, N>();
  test_participation<int, large_floating_constant, false, N>();
}

template <int N>
TEST_HOST_DEVICE_FUNC constexpr void test_width()
{
  using cuda::std::integral_constant;
  test_broadcast_participation<N>();

  test_value<int, implicit_conversion, N>({}, 3);
  test_value<double, float, N>(2.0f, 2.0);
  test_value<int, short, N>(3, 3);
  test_value<int, bool, N>(true, 1);
  test_value<int, integral_constant<bool, true>, N>({}, 1);
  test_value<signed char, integral_constant<int, -128>, N>({}, -128);
  test_value<unsigned char, integral_constant<int, 255>, N>({}, 255);
  test_value<int, integral_constant<unsigned, 7>, N>({}, 7);
  test_value<int, floating_constant<3>, N>({}, 3);
  test_value<float, floating_constant<3, 2>, N>({}, 1.5f);

  // character types
  test_character_rank<wchar_t, N>();
  test_character_rank<char16_t, N>();
  test_character_rank<char32_t, N>();
#if _CCCL_HAS_CHAR8_T()
  test_character_rank<char8_t, N>();
#endif // _CCCL_HAS_CHAR8_T()

  // complex types
  test_complex<cuda::std::complex<float>, N>();
  test_complex<cuda::complex<float>, N>();
#if _CCCL_HAS_INT128() // complex<double> requires a 16-byte mask element.
  test_complex<cuda::std::complex<double>, N>();
  test_complex<cuda::complex<double>, N>();
#endif // _CCCL_HAS_INT128()
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, (test_complex<::std::complex<float>, N>();))
#  if _CCCL_HAS_INT128()
  NV_IF_TARGET(NV_IS_HOST, (test_complex<::std::complex<double>, N>();))
#  endif // _CCCL_HAS_INT128()
#endif // _CCCL_HAS_HOST_STD_LIB()
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_width<1>();
  test_width<4>();
  return true;
}

int main(int, char**)
{
  assert(test()); // NOLINT(cert-dcl03-c,misc-static-assert)
  static_assert(test());
  return 0;
}
