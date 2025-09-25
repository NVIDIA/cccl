//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// <cuda/complex>

#include <cuda/__complex_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

template <class T>
__host__ __device__ constexpr void test_real_and_imag()
{
  using C = cuda::complex<T>;

  // 1. Test real() const signature
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::declval<const C>().real())>);
  static_assert(noexcept(cuda::std::declval<const C>().real()));

  // 2. Test real(T) signature
  static_assert(cuda::std::is_same_v<void, decltype(cuda::std::declval<C>().real(T{}))>);
  static_assert(noexcept(cuda::std::declval<C>().real(T{})));

  // 3. Test imag() const signature
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::declval<const C>().imag())>);
  static_assert(noexcept(cuda::std::declval<const C>().imag()));

  // 4. Test imag(T) signature
  static_assert(cuda::std::is_same_v<void, decltype(cuda::std::declval<C>().imag(T{}))>);
  static_assert(noexcept(cuda::std::declval<C>().imag(T{})));

  const C c1{};
  assert(c1.real() == T{});
  assert(c1.imag() == T{});

  C c2{1};
  assert(c2.real() == T{1});
  assert(c2.imag() == T{});

  c2.real(T(2));
  assert(c2.real() == T(2));
  assert(c2.imag() == T{});

  c2.imag(T(4));
  assert(c2.real() == T(2));
  assert(c2.imag() == T(4));
}

template <class T>
__host__ __device__ void test_real_and_imag_volatile()
{
  using C = cuda::complex<T>;

  // 1. Test real() const volatile signature
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::declval<const volatile C>().real())>);
  static_assert(noexcept(cuda::std::declval<const volatile C>().real()));

  // 2. Test real(T) volatile signature
  static_assert(cuda::std::is_same_v<void, decltype(cuda::std::declval<volatile C>().real(T{}))>);
  static_assert(noexcept(cuda::std::declval<volatile C>().real(T{})));

  // 3. Test imag() const volatile signature
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::declval<const volatile C>().imag())>);
  static_assert(noexcept(cuda::std::declval<const volatile C>().imag()));

  // 4. Test imag(T) volatile signature
  static_assert(cuda::std::is_same_v<void, decltype(cuda::std::declval<volatile C>().imag(T{}))>);
  static_assert(noexcept(cuda::std::declval<volatile C>().imag(T{})));

  const volatile C c1{};
  assert(c1.real() == T{});
  assert(c1.imag() == T{});

  volatile C c2{1};
  assert(c2.real() == T{1});
  assert(c2.imag() == T{});

  c2.real(T(2));
  assert(c2.real() == T(2));
  assert(c2.imag() == T{});

  c2.imag(T(4));
  assert(c2.real() == T(2));
  assert(c2.imag() == T(4));
}

__host__ __device__ constexpr bool test()
{
  test_real_and_imag<float>();
  test_real_and_imag<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_real_and_imag<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_real_and_imag<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_real_and_imag<signed char>();
  test_real_and_imag<signed short>();
  test_real_and_imag<signed int>();
  test_real_and_imag<signed long>();
  test_real_and_imag<signed long long>();
#if _CCCL_HAS_INT128()
  test_real_and_imag<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_real_and_imag<unsigned char>();
  test_real_and_imag<unsigned short>();
  test_real_and_imag<unsigned int>();
  test_real_and_imag<unsigned long>();
  test_real_and_imag<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_real_and_imag<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

__host__ __device__ void test_volatile()
{
  test_real_and_imag_volatile<float>();
  test_real_and_imag_volatile<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_real_and_imag_volatile<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_real_and_imag_volatile<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_real_and_imag_volatile<signed char>();
  test_real_and_imag_volatile<signed short>();
  test_real_and_imag_volatile<signed int>();
  test_real_and_imag_volatile<signed long>();
  test_real_and_imag_volatile<signed long long>();
#if _CCCL_HAS_INT128()
  test_real_and_imag_volatile<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_real_and_imag_volatile<unsigned char>();
  test_real_and_imag_volatile<unsigned short>();
  test_real_and_imag_volatile<unsigned int>();
  test_real_and_imag_volatile<unsigned long>();
  test_real_and_imag_volatile<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_real_and_imag_volatile<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

int main(int, char**)
{
  test();
  test_volatile();
  static_assert(test());
  return 0;
}
