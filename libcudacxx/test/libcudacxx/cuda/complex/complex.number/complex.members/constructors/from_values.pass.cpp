//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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
__host__ __device__ constexpr void test_constructor_from_values()
{
  // 1. Test that cuda::complex<T> constructible from T and T T
  static_assert(cuda::std::is_constructible_v<cuda::complex<T>, const T&>);
  static_assert(cuda::std::is_constructible_v<cuda::complex<T>, const T&, const T&>);

  // 2. Test that the constructor is noexcept
  static_assert(noexcept(cuda::complex<T>{cuda::std::declval<const T&>()}));
  static_assert(noexcept(cuda::complex<T>{cuda::std::declval<const T&>(), cuda::std::declval<const T&>()}));

  const cuda::complex<T> v1{T(2)};
  assert(v1.real() == T(2));
  assert(v1.imag() == T(0));

  const cuda::complex<T> v2{T(1), T(2)};
  assert(v2.real() == T(1));
  assert(v2.imag() == T(2));
}

__host__ __device__ constexpr bool test()
{
  test_constructor_from_values<float>();
  test_constructor_from_values<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_constructor_from_values<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_constructor_from_values<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_constructor_from_values<signed char>();
  test_constructor_from_values<signed short>();
  test_constructor_from_values<signed int>();
  test_constructor_from_values<signed long>();
  test_constructor_from_values<signed long long>();
#if _CCCL_HAS_INT128()
  test_constructor_from_values<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_constructor_from_values<unsigned char>();
  test_constructor_from_values<unsigned short>();
  test_constructor_from_values<unsigned int>();
  test_constructor_from_values<unsigned long>();
  test_constructor_from_values<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_constructor_from_values<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
