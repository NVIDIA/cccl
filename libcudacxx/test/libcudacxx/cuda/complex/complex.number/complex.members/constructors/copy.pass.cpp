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
__host__ __device__ constexpr void test_copy_constructor()
{
  using C = cuda::complex<T>;

  // 1. Test that cuda::complex<T> is copy constructible
  static_assert(cuda::std::is_copy_constructible_v<C>);

  // 2. If T is trivially copy constructible, then cuda::complex<T> is also trivially copy constructible
  static_assert(cuda::std::is_trivially_copy_constructible_v<C> == cuda::std::is_trivially_copy_constructible_v<T>);

  // 3. Test that the copy constructor is noexcept
  static_assert(noexcept(C{cuda::std::declval<const C&>()}));

  // 4. Test that the constructor is implicit
  static_assert(cuda::std::is_convertible_v<const C&, C>);

  const C v1{T(1), T(2)};
  assert(v1.real() == T(1));
  assert(v1.imag() == T(2));

  C v2{v1};
  assert(v2.real() == T(1));
  assert(v2.imag() == T(2));
}

__host__ __device__ constexpr bool test()
{
  test_copy_constructor<float>();
  test_copy_constructor<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_copy_constructor<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_copy_constructor<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_copy_constructor<signed char>();
  test_copy_constructor<signed short>();
  test_copy_constructor<signed int>();
  test_copy_constructor<signed long>();
  test_copy_constructor<signed long long>();
#if _CCCL_HAS_INT128()
  test_copy_constructor<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_copy_constructor<unsigned char>();
  test_copy_constructor<unsigned short>();
  test_copy_constructor<unsigned int>();
  test_copy_constructor<unsigned long>();
  test_copy_constructor<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_copy_constructor<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
