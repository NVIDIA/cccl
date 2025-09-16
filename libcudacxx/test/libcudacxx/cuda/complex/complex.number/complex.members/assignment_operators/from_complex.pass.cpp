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

template <class T, class U>
__host__ __device__ constexpr void test_assignment_from_complex()
{
  // 1. Test that cuda::complex<U> is assignable to cuda::complex<T>
  static_assert(cuda::std::is_assignable_v<cuda::complex<T>&, const cuda::complex<U>&>);

  // 2. Test that the assignment is noexcept
  static_assert(noexcept(cuda::std::declval<cuda::complex<T>&>() = cuda::std::declval<const cuda::complex<U>&>()));

  // 3. If T and U are the same type, test that cuda::complex<T> is trivially assignable from cuda::complex<U> if T is
  // trivially assignable from U
  static_assert(!cuda::std::is_same_v<T, U>
                || (cuda::std::is_trivially_assignable_v<T&, const U&>
                    == cuda::std::is_trivially_assignable_v<cuda::complex<T>&, const cuda::complex<U>&>) );

  const cuda::complex<U> u{U(1), U(2)};
  assert(u.real() == U(1));
  assert(u.imag() == U(2));

  const cuda::complex<T> t{u};
  assert(t.real() == T(1));
  assert(t.imag() == T(2));
}

template <class T>
__host__ __device__ constexpr void test()
{
  test_assignment_from_complex<T, float>();
  test_assignment_from_complex<T, double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_assignment_from_complex<T, long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_assignment_from_complex<T, __float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_assignment_from_complex<T, signed char>();
  test_assignment_from_complex<T, signed short>();
  test_assignment_from_complex<T, signed int>();
  test_assignment_from_complex<T, signed long>();
  test_assignment_from_complex<T, signed long long>();
#if _CCCL_HAS_INT128()
  test_assignment_from_complex<T, __int128_t>();
#endif // _CCCL_HAS_INT128()

  test_assignment_from_complex<T, unsigned char>();
  test_assignment_from_complex<T, unsigned short>();
  test_assignment_from_complex<T, unsigned int>();
  test_assignment_from_complex<T, unsigned long>();
  test_assignment_from_complex<T, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_assignment_from_complex<T, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr bool test()
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test<signed char>();
  test<signed short>();
  test<signed int>();
  test<signed long>();
  test<signed long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test<unsigned char>();
  test<unsigned short>();
  test<unsigned int>();
  test<unsigned long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
