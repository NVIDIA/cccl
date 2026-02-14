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
__host__ __device__ constexpr void test_assignment_from_value()
{
  // 1. Test that T is assignable to cuda::complex<T>
  static_assert(cuda::std::is_assignable_v<cuda::complex<T>&, const T&>);

  // 2. Test that the assignment is noexcept
  static_assert(noexcept(cuda::std::declval<cuda::complex<T>&>() = cuda::std::declval<const T&>()));

  cuda::complex<T> v{T(1), T(2)};
  assert(v.real() == T(1));
  assert(v.imag() == T(2));

  v = T(2);
  assert(v.real() == T(2));
  assert(v.imag() == T(0));
}

__host__ __device__ constexpr bool test()
{
  test_assignment_from_value<float>();
  test_assignment_from_value<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_assignment_from_value<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_assignment_from_value<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_assignment_from_value<signed char>();
  test_assignment_from_value<signed short>();
  test_assignment_from_value<signed int>();
  test_assignment_from_value<signed long>();
  test_assignment_from_value<signed long long>();
#if _CCCL_HAS_INT128()
  test_assignment_from_value<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_assignment_from_value<unsigned char>();
  test_assignment_from_value<unsigned short>();
  test_assignment_from_value<unsigned int>();
  test_assignment_from_value<unsigned long>();
  test_assignment_from_value<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_assignment_from_value<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
