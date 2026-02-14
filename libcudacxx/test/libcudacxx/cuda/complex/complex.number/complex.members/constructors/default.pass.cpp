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

template <class T>
__host__ __device__ constexpr void test_default_constructor()
{
  using C = cuda::complex<T>;

  // 1. Test that cuda::complex<T> is default constructible
  static_assert(cuda::std::is_default_constructible_v<C>);

  // 2. If T is trivially default constructible, then cuda::complex<T> is also trivially default constructible
  static_assert(
    cuda::std::is_trivially_default_constructible_v<C> == cuda::std::is_trivially_default_constructible_v<T>);

  // 3. Test that the constructor is noexcept
  static_assert(noexcept(C{}));

  C v{};
  assert(v.real() == T{});
  assert(v.imag() == T{});
}

__host__ __device__ constexpr bool test()
{
  test_default_constructor<float>();
  test_default_constructor<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_default_constructor<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_FLOAT128()
  test_default_constructor<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  test_default_constructor<signed char>();
  test_default_constructor<signed short>();
  test_default_constructor<signed int>();
  test_default_constructor<signed long>();
  test_default_constructor<signed long long>();
#if _CCCL_HAS_INT128()
  test_default_constructor<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_default_constructor<unsigned char>();
  test_default_constructor<unsigned short>();
  test_default_constructor<unsigned int>();
  test_default_constructor<unsigned long>();
  test_default_constructor<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_default_constructor<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
