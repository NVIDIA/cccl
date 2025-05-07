//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

template <class T, class In, class Ref>
__host__ __device__ constexpr void test_isqrt(In input, Ref ref)
{
  if (cuda::std::in_range<T>(input))
  {
    assert(cuda::isqrt(static_cast<T>(input)) == static_cast<T>(ref));
  }
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::isqrt(T{})), T>);
  static_assert(noexcept(cuda::isqrt(T{})));

  test_isqrt<T>(0, 0);
  test_isqrt<T>(1, 1);
  test_isqrt<T>(2, 1);
  test_isqrt<T>(4, 2);
  test_isqrt<T>(6, 2);
  test_isqrt<T>(43, 6);
  test_isqrt<T>(70, 8);
  test_isqrt<T>(99, 9);
  test_isqrt<T>(100, 10);
  test_isqrt<T>(2115, 45);
  test_isqrt<T>(2116, 46);
  test_isqrt<T>(9801, 99);
  test_isqrt<T>(2147483647, 46340);
  test_isqrt<T>(9223372036854775807, 3037000499);
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<signed short>();
  test_type<signed int>();
  test_type<signed long>();
  test_type<signed long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_type<unsigned char>();
  test_type<unsigned short>();
  test_type<unsigned int>();
  test_type<unsigned long>();
  test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
