//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/numeric>

// template<class R, class T>
// constexpr R saturating_cast(T x) noexcept;                     // freestanding

#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/numeric>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class To, class From>
TEST_FUNC constexpr void test(From x, To res, int zero_value)
{
  assert(cuda::std::saturating_cast<To>(static_cast<From>(zero_value + x)) == res);
}

template <class To, class From>
TEST_FUNC constexpr bool test_type(int zero_value)
{
  static_assert(cuda::std::is_same_v<To, decltype(cuda::std::saturating_cast<To>(From{}))>);
  static_assert(noexcept(cuda::std::saturating_cast<To>(From{})));

  constexpr auto from_min = cuda::std::numeric_limits<From>::min();
  constexpr auto from_max = cuda::std::numeric_limits<From>::max();

  constexpr auto to_min = cuda::std::numeric_limits<To>::min();
  constexpr auto to_max = cuda::std::numeric_limits<To>::max();

  test<To>(from_min, (cuda::std::in_range<To>(from_min)) ? static_cast<To>(from_min) : to_min, zero_value);
  if constexpr (cuda::std::is_signed_v<From>)
  {
    test<To>(From{-126}, (cuda::std::is_signed_v<To>) ? static_cast<To>(-126) : To{0}, zero_value);
    test<To>(From{-1}, (cuda::std::is_signed_v<To>) ? static_cast<To>(-1) : To{0}, zero_value);
  }
  test<To>(From{0}, To{0}, zero_value);
  test<To>(From{1}, To{1}, zero_value);
  test<To>(From{126}, To{126}, zero_value);
  test<To>(from_max, (cuda::std::in_range<To>(from_max)) ? static_cast<To>(from_max) : to_max, zero_value);

  return true;
}

template <class T>
TEST_FUNC constexpr bool test(int zero_value)
{
  test_type<T, signed char>(zero_value);
  test_type<T, signed short>(zero_value);
  test_type<T, signed int>(zero_value);
  test_type<T, signed long>(zero_value);
  test_type<T, signed long long>(zero_value);
#if _CCCL_HAS_INT128()
  test_type<T, __int128_t>(zero_value);
#endif // _CCCL_HAS_INT128()

  test_type<T, unsigned char>(zero_value);
  test_type<T, unsigned short>(zero_value);
  test_type<T, unsigned int>(zero_value);
  test_type<T, unsigned long>(zero_value);
  test_type<T, unsigned long long>(zero_value);
#if _CCCL_HAS_INT128()
  test_type<T, __uint128_t>(zero_value);
#endif // _CCCL_HAS_INT128()

  return true;
}

TEST_FUNC constexpr bool test(int zero_value)
{
  test<signed char>(zero_value);
  test<signed short>(zero_value);
  test<signed int>(zero_value);
  test<signed long>(zero_value);
  test<signed long long>(zero_value);
#if _CCCL_HAS_INT128()
  test<__int128_t>(zero_value);
#endif // _CCCL_HAS_INT128()

  test<unsigned char>(zero_value);
  test<unsigned short>(zero_value);
  test<unsigned int>(zero_value);
  test<unsigned long>(zero_value);
  test<unsigned long long>(zero_value);
#if _CCCL_HAS_INT128()
  test<__uint128_t>(zero_value);
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  volatile int zero_value = 0;

  test(zero_value);
  static_assert(test(0));

  return 0;
}
