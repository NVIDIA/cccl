//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/numeric>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class To, class From>
__host__ __device__ constexpr void test_overflow_cast(const From& from, const bool overflow)
{
  const auto result = cuda::overflow_cast<To>(from);
  assert(result.value == static_cast<To>(from));
  assert(result.overflow == overflow);
}

template <class To, class From>
__host__ __device__ constexpr void test_type()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::overflow_cast<To>(From{})), cuda::overflow_result<To>>);
  static_assert(noexcept(cuda::overflow_cast<To>(From{})));

  // 1. Casting zero should never overflow
  test_overflow_cast<To>(From{}, false);

  // 2. Casting positive one should never overflow
  test_overflow_cast<To>(From{1}, false);

  // 3. Casting negative one should overflow if the destination type is unsigned
  if constexpr (cuda::std::is_signed_v<From>)
  {
    test_overflow_cast<To>(From{-1}, cuda::std::is_unsigned_v<To>);
  }

  // 4. Casting the minimum value of From type
  constexpr bool min_overflows =
    cuda::std::cmp_less(cuda::std::numeric_limits<From>::min(), cuda::std::numeric_limits<To>::min());
  test_overflow_cast<To>(cuda::std::numeric_limits<From>::min(), min_overflows);

  // 5. Casting the maximum value of From type
  constexpr bool max_overflows =
    cuda::std::cmp_greater(cuda::std::numeric_limits<From>::max(), cuda::std::numeric_limits<To>::max());
  test_overflow_cast<To>(cuda::std::numeric_limits<From>::max(), max_overflows);
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  test_type<T, signed char>();
  test_type<T, unsigned char>();
  test_type<T, short>();
  test_type<T, unsigned short>();
  test_type<T, int>();
  test_type<T, unsigned int>();
  test_type<T, long>();
  test_type<T, unsigned long>();
  test_type<T, long long>();
  test_type<T, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<T, __int128_t>();
  test_type<T, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr bool test()
{
  test_type<signed char>();
  test_type<unsigned char>();
  test_type<short>();
  test_type<unsigned short>();
  test_type<int>();
  test_type<unsigned int>();
  test_type<long>();
  test_type<unsigned long>();
  test_type<long long>();
  test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test());
  return 0;
}
