//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/utility>

// template<typename _Tp, typename _Up>
//   constexpr bool in_range(_Tp v, _Up start, _Up end) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/utility>

#include "test_macros.h"

template <typename T, typename U>
__host__ __device__ constexpr void test()
{
  assert(cuda::in_range(T{5}, U{0}, U{10}));
  assert(!cuda::in_range(T{15}, U{0}, U{10}));
  assert(cuda::in_range(T{10}, U{0}, U{10}));

  if constexpr (cuda::std::is_signed_v<T> && cuda::std::is_unsigned_v<U>)
  {
    assert(!cuda::in_range(T{-5}, U{0}, U{10}));
    assert(cuda::in_range(T{5}, U{0}, U{10}));
  }
  if constexpr (cuda::std::is_unsigned_v<T> && cuda::std::is_signed_v<U>)
  {
    assert(cuda::in_range(T{5}, U{-10}, U{10}));
    assert(cuda::in_range(T{5}, U{-1}, U{10}));
    assert(cuda::in_range(T{0}, U{-1}, U{1}));
  }
  assert(!cuda::in_range(T{5}, U{cuda::std::numeric_limits<U>::max() - 1}, cuda::std::numeric_limits<U>::max()));
  assert(!cuda::in_range(T{5}, cuda::std::numeric_limits<U>::min(), U{cuda::std::numeric_limits<U>::min() + 1}));
  assert(cuda::in_range(T{5}, cuda::std::numeric_limits<U>::min(), cuda::std::numeric_limits<U>::max()));
}

template <typename T>
__host__ __device__ constexpr void test()
{
  test<T, unsigned char>();
  test<T, signed char>();
  test<T, unsigned short>();
  test<T, short>();
  test<T, unsigned int>();
  test<T, int>();
  test<T, unsigned long>();
  test<T, long>();
  test<T, unsigned long long>();
  test<T, long long>();
#if _CCCL_HAS_INT128()
  test<T, __int128_t>();
  test<T, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr bool test()
{
  static_assert(noexcept(cuda::in_range(1, 0, 10)));

  test<unsigned char>();
  test<signed char>();
  test<unsigned short>();
  test<short>();
  test<unsigned int>();
  test<int>();
  test<unsigned long>();
  test<long>();
  test<unsigned long long>();
  test<long long>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  return true;
}

int main(int, char**)
{
  static_assert(test());
  test();
  return 0;
}
