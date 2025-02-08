//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr bool operator==(cuda::overflow_check_result<T> a, cuda::overflow_check_result<T> b)
{
  return a.result == b.result && a.is_overflow == b.is_overflow;
}

template <class T, class U>
__host__ __device__ constexpr void test()
{
  using CommonType     = decltype(T{} + U{});
  using ResultType     = cuda::overflow_check_result<CommonType>;
  constexpr auto max_a = cuda::std::numeric_limits<T>::max();
  constexpr auto max_b = cuda::std::numeric_limits<U>::max();
  constexpr auto max_c = cuda::std::numeric_limits<CommonType>::max();
  constexpr auto min_a = cuda::std::numeric_limits<T>::min();
  constexpr auto min_b = cuda::std::numeric_limits<U>::min();
  // ensure that we return the right type
  static_assert(cuda::std::is_same_v<decltype(cuda::add_overflow(T{1}, U{1})), ResultType>, "");

  assert(cuda::add_overflow(T{1}, U{1}) == (ResultType{2, false}));
  assert(cuda::add_overflow(max_c, U(1)).is_overflow);
  if constexpr (sizeof(T) < sizeof(int) && sizeof(U) < sizeof(int))
  {
    assert(cuda::add_overflow(max_a, max_b) == (ResultType{max_a + max_b, false}));
    assert(cuda::add_overflow(min_a, min_b) == (ResultType{min_a + min_b, false}));
  }
  else if constexpr (cuda::std::is_signed_v<T> && cuda::std::is_signed_v<U>)
  {
    assert(cuda::add_overflow(T{-1}, U{1}) == (ResultType{0, false}));
    assert(cuda::add_overflow(T{1}, U{-1}) == (ResultType{0, false}));
    if constexpr (sizeof(T) >= sizeof(U))
    {
      assert(cuda::add_overflow(min_a, U(-1)).is_overflow);
      assert(cuda::add_overflow(max_a, U(1)).is_overflow);
    }
    else
    {
      assert(cuda::add_overflow(T{-1}, min_b).is_overflow);
      assert(cuda::add_overflow(T{1}, max_b).is_overflow);
    }
  }
  else if constexpr (cuda::std::is_unsigned_v<T> && cuda::std::is_unsigned_v<U>)
  {
    if constexpr (sizeof(T) >= sizeof(U))
    {
      assert(cuda::add_overflow(max_a, U{1}).is_overflow);
    }
    else
    {
      assert(cuda::add_overflow(T{1}, max_b).is_overflow);
    }
  }
  // opposite signed types
  else
  {
    if constexpr (sizeof(T) > sizeof(U))
    {
      assert(!cuda::add_overflow(T{1}, max_b).is_overflow);
    }
    else if constexpr (sizeof(T) < sizeof(U))
    {
      assert(!cuda::add_overflow(max_a, U{1}).is_overflow);
    }
    else // e.g. int vs. unsigned
    {
      if constexpr (cuda::std::is_unsigned_v<T>)
      {
        assert(!cuda::add_overflow(T{1}, max_b).is_overflow);
        assert(cuda::add_overflow(max_a, U{1}).is_overflow);
      }
      else
      {
        assert(!cuda::add_overflow(max_a, U{1}).is_overflow);
        assert(cuda::add_overflow(T{1}, max_b).is_overflow);
      }
    }
  }
}

template <class T>
__host__ __device__ constexpr void test()
{
  // Builtin integer types:
  test<T, char>();
  test<T, signed char>();
  test<T, unsigned char>();

  test<T, short>();
  test<T, unsigned short>();

  test<T, int>();
  test<T, unsigned int>();

  test<T, long>();
  test<T, unsigned long>();

  test<T, long long>();
  test<T, unsigned long long>();

#if !defined(TEST_COMPILER_NVRTC)
  // cstdint types:
  test<T, size_t>();
  test<T, ptrdiff_t>();
  test<T, intptr_t>();
  test<T, uintptr_t>();

  test<T, int8_t>();
  test<T, int16_t>();
  test<T, int32_t>();
  test<T, int64_t>();

  test<T, uint8_t>();
  test<T, uint16_t>();
  test<T, uint32_t>();
  test<T, uint64_t>();
#endif // !TEST_COMPILER_NVRTC
#if !defined(TEST_HAS_NO_INT128_T)
  test<T, __int128_t>();
  test<T, __uint128_t>();
#endif // !TEST_HAS_NO_INT128_T
}

__host__ __device__ constexpr bool test()
{
  // Builtin integer types:
  test<char>();
  test<signed char>();
  test<unsigned char>();

  test<short>();
  test<unsigned short>();

  test<int>();
  test<unsigned int>();

  test<long>();
  test<unsigned long>();

  test<long long>();
  test<unsigned long long>();

#if !defined(TEST_COMPILER_NVRTC)
  // cstdint types:
  test<size_t>();
  test<ptrdiff_t>();
  test<intptr_t>();
  test<uintptr_t>();

  test<int8_t>();
  test<int16_t>();
  test<int32_t>();
  test<int64_t>();

  test<uint8_t>();
  test<uint16_t>();
  test<uint32_t>();
  test<uint64_t>();
#endif // !TEST_COMPILER_NVRTC
#if !defined(TEST_HAS_NO_INT128_T)
  test<__int128_t>();
  test<__uint128_t>();
#endif // !TEST_HAS_NO_INT128_T
  return true;
}

int main(int arg, char** argv)
{
  test();
  return 0;
}
