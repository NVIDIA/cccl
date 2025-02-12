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
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ constexpr void test()
{
  using CommonType     = cuda::std::common_type_t<T, U>;
  constexpr auto max_a = cuda::std::numeric_limits<T>::max();
  constexpr auto max_b = cuda::std::numeric_limits<U>::max();
  constexpr auto max_c = cuda::std::numeric_limits<CommonType>::max();
  constexpr auto min_a = cuda::std::numeric_limits<T>::min();
  constexpr auto min_b = cuda::std::numeric_limits<U>::min();
  // ensure that we return the right type
  static_assert(cuda::std::is_same_v<decltype(cuda::is_add_overflow(T{1}, U{1})), bool>);
  // basic cases
  assert(!cuda::is_add_overflow(T{1}, U{1}));
  assert(cuda::is_add_overflow(max_c, U{1}));
  assert(cuda::is_add_overflow(T{1}, max_c));
  // never overflow
  if constexpr (!cuda::std::is_same_v<T, U> && sizeof(T) < 4 && sizeof(U) < 4)
  {
    assert(!cuda::is_add_overflow(max_a, max_b));
    assert(!cuda::is_add_overflow(min_a, min_b));
  }
  else if constexpr (cuda::std::is_signed_v<T> && cuda::std::is_signed_v<U>)
  {
    assert(!cuda::is_add_overflow(T{-1}, U{1})); // check for unintentional conversion
    assert(!cuda::is_add_overflow(T{1}, U{-1}));
    if constexpr (sizeof(T) >= sizeof(U))
    {
      assert(cuda::is_add_overflow(min_a, U{-1}));
      assert(cuda::is_add_overflow(max_a, U{1}));
    }
    else
    {
      assert(cuda::is_add_overflow(T{-1}, min_b));
      assert(cuda::is_add_overflow(T{1}, max_b));
    }
  }
  else if constexpr (cuda::std::is_unsigned_v<T> && cuda::std::is_unsigned_v<U>)
  {
    if constexpr (sizeof(T) >= sizeof(U))
    {
      assert(cuda::is_add_overflow(max_a, U{1}));
    }
    else
    {
      assert(cuda::is_add_overflow(T{1}, max_b));
    }
  }
  // opposite signed types
  else
  {
    if constexpr (sizeof(T) > sizeof(U))
    {
      assert(!cuda::is_add_overflow(T{1}, max_b));
      assert(cuda::std::is_unsigned_v<U> || cuda::is_add_overflow(max_a, T{1}));
    }
    else if constexpr (sizeof(T) < sizeof(U))
    {
      assert(!cuda::is_add_overflow(max_a, U{1}));
      assert(cuda::std::is_unsigned_v<T> || cuda::is_add_overflow(T{1}, max_b));
    }
    else // same sizee.g. int vs. unsigned
    {
      if constexpr (cuda::std::is_unsigned_v<T>)
      {
        assert(!cuda::is_add_overflow(T{1}, max_b));
        assert(cuda::is_add_overflow(max_a, U{1}));
      }
      else // cuda::std::is_unsigned_v<U>
      {
        assert(!cuda::is_add_overflow(max_a, U{1}));
        assert(cuda::is_add_overflow(T{1}, max_b));
      }
    }
  }
  unused(max_a);
  unused(max_b);
  unused(max_c);
  unused(min_a);
  unused(min_b);
}

template <class T>
__host__ __device__ constexpr void test()
{
  // Builtin integer types:
  // test<T, char>();
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
  // test<char>();
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
  static_assert(test());
  test();
  return 0;
}
