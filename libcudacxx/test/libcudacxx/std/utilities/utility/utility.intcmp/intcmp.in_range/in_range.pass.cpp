//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <utility>

// template<class R, class T>
//   constexpr bool in_range(T t) noexcept;               // C++20

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"

template <typename T>
struct Tuple
{
  T min;
  T max;
  T mid;
  __host__ __device__ constexpr Tuple()
  {
    min = cuda::std::numeric_limits<T>::min();
    max = cuda::std::numeric_limits<T>::max();
    if constexpr (cuda::std::is_signed_v<T>)
    {
      mid = T(-1);
    }
    else
    {
      mid = max >> 1;
    }
  }
};

template <typename T>
__host__ __device__ constexpr void test_in_range1()
{
  constexpr Tuple<T> tup;
  assert(cuda::std::in_range<T>(tup.min));
  assert(cuda::std::in_range<T>(tup.min + 1));
  assert(cuda::std::in_range<T>(tup.max));
  assert(cuda::std::in_range<T>(tup.max - 1));
  assert(cuda::std::in_range<T>(tup.mid));
  assert(cuda::std::in_range<T>(tup.mid - 1));
  assert(cuda::std::in_range<T>(tup.mid + 1));
}

__host__ __device__ constexpr void test_in_range()
{
  constexpr Tuple<uint8_t> utup8;
  constexpr Tuple<int8_t> stup8;
  assert(!cuda::std::in_range<int8_t>(utup8.max));
  assert(cuda::std::in_range<short>(utup8.max));
  assert(!cuda::std::in_range<uint8_t>(stup8.min));
  assert(cuda::std::in_range<int8_t>(utup8.mid));
  assert(!cuda::std::in_range<uint8_t>(stup8.mid));
  assert(!cuda::std::in_range<uint8_t>(-1));
}

template <class... Ts>
__host__ __device__ constexpr void test1(const cuda::std::tuple<Ts...>&)
{
  (test_in_range1<Ts>(), ...);
}

__host__ __device__ constexpr bool test()
{
  cuda::std::tuple<
#ifndef TEST_HAS_NO_INT128_T
    __int128_t,
    __uint128_t,
#endif
    unsigned long long,
    long long,
    unsigned long,
    long,
    unsigned int,
    int,
    unsigned short,
    short,
    unsigned char,
    signed char>
    types;
  test1(types);
  test_in_range();
  return true;
}

int main(int, char**)
{
  ASSERT_NOEXCEPT(cuda::std::in_range<int>(-1));
  test();
  static_assert(test());
  return 0;
}
