//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11

// <utility>

// template<class T, class U>
//   constexpr bool cmp_not_equal(T t, U u) noexcept;

#include <cuda/std/cassert>
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
__host__ __device__ constexpr void test_cmp_not_equal1()
{
  constexpr Tuple<T> tup;
  assert(!cuda::std::cmp_not_equal(T(0), T(0)));
  assert(!cuda::std::cmp_not_equal(T(10), T(10)));
  assert(!cuda::std::cmp_not_equal(tup.min, tup.min));
  assert(!cuda::std::cmp_not_equal(tup.max, tup.max));
  assert(cuda::std::cmp_not_equal(T(0), T(1)));
  assert(cuda::std::cmp_not_equal(T(1), T(0)));
  assert(cuda::std::cmp_not_equal(T(5), T(10)));
  assert(cuda::std::cmp_not_equal(T(10), T(5)));
  assert(cuda::std::cmp_not_equal(tup.min, tup.max));
  assert(cuda::std::cmp_not_equal(tup.max, tup.min));
  assert(cuda::std::cmp_not_equal(1, tup.max));
  assert(cuda::std::cmp_not_equal(tup.max, 1));
  assert(cuda::std::cmp_not_equal(1, tup.min));
  assert(cuda::std::cmp_not_equal(tup.min, 1));
  assert(cuda::std::cmp_not_equal(-2, tup.min));
  assert(cuda::std::cmp_not_equal(tup.min, -2));
  assert(cuda::std::cmp_not_equal(-2, tup.max));
  assert(cuda::std::cmp_not_equal(tup.max, -2));
}

template <typename T, typename U>
__host__ __device__ constexpr void test_cmp_not_equal2()
{
  constexpr Tuple<T> ttup;
  constexpr Tuple<U> utup;
  assert(!cuda::std::cmp_not_equal(T(0), U(0)));
  assert(!cuda::std::cmp_not_equal(T(10), U(10)));
  assert(cuda::std::cmp_not_equal(T(0), U(1)));
  assert(cuda::std::cmp_not_equal(T(1), U(0)));
  assert(cuda::std::cmp_not_equal(T(5), U(10)));
  assert(cuda::std::cmp_not_equal(T(10), U(5)));
  assert(cuda::std::cmp_not_equal(ttup.min, utup.max));
  assert(cuda::std::cmp_not_equal(utup.min, ttup.max));
}

template <class... Ts>
__host__ __device__ constexpr void test1(const cuda::std::tuple<Ts...>&)
{
  (test_cmp_not_equal1<Ts>(), ...);
}

template <class T, class... Us>
__host__ __device__ constexpr void test2_impl(const cuda::std::tuple<Us...>&)
{
  (test_cmp_not_equal2<T, Us>(), ...);
}

template <class... Ts, class UTuple>
__host__ __device__ constexpr void test2(const cuda::std::tuple<Ts...>&, const UTuple& utuple)
{
  (test2_impl<Ts>(utuple), ...);
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
  test2(types, types);
  return true;
}

int main(int, char**)
{
  ASSERT_NOEXCEPT(cuda::std::cmp_not_equal(0, 0));
  test();
  static_assert(test());
  return 0;
}
