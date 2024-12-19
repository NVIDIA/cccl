//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T, class U>
//   constexpr bool cmp_less(T t, U u) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"

template <typename T>
struct Tuple
{
  T min = cuda::std::numeric_limits<T>::min();
  T max = cuda::std::numeric_limits<T>::max();
  T mid = cuda::std::is_signed<T>::value ? T(-1) : max >> 1;

  __host__ __device__ constexpr Tuple() noexcept {}
};

template <typename T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test1()
{
  constexpr Tuple<T> tup{};
  assert(cuda::std::cmp_less(T(0), T(1)));
  assert(cuda::std::cmp_less(T(1), T(2)));
  assert(cuda::std::cmp_less(tup.min, tup.max));
  assert(cuda::std::cmp_less(tup.min, tup.mid));
  assert(cuda::std::cmp_less(tup.mid, tup.max));
  assert(!cuda::std::cmp_less(T(1), T(0)));
  assert(!cuda::std::cmp_less(T(10), T(5)));
  assert(!cuda::std::cmp_less(tup.max, tup.min));
  assert(!cuda::std::cmp_less(tup.mid, tup.min));
  assert(!cuda::std::cmp_less(tup.mid, tup.mid));
  assert(!cuda::std::cmp_less(tup.min, tup.min));
  assert(!cuda::std::cmp_less(tup.max, tup.max));
  assert(!cuda::std::cmp_less(tup.max, 1));
  assert(!cuda::std::cmp_less(1, tup.min));
  assert(!cuda::std::cmp_less(T(-1), T(-1)));
  assert(!cuda::std::cmp_less(-2, tup.min) == cuda::std::is_signed<T>::value);
  assert(cuda::std::cmp_less(tup.min, -2) == cuda::std::is_signed<T>::value);
  assert(cuda::std::cmp_less(-2, tup.max));
  assert(!cuda::std::cmp_less(tup.max, -2));
}

template <typename T, typename U>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test2()
{
  assert(cuda::std::cmp_less(T(0), U(1)));
  assert(!cuda::std::cmp_less(T(1), U(0)));
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  test1<T>();
#ifndef TEST_HAS_NO_INT128_T
  test2<T, __int128_t>();
  test2<T, __uint128_t>();
#endif // TEST_HAS_NO_INT128_T
  test2<T, unsigned long long>();
  test2<T, long long>();
  test2<T, unsigned long>();
  test2<T, long>();
  test2<T, unsigned int>();
  test2<T, int>();
  test2<T, unsigned short>();
  test2<T, short>();
  test2<T, unsigned char>();
  test2<T, signed char>();
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
#ifndef TEST_HAS_NO_INT128_T
  test<__int128_t>();
  test<__uint128_t>();
#endif // TEST_HAS_NO_INT128_T
  test<unsigned long long>();
  test<long long>();
  test<unsigned long>();
  test<long>();
  test<unsigned int>();
  test<int>();
  test<unsigned short>();
  test<short>();
  test<unsigned char>();
  test<signed char>();
  return true;
}

int main(int, char**)
{
  ASSERT_NOEXCEPT(cuda::std::cmp_less(0, 1));
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014
  return 0;
}
