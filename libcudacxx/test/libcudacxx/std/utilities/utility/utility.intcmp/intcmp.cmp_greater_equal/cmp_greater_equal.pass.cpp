//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

//   constexpr bool cmp_greater_equal(T t, U u) noexcept;

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
__host__ __device__ constexpr void test1()
{
  constexpr Tuple<T> tup{};
  assert(!cuda::std::cmp_greater_equal(T(0), T(1)));
  assert(!cuda::std::cmp_greater_equal(T(1), T(2)));
  assert(!cuda::std::cmp_greater_equal(tup.min, tup.max));
  assert(!cuda::std::cmp_greater_equal(tup.min, tup.mid));
  assert(!cuda::std::cmp_greater_equal(tup.mid, tup.max));
  assert(cuda::std::cmp_greater_equal(T(1), T(0)));
  assert(cuda::std::cmp_greater_equal(T(10), T(5)));
  assert(cuda::std::cmp_greater_equal(tup.max, tup.min));
  assert(cuda::std::cmp_greater_equal(tup.mid, tup.min));
  assert(cuda::std::cmp_greater_equal(tup.mid, tup.mid));
  assert(cuda::std::cmp_greater_equal(tup.min, tup.min));
  assert(cuda::std::cmp_greater_equal(tup.max, tup.max));
  assert(cuda::std::cmp_greater_equal(tup.max, 1));
  assert(cuda::std::cmp_greater_equal(1, tup.min));
  assert(cuda::std::cmp_greater_equal(T(-1), T(-1)));
  assert(cuda::std::cmp_greater_equal(-2, tup.min) == cuda::std::is_signed<T>::value);
  assert(cuda::std::cmp_greater_equal(tup.min, -2) == cuda::std::is_unsigned<T>::value);
  assert(!cuda::std::cmp_greater_equal(-2, tup.max));
  assert(cuda::std::cmp_greater_equal(tup.max, -2));
}

template <typename T, typename U>
__host__ __device__ constexpr void test2()
{
  assert(!cuda::std::cmp_greater_equal(T(0), U(1)));
  assert(cuda::std::cmp_greater_equal(T(1), U(0)));
  assert(cuda::std::cmp_greater_equal(T(0), U(0)));
  assert(cuda::std::cmp_greater_equal(T(1), U(1)));
}

template <class T>
__host__ __device__ constexpr void test()
{
  test1<T>();
#if _CCCL_HAS_INT128()
  test2<T, __int128_t>();
  test2<T, __uint128_t>();
#endif // _CCCL_HAS_INT128()
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

__host__ __device__ constexpr bool test()
{
#if _CCCL_HAS_INT128()
  test<__int128_t>();
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()
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
  static_assert(noexcept(cuda::std::cmp_greater_equal(1, 0)));
  test();
  static_assert(test(), "");
  return 0;
}
