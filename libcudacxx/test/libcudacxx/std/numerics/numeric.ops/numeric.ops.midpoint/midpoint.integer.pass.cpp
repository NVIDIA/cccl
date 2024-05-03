//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <numeric>

// template <class _Tp>
// _Tp midpoint(_Tp __a, _Tp __b) noexcept
//

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/numeric>

#include "test_macros.h"

template <typename T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void signed_test()
{
  constexpr T zero{0};
  constexpr T one{1};
  constexpr T two{2};
  constexpr T three{3};
  constexpr T four{4};

  ASSERT_SAME_TYPE(decltype(cuda::std::midpoint(T(), T())), T);
  ASSERT_NOEXCEPT(cuda::std::midpoint(T(), T()));
  using limits = cuda::std::numeric_limits<T>;

  assert(cuda::std::midpoint(one, three) == two);
  assert(cuda::std::midpoint(three, one) == two);

  assert(cuda::std::midpoint(zero, zero) == zero);
  assert(cuda::std::midpoint(zero, two) == one);
  assert(cuda::std::midpoint(two, zero) == one);
  assert(cuda::std::midpoint(two, two) == two);

  assert(cuda::std::midpoint(one, four) == two);
  assert(cuda::std::midpoint(four, one) == three);
  assert(cuda::std::midpoint(three, four) == three);
  assert(cuda::std::midpoint(four, three) == four);

  assert(cuda::std::midpoint(T(3), T(4)) == T(3));
  assert(cuda::std::midpoint(T(4), T(3)) == T(4));
  assert(cuda::std::midpoint(T(-3), T(4)) == T(0));
  assert(cuda::std::midpoint(T(-4), T(3)) == T(-1));
  assert(cuda::std::midpoint(T(3), T(-4)) == T(0));
  assert(cuda::std::midpoint(T(4), T(-3)) == T(1));
  assert(cuda::std::midpoint(T(-3), T(-4)) == T(-3));
  assert(cuda::std::midpoint(T(-4), T(-3)) == T(-4));

  assert(cuda::std::midpoint(limits::min(), limits::max()) == T(-1));
  assert(cuda::std::midpoint(limits::max(), limits::min()) == T(0));

  assert(cuda::std::midpoint(limits::min(), T(6)) == limits::min() / 2 + 3);
  assert(cuda::std::midpoint(T(6), limits::min()) == limits::min() / 2 + 3);
  assert(cuda::std::midpoint(limits::max(), T(6)) == limits::max() / 2 + 4);
  assert(cuda::std::midpoint(T(6), limits::max()) == limits::max() / 2 + 3);

  assert(cuda::std::midpoint(limits::min(), T(-6)) == limits::min() / 2 - 3);
  assert(cuda::std::midpoint(T(-6), limits::min()) == limits::min() / 2 - 3);
  assert(cuda::std::midpoint(limits::max(), T(-6)) == limits::max() / 2 - 2);
  assert(cuda::std::midpoint(T(-6), limits::max()) == limits::max() / 2 - 3);
}

template <typename T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void unsigned_test()
{
  constexpr T zero{0};
  constexpr T one{1};
  constexpr T two{2};
  constexpr T three{3};
  constexpr T four{4};

  ASSERT_SAME_TYPE(decltype(cuda::std::midpoint(T(), T())), T);
  ASSERT_NOEXCEPT(cuda::std::midpoint(T(), T()));
  using limits     = cuda::std::numeric_limits<T>;
  const T half_way = (limits::max() - limits::min()) / 2;

  assert(cuda::std::midpoint(one, three) == two);
  assert(cuda::std::midpoint(three, one) == two);

  assert(cuda::std::midpoint(zero, zero) == zero);
  assert(cuda::std::midpoint(zero, two) == one);
  assert(cuda::std::midpoint(two, zero) == one);
  assert(cuda::std::midpoint(two, two) == two);

  assert(cuda::std::midpoint(one, four) == two);
  assert(cuda::std::midpoint(four, one) == three);
  assert(cuda::std::midpoint(three, four) == three);
  assert(cuda::std::midpoint(four, three) == four);

  assert(cuda::std::midpoint(limits::min(), limits::max()) == T(half_way));
  assert(cuda::std::midpoint(limits::max(), limits::min()) == T(half_way + 1));

  assert(cuda::std::midpoint(limits::min(), T(6)) == limits::min() / 2 + 3);
  assert(cuda::std::midpoint(T(6), limits::min()) == limits::min() / 2 + 3);
  assert(cuda::std::midpoint(limits::max(), T(6)) == half_way + 4);
  assert(cuda::std::midpoint(T(6), limits::max()) == half_way + 3);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  signed_test<signed char>();
  signed_test<short>();
  signed_test<int>();
  signed_test<long>();
  signed_test<long long>();

  signed_test<cuda::std::int8_t>();
  signed_test<cuda::std::int16_t>();
  signed_test<cuda::std::int32_t>();
  signed_test<cuda::std::int64_t>();

  unsigned_test<unsigned char>();
  unsigned_test<unsigned short>();
  unsigned_test<unsigned int>();
  unsigned_test<unsigned long>();
  unsigned_test<unsigned long long>();

  unsigned_test<cuda::std::uint8_t>();
  unsigned_test<cuda::std::uint16_t>();
  unsigned_test<cuda::std::uint32_t>();
  unsigned_test<cuda::std::uint64_t>();

#ifndef TEST_HAS_NO_INT128_T
  unsigned_test<__uint128_t>();
  signed_test<__int128_t>();
#endif // !TEST_HAS_NO_INT128_T

  //     int_test<char>();
  signed_test<cuda::std::ptrdiff_t>();
  unsigned_test<cuda::std::size_t>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014
  return 0;
}
