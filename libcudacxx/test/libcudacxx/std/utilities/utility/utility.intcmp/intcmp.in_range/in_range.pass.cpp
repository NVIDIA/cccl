//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class R, class T>
//   constexpr bool in_range(T t) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
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
__host__ __device__ constexpr void test_in_range1()
{
  constexpr Tuple<T> tup{};
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
  constexpr Tuple<uint8_t> utup8{};
  constexpr Tuple<int8_t> stup8{};
  assert(!cuda::std::in_range<int8_t>(utup8.max));
  assert(cuda::std::in_range<short>(utup8.max));
  assert(!cuda::std::in_range<uint8_t>(stup8.min));
  assert(cuda::std::in_range<int8_t>(utup8.mid));
  assert(!cuda::std::in_range<uint8_t>(stup8.mid));
  assert(!cuda::std::in_range<uint8_t>(-1));
}

__host__ __device__ constexpr bool test()
{
  test_in_range();
#if _CCCL_HAS_INT128()
  test_in_range1<__int128_t>();
  test_in_range1<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  test_in_range1<unsigned long long>();
  test_in_range1<long long>();
  test_in_range1<unsigned long>();
  test_in_range1<long>();
  test_in_range1<unsigned int>();
  test_in_range1<int>();
  test_in_range1<unsigned short>();
  test_in_range1<short>();
  test_in_range1<unsigned char>();
  test_in_range1<signed char>();
  return true;
}

int main(int, char**)
{
  static_assert(noexcept(cuda::std::in_range<int>(-1)));
  test();
  static_assert(test(), "");
  return 0;
}
