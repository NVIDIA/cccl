//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<class I, sentinel_for<I> S>
//   requires (!sized_sentinel_for<S, I>)
//     constexpr iter_difference_t<I> ranges::distance(I first, S last);
//
// template<class I, sized_sentinel_for<decay_t<I>> S>
//   constexpr iter_difference_t<I> ranges::distance(const I& first, S last);

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"

template <class It>
struct EvilSentinel
{
  It p_;
  __host__ __device__ friend constexpr bool operator==(EvilSentinel s, It p)
  {
    return s.p_ == p;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator==(It p, EvilSentinel s)
  {
    return s.p_ == p;
  }
  __host__ __device__ friend constexpr bool operator!=(EvilSentinel s, It p)
  {
    return s.p_ != p;
  }
  __host__ __device__ friend constexpr bool operator!=(It p, EvilSentinel s)
  {
    return s.p_ != p;
  }
#endif
  __host__ __device__ friend constexpr auto operator-(EvilSentinel s, It p)
  {
    return s.p_ - p;
  }
  __host__ __device__ friend constexpr auto operator-(It p, EvilSentinel s)
  {
    return p - s.p_;
  }
// Older clang confuses the all deleted overloads
#if (!defined(TEST_CLANG_VER) || TEST_CLANG_VER >= 1000)
  __host__ __device__ friend constexpr void operator-(EvilSentinel s, int (&)[3])       = delete;
  __host__ __device__ friend constexpr void operator-(EvilSentinel s, const int (&)[3]) = delete;
// Older gcc confuses the rvalue overloads with the lvalue overloads and complains about duplicated function definitions
#  if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10)
  __host__ __device__ friend constexpr void operator-(EvilSentinel s, int (&&)[3])       = delete;
  __host__ __device__ friend constexpr void operator-(EvilSentinel s, const int (&&)[3]) = delete;
#  endif
#endif
};
static_assert(cuda::std::sized_sentinel_for<EvilSentinel<int*>, int*>);
static_assert(!cuda::std::sized_sentinel_for<EvilSentinel<int*>, const int*>);
static_assert(cuda::std::sized_sentinel_for<EvilSentinel<const int*>, int*>);
static_assert(cuda::std::sized_sentinel_for<EvilSentinel<const int*>, const int*>);

__host__ __device__ constexpr bool test()
{
  {
    int a[] = {1, 2, 3};
    assert(cuda::std::ranges::distance(a, a + 3) == 3);
    assert(cuda::std::ranges::distance(a, a) == 0);
    assert(cuda::std::ranges::distance(a + 3, a) == -3);
  }
  {
    int a[] = {1, 2, 3};
    assert(cuda::std::ranges::distance(a, EvilSentinel<int*>{a + 3}) == 3);
    assert(cuda::std::ranges::distance(a, EvilSentinel<int*>{a}) == 0);
    assert(cuda::std::ranges::distance(a + 3, EvilSentinel<int*>{a}) == -3);
    assert(cuda::std::ranges::distance(cuda::std::move(a), EvilSentinel<int*>{a + 3}) == 3);
  }
  {
    const int a[] = {1, 2, 3};
    assert(cuda::std::ranges::distance(a, EvilSentinel<const int*>{a + 3}) == 3);
    assert(cuda::std::ranges::distance(a, EvilSentinel<const int*>{a}) == 0);
    assert(cuda::std::ranges::distance(a + 3, EvilSentinel<const int*>{a}) == -3);
    assert(cuda::std::ranges::distance(cuda::std::move(a), EvilSentinel<const int*>{a + 3}) == 3);
    static_assert(
      !cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), const int(&)[3], EvilSentinel<int*>>);
    static_assert(
      !cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), const int(&&)[3], EvilSentinel<int*>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
