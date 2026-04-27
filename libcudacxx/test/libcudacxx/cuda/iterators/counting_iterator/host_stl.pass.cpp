//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/iterator>
#include <cuda/std/cassert>

#include <iterator>

#include "test_macros.h"

template <typename Iter1T, typename Iter2T = Iter1T>
constexpr void test()
{
  using T = typename Iter1T::value_type;
  static_assert(cuda::std::same_as<T, typename Iter2T::value_type>);

  Iter1T iter1{T{42}};
  Iter2T iter2{T{77}};
  constexpr int dist = 77 - 42;

  // Distance is only defined for iterators of the same type
  if constexpr (::cuda::std::same_as<Iter1T, Iter2T>)
  {
    auto diff = ::std::distance(iter1, iter2);
    static_assert(cuda::std::is_same_v<decltype(diff), cuda::std::iter_difference_t<Iter1T>>);
    static_assert(noexcept(::std::distance(iter1, iter2)));
    assert(diff == dist);
  }

  {
    auto iter = ::std::next(iter1, dist);
    static_assert(cuda::std::is_same_v<decltype(iter), Iter1T>);
    static_assert(noexcept(::std::next(iter1, dist)));
    assert(iter == iter2);
  }

  {
    auto iter = ::std::prev(iter2, dist);
    static_assert(cuda::std::is_same_v<decltype(iter), Iter2T>);
    static_assert(noexcept(::std::prev(iter2, dist)));
    assert(iter == iter1);
  }

  {
    ::std::advance(iter1, dist);
    static_assert(cuda::std::is_same_v<decltype(::std::advance(iter1, dist)), void>);
    static_assert(noexcept(::std::advance(iter1, dist)));
    assert(iter1 == iter2);
  }
}

constexpr bool test()
{
  test<cuda::counting_iterator<cuda::std::int8_t>>();
  test<cuda::counting_iterator<cuda::std::uint8_t>>();
  test<cuda::counting_iterator<int>>();
  test<cuda::counting_iterator<cuda::std::int64_t>>();
  test<cuda::counting_iterator<cuda::std::uint64_t>>();
  test<cuda::counting_iterator<int, int>>();
  test<cuda::counting_iterator<int>, cuda::counting_iterator<int, int>>();
  test<cuda::counting_iterator<int, short>, cuda::counting_iterator<int, long long>>();

  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 test();
                 static_assert(test());
               }))

  return 0;
}
