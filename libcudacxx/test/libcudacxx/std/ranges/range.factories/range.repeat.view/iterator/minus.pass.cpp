//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// friend constexpr iterator operator-(iterator i, difference_type n);
// friend constexpr difference_type operator-(const iterator& x, const iterator& y);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstdint>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // <iterator> - difference_type
  {
    using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::repeat_view<int>>;
    cuda::std::ranges::repeat_view<int> v(0);
    Iter iter = v.begin() + 10;
    assert(iter - 5 == v.begin() + 5);
    static_assert(cuda::std::same_as<decltype(iter - 5), Iter>);
  }

  // <iterator> - <iterator>
  {
    // unbound
    {
      cuda::std::ranges::repeat_view<int> v(0);
      auto iter1 = v.begin() + 10;
      auto iter2 = v.begin() + 5;
      assert(iter1 - iter2 == 5);
      static_assert(cuda::std::same_as<decltype(iter1 - iter2), ptrdiff_t>);
    }

    // bound && signed bound sentinel
    {
      cuda::std::ranges::repeat_view<int, int> v(0, 20);
      auto iter1 = v.begin() + 10;
      auto iter2 = v.begin() + 5;
      assert(iter1 - iter2 == 5);
      static_assert(cuda::std::same_as<decltype(iter1 - iter2), int>);
    }

    // bound && unsigned bound sentinel
    {
      cuda::std::ranges::repeat_view<int, unsigned> v(0, 20);
      auto iter1 = v.begin() + 10;
      auto iter2 = v.begin() + 5;
      assert(iter1 - iter2 == 5);
      static_assert(sizeof(decltype(iter1 - iter2)) > sizeof(unsigned));
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if !defined(TEST_COMPILER_CLANG) || __clang__ > 9
  static_assert(test());
#endif // clang > 9

  return 0;
}
