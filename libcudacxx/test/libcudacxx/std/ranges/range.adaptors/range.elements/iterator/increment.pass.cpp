//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires forward_range<Base>;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>>
__host__ __device__ constexpr void testOne()
{
  using Range                = cuda::std::ranges::subrange<Iter, Sent>;
  cuda::std::tuple<int> ts[] = {{1}, {2}, {3}};
  auto ev                    = Range{Iter{ts}, Sent{Iter{ts + 3}}} | cuda::std::views::elements<0>;
  using ElementIter          = cuda::std::ranges::iterator_t<decltype(ev)>;

  { // ++i
    auto it               = ev.begin();
    decltype(auto) result = ++it;

    static_assert(cuda::std::is_same_v<decltype(result), ElementIter&>);
    assert(&result == &it);

    assert(base(it.base()) == &ts[1]);
  }

  { // i++
    if constexpr (cuda::std::forward_iterator<Iter>)
    {
      auto it               = ev.begin();
      decltype(auto) result = it++;

      static_assert(cuda::std::is_same_v<decltype(result), ElementIter>);

      assert(base(it.base()) == &ts[1]);
      assert(base(result.base()) == &ts[0]);
    }
    else
    {
      auto it = ev.begin();
      it++;

      static_assert(cuda::std::is_same_v<decltype(it++), void>);
      assert(base(it.base()) == &ts[1]);
    }
  }
}

__host__ __device__ constexpr bool test()
{
  using Ptr = cuda::std::tuple<int>*;
#if !defined(TEST_COMPILER_CLANG) || TEST_STD_VER >= 2020 // clang ICEs on the rvalue subrange
  testOne<cpp20_input_iterator<Ptr>>();
#endif // !clang || C++20
  testOne<forward_iterator<Ptr>>();
  testOne<bidirectional_iterator<Ptr>>();
  testOne<random_access_iterator<Ptr>>();
  testOne<contiguous_iterator<Ptr>>();
  testOne<Ptr>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
