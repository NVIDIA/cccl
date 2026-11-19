//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template <class T1, class T2> constexpr void swap(pair<T1, T2>& x, pair<T1, T2>& y);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct NotSwappable
{
  NotSwappable(const NotSwappable&)            = delete;
  NotSwappable& operator=(const NotSwappable&) = delete;
};

TEST_FUNC constexpr bool test()
{
  {
    using P1 = cuda::std::pair<int, short>;
    P1 p1(3, static_cast<short>(4));
    P1 p2(5, static_cast<short>(6));
    swap(p1, p2);
    assert(p1.first == 5);
    assert(p1.second == 6);
    assert(p2.first == 3);
    assert(p2.second == 4);
  }

  return true;
}

static_assert(cuda::std::is_swappable_v<cuda::std::pair<int, int>>);
static_assert(noexcept(swap(cuda::std::declval<cuda::std::pair<int, int>&>(),
                            cuda::std::declval<cuda::std::pair<int, int>&>())));
static_assert(!cuda::std::is_swappable_v<cuda::std::pair<NotSwappable, int>>);

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
