//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template <class... Args1, class... Args2>
//     pair(piecewise_construct_t, tuple<Args1...> first_args,
//                                 tuple<Args2...> second_args);

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  {
    using P1 = cuda::std::pair<int, int*>;
    using P2 = cuda::std::pair<int*, int>;
    using P3 = cuda::std::pair<P1, P2>;
    P3 p3(
      cuda::std::piecewise_construct, cuda::std::tuple<int, int*>(3, nullptr), cuda::std::tuple<int*, int>(nullptr, 4));
    assert(p3.first == P1(3, nullptr));
    assert(p3.second == P2(nullptr, 4));
  }
  {
    int x = 3;
    int y = 4;
    cuda::std::pair<int&, int&> p(
      cuda::std::piecewise_construct, cuda::std::forward_as_tuple(x), cuda::std::forward_as_tuple(y));
    assert(p.first == 3);
    assert(p.second == 4);
    p.first  = 5;
    p.second = 6;
    assert(x == 5);
    assert(y == 6);
  }
  return true;
}

template <bool Expected, class Pair, class Tuple1, class Tuple2>
TEST_FUNC void test_piecewise_constructible()
{
  static_assert(cuda::std::is_constructible_v<Pair, cuda::std::piecewise_construct_t, Tuple1, Tuple2> == Expected);
}

// [pairs.pair]/18: the constructor does not participate unless both elements are constructible from their packs.
// [pairs.pair] note 2: a reference element that would bind to a temporary is deleted.
TEST_FUNC void test_constraints()
{
  test_piecewise_constructible<true, cuda::std::pair<int, int>, cuda::std::tuple<int>, cuda::std::tuple<int>>();
  test_piecewise_constructible<true, cuda::std::pair<int, int>, cuda::std::tuple<>, cuda::std::tuple<>>();
  test_piecewise_constructible<true,
                               cuda::std::pair<cuda::std::pair<int, int>, int>,
                               cuda::std::tuple<int, int>,
                               cuda::std::tuple<int>>();

  test_piecewise_constructible<false, cuda::std::pair<int, int>, cuda::std::tuple<void*>, cuda::std::tuple<int>>();
  test_piecewise_constructible<false, cuda::std::pair<int, int>, cuda::std::tuple<int>, cuda::std::tuple<void*>>();
  test_piecewise_constructible<false, cuda::std::pair<int&, int>, cuda::std::tuple<>, cuda::std::tuple<int>>();
  test_piecewise_constructible<false,
                               cuda::std::pair<cuda::std::pair<int, int>, int>,
                               cuda::std::tuple<int>,
                               cuda::std::tuple<int>>();

  // An lvalue reference element binds directly to the tuple element.
  test_piecewise_constructible<true, cuda::std::pair<int&, int&>, cuda::std::tuple<int&>, cuda::std::tuple<int&>>();
  test_piecewise_constructible<true, cuda::std::pair<const int&, int>, cuda::std::tuple<int&>, cuda::std::tuple<int>>();

#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
  // A conversion materializes a temporary, so the reference member would dangle.
  test_piecewise_constructible<false, cuda::std::pair<const int&, int>, cuda::std::tuple<long>, cuda::std::tuple<int>>();
  test_piecewise_constructible<false, cuda::std::pair<int, const int&>, cuda::std::tuple<int>, cuda::std::tuple<long>>();
  test_piecewise_constructible<false, cuda::std::pair<int&&, int&&>, cuda::std::tuple<long>, cuda::std::tuple<int>>();
  test_piecewise_constructible<false, cuda::std::pair<int&&, int&&>, cuda::std::tuple<int>, cuda::std::tuple<long>>();

  // reference_constructs_from_temporary_v<int&&, int> and
  // reference_constructs_from_temporary_v<const int&, int> are true, so these are deleted.
  test_piecewise_constructible<false, cuda::std::pair<int&&, int&&>, cuda::std::tuple<int>, cuda::std::tuple<int>>();
  test_piecewise_constructible<false,
                               cuda::std::pair<const int&, const int&>,
                               cuda::std::tuple<int>,
                               cuda::std::tuple<int>>();
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY
}

int main(int, char**)
{
  test();
  static_assert(test());
  test_constraints();
  return 0;
}
