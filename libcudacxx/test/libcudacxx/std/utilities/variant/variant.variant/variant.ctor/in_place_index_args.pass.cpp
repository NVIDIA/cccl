//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types> class variant;

// template <size_t I, class ...Args>
// constexpr explicit variant(in_place_index_t<I>, Args&&...);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_convertible.h"
#include "test_macros.h"

__host__ __device__ void test_ctor_sfinae()
{
  {
    using V = cuda::std::variant<int>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_index_t<0>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<0>, int>(), "");
  }
  {
    using V = cuda::std::variant<int, long, long long>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_index_t<1>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<1>, int>(), "");
  }
  {
    using V = cuda::std::variant<int, long, int*>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_index_t<2>, int*>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<2>, int*>(), "");
  }
  { // args not convertible to type
    using V = cuda::std::variant<int, long, int*>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_index_t<0>, int*>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<0>, int*>(), "");
  }
  { // index not in variant
    using V = cuda::std::variant<int, long, int*>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_index_t<3>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<3>, int>(), "");
  }
}

__host__ __device__ void test_ctor_basic()
{
  {
    constexpr cuda::std::variant<int> v(cuda::std::in_place_index<0>, 42);
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v) == 42, "");
  }
  {
    constexpr cuda::std::variant<int, long, long> v(cuda::std::in_place_index<1>, 42);
    static_assert(v.index() == 1, "");
    static_assert(cuda::std::get<1>(v) == 42, "");
  }
  {
    constexpr cuda::std::variant<int, const int, long> v(cuda::std::in_place_index<1>, 42);
    static_assert(v.index() == 1, "");
    static_assert(cuda::std::get<1>(v) == 42, "");
  }
  {
    using V = cuda::std::variant<const int, volatile int, int>;
    int x   = 42;
    V v(cuda::std::in_place_index<0>, x);
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == x);
  }
  {
    using V = cuda::std::variant<const int, volatile int, int>;
    int x   = 42;
    V v(cuda::std::in_place_index<1>, x);
    assert(v.index() == 1);
    assert(cuda::std::get<1>(v) == x);
  }
  {
    using V = cuda::std::variant<const int, volatile int, int>;
    int x   = 42;
    V v(cuda::std::in_place_index<2>, x);
    assert(v.index() == 2);
    assert(cuda::std::get<2>(v) == x);
  }
}

int main(int, char**)
{
  test_ctor_basic();
  test_ctor_sfinae();

  return 0;
}
