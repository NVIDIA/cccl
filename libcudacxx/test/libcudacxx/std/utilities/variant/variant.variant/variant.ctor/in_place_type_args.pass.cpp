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

// template <class Tp, class ...Args>
// constexpr explicit variant(in_place_type_t<Tp>, Args&&...);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_convertible.h"
#include "test_macros.h"

__host__ __device__ void test_ctor_sfinae()
{
  {
    using V = cuda::std::variant<int>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_type_t<int>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<int>, int>(), "");
  }
  {
    using V = cuda::std::variant<int, long, long long>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_type_t<long>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<long>, int>(), "");
  }
  {
    using V = cuda::std::variant<int, long, int*>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_type_t<int*>, int*>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<int*>, int*>(), "");
  }
  { // duplicate type
    using V = cuda::std::variant<int, long, int>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<int>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<int>, int>(), "");
  }
  { // args not convertible to type
    using V = cuda::std::variant<int, long, int*>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<int>, int*>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<int>, int*>(), "");
  }
  { // type not in variant
    using V = cuda::std::variant<int, long, int*>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<long long>, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<long long>, int>(), "");
  }
}

__host__ __device__ void test_ctor_basic()
{
  {
    constexpr cuda::std::variant<int> v(cuda::std::in_place_type<int>, 42);
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v) == 42, "");
  }
  {
    constexpr cuda::std::variant<int, long> v(cuda::std::in_place_type<long>, 42);
    static_assert(v.index() == 1, "");
    static_assert(cuda::std::get<1>(v) == 42, "");
  }
  {
    constexpr cuda::std::variant<int, const int, long> v(cuda::std::in_place_type<const int>, 42);
    static_assert(v.index() == 1, "");
    static_assert(cuda::std::get<1>(v) == 42, "");
  }
  {
    using V = cuda::std::variant<const int, volatile int, int>;
    int x   = 42;
    V v(cuda::std::in_place_type<const int>, x);
    assert(v.index() == 0);
    assert(cuda::std::get<0>(v) == x);
  }
  {
    using V = cuda::std::variant<const int, volatile int, int>;
    int x   = 42;
    V v(cuda::std::in_place_type<volatile int>, x);
    assert(v.index() == 1);
    assert(cuda::std::get<1>(v) == x);
  }
  {
    using V = cuda::std::variant<const int, volatile int, int>;
    int x   = 42;
    V v(cuda::std::in_place_type<int>, x);
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
