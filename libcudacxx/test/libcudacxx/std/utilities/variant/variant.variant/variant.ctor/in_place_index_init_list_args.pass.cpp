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

// template <size_t I, class Up, class ...Args>
// constexpr explicit
// variant(in_place_index_t<I>, initializer_list<Up>, Args&&...);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_convertible.h"
#include "test_macros.h"

struct InitList
{
  cuda::std::size_t size;
  __host__ __device__ constexpr InitList(cuda::std::initializer_list<int> il)
      : size(il.size())
  {}
};

struct InitListArg
{
  cuda::std::size_t size;
  int value;
  __host__ __device__ constexpr InitListArg(cuda::std::initializer_list<int> il, int v)
      : size(il.size())
      , value(v)
  {}
};

__host__ __device__ void test_ctor_sfinae()
{
  using IL = cuda::std::initializer_list<int>;
  { // just init list
    using V = cuda::std::variant<InitList, InitListArg, int>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_index_t<0>, IL>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<0>, IL>(), "");
  }
  { // too many arguments
    using V = cuda::std::variant<InitList, InitListArg, int>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_index_t<0>, IL, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<0>, IL, int>(), "");
  }
  { // too few arguments
    using V = cuda::std::variant<InitList, InitListArg, int>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_index_t<1>, IL>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<1>, IL>(), "");
  }
  { // init list and arguments
    using V = cuda::std::variant<InitList, InitListArg, int>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_index_t<1>, IL, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<1>, IL, int>(), "");
  }
  { // not constructible from arguments
    using V = cuda::std::variant<InitList, InitListArg, int>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_index_t<2>, IL>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<2>, IL>(), "");
  }
  { // index not in variant
    using V = cuda::std::variant<InitList, InitListArg, int>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_index_t<3>, IL>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_index_t<3>, IL>(), "");
  }
}

__host__ __device__ void test_ctor_basic()
{
  {
    constexpr cuda::std::variant<InitList, InitListArg, InitList> v(cuda::std::in_place_index<0>, {1, 2, 3});
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v).size == 3, "");
  }
  {
    constexpr cuda::std::variant<InitList, InitListArg, InitList> v(cuda::std::in_place_index<2>, {1, 2, 3});
    static_assert(v.index() == 2, "");
    static_assert(cuda::std::get<2>(v).size == 3, "");
  }
  {
    constexpr cuda::std::variant<InitList, InitListArg, InitListArg> v(cuda::std::in_place_index<1>, {1, 2, 3, 4}, 42);
    static_assert(v.index() == 1, "");
    static_assert(cuda::std::get<1>(v).size == 4, "");
    static_assert(cuda::std::get<1>(v).value == 42, "");
  }
}

int main(int, char**)
{
  test_ctor_basic();
  test_ctor_sfinae();

  return 0;
}
