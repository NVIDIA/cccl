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

// template <class Tp, class Up, class ...Args>
// constexpr explicit
// variant(in_place_type_t<Tp>, initializer_list<Up>, Args&&...);

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
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_type_t<InitList>, IL>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<InitList>, IL>(), "");
  }
  { // too many arguments
    using V = cuda::std::variant<InitList, InitListArg, int>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<InitList>, IL, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<InitList>, IL, int>(), "");
  }
  { // too few arguments
    using V = cuda::std::variant<InitList, InitListArg, int>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<InitListArg>, IL>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<InitListArg>, IL>(), "");
  }
  { // init list and arguments
    using V = cuda::std::variant<InitList, InitListArg, int>;
    static_assert(cuda::std::is_constructible<V, cuda::std::in_place_type_t<InitListArg>, IL, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<InitListArg>, IL, int>(), "");
  }
  { // not constructible from arguments
    using V = cuda::std::variant<InitList, InitListArg, int>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<int>, IL>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<int>, IL>(), "");
  }
  { // duplicate types in variant
    using V = cuda::std::variant<InitListArg, InitListArg, int>;
    static_assert(!cuda::std::is_constructible<V, cuda::std::in_place_type_t<InitListArg>, IL, int>::value, "");
    static_assert(!test_convertible<V, cuda::std::in_place_type_t<InitListArg>, IL, int>(), "");
  }
}

__host__ __device__ void test_ctor_basic()
{
  {
    constexpr cuda::std::variant<InitList, InitListArg> v(cuda::std::in_place_type<InitList>, {1, 2, 3});
    static_assert(v.index() == 0, "");
    static_assert(cuda::std::get<0>(v).size == 3, "");
  }
  {
    constexpr cuda::std::variant<InitList, InitListArg> v(cuda::std::in_place_type<InitListArg>, {1, 2, 3, 4}, 42);
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
