//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types> class variant;

// ~variant();

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_macros.h"

struct NonTDtor
{
  STATIC_MEMBER_VAR(count, int)
  NonTDtor() = default;
  __host__ __device__ ~NonTDtor()
  {
    ++count();
  }
};
static_assert(!cuda::std::is_trivially_destructible<NonTDtor>::value, "");

struct NonTDtor1
{
  STATIC_MEMBER_VAR(count, int)
  NonTDtor1() = default;
  __host__ __device__ ~NonTDtor1()
  {
    ++count();
  }
};
static_assert(!cuda::std::is_trivially_destructible<NonTDtor1>::value, "");

struct TDtor
{
  __host__ __device__ TDtor(const TDtor&) {} // non-trivial copy
  ~TDtor() = default;
};
static_assert(!cuda::std::is_trivially_copy_constructible<TDtor>::value, "");
static_assert(cuda::std::is_trivially_destructible<TDtor>::value, "");

int main(int, char**)
{
  {
    using V = cuda::std::variant<int, long, TDtor>;
    static_assert(cuda::std::is_trivially_destructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<NonTDtor, int, NonTDtor1>;
    static_assert(!cuda::std::is_trivially_destructible<V>::value, "");
    {
      V v(cuda::std::in_place_index<0>);
      assert(NonTDtor::count() == 0);
      assert(NonTDtor1::count() == 0);
    }
    assert(NonTDtor::count() == 1);
    assert(NonTDtor1::count() == 0);
    NonTDtor::count() = 0;
    {
      V v(cuda::std::in_place_index<1>);
    }
    assert(NonTDtor::count() == 0);
    assert(NonTDtor1::count() == 0);
    {
      V v(cuda::std::in_place_index<2>);
      assert(NonTDtor::count() == 0);
      assert(NonTDtor1::count() == 0);
    }
    assert(NonTDtor::count() == 0);
    assert(NonTDtor1::count() == 1);
  }

  return 0;
}
