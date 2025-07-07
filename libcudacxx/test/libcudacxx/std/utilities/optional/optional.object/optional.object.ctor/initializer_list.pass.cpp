//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// template <class U, class... Args>
//     constexpr
//     explicit optional(in_place_t, initializer_list<U> il, Args&&... args);

#include <cuda/std/cassert>
#include <cuda/std/inplace_vector>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::in_place;
using cuda::std::in_place_t;
using cuda::std::optional;

class X
{
  int i_;
  int j_ = 0;

public:
  __host__ __device__ X()
      : i_(0)
  {}
  __host__ __device__ X(int i)
      : i_(i)
  {}
  __host__ __device__ X(int i, int j)
      : i_(i)
      , j_(j)
  {}

  __host__ __device__ ~X() {}

  __host__ __device__ friend bool operator==(const X& x, const X& y)
  {
    return x.i_ == y.i_ && x.j_ == y.j_;
  }
};

class Y
{
  int i_;
  int j_ = 0;

public:
  __host__ __device__ constexpr Y()
      : i_(0)
  {}
  __host__ __device__ constexpr Y(int i)
      : i_(i)
  {}
  __host__ __device__ constexpr Y(cuda::std::initializer_list<int> il)
      : i_(il.begin()[0])
      , j_(il.begin()[1])
  {}

  __host__ __device__ friend constexpr bool operator==(const Y& x, const Y& y)
  {
    return x.i_ == y.i_ && x.j_ == y.j_;
  }
};

#if TEST_HAS_EXCEPTIONS()
class Z
{
  int i_;
  int j_ = 0;

public:
  Z()
      : i_(0)
  {}
  Z(int i)
      : i_(i)
  {}
  Z(cuda::std::initializer_list<int> il)
      : i_(il.begin()[0])
      , j_(il.begin()[1])
  {
    TEST_THROW(6);
  }

  friend bool operator==(const Z& x, const Z& y)
  {
    return x.i_ == y.i_ && x.j_ == y.j_;
  }
};

void test_exceptions()
{
  static_assert(cuda::std::is_constructible<optional<Z>, cuda::std::initializer_list<int>&>::value, "");
  try
  {
    optional<Z> opt(in_place, {3, 1});
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  {
    static_assert(!cuda::std::is_constructible<X, cuda::std::initializer_list<int>&>::value, "");
    static_assert(!cuda::std::is_constructible<optional<X>, cuda::std::initializer_list<int>&>::value, "");
  }
  {
    optional<cuda::std::inplace_vector<int, 3>> opt(in_place, {3, 1});
    assert(static_cast<bool>(opt) == true);
    assert((*opt == cuda::std::inplace_vector<int, 3>{3, 1}));
    assert(opt->size() == 2);
  }
  {
    static_assert(cuda::std::is_constructible<optional<Y>, cuda::std::initializer_list<int>&>::value, "");

    {
      optional<Y> opt(in_place, {3, 1});
      assert(static_cast<bool>(opt) == true);
      assert((*opt == Y{3, 1}));
    }

    {
      constexpr optional<Y> opt(in_place, {3, 1});
      static_assert(static_cast<bool>(opt) == true, "");
      static_assert(*opt == Y{3, 1}, "");
    }

    struct test_constexpr_ctor : public optional<Y>
    {
      __host__ __device__ constexpr test_constexpr_ctor(in_place_t, cuda::std::initializer_list<int> i)
          : optional<Y>(in_place, i)
      {}
    };
  }
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
