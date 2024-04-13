//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11

// <cuda/std/optional>

// template <class... Args>
//   constexpr explicit optional(in_place_t, Args&&... args);

#include <cuda/std/cassert>
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
  __host__ __device__ constexpr Y(int i, int j)
      : i_(i)
      , j_(j)
  {}

  __host__ __device__ friend constexpr bool operator==(const Y& x, const Y& y)
  {
    return x.i_ == y.i_ && x.j_ == y.j_;
  }
};

#ifndef TEST_HAS_NO_EXCEPTIONS
class Z
{
public:
  Z(int)
  {
    TEST_THROW(6);
  }
};

void test_exceptions()
{
  try
  {
    const optional<Z> opt(in_place, 1);
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    constexpr optional<int> opt(in_place, 5);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(*opt == 5, "");

    struct test_constexpr_ctor : public optional<int>
    {
      __host__ __device__ constexpr test_constexpr_ctor(in_place_t, int i)
          : optional<int>(in_place, i)
      {}
    };
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    optional<const int> opt(in_place, 5);
    assert(*opt == 5);
  }
#ifndef TEST_COMPILER_ICC
  {
    const optional<X> opt(in_place);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == X());
  }
  {
    const optional<X> opt(in_place, 5);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == X(5));
  }
  {
    const optional<X> opt(in_place, 5, 4);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == X(5, 4));
  }
#endif // TEST_COMPILER_ICC
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    constexpr optional<Y> opt(in_place);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(*opt == Y(), "");

    struct test_constexpr_ctor : public optional<Y>
    {
      __host__ __device__ constexpr test_constexpr_ctor(in_place_t)
          : optional<Y>(in_place)
      {}
    };
  }
  {
    constexpr optional<Y> opt(in_place, 5);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(*opt == Y(5), "");

    struct test_constexpr_ctor : public optional<Y>
    {
      __host__ __device__ constexpr test_constexpr_ctor(in_place_t, int i)
          : optional<Y>(in_place, i)
      {}
    };
  }
  {
    constexpr optional<Y> opt(in_place, 5, 4);
    static_assert(static_cast<bool>(opt) == true, "");
    static_assert(*opt == Y(5, 4), "");

    struct test_constexpr_ctor : public optional<Y>
    {
      __host__ __device__ constexpr test_constexpr_ctor(in_place_t, int i, int j)
          : optional<Y>(in_place, i, j)
      {}
    };
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))

#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS

  return 0;
}
