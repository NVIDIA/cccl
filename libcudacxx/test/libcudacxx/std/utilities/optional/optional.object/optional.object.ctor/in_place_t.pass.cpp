//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//

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
  __host__ __device__ constexpr X()
      : i_(0)
  {}
  __host__ __device__ constexpr X(int i)
      : i_(i)
  {}
  __host__ __device__ constexpr X(int i, int j)
      : i_(i)
      , j_(j)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 ~X() {}

  __host__ __device__ constexpr friend bool operator==(const X& x, const X& y)
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

template <class T, class... Args>
__host__ __device__ constexpr void test(Args... vals)
{
  const T expected{vals...};
  {
    optional<T> opt(in_place, vals...);
    assert(opt.has_value());
    assert(*opt == expected);
  }

  {
    const optional<T> opt(in_place, vals...);
    assert(opt.has_value());
    assert(*opt == expected);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      assert(cuda::std::addressof(expected) == opt.operator->());
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<const int>();

  test<int>(42);
  test<const int>(42);

  test<X>();
  test<X>(42);
  test<X>(42, 1337);

  test<Y>();
  test<Y>(42);
  test<Y>(42, 1337);

#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&>(42);
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

#if TEST_HAS_EXCEPTIONS()
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
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
