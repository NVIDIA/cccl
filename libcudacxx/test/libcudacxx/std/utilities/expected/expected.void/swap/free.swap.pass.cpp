//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: gcc-6

// Older Clangs do not support the C++20 feature to constrain destructors

// friend constexpr void swap(expected& x, expected& y) noexcept(noexcept(swap(x,y)));

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../../types.h"
#include "test_macros.h"

// Test constraint
static_assert(cuda::std::is_swappable_v<cuda::std::expected<void, int>>, "");

struct NotSwappable
{
  __host__ __device__ NotSwappable& operator=(const NotSwappable&) = delete;
};
__host__ __device__ void swap(NotSwappable&, NotSwappable&) = delete;

// !is_swappable_v<E>
static_assert(!cuda::std::is_swappable_v<cuda::std::expected<void, NotSwappable>>, "");

struct NotMoveContructible
{
  NotMoveContructible(NotMoveContructible&&) = delete;
  __host__ __device__ friend void swap(NotMoveContructible&, NotMoveContructible&) {}
};

// !is_move_constructible_v<E>
static_assert(!cuda::std::is_swappable_v<cuda::std::expected<void, NotMoveContructible>>, "");

// Test noexcept
struct MoveMayThrow
{
  __host__ __device__ MoveMayThrow(MoveMayThrow&&) noexcept(false);
  __host__ __device__ friend void swap(MoveMayThrow&, MoveMayThrow&) noexcept {}
};
static_assert(cuda::std::is_nothrow_swappable_v<cuda::std::expected<void, int>>, "");

#ifndef TEST_COMPILER_ICC
// !is_nothrow_move_constructible_v<E>
static_assert(!cuda::std::is_nothrow_swappable_v<cuda::std::expected<void, MoveMayThrow>>, "");

struct SwapMayThrow
{
  __host__ __device__ friend void swap(SwapMayThrow&, SwapMayThrow&) noexcept(false) {}
};
static_assert(!cuda::std::is_nothrow_swappable_v<cuda::std::expected<void, SwapMayThrow>>, "");
#endif // TEST_COMPILER_ICC

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // this->has_value() && rhs.has_value()
  {
    cuda::std::expected<void, int> x;
    cuda::std::expected<void, int> y;
    swap(x, y);

    assert(x.has_value());
    assert(y.has_value());
  }

  // !this->has_value() && !rhs.has_value()
  {
    cuda::std::expected<void, ADLSwap> x(cuda::std::unexpect, 5);
    cuda::std::expected<void, ADLSwap> y(cuda::std::unexpect, 10);
    swap(x, y);

    assert(!x.has_value());
    assert(x.error().i == 10);
    assert(x.error().adlSwapCalled);
    assert(!y.has_value());
    assert(y.error().i == 5);
    assert(y.error().adlSwapCalled);
  }

  // this->has_value() && !rhs.has_value()
  {
    Traced::state s{};
    cuda::std::expected<void, Traced> e1(cuda::std::in_place);
    cuda::std::expected<void, Traced> e2(cuda::std::unexpect, s, 10);

    e1.swap(e2);

    assert(!e1.has_value());
    assert(e1.error().data_ == 10);
    assert(e2.has_value());

    assert(s.moveCtorCalled);
    assert(s.dtorCalled);
  }

  // !this->has_value() && rhs.has_value()
  {
    Traced::state s{};
    cuda::std::expected<void, Traced> e1(cuda::std::unexpect, s, 10);
    cuda::std::expected<void, Traced> e2(cuda::std::in_place);

    e1.swap(e2);

    assert(e1.has_value());
    assert(!e2.has_value());
    assert(e2.error().data_ == 10);

    assert(s.moveCtorCalled);
    assert(s.dtorCalled);
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  // !e1.has_value() && e2.has_value()
  {
    bool e1Destroyed = false;
    cuda::std::expected<void, ThrowOnMove> e1(cuda::std::unexpect, e1Destroyed);
    cuda::std::expected<void, ThrowOnMove> e2(cuda::std::in_place);
    try
    {
      swap(e1, e2);
      assert(false);
    }
    catch (Except)
    {
      assert(!e1.has_value());
      assert(e2.has_value());
      assert(!e1Destroyed);
    }
  }

  // e1.has_value() && !e2.has_value()
  {
    bool e2Destroyed = false;
    cuda::std::expected<void, ThrowOnMove> e1(cuda::std::in_place);
    cuda::std::expected<void, ThrowOnMove> e2(cuda::std::unexpect, e2Destroyed);
    try
    {
      swap(e1, e2);
      assert(false);
    }
    catch (Except)
    {
      assert(e1.has_value());
      assert(!e2.has_value());
      assert(!e2Destroyed);
    }
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
