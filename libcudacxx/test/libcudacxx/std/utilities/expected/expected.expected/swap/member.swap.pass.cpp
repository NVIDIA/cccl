//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr void swap(expected& rhs) noexcept(see below);
//
// Constraints:
// is_swappable_v<T> is true and
// is_swappable_v<E> is true and
// is_move_constructible_v<T> && is_move_constructible_v<E> is true, and
// is_nothrow_move_constructible_v<T> || is_nothrow_move_constructible_v<E> is true.
//
// Throws: Any exception thrown by the expressions in the Effects.
// Remarks: The exception specification is equivalent to:
// is_nothrow_move_constructible_v<T> && is_nothrow_swappable_v<T> &&
// is_nothrow_move_constructible_v<E> && is_nothrow_swappable_v<E>

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../../types.h"
#include "test_macros.h"

// Test Constraints:
template <class T, class E>
_LIBCUDACXX_CONCEPT_FRAGMENT(HasMemberSwap_,
                             requires(cuda::std::expected<T, E> x, cuda::std::expected<T, E> y)((x.swap(y))));
template <class T, class E>
_LIBCUDACXX_CONCEPT HasMemberSwap = _LIBCUDACXX_FRAGMENT(HasMemberSwap_, T, E);

static_assert(HasMemberSwap<int, int>, "");

struct NotSwappable
{};
__host__ __device__ void swap(NotSwappable&, NotSwappable&) = delete;

// !is_swappable_v<T>
static_assert(!HasMemberSwap<NotSwappable, int>, "");

// !is_swappable_v<E>
static_assert(!HasMemberSwap<int, NotSwappable>, "");

struct NotMoveContructible
{
  NotMoveContructible(NotMoveContructible&&) = delete;
  __host__ __device__ friend void swap(NotMoveContructible&, NotMoveContructible&) {}
};

// !is_move_constructible_v<T>
static_assert(!HasMemberSwap<NotMoveContructible, int>, "");

// !is_move_constructible_v<E>
static_assert(!HasMemberSwap<int, NotMoveContructible>, "");

struct MoveMayThrow
{
  __host__ __device__ MoveMayThrow(MoveMayThrow&&) noexcept(false);
  __host__ __device__ friend void swap(MoveMayThrow&, MoveMayThrow&) noexcept {}
};

// !is_nothrow_move_constructible_v<T> && is_nothrow_move_constructible_v<E>
static_assert(HasMemberSwap<MoveMayThrow, int>, "");

// is_nothrow_move_constructible_v<T> && !is_nothrow_move_constructible_v<E>
static_assert(HasMemberSwap<int, MoveMayThrow>, "");

#ifndef TEST_COMPILER_ICC
// !is_nothrow_move_constructible_v<T> && !is_nothrow_move_constructible_v<E>
static_assert(!HasMemberSwap<MoveMayThrow, MoveMayThrow>, "");
#endif // TEST_COMPILER_ICC

// Test noexcept
template <class T, class E, bool = HasMemberSwap<T, E>>
constexpr bool MemberSwapNoexcept = false;

template <class T, class E>
constexpr bool MemberSwapNoexcept<T, E, true> =
  noexcept(cuda::std::declval<cuda::std::expected<T, E>&>().swap(cuda::std::declval<cuda::std::expected<T, E>&>()));

static_assert(MemberSwapNoexcept<int, int>, "");

#ifndef TEST_COMPILER_ICC
// !is_nothrow_move_constructible_v<T>
static_assert(!MemberSwapNoexcept<MoveMayThrow, int>, "");

// !is_nothrow_move_constructible_v<E>
static_assert(!MemberSwapNoexcept<int, MoveMayThrow>, "");

struct SwapMayThrow
{
  __host__ __device__ friend void swap(SwapMayThrow&, SwapMayThrow&) noexcept(false) {}
};

// !is_nothrow_swappable_v<T>
static_assert(!MemberSwapNoexcept<SwapMayThrow, int>, "");

// !is_nothrow_swappable_v<E>
static_assert(!MemberSwapNoexcept<int, SwapMayThrow>, "");
#endif // TEST_COMPILER_ICC

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // this->has_value() && rhs.has_value()
  {
    cuda::std::expected<ADLSwap, int> x(cuda::std::in_place, 5);
    cuda::std::expected<ADLSwap, int> y(cuda::std::in_place, 10);
    x.swap(y);

    assert(x.has_value());
    assert(x->i == 10);
    assert(x->adlSwapCalled);
    assert(y.has_value());
    assert(y->i == 5);
    assert(y->adlSwapCalled);
  }

  // !this->has_value() && !rhs.has_value()
  {
    cuda::std::expected<int, ADLSwap> x(cuda::std::unexpect, 5);
    cuda::std::expected<int, ADLSwap> y(cuda::std::unexpect, 10);
    x.swap(y);

    assert(!x.has_value());
    assert(x.error().i == 10);
    assert(x.error().adlSwapCalled);
    assert(!y.has_value());
    assert(y.error().i == 5);
    assert(y.error().adlSwapCalled);
  }

  // this->has_value() && !rhs.has_value()
  // && is_nothrow_move_constructible_v<E>
  {
    cuda::std::expected<TrackedMove<true>, TrackedMove<true>> e1(cuda::std::in_place, 5);
    cuda::std::expected<TrackedMove<true>, TrackedMove<true>> e2(cuda::std::unexpect, 10);

    e1.swap(e2);

    assert(!e1.has_value());
    assert(e1.error().i == 10);
    assert(e2.has_value());
    assert(e2->i == 5);

    assert(e1.error().numberOfMoves == 2);
    assert(!e1.error().swapCalled);
    assert(e2->numberOfMoves == 1);
    assert(!e2->swapCalled);
  }

  // this->has_value() && !rhs.has_value()
  // && !is_nothrow_move_constructible_v<E>
  {
    cuda::std::expected<TrackedMove<true>, TrackedMove<false>> e1(cuda::std::in_place, 5);
    cuda::std::expected<TrackedMove<true>, TrackedMove<false>> e2(cuda::std::unexpect, 10);

    e1.swap(e2);

    assert(!e1.has_value());
    assert(e1.error().i == 10);
    assert(e2.has_value());
    assert(e2->i == 5);

    assert(e1.error().numberOfMoves == 1);
    assert(!e1.error().swapCalled);
    assert(e2->numberOfMoves == 2);
    assert(!e2->swapCalled);
  }

  // !this->has_value() && rhs.has_value()
  // && is_nothrow_move_constructible_v<E>
  {
    cuda::std::expected<TrackedMove<true>, TrackedMove<true>> e1(cuda::std::unexpect, 10);
    cuda::std::expected<TrackedMove<true>, TrackedMove<true>> e2(cuda::std::in_place, 5);

    e1.swap(e2);

    assert(e1.has_value());
    assert(e1->i == 5);
    assert(!e2.has_value());
    assert(e2.error().i == 10);

    assert(e1->numberOfMoves == 1);
    assert(!e1->swapCalled);
    assert(e2.error().numberOfMoves == 2);
    assert(!e2.error().swapCalled);
  }

  // !this->has_value() && rhs.has_value()
  // && !is_nothrow_move_constructible_v<E>
  {
    cuda::std::expected<TrackedMove<true>, TrackedMove<false>> e1(cuda::std::unexpect, 10);
    cuda::std::expected<TrackedMove<true>, TrackedMove<false>> e2(cuda::std::in_place, 5);

    e1.swap(e2);

    assert(e1.has_value());
    assert(e1->i == 5);
    assert(!e2.has_value());
    assert(e2.error().i == 10);

    assert(e1->numberOfMoves == 2);
    assert(!e1->swapCalled);
    assert(e2.error().numberOfMoves == 1);
    assert(!e2.error().swapCalled);
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  // !e1.has_value() && e2.has_value()
  {
    cuda::std::expected<ThrowOnMoveConstruct, int> e1(cuda::std::unexpect, 5);
    cuda::std::expected<ThrowOnMoveConstruct, int> e2(cuda::std::in_place);
    try
    {
      e1.swap(e2);
      assert(false);
    }
    catch (Except)
    {
      assert(!e1.has_value());
      assert(e1.error() == 5);
    }
  }

  // e1.has_value() && !e2.has_value()
  {
    cuda::std::expected<int, ThrowOnMoveConstruct> e1(5);
    cuda::std::expected<int, ThrowOnMoveConstruct> e2(cuda::std::unexpect);
    try
    {
      e1.swap(e2);
      assert(false);
    }
    catch (Except)
    {
      assert(e1.has_value());
      assert(*e1 == 5);
    }
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
