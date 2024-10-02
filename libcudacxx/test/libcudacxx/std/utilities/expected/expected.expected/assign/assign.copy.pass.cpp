//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Older Clangs do not support the C++20 feature to constrain destructors

// constexpr expected& operator=(const expected& rhs);
//
// Effects:
// - If this->has_value() && rhs.has_value() is true, equivalent to val = *rhs.
// - Otherwise, if this->has_value() is true, equivalent to:
//  reinit-expected(unex, val, rhs.error())
// - Otherwise, if rhs.has_value() is true, equivalent to:
//  reinit-expected(val, unex, *rhs)
// - Otherwise, equivalent to unex = rhs.error().
//
// - Then, if no exception was thrown, equivalent to: has_val = rhs.has_value(); return *this;
//
// Returns: *this.
//
// Remarks: This operator is defined as deleted unless:
// - is_copy_assignable_v<T> is true and
// - is_copy_constructible_v<T> is true and
// - is_copy_assignable_v<E> is true and
// - is_copy_constructible_v<E> is true and
// - is_nothrow_move_constructible_v<T> || is_nothrow_move_constructible_v<E> is true.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../../types.h"
#include "test_macros.h"

struct NotCopyConstructible
{
  NotCopyConstructible(const NotCopyConstructible&)            = delete;
  NotCopyConstructible& operator=(const NotCopyConstructible&) = default;
};

struct NotCopyAssignable
{
  NotCopyAssignable(const NotCopyAssignable&)            = default;
  NotCopyAssignable& operator=(const NotCopyAssignable&) = delete;
};

struct MoveMayThrow
{
  MoveMayThrow(MoveMayThrow const&)            = default;
  MoveMayThrow& operator=(const MoveMayThrow&) = default;
  __host__ __device__ MoveMayThrow(MoveMayThrow&&) noexcept(false) {}
  __host__ __device__ MoveMayThrow& operator=(MoveMayThrow&&) noexcept(false)
  {
    return *this;
  }
};

// Test constraints
static_assert(cuda::std::is_copy_assignable_v<cuda::std::expected<int, int>>, "");

// !is_copy_assignable_v<T>
static_assert(!cuda::std::is_copy_assignable_v<cuda::std::expected<NotCopyAssignable, int>>, "");

// !is_copy_constructible_v<T>
static_assert(!cuda::std::is_copy_assignable_v<cuda::std::expected<NotCopyConstructible, int>>, "");

// !is_copy_assignable_v<E>
static_assert(!cuda::std::is_copy_assignable_v<cuda::std::expected<int, NotCopyAssignable>>, "");

// !is_copy_constructible_v<E>
static_assert(!cuda::std::is_copy_assignable_v<cuda::std::expected<int, NotCopyConstructible>>, "");

// !is_nothrow_move_constructible_v<T> && is_nothrow_move_constructible_v<E>
static_assert(cuda::std::is_copy_assignable_v<cuda::std::expected<MoveMayThrow, int>>, "");

// is_nothrow_move_constructible_v<T> && !is_nothrow_move_constructible_v<E>
static_assert(cuda::std::is_copy_assignable_v<cuda::std::expected<int, MoveMayThrow>>, "");

#ifndef TEST_COMPILER_ICC
// !is_nothrow_move_constructible_v<T> && !is_nothrow_move_constructible_v<E>
static_assert(
  cuda::std::__expected_can_copy_assign<MoveMayThrow, MoveMayThrow> == cuda::std::__smf_availability::__deleted, "");
static_assert(!cuda::std::is_copy_assignable_v<cuda::std::expected<MoveMayThrow, MoveMayThrow>>, "");
#endif // TEST_COMPILER_ICC

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // If this->has_value() && rhs.has_value() is true, equivalent to val = *rhs.
  {
    Traced::state oldState{};
    Traced::state newState{};
    cuda::std::expected<Traced, int> e1(cuda::std::in_place, oldState, 5);
    const cuda::std::expected<Traced, int> e2(cuda::std::in_place, newState, 10);
    decltype(auto) x = (e1 = e2);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<Traced, int>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);
    assert(oldState.copyAssignCalled);
  }

  // - Otherwise, if this->has_value() is true, equivalent to:
  // reinit-expected(unex, val, rhs.error())
  //  E move is not noexcept
  //  In this case, it should call the branch
  //
  //  U tmp(cuda::std::move(oldval));
  //  destroy_at(addressof(oldval));
  //  try {
  //    construct_at(addressof(newval), cuda::std::forward<Args>(args)...);
  //  } catch (...) {
  //    construct_at(addressof(oldval), cuda::std::move(tmp));
  //    throw;
  //  }
  //
  {
    TracedNoexcept::state oldState{};
    Traced::state newState{};
    cuda::std::expected<TracedNoexcept, Traced> e1(cuda::std::in_place, oldState, 5);
    const cuda::std::expected<TracedNoexcept, Traced> e2(cuda::std::unexpect, newState, 10);

    decltype(auto) x = (e1 = e2);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<TracedNoexcept, Traced>&>, "");
    assert(&x == &e1);

    assert(!e1.has_value());
    assert(e1.error().data_ == 10);

    assert(!oldState.copyAssignCalled);
    assert(oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(!oldState.copyCtorCalled);
    assert(newState.copyCtorCalled);
    assert(!newState.moveCtorCalled);
    assert(!newState.dtorCalled);
  }

  // - Otherwise, if this->has_value() is true, equivalent to:
  // reinit-expected(unex, val, rhs.error())
  //  E move is noexcept
  //  In this case, it should call the branch
  //
  //  destroy_at(addressof(oldval));
  //  construct_at(addressof(newval), cuda::std::forward<Args>(args)...);
  //
  {
    Traced::state oldState{};
    TracedNoexcept::state newState{};
    cuda::std::expected<Traced, TracedNoexcept> e1(cuda::std::in_place, oldState, 5);
    const cuda::std::expected<Traced, TracedNoexcept> e2(cuda::std::unexpect, newState, 10);

    decltype(auto) x = (e1 = e2);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<Traced, TracedNoexcept>&>, "");
    assert(&x == &e1);

    assert(!e1.has_value());
    assert(e1.error().data_ == 10);

    assert(!oldState.copyAssignCalled);
    assert(!oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(!oldState.copyCtorCalled);
    assert(newState.copyCtorCalled);
    assert(!newState.moveCtorCalled);
    assert(!newState.dtorCalled);
  }

  // - Otherwise, if rhs.has_value() is true, equivalent to:
  // reinit-expected(val, unex, *rhs)
  //  T move is not noexcept
  //  In this case, it should call the branch
  //
  //  U tmp(cuda::std::move(oldval));
  //  destroy_at(addressof(oldval));
  //  try {
  //    construct_at(addressof(newval), cuda::std::forward<Args>(args)...);
  //  } catch (...) {
  //    construct_at(addressof(oldval), cuda::std::move(tmp));
  //    throw;
  //  }
  //
  {
    TracedNoexcept::state oldState{};
    Traced::state newState{};
    cuda::std::expected<Traced, TracedNoexcept> e1(cuda::std::unexpect, oldState, 5);
    const cuda::std::expected<Traced, TracedNoexcept> e2(cuda::std::in_place, newState, 10);

    decltype(auto) x = (e1 = e2);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<Traced, TracedNoexcept>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);

    assert(!oldState.copyAssignCalled);
    assert(oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(!oldState.copyCtorCalled);
    assert(newState.copyCtorCalled);
    assert(!newState.moveCtorCalled);
    assert(!newState.dtorCalled);
  }

  // - Otherwise, if rhs.has_value() is true, equivalent to:
  // reinit-expected(val, unex, *rhs)
  //  T move is noexcept
  //  In this case, it should call the branch
  //
  //  destroy_at(addressof(oldval));
  //  construct_at(addressof(newval), cuda::std::forward<Args>(args)...);
  //
  {
    Traced::state oldState{};
    TracedNoexcept::state newState{};
    cuda::std::expected<TracedNoexcept, Traced> e1(cuda::std::unexpect, oldState, 5);
    const cuda::std::expected<TracedNoexcept, Traced> e2(cuda::std::in_place, newState, 10);

    decltype(auto) x = (e1 = e2);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<TracedNoexcept, Traced>&>, "");
    assert(&x == &e1);

    assert(e1.has_value());
    assert(e1.value().data_ == 10);

    assert(!oldState.copyAssignCalled);
    assert(!oldState.moveCtorCalled);
    assert(oldState.dtorCalled);
    assert(!oldState.copyCtorCalled);
    assert(newState.copyCtorCalled);
    assert(!newState.moveCtorCalled);
    assert(!newState.dtorCalled);
  }

  // Otherwise, equivalent to unex = rhs.error().
  {
    Traced::state oldState{};
    Traced::state newState{};
    cuda::std::expected<int, Traced> e1(cuda::std::unexpect, oldState, 5);
    const cuda::std::expected<int, Traced> e2(cuda::std::unexpect, newState, 10);
    decltype(auto) x = (e1 = e2);
    static_assert(cuda::std::same_as<decltype(x), cuda::std::expected<int, Traced>&>, "");
    assert(&x == &e1);

    assert(!e1.has_value());
    assert(e1.error().data_ == 10);
    assert(oldState.copyAssignCalled);
  }
  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  struct ThrowOnCopyMoveMayThrow
  {
    ThrowOnCopyMoveMayThrow() = default;
    ThrowOnCopyMoveMayThrow(const ThrowOnCopyMoveMayThrow&)
    {
      throw Except{};
    };
    ThrowOnCopyMoveMayThrow& operator=(const ThrowOnCopyMoveMayThrow&) = default;
    ThrowOnCopyMoveMayThrow(ThrowOnCopyMoveMayThrow&&) noexcept(false) {}
  };

  // assign value throw on copy
  {
    cuda::std::expected<ThrowOnCopyMoveMayThrow, int> e1(cuda::std::unexpect, 5);
    const cuda::std::expected<ThrowOnCopyMoveMayThrow, int> e2(cuda::std::in_place);
    try
    {
      e1 = e2;
      assert(false);
    }
    catch (Except)
    {
      assert(!e1.has_value());
      assert(e1.error() == 5);
    }
  }

  // assign error throw on copy
  {
    cuda::std::expected<int, ThrowOnCopyMoveMayThrow> e1(5);
    const cuda::std::expected<int, ThrowOnCopyMoveMayThrow> e2(cuda::std::unexpect);
    try
    {
      e1 = e2;
      assert(false);
    }
    catch (Except)
    {
      assert(e1.has_value());
      assert(e1.value() == 5);
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
